# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
import os
from pathlib import Path
import logging
import time
import shutil

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np

from .charge3net.models.densitymodel import PainnDensityModel
from .utils import predictions as pred_utils

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        log_dir: str,
        gpu_id: int,
        global_rank: int,
        load_checkpoint_path=None,
        log_steps=50,
    ):
        self.local_rank = gpu_id
        self.global_rank = global_rank
        self.model = model.to(self.local_rank)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        self.start_epoch = 0
        self.step = 0
        self.best_nmape = float("inf")

        if load_checkpoint_path is not None:
            assert Path(load_checkpoint_path).exists(), f"file {load_checkpoint_path} does not exist"
            print(f"Loading checkpoint {load_checkpoint_path}")
            self._load_checkpoint(load_checkpoint_path)

        self.log_dir = Path(log_dir)
        
        # add slurm job to log directory (if using slurm)
        if "SLURM_JOB_ID" in os.environ:
            job_id = os.environ["SLURM_JOB_ID"]
            if "SLURM_ARRAY_TASK_ID" in os.environ:
                job_id += '_' + os.environ['SLURM_ARRAY_TASK_ID']
            self.log_dir = self.log_dir / job_id

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_steps = log_steps
        self.checkpoint_path = self.log_dir / "checkpoint.pt"

        if self.local_rank == 0 and self.global_rank == 0:
            # setup tensorboard and print logging
            self.tensorboard = SummaryWriter(self.log_dir)
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
                handlers=[logging.FileHandler(self.log_dir / "log.txt"), logging.StreamHandler()],
            )

        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{self.local_rank}")
        if "pytorch-lightning_version" in checkpoint: return self._load_checkpoint_legacy(checkpoint)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.start_epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]

    def _load_checkpoint_legacy(self, checkpoint):
        # loads old lightning checkpoints
        self.model.load_state_dict({k.replace("network.", ""): v for k, v in checkpoint["state_dict"].items()})
        self.optimizer.load_state_dict(checkpoint["optimizer_states"][0])
        self.scheduler.load_state_dict(checkpoint["lr_schedulers"][0])
        self.start_epoch = checkpoint["epoch"]
        self.step = checkpoint["global_step"]

    def _save_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "step": self.step,
            "model": self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_nmape": self.best_nmape,
        }
        torch.save(checkpoint, self.checkpoint_path)

    def _train_epoch(self):
        for batch in self.train_dl:
            batch = self._to_device(batch)
            output = self.model(batch)
            loss = self.criterion(output, batch["probe_target"])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1

            if self.local_rank == 0 and self.global_rank == 0:
                # logging/checkpointing on rank 0 (otherwise would repeat on each node)
                if self.step % self.log_steps == 0:
                    last_lr = self.optimizer.param_groups[-1]['lr']
                    logging.info(f"step: {self.step} train/loss: {loss.item():.6f} lr: {last_lr:.6f}")
                    self.tensorboard.add_scalar("train/loss", loss.item(), global_step=self.step)
                    self.tensorboard.add_scalar("lr", last_lr, global_step=self.step)
                    self.tensorboard.flush()


    def _to_device(self, batch):
        # Moves batch (dict of tensors) to proper device
        return {
            k: v.to(self.local_rank) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    @torch.no_grad()
    def _valid_epoch(self):
        total_nmape, total_count = 0.0, 0
        for batch in self.valid_dl:
            batch = self._to_device(batch)
            preds = self.model(batch)
            diff = batch["probe_target"] - preds
            nmape = torch.abs(diff).sum(1) / torch.abs(batch["probe_target"]).sum(1) * 100.
            total_count += nmape.shape[0]
            total_nmape += nmape.sum(0)

        nmape = total_nmape / total_count
        if len(nmape.shape): # add spin metrics
            return  {
                "val/IntegralNormalizedMeanAbsoluteError": nmape[0].item(),
                "val_spin/IntegralNormalizedMeanAbsoluteError": nmape[1].item(),
            }
        return {"val/IntegralNormalizedMeanAbsoluteError": nmape.item()}
        

    def fit(self, train_dl, valid_dl, steps):

        self.train_dl = train_dl
        self.valid_dl = [batch for batch in valid_dl] # cache validation dataloader

        epoch = self.start_epoch
        while self.step < steps:
            self.train_dl.sampler.set_epoch(epoch)
            self.model.train()
            self._train_epoch()
            self.model.eval()
            metrics = self._valid_epoch()

            if self.local_rank == 0 and self.global_rank == 0:
                # logging/checkpointing on rank 0 (otherwise would repeat on each node)
                logging.info(f"step: {self.step} {' '.join(f'{k}: {v:.4f}' for k, v in metrics.items())}")
                for k, v in metrics.items(): self.tensorboard.add_scalar(k, v, global_step=self.step)
                self.tensorboard.flush()
                
                if metrics["val/IntegralNormalizedMeanAbsoluteError"] <= self.best_nmape:
                    self.best_nmape = metrics["val/IntegralNormalizedMeanAbsoluteError"]
                    self._save_checkpoint(epoch)



    @torch.no_grad()
    def test(self, test_dl, cube_dir=None, max_predict_batch_probes=2500):
        self.model.eval()
        tmp_nmape_dir = self.log_dir / ".tmp_nmape"
        tmp_nmape_dir.mkdir(exist_ok=True)

        if cube_dir is not None:
            cube_dir = Path(cube_dir) / "cubes"
            cube_dir.mkdir(exist_ok=True, parents=True)
            tmp_cube_dir = self.log_dir / ".tmp_cubes"
            tmp_cube_dir.mkdir(exist_ok=True)

        for i, batch in enumerate(test_dl):
            if self.local_rank == 0 and self.global_rank == 0:
                logging.info(f"testing {i} / {len(test_dl)} (rank 0)")

            # write out nmape (or partial nmape)
            predictions = self._test_step(batch, max_predict_batch_probes)
            # TODO: handle spin density here (should only compute with preds[:, :, 0] and targets[:, :, 0]
            diff_sum, targ_sum = pred_utils.compute_nmape_components(predictions["preds"], predictions["targets"])
            out_dict = {
                "diff_sum": diff_sum,
                "target_sum": targ_sum,
                "num_probes": torch.prod(torch.tensor(predictions["preds"].shape)).item(),
                "filename": predictions["filename"],
                "probe_offset": predictions["probe_offset"].item(),
                "grid_shape": predictions["grid_shape"].tolist(),
                "time": predictions["time"]
            }
            torch.save(out_dict, tmp_nmape_dir / f"pred_{predictions['filename']}_offset_{predictions['probe_offset']}.pt")

            if cube_dir is None:
                continue

            # write out cubes (or partial cubes)
            preds = predictions["preds"].cpu().numpy()
            grid_shape = predictions["grid_shape"].cpu()

            if not predictions["partial"]:
                cube = preds.reshape(grid_shape.numpy())
                np.save(cube_dir / f"{predictions['filename']}.npy", cube)
            else:
                out_dict = {
                    "density": preds,
                    "grid_shape": grid_shape,
                    "probe_offset": predictions["probe_offset"].item(),
                    "filename": predictions["filename"],
                }
                torch.save(out_dict, tmp_cube_dir / f"pred_{predictions['filename']}_offset_{predictions['probe_offset']}.pt")
            

        # End of testing, once all nodes have finished
        torch.distributed.barrier()
        if self.local_rank == 0 and self.global_rank == 0:
            preds = [torch.load(f, map_location=self.local_rank) for f in tmp_nmape_dir.iterdir() if "pred" in f.name]
            pred_utils.save_preds(preds, self.log_dir)
            if cube_dir is not None:
                pred_utils.combine_partial_cubes(tmp_cube_dir, cube_dir)

            # cleanup
            shutil.rmtree(tmp_nmape_dir)
            if cube_dir is not None:
                shutil.rmtree(tmp_cube_dir)
                
    def _test_step(self, batch, max_predict_batch_probes):
        start_time = time.time()
            
        # see NOTE below
        batch = self._to_device(batch)

        if batch["num_probes"] > max_predict_batch_probes:            
            all_loss, all_preds, all_targets, atom_repr = [], [], [], None


            for i, sub_batch in enumerate(pred_utils.split_batch(batch, max_predict_batch_probes)):
                # NOTE: much slower to move sub-batch one at a time instead of all at once above
                # sub_batch = self._to_device(sub_batch)

                # if self.local_rank == 0 and self.global_rank == 0:
                #     logging.info(f"sub-batch {i} (rank 0)")

                # atom representations only need to be calculated once
                if atom_repr is None:
                    atom_repr = self.model.module.atom_model(sub_batch)
                if isinstance(self.model.module, PainnDensityModel):
                    # PaiNN takes (scalar, vector) tuple as two args
                    outputs = self.model.module.probe_model(sub_batch, *atom_repr)
                else:
                    outputs = self.model.module.probe_model(sub_batch, atom_repr)

                loss = self.criterion(outputs, sub_batch["probe_target"])
                all_loss.append(loss)
                all_preds.append(outputs)
                all_targets.append(sub_batch["probe_target"])

            preds = torch.cat(all_preds, dim=1)
            loss = torch.mean(torch.tensor(all_loss))
            targets = torch.cat(all_targets, dim=1) 

        else:
            preds = self.model(batch)
            loss = self.criterion(preds, batch["probe_target"])
            targets = batch["probe_target"]

        return {
            "loss": loss, 
            "preds": preds, 
            "targets": targets, 
            "filename":batch["filename"][0],
            "probe_offset": batch["probe_offset"][0],
            "grid_shape": batch["grid_shape"][0],
            "partial": batch["partial"][0],
            "time": time.time() - start_time + batch["load_time"][0],
        }





