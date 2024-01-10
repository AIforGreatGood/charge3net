# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
import os
import sys

sys.path.append(os.getcwd())

import torch
import numpy as np
from hydra.utils import instantiate
from torch.distributed import destroy_process_group, init_process_group

from src.trainer import Trainer

def test(rank, cfg, env):
    print(f"Initializing on rank {rank} with environment {env}")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # global rank will be unique across all nodes/gpus
    global_rank = env["group_rank"] * cfg.nprocs + rank

    # initialize NCCL
    init_process_group(backend="nccl", init_method="env://", rank=global_rank, world_size=env["world_size"])
    torch.cuda.set_device(rank)


    model = instantiate(cfg.model.model)
    optimizer = instantiate(cfg.model.optimizer)(model.parameters())
    scheduler = instantiate(cfg.model.lr_scheduler)(optimizer)
    criterion = instantiate(cfg.model.criterion)

    datamodule = instantiate(cfg.data)

    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        criterion=criterion,
        log_dir=cfg.log_dir,
        gpu_id=rank,
        global_rank=global_rank,
        load_checkpoint_path=cfg.checkpoint_path,
    )
    if cfg.cube_dir is not None: 
        assert cfg.data.test_probes is None, "Cannot write cube without data.test_probes=null"
    trainer.test(test_dl=datamodule.test_dataloader(), cube_dir=cfg.cube_dir)

    # necessary at end of main code
    destroy_process_group()