# Copyright (c) 2023, MASSACHUSETTS INSTITUTE OF TECHNOLOGY
# Subject to FAR 52.227-11 - Patent Rights - Ownership by the Contractor (May 2014).
import os
os.environ["HYDRA_FULL_ERROR"] = "1"
# os.environ["NCCL_DEBUG"] = "INFO"
# os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

import subprocess
import sys

import hydra
from omegaconf import OmegaConf
import torch.multiprocessing as mp

sys.path.append(os.getcwd())
from src.train import train


@hydra.main(config_path=None, version_base=None)
def train_from_config(cfg):

    # device environment
    env = {
        # port to use for distributed messaging
        "master_port": str(29500),
        # address to use for distributed messaging
        "master_addr": "localhost",
        # total number of processes running 
        "world_size": cfg.nnodes * cfg.nprocs,
        # what node we're on (pull from SLURM)
        "group_rank": int(os.environ.get("SLURM_NODEID", "0"))

    }

    # multinode, assign master address to first slurm node in job
    if cfg.nnodes > 1:
        cmd = "scontrol show hostnames " + os.getenv('SLURM_JOB_NODELIST')
        env["master_addr"] = subprocess.check_output(cmd.split()).decode().splitlines()[0]
        print("master_addr", env["master_addr"])

    # setup environment variables that nccl needs
    os.environ["MASTER_ADDR"] = env["master_addr"]
    os.environ["MASTER_PORT"] = env["master_port"]
    os.environ["WORLD_SIZE"] = str(env["world_size"])

    # See https://github.com/facebookresearch/hydra/issues/2772
    OmegaConf.resolve(cfg)

    # this calls the function run(rank, args) for each rank in range(0, nprocs)
    mp.spawn(train, args=(cfg, env), nprocs=cfg.nprocs)

if __name__ == "__main__":
    train_from_config()