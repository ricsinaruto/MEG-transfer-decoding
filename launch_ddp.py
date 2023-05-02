import os
from training import main as train
from args import Args
import torch

import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group


def launch(rank: int, world_size: int):
    """ Initialize the distributed environment. """
    # world_size: number of processes
    # rank: id of the current process
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

    train(Args, gpu_id=rank)
    destroy_process_group()


if __name__ == '__main__':
    num_gpus = torch.cuda.device_count()
    Args.num_gpus = num_gpus
    mp.spawn(launch, args=(num_gpus,), nprocs=num_gpus)
