import torch
import torch.distributed as dist
import os

def setup_ddp(rank, world_size, port='12355'):
    """Initializes the distributed environment."""
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', port)

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"DDP Setup: Rank {rank}/{world_size} on GPU {torch.cuda.current_device()}")

def cleanup_ddp():
    """Cleans up the distributed environment."""
    dist.destroy_process_group()
    print("DDP Cleanup successful.")

def is_main_process(rank):
    """Checks if the current process is the main process (rank 0)."""
    return rank == 0

def reduce_tensor(tensor, world_size):
    """Reduces tensor across all processes for averaging."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def barrier():
    """Synchronizes all processes."""
    dist.barrier()