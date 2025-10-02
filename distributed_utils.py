import os
import torch

from megatron.core import parallel_state

def init_distributed(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
):
    parallel_state.destroy_model_parallel()

    # Torch setup
    rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = torch.cuda.device_count()
    print(f'rank: {rank}, world_size: {world_size}')
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(world_size=world_size, rank=rank)

    # Init mcore
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
    )

def destroy_distributed():
    torch.distributed.destroy_process_group()
