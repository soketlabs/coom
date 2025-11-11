import torch
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from torch.optim import Adam

from checkpointing import load_distributed_checkpoint, save_distributed_checkpoint
from coom.training.utils import print_rank_0
from dataloader import get_train_data_iterator
from distributed_utils import destroy_distributed, init_distributed
from lossfunc import forward_step_func
from model_provider_eka import model_provider

# export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
# export MASTER_PORT=${MASTER_PORT:-"6000"}
# export NNODES=${NNODES:-"1"}
# export GPUS_PER_NODE=8
# export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


def main():
    init_distributed(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)

    model = model_provider()
    device = torch.device("cuda")
    model.to(device)
    # exit()

    print_rank_0(model)

    optim = Adam(model.parameters())

    train_iterator = get_train_data_iterator()

    forward_backward_func = get_forward_backward_func()

    # Running the model for 5 iterations
    for _ in range(1):
        optim.zero_grad()

        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=train_iterator,
            model=model,
            num_microbatches=1,
            seq_length=64,
            micro_batch_size=8,
            decoder_seq_length=64,
            forward_only=False,
        )

        optim.step()

        print_rank_0(f"Losses reduced :{losses_reduced}")

    # Saving the model
    save_distributed_checkpoint(gpt_model=model, checkpoint_path="./ckpt")

    # Loading the model
    model = load_distributed_checkpoint(gpt_model=model, checkpoint_path="./ckpt")
    model.to(device)
    print_rank_0("Successfully loaded the model")

    destroy_distributed()


if __name__ == "__main__":
    main()
