import math
import time
from dataclasses import dataclass, field

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from distributed_utils import init_distributed, destroy_distributed
from model_provider import model_provider
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from coom.training.utils import print_rank_0

from dataloader import get_train_data_iterator
from lossfunc import forward_step_func
from checkpointing import save_distributed_checkpoint, load_distributed_checkpoint


@dataclass
class TrainingConfig:
    # Parallelism
    tensor_model_parallel_size: int = 2

    # Optimizer
    lr: float = 3e-5            # matches Pragna-1B blog
    weight_decay: float = 0.1   # standard for LLMs (Llama/GPT-3 use 0.1)
    beta1: float = 0.9
    beta2: float = 0.95         # GPT-3 / Llama standard (not 0.999)
    eps: float = 1e-8
    max_grad_norm: float = 1.0  # gradient clipping threshold

    # LR schedule
    warmup_steps: int = 2000    # linear warmup before cosine decay
    total_steps: int = 1_000_000  # set to actual token_budget / batch_tokens

    # Training loop
    num_train_steps: int = 5    # demo; set to total_steps for real run
    seq_length: int = 64        # demo; use 2048+ for production
    micro_batch_size: int = 8
    num_microbatches: int = 1   # increase with pipeline parallelism

    # Checkpointing
    checkpoint_path: str = "./ckpt"
    save_every_n_steps: int = 1000

    # Precision
    use_bf16: bool = True       # BF16 matches Pragna-1B training

    # Logging
    log_every_n_steps: int = 10

    # Reproducibility
    seed: int = 123


def get_cosine_schedule_with_warmup(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Linear warmup for warmup_steps, then cosine decay to min_lr_ratio * lr.
    This matches the Pragna-1B training schedule described in the blog.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # linear warmup: 0 → 1
            return float(current_step) / float(max(1, warmup_steps))

        # cosine decay: 1 → min_lr_ratio
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # rescale from [0,1] to [min_lr_ratio, 1]
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def main(cfg: TrainingConfig = TrainingConfig()):

    # --- distributed setup ---
    init_distributed(tensor_model_parallel_size=cfg.tensor_model_parallel_size)
    model_parallel_cuda_manual_seed(cfg.seed)

    # --- model ---
    model = model_provider()
    device = torch.device("cuda")
    model.to(device)
    print_rank_0(model)

    # --- optimizer (AdamW, not Adam) ---
    # Exclude bias and LayerNorm/RMSNorm params from weight decay — standard practice.
    no_decay_names = ["bias", "norm.weight", "norm.bias"]
    param_groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay_names)
            ],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay_names)
            ],
            "weight_decay": 0.0,
        },
    ]
    optim = AdamW(
        param_groups,
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
    )

    # --- LR scheduler ---
    scheduler = get_cosine_schedule_with_warmup(
        optim,
        warmup_steps=cfg.warmup_steps,
        total_steps=cfg.total_steps,
    )

    # --- data ---
    train_iterator = get_train_data_iterator()

    # --- Megatron forward+backward function ---
    forward_backward_func = get_forward_backward_func()

    # --- BF16 autocast context ---
    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=torch.bfloat16)
        if cfg.use_bf16
        else torch.cuda.amp.autocast(enabled=False)
    )

    print_rank_0(
        f"\n=== Starting EKA pre-training ==="
        f"\n  steps:        {cfg.num_train_steps}"
        f"\n  seq_length:   {cfg.seq_length}"
        f"\n  micro_bs:     {cfg.micro_batch_size}"
        f"\n  lr:           {cfg.lr}"
        f"\n  warmup_steps: {cfg.warmup_steps}"
        f"\n  grad_clip:    {cfg.max_grad_norm}"
        f"\n  precision:    {'BF16' if cfg.use_bf16 else 'FP32'}"
        f"\n================================\n"
    )

    step_start = time.time()

    try:
        for step in range(1, cfg.num_train_steps + 1):
            optim.zero_grad()

            with autocast_ctx:
                losses_reduced = forward_backward_func(
                    forward_step_func=forward_step_func,
                    data_iterator=train_iterator,
                    model=model,
                    num_microbatches=cfg.num_microbatches,
                    seq_length=cfg.seq_length,
                    micro_batch_size=cfg.micro_batch_size,
                    decoder_seq_length=cfg.seq_length,
                    forward_only=False,
                )

            # --- gradient clipping ---
            # Returns the total grad norm BEFORE clipping — useful for monitoring.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.max_grad_norm
            )

            optim.step()
            scheduler.step()

            # --- logging ---
            if step % cfg.log_every_n_steps == 0:
                elapsed = time.time() - step_start
                tokens_per_sec = (
                    cfg.log_every_n_steps
                    * cfg.micro_batch_size
                    * cfg.seq_length
                    * cfg.num_microbatches
                ) / elapsed

                current_lr = scheduler.get_last_lr()[0]
                loss_val = (
                    losses_reduced[0]["avg"] if losses_reduced else float("nan")
                )

                print_rank_0(
                    f"step {step:>7d} | "
                    f"loss: {loss_val:.4f} | "
                    f"grad_norm: {float(grad_norm):.4f} | "
                    f"lr: {current_lr:.2e} | "
                    f"tok/sec: {tokens_per_sec:.0f}"
                )
                step_start = time.time()

            # --- periodic checkpoint ---
            if step % cfg.save_every_n_steps == 0:
                ckpt = f"{cfg.checkpoint_path}/step_{step}"
                save_distributed_checkpoint(gpt_model=model, checkpoint_path=ckpt)
                print_rank_0(f"Checkpoint saved at step {step} → {ckpt}")

    except KeyboardInterrupt:
        print_rank_0("Interrupted — saving emergency checkpoint.")
        save_distributed_checkpoint(
            gpt_model=model, checkpoint_path=f"{cfg.checkpoint_path}/interrupted"
        )

    # --- final checkpoint + verify load ---
    save_distributed_checkpoint(
        gpt_model=model, checkpoint_path=f"{cfg.checkpoint_path}/final"
    )
    model = load_distributed_checkpoint(
        gpt_model=model, checkpoint_path=f"{cfg.checkpoint_path}/final"
    )
    model.to(device)
    print_rank_0("Final checkpoint verified — load successful.")

    destroy_distributed()


if __name__ == "__main__":
    main()