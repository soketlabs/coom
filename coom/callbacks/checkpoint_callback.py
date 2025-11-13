# import logging
# import os
# import shutil
# from typing import Any, Dict, Optional

# import torch
# import yaml
# from lightning import LightningModule, Trainer
# from lightning.pytorch.callbacks import Callback

# # Optional: import distributed helpers from Megatron-Core
# try:
#     from megatron.core import dist_checkpointing, mpu

#     def is_global_rank_zero():
#         try:
#             return mpu.get_data_parallel_rank() == 0
#         except Exception:
#             if not torch.distributed.is_initialized():
#                 return True
#             return torch.distributed.get_rank() == 0

#     _HAS_DIST_CHECKPOINTING = True
# except Exception:
#     mpu = None
#     dist_checkpointing = None

#     def is_global_rank_zero():
#         if not torch.distributed.is_initialized():
#             return True
#         return torch.distributed.get_rank() == 0

#     _HAS_DIST_CHECKPOINTING = False

# logger = logging.getLogger(__name__)


# def barrier_safe():
#     """Helper to safely call barrier and catch exceptions."""
#     if torch.distributed.is_initialized():
#         try:
#             rank = torch.distributed.get_rank()
#             logger.debug(f"[rank {rank}] Entering barrier")
#             torch.distributed.barrier()
#             logger.debug(f"[rank {rank}] Exited barrier")
#         except Exception as e:
#             rank = (
#                 torch.distributed.get_rank()
#                 if torch.distributed.is_initialized()
#                 else -1
#             )
#             logger.error(f"[rank {rank}] Barrier failed: {e}")
#     else:
#         logger.debug("Barrier skipped (distributed not initialized)")


# class AsyncModelCheckpoint(Callback):
#     """
#     Checkpointing callback using Megatron-Core's async_save=True.

#     When async_save=True:
#     - Megatron-Core handles threading internally
#     - No manual barriers needed (Megatron manages synchronization)
#     - Training continues while checkpoint writes happen in background
#     - finalize() must be called before next checkpoint or training end
#     """

#     def __init__(
#         self,
#         dirpath: str,
#         filename: str = "epoch={epoch}-step={step}",
#         save_interval: int = 1000,
#         save_last: bool = True,
#         save_ema: bool = False,
#         context: Optional[Dict[str, Any]] = None,
#         async_save: bool = True,  # Enable Megatron async by default
#     ):
#         super().__init__()
#         self.dirpath = dirpath
#         self.filename = filename
#         self.save_interval = int(save_interval)
#         self.save_last = save_last
#         self.save_ema = save_ema
#         self.context = context or {}
#         self.async_save = async_save

#         # Track the async request handle (returned by dist_checkpointing.save)
#         self._async_request = None

#         os.makedirs(self.dirpath, exist_ok=True)

#     def _format_ckpt_basename(self, trainer: Trainer, ema=False) -> str:
#         fname = self.filename.format(
#             epoch=trainer.current_epoch, step=trainer.global_step
#         )
#         if ema:
#             fname += "-EMA"
#         return fname

#     def _format_ckpt_path(self, trainer: Trainer, ema=False) -> str:
#         base = self._format_ckpt_basename(trainer, ema=ema)
#         return os.path.join(self.dirpath, f"{base}.ckpt")

#     def _dist_checkpoint_dir_from_path(self, path: str) -> str:
#         if path.endswith(".ckpt"):
#             return path[: -len(".ckpt")]
#         return path + ".dist"

#     def _unfinished_marker_path(self, path: str) -> str:
#         return path + ".unfinished"

#     def _maybe_rank(self):
#         if torch.distributed.is_initialized():
#             return torch.distributed.get_rank()
#         return 0

#     def _remove_unfinished_checkpoints(self):
#         if not is_global_rank_zero():
#             return
#         for name in os.listdir(self.dirpath):
#             if name.endswith(".unfinished"):
#                 unfinished_marker = os.path.join(self.dirpath, name)
#                 ckpt_name = name.replace(".unfinished", "")
#                 ckpt_path = os.path.join(self.dirpath, ckpt_name)
#                 logger.warning(f"Removing unfinished checkpoint: {ckpt_path}")
#                 if os.path.isdir(ckpt_path):
#                     shutil.rmtree(ckpt_path, ignore_errors=True)
#                 elif os.path.exists(ckpt_path):
#                     try:
#                         os.remove(ckpt_path)
#                     except Exception:
#                         pass
#                 try:
#                     os.remove(unfinished_marker)
#                 except Exception:
#                     pass

#     def _finalize_async_save(self):
#         """
#         Finalize any pending async checkpoint.
#         Must be called before starting a new checkpoint or ending training.
#         """
#         if self._async_request is None:
#             return

#         rank = self._maybe_rank()
#         logger.info(f"[rank {rank}] Finalizing previous async checkpoint...")

#         try:
#             # This blocks until the async save completes
#             # Megatron-Core handles the distributed coordination internally
#             dist_checkpointing.finalize(self._async_request)
#             logger.info(f"[rank {rank}] Async checkpoint finalized")
#         except Exception as e:
#             logger.exception(f"[rank {rank}] Failed to finalize async checkpoint: {e}")
#         finally:
#             self._async_request = None

#     def _do_distributed_save(
#         self, trainer: Trainer, pl_module: LightningModule, path: str, ema: bool = False
#     ):
#         """
#         Distributed checkpoint save using async_save=True.
#         Returns immediately, checkpoint continues in background.
#         """
#         rank = self._maybe_rank()
#         dist_dir = self._dist_checkpoint_dir_from_path(path)

#         logger.info(f"[rank {rank}] Starting async checkpoint: {path}")

#         # IMPORTANT: Finalize any previous async save before starting new one
#         if self._async_request is not None:
#             self._finalize_async_save()

#         # Rank 0: Create directories and marker (minimal blocking work)
#         if is_global_rank_zero():
#             os.makedirs(dist_dir, exist_ok=True)
#             marker = self._unfinished_marker_path(path)
#             os.makedirs(os.path.dirname(marker), exist_ok=True)
#             open(marker, "w").close()
#             logger.debug(f"[rank {rank}] Created unfinished marker")

#         # Brief barrier to ensure marker is created before proceeding
#         barrier_safe()

#         # Get sharded state dict
#         sharded_state_dict = None
#         use_dist_checkpoint = False

#         try:
#             sharded_state_dict = pl_module.sharded_state_dict(prefix="")
#             use_dist_checkpoint = (
#                 _HAS_DIST_CHECKPOINTING and sharded_state_dict is not None
#             )
#         except Exception as e:
#             logger.debug(f"[rank {rank}] No sharded_state_dict: {e}")

#         if use_dist_checkpoint and self.async_save:
#             # Async distributed checkpoint - returns immediately
#             logger.info(f"[rank {rank}] Launching async distributed checkpoint")
#             try:
#                 self._async_request = dist_checkpointing.save(
#                     sharded_state_dict=sharded_state_dict,
#                     checkpoint_dir=dist_dir,
#                     async_save=True,  # <-- Key: Megatron handles async internally
#                 )
#                 logger.info(f"[rank {rank}] Async save launched (non-blocking)")

#                 # Optionally save metadata immediately (since it's small)
#                 if is_global_rank_zero():
#                     self._save_metadata_sync(path, ema, pl_module)

#             except Exception as e:
#                 logger.exception(f"[rank {rank}] Async distributed save failed: {e}")
#                 self._async_request = None

#         elif use_dist_checkpoint and not self.async_save:
#             # Synchronous distributed checkpoint
#             logger.info(f"[rank {rank}] Using synchronous distributed checkpoint")
#             try:
#                 dist_checkpointing.save(
#                     sharded_state_dict=sharded_state_dict,
#                     checkpoint_dir=dist_dir,
#                     async_save=False,
#                 )
#                 logger.info(f"[rank {rank}] Synchronous distributed save complete")

#                 if is_global_rank_zero():
#                     self._save_metadata_sync(path, ema, pl_module)

#                 barrier_safe()
#             except Exception as e:
#                 logger.exception(
#                     f"[rank {rank}] Synchronous distributed save failed: {e}"
#                 )

#         else:
#             # Fallback: single-file checkpoint (synchronous)
#             logger.info(f"[rank {rank}] Using single-file checkpoint (fallback)")

#             # All ranks must execute the same code path
#             if is_global_rank_zero():
#                 try:
#                     trainer.save_checkpoint(path)
#                     logger.info(f"[rank {rank}] Single-file save complete")
#                     self._save_metadata_sync(path, ema, pl_module)
#                 except Exception as e:
#                     logger.exception(f"[rank {rank}] Single-file save failed: {e}")
#             else:
#                 logger.info(
#                     f"[rank {rank}] Waiting for rank 0 to complete single-file save"
#                 )

#             # Barrier after fallback save - all ranks synchronize
#             logger.info(f"[rank {rank}] About to hit barrier after fallback save")
#             barrier_safe()
#             logger.info(f"[rank {rank}] Passed barrier after fallback save")

#         logger.info(f"[rank {rank}] Checkpoint initiated: {path}")

#         # Final barrier before returning to ensure all ranks are done
#         logger.info(f"[rank {rank}] Final barrier before return")
#         barrier_safe()
#         logger.info(f"[rank {rank}] Checkpoint function complete")

#     def _save_metadata_sync(self, path: str, ema: bool, pl_module: LightningModule):
#         """Save EMA, context, and cleanup markers (rank 0 only)."""
#         rank = self._maybe_rank()

#         # Save EMA if needed
#         if ema and hasattr(pl_module, "ema_model"):
#             try:
#                 ema_state = pl_module.ema_model.state_dict()
#                 ema_file = path.replace(".ckpt", "-EMA.ckpt")
#                 torch.save(ema_state, ema_file)
#                 logger.info(f"[rank {rank}] EMA saved")
#             except Exception as e:
#                 logger.error(f"[rank {rank}] EMA save failed: {e}")

#         # Save context
#         try:
#             ctx_dir = os.path.join(os.path.dirname(path), "context")
#             os.makedirs(ctx_dir, exist_ok=True)
#             with open(os.path.join(ctx_dir, "train_context.yaml"), "w") as f:
#                 yaml.safe_dump(self.context, f)
#         except Exception as e:
#             logger.warning(f"[rank {rank}] Context save failed: {e}")

#         # Remove unfinished marker (checkpoint is done or in progress safely)
#         marker = self._unfinished_marker_path(path)
#         try:
#             os.remove(marker)
#             logger.debug(f"[rank {rank}] Removed unfinished marker")
#         except FileNotFoundError:
#             pass

#         # Create symlink
#         if self.save_last:
#             symlink_path = os.path.join(self.dirpath, "last.ckpt")
#             target = os.path.basename(path)
#             try:
#                 if os.path.islink(symlink_path) or os.path.exists(symlink_path):
#                     os.remove(symlink_path)
#                 os.symlink(target, symlink_path)
#             except OSError as e:
#                 logger.warning(f"[rank {rank}] Symlink failed: {e}")

#     # ---------------------
#     # Lightning hooks
#     # ---------------------
#     def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
#         if is_global_rank_zero():
#             self._remove_unfinished_checkpoints()
#             logger.info("Cleaned unfinished checkpoints")
#         barrier_safe()

#     def on_train_batch_end(
#         self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx
#     ):
#         if trainer.global_step == 0:
#             return

#         if (trainer.global_step % self.save_interval) != 0:
#             return

#         path = self._format_ckpt_path(trainer, ema=self.save_ema)
#         rank = self._maybe_rank()
#         logger.info(
#             f"[rank {rank}] Triggering checkpoint at step {trainer.global_step}"
#         )

#         try:
#             self._do_distributed_save(trainer, pl_module, path, ema=self.save_ema)
#         except Exception as e:
#             logger.exception(f"[rank {rank}] Checkpoint failed: {e}")
#             # CRITICAL: Ensure barrier even on failure so other ranks don't hang
#             barrier_safe()

#         logger.info(f"[rank {rank}] on_train_batch_end complete")

#     def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
#         # CRITICAL: Finalize any pending async checkpoint before ending
#         if self._async_request is not None:
#             logger.info(
#                 f"[rank {self._maybe_rank()}] Finalizing pending checkpoint before train end"
#             )
#             self._finalize_async_save()

#         # Save final checkpoint
#         path = self._format_ckpt_path(trainer, ema=self.save_ema)
#         logger.info(f"[rank {self._maybe_rank()}] Final checkpoint")

#         try:
#             # Final checkpoint should be synchronous to ensure it completes
#             original_async = self.async_save
#             self.async_save = False
#             self._do_distributed_save(trainer, pl_module, path, ema=self.save_ema)
#             self.async_save = original_async
#         except Exception:
#             logger.exception(f"[rank {self._maybe_rank()}] Final checkpoint failed")

#         barrier_safe()


# new without  rank comm

# import logging
# import os
# import shutil
# import threading
# from typing import Any, Dict, Optional

# import torch
# import yaml
# from lightning import LightningModule, Trainer
# from lightning.pytorch.callbacks import Callback

# # Optional: import distributed helpers from Megatron-Core
# try:
#     from megatron.core import dist_checkpointing, mpu

#     def is_global_rank_zero():
#         try:
#             return mpu.get_data_parallel_rank() == 0
#         except Exception:
#             if not torch.distributed.is_initialized():
#                 return True
#             return torch.distributed.get_rank() == 0

#     _HAS_DIST_CHECKPOINTING = True
# except Exception:
#     mpu = None
#     dist_checkpointing = None

#     def is_global_rank_zero():
#         if not torch.distributed.is_initialized():
#             return True
#         return torch.distributed.get_rank() == 0

#     _HAS_DIST_CHECKPOINTING = False

# logger = logging.getLogger(__name__)


# class AsyncModelCheckpoint(Callback):
#     """
#     Truly async checkpointing - each rank saves independently without coordination.

#     Each rank saves its own shard to: {checkpoint_dir}/rank_{rank}/model.pt
#     No barriers, no cross-rank communication during save.
#     """

#     def __init__(
#         self,
#         dirpath: str,
#         filename: str = "epoch={epoch}-step={step}",
#         save_interval: int = 1000,
#         save_last: bool = True,
#         save_ema: bool = False,
#         context: Optional[Dict[str, Any]] = None,
#     ):
#         super().__init__()
#         self.dirpath = dirpath
#         self.filename = filename
#         self.save_interval = int(save_interval)
#         self.save_last = save_last
#         self.save_ema = save_ema
#         self.context = context or {}

#         # Track async save threads per rank
#         self._save_thread: Optional[threading.Thread] = None
#         self._thread_lock = threading.Lock()

#         os.makedirs(self.dirpath, exist_ok=True)

#     def _format_ckpt_basename(self, trainer: Trainer, ema=False) -> str:
#         fname = self.filename.format(
#             epoch=trainer.current_epoch, step=trainer.global_step
#         )
#         if ema:
#             fname += "-EMA"
#         return fname

#     def _format_ckpt_path(self, trainer: Trainer, ema=False) -> str:
#         base = self._format_ckpt_basename(trainer, ema=ema)
#         return os.path.join(self.dirpath, base)

#     def _get_rank(self):
#         if torch.distributed.is_initialized():
#             return torch.distributed.get_rank()
#         return 0

#     def _get_world_size(self):
#         if torch.distributed.is_initialized():
#             return torch.distributed.get_world_size()
#         return 1

#     def _remove_unfinished_checkpoints(self):
#         """Only rank 0 cleans up unfinished checkpoints."""
#         if not is_global_rank_zero():
#             return

#         for name in os.listdir(self.dirpath):
#             if name.endswith(".unfinished"):
#                 unfinished_marker = os.path.join(self.dirpath, name)
#                 ckpt_name = name.replace(".unfinished", "")
#                 ckpt_path = os.path.join(self.dirpath, ckpt_name)
#                 logger.warning(f"Removing unfinished checkpoint: {ckpt_path}")
#                 if os.path.isdir(ckpt_path):
#                     shutil.rmtree(ckpt_path, ignore_errors=True)
#                 try:
#                     os.remove(unfinished_marker)
#                 except Exception:
#                     pass

#     def _save_rank_shard(
#         self,
#         trainer: Trainer,
#         pl_module: LightningModule,
#         checkpoint_dir: str,
#         ema: bool = False,
#     ):
#         """
#         Save this rank's shard independently (runs in background thread).
#         No barriers, no coordination.
#         """
#         rank = self._get_rank()

#         try:
#             # Create rank-specific directory
#             rank_dir = os.path.join(checkpoint_dir, f"rank_{rank}")
#             os.makedirs(rank_dir, exist_ok=True)

#             # Mark as unfinished (per-rank marker)
#             marker_path = os.path.join(rank_dir, ".unfinished")
#             with open(marker_path, "w") as f:
#                 f.write(f"{rank}\n")

#             logger.info(f"[rank {rank}] Starting checkpoint save to {rank_dir}")

#             # Get this rank's state dict
#             # For FSDP/DDP, this will be the local shard
#             try:
#                 if hasattr(pl_module, "sharded_state_dict"):
#                     # Megatron-style sharded state dict
#                     state_dict = pl_module.sharded_state_dict(prefix="")
#                 else:
#                     # Regular PyTorch state dict (will be full model on rank 0, shard on DDP)
#                     state_dict = pl_module.state_dict()

#                 # Save the state dict
#                 model_path = os.path.join(rank_dir, "model.pt")
#                 torch.save(state_dict, model_path)
#                 logger.info(f"[rank {rank}] Saved model state to {model_path}")

#             except Exception as e:
#                 logger.exception(f"[rank {rank}] Failed to save model state: {e}")

#             # Save optimizer state (each rank's optimizer shard)
#             try:
#                 if trainer.strategy.optimizer is not None:
#                     opt_states = []
#                     optimizers = trainer.optimizers
#                     if not isinstance(optimizers, list):
#                         optimizers = [optimizers]

#                     for opt in optimizers:
#                         opt_states.append(opt.state_dict())

#                     opt_path = os.path.join(rank_dir, "optimizer.pt")
#                     torch.save(opt_states, opt_path)
#                     logger.info(f"[rank {rank}] Saved optimizer state to {opt_path}")
#             except Exception as e:
#                 logger.warning(f"[rank {rank}] Failed to save optimizer state: {e}")

#             # Save trainer state (lightweight metadata)
#             try:
#                 trainer_state = {
#                     "global_step": trainer.global_step,
#                     "current_epoch": trainer.current_epoch,
#                     "rank": rank,
#                     "world_size": self._get_world_size(),
#                 }
#                 trainer_path = os.path.join(rank_dir, "trainer_state.pt")
#                 torch.save(trainer_state, trainer_path)
#                 logger.info(f"[rank {rank}] Saved trainer state")
#             except Exception as e:
#                 logger.warning(f"[rank {rank}] Failed to save trainer state: {e}")

#             # Save EMA if needed (rank 0 only)
#             if ema and hasattr(pl_module, "ema_model") and is_global_rank_zero():
#                 try:
#                     ema_state = pl_module.ema_model.state_dict()
#                     ema_path = os.path.join(rank_dir, "ema.pt")
#                     torch.save(ema_state, ema_path)
#                     logger.info(f"[rank {rank}] Saved EMA state")
#                 except Exception as e:
#                     logger.warning(f"[rank {rank}] Failed to save EMA: {e}")

#             # Save context (rank 0 only)
#             if is_global_rank_zero() and self.context:
#                 try:
#                     context_path = os.path.join(checkpoint_dir, "context.yaml")
#                     with open(context_path, "w") as f:
#                         yaml.safe_dump(self.context, f)
#                     logger.info(f"[rank {rank}] Saved context")
#                 except Exception as e:
#                     logger.warning(f"[rank {rank}] Failed to save context: {e}")

#             # Remove unfinished marker
#             try:
#                 os.remove(marker_path)
#                 logger.info(f"[rank {rank}] Checkpoint complete: {rank_dir}")
#             except Exception:
#                 pass

#             # Create symlink (rank 0 only)
#             if self.save_last and is_global_rank_zero():
#                 try:
#                     symlink_path = os.path.join(self.dirpath, "last")
#                     target = os.path.basename(checkpoint_dir)
#                     if os.path.islink(symlink_path) or os.path.exists(symlink_path):
#                         os.remove(symlink_path)
#                     os.symlink(target, symlink_path)
#                     logger.info(f"[rank {rank}] Updated 'last' symlink")
#                 except Exception as e:
#                     logger.warning(f"[rank {rank}] Failed to create symlink: {e}")

#         except Exception as e:
#             logger.exception(f"[rank {rank}] Checkpoint save failed: {e}")
#         finally:
#             with self._thread_lock:
#                 self._save_thread = None

#     def _trigger_async_save(
#         self,
#         trainer: Trainer,
#         pl_module: LightningModule,
#         checkpoint_dir: str,
#         ema: bool = False,
#     ):
#         """Launch async save in background thread (per rank, no coordination)."""
#         rank = self._get_rank()

#         with self._thread_lock:
#             # If previous save still running, wait for it
#             if self._save_thread is not None and self._save_thread.is_alive():
#                 logger.warning(
#                     f"[rank {rank}] Previous checkpoint still saving, waiting..."
#                 )
#                 self._save_thread.join(timeout=300)  # Wait up to 5 minutes
#                 if self._save_thread.is_alive():
#                     logger.error(f"[rank {rank}] Previous checkpoint timed out!")
#                     return

#             # Launch new save thread
#             self._save_thread = threading.Thread(
#                 target=self._save_rank_shard,
#                 args=(trainer, pl_module, checkpoint_dir, ema),
#                 daemon=False,  # Don't use daemon so it completes even if main exits
#             )
#             self._save_thread.start()
#             logger.info(f"[rank {rank}] Launched async checkpoint thread")

#     # ---------------------
#     # Lightning hooks
#     # ---------------------
#     def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
#         # Only rank 0 cleans up, no coordination needed
#         if is_global_rank_zero():
#             self._remove_unfinished_checkpoints()
#             logger.info("Cleaned unfinished checkpoints")

#     def on_train_batch_end(
#         self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx
#     ):
#         if trainer.global_step == 0:
#             return

#         if (trainer.global_step % self.save_interval) != 0:
#             return

#         checkpoint_dir = self._format_ckpt_path(trainer, ema=self.save_ema)
#         rank = self._get_rank()

#         logger.info(
#             f"[rank {rank}] Triggering async checkpoint at step {trainer.global_step}"
#         )

#         # Each rank independently launches its save thread
#         # No barriers, no waiting, returns immediately
#         self._trigger_async_save(trainer, pl_module, checkpoint_dir, ema=self.save_ema)

#         logger.info(f"[rank {rank}] Checkpoint triggered, continuing training")

#     def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
#         rank = self._get_rank()

#         # Wait for any pending async save to complete
#         with self._thread_lock:
#             if self._save_thread is not None and self._save_thread.is_alive():
#                 logger.info(
#                     f"[rank {rank}] Waiting for pending checkpoint to complete..."
#                 )
#                 self._save_thread.join(timeout=600)

#         # Final checkpoint (synchronous)
#         checkpoint_dir = self._format_ckpt_path(trainer, ema=self.save_ema)
#         logger.info(f"[rank {rank}] Saving final checkpoint")

#         self._save_rank_shard(trainer, pl_module, checkpoint_dir, ema=self.save_ema)

#         logger.info(f"[rank {rank}] Final checkpoint complete")


### new
import logging
import os
import shutil
import threading
from typing import Any, Dict, Optional

import torch
import yaml
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

# Optional: import distributed helpers from Megatron-Core
try:
    from megatron.core import dist_checkpointing, mpu

    # Try to find megatron's sharded writers - support multiple possible import paths
    _MEGATRON_SAVE_SHARDED = None
    _MEGATRON_SHARDED_STATE_DICT = None
    try:
        # newer layout
        from megatron.core.dist_checkpointing.serialization import save_sharded_tensor

        _MEGATRON_SAVE_SHARDED = save_sharded_tensor
    except Exception:
        try:
            # alternative path
            from megatron.core.dist_checkpointing.serializers import save_sharded_tensor

            _MEGATRON_SAVE_SHARDED = save_sharded_tensor
        except Exception:
            _MEGATRON_SAVE_SHARDED = None

    try:
        from megatron.core.dist_checkpointing.mapping import ShardedStateDict

        _MEGATRON_SHARDED_STATE_DICT = ShardedStateDict
    except Exception:
        _MEGATRON_SHARDED_STATE_DICT = None

    def is_global_rank_zero():
        try:
            return mpu.get_data_parallel_rank() == 0
        except Exception:
            if not torch.distributed.is_initialized():
                return True
            return torch.distributed.get_rank() == 0

    _HAS_DIST_CHECKPOINTING = True
except Exception:
    mpu = None
    dist_checkpointing = None
    _MEGATRON_SAVE_SHARDED = None
    _MEGATRON_SHARDED_STATE_DICT = None

    def is_global_rank_zero():
        if not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    _HAS_DIST_CHECKPOINTING = False

logger = logging.getLogger(__name__)


class AsyncModelCheckpoint(Callback):
    """
    Truly async checkpointing - each rank saves independently without coordination.

    If Megatron-Core is available, attempt to use its sharded serialization
    functions per-rank WITHOUT calling the distributed orchestration (so NO NCCL/collectives).
    Otherwise fall back to torch.save rank-local behavior.

    Parameters
    ----------
    dirpath: str
        Top-level checkpoint directory (shared filesystem).
    filename: str
        Checkpoint basename format, accepts {epoch} and {step}.
    save_interval: int
        Save every N global steps.
    save_last: bool
        Update 'last' symlink on rank 0.
    save_ema: bool
        If True and pl_module has `ema_model`, save it on rank 0.
    context: dict
        Optional metadata to write (rank 0 only).
    local_tmp_dir: Optional[str]
        If provided, write the rank-local checkpoint into this local dir first (fast),
        then copy to the shared `dirpath`. Use e.g. "/local_scratch".
    """

    def __init__(
        self,
        dirpath: str,
        filename: str = "epoch={epoch}-step={step}",
        save_interval: int = 1000,
        save_last: bool = True,
        save_ema: bool = False,
        context: Optional[Dict[str, Any]] = None,
        local_tmp_dir: Optional[str] = None,
    ):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.save_interval = int(save_interval)
        self.save_last = save_last
        self.save_ema = save_ema
        self.context = context or {}
        self.local_tmp_dir = local_tmp_dir

        # Track async save threads per rank
        self._save_thread: Optional[threading.Thread] = None
        self._thread_lock = threading.Lock()

        os.makedirs(self.dirpath, exist_ok=True)

    def _format_ckpt_basename(self, trainer: Trainer, ema=False) -> str:
        fname = self.filename.format(
            epoch=trainer.current_epoch, step=trainer.global_step
        )
        if ema:
            fname += "-EMA"
        return fname

    def _format_ckpt_path(self, trainer: Trainer, ema=False) -> str:
        base = self._format_ckpt_basename(trainer, ema=ema)
        # this is the per-checkpoint directory under self.dirpath
        return os.path.join(self.dirpath, base)

    def _get_rank(self) -> int:
        if torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
        return 0

    def _get_world_size(self) -> int:
        if torch.distributed.is_initialized():
            return int(torch.distributed.get_world_size())
        return 1

    def _remove_unfinished_checkpoints(self):
        """Only global rank 0 cleans up unfinished checkpoints.
        Search recursively for '.unfinished' marker files and remove the parent checkpoint dir.
        """
        if not is_global_rank_zero():
            return

        # Walk the dirpath for any '.unfinished' markers
        for root, dirs, files in os.walk(self.dirpath):
            if ".unfinished" in files:
                marker_path = os.path.join(root, ".unfinished")
                parent = os.path.dirname(root)
                # determine checkpoint base dir (one level up if rank_<n> is last level)
                ckpt_dir = (
                    parent
                    if parent and parent != os.path.abspath(self.dirpath)
                    else root
                )

                logger.warning(
                    f"Removing unfinished checkpoint dir: {ckpt_dir} (marker: {marker_path})"
                )
                try:
                    if os.path.isdir(ckpt_dir) and os.path.abspath(
                        ckpt_dir
                    ) != os.path.abspath(self.dirpath):
                        shutil.rmtree(ckpt_dir, ignore_errors=True)
                except Exception as e:
                    logger.exception(
                        f"Error removing unfinished checkpoint dir {ckpt_dir}: {e}"
                    )

                # remove the marker if it still exists
                try:
                    if os.path.exists(marker_path):
                        os.remove(marker_path)
                except Exception:
                    pass

    # -----------------------
    # Megatron local-save helpers
    # -----------------------
    def _megatron_save_local_shards(
        self, state_dict: Dict[str, Any], rank_dir: str
    ) -> bool:
        """
        Try to use Megatron-Core sharded writers to write local shards only.
        Returns True if this code path was used, False if fall-back is needed.
        IMPORTANT: This function must NOT call any torch.distributed collectives.
        """
        if not _HAS_DIST_CHECKPOINTING:
            return False

        if _MEGATRON_SAVE_SHARDED is None or _MEGATRON_SHARDED_STATE_DICT is None:
            # megatron present but import of writer or mapping failed.
            return False

        try:
            # Wrap into ShardedStateDict if available (Megatron expects mapping)
            # The ShardedStateDict constructor accepts a dict mapping names -> state_dicts.
            try:
                sharded = _MEGATRON_SHARDED_STATE_DICT({"model": state_dict})
            except Exception:
                # If wrapping fails, fallback to writing tensors individually using save_sharded_tensor
                sharded = None

            if sharded is not None:
                # Iterate and write each ShardedTensor locally using Megatron writer,
                # but do NOT call any global orchestration functions.
                for name, entry in sharded.items():
                    # entry is expected to be a ShardedTensor-like object; save locally with provided writer.
                    try:
                        # writer should accept (sharded_tensor, filepath, rank=..., world_size=...)
                        # but different versions have different signatures - try common forms.
                        target_path = os.path.join(rank_dir, f"{name}.bin")
                        try:
                            _MEGATRON_SAVE_SHARDED(
                                entry, target_path, rank=self._get_rank()
                            )
                        except TypeError:
                            # fallback signature without rank kwarg
                            _MEGATRON_SAVE_SHARDED(entry, target_path)
                    except Exception as e:
                        logger.exception(f"Megeatron writer failed for {name}: {e}")
                        # If any write fails, return False to fall back.
                        return False
                return True
            else:
                # No ShardedStateDict available: try to save each tensor value with save_sharded_tensor
                for key, val in state_dict.items():
                    target_path = os.path.join(rank_dir, f"{key}.bin")
                    try:
                        try:
                            _MEGATRON_SAVE_SHARDED(
                                val, target_path, rank=self._get_rank()
                            )
                        except TypeError:
                            _MEGATRON_SAVE_SHARDED(val, target_path)
                    except Exception as e:
                        logger.exception(
                            f"Megeatron writer failed for tensor {key}: {e}"
                        )
                        return False
                return True

        except Exception as e:
            logger.exception(f"Unexpected error in Megatron local save: {e}")
            return False

    def _fallback_torch_save(self, state_dict: Dict[str, Any], rank_dir: str) -> None:
        """Fallback to torch.save of the full state_dict into model.pt."""
        model_path = os.path.join(rank_dir, "model.pt")
        torch.save(state_dict, model_path)
        logger.info(
            f"[rank {self._get_rank()}] torch.save fallback saved model to {model_path}"
        )

    # -----------------------
    # Core per-rank save routine (invoked in thread)
    # -----------------------
    def _save_rank_shard(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint_dir: str,
        ema: bool = False,
    ):
        """
        Save this rank's shard independently (runs in background thread).
        No collectives called here.
        """
        rank = self._get_rank()

        try:
            # Ensure top-level checkpoint_dir exists
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Optionally use a local temp directory per-rank to reduce shared FS contention
            if self.local_tmp_dir:
                base_dir = os.path.join(
                    self.local_tmp_dir, os.path.basename(checkpoint_dir)
                )
            else:
                base_dir = checkpoint_dir

            rank_dir = os.path.join(base_dir, f"rank_{rank}")
            os.makedirs(rank_dir, exist_ok=True)

            # Mark as unfinished (per-rank marker)
            marker_path = os.path.join(rank_dir, ".unfinished")
            try:
                with open(marker_path, "w") as f:
                    f.write(f"{rank}\n")
            except Exception:
                logger.warning(
                    f"[rank {rank}] Could not write unfinished marker at {marker_path}"
                )

            logger.info(f"[rank {rank}] Starting checkpoint save to {rank_dir}")

            # Extract local state dict (Megatron-aware if possible)
            state_dict = None
            try:
                if hasattr(pl_module, "sharded_state_dict"):
                    state_dict = pl_module.sharded_state_dict(prefix="")
                else:
                    # Some LM frameworks store torch.nn.Module under pl_module.model or similar.
                    # Try best-effort to extract underlying module's state_dict if available.
                    try:
                        # If LightningModule wraps the real model in `pl_module.model`
                        model_obj = getattr(pl_module, "model", pl_module)
                        state_dict = model_obj.state_dict()
                    except Exception:
                        state_dict = pl_module.state_dict()
            except Exception as e:
                logger.exception(f"[rank {rank}] Could not get state_dict: {e}")
                state_dict = pl_module.state_dict()

            # Try Megatron local writer first (no collectives)
            used_megatron_writer = False
            try:
                if state_dict is not None and _HAS_DIST_CHECKPOINTING:
                    used_megatron_writer = self._megatron_save_local_shards(
                        state_dict, rank_dir
                    )
            except Exception as e:
                logger.exception(f"[rank {rank}] Megatron local writer raised: {e}")
                used_megatron_writer = False

            # If Megatron path not available or failed, fallback to torch.save
            if not used_megatron_writer:
                try:
                    self._fallback_torch_save(state_dict, rank_dir)
                except Exception as e:
                    logger.exception(f"[rank {rank}] Fallback torch.save failed: {e}")

            # Save optimizer state (each rank's optimizer shard) if available locally
            try:
                optimizers = getattr(trainer, "optimizers", None)
                if optimizers:
                    if not isinstance(optimizers, (list, tuple)):
                        optimizers = [optimizers]

                    opt_states = []
                    for opt in optimizers:
                        try:
                            if opt is not None:
                                opt_states.append(opt.state_dict())
                        except Exception as e_opt:
                            logger.warning(
                                f"[rank {rank}] Failed to get optimizer state for one optimizer: {e_opt}"
                            )

                    if opt_states:
                        opt_path = os.path.join(rank_dir, "optimizer.pt")
                        torch.save(opt_states, opt_path)
                        logger.info(
                            f"[rank {rank}] Saved optimizer state to {opt_path}"
                        )
                else:
                    logger.debug(
                        f"[rank {rank}] No trainer.optimizers available; skipping optimizer save."
                    )
            except Exception as e:
                logger.warning(f"[rank {rank}] Failed to save optimizer state: {e}")

            # Save trainer state (lightweight metadata)
            try:
                trainer_state = {
                    "global_step": int(trainer.global_step),
                    "current_epoch": int(trainer.current_epoch),
                    "rank": int(rank),
                    "world_size": int(self._get_world_size()),
                }
                trainer_path = os.path.join(rank_dir, "trainer_state.pt")
                torch.save(trainer_state, trainer_path)
                logger.info(f"[rank {rank}] Saved trainer state to {trainer_path}")
            except Exception as e:
                logger.warning(f"[rank {rank}] Failed to save trainer state: {e}")

            # Save EMA if needed (rank 0 only) — EMA is usually global; save once under checkpoint_dir
            if ema and hasattr(pl_module, "ema_model") and is_global_rank_zero():
                try:
                    ema_state = pl_module.ema_model.state_dict()
                    ema_target_dir = base_dir if self.local_tmp_dir else checkpoint_dir
                    ema_path = os.path.join(ema_target_dir, "ema.pt")
                    torch.save(ema_state, ema_path)
                    logger.info(f"[rank {rank}] Saved EMA state to {ema_path}")
                except Exception as e:
                    logger.warning(f"[rank {rank}] Failed to save EMA: {e}")

            # Save context (rank 0 only)
            if is_global_rank_zero() and self.context:
                try:
                    context_target_dir = (
                        base_dir if self.local_tmp_dir else checkpoint_dir
                    )
                    context_path = os.path.join(context_target_dir, "context.yaml")
                    with open(context_path, "w") as f:
                        yaml.safe_dump(self.context, f)
                    logger.info(f"[rank {rank}] Saved context to {context_path}")
                except Exception as e:
                    logger.warning(f"[rank {rank}] Failed to save context: {e}")

            # Remove unfinished marker
            try:
                if os.path.exists(marker_path):
                    os.remove(marker_path)
                logger.info(f"[rank {rank}] Checkpoint complete: {rank_dir}")
            except Exception:
                pass

            # If we wrote to local_tmp_dir, copy the rank_dir to shared checkpoint_dir
            if self.local_tmp_dir:
                try:
                    # Ensure shared checkpoint_dir exists
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    dest_rank_dir = os.path.join(checkpoint_dir, f"rank_{rank}")
                    # Use rsync-like behavior: copytree -> replace
                    if os.path.exists(dest_rank_dir):
                        shutil.rmtree(dest_rank_dir, ignore_errors=True)
                    shutil.copytree(rank_dir, dest_rank_dir)
                    logger.info(
                        f"[rank {rank}] Copied local checkpoint {rank_dir} -> {dest_rank_dir}"
                    )
                    # Optionally remove local copy to save space
                    try:
                        shutil.rmtree(rank_dir, ignore_errors=True)
                    except Exception:
                        pass
                except Exception as e:
                    logger.exception(
                        f"[rank {rank}] Failed to copy local checkpoint to shared fs: {e}"
                    )

            # Create/refresh 'last' symlink (rank 0 only)
            if self.save_last and is_global_rank_zero():
                try:
                    symlink_path = os.path.join(self.dirpath, "last")
                    target = os.path.abspath(checkpoint_dir)
                    if os.path.islink(symlink_path) or os.path.exists(symlink_path):
                        os.remove(symlink_path)
                    os.symlink(target, symlink_path)
                    logger.info(f"[rank {rank}] Updated 'last' symlink -> {target}")
                except Exception as e:
                    logger.warning(f"[rank {rank}] Failed to create symlink: {e}")

        except Exception as e:
            logger.exception(f"[rank {rank}] Checkpoint save failed: {e}")
        finally:
            with self._thread_lock:
                # clear thread reference so new saves can be launched
                self._save_thread = None

    def _trigger_async_save(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint_dir: str,
        ema: bool = False,
    ):
        """Launch async save in background thread (per rank, no coordination)."""
        rank = self._get_rank()

        with self._thread_lock:
            # If previous save still running, wait up to timeout for it to finish.
            if self._save_thread is not None and self._save_thread.is_alive():
                logger.warning(
                    f"[rank {rank}] Previous checkpoint still saving, waiting..."
                )
                self._save_thread.join(timeout=300)  # Wait up to 5 minutes
                if self._save_thread.is_alive():
                    logger.error(
                        f"[rank {rank}] Previous checkpoint timed out! Skipping this save."
                    )
                    return

            # Launch new save thread (non-daemon so it won't be killed prematurely)
            self._save_thread = threading.Thread(
                target=self._save_rank_shard,
                args=(trainer, pl_module, checkpoint_dir, ema),
                daemon=False,
            )
            self._save_thread.start()
            logger.info(f"[rank {rank}] Launched async checkpoint thread")

    # ---------------------
    # Lightning hooks
    # ---------------------
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        # Only rank 0 cleans up, no coordination needed
        if is_global_rank_zero():
            self._remove_unfinished_checkpoints()
            logger.info("Cleaned unfinished checkpoints")

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx
    ):
        # don't save at global_step == 0
        if trainer.global_step == 0:
            return

        if (trainer.global_step % self.save_interval) != 0:
            return

        checkpoint_dir = self._format_ckpt_path(trainer, ema=self.save_ema)
        rank = self._get_rank()

        logger.info(
            f"[rank {rank}] Triggering async checkpoint at step {trainer.global_step}"
        )

        # Each rank independently launches its save thread — no collectives called here.
        self._trigger_async_save(trainer, pl_module, checkpoint_dir, ema=self.save_ema)

        logger.info(f"[rank {rank}] Checkpoint triggered, continuing training")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        rank = self._get_rank()

        # Wait for any pending async save to complete
        with self._thread_lock:
            if self._save_thread is not None and self._save_thread.is_alive():
                logger.info(
                    f"[rank {rank}] Waiting for pending checkpoint to complete..."
                )
                self._save_thread.join(timeout=600)

        # Final checkpoint (synchronous)
        checkpoint_dir = self._format_ckpt_path(trainer, ema=self.save_ema)
        logger.info(f"[rank {rank}] Saving final checkpoint (synchronous)")
        # Do a synchronous save to ensure final checkpoint exists
        self._save_rank_shard(trainer, pl_module, checkpoint_dir, ema=self.save_ema)
        logger.info(f"[rank {rank}] Final checkpoint complete")
