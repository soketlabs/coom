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
##All rank
# import logging
# import os
# import shutil
# import threading
# from typing import Any, Dict, Optional

# import torch
# import yaml
# from lightning import LightningModule, Trainer
# from lightning.pytorch.callbacks import Callback

# logger = logging.getLogger(__name__)

# # ----------------------
# # Megatron-Core imports
# # ----------------------
# _HAS_DIST_CHECKPOINTING = False
# _HAS_ASYNC_SAVE = False
# _dist_checkpointing = None
# _mpu = None
# _AsyncCallsQueue = None
# _AsyncRequest = None

# try:
#     from megatron.core import dist_checkpointing as _dist_checkpointing
#     from megatron.core import mpu as _mpu

#     _HAS_DIST_CHECKPOINTING = True
#     logger.info("Megatron-Core dist_checkpointing loaded")

#     # Try to import async utilities
#     try:
#         from megatron.core.dist_checkpointing.strategies.async_utils import (
#             AsyncCallsQueue,
#             AsyncRequest,
#         )
#         from megatron.core.dist_checkpointing.strategies.torch import (
#             TorchDistSaveShardedStrategy as _TorchDistSaveShardedStrategy,
#         )

#         _AsyncCallsQueue = AsyncCallsQueue
#         _AsyncRequest = AsyncRequest
#         _HAS_ASYNC_SAVE = True
#         logger.info("Megatron-Core async save utilities loaded")
#     except ImportError:
#         try:
#             # Alternative import path
#             from megatron.core.dist_checkpointing.core import (
#                 AsyncCallsQueue,
#                 AsyncRequest,
#             )

#             _AsyncCallsQueue = AsyncCallsQueue
#             _AsyncRequest = AsyncRequest
#             _HAS_ASYNC_SAVE = True
#             logger.info("Megatron-Core async save utilities loaded (alt path)")
#         except ImportError:
#             logger.warning("Megatron async save not available, will use sync mode")
#             _HAS_ASYNC_SAVE = False

# except ImportError:
#     logger.warning("Megatron-Core not available, using fallback checkpoint mode")
#     _HAS_DIST_CHECKPOINTING = False


# def is_global_rank_zero() -> bool:
#     """Return True if this process is global rank 0 (tensor/pipeline parallel aware)."""
#     try:
#         if _mpu is not None:
#             # For Megatron models with tensor/pipeline parallelism
#             return _mpu.get_data_parallel_rank() == 0
#     except Exception:
#         pass

#     if not torch.distributed.is_initialized():
#         return True
#     return torch.distributed.get_rank() == 0


# def safe_barrier(timeout_sec: float = 300):
#     """
#     Safe distributed barrier that catches exceptions.
#     Uses store-based barrier for better error handling.
#     """
#     if not torch.distributed.is_initialized():
#         return

#     try:
#         # Use store-based barrier with timeout for better error handling
#         rank = torch.distributed.get_rank()
#         world_size = torch.distributed.get_world_size()

#         # Create a unique key for this barrier
#         import time

#         barrier_key = f"checkpoint_barrier_{time.time()}"

#         store = torch.distributed.distributed_c10d._get_default_store()

#         # Each rank sets its key
#         store.set(f"{barrier_key}_rank_{rank}", "1")

#         # Rank 0 waits for all ranks
#         if rank == 0:
#             deadline = time.time() + timeout_sec
#             for r in range(world_size):
#                 while time.time() < deadline:
#                     try:
#                         store.get(f"{barrier_key}_rank_{r}")
#                         break
#                     except Exception:
#                         time.sleep(0.1)
#                 else:
#                     logger.warning(f"Barrier timeout waiting for rank {r}")
#                     raise TimeoutError(f"Barrier timeout waiting for rank {r}")

#             # Signal completion
#             store.set(f"{barrier_key}_complete", "1")
#         else:
#             # Other ranks wait for completion signal
#             deadline = time.time() + timeout_sec
#             while time.time() < deadline:
#                 try:
#                     store.get(f"{barrier_key}_complete")
#                     break
#                 except Exception:
#                     time.sleep(0.1)
#             else:
#                 logger.warning(f"[rank {rank}] Barrier timeout")
#                 raise TimeoutError("Barrier timeout")

#     except Exception as e:
#         logger.warning(f"Barrier failed: {e}, attempting standard barrier")
#         try:
#             torch.distributed.barrier()
#         except Exception as e2:
#             logger.error(f"Standard barrier also failed: {e2}")


# # ----------------------
# # AsyncModelCheckpoint
# # ----------------------
# class AsyncModelCheckpoint(Callback):
#     """
#     Advanced async checkpointing with Megatron-Core sharded distributed saving.

#     ALL RANKS SAVE - but Megatron handles sharding automatically:
#     - For Megatron models: Uses sharded_state_dict() - each rank saves only its shard
#     - For DDP models: Wraps state_dict in ShardedTensor format - automatic deduplication
#     - Megatron dist_checkpointing.save() coordinates across ranks WITHOUT collectives

#     Features:
#     - Megatron-Core async distributed checkpointing (true async, no barriers)
#     - Automatic shard deduplication (no wasted storage)
#     - Thread-based async saves with proper finalization
#     - EMA support
#     - Context/metadata saving
#     - Automatic cleanup of incomplete checkpoints

#     Parameters
#     ----------
#     dirpath : str
#         Checkpoint directory path (shared filesystem)
#     filename : str
#         Checkpoint name format (supports {epoch}, {step})
#     save_interval : int
#         Save every N training steps
#     save_last : bool
#         Maintain 'last' symlink to latest checkpoint
#     save_ema : bool
#         Save EMA model weights (rank 0 only)
#     context : dict
#         Additional metadata to save
#     local_tmp_dir : str, optional
#         Local fast storage for temporary writes before copying to shared FS
#     async_save : bool
#         Use async save if available. Default True.
#     """

#     def __init__(
#         self,
#         dirpath: str,
#         filename: str = "epoch={epoch}-step={step}",
#         save_interval: int = 1000,
#         save_last: bool = True,
#         save_ema: bool = False,
#         context: Optional[Dict[str, Any]] = None,
#         local_tmp_dir: Optional[str] = None,
#         async_save: bool = True,
#     ):
#         super().__init__()
#         self.dirpath = dirpath
#         self.filename = filename
#         self.save_interval = int(save_interval)
#         self.save_last = save_last
#         self.save_ema = save_ema
#         self.context = context or {}
#         self.local_tmp_dir = local_tmp_dir
#         self.async_save = async_save and _HAS_ASYNC_SAVE

#         # Track async operations
#         self._async_request = None
#         self._async_queue = None
#         self._local_thread: Optional[threading.Thread] = None
#         self._thread_lock = threading.Lock()

#         # Initialize save strategy for async support
#         self._save_strategy = None
#         if self.async_save and _TorchDistSaveShardedStrategy is not None:
#             try:
#                 self._save_strategy = _TorchDistSaveShardedStrategy(
#                     backend="torch_dist",
#                     version=1,
#                 )
#                 logger.info("Initialized async-capable save strategy")
#             except Exception as e:
#                 logger.warning(f"Failed to initialize save strategy: {e}")
#                 self._save_strategy = None

#         os.makedirs(self.dirpath, exist_ok=True)

#         # Log configuration
#         if _HAS_DIST_CHECKPOINTING:
#             mode = "async" if self.async_save else "sync"
#             logger.info(f"Using Megatron-Core distributed checkpointing ({mode} mode)")
#             logger.info("All ranks will save, Megatron handles automatic sharding")
#         else:
#             logger.info(
#                 "WARNING: Megatron not available, falling back to per-rank torch.save"
#             )
#             logger.info("This will result in duplicate model saves across ranks")

#     # ----------------------
#     # Utilities
#     # ----------------------
#     def _format_ckpt_basename(self, trainer: Trainer, ema=False) -> str:
#         fname = self.filename.format(
#             epoch=trainer.current_epoch, step=trainer.global_step
#         )
#         if ema:
#             fname += "-EMA"
#         return fname

#     def _format_ckpt_path(self, trainer: Trainer, ema=False) -> str:
#         return os.path.join(self.dirpath, self._format_ckpt_basename(trainer, ema=ema))

#     def _get_rank(self) -> int:
#         if torch.distributed.is_initialized():
#             return int(torch.distributed.get_rank())
#         return 0

#     def _get_world_size(self) -> int:
#         if torch.distributed.is_initialized():
#             return int(torch.distributed.get_world_size())
#         return 1

#     def _unfinished_marker(self, checkpoint_dir: str) -> str:
#         return os.path.join(checkpoint_dir, ".unfinished")

#     def _remove_unfinished_checkpoints(self):
#         """Clean up incomplete checkpoints (rank 0 only)."""
#         if not is_global_rank_zero():
#             return

#         for root, dirs, files in os.walk(self.dirpath):
#             if ".unfinished" in files:
#                 marker_path = os.path.join(root, ".unfinished")
#                 # Determine checkpoint directory to remove
#                 ckpt_dir = os.path.dirname(marker_path)
#                 if ckpt_dir == self.dirpath:
#                     continue  # Don't remove root dir

#                 logger.warning(f"Removing unfinished checkpoint: {ckpt_dir}")
#                 try:
#                     if os.path.isdir(ckpt_dir):
#                         shutil.rmtree(ckpt_dir, ignore_errors=True)
#                     try:
#                         os.remove(marker_path)
#                     except Exception:
#                         pass
#                 except Exception as e:
#                     logger.exception(f"Failed to remove {ckpt_dir}: {e}")

#     # ----------------------
#     # Async finalization
#     # ----------------------
#     def _finalize_async_save(
#         self, timeout: Optional[float] = 600, from_worker: bool = False
#     ):
#         """
#         Wait for any pending async save to complete.

#         Parameters
#         ----------
#         timeout : float
#             Timeout in seconds
#         from_worker : bool
#             If True, called from within worker thread (don't try to join self)
#         """
#         rank = self._get_rank()

#         with self._thread_lock:
#             # Finalize Megatron async request
#             if self._async_request is not None:
#                 logger.info(f"[rank {rank}] Waiting for Megatron async save...")
#                 try:
#                     # Different Megatron versions have different finalization methods
#                     if hasattr(self._async_request, "wait"):
#                         self._async_request.wait(timeout=timeout)
#                     elif hasattr(self._async_request, "result"):
#                         self._async_request.result(timeout=timeout)
#                     elif callable(self._async_request):
#                         # Some versions return a callable finalizer
#                         self._async_request()
#                     logger.info(f"[rank {rank}] Megatron async save completed")
#                 except Exception as e:
#                     logger.exception(f"[rank {rank}] Async finalization failed: {e}")
#                 finally:
#                     self._async_request = None

#             # Finalize local thread (but NOT if we're calling from within that thread!)
#             if not from_worker:
#                 if self._local_thread is not None and self._local_thread.is_alive():
#                     logger.info(f"[rank {rank}] Waiting for local save thread...")
#                     self._local_thread.join(timeout=timeout or 300)
#                     if self._local_thread.is_alive():
#                         logger.error(f"[rank {rank}] Local save thread timed out!")
#                     self._local_thread = None

#     # ----------------------
#     # Create sharded state dict wrapper
#     # ----------------------
#     def _create_sharded_state_dict(self, pl_module: LightningModule) -> Dict[str, Any]:
#         """
#         Create a sharded state dict that Megatron can use to deduplicate.

#         For Megatron models: Use native sharded_state_dict()
#         For DDP models: Wrap in ShardedTensor format so Megatron knows it's replicated
#         """
#         rank = self._get_rank()

#         # Try Megatron's native sharded_state_dict first
#         if hasattr(pl_module, "sharded_state_dict"):
#             try:
#                 sharded_sd = pl_module.sharded_state_dict(prefix="")
#                 logger.info(f"[rank {rank}] Using native sharded_state_dict from model")
#                 return sharded_sd
#             except Exception as e:
#                 logger.warning(f"[rank {rank}] sharded_state_dict() failed: {e}")

#         # Fallback: Create sharded state dict from regular state dict
#         # Extract model state dict
#         model = pl_module
#         if hasattr(pl_module, "model"):
#             model = pl_module.model

#         # Unwrap DDP if needed
#         from torch.nn.parallel import DistributedDataParallel as DDP

#         if isinstance(model, DDP):
#             model = model.module
#             logger.debug(f"[rank {rank}] Unwrapped DDP module")

#         state_dict = model.state_dict()

#         # Try to wrap in Megatron's ShardedTensor format
#         try:
#             from megatron.core.dist_checkpointing.mapping import (
#                 LocalNonpersitentObject,
#                 ShardedTensor,
#             )

#             # For DDP, all ranks have identical copies
#             # Mark tensors as replicated so Megatron saves only one copy
#             sharded_state_dict = {}

#             for key, tensor in state_dict.items():
#                 if isinstance(tensor, torch.Tensor):
#                     # Create a ShardedTensor that indicates this is replicated across all ranks
#                     # Only rank 0 will actually save it
#                     sharded_state_dict[key] = ShardedTensor.from_rank_offsets(
#                         key=key,
#                         tensor=tensor,
#                         replica_id=rank,  # Mark which replica this is
#                         # Tell Megatron this is replicated - only save from rank 0
#                         prepend_axis_num=0,
#                     )
#                 else:
#                     # Non-tensor data
#                     sharded_state_dict[key] = tensor

#             logger.info(
#                 f"[rank {rank}] Wrapped state_dict in ShardedTensor format (replicated)"
#             )
#             return sharded_state_dict

#         except ImportError:
#             logger.warning(
#                 f"[rank {rank}] Cannot import ShardedTensor, returning raw state_dict"
#             )
#             # Return raw state dict - Megatron will save it as-is from each rank
#             return {"model": state_dict}
#         except Exception as e:
#             logger.exception(
#                 f"[rank {rank}] Failed to create ShardedTensor wrapper: {e}"
#             )
#             return {"model": state_dict}

#     # ----------------------
#     # Megatron distributed async save
#     # ----------------------
#     def _megatron_async_save(
#         self,
#         pl_module: LightningModule,
#         checkpoint_dir: str,
#         from_worker: bool = False,
#     ) -> bool:
#         """
#         Use Megatron-Core's distributed save with proper error handling.
#         All ranks participate, Megatron handles sharding/deduplication automatically.

#         Parameters
#         ----------
#         pl_module : LightningModule
#             The model to checkpoint
#         checkpoint_dir : str
#             Directory to save checkpoint
#         from_worker : bool
#             If True, called from within worker thread

#         Returns
#         -------
#         bool
#             True if successful, False if should fall back.
#         """
#         if not _HAS_DIST_CHECKPOINTING or _dist_checkpointing is None:
#             return False

#         rank = self._get_rank()

#         # Ensure previous save is complete (but don't join self if in worker thread)
#         self._finalize_async_save(timeout=30.0, from_worker=from_worker)

#         try:
#             # Create checkpoint directory (rank 0)
#             if is_global_rank_zero():
#                 os.makedirs(checkpoint_dir, exist_ok=True)
#                 # Create unfinished marker
#                 marker = self._unfinished_marker(checkpoint_dir)
#                 with open(marker, "w") as f:
#                     f.write(f"started at step {self._get_rank()}\n")

#             # Brief barrier to ensure directory exists
#             # This is necessary - Megatron will do more collectives anyway
#             logger.info(f"[rank {rank}] About to call barrier...")
#             safe_barrier()
#             logger.info(f"[rank {rank}] Barrier complete")

#             # Get sharded state dict (all ranks)
#             logger.info(f"[rank {rank}] Getting sharded state dict...")
#             sharded_state_dict = self._create_sharded_state_dict(pl_module)
#             logger.info(f"[rank {rank}] Got sharded state dict")

#             # Log what we're saving
#             num_tensors = sum(
#                 1 for v in sharded_state_dict.values() if isinstance(v, torch.Tensor)
#             )
#             logger.info(
#                 f"[rank {rank}] Starting Megatron distributed save "
#                 f"({num_tensors} tensors) to {checkpoint_dir}"
#             )

#             # Call Megatron save with proper API for your version
#             try:
#                 if self.async_save:
#                     # Try async_sharded_save parameter first (correct parameter name!)
#                     logger.info(
#                         f"[rank {rank}] Calling dist_checkpointing.save with async_sharded_save=True..."
#                     )
#                     try:
#                         self._async_request = _dist_checkpointing.save(
#                             sharded_state_dict=sharded_state_dict,
#                             checkpoint_dir=checkpoint_dir,
#                             async_sharded_save=True,  # ← Correct parameter name!
#                         )
#                         logger.info(
#                             f"[rank {rank}] ✓ Async save launched (non-blocking), "
#                             "training continues..."
#                         )
#                     except TypeError as e:
#                         logger.warning(
#                             f"[rank {rank}] async_sharded_save parameter failed: {e}"
#                         )
#                         # Fallback: try with strategy-based async
#                         if (
#                             "async_sharded_save" in str(e)
#                             and self._save_strategy is not None
#                         ):
#                             logger.info(
#                                 f"[rank {rank}] Trying strategy-based async save"
#                             )
#                             self._async_request = self._save_strategy.async_save(
#                                 sharded_state_dict, checkpoint_dir
#                             )
#                             logger.info(
#                                 f"[rank {rank}] ✓ Strategy-based async save launched"
#                             )
#                         else:
#                             raise
#                 else:
#                     # Synchronous save (compatible with all versions)
#                     logger.info(
#                         f"[rank {rank}] Calling dist_checkpointing.save (synchronous)..."
#                     )
#                     _dist_checkpointing.save(
#                         sharded_state_dict=sharded_state_dict,
#                         checkpoint_dir=checkpoint_dir,
#                     )
#                     logger.info(f"[rank {rank}] ✓ Synchronous save complete")

#             except RuntimeError as e:
#                 # NCCL errors - ranks may be out of sync
#                 if "NCCL" in str(e):
#                     logger.error(
#                         f"[rank {rank}] NCCL error during save: {e}\n"
#                         "This usually means:\n"
#                         "1. Ranks are out of sync (one rank failed/hung)\n"
#                         "2. NCCL backend not properly initialized\n"
#                         "3. Network issues between nodes\n"
#                         "Falling back to per-rank save..."
#                     )
#                     # Try to recover NCCL state
#                     try:
#                         if torch.distributed.is_initialized():
#                             torch.cuda.synchronize()
#                     except Exception:
#                         pass
#                     return False
#                 else:
#                     raise

#             # Remove unfinished marker (rank 0, only for sync saves)
#             if is_global_rank_zero() and self._async_request is None:
#                 try:
#                     marker = self._unfinished_marker(checkpoint_dir)
#                     if os.path.exists(marker):
#                         os.remove(marker)
#                     logger.info(f"[rank {rank}] Removed unfinished marker")
#                 except Exception:
#                     pass

#             return True

#         except Exception as e:
#             logger.exception(f"[rank {rank}] Megatron distributed save failed: {e}")
#             self._async_request = None

#             # Try to recover from error state
#             try:
#                 if torch.distributed.is_initialized():
#                     torch.cuda.synchronize()
#             except Exception:
#                 pass

#             return False

#     # ----------------------
#     # Fallback: per-rank torch.save (NOT RECOMMENDED)
#     # ----------------------
#     def _fallback_save_worker(
#         self,
#         trainer: Trainer,
#         pl_module: LightningModule,
#         checkpoint_dir: str,
#         ema: bool,
#     ):
#         """
#         Fallback when Megatron not available.
#         WARNING: This saves full model on each rank (wasteful for DDP).
#         """
#         rank = self._get_rank()

#         try:
#             os.makedirs(checkpoint_dir, exist_ok=True)
#             rank_dir = os.path.join(checkpoint_dir, f"rank_{rank}")
#             os.makedirs(rank_dir, exist_ok=True)

#             # Unfinished marker
#             marker = os.path.join(rank_dir, ".unfinished")
#             with open(marker, "w") as f:
#                 f.write(f"{rank}\n")

#             logger.warning(
#                 f"[rank {rank}] Using FALLBACK per-rank save "
#                 "(Megatron not available) - may result in duplicates!"
#             )

#             # Extract model
#             model = pl_module
#             if hasattr(pl_module, "model"):
#                 model = pl_module.model

#             from torch.nn.parallel import DistributedDataParallel as DDP

#             if isinstance(model, DDP):
#                 model = model.module

#             state_dict = model.state_dict()

#             # Save model (all ranks, will be duplicated in DDP)
#             model_path = os.path.join(rank_dir, "model.pt")
#             torch.save(state_dict, model_path)
#             model_size_mb = os.path.getsize(model_path) / (1024**2)
#             logger.info(f"[rank {rank}] ✓ Saved model ({model_size_mb:.2f} MB)")

#             # Save optimizer (all ranks)
#             try:
#                 optimizers = getattr(trainer, "optimizers", None)
#                 if optimizers:
#                     if not isinstance(optimizers, (list, tuple)):
#                         optimizers = [optimizers]

#                     opt_states = []
#                     for opt in optimizers:
#                         if opt is not None:
#                             try:
#                                 opt_states.append(opt.state_dict())
#                             except Exception as e:
#                                 logger.warning(f"[rank {rank}] Optimizer failed: {e}")

#                     if opt_states:
#                         opt_path = os.path.join(rank_dir, "optimizer.pt")
#                         torch.save(opt_states, opt_path)
#                         opt_size_mb = os.path.getsize(opt_path) / (1024**2)
#                         logger.info(
#                             f"[rank {rank}] ✓ Saved optimizer ({opt_size_mb:.2f} MB)"
#                         )
#             except Exception as e:
#                 logger.warning(f"[rank {rank}] Failed to save optimizer: {e}")

#             # Save trainer state
#             try:
#                 trainer_state = {
#                     "global_step": int(trainer.global_step),
#                     "current_epoch": int(trainer.current_epoch),
#                     "rank": int(rank),
#                     "world_size": int(self._get_world_size()),
#                 }
#                 trainer_path = os.path.join(rank_dir, "trainer_state.pt")
#                 torch.save(trainer_state, trainer_path)
#                 logger.info(f"[rank {rank}] ✓ Saved trainer state")
#             except Exception as e:
#                 logger.warning(f"[rank {rank}] Failed to save trainer state: {e}")

#             # EMA and context (rank 0 only)
#             if is_global_rank_zero():
#                 if ema and hasattr(pl_module, "ema_model"):
#                     try:
#                         ema_path = os.path.join(checkpoint_dir, "ema.pt")
#                         torch.save(pl_module.ema_model.state_dict(), ema_path)
#                         logger.info(f"[rank {rank}] ✓ Saved EMA")
#                     except Exception as e:
#                         logger.warning(f"[rank {rank}] EMA save failed: {e}")

#                 if self.context:
#                     try:
#                         ctx_path = os.path.join(checkpoint_dir, "context.yaml")
#                         with open(ctx_path, "w") as f:
#                             yaml.safe_dump(self.context, f)
#                         logger.info(f"[rank {rank}] ✓ Saved context")
#                     except Exception as e:
#                         logger.warning(f"[rank {rank}] Context save failed: {e}")

#             # Remove marker
#             try:
#                 if os.path.exists(marker):
#                     os.remove(marker)
#             except Exception:
#                 pass

#             # Update symlink (rank 0)
#             if self.save_last and is_global_rank_zero():
#                 try:
#                     symlink_path = os.path.join(self.dirpath, "last")
#                     target = os.path.basename(checkpoint_dir)
#                     if os.path.islink(symlink_path) or os.path.exists(symlink_path):
#                         os.remove(symlink_path)
#                     os.symlink(target, symlink_path)
#                     logger.info(f"[rank {rank}] ✓ Updated 'last' → {target}")
#                 except Exception as e:
#                     logger.warning(f"[rank {rank}] Symlink failed: {e}")

#             logger.info(f"[rank {rank}] Fallback checkpoint save complete")

#         except Exception as e:
#             logger.exception(f"[rank {rank}] Fallback save worker failed: {e}")

#     # ----------------------
#     # Main save trigger
#     # ----------------------
#     def _trigger_checkpoint(
#         self,
#         trainer: Trainer,
#         pl_module: LightningModule,
#         checkpoint_dir: str,
#         ema: bool = False,
#     ):
#         """
#         Trigger checkpoint save:
#         1. Try Megatron distributed save (all ranks, automatic sharding)
#         2. Fall back to per-rank save if Megatron unavailable

#         ALWAYS uses threading to ensure non-blocking behavior.
#         If previous checkpoint is still running, allows overlap (don't skip).
#         """
#         rank = self._get_rank()

#         # Check if previous save is still running (but DON'T wait or skip)
#         with self._thread_lock:
#             if self._local_thread is not None and self._local_thread.is_alive():
#                 logger.warning(
#                     f"[rank {rank}] Previous checkpoint still running "
#                     f"(thread: {self._local_thread.name}), launching new one anyway..."
#                 )
#                 # Don't skip! Allow overlapping checkpoints if needed
#                 # The old thread will finish eventually

#         # Launch checkpoint in background thread (GUARANTEED non-blocking)
#         logger.info(f"[rank {rank}] Launching checkpoint in background thread")

#         def _checkpoint_worker():
#             """Background worker that handles the actual checkpoint save."""
#             worker_rank = self._get_rank()
#             thread_name = threading.current_thread().name

#             try:
#                 logger.info(
#                     f"[rank {worker_rank}] [{thread_name}] Checkpoint worker started"
#                 )

#                 # Try Megatron distributed save (preferred)
#                 if _HAS_DIST_CHECKPOINTING:
#                     # Pass from_worker=True to prevent thread from joining itself
#                     logger.info(
#                         f"[rank {worker_rank}] [{thread_name}] Attempting Megatron save..."
#                     )
#                     success = self._megatron_async_save(
#                         pl_module,
#                         checkpoint_dir,
#                         from_worker=True,  # ← Important: prevents self-join
#                     )
#                     if success:
#                         logger.info(
#                             f"[rank {worker_rank}] [{thread_name}] "
#                             "✓ Megatron checkpoint complete (automatic sharding)"
#                         )

#                         # Save EMA and context separately (rank 0 only)
#                         if is_global_rank_zero():
#                             if ema and hasattr(pl_module, "ema_model"):
#                                 try:
#                                     ema_path = os.path.join(checkpoint_dir, "ema.pt")
#                                     torch.save(
#                                         pl_module.ema_model.state_dict(), ema_path
#                                     )
#                                     logger.info(
#                                         f"[rank {worker_rank}] [{thread_name}] Saved EMA"
#                                     )
#                                 except Exception as e:
#                                     logger.warning(
#                                         f"[rank {worker_rank}] [{thread_name}] EMA save failed: {e}"
#                                     )

#                             if self.context:
#                                 try:
#                                     ctx_path = os.path.join(
#                                         checkpoint_dir, "context.yaml"
#                                     )
#                                     with open(ctx_path, "w") as f:
#                                         yaml.safe_dump(self.context, f)
#                                     logger.info(
#                                         f"[rank {worker_rank}] [{thread_name}] Saved context"
#                                     )
#                                 except Exception as e:
#                                     logger.warning(
#                                         f"[rank {worker_rank}] [{thread_name}] Context save failed: {e}"
#                                     )

#                         return

#                 # Fallback: per-rank torch.save
#                 logger.warning(
#                     f"[rank {worker_rank}] [{thread_name}] Using fallback per-rank save"
#                 )
#                 self._fallback_save_worker(trainer, pl_module, checkpoint_dir, ema)
#                 logger.info(
#                     f"[rank {worker_rank}] [{thread_name}] ✓ Fallback save complete"
#                 )

#             except Exception as e:
#                 logger.exception(
#                     f"[rank {worker_rank}] [{thread_name}] Checkpoint worker failed: {e}"
#                 )
#             finally:
#                 # Clear thread reference when done
#                 with self._thread_lock:
#                     if self._local_thread is not None:
#                         # Only clear if we're the current thread
#                         if threading.current_thread() == self._local_thread:
#                             self._local_thread = None
#                             logger.info(
#                                 f"[rank {worker_rank}] [{thread_name}] Thread finalized"
#                             )

#         # Launch thread
#         with self._thread_lock:
#             new_thread = threading.Thread(
#                 target=_checkpoint_worker,
#                 daemon=False,
#                 name=f"checkpoint-rank{rank}-step{trainer.global_step}",
#             )
#             new_thread.start()
#             self._local_thread = new_thread  # Update to newest thread

#         logger.info(
#             f"[rank {rank}] ✓ Checkpoint launched in background, "
#             "training continues immediately"
#         )

#     # ----------------------
#     # Lightning hooks
#     # ----------------------
#     def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
#         """Clean up incomplete checkpoints at training start."""
#         if is_global_rank_zero():
#             self._remove_unfinished_checkpoints()
#             logger.info("Cleaned unfinished checkpoints")

#             if _HAS_DIST_CHECKPOINTING:
#                 mode = "async" if self.async_save else "sync"
#                 logger.info(f"Checkpointing mode: Megatron distributed ({mode})")
#                 logger.info("✓ All ranks save, automatic deduplication enabled")
#             else:
#                 logger.warning("Checkpointing mode: Fallback per-rank torch.save")
#                 logger.warning(
#                     "⚠ Storage will be wasted for DDP (consider installing Megatron)"
#                 )

#     def on_train_batch_end(
#         self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx
#     ):
#         """Trigger checkpoint save at specified intervals."""
#         if trainer.global_step == 0:
#             return

#         if (trainer.global_step % self.save_interval) != 0:
#             return

#         checkpoint_dir = self._format_ckpt_path(trainer, ema=self.save_ema)
#         rank = self._get_rank()

#         logger.info(f"[rank {rank}] Checkpoint trigger at step {trainer.global_step}")

#         self._trigger_checkpoint(trainer, pl_module, checkpoint_dir, ema=self.save_ema)

#         logger.info(f"[rank {rank}] Continuing training...")

#     def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
#         """Finalize all async saves and save final checkpoint."""
#         rank = self._get_rank()

#         logger.info(f"[rank {rank}] Training ended, finalizing checkpoints...")

#         # Wait for any pending async saves (from_worker=False since we're in main thread)
#         self._finalize_async_save(timeout=600, from_worker=False)

#         # Save final checkpoint (synchronous to ensure completion)
#         checkpoint_dir = self._format_ckpt_path(trainer, ema=self.save_ema)
#         logger.info(f"[rank {rank}] Saving final checkpoint (synchronous)")

#         # Force synchronous save for final checkpoint
#         original_async = self.async_save
#         self.async_save = False

#         try:
#             self._trigger_checkpoint(
#                 trainer, pl_module, checkpoint_dir, ema=self.save_ema
#             )

#             # Wait for thread to complete
#             if self._local_thread is not None and self._local_thread.is_alive():
#                 self._local_thread.join(timeout=600)
#         finally:
#             self.async_save = original_async

#         logger.info(f"[rank {rank}] All checkpoints saved, training complete")


# base megatron core
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.core.module import LightningModule
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.utilities.rank_zero import (
    rank_zero_info,
    rank_zero_only,
    rank_zero_warn,
)

# Megatron-Core imports (may raise if not installed in your env)
try:
    from megatron.core.dist_checkpointing import serialization as mcore_serialization
    from megatron.core.dist_checkpointing import strategies as mcore_strategies

    # Removed from megatron.core.dist_checkpointing.dict_utils import get_checkpoint_version
    from megatron.core.dist_checkpointing.strategies.base import AsyncRequest
except ImportError as e:
    mcore_serialization = None
    mcore_strategies = None
    rank_zero_warn(f"Megatron-Core dist_checkpointing not available. Error: {e}")


class AsyncModelCheckpoint(Callback):
    """
    Advanced async checkpointing with Megatron-Core sharded distributed saving.

    Leverages MCore strategies for high-performance, sharded I/O.

    Parameters
    ----------
    dirpath : str
        Checkpoint directory path (shared filesystem)
    filename : str
        Checkpoint name format (supports {step}, {global_step})
    save_interval : int
        Save every N training steps
    save_last : bool
        Maintain 'last' symlink to latest checkpoint (rank 0)
    context : dict
        Additional metadata to save (written to common part)
    local_tmp_dir : str | None
        If provided, write to a local fast path first then move to shared FS
    async_save : bool
        Use Megatron-Core async save features (default True)
    backend : str
        The MCore backend to use (e.g., 'torch_dist')
    """

    def __init__(
        self,
        dirpath: str,
        filename: str = "iter_{step:07d}",
        save_interval: int = 1000,
        save_last: bool = True,
        context: Optional[Dict[str, Any]] = None,
        local_tmp_dir: Optional[str] = None,
        async_save: bool = True,
        backend: str = "torch_dist",
        backend_version: int = 1,
    ):
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.save_interval = int(save_interval)
        self.save_last = save_last
        self.context = context or {}
        self.local_tmp_dir = Path(local_tmp_dir) if local_tmp_dir else None

        self.async_save_enabled = async_save and (mcore_serialization is not None)
        self.backend = backend
        self.backend_version = backend_version

        # Track outstanding async requests: list of tuples (thread, async_request, checkpoint_dir)
        self._outstanding = []
        # Protect access to _outstanding
        self._lock = threading.Lock()

        # Make sure checkpoint directory exists (rank 0 only)
        if rank_zero_only.rank == 0:
            self.dirpath.mkdir(parents=True, exist_ok=True)

        if mcore_serialization is None:
            raise RuntimeError(
                "Megatron-Core serialization module is required for this callback."
            )

    # ---------- Helper utilities ----------
    def _format_ckpt_dir(self, step: int) -> Path:
        name = self.filename.format(epoch=None, step=step, global_step=step)
        return self.dirpath / name

    @staticmethod
    def _is_megatron_model(pl_module: LightningModule) -> bool:
        # Check if the model exposes `sharded_state_dict()` helper
        return hasattr(pl_module, "sharded_state_dict")

    # ---------- MCore Interaction & Async Management ----------

    @rank_zero_only
    def _write_common_state(self, checkpoint_dir: Path, trainer: Trainer):
        """Save the non-sharded (common) part of the checkpoint (rank 0)."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        common_state = {
            "global_step": trainer.global_step,
            "epoch": trainer.current_epoch,
            **self.context,
            # Explicitly removed 'mcore_checkpoint_version'
        }

        common_file = checkpoint_dir / "common.pt"
        torch.save(common_state, str(common_file))

    def _finalize_async_save(
        self,
        async_request: AsyncRequest,
        source_dir: Path,
        target_dir: Path,
        thread_obj: threading.Thread,
    ):
        """Target function for background thread to finalize the MCore async request."""
        try:
            # wait() blocks until all ranks have completed their I/O
            async_request.wait()
            rank_zero_info(f"Finished async distributed save to {source_dir}.")

            # If using a local tmp dir, move to shared FS now (rank 0 only manages the move)
            if self.local_tmp_dir and rank_zero_only.rank == 0:
                rank_zero_info(
                    f"Moving checkpoint from local {source_dir} to shared {target_dir}"
                )
                # Use os.rename which is atomic if possible within the same filesystem
                if source_dir.rename(target_dir):
                    pass
                else:
                    shutil.move(source_dir, target_dir)

        except Exception as e:
            rank_zero_warn(
                f"Error in finalizing async checkpoint save for {target_dir}: {e}"
            )
        finally:
            # Remove this request from _outstanding list in a thread-safe manner
            with self._lock:
                # Find the current thread object in the list of outstanding operations and remove it
                for i, (thread, req, _) in enumerate(self._outstanding):
                    if thread is thread_obj:
                        self._outstanding.pop(i)
                        break

    def _schedule_async_save(
        self, sharded_state_dict: Dict[str, Any], source_dir: Path, target_dir: Path
    ):
        """
        Obtain an MCore SaveShardedStrategy and initiate an async save request.
        """
        try:
            # Get the appropriate strategy based on backend and version
            sharded_strategy = mcore_serialization.get_default_save_sharded_strategy(
                backend=self.backend, version=self.backend_version
            )
        except Exception as e:
            rank_zero_warn(
                f"Failed to get MCore save strategy: {e}. Falling back to synchronous if possible."
            )
            # Fallback path would go here
            return

        if not hasattr(sharded_strategy, "async_save"):
            rank_zero_warn(
                f"Strategy {type(sharded_strategy).__name__} does not support async save."
            )
            # Fallback path would go here
            return

        rank_zero_info(
            f"Initiating async distributed save to {source_dir} using {type(sharded_strategy).__name__}"
        )
        source_dir.mkdir(parents=True, exist_ok=True)

        # This initiates the non-blocking I/O operations across all ranks
        async_request: AsyncRequest = sharded_strategy.async_save(
            sharded_state_dict, str(source_dir)
        )

        # Start a background thread to wait for completion and move files if necessary
        # We pass the thread object itself to the finalizer for easy removal from the list
        thread = threading.Thread(
            target=self._finalize_async_save,
            args=(async_request, source_dir, target_dir, threading.current_thread()),
        )
        thread.daemon = True
        thread.start()

        with self._lock:
            self._outstanding.append((thread, async_request, target_dir))

    def _trigger_checkpoint_save(self, trainer: Trainer, pl_module: LightningModule):
        """Main entry point to initiate a save operation."""
        if not self._is_megatron_model(pl_module):
            rank_zero_warn(
                "Model must implement `sharded_state_dict()` for this callback to work."
            )
            return

        step = trainer.global_step
        target_dir = self._format_ckpt_dir(step)
        source_dir = (
            (self.local_tmp_dir / target_dir.name) if self.local_tmp_dir else target_dir
        )

        # 1. Get the sharded state dict from the model
        sharded_state_dict = pl_module.sharded_state_dict()

        # 2. Write common (non-sharded) state on rank 0
        self._write_common_state(source_dir, trainer)

        # 3. Schedule the distributed sharded save
        if self.async_save_enabled:
            self._schedule_async_save(sharded_state_dict, source_dir, target_dir)
        else:
            # Synchronous path: blocks all ranks until save is complete
            mcore_serialization.save(
                sharded_state_dict=sharded_state_dict,
                checkpoint_dir=str(source_dir),
            )
            if self.local_tmp_dir and rank_zero_only.rank == 0:
                shutil.move(source_dir, target_dir)
            rank_zero_info(f"Finished synchronous distributed save to {target_dir}.")

    # ---------- Lightning Callbacks Hooks ----------

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, *args, **kwargs
    ):
        """Check if it's time to save a checkpoint after a batch/step ends."""
        if trainer.global_step > 0 and trainer.global_step % self.save_interval == 0:
            self._trigger_checkpoint_save(trainer, pl_module)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule):
        """Ensure all pending I/O operations are complete before the job finishes."""
        rank_zero_info(
            "Waiting for all async checkpoint saves to finalize before exiting..."
        )
        # Join all background threads
        for thread, req, _ in self._outstanding:
            if thread.is_alive():
                thread.join()  # blocks until the finalize function is done
        self._outstanding = []
        rank_zero_info("All async checkpoint saves finalized.")

        # Handle 'last' symlink creation on rank 0
        if self.save_last and rank_zero_only.rank == 0 and trainer.global_step > 0:
            last_ckpt_dir_name = self._format_ckpt_dir(trainer.global_step).name
            last_link = self.dirpath / "last"
            if last_link.is_symlink() or last_link.is_file():
                last_link.unlink()
            last_link.symlink_to(last_ckpt_dir_name, target_is_directory=True)
            rank_zero_info(f"Updated 'last' symlink to {last_ckpt_dir_name}")
