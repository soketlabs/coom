import time
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.model_summary import summarize
from loguru import logger
from typing_extensions import override


class PrecisionSummary(Callback):
    def __init__(self, max_depth: int = 1, **summarize_kwargs: Any) -> None:
        self._max_depth: int = max_depth
        self._summarize_kwargs: dict[str, Any] = summarize_kwargs

    @override
    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        summary = summarize(
            pl_module, max_depth=self._max_depth, **self._summarize_kwargs
        )

        logger.info(f"Model Precision: {summary._precision_megabytes} MB")
        logger.info(f"Precision: {summary._model.trainer.precision}")


class TimingCallback(Callback):
    """Measures and records per-step timing for integration with FLOPs callbacks."""

    def __init__(self):
        super().__init__()
        self.total_time = 0.0
        self.total_steps = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        elapsed = time.time() - self.start_time
        self.total_time += elapsed
        self.total_steps += 1

        avg_time = self.total_time / self.total_steps

        # log both latest and averaged timing to progress bar
        trainer.progress_bar_metrics["train_step_timing in s"] = elapsed
        trainer.progress_bar_metrics["avg_train_step_timing in s"] = avg_time
