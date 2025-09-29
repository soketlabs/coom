from lightning.pytorch.profilers import PyTorchProfiler
from torch.profiler import tensorboard_trace_handler
import os
from typing import Optional

def pytorch_profiler(experiment_name: str, sub_experiment_name: Optional[str] = None, **kwargs):
    if sub_experiment_name is not None:
        profiler_dir = os.path.join("profiler_logs", experiment_name, sub_experiment_name)
    else:
        profiler_dir = os.path.join("profiler_logs", experiment_name)

    os.makedirs(profiler_dir, exist_ok=True)

    kwargs.setdefault("dirpath", profiler_dir)
    kwargs.setdefault("filename", "profile")
    kwargs.setdefault("export_to_tensorboard", True)
    kwargs.setdefault("export_to_chrome", True) 
    kwargs.setdefault("on_trace_ready", tensorboard_trace_handler(profiler_dir))

    return PyTorchProfiler(**kwargs)
