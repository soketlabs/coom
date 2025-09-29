from nemo.lightning.pytorch.callbacks.pytorch_profiler import PytorchProfilerCallback

class EKAProfiler(PytorchProfilerCallback):
    """
    Currently identical to NeMo's PytorchProfilerCallback.
    Defined separately for modularity, to allow future changes or extensions.
    """
