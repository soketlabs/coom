import multiprocessing
from coom.train import Trainer
from coom.profiler import pytorch_profiler

def main():
    # Ensure that the configs directory in coom contains a subfolder
    # with the same name, and that it includes the appropriate configuration files.
    experiment_name = "experiment_0"

    # This is in case someone wants to do multiple runs with the same configs.
    sub_experiment_name = "trial"

    # Command to visualise profiles:
    # tensorboard --logdir=profiler_logs
    profiler = pytorch_profiler(
        experiment_name=experiment_name,
        sub_experiment_name=sub_experiment_name,
        export_to_chrome=True,
        export_to_tensorboard=True,
        nvtx_emit=True
    )

    # profiler_summary=True will display profiler summary on terminal.
    trainer = Trainer(
        experiment_name=experiment_name,
        sub_experiment_name=sub_experiment_name,
        profiler=profiler,
        profiler_summary=True,
    )

    trainer.train()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  
    main()