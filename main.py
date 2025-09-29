import multiprocessing
from coom.train import Trainer


def main():
    """Run the training experiment with the specified configurations."""
    # Ensure that the configs directory in coom contains a subfolder
    # with the same name, and that it includes the appropriate configuration files.
    experiment_name = "experiment_0"

    # This is in case someone wants to do multiple runs with the same configs.
    sub_experiment_name = "trial"


    trainer = Trainer(
        experiment_name=experiment_name,
        sub_experiment_name=sub_experiment_name,
    )

    trainer.train()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
