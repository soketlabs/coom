import os
from pathlib import Path

from hydra import compose, initialize_config_dir
# from lightning.pytorch.loggers import WandbLogger
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from omegaconf import DictConfig, OmegaConf

from . import config_classes, model

def load_cfg(config_path: str, config_name: str) -> DictConfig:
    """
    Load a Hydra configuration as a native Python dictionary.

    Args:
        config_path (str): Path to configuration directory.
        config_name (str): Name of the config file (without extension).

    Returns:
        DictConfig: Loaded configuration as a dictionary.
    """
    abs_path = str(Path(config_path).resolve())
    with initialize_config_dir(config_dir=abs_path, version_base=None):
        cfg = compose(config_name=config_name)
        native_cfg = OmegaConf.to_container(cfg, resolve=True)
    return native_cfg


class Trainer:
    """
    A class for training models with configuration management.

    This class handles loading configurations, initializing components,
    and orchestrating the training process.
    """

    def __init__(self, experiment_name, sub_experiment_name, use_wandb=False, profiler=None, profiler_summary=False):
        """
        Initialize the Trainer with experiment and sub-experiment names.

        Args:
            experiment_name (str): Name of the experiment.
            sub_experiment_name (str): Name of the sub-experiment.
            use_wandb (bool): Whether to use Weights & Biases logging.
        """
        self.experiment_name = experiment_name
        self.sub_experiment_name = sub_experiment_name
        self.config_base_path = "./../coom/configs"
        self.use_wandb = use_wandb  # Will be used in the future.
        self.profiler = profiler
        self.profiler_summary = profiler_summary

        # Configuration containers
        self.main_cfg = None
        self.model_cfg = None
        self.data_cfg = None
        self.opt_cfg = None
        self.trainer_cfg = None
        self.logger_cfg = None

        # Component containers
        self.model = None
        self.data_module = None
        self.trainer = None
        self.logger = None
        self.optimizer = None

        self.load_configurations()

    def load_configurations(self):
        """
        Load all configuration files for the experiment.
        """
        self.main_cfg = load_cfg(
            self.config_base_path,
            f"{self.experiment_name}/main_config"
        )[self.experiment_name]

        prefix = self.main_cfg["config_path_prefix"]

        def full_path(key):
            return f"{self.experiment_name}{os.path.join(prefix, self.main_cfg[key])}"

        self.model_cfg = load_cfg(self.config_base_path, full_path("base_model_configuration_path"))[self.experiment_name]
        self.data_cfg = load_cfg(self.config_base_path, full_path("dataLoader_config_path"))[self.experiment_name]
        self.opt_cfg = load_cfg(self.config_base_path, full_path("optimizer_config_path"))[self.experiment_name]
        self.trainer_cfg = load_cfg(self.config_base_path, full_path("trainer_config_path"))[self.experiment_name]
        self.logger_cfg = load_cfg(self.config_base_path, full_path("logger_config_path"))[self.experiment_name]

        print("All configurations loaded successfully!")

    def initialize_model(self):
        """
        Initialize the model based on the loaded configuration.
        """
        if self.main_cfg is None:
            raise ValueError("Main configuration not loaded. Call load_configurations() first.")

        ModelClass = getattr(model, self.main_cfg["base_model"])
        ConfigClass = getattr(config_classes, self.main_cfg["base_model_config"])

        model_config = ConfigClass(**self.model_cfg)
        self.model = ModelClass(model_config)

        print(f"Model {self.main_cfg['base_model']} initialized successfully!")

    def initialize_data_module(self, data_module_type):
        """
        Initialize the data module based on configuration.

        Args:
            data_module_type (str): "Mock" (currently only supported option)
        """
        if self.data_cfg is None:
            raise ValueError("Data configuration not loaded. Call load_configurations() first.")

        if data_module_type == "Mock":
            self.data_module = llm.MockDataModule(
                seq_length=self.data_cfg["seq_length"],
                global_batch_size=self.data_cfg["global_batch_size"],
            )
        else:
            raise NotImplementedError("Only 'Mock' data module is implemented.")

        print("Data module initialized successfully!")

    def initialize_optimizer(self):
        """
        Initialize the optimizer based on configuration.
        """
        if self.opt_cfg is None:
            raise ValueError("Optimizer configuration not loaded. Call load_configurations() first.")

        optimizer_config = OptimizerConfig(**self.opt_cfg)
        self.optimizer = nl.MegatronOptimizerModule(config=optimizer_config)

        print("Optimizer initialized successfully!")

    def initialize_trainer(self):
        """
        Initialize the training strategy and trainer.
        """
        if self.trainer_cfg is None:
            raise ValueError("Trainer configuration not loaded. Call load_configurations() first.")

        strategy = nl.MegatronStrategy(**self.trainer_cfg["strategy"])

        profiler = self.profiler

        self.trainer = nl.Trainer(
            devices=self.trainer_cfg["devices"],
            max_steps=self.trainer_cfg["max_steps"],
            accelerator=self.trainer_cfg["accelerator"],
            strategy=strategy,
            plugins=nl.MegatronMixedPrecision(precision=self.trainer_cfg["precision"]),
            profiler=profiler
        )

        print("Trainer initialized successfully!")

    def initialize_logger(self):
        """
        Initialize the logger based on configuration.
        """
        if self.logger_cfg is None:
            raise ValueError("Logger configuration not loaded. Call load_configurations() first.")

        self.logger = nl.NeMoLogger(
            log_dir=self.logger_cfg["log_dir"],
            update_logger_directory=True,
            name=self.sub_experiment_name,
            use_datetime_version=False,
        )

        print("Logger initialized successfully!")

    def initialize_all_components(self):
        """
        Initialize all components required for training.
        """
        
        self.initialize_model()
        self.initialize_data_module(data_module_type="Mock")
        self.initialize_optimizer()
        self.initialize_trainer()
        self.initialize_logger()

        print("All components initialized successfully!")

    def validate_components(self):
        """
        Ensure all required components are initialized.

        Raises:
            ValueError: If any required component is missing.
        """
        required_components = [
            ("model", self.model),
            ("data_module", self.data_module),
            ("trainer", self.trainer),
            ("logger", self.logger),
            ("optimizer", self.optimizer),
        ]

        missing = [name for name, comp in required_components if comp is None]

        if missing:
            raise ValueError(
                f"Missing components: {', '.join(missing)}. "
                "Call initialize_all_components() first."
            )

        return True

    def start_training(self):
        """
        Start the training process.
        """
        self.validate_components()

        print("Starting training...")

        llm.train(
            model=self.model,
            data=self.data_module,
            trainer=self.trainer,
            log=self.logger,
            tokenizer="data",
            optim=self.optimizer,
        )

        print("Training completed successfully!")

    def train(self):
        """
        Full training pipeline: initialize and train.
        """
        self.initialize_all_components()
        self.start_training()
        if self.trainer and self.trainer.profiler and self.profiler_summary:
            print("\n=== Profiler Summary ===")
            print(self.trainer.profiler.summary())

    def get_config_summary(self):
        """
        Get a summary of the loaded configuration.

        Returns:
            dict: Configuration summary
        """
        if self.main_cfg is None:
            return {"status": "Configurations not loaded"}

        return {
            "experiment_name": self.experiment_name,
            "base_model": self.main_cfg.get("base_model"),
            "base_model_config": self.main_cfg.get("base_model_config"),
            "devices": self.trainer_cfg.get("devices") if self.trainer_cfg else None,
            "max_steps": self.trainer_cfg.get("max_steps") if self.trainer_cfg else None,
            "seq_length": self.data_cfg.get("seq_length") if self.data_cfg else None,
            "global_batch_size": self.data_cfg.get("global_batch_size") if self.data_cfg else None,
            "log_dir": self.logger_cfg.get("log_dir") if self.logger_cfg else None,
        }
