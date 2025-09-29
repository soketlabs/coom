import os
from pathlib import Path
import importlib

from hydra import compose, initialize_config_dir
# from lightning.pytorch.loggers import WandbLogger
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from omegaconf import DictConfig, OmegaConf

from coom import config_classes, model, data_module, profiler
from coom.data_module.load_and_validate_paths import load_data_paths


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

    def __init__(self, experiment_name, sub_experiment_name, use_wandb=False):
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

        # Configuration containers
        self.main_cfg = None
        self.model_cfg = None
        self.data_cfg = None
        self.opt_cfg = None
        self.trainer_cfg = None
        self.logger_cfg = None
        self.profiler_cfg = None
        self.callback_cfg = None

        # Component containers
        self.model = None
        self.data_module = None
        self.trainer = None
        self.logger = None
        self.optimizer = None
        self.profiler = None
        self.callbacks = []

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

        def full_path(config, key):
            return f"{self.experiment_name}{os.path.join(prefix, config[key])}"

        self.model_cfg = load_cfg(self.config_base_path, full_path(self.main_cfg, "base_model_configuration_path"))[self.experiment_name]
        self.data_cfg = load_cfg(self.config_base_path, full_path(self.main_cfg, "dataLoader_config_path"))[self.experiment_name]
        self.opt_cfg = load_cfg(self.config_base_path, full_path(self.main_cfg, "optimizer_config_path"))[self.experiment_name]
        self.trainer_cfg = load_cfg(self.config_base_path, full_path(self.main_cfg, "trainer_config_path"))[self.experiment_name]
        self.logger_cfg = load_cfg(self.config_base_path, full_path(self.main_cfg, "logger_config_path"))[self.experiment_name]
        self.profiler_cfg = load_cfg(self.config_base_path, full_path(self.main_cfg, "profiler_config_path"))[self.experiment_name]
        self.data_paths_cfg = load_cfg(self.config_base_path, full_path(self.data_cfg, "data_paths_config_file"))[self.experiment_name]

        # Load callback configuration if specified
        if "callback_config_path" in self.main_cfg:
            self.callback_cfg = load_cfg(self.config_base_path, full_path(self.main_cfg, "callback_config_path"))[self.experiment_name]

        print("All configurations loaded successfully!")

    def initialize_callbacks(self):
        """
        Initialize callbacks based on the callback configuration.

        Expected callback_cfg structure:
        {
            "callbacks": [
                {
                    "class_name": "ModelCheckpoint",
                    "module_path": "nemo.lightning.pytorch.callbacks",
                    "parameters": {
                        "dirpath": "checkpoints/",
                        "filename": "{epoch}-{step}",
                        "save_top_k": 3,
                        "monitor": "val_loss"
                    }
                },
                {
                    "class_name": "EarlyStopping",
                    "module_path": "nemo.lightning.pytorch.callbacks",
                    "parameters": {
                        "monitor": "val_loss",
                        "patience": 10,
                        "mode": "min"
                    }
                }
            ]
        }
        """
        self.callbacks = []

        if self.callback_cfg is None or "callbacks" not in self.callback_cfg:
            print("No callback configuration found or callbacks section missing.")
            return

        print("Initializing callbacks...")

        for callback_config in self.callback_cfg["callbacks"]:
            try:
                # Extract callback information
                class_name = callback_config["class_name"]
                module_path = callback_config["module_path"]
                parameters = callback_config.get("parameters", {})

                # Dynamically import the module
                module = importlib.import_module(module_path)

                # Get the callback class
                callback_class = getattr(module, class_name)

                # Create callback instance with parameters
                callback_instance = callback_class(**parameters)

                # Add to callbacks list
                self.callbacks.append(callback_instance)

                print(f"Successfully initialized callback: {class_name}")

            except Exception as e:
                print(f"Error initializing callback {callback_config.get('class_name', 'Unknown')}: {str(e)}")
                continue

        print(f"Initialized {len(self.callbacks)} callbacks successfully!")

    def initialize_profiler(self):
        """
        Initialize the profiler based on configuration.
        """
        if not self.profiler_cfg["enable_profiler"]:
            self.profiler = None
            print("Profiler disabled.")
            return

        print("Initializing profiler with configuration...")

        if self.sub_experiment_name is not None:
            profiler_dir = os.path.join("profiler_logs", self.experiment_name, self.sub_experiment_name)
        else:
            profiler_dir = os.path.join("profiler_logs", self.experiment_name)

        os.makedirs(profiler_dir, exist_ok=True)

        self.profiler = profiler.EKAProfiler(
            start_step=self.profiler_cfg["start_step"],
            end_step=self.profiler_cfg["end_step"],
            warmup_steps=self.profiler_cfg["warmup_steps"],
            active_steps=self.profiler_cfg["active_steps"],
            trace_dir=profiler_dir
        )

    def initialize_model(self):
        """
        Initialize the model based on the loaded configuration.
        """
        if self.main_cfg is None:
            raise ValueError("Main configuration not loaded. Call load_configurations() first.")

        model_class = getattr(model, self.main_cfg["base_model"])
        config_class = getattr(config_classes, self.main_cfg["base_model_config"])

        model_config = config_class(
            **{
                **self.model_cfg,
                "seq_length": self.data_cfg["seq_length"],
                "vocab_size": self.data_cfg["vocab_size"],
            })
        self.model = model_class(model_config)

        print(f"Model {self.main_cfg['base_model']} initialized successfully!")

    def initialize_data_module(self, data_module_type):
        """
        Initialize the data module based on configuration.

        Args:
            data_module_type (str): "Mock" or "Real"
        """
        if self.data_cfg is None:
            raise ValueError("Data configuration not loaded. Call load_configurations() first.")

        if data_module_type == "Mock":
            self.data_module = llm.MockDataModule(
                seq_length=self.data_cfg["seq_length"],
                global_batch_size=self.data_cfg["global_batch_size"],
            )
        elif data_module_type == "Real":
            if self.data_cfg["streaming"]:
                os.environ["MSC_CONFIG"] = os.path.join(self.config_base_path, f"{self.experiment_name}{os.path.join(self.main_cfg['config_path_prefix'], self.data_cfg['msc_config'])}")

            processed_paths = load_data_paths(self.data_paths_cfg)

            datamodule_class = getattr(data_module, self.main_cfg["data_model"])
            self.data_module = datamodule_class(
                paths=processed_paths,
                seq_length=self.data_cfg["seq_length"],
                micro_batch_size=self.data_cfg["micro_batch_size"],
                global_batch_size=self.data_cfg["global_batch_size"],
                object_storage_cache_path=self.data_cfg["object_storage_cache_path"],
                mmap_bin_files=not self.data_cfg["streaming"]
            )

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

        # Combine all callbacks (profiler + configured callbacks)
        all_callbacks = []
        if self.profiler is not None:
            all_callbacks.append(self.profiler)
        all_callbacks.extend(self.callbacks)

        self.trainer = nl.Trainer(
            devices=self.trainer_cfg["devices"],
            max_steps=self.trainer_cfg["max_steps"],
            accelerator=self.trainer_cfg["accelerator"],
            strategy=strategy,
            callbacks=all_callbacks,
            plugins=nl.MegatronMixedPrecision(precision=self.trainer_cfg["precision"]),
            limit_val_batches=0,
        )

        print("Trainer initialized successfully!")

    def initialize_logger(self):
        """
        Initialize the logger based on configuration.
        """
        if self.logger_cfg is None:
            raise ValueError("Logger configuration not loaded. Call load_configurations() first.")

        self.logger = nl.NeMoLogger(
            log_dir=os.path.join(self.logger_cfg["log_dir"], self.experiment_name, self.sub_experiment_name),
            update_logger_directory=True,
            use_datetime_version=False,
        )

        print("Logger initialized successfully!")

    def initialize_all_components(self):
        """
        Initialize all components required for training.
        """

        self.initialize_profiler()
        self.initialize_callbacks()
        self.initialize_model()
        self.initialize_data_module(data_module_type=self.data_cfg["dataModuleType"])
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
            # tokenizer="data",
            optim=self.optimizer,
        )

        print("Training completed successfully!")

    def train(self):
        """
        Full training pipeline: initialize and train.
        """
        self.initialize_all_components()
        self.start_training()

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
            "num_callbacks": len(self.callbacks),
        }
