# lightning_data_module.py

import lightning.pytorch as pl
from loguru import logger

from coom.data_module.eka_dataload import EkaPretrainDataLoader


class EkaLightningDataModule(pl.LightningDataModule):
    """
    Simple Lightning DataModule wrapper around EkaPretrainDataLoader.
    Makes it compatible with PyTorch Lightning Trainer.
    """

    def __init__(self, data_loader: EkaPretrainDataLoader):
        super().__init__()
        self.data_loader = data_loader
        self.tokenizer = None
        self.train_loader = None

        # Expose attributes for callbacks (FLOPs, etc.)
        self.global_batch_size = getattr(data_loader, "batch_size", 1)
        self.tokenizer_path = getattr(data_loader, "tokenizer_path", None)
        self.vocab_size = None

    def prepare_data(self):
        """Loads tokenizer if not already loaded."""
        self.tokenizer = self.data_loader.prepare_and_load_megatron_tokenizer()
        if self.tokenizer:
            self.vocab_size = getattr(self.tokenizer, "vocab_size", None)
            logger.info(f"Tokenizer loaded (vocab_size={self.vocab_size})")

    def setup(self, stage=None):
        """Builds DataLoader (train only, for now)."""
        if self.tokenizer is None:
            self.prepare_data()
        self.train_loader = self.data_loader.get_train_data_iterator(self.tokenizer)
        logger.info(f"Train DataLoader ready. Batches: {len(self.train_loader)}")

    def train_dataloader(self):
        """Return train dataloader for Lightning Trainer."""
        return self.train_loader
