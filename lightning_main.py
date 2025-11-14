# import pytorch_lightning as pl
import os

import lightning as L
import numpy as np
import torch
from datasets import Dataset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from loguru import logger
from megatron.core.datasets.gpt_dataset import (
    GPTDataset,
    GPTDatasetConfig,
)
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.utils import Split
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.tokenizers import MegatronTokenizer

# from megatron.training.tokenizer.tokenizer import _NullTokenizer
from torch.utils.data import DataLoader

from coom.callbacks.checkpoint_callback import AsyncModelCheckpoint
from coom.callbacks.flops_callbacks import (
    FLOPsMeasurementCallback,
    MM_FLOPsMeasurementCallback,
)
from coom.callbacks.model_callbacks import PrecisionSummary, TimingCallback
from coom.data_module.dataloader import get_train_data_iterator
from coom.data_module.eka_data_module import EkaLightningDataModule
from coom.data_module.eka_dataload import EkaPretrainDataLoader

# ... (Keep remaining imports)
from coom.lightning_model import EKALightningModel
from distributed_utils import destroy_distributed, init_distributed
from model_provider_eka import model_provider

# def load_dummy_data():
#     data_dict = {
#         "text": [
#             "What is a cat?",
#             "Explain photosynthesis.",
#             "Write a poem about a tree.",
#         ]
#     }

#     return Dataset.from_dict(data_dict)

# def collator_fn(tokenizer, batch):
#     enc_batch = []
#     for b in batch:
#         enc = tokenizer(b["text"], return_tensors="pt", padding=True, truncation=True)
#         enc['labels'] = enc['input_ids'].clone()
#         enc_batch.append(enc)
#     return enc_batch[0]


def main():
    init_distributed()
    model_parallel_cuda_manual_seed(123)
    # num_nodes = int(os.getenv("SLURM_NNODES"))
    # logger.info(f"num_nodes: {num_nodes}")

    # tokenizer = AutoTokenizer.from_pretrained("soketlabs/pragna-1b")

    model = model_provider()
    # data = load_dummy_data()
    # data_loader = DataLoader(
    #     data, batch_size=1, collate_fn=lambda x: collator_fn(tokenizer, x)
    # )
    data_prep = False
    data_module = EkaPretrainDataLoader(
        data_prefix="/projects/data/teams/coom_team/training/data_prep/wikitext_text_document",
        tokenizer_path="/projects/data/teams/coom_team/training/model_info/tokenizer",
        cache_path="/projects/data/teams/coom_team/training/.cache/gpt_indices",
        random_seed=1000,
        seq_len=1024,
        batch_size=16,
        shuffle=True,
        model_type="gpt",
    )

    if data_prep:
        data_module.data_build_save_tok(
            "wikitext",
            "/projects/data/teams/coom_team/training/data_prep/wikitext_train.jsonl",
            "mistralai/Mixtral-8x7B-v0.1",
        )
        print("data made into jsonl")
    else:
        # === Wrap into LightningDataModule ===
        lightning_data_module = EkaLightningDataModule(data_module)

        # Prepare data and build dataloader
        lightning_data_module.prepare_data()
        lightning_data_module.setup()

        print(f"Tokenizer vocab_size: {lightning_data_module.vocab_size}")
    # data = get_train_data_iterator()
    lightning_model = EKALightningModel(model)

    # print(lightning_model)
    wandb_logger = WandbLogger(
        project="EKA_training_prototype_6th_Nov",
        save_dir="/projects/data/teams/coom_team/training/.cache/wandb",
        name="EKA_training_prototype_6th_Nov_run1",
        log_model=False,
    )

    ckpt_dir_path = "/projects/data/teams/coom_team/training/checkpoint_path/eka_train_prototype_13_11_25"
    os.makedirs(ckpt_dir_path, exist_ok=True)
    callbacks = [
        PrecisionSummary(),
        TimingCallback(),
        FLOPsMeasurementCallback(data_config=lightning_data_module),
        MM_FLOPsMeasurementCallback(data_config=lightning_data_module),
        # ModelCheckpoint(
        #     every_n_train_steps=2,  # Corresponds to `save_steps`
        #     save_top_k=10,  # Corresponds to `save_total_limit`
        #     mode="max",  # Keep the 10 checkpoints with the highest step numbers
        #     dirpath=ckpt_dir_path,  # Directory to save checkpoints
        #     filename="eka",
        #     # filename="dhvani-{epoch}-{step}",
        #     save_last=True,  # Optionally, save a 'last.ckpt' as well
        #     monitor="step",
        #     verbose=True,
        #     save_weights_only=False,
        #     train_time_interval=10,
        #     save_on_train_epoch_end=False,
        # ),
        AsyncModelCheckpoint(
            dirpath=ckpt_dir_path,
            filename="epoch={epoch}-step={step}",
            save_interval=100,
            save_last=True,
            async_save=True,
        ),
    ]
    # for cb in callbacks:
    #     assert isinstance(cb, pl.Callback), f"Bad callback: {type(cb)}"

    trainer = L.Trainer(
        max_epochs=500,
        logger=wandb_logger,
        callbacks=callbacks,
        precision="bf16-true",
        # num_nodes=num_nodes,
        devices=8,
    )
    trainer.fit(lightning_model, datamodule=lightning_data_module)

    destroy_distributed()


if __name__ == "__main__":
    main()
