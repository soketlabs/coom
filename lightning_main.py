import numpy as np
import pytorch_lightning as pl
import torch
from datasets import Dataset
from lightning.pytorch.loggers import WandbLogger
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
        batch_size=2,
        shuffle=True,
        model_type="gpt",
    )
    if data_prep:
        data_module.data_build_save_tok(
            "wikitext",
            "/projects/data/teams/coom_team/training/data_prep/wikitext_train.jsonl",
            "gpt2",
        )
        print("data made into jsonl")
    else:
        tokenizer = data_module.prepare_and_load_megatron_tokenizer()
        data = data_module.get_train_data_iterator(tokenizer)

        lightning_model = EKALightningModel(model)

        print(lightning_model)
        wandb_logger = WandbLogger(
            project="EKA_training_prototype_6th_Nov",
            save_dir="/projects/data/teams/coom_team/training/.cache/wandb",
            name="EKA_training_prototype_6th_Nov_run1",
            log_model=False,
        )

        trainer = pl.Trainer(
            max_epochs=500,
            logger=wandb_logger,
        )
        trainer.fit(lightning_model, train_dataloaders=data)

        destroy_distributed()


if __name__ == "__main__":
    main()
