import torch
from loguru import logger
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.core.tokenizers import MegatronTokenizer
from torch.utils.data import DataLoader

from coom.training.tokenizer.tokenizer import _GPT2BPETokenizer
from coom.training.utils import print_rank_0

_SEQUENCE_LENGTH = 1024


def prepare_and_load_megatron_tokenizer():
    # Ensure HF tokenizer artifacts were already saved to self.tokenizer_dir
    MegatronTokenizer.write_metadata(
        tokenizer_path="/projects/data/teams/coom_team/training/model_info/tokenizer",
        tokenizer_library="huggingface",
        model_type="gpt2",
        tokenizer_class=None,
        chat_template=None,
        overwrite=True,
        metadata_path=None,
    )
    return MegatronTokenizer.from_pretrained(
        tokenizer_path="/projects/data/teams/coom_team/training/model_info/tokenizer",
        metadata_path=None,
    )


def get_train_data_iterator():
    tokenizer = prepare_and_load_megatron_tokenizer
    config = GPTDatasetConfig(
        random_seed=0,
        sequence_length=_SEQUENCE_LENGTH,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=tokenizer,
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [1000, None, None], lambda: True, config
    ).build()

    train_dataloader = DataLoader(datasets[0], batch_size=1, shuffle=True)
    logger.info(f"dataset : {iter(train_dataloader)}")

    # train_iterator = iter(train_dataloader)

    return train_dataloader
