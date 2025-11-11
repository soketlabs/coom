# Load your HF dataset (e.g., "wikitext")
import json
import os

import numpy as np
from datasets import load_dataset
from loguru import logger
from megatron.core.datasets.gpt_dataset import (
    GPTDataset,
    GPTDatasetConfig,
)
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.utils import Split
from megatron.core.tokenizers import MegatronTokenizer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class EkaPretrainDataLoader:
    def __init__(
        self,
        data_prefix,
        tokenizer_path,
        cache_path,
        random_seed,
        seq_len,
        batch_size,
        shuffle,
        model_type,
    ):
        self.data_prefix = data_prefix
        self.tokenizer_path = tokenizer_path
        self.cache_path = cache_path
        self.random_seed = random_seed
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.model_type = model_type

    def get_train_data_iterator(self, tokenizer):
        path_prefix = self.data_prefix
        indexed_dataset = IndexedDataset(
            path_prefix=path_prefix, multimodal=False, mmap=True
        )
        num_docs = indexed_dataset.sequence_lengths.shape[0]
        all_indices = np.arange(num_docs, dtype=np.int64)
        split_at = int(0.95 * num_docs)
        train_indices = all_indices[:split_at]
        cfg = GPTDatasetConfig(
            random_seed=self.random_seed,
            sequence_length=self.seq_len,
            tokenizer=tokenizer,
            add_extra_token_to_sequence=True,
            create_attention_mask=True,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            split="train",
            path_to_cache=self.cache_path,
        )
        train_ds = GPTDataset(
            indexed_dataset=indexed_dataset,
            dataset_path=path_prefix,
            indexed_indices=train_indices,
            num_samples=None,
            index_split=Split.train,
            config=cfg,
        )

        train_dataloader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=self.shuffle
        )
        logger.info(f"dataset : {len(train_dataloader)}")

        return train_dataloader

    def prepare_and_load_megatron_tokenizer(self):
        # Ensure HF tokenizer artifacts were already saved to self.tokenizer_dir
        MegatronTokenizer.write_metadata(
            tokenizer_path=self.tokenizer_path,
            tokenizer_library="huggingface",
            model_type=self.model_type,
            tokenizer_class=None,
            chat_template=None,
            overwrite=True,
            metadata_path=None,
        )
        return MegatronTokenizer.from_pretrained(
            tokenizer_path=self.tokenizer_path,
            metadata_path=None,
        )

    def data_build_save_tok(self, hf_data_name, output_json, tokenizer_name):
        # Load the dataset (adjust as necessary)
        dataset = load_dataset(
            hf_data_name, f"{hf_data_name}-103-raw-v1", split="train"
        )

        output_jsonl_path = output_json

        with open(output_jsonl_path, "w", encoding="utf-8") as f:
            for document in dataset:
                if document["text"].strip() == "":
                    continue

                json_object = {"text": document["text"]}
                f.write(json.dumps(json_object) + "\n")
        print("Tokenizer", tokenizer_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer_path = f"{self.tokenizer_path}/{tokenizer_name}"
        print("Tokenizer", tokenizer_path)
        os.makedirs(tokenizer_path, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_path)
