import os

import numpy as np
import torch
from datasets import load_dataset
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder
from megatron.core.datasets.utils import Split
from megatron.core.tokenizers import MegatronTokenizer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class DataModule:
    def __init__(
        self,
        data_dir_prefix,
        tokenizer_name,
        hf_dataset_name,
        tokenizer_dir,
        seq_len=128,
        batch_size=8,
        random_seed=1000,
    ):
        self.data_dir_prefix = data_dir_prefix
        self.tokenizer_name = tokenizer_name
        self.hf_dataset_name = hf_dataset_name
        self.tokenizer_dir = tokenizer_dir
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.random_seed = random_seed
        os.makedirs(self.tokenizer_dir, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.tokenizer.save_pretrained(self.tokenizer_dir)

    def prepare_and_load_megatron_tokenizer(self):
        # Ensure HF tokenizer artifacts were already saved to self.tokenizer_dir
        MegatronTokenizer.write_metadata(
            tokenizer_path=self.tokenizer_dir,
            tokenizer_library="huggingface",
            model_type="gpt",
            tokenizer_class=None,
            chat_template=None,
            overwrite=True,
            metadata_path=None,
        )
        return MegatronTokenizer.from_pretrained(
            tokenizer_path=self.tokenizer_dir,
            metadata_path=None,
        )

    def build_indexed_dataset(self):
        bin_path = f"{self.data_dir_prefix}.bin"
        idx_path = f"{self.data_dir_prefix}.idx"
        # if os.path.exists(bin_path) and os.path.exists(idx_path):
        #     return
        ds = load_dataset(self.hf_dataset_name, split="train")
        eod_id = self.tokenizer.eos_token_id
        builder = IndexedDatasetBuilder(bin_path)
        for ex in ds:
            ids = self.tokenizer.encode(ex["text"], truncation=True, padding=False)
            ids.append(eod_id)
            ids_np = np.asarray(ids, dtype=np.int64)
            doc_len = np.asarray([len(ids_np)], dtype=np.int32)
            builder.add_document(ids_np, doc_len)
        builder.finalize(idx_path)

    def make_loader(self, meg_tok):
        path_prefix = self.data_dir_prefix
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
            tokenizer=meg_tok,
            add_extra_token_to_sequence=True,
            create_attention_mask=True,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            split="train",
            path_to_cache="./gpt_indices",
        )
        train_ds = GPTDataset(
            indexed_dataset=indexed_dataset,
            dataset_path=path_prefix,
            indexed_indices=train_indices,
            num_samples=None,
            index_split=Split.train,
            config=cfg,
        )
        return DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)


def inspect_batch(batch, meg_tok, max_rows=2, max_len=80):
    print("Keys:", list(batch.keys()))
    t = batch["tokens"]
    l = batch["labels"]
    am = batch["attention_mask"]
    lm = batch["loss_mask"]
    pid = batch["position_ids"]

    def stats(name, x):
        x_min = int(x.min().item())
        x_max = int(x.max().item())
        print(
            f"{name}: shape={tuple(x.shape)}, dtype={x.dtype}, min={x_min}, max={x_max}"
        )

    stats("tokens", t)
    stats("labels", l)
    stats("attention_mask", am)
    stats("loss_mask", lm)
    stats("position_ids", pid)
    rows = min(max_rows, t.shape[0])
    for i in range(rows):
        row = t[i][:max_len].tolist()
        print(f"sample {i} token_ids[:{max_len}]: {row}")
        txt = None
        for fn in ("decode", "detokenize"):
            if hasattr(meg_tok, fn):
                try:
                    txt = getattr(meg_tok, fn)(row)
                    break
                except Exception:
                    pass
        print(f"decoded[:{max_len}]: {repr(txt) if txt is not None else '<no decode>'}")


if __name__ == "__main__":
    dm = DataModule(
        data_dir_prefix="./imdb_megatron",
        tokenizer_name="gpt2",
        hf_dataset_name="imdb",
        tokenizer_dir="./tok_gpt2_hf",
        seq_len=128,
        batch_size=4,
        random_seed=1000,
    )
    meg_tok = dm.prepare_and_load_megatron_tokenizer()
    dm.build_indexed_dataset()
    loader = dm.make_loader(meg_tok)
    batch = next(iter(loader))
    inspect_batch(batch, meg_tok, max_rows=2, max_len=80)
