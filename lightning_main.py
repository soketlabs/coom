from distributed_utils import init_distributed, destroy_distributed
from model_provider import model_provider
from coom.lightning_model import EKALightningModel
from transformers import AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

def load_dummy_data():
    data_dict = {
        "text": [
            "What is a cat?",
            "Explain photosynthesis.",
            "Write a poem about a tree.",
        ]
    }

    return Dataset.from_dict(data_dict)

def collator_fn(tokenizer, batch):
    enc_batch = []
    for b in batch:
        enc = tokenizer(b["text"], return_tensors="pt", padding=True, truncation=True)
        enc['labels'] = enc['input_ids'].clone()
        enc_batch.append(enc)
    return enc_batch[0]

def main():
    init_distributed()
    model_parallel_cuda_manual_seed(123)

    tokenizer = AutoTokenizer.from_pretrained("soketlabs/pragna-1b")

    model = model_provider()
    data = load_dummy_data()
    data_loader = DataLoader(data, batch_size=1, collate_fn=lambda x: collator_fn(tokenizer, x))

    lightning_model = EKALightningModel(model)

    print(lightning_model)

    trainer = pl.Trainer(max_epochs=500)
    trainer.fit(lightning_model, train_dataloaders=data_loader)

    destroy_distributed()

if __name__ == "__main__":
    main()