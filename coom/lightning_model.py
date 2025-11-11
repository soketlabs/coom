import lightning as L
import pytorch_lightning as pl
import torch
from loguru import logger

from coom.model.transformer import EKAModel


class EKALightningModel(L.LightningModule):
    def __init__(self, model: EKAModel, **kwargs):
        super().__init__()
        self.model = model
        # for n, p in model.named_parameters():
        #     if p.device.type == "meta":
        #         print(f"Uninitialized (meta) parameter: {n}")
        #     if p.device.type != "meta":
        #         print(f"{n:60} | device={p.device} | shape={tuple(p.shape)}")
        #         print(f"{n}: mean={p.data.mean():.4f}, std={p.data.std():.4f}")

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ):
        return self.model(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=attention_mask,
            labels=labels,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5, eps=1e-8)

    def training_step(self, batch, batch_idx):
        input_ids = batch["tokens"]
        labels = batch["labels"]
        # logger.info(input_ids)
        loss = self.forward(input_ids=input_ids, labels=labels)
        loss = loss[:, -1].mean()

        self.log("train_loss", loss.item(), on_step=True)
        # logger.info(loss)
        # exit()

        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return y_hat
