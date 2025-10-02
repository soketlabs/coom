import torch
import pytorch_lightning as pl

from coom.model.transformer import EKAModel

class EKALightningModel(pl.LightningModule):
    def __init__(self, model: EKAModel, **kwargs):
        super().__init__()
        self.model = model
        
    def forward(
        self, 
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor = None
    ):
        return self.model(
            input_ids=input_ids, 
            position_ids=None, 
            attention_mask=attention_mask,
            labels=labels
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['labels']
        loss = self.forward(input_ids=input_ids, labels=labels)
        loss = loss[:, -1].mean()

        self.log('train_loss', loss.item(), on_epoch=True, on_step=True)
        print(loss)
        # exit()

        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return y_hat    
        
    
