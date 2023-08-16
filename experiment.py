import os
import math
import torch
from torch import optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DistillExperiment(pl.LightningModule):

    def __init__(self,
                 distill_model,
                 params: dict) -> None:
        super(DistillExperiment, self).__init__()

        self.model = distill_model
        self.params = params

    def forward(self, img: torch.tensor, context: str, **kwargs) -> torch.tensor:
        return self.model(img, context, **kwargs)

    def training_step(self, batch, batch_idx):
        # compute student output, weights for the loss, and target
        img, context = batch['image_tensor'], batch['text']
        recon, lambda_t, target = self(img, context)

        train_loss = self.model.loss_function(recon, lambda_t, target)

        self.log_dict({key: val.item() for key, val in train_loss.items()},
                      sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx):
        img, context = batch['image_tensor'], batch['text']
        recon, lambda_t, target = self(img, context)

        val_loss = self.model.loss_function(recon, lambda_t, target)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()},
                      sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=3e-4)
