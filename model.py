import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

from dataset import DatasetFromFolder

class SRCNNModel(pl.LightningModule):
    def __init__(self):
        super(SRCNNModel, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2)
        )

    def forward(self, input):
        return self.net(input)

    def training_step(self, batch,  batch_nb):
        img_lr = batch['lr']
        img_hr = batch['hr']

        img_sr = self.forward(img_lr)

        return {'loss': F.mse_loss(img_sr, img_hr)}

    def validation_step(self, batch, batch_nb, dataloader_i):
        img_lr = batch['lr']
        img_hr = batch['hr']

        img_sr = self.forward(img_lr)

        return {'val_loss': F.mse_loss(img_sr, img_hr)}

    def configure_optimizers(self):
        return [optim.Adam(self.parameters(), lr=1e-3)]

    @pl.data_loader
    def tng_dataloader(self):
        dataset = DatasetFromFolder('./data/General-100/train', scale_factor=4, patch_size=96)
        return DataLoader(dataset, batch_size=16)

    @pl.data_loader
    def val_dataloader(self):
        dataset = DatasetFromFolder('./data/General-100/val', scale_factor=4, mode='eval')
        return DataLoader(dataset, batch_size=1)

    @pl.data_loader
    def test_dataloader(self):
        dataset = DatasetFromFolder('./data/General-100/test', scale_factor=4, mode='eval')
        return DataLoader(dataset, batch_size=1)