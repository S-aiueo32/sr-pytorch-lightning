from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

from .datasets import DatasetFromFolder
from .networks import SRCNN


class SRCNNModel(pl.LightningModule):
    def __init__(self, opt):
        super(SRCNNModel, self).__init__()
        self.dataroot = Path(opt.dataroot)

        self.net = SRCNN()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        return self.net(input)

    def training_step(self, batch,  batch_nb):
        img_lr = batch['lr']
        img_hr = batch['hr']
        img_sr = self.forward(img_lr)
        return {'loss': self.criterion(img_sr, img_hr)}

    def validation_step(self, batch, batch_nb):
        img_lr = batch['lr']
        img_hr = batch['hr']
        img_sr = self.forward(img_lr)
        return {'val_loss': self.criterion(img_sr, img_hr)}

    def configure_optimizers(self):
        return [optim.Adam(self.parameters(), lr=1e-4)]

    @pl.data_loader
    def tng_dataloader(self):
        dataset = DatasetFromFolder(
            data_dir=self.dataroot / 'train',
            scale_factor=4,
            patch_size=96,
            preupsample=True
        )
        return DataLoader(dataset, batch_size=16)

    @pl.data_loader
    def val_dataloader(self):
        dataset = DatasetFromFolder(
            data_dir=self.dataroot / 'val',
            scale_factor=4,
            mode='eval',
            preupsample=True
        )
        return DataLoader(dataset, batch_size=1)

    @pl.data_loader
    def test_dataloader(self):
        dataset = DatasetFromFolder(
            data_dir=self.dataroot / 'test',
            scale_factor=4,
            mode='eval',
            preupsample=True
        )
        return DataLoader(dataset, batch_size=1)