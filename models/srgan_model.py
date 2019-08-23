import argparse
from math import sqrt, ceil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
import pytorch_lightning as pl

from .datasets import DatasetFromFolder
from .networks import SRResNet, Discriminator
from .losses import VGGLoss, GANLoss


class SRGANModel(pl.LightningModule):
    """
    LightningModule for SRGAN, https://arxiv.org/pdf/1609.04802.
    """
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         add_help=False)
        parser.add_argument('--ngf', type=int, default=64)
        parser.add_argument('--n_blocks', type=int, default=16)
        parser.add_argument('--ndf', type=int, default=64)
        return parser

    def __init__(self, opt):

        super(SRGANModel, self).__init__()

        # store parameters
        self.scale_factor = opt.scale_factor
        self.batch_size = opt.batch_size
        self.patch_size = opt.patch_size

        # network definition
        self.net_G = SRResNet(opt.scale_factor, opt.ngf, opt.n_blocks)
        self.net_D = Discriminator(opt.ndf)

        # criterion definition
        self.criterion_VGG = VGGLoss()
        self.criterion_GAN = GANLoss('wgangp')

    def forward(self, input):
        return self.net_G(input)

    def training_step(self, batch, batch_nb, optimizer_i):
        img_lr = batch['lr']  # [0, 1]
        img_hr = batch['hr']  # [0, 1]
        img_sr = self.forward(img_lr)  # [-1, 1]

        if optimizer_i == 0:  # train generator
            # content losses
            mse_loss = self.criterion_MSE(img_sr, img_hr * 2 - 1)
            vgg_loss = self.criterion_VGG((img_sr + 1) / 2, img_hr)
            # adversarial loss
            gan_loss = self.criterion_GAN(self.net_D(img_sr), True)

            # combined generator loss
            g_loss = 0.5 * (mse_loss + vgg_loss) + 1e-3 * gan_loss

            if self.global_step % self.trainer.add_log_row_interval == 0:
                nrow = ceil(sqrt(self.batch_size))
                self.experiment.add_image(
                    tag='train/lr_img',
                    img_tensor=make_grid(img_lr, nrow=nrow, padding=0),
                    global_step=self.global_step
                )
                self.experiment.add_image(
                    tag='train/hr_img',
                    img_tensor=make_grid(img_hr, nrow=nrow, padding=0),
                    global_step=self.global_step
                )
                self.experiment.add_image(
                    tag='train/sr_img',
                    img_tensor=make_grid(
                        img_sr / 2 + 0.5, nrow=nrow, padding=0),
                    global_step=self.global_step
                )

            return {'loss': g_loss, 'prog': {'train/g_loss': g_loss,
                                             'train/mse_loss': mse_loss,
                                             'train/vgg_loss': vgg_loss}}

        elif optimizer_i == 1:
            # for real image
            d_out_real = self.net_D(img_hr * 2 - 1)
            d_loss_real = self.criterion_GAN(d_out_real, True)
            # for fake image
            d_out_fake = self.net_D(img_sr.detach())
            d_loss_fake = self.criterion_GAN(d_out_fake, False)

            # combined discriminator loss
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            return {'loss': d_loss, 'prog': {'d_loss': d_loss}}

    def validation_step(self, batch, batch_nb):
        with torch.no_grad():
            img_lr = batch['lr']
            img_hr = batch['hr']
            img_sr = self.forward(img_lr)

            mse_loss = self.criterion_MSE((img_sr + 1) / 2, img_hr)
            vgg_loss = self.criterion_VGG((img_sr + 1) / 2, img_hr)
            gan_loss = self.criterion_GAN(self.net_D(img_sr), True)

            g_loss = 0.5 * (mse_loss + vgg_loss) + 1e-3 * gan_loss

        return {'g_loss': g_loss}

    def validation_end(self, outputs):
        val_loss = torch.tensor([output['g_loss'] for output in outputs])
        val_loss = torch.mean(val_loss)
        return {'val_loss': val_loss.item()}

    def configure_optimizers(self):
        optimizer_G = optim.Adam(self.net_G.parameters(), lr=1e-4)
        optimizer_D = optim.Adam(self.net_D.parameters(), lr=1e-4)
        scheduler_G = StepLR(optimizer_G, step_size=1e+5, gamma=0.1)
        scheduler_D = StepLR(optimizer_D, step_size=1e+5, gamma=0.1)
        return [optimizer_G, optimizer_D], [scheduler_G, scheduler_D]

    @pl.data_loader
    def tng_dataloader(self):
        dataset = DatasetFromFolder(
            data_dir='./data/DIV2K/train',
            scale_factor=self.scale_factor,
            patch_size=self.patch_size
        )
        return DataLoader(dataset, self.batch_size, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        dataset = DatasetFromFolder(
            data_dir='./data/DIV2K/val',
            scale_factor=self.scale_factor,
            mode='eval'
        )
        return DataLoader(dataset, batch_size=1)

    @pl.data_loader
    def test_dataloader(self):
        def get_loader(name):
            dataset = DatasetFromFolder(
                data_dir=f'./data/{name}/HR',
                scale_factor=self.scale_factor,
                mode='eval'
            )
            return DataLoader(dataset, batch_size=1)
        dataset_names = ['Set5', 'Set14', 'BSD100', 'Urban100']
        return [get_loader(name) for name in dataset_names]
