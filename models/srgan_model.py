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
    LightningModule for SRGAN.
    """

    def __init__(self, opt):

        super(SRGANModel, self).__init__()

        # store parameters
        self.dataroot = Path(opt.dataroot)
        self.batch_size = opt.batch_size

        # network definition
        self.net_G = SRResNet(base_filters=64)
        self.net_D = Discriminator(base_filters=64)

        # criterion definition
        self.criterion_MSE = nn.MSELoss()
        self.criterion_VGG = VGGLoss(scale=0.006)
        self.criterion_GAN = GANLoss('vanilla')

    def forward(self, input):
        return self.net_G(input)

    def training_step(self, batch, batch_nb, optimizer_i):
        img_lr = batch['lr'] # [0, 1]
        img_hr = batch['hr']  # [0, 1]
        img_sr = self.forward(img_lr)  # [-1, 1]

        if optimizer_i == 0:  # train generator
            # content losses
            mse_loss = self.criterion_MSE((img_sr + 1) / 2, img_hr)
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
                    img_tensor=make_grid(img_sr / 2 + 0.5, nrow=nrow, padding=0),
                    global_step=self.global_step
                )

            return {'loss': g_loss, 'prog': {'train/g_loss': g_loss}}

        elif optimizer_i == 1:
            # for real image
            d_out_real = self.net_D(img_hr * 2 - 1)
            d_loss_real = self.criterion_GAN(d_out_real, True)
            # for fake image
            d_out_fake = self.net_D(img_sr.detach())
            d_loss_fake = self.criterion_GAN(d_out_fake, False)

            # combined discriminator loss
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            return {'loss': d_loss, 'prog': {'train/d_loss': d_loss}}

    def validation_step(self, batch, batch_nb):
        with torch.no_grad():
            img_lr = batch['lr']
            img_hr = batch['hr']
            img_sr = self.forward(img_lr)

            mse_loss = self.criterion_MSE((img_sr + 1) / 2, img_hr)
            vgg_loss = self.criterion_VGG((img_sr + 1) / 2, img_hr)
            gan_loss = self.criterion_GAN(self.net_D(img_sr), True)

            g_loss = 0.5 * (mse_loss + vgg_loss) + 1e-3 * gan_loss
            
            d_out_real = self.net_D(img_hr * 2 - 1)
            d_loss_real = self.criterion_GAN(d_out_real, True)
            d_out_fake = self.net_D(img_sr.detach())
            d_loss_fake = self.criterion_GAN(d_out_fake, False)

            d_loss = 0.5 * (d_loss_real + d_loss_fake)

        return {'g_loss': g_loss, 'd_loss': d_loss}

    
    def validation_end(self, outputs):
        g_loss_mean = 0
        d_loss_mean = 0
        for output in outputs:
            g_loss_mean += output['g_loss']
            d_loss_mean += output['d_loss']

        g_loss_mean /= len(outputs)
        d_loss_mean /= len(outputs)
        tqdm_dic = {'val/g_loss': g_loss_mean.item(), 'val/d_loss': d_loss_mean.item()}
        return tqdm_dic

    def configure_optimizers(self):
        optimizer_G = optim.Adam(self.net_G.parameters(), lr=1e-4)
        optimizer_D = optim.Adam(self.net_D.parameters(), lr=1e-4)
        scheduler_G = StepLR(optimizer_G, step_size=1e+5, gamma=0.1)
        scheduler_D = StepLR(optimizer_D, step_size=1e+5, gamma=0.1)
        return [optimizer_G, optimizer_D], [scheduler_G, scheduler_D]

    @pl.data_loader
    def tng_dataloader(self):
        dataset = DatasetFromFolder(
            data_dir=self.dataroot / 'train',
            scale_factor=4,
            patch_size=96
        )
        return DataLoader(dataset, batch_size=16, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        dataset = DatasetFromFolder(
            data_dir=self.dataroot / 'val',
            scale_factor=4,
            mode='eval'
        )
        return DataLoader(dataset, batch_size=1)
