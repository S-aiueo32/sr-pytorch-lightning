import argparse
from collections import OrderedDict
from math import sqrt, ceil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import make_grid
import pytorch_lightning as pl
from kornia.losses import SSIM
from kornia.color import rgb_to_grayscale

from .datasets import DatasetFromFolder
from .networks import SRResNet, Discriminator
from .losses import GANLoss, TVLoss, VGGLoss, PSNR


class SRGANModel(pl.LightningModule):
    """
    LightningModule for SRGAN, https://arxiv.org/pdf/1609.04802.
    """
    @staticmethod
    def add_model_specific_args(parent):
        parser = argparse.ArgumentParser(parents=[parent], add_help=False)
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

        # networks
        self.net_G = SRResNet(opt.scale_factor, opt.ngf, opt.n_blocks)
        self.net_D = Discriminator(opt.ndf)

        # training criterions
        self.criterion_MSE = nn.MSELoss()
        self.criterion_VGG = VGGLoss(net_type='vgg19', layer='relu5_4')
        self.criterion_GAN = GANLoss(gan_mode='wgangp')
        self.criterion_TV = TVLoss()

        # validation metrics
        self.criterion_PSNR = PSNR()
        self.criterion_SSIM = SSIM(window_size=11, reduction='mean')

    def forward(self, input):
        return self.net_G(input)

    def training_step(self, batch, batch_nb, optimizer_i):
        img_lr = batch['lr']  # \in [0, 1]
        img_hr = batch['hr']  # \in [0, 1]

        if optimizer_i == 0:  # train discriminator
            self.img_sr = self.forward(img_lr)  # \in [0, 1]

            # for real image
            d_out_real = self.net_D(img_hr)
            d_loss_real = self.criterion_GAN(d_out_real, True)
            # for fake image
            d_out_fake = self.net_D(self.img_sr.detach())
            d_loss_fake = self.criterion_GAN(d_out_fake, False)

            # combined discriminator loss
            d_loss = 1 + d_loss_real + d_loss_fake

            return {'loss': d_loss, 'prog': {'tng/d_loss': d_loss}}

        elif optimizer_i == 1:  # train generator
            # content loss
            mse_loss = self.criterion_MSE(self.img_sr * 2 - 1,  # \in [-1, 1]
                                          img_hr * 2 - 1)  # \in [-1, 1]
            vgg_loss = self.criterion_VGG(self.img_sr, img_hr)
            content_loss = (vgg_loss + mse_loss) / 2
            # adversarial loss
            adv_loss = self.criterion_GAN(self.net_D(self.img_sr), True)
            # tv loss
            tv_loss = self.criterion_TV(self.img_sr)

            # combined generator loss
            g_loss = content_loss + 1e-3 * adv_loss + 2e-8 * tv_loss

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
                    img_tensor=make_grid(self.img_sr, nrow=nrow, padding=0),
                    global_step=self.global_step
                )

            return {'loss': g_loss, 'prog': {'tng/g_loss': g_loss,
                                             'tng/content_loss': content_loss,
                                             'tng/adv_loss': adv_loss,
                                             'tng/tv_loss': tv_loss}}

    def validation_step(self, batch, batch_nb):
        with torch.no_grad():
            img_lr = batch['lr']
            img_hr = batch['hr']
            img_sr = self.forward(img_lr)

            img_hr_ = rgb_to_grayscale(img_hr)
            img_sr_ = rgb_to_grayscale(img_sr)

            psnr = self.criterion_PSNR(img_sr_, img_hr_)
            ssim = 1 - self.criterion_SSIM(img_sr_, img_hr_)  # invert

        return {'psnr': psnr, 'ssim': ssim}

    def validation_end(self, outputs):
        val_psnr_mean = 0
        val_ssim_mean = 0
        for output in outputs:
            val_psnr_mean += output['psnr']
            val_ssim_mean += output['ssim']
        val_psnr_mean /= len(outputs)
        val_ssim_mean /= len(outputs)
        return {'val/psnr': val_psnr_mean.item(),
                'val/ssim': val_ssim_mean.item()}

    def configure_optimizers(self):
        optimizer_G = optim.Adam(self.net_G.parameters(), lr=1e-4)
        optimizer_D = optim.Adam(self.net_D.parameters(), lr=1e-4)
        scheduler_G = StepLR(optimizer_G, step_size=1e+5, gamma=0.1)
        scheduler_D = StepLR(optimizer_D, step_size=1e+5, gamma=0.1)
        return [optimizer_D, optimizer_G], [scheduler_D, scheduler_G]

    @pl.data_loader
    def tng_dataloader(self):
        dataset = DatasetFromFolder(
            data_dir='./data/DIV2K/train',
            scale_factor=self.scale_factor,
            patch_size=self.patch_size
        )
        return DataLoader(dataset, self.batch_size, shuffle=True, num_workers=4)

    @pl.data_loader
    def val_dataloader(self):
        dataset = DatasetFromFolder(
            data_dir='./data/DIV2K/val',
            scale_factor=self.scale_factor,
            mode='eval'
        )
        return DataLoader(dataset, batch_size=1, num_workers=4)

    @pl.data_loader
    def test_dataloader(self):
        def get_loader(name):
            dataset = DatasetFromFolder(
                data_dir=f'./data/{name}/HR',
                scale_factor=self.scale_factor,
                mode='eval'
            )
            return DataLoader(dataset, batch_size=1, num_workers=4)

        out_dict = OrderedDict()
        for name in ['Set5', 'Set14', 'BSD100', 'Urban100']:
            out_dict[name] = get_loader(name)

        return out_dict
