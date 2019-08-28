from math import log2

import torch
from torch import nn


class SRCNN(nn.Sequential):
    """
    PyTorch Module for SRCNN, https://arxiv.org/pdf/1501.00092.pdf.
    """

    def __init__(self):
        super(SRCNN, self).__init__(
            nn.Conv2d(3, 64, 9, padding=4),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(32, 3, 5, padding=2)
        )


class SRResNet(nn.Module):
    """
    PyTorch Module for SRGAN, https://arxiv.org/pdf/1609.04802.
    """

    def __init__(self, scale_factor=4, ngf=64, n_blocks=16):
        super(SRResNet, self).__init__()

        self.head = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(3, ngf, kernel_size=9),
            nn.PReLU()
        )
        self.body = nn.Sequential(
            *[SRGANBlock(ngf) for _ in range(n_blocks)],
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf, kernel_size=3),
            nn.BatchNorm2d(ngf)
        )
        self.tail = nn.Sequential(
            UpscaleBlock(scale_factor, ngf, act='prelu'),
            nn.ReflectionPad2d(4),
            nn.Conv2d(ngf, 3, kernel_size=9),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x) + x
        x = self.tail(x)
        return (x + 1) / 2


class EDSR(nn.Module):
    """
    PyTorch Module for EDSR, https://arxiv.org/pdf/1707.02921.
    """

    def __init__(self, scale_factor=4, ngf=256, n_blocks=32, res_scale=0.1):
        super(EDSR, self).__init__()

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, ngf, kernel_size=3),
        )
        self.body = nn.Sequential(
            *[EDSRBlock(ngf, res_scale) for _ in range(n_blocks)],
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf, kernel_size=3),
        )
        self.tail = nn.Sequential(
            UpscaleBlock(scale_factor, ngf),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, 3, kernel_size=3)
        )

        # mean value of DIV2K
        self.register_buffer(
            name='mean',
            tensor=torch.tensor([[[0.4488]], [[0.4371]], [[0.4040]]],
                                requires_grad=False)
        )

    def __normalize(self, x):
        x.sub_(self.mean.detach())

    def __denormalize(self, x):
        x.add_(self.mean.detach())

    def forward(self, x):
        self.__normalize(x)

        x = self.head(x)
        x = self.body(x) + x
        x = self.tail(x)

        self.__denormalize(x)

        return x


#################
# Building Blocks
#################
class SRGANBlock(nn.Module):
    """
    Building block of SRGAN.
    """

    def __init__(self, dim):
        super(SRGANBlock, self).__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.BatchNorm2d(dim),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.net(x)


class EDSRBlock(nn.Module):
    """
    Building block of EDSR.
    """

    def __init__(self, dim, res_scale=0.1):
        super(EDSRBlock, self).__init__()
        self.res_scale = res_scale
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
        )

    def forward(self, x):
        return x + self.net(x) * self.res_scale


class UpscaleBlock(nn.Sequential):
    """
    Upscale block using sub-pixel convolutions.
    `scale_factor` can be selected from {2, 3, 4, 8}.
    """

    def __init__(self, scale_factor, dim, act=None):
        assert scale_factor in [2, 3, 4, 8]

        layers = []
        for _ in range(int(log2(scale_factor))):
            r = 2 if scale_factor % 2 == 0 else 3
            layers += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim * r * r, kernel_size=3),
                nn.PixelShuffle(r),
            ]

            if act == 'relu':
                layers += [nn.ReLU(True)]
            elif act == 'prelu':
                layers += [nn.PReLU()]

        super(UpscaleBlock, self).__init__(*layers)


class Discriminator(nn.Sequential):
    """
    Discriminator for SRGAN.
    Dense layers are replaced with global poolings and 1x1 convolutions.
    """

    def __init__(self, ndf):

        def ConvBlock(in_channels, out_channels, stride):
            out = [
                nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(out_channels),
            ]
            return out

        super(Discriminator, self).__init__(
            nn.Conv2d(3, ndf, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),

            *ConvBlock(ndf, ndf, 2),

            *ConvBlock(ndf, ndf * 2, 1),
            *ConvBlock(ndf * 2, ndf * 2, 2),

            *ConvBlock(ndf * 2, ndf * 4, 1),
            *ConvBlock(ndf * 4, ndf * 4, 2),

            *ConvBlock(ndf * 4, ndf * 8, 1),
            *ConvBlock(ndf * 8, ndf * 8, 2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ndf * 8, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )
