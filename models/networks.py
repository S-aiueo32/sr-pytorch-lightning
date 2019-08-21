from math import log2

from torch import nn


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.net(x)

class SRResNet(nn.Module):
    def __init__(self, scale_factor=4, base_filters=64, n_blocks=16):
        super(SRResNet, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(3, base_filters, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        body = [ResidualBlock(base_filters) for _ in range(n_blocks)]
        body += [nn.Conv2d(base_filters, base_filters, kernel_size=3, padding=1),
                 nn.BatchNorm2d(base_filters)]
        self.body = nn.Sequential(*body)

        tail = []
        for _ in range(int(log2(scale_factor))):
            r = 2 if scale_factor % 2 == 0 else 3
            tail += [
                nn.Conv2d(base_filters, base_filters * (2 ** 2), kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU()
            ]
        tail += [nn.Conv2d(base_filters, 3, kernel_size=9, padding=4),
                 nn.Tanh()]
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x) + x
        x = self.tail(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.PReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.net(x)

class Discriminator(nn.Sequential):
    def __init__(self, base_filters):
        
        def ConvBlock(in_channels, out_channels, stride):
            out = [
                nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(out_channels),
            ]
            return out

        super(Discriminator, self).__init__(
            nn.Conv2d(3, base_filters, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),

            *ConvBlock(base_filters, base_filters, 2),

            *ConvBlock(base_filters, base_filters * 2, 1),
            *ConvBlock(base_filters * 2, base_filters * 2, 2),

            *ConvBlock(base_filters * 2, base_filters * 4, 1),
            *ConvBlock(base_filters * 4, base_filters * 4, 2),

            *ConvBlock(base_filters * 4, base_filters * 8, 1),
            *ConvBlock(base_filters * 8, base_filters * 8, 2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_filters * 8, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )
