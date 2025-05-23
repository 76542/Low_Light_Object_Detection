import torch
import torch.nn as nn

class SRGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(SRGANDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride):
            return nn.Sequential(
                nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_filters),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            discriminator_block(64, 64, stride=2),
            discriminator_block(64, 128, stride=1),
            discriminator_block(128, 128, stride=2),
            discriminator_block(128, 256, stride=1),
            discriminator_block(256, 256, stride=2),
            discriminator_block(256, 512, stride=1),
            discriminator_block(512, 512, stride=2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        out = self.model(x)
        return torch.sigmoid(out.view(out.size(0)))