import torch
import torch.nn as nn

# ----------------------------------------
# Residual Block used inside the Generator
# ----------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(), #parametric ReLu allows to learn the negative slope
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)  #skip connection : adds input to output 


# ----------------------------------------
# SRGAN Generator Network
# ----------------------------------------
class SRGANGenerator(nn.Module):
    def __init__(self, in_channels=3, num_res_blocks=16):
        super(SRGANGenerator, self).__init__()

        # First convolution + activation
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        # 16 residual blocks
        res_blocks = [ResidualBlock(64) for _ in range(num_res_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)

        # Post-residual merge conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        # Upsampling (PixelShuffle doubles resolution)
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU()
        )

        # Final output
        self.conv3 = nn.Conv2d(64, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.conv1(x)              # First conv
        res = self.res_blocks(initial)       # Residual pathway
        res = self.conv2(res)                # Merge back
        out = initial + res                  # Residual connection
        out = self.upsample(out)             # Upsample by 4x
        out = self.conv3(out)                # Final conv
        return torch.clamp(out, 0.0, 1.0)     # Clamp to valid image range