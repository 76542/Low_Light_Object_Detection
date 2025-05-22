import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# --------------------------------------------
# VGG-based Content Loss (Perceptual Loss)
# --------------------------------------------
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg)[:36]).eval()  # Up to relu5_4

        # Freeze VGG weights
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss()

    def forward(self, sr, hr):
        """
        sr: Super-resolved image (output of generator)
        hr: High-resolution ground truth image
        """
        # Normalize to VGG's expected input range
        def preprocess(x):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(x.device)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(x.device)
            return (x - mean) / std

        sr_vgg = self.feature_extractor(preprocess(sr))
        hr_vgg = self.feature_extractor(preprocess(hr))
        return self.criterion(sr_vgg, hr_vgg)

# --------------------------------------------
# (Optional) PSNR Metric
# --------------------------------------------
def psnr(sr, hr, max_val=1.0):
    mse = F.mse_loss(sr, hr)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(max_val / torch.sqrt(mse))
