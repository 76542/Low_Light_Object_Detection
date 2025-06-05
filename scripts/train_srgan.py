import os
import sys

# Add project root to sys.path so 'models', 'data', etc. can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image

from models.srgan.generator import SRGANGenerator  # now includes RFA!
from models.srgan.discriminator import SRGANDiscriminator
from models.srgan.utils import VGGLoss
from data.dataloaders import DIV2KDataset
from PIL import Image

# -------- Device selection for Apple Silicon --------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple Silicon MPS backend.")
else:
    device = torch.device("cpu")
    print("⚠️ MPS not available, using CPU.")

def train_srgan():
    # -------- Config --------
    batch_size = 4  # Keep small on M1 Mac
    num_epochs = 20  # You can increase this later
    warmup_epochs = 5
    lr = 1e-4

    # -------- Model Setup --------
    generator = SRGANGenerator().to(device)
    discriminator = SRGANDiscriminator().to(device)
    content_loss = VGGLoss().to(device)

    # -------- Optimizers --------
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    # -------- Data Transforms --------
    hr_transform = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])

    lr_transform = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # -------- Dataset & DataLoader --------
    train_dataset = DIV2KDataset(
        root_dir='data/div2k',
        lr_transform=lr_transform,
        hr_transform=hr_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # -------- Losses --------
    pixel_loss_fn = nn.MSELoss()
    bce_loss = nn.BCELoss()

    # -------- Training Loop --------
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        for i, (lr, hr) in enumerate(train_loader):
            lr, hr = lr.to(device), hr.to(device)

            # Generate SR image
            sr = generator(lr)

            # -------- Train Discriminator --------
            real_labels = torch.ones(hr.size(0)).to(device)
            fake_labels = torch.zeros(hr.size(0)).to(device)


            d_real = discriminator(hr)
            d_fake = discriminator(sr.detach())

            d_loss_real = bce_loss(d_real, real_labels)
            d_loss_fake = bce_loss(d_fake, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # -------- Train Generator --------
            d_fake = discriminator(sr)
            g_loss_gan = bce_loss(d_fake, real_labels)
            g_loss_content = content_loss(sr, hr)

            if epoch < warmup_epochs:
                g_loss = pixel_loss_fn(sr, hr)
            else:
                g_loss = g_loss_content + 1e-3 * g_loss_gan

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            # -------- Logging --------
            if (i + 1) % 10 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] [Batch {i+1}] "
                      f"D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

        # -------- Save Images & Checkpoints --------
        if (epoch + 1) % 5 == 0:
            os.makedirs("checkpoints/srgan", exist_ok=True)
            save_image(sr[:4], f"checkpoints/srgan/sr_epoch_{epoch+1}.png", nrow=2, normalize=True)
            torch.save(generator.state_dict(), f"checkpoints/srgan/generator_epoch_{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"checkpoints/srgan/discriminator_epoch_{epoch+1}.pth")
            print(f"✅ Checkpoint saved for epoch {epoch+1}")

if __name__ == "__main__":
    train_srgan()
