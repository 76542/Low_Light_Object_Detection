import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.zero_dce.model import ZeroDCE
from models.zero_dce.utils import (
    spatial_consistency_loss,
    exposure_loss,
    color_constancy_loss,
    illumination_smoothness_loss
)
from data.dataloaders import LowLightDataset

# ========== Config ==========
root_dir = "data/lol_dataset"
num_epochs = 25
batch_size = 4
lr = 1e-4

# ========== Device (MPS support) ==========
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple MPS (GPU)")
else:
    device = torch.device("cpu")
    print("⚠️ MPS not available, using CPU")

# ========== Training Loop ==========
def train():
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = LowLightDataset(root_dir=root_dir, transform=transform, split="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = ZeroDCE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        print(f"\n🌀 Epoch [{epoch+1}/{num_epochs}]")
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            low_img, high_img = batch
            low_img = low_img.to(device)

            enhanced_img, curves = model(low_img)

            loss1 = spatial_consistency_loss(enhanced_img, low_img)
            loss2 = exposure_loss(enhanced_img)
            loss3 = color_constancy_loss(enhanced_img)
            loss4 = illumination_smoothness_loss(curves)

            loss = loss1 + loss2 + loss3 + loss4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        end_time = time.time()

        print(f"✅ Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")
        print(f"⏱️ Epoch Time: {end_time - start_time:.2f} seconds")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            os.makedirs("checkpoints/zero_dce", exist_ok=True)
            ckpt_path = f"checkpoints/zero_dce/zerodce_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    train()
