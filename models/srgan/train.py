# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import models
# from torch.utils.data import DataLoader
# from torchvision.transforms import ToTensor, Compose, Resize
# from torchvision.transforms.functional import InterpolationMode
# from tqdm import tqdm

# from models.srgan.generator import Generator
# from models.srgan.discriminator import Discriminator
# from data.dataloaders import SRDataset  # Custom dataset for SRGAN


# # Perceptual loss using VGG19 features
# class VGGPerceptualLoss(nn.Module):
#     def __init__(self):
#         super(VGGPerceptualLoss, self).__init__()
#         vgg = models.vgg19(pretrained=True).features
#         self.vgg_layers = nn.Sequential(*list(vgg[:36])).eval()
#         for param in self.vgg_layers.parameters():
#             param.requires_grad = False

#     def forward(self, sr, hr):
#         return nn.functional.mse_loss(self.vgg_layers(sr), self.vgg_layers(hr))

# # SRGAN training function
# def train_srgan():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     generator = Generator().to(device)
#     discriminator = Discriminator().to(device)
#     vgg_loss = VGGPerceptualLoss().to(device)

#     adversarial_criterion = nn.BCELoss()
#     g_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
#     d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4)

#     # Define transforms for LR and HR images
#     transform = Compose([
#         Resize((96, 96), interpolation=InterpolationMode.BICUBIC),
#         ToTensor()
#     ])
#     target_transform = Compose([
#         Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
#         ToTensor()
#     ])

#     # Load dataset
#     train_dataset = SRDataset(root_dir="data/div2k", transform=transform, target_transform=target_transform)
#     train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

#     # Train for 10 epochs
#     for epoch in range(10):
#         g_loss_epoch = 0.0
#         d_loss_epoch = 0.0

#         for lr_imgs, hr_imgs in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
#             lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

#             # === Train Discriminator ===
#             real_labels = torch.ones((lr_imgs.size(0),), device=device)
#             fake_labels = torch.zeros((lr_imgs.size(0),), device=device)

#             with torch.no_grad():
#                 sr_imgs = generator(lr_imgs)

#             real_outputs = discriminator(hr_imgs)
#             fake_outputs = discriminator(sr_imgs.detach())

#             d_loss_real = adversarial_criterion(real_outputs, real_labels)
#             d_loss_fake = adversarial_criterion(fake_outputs, fake_labels)
#             d_loss = (d_loss_real + d_loss_fake) / 2

#             d_optimizer.zero_grad()
#             d_loss.backward()
#             d_optimizer.step()

#             # === Train Generator ===
#             sr_imgs = generator(lr_imgs)
#             fake_outputs = discriminator(sr_imgs)

#             content_loss = vgg_loss(sr_imgs, hr_imgs)
#             adversarial_loss = adversarial_criterion(fake_outputs, real_labels)
#             g_loss = content_loss + 1e-3 * adversarial_loss

#             g_optimizer.zero_grad()
#             g_loss.backward()
#             g_optimizer.step()

#             g_loss_epoch += g_loss.item()
#             d_loss_epoch += d_loss.item()

#         print(f"[Epoch {epoch+1}] G Loss: {g_loss_epoch:.4f} | D Loss: {d_loss_epoch:.4f}")

#         # Save models every 5 epochs
#         if (epoch + 1) % 2 == 0:
#             os.makedirs("checkpoints/srgan", exist_ok=True)
#             torch.save(generator.state_dict(), f"checkpoints/srgan/generator_epoch_{epoch+1}.pth")
#             torch.save(discriminator.state_dict(), f"checkpoints/srgan/discriminator_epoch_{epoch+1}.pth")
#             print("✅ Checkpoints saved.")

# if __name__ == "__main__":
#     train_srgan()