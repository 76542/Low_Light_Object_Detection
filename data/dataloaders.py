import os
from torch.utils.data import Dataset
from PIL import Image

# ------------------------------
# Dataset for Zero-DCE (LoL)
# ------------------------------
class LowLightDataset(Dataset):
    def __init__(self, root_dir, transform=None, split="train"):
        self.low_dir = os.path.join(root_dir, split, "low")
        self.high_dir = os.path.join(root_dir, split, "high")
        self.transform = transform
        self.image_filenames = os.listdir(self.low_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        low_path = os.path.join(self.low_dir, self.image_filenames[idx])
        high_path = os.path.join(self.high_dir, self.image_filenames[idx])

        low_img = Image.open(low_path).convert("RGB")
        high_img = Image.open(high_path).convert("RGB")

        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)

        return low_img, high_img


# ------------------------------
# Dataset for SRGAN (DIV2K)
# ------------------------------
class DIV2KDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to dataset containing 'HR' and 'LR' folders
            transform (callable, optional): Optional transform to be applied on both HR and LR images
        """
        self.hr_dir = os.path.join(root_dir, 'HR')  # High-resolution images
        self.lr_dir = os.path.join(root_dir, 'LR')  # Low-resolution images
        self.transform = transform

        self.hr_images = sorted(os.listdir(self.hr_dir))
        self.lr_images = sorted(os.listdir(self.lr_dir))

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])

        hr_img = Image.open(hr_path).convert("RGB")
        lr_img = Image.open(lr_path).convert("RGB")

        if self.transform:
            hr_img = self.transform(hr_img)
            lr_img = self.transform(lr_img)

        return lr_img, hr_img