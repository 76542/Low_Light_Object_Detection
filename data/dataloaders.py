import os
from torch.utils.data import Dataset
from PIL import Image

class DIV2KDataset(Dataset):
    def __init__(self, root_dir, lr_transform=None, hr_transform=None):
        self.hr_dir = os.path.join(root_dir, 'HR')
        self.lr_dir = os.path.join(root_dir, 'LR')
        self.lr_transform = lr_transform
        self.hr_transform = hr_transform

        self.hr_images = sorted(os.listdir(self.hr_dir))
        self.lr_images = sorted(os.listdir(self.lr_dir))

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])

        hr_img = Image.open(hr_path).convert("RGB")
        lr_img = Image.open(lr_path).convert("RGB")

        if self.hr_transform:
            hr_img = self.hr_transform(hr_img)
        if self.lr_transform:
            lr_img = self.lr_transform(lr_img)

        return lr_img, hr_img
