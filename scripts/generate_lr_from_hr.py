import os
from PIL import Image
from tqdm import tqdm

def downscale_images(hr_dir, lr_dir, scale=4):
    os.makedirs(lr_dir, exist_ok=True)
    hr_files = sorted(os.listdir(hr_dir))

    for filename in tqdm(hr_files, desc="Downscaling images"):
        hr_path = os.path.join(hr_dir, filename)
        lr_path = os.path.join(lr_dir, filename)

        hr_img = Image.open(hr_path).convert("RGB")
        w, h = hr_img.size
        lr_img = hr_img.resize((w // scale, h // scale), Image.BICUBIC)
        lr_img.save(lr_path)

if __name__ == "__main__":
    hr_dir = "data/div2k/HR"
    lr_dir = "data/div2k/LR"
    downscale_images(hr_dir, lr_dir)
