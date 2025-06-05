import os
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Import the Zero-DCE model and loss functions
from models.zero_dce.model import ZeroDCE
from models.zero_dce.utils import (
    spatial_consistency_loss, 
    exposure_loss, 
    color_constancy_loss, 
    illumination_smoothness_loss
)

# Import your dataset loader (assumed to be implemented in data/dataloaders.py)
# This dataset loader should return pairs of (low_light_image, ground_truth or placeholder)
from data.dataloaders import LowLightDataset

def main():
    # -------------------------------
    # Configuration and Hyperparameters
    # -------------------------------
    num_epochs = 50               # Number of training epochs
    batch_size = 4                # Batch size for DataLoader
    learning_rate = 1e-4          # Learning rate for the optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # -------------------------------
    # Model Setup
    # -------------------------------
    # Initialize the Zero-DCE model and move it to the selected device (GPU/CPU)
    model = ZeroDCE(channels=3, iterations=8).to(device)
    
    # Define the optimizer (Adam) to update the model's weights
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # -------------------------------
    # Data Loading
    # -------------------------------
    # Define the transformation – converts images to tensors (normalize pixel values to [0,1])
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Create an instance of the low-light dataset.
    # The LowLightDataset should accept at least a root directory, a transform, and a mode (e.g., 'train').
    train_dataset = LowLightDataset(root_dir="data/lol_dataset", transform=transform, mode="train")
    
    # Create a DataLoader to iterate over the dataset during training.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # -------------------------------
    # Training Loop
    # -------------------------------
    model.train()  # Set model to training mode
    for epoch in range(num_epochs):
        epoch_loss = 0.0  # To accumulate loss over the epoch
        for batch_idx, (input_img, _) in enumerate(train_loader):
            # Move input images to device
            input_img = input_img.to(device)
            
            # Zero the gradients for the optimizer
            optimizer.zero_grad()
            
            # Forward pass: obtain the enhanced image and the list of enhancement curves
            enhanced_img, curves = model(input_img)
            
            # Compute the losses:
            # 1. Spatial Consistency Loss (preserves structure)
            loss_spatial = spatial_consistency_loss(enhanced_img, input_img)
            # 2. Exposure Loss (guides brightness to a target mean)
            loss_exposure = exposure_loss(enhanced_img, patch_size=16, mean_val=0.6)
            # 3. Color Constancy Loss (ensures balanced colors)
            loss_color = color_constancy_loss(enhanced_img)
            # 4. Illumination Smoothness Loss (ensures smooth curves)
            loss_smoothness = illumination_smoothness_loss(curves)
            
            # Total loss is the sum of all loss terms
            total_loss = loss_spatial + loss_exposure + loss_color + loss_smoothness
            
            # Backward pass: compute gradients
            total_loss.backward()
            # Update the weights
            optimizer.step()
            
            # Accumulate the total loss for logging
            epoch_loss += total_loss.item()
        
        # Log progress for the current epoch
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        # Save model checkpoint every 10 epochs (adjust as needed)
        if (epoch + 1) % 10 == 0:
            checkpoint_dir = "checkpoints/zero_dce"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"zerodce_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at {checkpoint_path}")

if __name__ == "__main__":
    main()
