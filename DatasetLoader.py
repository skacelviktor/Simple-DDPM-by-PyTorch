"""
Dataset loader for DDPM (Denoising Diffusion Probabilistic Models)

This code provides a PyTorch Dataset class for loading and preprocessing
images for DDPM training. It handles:
- Loading images from disk
- Resizing to uniform dimensions
- Converting to tensors
- Normalizing to [-1, 1] range
"""

from torch.utils.data.dataset import Dataset
import torchvision.transforms as T
from PIL import Image
import os

class ImageDataset(Dataset):
    """
    Custom Dataset for loading images for DDPM training.
    
    Automatically preprocesses images to consistent size and normalizes
    pixel values to [-1, 1] range, which is standard for diffusion models.
    """
    
    def __init__(self, image_path, image_size=28):
        self.image_path = image_path
        
        # Define preprocessing pipeline
        # Applied sequentially to each image during loading
        self.transform = T.Compose([
            # 1. Resize to uniform size
            T.Resize((image_size, image_size)),
            
            # 2. Convert PIL Image to PyTorch Tensor
            T.ToTensor(),
            
            # 3. Normalize to [-1, 1] range
            # Formula: (x - mean) / std
            # With mean=0.5 and std=0.5: (x - 0.5) / 0.5
            T.Normalize((0.5,), (0.5,))
        ])
        
        # Load list of all image file paths
        self.image_files = self._load_image_files()

    def _load_image_files(self):
        # supported image formats
        valid_extensions = ('.png', '.jpg', '.jpeg')
        
        # List all files in directory
        return [os.path.join(os.getcwd(), self.image_path, f) for f in os.listdir(self.image_path) 
                if f.lower().endswith(valid_extensions)]
    
    def __len__(self):
        #Number of images
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get file path for this index
        image_file = self.image_files[idx]
        
        # Load the image and convert it to grayscale ('L' mode = 8-bit grayscale)
        # For color images, use "RGB" instead and set in_channels = 3 in the U-Net
        image = Image.open(image_file).convert('L')
        
        # Apply preprocessing pipeline: resize, convert to tensor, normalize
        return self.transform(image)