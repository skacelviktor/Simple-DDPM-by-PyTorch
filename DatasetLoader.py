from torch.utils.data.dataset import Dataset
import torchvision.transforms as T
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, image_path, image_size=28): # Pro MNIST obvykle 28x28
        self.image_path = image_path
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(), # Převede [0, 255] na [0.0, 1.0]
            # Normalizace: (x - 0.5) / 0.5 převede [0, 1] na [-1, 1]
            T.Normalize((0.5,), (0.5,)) 
        ])
        self.image_files = self._load_image_files()

    def _load_image_files(self):
        valid_extensions = ('.png', '.jpg', '.jpeg')
        return [os.path.join(os.getcwd(), self.image_path, f) for f in os.listdir(self.image_path) 
                if f.lower().endswith(valid_extensions)]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = Image.open(image_file).convert('L')
        return self.transform(image)