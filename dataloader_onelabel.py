from pathlib import Path
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms.functional import pil_to_tensor, to_tensor
from PIL import Image
import numpy as np
import torch

class CustomDataset(VisionDataset):
    def __init__(self, subset):
        self.image_folder = Path('rs19_val\jpgs\\rs19_val\\rs00091.jpg')
        self.mask_folder = Path('rs19_val\\uint8\\objects\\91_5.png')
        self.mask_folderi = Path('rs19_val\\uint8\\objects\\91_5i.png') # inverted

        self.image_list = np.array(Path(self.image_folder))
        self.mask_list = np.array(Path(self.mask_folder))
        self.mask_listi = np.array(Path(self.mask_folderi))

        if subset == 'Train':
            self.image_names = [self.image_list]*128
            self.mask_names = [self.mask_list]*128 #+ [self.mask_listi]*64 # +inverted
        elif subset == 'Test':
            self.image_names = [self.image_list]*16
            self.mask_names = [self.mask_list]*16 #+ [self.mask_listi]*8 # +inverted

    def __len__(self) -> int:
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_path = str(self.image_names[idx])
        mask_path = str(self.mask_names[idx])

        with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
            image = Image.open(image_file)
            mask = Image.open(mask_file)
            mask = mask.convert("L")
            
            image = image.resize((224, 224), Image.BILINEAR)
            mask = mask.resize((224, 224), Image.BILINEAR)

            image = torch.div(pil_to_tensor(image).float(), 254)
            mask = torch.div(pil_to_tensor(mask).float(), 254)
            mask_norm = torch.squeeze(mask).long()

            sample = [image, mask_norm]
            return sample
