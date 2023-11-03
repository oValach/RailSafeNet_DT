from pathlib import Path
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms.functional import pil_to_tensor, to_tensor
from PIL import Image
import numpy as np
import torch
import cv2
import re


class CustomDataset(VisionDataset):
    def __init__(self, image_folder, mask_folder, seed, subset, test_val_fraction=0.1):
        self.image_folder = Path(image_folder)
        self.mask_folder = Path(mask_folder)
        self.test_val_fraction = test_val_fraction

        # all files
        self.image_list = np.array(sorted(Path(self.image_folder).glob("*")))
        self.mask_list = np.array(sorted(Path(self.mask_folder).glob("*")))

        for file_path in self.image_list:
            if 'desktop.ini' in file_path.name:
                file_path.unlink()
        for file_path in self.mask_list:
            if 'desktop.ini' in file_path.name:
                file_path.unlink()

        self.mask_list = np.array(sorted(self.mask_list, key=lambda path: int(re.findall(r'\d+', path.stem)[0]) if re.findall(r'\d+', path.stem) else 0))

        if seed:  # rng locked data shuffle and split
            np.random.seed(seed)
            indices = np.arange(len(self.image_list))
            np.random.shuffle(indices)
            self.image_list = self.image_list[indices]
            self.mask_list = self.mask_list[indices]
        if subset == 'Train':  # split dataset to 1-2*fraction of train data, default fraction == 0.1
            self.image_names = self.image_list[:int(np.ceil(len(self.image_list) * (1 - self.test_val_fraction*2)))]
            self.mask_names = self.mask_list[:int(np.ceil(len(self.mask_list) * (1 - self.test_val_fraction*2)))]
        elif subset == 'Test':  # test data of lenght of fraction
            self.image_names = self.image_list[int(np.ceil(len(self.image_list) * (1 - self.test_val_fraction*2))):int(np.ceil(len(self.image_list) * (1 - self.test_val_fraction)))]
            self.mask_names = self.mask_list[int(np.ceil(len(self.mask_list) * (1 - self.test_val_fraction*2))):int(np.ceil(len(self.mask_list) * (1 - self.test_val_fraction)))]
        elif subset == 'Val':  # val data - different part of data than test, also of length fraction
            self.image_names = self.image_list[int(np.ceil(len(self.image_list) * (1 - self.test_val_fraction))):]
            self.mask_names = self.mask_list[int(np.ceil(len(self.mask_list) * (1 - self.test_val_fraction))):]
        else:
            print('Invalid data subset.')

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = self.image_names[idx]
        mask_path = self.mask_names[idx]

        with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:

            im_jpg = cv2.imread(image_file.name)
            im_jpg = cv2.resize(im_jpg, (224, 224), interpolation=cv2.INTER_NEAREST)
            image = torch.tensor(im_jpg, dtype=torch.float32)
            image = torch.div(image.permute(2, 0, 1), 254) # input normalization

            id_map = cv2.imread(mask_file.name, cv2.IMREAD_GRAYSCALE)
            id_map = cv2.resize(id_map, (224, 224), interpolation=cv2.INTER_NEAREST) # keep the initial values after resizing
            mask = torch.tensor(id_map, dtype=torch.float32).long() # pixel value == class id, if 255 == not classified

            # ignore not well segmented classes
            ignore = True
            if ignore:
                ignore_list = [0,1,2,6,8,9,15,16,19,20]
                for cls in ignore_list:
                    mask[mask==cls] = 255

            #import matplotlib.pyplot as plt
            #mask[mask == 255] = 0

            #plt.figure()
            #plt.subplot(1, 2, 1)
            #plt.imshow(mask, cmap='gray')
            #plt.title('Ground truth')

            #plt.subplot(1, 2, 2) 
            #plt.imshow(image[0], cmap='gray')
            #plt.title('Output')
            #plt.show()

            sample = [image, mask]
            return sample
