import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_image_files = os.listdir(lr_dir)
        self.hr_image_files = os.listdir(hr_dir)

    def __len__(self):
        return len(self.lr_image_files)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_image_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_image_files[idx])

        lr_image = Image.open(lr_path).convert('RGB')
        hr_image = Image.open(hr_path).convert('RGB')

        lr_tensor = self.transform(lr_image)
        hr_tensor = self.transform(hr_image)

        return lr_tensor, hr_tensor

    @staticmethod
    def transform(image):
        return torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
