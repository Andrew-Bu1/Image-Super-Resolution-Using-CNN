from PIL import Image
import torch
import os
from torch.utils.data import Dataset


def process_image(image_path: str, scale=2, batch_size=32):
    """
    Processes an image and extracts batches for training.

    Args:
        image_path: Path to the image.
        scale: Scaling factor for downsampling.
        batch_size: Number of batches to extract.

    Returns:
        A tuple of tensors containing LR and HR batches.
    """
    with Image.open(image_path) as img:
        # Get the original image dimensions
        original_width, original_height = img.size

        # Calculate scaling factor based on target size
        if original_width > original_height:
            scale = batch_size / original_width
        else:
            scale = batch_size / original_height

        # Resize the image with downscaling
        img_resized = img.resize(
            (int(original_width * scale), int(original_height * scale)))

        # Create a copy for original HR
        img_hr = img.copy()

        # Extract batches
        lr_batches, hr_batches = [], []
        for i in range(0, img_resized.shape[0] - batch_size + 1, batch_size):
            for j in range(0, img_resized.shape[1] - batch_size + 1, batch_size):
                # Convert to tensors
                lr_batch_tensor = torch.from_numpy(
                    img_resized[i:i + batch_size, j:j + batch_size]).float()
                hr_batch_tensor = torch.from_numpy(
                    img_hr[i:i + batch_size, j:j + batch_size]).float()

                lr_batches.append(lr_batch_tensor)
                hr_batches.append(hr_batch_tensor)

        lr_batches_batch = torch.stack(lr_batches[:batch_size])
        hr_batches_batch = torch.stack(hr_batches[:batch_size])

    return lr_batches_batch, hr_batches_batch


class SRDataset(Dataset):
    """
    Dataset class for loading and processing image pairs for SR training.

    Args:
        image_dir: Directory containing LR images.
        hr_image_dir: Directory containing HR images (optional).
        scale: Scaling factor for downsampling.
        batch_size: Number of batches to extract.
    """

    def __init__(self, image_dir, hr_image_dir=None, scale=2, batch_size=32):
        self.image_dir = image_dir
        self.hr_image_dir = hr_image_dir
        self.scale = scale
        self.batch_size = batch_size
        self.image_paths = os.listdir(image_dir)

        if not hr_image_dir:
            self.hr_image_paths = self.image_paths
        else:
            self.hr_image_paths = os.listdir(hr_image_dir)

    def __getitem__(self, index):
        low_image_path = os.path.join(self.image_dir, self.image_paths[index])
        high_image_path = os.path.join(
            self.hr_image_dir, self.hr_image_paths[index])

        try:
            low_patch_tensor, high_patch_tensor = process_image(
                low_image_path, scale=self.scale, batch_size=self.batch_size)
        except Exception as e:
            print(f"Error loading image {low_image_path}: {e}")
            raise

        return low_patch_tensor, high_patch_tensor

    def __len__(self):
        return len(self.image_paths) // self.batch_size
