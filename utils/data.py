import numpy as np
from PIL import Image
import os
from pathlib import Path
import shutil
from modules import constants as const


def resize_image(image_path: str, scale):
    with Image.open(image_path) as img:
        image_resize = img.resize(
            (int(img.width * scale), int(img.height * scale)))
        image_resize.save(image_path)
    return image_path


def downsize_upsize_image(image_path: str, scale):
    scaled = resize_image(image_path, 1 / scale)
    scaled = resize_image(scaled, scale)
    return scaled


def crop_image(image_path: str):
    with Image.open(image_path) as img:
        width, height = img.size
        left = (width - 800)/2
        top = (height - 800)/2
        right = (width + 800)/2
        bottom = (height + 800)/2

        img_cropped = img.crop((left, top, right, bottom))
        img_cropped.save(image_path)

    return image_path


print(crop_image('utils/0900.png'))


def proccess_image():

    # Define the data folder path
    data_folder = Path("data/DIV2K")

    # Get all image files recursively
    for image_file in data_folder.rglob("*.png"):

        # Convert Path object to string for script execution
        image_path = str(image_file)
        crop_image(image_path)
        downsize_upsize_image(image_path)


def sort_image(number_):

    # ` Define the data folder path and subfolders
    data_folder = "data/DIV2K"
    train_folder = "data/train"
    test_folder = "data/test"
    val_folder = "data/validation"

    # Get all image filenames in the DIV2K folder
    image_filenames = sorted(os.listdir(data_folder))

    # Pick the first 60 images for train
    train_images = image_filenames[:const.DEFAULT_TRAIN_DATASET]

    # Copy the first 20 images to test and validation folders
    test_images = train_images[:const.DEFAULT_TEST_DATASET]
    val_images = train_images[:const.DEFAULT_VALIDATION_DATASET]

    # Create folders if they don't exist
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)

    # Copy images to respective folders
    for filename in train_images:
        shutil.copy(os.path.join(data_folder, filename),
                    os.path.join(train_folder, filename))

    for filename in test_images:
        shutil.copy(os.path.join(data_folder, filename),
                    os.path.join(test_folder, filename))

    for filename in val_images:
        shutil.copy(os.path.join(data_folder, filename),
                    os.path.join(val_folder, filename))

    print("Images successfully copied!")
