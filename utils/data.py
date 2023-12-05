import numpy as np
from PIL import Image
import os


def resize_image(image_path: str, scale):
    with Image.open(image_path) as img:
        image_resize = img.resize(
            (int(img.width * scale), int(img.height * scale)))
        image_resizedPath = image_path.split('/')[-1][:-4] + "_resized.png"
        image_resize.save(image_resizedPath)
    return image_resizedPath


def downsize_upsize_image(image_path: str, scale):
    scaled = resize_image(image_path, 1 / scale)
    scaled = resize_image(scaled, scale)
    return scaled


def crop_image(image_path: str):
    image_croppedPath = image_path.split('/')[-1]
    with Image.open(image_path) as img:
        width, height = img.size
        left = (width - 800)/2
        top = (height - 800)/2
        right = (width + 800)/2
        bottom = (height + 800)/2

        img_cropped = img.crop((left, top, right, bottom))
        img_cropped.save(image_croppedPath)

    return image_croppedPath
