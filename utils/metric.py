import torch


def calculate_psnr(img1, img2, max_pixel_value=255.):
    return 10. * torch.log10(max_pixel_value**2 / torch.mean((img1 - img2)))
