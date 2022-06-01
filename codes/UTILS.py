import cv2
import math
import numpy as np

import torch

from PIL import Image
#from skimage.metrics import peak_signal_noise_ratio
#from skimage.metrics import structural_similarity as compare_ssim


def compute_psnr(images, labels):

    batch, _, _, _ = images.size()
    PSNR = 0
    for i in range(batch):
        PSNR += psnr(images[i] * 255, labels[i] * 255)

    PSNR = PSNR / batch

    return PSNR


def psnr(img1, img2):

    img1 = img1.cpu().detach().numpy()
    img2 = img2.cpu().detach().numpy()
    img1 = np.transpose(np.float64(img1), (1, 2, 0))
    img2 = np.transpose(np.float64(img2), (1, 2, 0))
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

"""
def compute_ssim(images, labels):

    batch, _, _, _ = images.size()
    SSIM = 0
    for i in range(batch):

        SSIM += ssim(images[i] * 255, labels[i] * 255)

    SSIM = SSIM / batch
    return SSIM


def ssim(img1, img2):

    img1 = img1.cpu().detach().numpy()
    img2 = img2.cpu().detach().numpy()
    img1 = np.transpose(np.uint8(img1), (1, 2, 0))
    img2 = np.transpose(np.uint8(img2), (1, 2, 0))
    ssim_value = compare_ssim(img1, img2, multichannel=True)

    return ssim_value
"""



