#download dataset
#import kaggle
import zipfile
import os
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
from collections import OrderedDict
import gdown
import pickle

#for image related ops
import cv2
import pydicom
import skimage
from skimage import io, img_as_float
from skimage.filters import gaussian

def CLAHE(image_path):
    img = cv2.imread(image_path, 0)
    clahe = cv2.createCLAHE(clipLimit=40, tileGridSize=(8,8))
    clahe_image = clahe.apply(img)
    return clahe_image

def UM(img_path, sigma, amount):
    img = cv2.imread(img_path, 0)
    img = img / 255.
    gaussian_img = gaussian(img, sigma=sigma, mode="constant", cval=0.0)
    unsharped_mask = (img - gaussian_img) * amount
    um_img = img + unsharped_mask
    um_img = np.clip(um_img, float(0), float(1)) # Interval [0.0, 1.0]
    um_img = (um_img * 255).astype(np.uint8) # Interval [0,255]
    return um_img

def histogram(data):
    '''Generates the histogram for the given data.
    Parameters:
    data: data to make the histogram.
    Returns: histogram, bins.
    '''

    pixels, count = np.unique(data, return_counts=True)
    hist = OrderedDict()

    for i in range(len(pixels)):
        hist[pixels[i]] = count[i]

    return np.array(list(hist.values())), np.array(list(hist.keys()))

def HEF(image_path, D0):
    img = cv2.imread(image_path, 0)
    img_fft = np.fft.fft2(img)
    img_ffts = np.fft.fftshift(img_fft)

    #High-pass Gaussian filter
    (P, Q) = img_ffts.shape
    H = np.zeros((P,Q))
    for u in range(P):
        for v in range(Q):
            H[u, v] = 1.0 - np.exp(- ((u - P / 2.0) ** 2 + (v - Q / 2.0) ** 2) / (2 * (D0 ** 2)))
    k1 = 0.5 ; k2 = 0.75
    HFEfilt = k1 + k2 * H # Apply High-frequency emphasis

    # Apply HFE filter to FFT of original image
    HFE = HFEfilt * img_ffts

    img_hef = np.real(np.fft.ifft2(np.fft.ifftshift(HFE)))  # HFE filtering done

    # HE part
    # Building the histogram
    hist, bins = histogram(img_hef)
    # Calculating probability for each pixel
    pixel_probability = hist / hist.sum()
    # Calculating the CDF (Cumulative Distribution Function)
    cdf = np.cumsum(pixel_probability)
    cdf_normalized = cdf * 255
    hist_eq = {}
    for i in range(len(cdf)):
        hist_eq[bins[i]] = int(cdf_normalized[i])

    for i in range(P):
        for j in range(Q):
            img[i][j] = hist_eq[img_hef[i][j]]

    return img.astype(np.uint8)
