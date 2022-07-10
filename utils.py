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

def copy_resize_images(image_source_dir, image_destination_dir, unique_affix, target_size=(224, 224)):
    images = []
    for root, _, files in os.walk(image_source_dir, topdown=True):
        for file in files:
            if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".dcm")) and unique_affix in file:
                images.append(root + "/" + file)
            else:
                continue
    print(len(images))
    for image in images:
        img_name = os.path.basename(image)
        img_name = os.path.splitext(img_name)[0]
        if image.endswith(".dcm"):
            img = pydicom.dcmread(image)
            img_pixel_array = img.pixel_array
            img_pixel_array = (img_pixel_array / img_pixel_array.max()) * 255
            img_pixel_array = img_pixel_array.astype(np.uint8)
        else:
            img_pixel_array = cv2.imread(image, 0)
        img = cv2.resize(img_pixel_array, target_size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(image_destination_dir + "/" + img_name + ".png", img)
    print(f"finish write {len(images)} image to {image_destination_dir}")

def create_df(dataset_dir):
    # create dataframe of images_path and its class for training set
    x = []
    y = []
    normal_count = 0
    tb_count = 0
    other_lung_disease_count = 0
    covid_count = 0
    pneumonia_count = 0

    for root, _, files in os.walk(dataset_dir, topdown=True):
        if "ipynb" in root:
            continue
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                x.append(os.path.join(root,
                                      file))  # create a pair of dir and filename e.g. ("dataset/image", "covid19_1.png")
                if "covid19_pneumonia" in root:
                    # if other_lung_disease_count >= 1442:
                    #    continue
                    y.append("other_lung_disease")
                    other_lung_disease_count += 1
                elif "drug_res" in root or "montgomery" in root or "shenzhen" in root:
                    y.append("tuberculosis")
                    tb_count += 1
                elif "tb_db" in root:
                    # if normal_count >= 1442:
                    #    continue
                    y.append("normal")
                    normal_count += 1
                """
                if "COVID" in file:
                    y.append("covid19")
                    covid_count += 1
                elif "PNEUMONIA" in file:
                    y.append("pneumonia")
                    pneumonia_count += 1
                """
            else:
                continue
    # print(f"finished, normal data: {normal_count}, tb data: {tb_count}, other lung disease data: {other_lung_disease_count}")
    print(
        f"finished, normal data: {normal_count}, tb data: {tb_count}, other lung disease data: {other_lung_disease_count}")
    data = list(zip(x, y))
    df = pd.DataFrame(data, columns=["images_path", "labels"])
    df = df.sample(frac=1).reset_index(drop=True)
    print("==========================================================")
    print(df.value_counts("labels"))
    return df


def apply_enhancement(algorithm, image_source_dir):  # algorithm has three option (CLAHE, UM, and HEF)
    images = []
    for root, _, files in os.walk(image_source_dir, topdown=True):
        if "ipynb" in root:
            continue
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                images.append((root, file))  # create a pair of dir and filename e.g. ("dataset/image", "covid19_1.png")
            else:
                continue
    print(len(images))
    if algorithm == "CLAHE":
        for img_dir, img in images:
            img_path = os.path.join(img_dir, img)
            clahe_img = CLAHE(img_path)

            folder_name = os.path.basename(img_dir)
            save_dir = os.path.join("x-ray dataset_CLAHE", folder_name)
            cv2.imwrite(f"{save_dir}/CLAHE_{img}", clahe_img)
        print("finish applying CLAHE")

    elif algorithm == "UM":
        for img_dir, img in images:
            img_path = os.path.join(img_dir, img)
            um_img = UM(img_path, sigma=5, amount=1)

            folder_name = os.path.basename(img_dir)
            save_dir = os.path.join("x-ray dataset_UM", folder_name)
            cv2.imwrite(f"{save_dir}/UM_{img}", um_img)
        print("finish applying UM")

    elif algorithm == "HEF":
        for img_dir, img in images:
            img_path = os.path.join(img_dir, img)
            hef_img = HEF(img_path, D0=40)

            folder_name = os.path.basename(img_dir)
            save_dir = os.path.join("x-ray dataset_HEF", folder_name)
            cv2.imwrite(f"{save_dir}/HEF_{img}", hef_img)
        print("finish applying HEF")
    else:
        print("please select image enhancement algorithm between 'CLAHE', 'UM', or 'HEF'")