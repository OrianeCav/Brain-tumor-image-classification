import os
import re
import shutil

import numpy as np
import matplotlib.pyplot as plt
import torch
from IPython.display import display, Image
import torchvision
from torchvision import models, transforms, datasets
from PIL import Image as PILImage


class PreprocessData:
    def __init__(self):
        # Get the name of the folders and the files in a list
        self.no_folder_path = "../Brain-tumor-image-classification/Data/brain_tumor/brain_tumor_dataset/no"
        self.no_file_names = os.listdir(self.no_folder_path)
        self.no_image_paths = [
            os.path.join(self.no_folder_path, file_name)
            for file_name in self.no_file_names
        ]

        self.yes_folder_path = "../Brain-tumor-image-classification/Data/brain_tumor/brain_tumor_dataset/yes"
        self.yes_file_names = os.listdir(self.yes_folder_path)
        self.yes_image_paths = [
            os.path.join(self.yes_folder_path, file_name)
            for file_name in self.yes_file_names
        ]

    def rename_images(self):
        """Rename images because the names were inconsistent for the images with a 'no' label"""
        for i, current_file_name in enumerate(self.no_file_names):
            file_extension = os.path.splitext(current_file_name)[1]
            new_name = "N" + str(i + 1) + file_extension
            if os.path.exists(os.path.join(no_folder_path, new_name)):
                new_name = "N" + str(i + 1) + "_new" + file_extension
            if file_extension == "":
                os.remove(os.path.join(no_folder_path, current_file_name))
            else:
                os.rename(
                    os.path.join(no_folder_path, current_file_name),
                    os.path.join(no_folder_path, new_name),
                )

        self.no_folder_path = "../Data/brain_tumor/brain_tumor_dataset/no"
        self.no_file_names = os.listdir(no_folder_path)
        self.no_image_paths = [
            os.path.join(no_folder_path, file_name) for file_name in no_file_names
        ]

        for current_file_name in no_file_names:
            if "_new" in current_file_name:
                new_name = current_file_name.replace("_new", "")
                os.rename(
                    os.path.join(no_folder_path, current_file_name),
                    os.path.join(no_folder_path, new_name),
                )

        # Update the image path of the 'no' images in a list
        self.no_folder_path = "../Data/brain_tumor/brain_tumor_dataset/no"
        self.no_file_names = os.listdir(no_folder_path)
        self.no_image_paths = [
            os.path.join(no_folder_path, file_name) for file_name in no_file_names
        ]

    def create_train_test_set(self):
        """Split into a train and a test set"""

        # Create a train and test directory
        os.makedirs("../Data/brain_tumor/brain_tumor_dataset/train", exist_ok=True)
        os.makedirs("../Data/brain_tumor/brain_tumor_dataset/test", exist_ok=True)
        # Create the class folder in the test and train folders
        os.makedirs("../Data/brain_tumor/brain_tumor_dataset/train/yes", exist_ok=True)
        os.makedirs("../Data/brain_tumor/brain_tumor_dataset/train/no", exist_ok=True)
        os.makedirs("../Data/brain_tumor/brain_tumor_dataset/test/yes", exist_ok=True)
        os.makedirs("../Data/brain_tumor/brain_tumor_dataset/test/no", exist_ok=True)

        # Copy 33% of the images in the test directory and the rest of them in the train directory
        for i, img in enumerate(self.no_image_paths + self.yes_image_paths):
            file_name = os.path.basename(img)
            if i % 3 == 0:
                if file_name[0] == "Y":
                    set_path = "../Data/brain_tumor/brain_tumor_dataset/test/yes"
                else:
                    set_path = "../Data/brain_tumor/brain_tumor_dataset/test/no"
            else:
                if file_name[0] == "Y":
                    set_path = "../Data/brain_tumor/brain_tumor_dataset/train/yes"
                else:
                    set_path = "../Data/brain_tumor/brain_tumor_dataset/train/no"

            shutil.copy(img, os.path.join(set_path, file_name))

    def resize_img_and_create_dataset(self):
        """Resize the images so they all have the dimensions of the smallest one"""

        # Find the smallest dimension
        min_dim = 5000
        image_sizes = {}
        for img_path in self.no_image_paths + self.yes_image_paths:
            with PILImage.open(img_path) as img:
                if img.size in image_sizes.keys():
                    image_sizes[img.size] += 1
                else:
                    image_sizes[img.size] = 1

        for h, w in image_sizes.keys():
            min_dim = min(min_dim, h, w)

        resized_images = transforms.Compose(
            [
                transforms.Resize((min_dim, min_dim)),
                transforms.ToTensor(),
            ]
        )

        # Create the data sets
        data_dir = (
            "../Brain-tumor-image-classification/Data/brain_tumor/brain_tumor_dataset/"
        )
        dsets = {
            x: datasets.ImageFolder(os.path.join(data_dir, x), transform=resized_images)
            for x in ["train", "test"]
        }

        return dsets
