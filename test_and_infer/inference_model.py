import os
import re
import shutil
import random

import joblib
import pandas as pd
import numpy as np
from numpy.random import random, permutation
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils, models, transforms, datasets
from tqdm import tqdm
from tqdm import tqdm_notebook
from sklearn.model_selection import KFold
import itertools

from test_model import PredictTestSet
from data_preprocessing.data_preprocessing import PreprocessData


class ShowPred:
    def __init__(self, model_path):
        self.model_path = model_path
        pred = PredictTestSet(self.model_path)
        self.predictions, self.classes = pred.predict()
        pre = PreprocessData()
        self.dsets = pre.resize_img_and_create_dataset()

    def imshow(self, inp, title=None):
        #   Imshow for Tensor.
        inp = inp.numpy().transpose((1, 2, 0))
        plt.imshow(inp)
        if title is not None:
            plt.title(title)

        plt.show()

    def show_test_images(self, n_images, classified_type="all"):
        """
        Choose randomly n_images to show with their prediction and true label.
        Also allow to show only correctly classified images or only incorrectly classified images
        """

        if classified_type == "correct":
            all_index = [
                i
                for i in range(len(self.predictions))
                if self.predictions[i] == self.classes[i]
            ]
            idx = permutation(all_index)[:n_images]

        elif classified_type == "incorrect":
            all_index = [
                i
                for i in range(len(self.predictions))
                if self.predictions[i] != self.classes[i]
            ]
            idx = permutation(all_index)[:n_images]

        elif classified_type == "all":
            all_index = range(len(self.predictions))

        idx = permutation(all_index)[:n_images]

        loader = torch.utils.data.DataLoader(
            [self.dsets["test"][x] for x in idx], batch_size=n_images, shuffle=True
        )

        for data in loader:
            inputs, _ = data

        preds = [self.predictions[i] for i in idx]
        labels = [self.classes[i] for i in idx]

        out = utils.make_grid(inputs)

        self.imshow(
            out,
            title=f"Labels: {['yes' if x == 1 else 'no' for x in labels]}, predictions:{['yes' if x == 1 else 'no' for x in preds]}",
        )


if __name__ == "__main__":
    show = ShowPred("trained_best_created_model.joblib")
    show.show_test_images(4, "incorrect")
