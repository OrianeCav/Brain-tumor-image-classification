import os
import re
import shutil
import random

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from tqdm import tqdm
from tqdm import tqdm_notebook
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import KFold
import itertools

from data_preprocessing.data_preprocessing import PreprocessData


class PredictTestSet:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

        self.device = "cpu"
        if torch.cuda.is_available():
            print("using gpu")
            self.device = torch.device("gpu")
        elif torch.backends.mps.is_available():
            print("using mps")
            self.device = torch.device("mps")

    # Data loaders
    def create_test_data_loader(self):
        pre = PreprocessData()
        dsets = pre.resize_img_and_create_dataset()

        test_loader = torch.utils.data.DataLoader(
            dsets["test"], batch_size=8, shuffle=False, num_workers=2
        )

        return test_loader

    def predict(self, test_loader=None):
        self.model.eval()
        predictions = []
        labels_all = []
        if test_loader is None:
            test_loader = self.create_test_data_loader()

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())
                labels_all.extend(labels.cpu().numpy())

        return labels_all, predictions

    def show_perf_metrics(self):
        predictions = self.predict()

        results = {
            "Accuracy": [accuracy_score(predictions[0], predictions[1])],
            "Precision": [precision_score(predictions[0], predictions[1])],
            "Recall": [recall_score(predictions[0], predictions[1])],
            "F1-score": [f1_score(predictions[0], predictions[1])],
        }

        results_pd = pd.DataFrame(results, index=[""])
        print(results_pd)


if __name__ == "__main__":
    pred = PredictTestSet("trained_best_created_model.joblib")
    pred.show_perf_metrics()
