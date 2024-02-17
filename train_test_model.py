import os
import re
import shutil
import random

import click
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
import itertools

from data_preprocessing.data_preprocessing import PreprocessData
from existing_models import train_efficient_net
from models_created import train_best_created_model
from test_and_infer.test_model import PredictTestSet
from test_and_infer.inference_model import ShowPred


@click.command()
@click.option(
    "--model",
    default=None,
    help="Model chosen for training/test.",
)
def main(model):
    if model == "efficient_net":
        train_efficient_net.main()

        model_path = "trained_model_efficient_net7.joblib"

    elif model == "created":
        train_best_created_model.main()

        model_path = "trained_best_created_model.joblib"

    else:
        return "This model has not been developed yet"

    pred = PredictTestSet(model_path)
    pred.show_perf_metrics()

    show = ShowPred(model_path)
    show.show_test_images(5, "all")


if __name__ == "__main__":
    main()
