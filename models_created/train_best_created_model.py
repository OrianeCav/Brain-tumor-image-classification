import os
import re
import shutil
import random

import yaml
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
from sklearn.model_selection import KFold
import itertools

from data_preprocessing.data_preprocessing import PreprocessData
from models_created.models import classifier_large


# Data loaders
def create_train_data_loader(batch_size):
    pre = PreprocessData()
    dsets = pre.resize_img_and_create_dataset()

    train_loader = torch.utils.data.DataLoader(
        dsets["train"], batch_size=batch_size, shuffle=True, num_workers=2
    )

    return train_loader


def train(model, data_loader, loss_fn, optimizer, n_epochs=1, verbose=True):
    model = model.to(device)
    model.train(True)
    loss_train = np.zeros(n_epochs)
    acc_train = np.zeros(n_epochs)
    for epoch_num in range(n_epochs):  # tqdm(range(n_epochs)):
        running_corrects = 0.0
        running_loss = 0.0
        size = 0

        for data in data_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            bs = labels.size(0)
            output = model.forward(inputs)
            loss = loss_fn(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = output.max(1, keepdim=True)[1]
            running_loss += loss
            running_corrects += pred.eq(labels.view_as(pred)).sum()
            size += bs
        epoch_loss = running_loss.item() / size
        epoch_acc = running_corrects.item() / size
        loss_train[epoch_num] = epoch_loss
        acc_train[epoch_num] = epoch_acc
        if verbose:
            print("Train - Loss: {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_acc))
    return loss_train, acc_train


if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        print("using gpu")
        device = torch.device("gpu")
    elif torch.backends.mps.is_available():
        print("using mps")
        device = torch.device("mps")

    with open("models_created/best_model_params.yml", "r") as file:
        model_params = yaml.safe_load(file)

    # Access individual parameters
    learning_rate = model_params["learning_rate"]
    batch_size = model_params["batch_size"]
    num_epochs = model_params["num_epochs"]
    conv_kernel_number = model_params["conv_kernel_number"]
    conv_kernel_size = model_params["conv_kernel_size"]
    maxpool_kernel_size = model_params["maxpool_kernel_size"]
    dropout_rate = model_params["dropout_rate"]

    conv_class = classifier_large(
        conv_kernel_number, conv_kernel_size, maxpool_kernel_size, dropout_rate
    )

    loss_fn = nn.NLLLoss()
    optimizer_cl = torch.optim.RMSprop(conv_class.parameters())
    train_loader = create_train_data_loader(batch_size)

    loss_train, acc_train = train(
        conv_class, train_loader, loss_fn, optimizer_cl, n_epochs=num_epochs
    )

    joblib.dump(conv_class, "trained_best_created_model.joblib")
