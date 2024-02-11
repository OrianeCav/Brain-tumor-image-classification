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
from sklearn.model_selection import KFold
import itertools

from data_preprocessing.data_preprocessing import PreprocessData


# Data loaders
def create_train_data_loader():
    pre = PreprocessData()
    dsets = pre.resize_img_and_create_dataset()

    train_loader = torch.utils.data.DataLoader(
        dsets["train"], batch_size=8, shuffle=True, num_workers=2
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


def load_and_tune_eff_net():
    model_eff_net7 = models.efficientnet_b7(weights="DEFAULT")
    model_eff_net7 = model_eff_net7.to(device)

    for param in model_eff_net7.parameters():
        param.requires_grad = False

    model_eff_net7.classifier._modules["1"] = nn.Linear(2560, 2)
    model_eff_net7.classifier._modules["2"] = torch.nn.LogSoftmax(dim=1)

    model_eff_net7 = model_eff_net7.to(device)

    return model_eff_net7


if __name__ == "__main__":
    device = "cpu"
    if torch.cuda.is_available():
        print("using gpu")
        device = torch.device("gpu")
    elif torch.backends.mps.is_available():
        print("using mps")
        device = torch.device("mps")

    model_eff_net7 = load_and_tune_eff_net()

    loss_fn = nn.NLLLoss()
    learning_rate = 1e-3
    optimizer_cl = torch.optim.RMSprop(model_eff_net7.parameters())
    train_loader = create_train_data_loader()

    loss_train, acc_train = train(
        model_eff_net7, train_loader, loss_fn, optimizer_cl, n_epochs=30
    )

    joblib.dump(model_eff_net7, "trained_model_efficient_net7.joblib")
