import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # For progress bars
import wandb
import json
import eval_cifar100
import eval_ood

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # A simple network with two convolutional layers, pooling and two fc layers.
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)  # CIFAR100 images are 32x32 so after two poolings: 32/2/2=8
        self.fc2 = nn.Linear(256, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [B, 32, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))   # [B, 64, 8, 8]
        x = x.view(x.size(0), -1)                # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if CONFIG["model"] == "SimpleCNN":
    model = SimpleCNN()
elif CONFIG["model"] == "ResNet18":
    # A more sophisticated network without pretrained weights
    model = torchvision.models.resnet18(num_classes=100)
elif CONFIG["model"] == "PretrainedResNet18":
    # Transfer learning: start from pretrained ResNet18 and fine-tune
    model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 100)
    for param in list(model.parameters())[:-2]:
        param.requires_grad = False
elif CONFIG["model"] == "PretrainedResNet50":
    # Transfer learning: start from pretrained ResNet18 and fine-tune
    model = torchvision.models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 100)
    for param in list(model.parameters())[:-2]:
        param.requires_grad = False
elif CONFIG["model"] == "PretrainedWideResNet101":
    # Transfer learning: start from pretrained ResNet18 and fine-tune
    model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet101_2', pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 100)
    for param in list(model.parameters())[:-2]:
        param.requires_grad = False
elif CONFIG["model"] = "pretrainedcifarresent44":
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet44", pretrained=True)
else:
    raise ValueError("Unsupported model type selected.")