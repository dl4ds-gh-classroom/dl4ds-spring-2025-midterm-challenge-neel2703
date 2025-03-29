import eval_cifar100
import eval_ood
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
from starter_code import SimpleCNN

CONFIG = {
    # Change "model" to choose the architecture:
    # "SimpleCNN" for a manually defined simple network,
    # "ResNet18" for a more sophisticated network,
    # "PretrainedResNet18" for transfer learning.
    "model": "PretrainedResNet18",
    "batch_size": 64, 
    "learning_rate": 0.1,
    "epochs": 175,  # Increase for a real training run
    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # "device": "cpu",
    "data_dir": "./data",  # Make sure this directory exists
    "ood_dir": "./data/ood-test",
    "wandb_project": "neelg2703-sp25-ds542-challenge",
    "seed": 42,
}

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

# Validation and test transforms (no augmentation)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

############################################################################
# Data Loading
############################################################################
# Load the training dataset
full_trainset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True,
                                                download=True, transform=transform_train)

# Split into training and validation (80/20 split)
train_size = int(0.8 * len(full_trainset))
val_size = len(full_trainset) - train_size
trainset, valset = torch.utils.data.random_split(full_trainset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
valloader = torch.utils.data.DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

# Test set
testset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=False,
                                            download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

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
    # for param in list(model.parameters())[:-2]:
    #     param.requires_grad = False
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

model = model.to(CONFIG["device"])

state_dict = torch.load('model_ckpts/PretrainedResNet18_175.pth', map_location=torch.device(CONFIG["device"]))
model.load_state_dict(state_dict, strict=False)
model.eval()

model_name = CONFIG["model"]
epochs_num = CONFIG["epochs"]

# Clean CIFAR-100 Test Set Evaluation
predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, torch.device(CONFIG["device"]), 'model_ckpts/PretrainedResNet18_175.pth')
print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

# OOD Evaluation
all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

# Create Submission File (OOD)
submission_df_ood = eval_ood.create_ood_df(all_predictions)
submission_df_ood.to_csv(f"submission_ood_{model_name}_{epochs_num}.csv", index=False)
print(f"submission_ood_{model_name}_{epochs_num}.csv created successfully.") 