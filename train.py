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
import argparse

parser = argparse.ArgumentParser(description='training code')
parser.add_argument("--model", help="Model for training", required=True, type=str)
parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', default='model_ckpts', type=str)
# parser.add_argument('--checkpoint_path', help='Resume from this checkpoint', default=None, type=str)
parser.add_argument('--bs', help='batch size', default=8, type=int)
parser.add_argument('--lr', help='learning rate', default=0.001, type=float)
parser.add_argument('--epochs', help='number of epochs to train', type=str, default=200)
parser.add_argument('--num_workers', help='number of workers in dataloader', default=0, type=int)
args = parser.parse_args()

data_dir = './data'
ood_dir = './data/ood-test'
seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "model": args.model,
    "batch_size": args.bs, 
    "learning_rate": args.lr,
    "epochs": args.epochs,  # Increase for a real training run
    "num_workers": args.num_workers,
    "device": device,
    "data_dir": data_dir,  # Make sure this directory exists
    "ood_dir": ood_dir,
    "wandb_project": "neelg2703-sp25-ds542-challenge",
    "seed": seed,
}

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


args = parser.parse_args()

def train(epoch, model, trainloader, optimizer, criterion):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

def validate(model, valloader, criterion, device):
    model.eval()  # Set to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    val_loss = running_loss / len(valloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

def unfreeze_layers(model, num_layers):
    for i, (name, param) in enumerate(reversed(list(model.named_parameters()))):
        if i < num_layers:
            param.requires_grad = True

def save_ckpt(model, optimizer, epoch, loss, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, filepath)

def load_ckpt(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def main():

    torch.manual_seed(CONFIG["seed"])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(0, shear=10),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        transforms.RandomErasing(p=0.5)
    ])

    # Validation and test transforms (no augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Load the training dataset
    full_trainset = torchvision.datasets.CIFAR100(root=CONFIG["data_dir"], train=True,
                                                  download=True, transform=transform_train)

    # Split into training and validation (80/20 split) 
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = torch.utils.data.random_split(full_trainset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)

    # Test set
    testset = torchvision.datasets.CIFAR100(root=data_dir, train=False,
                                             download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
    
    if args.model == "SimpleCNN":
        model = SimpleCNN()
    elif args.model == "ResNet18":
        # A more sophisticated network without pretrained weights
        model = torchvision.models.resnet18(num_classes=100)
    elif args.model == "PretrainedResNet18":
        # Transfer learning: start from pretrained ResNet18 and fine-tune
        model = torchvision.models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 100)
        for param in list(model.parameters())[:-2]:
            param.requires_grad = False
    elif args.model == "PretrainedResNet50":
        # Transfer learning: start from pretrained ResNet18 and fine-tune
        model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 100)
        for param in list(model.parameters())[:-2]:
            param.requires_grad = False
    elif args.model == "PretrainedWideResNet101":
        # Transfer learning: start from pretrained ResNet18 and fine-tune
        model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet101_2', pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 100)
        for param in list(model.parameters())[:-2]:
            param.requires_grad = False
    elif args.model == "PretrainedCifarResNet44":
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet44", pretrained=True)
        freeze_layers(model)
    else:
        raise ValueError("Unsupported model type selected.")

    model = model.to(device)

    print("\nModel summary:")
    print(f"{model}\n")

    # Optional: find optimal batch size (only run once per machine)
    SEARCH_BATCH_SIZES = False
    if SEARCH_BATCH_SIZES:
        from utils import find_optimal_batch_size
        print("Finding optimal batch size...")
        optimal_batch_size = find_optimal_batch_size(model, full_trainset, device, args.num_workers)
        args.bs = optimal_batch_size
        print(f"Using batch size: {args.bs}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

    # Initialize wandb
    # wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    # wandb.watch(model)
    model_name = args.model
    epochs_num = args.epochs

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        if epoch == 10:  
            unfreeze_layers(model.layer4.parameters(), len(list(model.layer4.parameters())))  
        
        elif epoch == 70:  
            unfreeze_layers(model.layer3.parameters(), len(list(model.layer3.parameters())))
        
        elif epoch == 130:  
            unfreeze_layers(model.parameters(), len(list(model.parameters())))
            
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion)
        val_loss, val_acc = validate(model, valloader, criterion, device)
        scheduler.step()
        # Save the best model based on validation accuracy
        if int(epoch) % 25 == 0:
            # best_val_acc = val_acc
            save_ckpt(model, optimizer, epoch, val_loss, f"{args.checkpoint_dir}/{model_name}_{epoch}.pth")

    # Clean CIFAR-100 Test Set Evaluation
    # predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, device)
    # print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # # OOD Evaluation
    # all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

    # # Create Submission File (OOD)
    # submission_df_ood = eval_ood.create_ood_df(all_predictions)
    # model_name = CONFIG["model"]
    # epochs_num = CONFIG["epochs"]
    # submission_df_ood.to_csv(f"submission_ood_{model_name}_{epochs_num}.csv", index=False)
    # print(f"submission_ood_{model_name}_{epochs_num}.csv created successfully.")

if __name__ == '__main__':
    main()
