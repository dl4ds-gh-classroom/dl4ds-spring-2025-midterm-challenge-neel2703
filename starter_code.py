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

################################################################################
# Model Definition
################################################################################
# Part 1 -- Simple CNN model
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

################################################################################
# Define a one epoch training function
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    device = CONFIG["device"]
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)
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

################################################################################
# Define a validation function
################################################################################
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

def unfreeze_layers(model, num_layers):
    for i, (name, param) in enumerate(reversed(list(model.named_parameters()))):
        if i < num_layers:
            param.requires_grad = True

def main():
    ############################################################################
    # Configuration Dictionary (Modify as needed)
    ############################################################################
    CONFIG = {
        # Change "model" to choose the architecture:
        # "SimpleCNN" for a manually defined simple network,
        # "ResNet18" for a more sophisticated network,
        # "PretrainedResNet18" for transfer learning.
        "model": "PretrainedResNet18",
        "batch_size": 32, 
        "learning_rate": 0.001,
        "epochs": 200,  # Increase for a real training run
        "num_workers": 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",  # Make sure this directory exists
        "ood_dir": "./data/ood-test",
        "wandb_project": "neelg2703-sp25-ds542-challenge",
        "seed": 42,
    }

    import pprint
    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    ############################################################################
    # Data Transformation
    ############################################################################
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
    
    ############################################################################
    # Instantiate model based on CONFIG selection and move to device
    ############################################################################
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

    model = model.to(CONFIG["device"])

    print("\nModel summary:")
    print(f"{model}\n")

    # Optional: find optimal batch size (only run once per machine)
    SEARCH_BATCH_SIZES = False
    if SEARCH_BATCH_SIZES:
        from utils import find_optimal_batch_size
        print("Finding optimal batch size...")
        optimal_batch_size = find_optimal_batch_size(model, full_trainset, CONFIG["device"], CONFIG["num_workers"])
        CONFIG["batch_size"] = optimal_batch_size
        print(f"Using batch size: {CONFIG['batch_size']}")

    ############################################################################
    # Loss Function, Optimizer and Scheduler
    ############################################################################
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=CONFIG["learning_rate"], momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.9)

    # Initialize wandb
    # wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    # wandb.watch(model)
    model_name = CONFIG["model"]
    epochs_num = CONFIG["epochs"]

    ############################################################################
    # Training Loop
    ############################################################################
    best_val_acc = 0.0

    # for epoch in range(CONFIG["epochs"]):
        # train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        # val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        # scheduler.step()

        # Log metrics to WandB
        # wandb.log({
        #     "epoch": epoch + 1,
        #     "train_loss": train_loss,
        #     "train_acc": train_acc,
        #     "val_loss": val_loss,
        #     "val_acc": val_acc,
        #     "lr": optimizer.param_groups[0]["lr"]
        # })


    for epoch in range(CONFIG["epochs"]):
        if epoch == 75:  # After 30 epochs, unfreeze the last convolutional block
            unfreeze_layers(model, 21)  # ResNet50's last block has 33 layers
        elif epoch == 110:  # After 60 epochs, unfreeze all layers
            unfreeze_layers(model, len(list(model.parameters())))
            
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()
        # Save the best model based on validation accuracy
        if int(epoch) % 25 == 0:
            # best_val_acc = val_acc
            torch.save(model.state_dict(), f"model_ckpts/{model_name}_{epoch}.pth")
            # predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
            # print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

            # # OOD Evaluation
            # all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

            # # Create Submission File (OOD)
            # submission_df_ood = eval_ood.create_ood_df(all_predictions)
            # model_name = CONFIG["model"]
            # epochs_num = CONFIG["epochs"]
            # submission_df_ood.to_csv(f"submission_ood_{model_name}_{epoch}.csv", index=False)
            # print(f"submission_ood_{model_name}_{epoch}.csv created successfully.")
            # wandb.save("best_model.pth")
    
    # wandb.finish()

    ############################################################################
    # Evaluation -- using provided evaluation functions
    ############################################################################
    # import eval_cifar100
    # import eval_ood

    # Clean CIFAR-100 Test Set Evaluation
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # OOD Evaluation
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

    # Create Submission File (OOD)
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    model_name = CONFIG["model"]
    epochs_num = CONFIG["epochs"]
    submission_df_ood.to_csv(f"submission_ood_{model_name}_{epochs_num}.csv", index=False)
    print(f"submission_ood_{model_name}_{epochs_num}.csv created successfully.")

if __name__ == '__main__':
    main()
