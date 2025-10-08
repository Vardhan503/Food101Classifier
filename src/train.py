"""
This script trains and evaluates a model on the Food101 dataset.

The script:
1. Loads and prepares the training and testing datasets using DataLoader.
2. Defines the model (ResNet-18 in this case) and modifies the final layer to match the number of classes in the Food101 dataset (101 classes).
3. Defines the optimizer (Adam) and the loss function (CrossEntropyLoss).
4. Trains the model for a specified number of epochs.
5. Evaluates the model on the test set after each epoch.
6. Saves the model's state after each epoch.

Modules:
    - torch: For model building, training, and testing.
    - tqdm: For displaying progress bars during training.
    - src.engine: For training and testing functions.
    - src.dataloader_setup: For data loading functionality.
    - src.utils: For model saving functionality.
"""

import torch
from torch.optim import Adam
from src.engine import train_one_epoch, test_one_epoch
from src.dataloader_setup import create_dataloaders
from torchvision import models, transforms
from src.utils import save_model  # Import the save_model function
from tqdm import tqdm
import os

# Hyperparameters
NUM_EPOCHS = 1  # Number of training epochs
BATCH_SIZE = 32  # Batch size for training and testing
NUM_WORKERS = os.cpu_count()  # Number of workers for data loading
LEARNING_RATE = 0.001  # Learning rate for the optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU

# Paths for train and test data
train_dir = "/content/data/food101/train"
test_dir = "/content/data/food101/test"

# Transformations for training and testing datasets
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Crop and resize the image to 224x224 randomly
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

test_transform = transforms.Compose([
    transforms.Resize(256),  # Resize the image to 256
    transforms.CenterCrop(224),  # Crop the image to 224x224
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Load data using create_dataloaders function from dataloader_setup
train_dataloader, test_dataloader, class_names = create_dataloaders(
    train_dir, test_dir, train_transform, test_transform, BATCH_SIZE, NUM_WORKERS)

# Initialize ResNet-18 model from torchvision.models
model = models.resnet18()

# Modify the final layer to fit the number of classes in Food101 (101 classes)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))

# Move the model to the chosen device (GPU or CPU)
model.to(device)

# Initialize optimizer and loss function
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)  # Adam optimizer
loss_fn = torch.nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification

# Training and testing loop
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    
    # Training for one epoch
    train_loss, train_acc = train_one_epoch(
        model, train_dataloader, optimizer, loss_fn, device)
    
    # Testing for one epoch
    test_loss, test_acc = test_one_epoch(model, test_dataloader, loss_fn, device)

    # Print the results of the current epoch
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    # Save the model after every epoch
    save_model(model, f"model_epoch_{epoch+1}.pth")  # Save using utils.py
