"""
This script evaluates a trained model on the Food101 test dataset.

The script:
1. Loads a trained model from the specified checkpoint file.
2. Evaluates the model on the test dataset.
3. Prints the test loss and accuracy.

Modules:
    - torch: For model loading and evaluation.
    - tqdm: For displaying progress bars during evaluation.
    - src.engine: For the evaluation functions.
    - src.dataloader_setup: For data loading functionality.
    - src.utils: For model loading functionality.
"""

import torch
from src.engine import test_one_epoch
from src.dataloader_setup import create_dataloaders
from src.utils import load_model  # Import the load_model function
from torchvision import transforms
import os

# Hyperparameters
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths for train and test data
train_dir = "/content/data/food101/train"
test_dir = "/content/data/food101/test"

# Transformations for testing datasets
test_transform = transforms.Compose([
    transforms.Resize(256),  # Resize the image to 256
    transforms.CenterCrop(224),  # Crop the image to 224x224
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Load test data
_, test_dataloader, class_names = create_dataloaders(
    train_dir, test_dir, None, test_transform, BATCH_SIZE, NUM_WORKERS)

# Initialize the model (ResNet-18)
model = torch.nn.Module()  # Placeholder model, to be replaced by actual model

# Load the model weights from a checkpoint file
model = load_model(model, 'model_epoch_1.pth')  # Change the filename as needed

# Move the model to the chosen device (GPU or CPU)
model.to(device)

# Loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Evaluate the model
test_loss, test_acc = test_one_epoch(model, test_dataloader, loss_fn, device)

# Print the test results
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
