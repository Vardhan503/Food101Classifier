
"""
This module contains functions for training and testing a neural network model for one epoch.

The following functions are defined in this module:
1. train_one_epoch: Trains the model for one epoch, computes the average loss and accuracy on the training set.
2. test_one_epoch: Evaluates the model on the test dataset for one epoch, computes the average loss and accuracy.

These functions are intended to be used for iterating over epochs during model training and evaluation. They can handle the training loop and testing loop, including optimization, loss calculation, and accuracy computation.

The functions assume the model, data, and device (CPU or GPU) are appropriately set up.

Usage Example:
-------------
    # Example usage in your main training script:
    train_loss, train_acc = train_one_epoch(model, train_dataloader, optimizer, loss_fn, device)
    test_loss, test_acc = test_one_epoch(model, test_dataloader, loss_fn, device)

"""
import torch
from tqdm import tqdm

def train_one_epoch(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device):
    """
    Trains the model for one epoch.

    Args:
    - model (torch.nn.Module): The neural network model to train.
    - train_dataloader (torch.utils.data.DataLoader): The DataLoader for the training dataset.
    - optimizer (torch.optim.Optimizer): The optimizer used for training.
    - loss_fn (torch.nn.Module): The loss function used for training.
    - device (torch.device): The device (CPU/GPU) on which training happens.

    Returns:
    - tuple: (train_loss (float), train_acc (float))
      - train_loss: The average training loss for the epoch.
      - train_acc: The training accuracy as a percentage (0-100%).
    """
    model.train()
    train_loss = 0
    train_correct = 0
    total_train = 0

    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)  # Move data to device
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Calculate accuracy by comparing predictions and true labels
        preds = torch.argmax(y_pred, dim=1)
        train_correct += (preds == y).sum().item()
        total_train += y.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate average loss and accuracy
    train_loss /= len(train_dataloader)
    train_acc = 100 * train_correct / total_train  # Accuracy as percentage

    return train_loss, train_acc


def test_one_epoch(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module, 
    device: torch.device):
    """
    Evaluates the model on the test dataset for one epoch.

    Args:
    - model (torch.nn.Module): The trained model to evaluate.
    - test_dataloader (torch.utils.data.DataLoader): The DataLoader for the test dataset.
    - loss_fn (torch.nn.Module): The loss function used for evaluation.
    - device (torch.device): The device (CPU/GPU) on which evaluation happens.

    Returns:
    - tuple: (test_loss (float), test_acc (float))
      - test_loss: The average test loss for the epoch.
      - test_acc: The test accuracy as a percentage (0-100%).
    """
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    test_correct = 0
    total_test = 0

    with torch.no_grad():  # No gradient computation during testing
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)  # Move data to device
            test_pred = model(X)
            test_loss += loss_fn(test_pred, y).item()

            # Calculate accuracy
            preds = torch.argmax(test_pred, dim=1)
            test_correct += (preds == y).sum().item()
            total_test += y.size(0)

    # Calculate average loss and accuracy
    test_loss /= len(test_dataloader)
    test_acc = 100 * test_correct / total_test  # Accuracy as percentage

    return test_loss, test_acc
