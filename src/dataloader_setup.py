
'''
This module contains the function to load images using `ImageFolder` and create train/test dataloaders.

The `create_dataloaders` function sets up the necessary transformations, loads the images into datasets, and returns dataloaders for training and testing.

The function is flexible and allows you to:
- Define custom transformations for training and testing data.
- Specify batch sizes and the number of workers for data loading.

Modules Required:
- torch
- torchvision
- os

Function:
    create_dataloaders(train_dir, test_dir, train_transform, test_transform, BATCH_SIZE, NUM_WORKERS)
'''

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    train_transform: transforms.Compose,
    test_transform: transforms.Compose,
    BATCH_SIZE: int,
    NUM_WORKERS: int
):
    """
    Create dataloaders for training and testing datasets using ImageFolder.

    This function accepts the directories for training and testing data, applies the necessary transformations,
    and returns DataLoaders for each dataset. It is designed to work with any dataset structured with class 
    subdirectories under the specified directories.

    Args:
        train_dir (str): The directory containing the training data.
        test_dir (str): The directory containing the testing data.
        train_transform (transforms.Compose): The transformations to apply to the training data.
        test_transform (transforms.Compose): The transformations to apply to the testing data.
        BATCH_SIZE (int): The batch size for the dataloaders.
        NUM_WORKERS (int): The number of workers for data loading.

    Returns:
        tuple: 
            - train_dataloader (DataLoader): DataLoader for training data.
            - test_dataloader (DataLoader): DataLoader for testing data.
            - class_names (list): List of class names (labels).
    """

    # Load training and testing data using ImageFolder
    train_data = ImageFolder(root=train_dir, transform=train_transform)
    test_data = ImageFolder(root=test_dir, transform=test_transform)

    # Get class names (folder names) from the training dataset
    class_names = train_data.classes

    # Create the training and testing dataloaders
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS,
                                  pin_memory=True)
    
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False,
                                 num_workers=NUM_WORKERS,
                                 pin_memory=True)
  
    return train_dataloader, test_dataloader, class_names
