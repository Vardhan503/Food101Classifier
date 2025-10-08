
"""
This module contains functions for building two models using transfer learning:
1. VGG16
2. EfficientNet (EfficientNet-B0)

Both models are modified to adapt to the number of classes in the Food101 dataset (101 classes).

Models are built using pre-trained weights and only the final classification layer is modified.

Modules:
    - torch: For model building and training.
    - torchvision.models: For loading pre-trained models.
    - torch.nn: For defining the custom layers.
"""

import torch
import torchvision
import torch.nn as nn

def build_vgg16(num_classes: int):
    """
    Builds the VGG16 model with transfer learning.

    Args:
    - num_classes (int): The number of output classes in the dataset.

    Returns:
    - model (torch.nn.Module): The VGG16 model with a modified final layer.
    """
    # Load the pre-trained VGG16 model
    model = torchvision.models.vgg16(pretrained=True)

    # Freeze all layers except the final fully connected layer
    for param in model.parameters():
        param.requires_grad = False

    # Modify the final fully connected layer to fit the number of classes
    model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=num_classes)

    return model


def build_efficientnet_b0(num_classes: int):
    """
    Builds the EfficientNet-B0 model with transfer learning.

    Args:
    - num_classes (int): The number of output classes in the dataset.

    Returns:
    - model (torch.nn.Module): The EfficientNet-B0 model with a modified final layer.
    """
    # Load the pre-trained EfficientNet-B0 model
    model = torchvision.models.efficientnet_b0(pretrained=True)

    # Freeze all layers except the final classification layer
    for param in model.parameters():
        param.requires_grad = False

    # Modify the final fully connected layer to fit the number of classes
    model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)

    return model
