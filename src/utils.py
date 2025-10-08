
"""
This module contains two functions for saving and loading PyTorch models.

1. `save_model`: Saves the model's learned parameters (weights) to a specified file.
2. `load_model`: Loads the model's parameters (weights) from a specified file.

These utility functions help in persisting and reusing trained models for further inference or fine-tuning.

Functions:
    - save_model(model, filename): Saves the model to the specified file.
    - load_model(model, filename): Loads the model from the specified file.

"""

import torch
import os

def save_model(model, filename: str):
    """
    Saves the trained model weights to the specified file.

    Args:
    - model: The trained PyTorch model.
    - filename: The path where the model weights will be saved.
    
    Returns:
    - None
    """
    try:
        torch.save(model.state_dict(), filename)
        print(f"Model saved to {filename}")
    except Exception as e:
        print(f"Error saving model: {e}")


def load_model(model, filename: str):
    """
    Loads the model weights from the specified file.

    Args:
    - model: The PyTorch model to which the weights will be loaded.
    - filename: The path where the model weights are saved.
    
    Returns:
    - model: The PyTorch model with loaded weights.
    """
    if not os.path.exists(filename):
        print(f"Error: The file {filename} does not exist.")
        return None

    try:
        model.load_state_dict(torch.load(filename))
        model.eval()  # Set the model to evaluation mode
        print(f"Model loaded from {filename}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    return model
