# Food101Classifier 🍔🍕🍜

A deep learning project to classify food images into 101 categories using the **Food101 dataset**. This repository implements a complete pipeline using **PyTorch**, including automatic data downloading, organization, efficient data loading, and transfer learning with models like **ResNet-18**, **VGG16**, and **EfficientNet-B0**.

## 📌 Project Overview

The goal of this project is to build a robust image classifier capable of distinguishing between 101 different types of food (e.g., pizza, sushi, hamburger).

Key features:

  * **Automated Data Setup**: Scripts to download and restructure the raw Food101 dataset into a standard `train/test` folder format.
  * **Transfer Learning**: Utilizes pre-trained models (ImageNet weights) finetuned for the specific 101 classes.
  * **Modular Design**: Code is split into reusable modules for data loading, model building, and training logic.
  * **Evaluation**: Tracks training/testing loss and accuracy per epoch.

## 📂 Project Structure

```text
Food101Classifier/
├── notebooks/
│   └── Food101_Project.ipynb   # Jupyter notebook for experimentation
├── src/
│   ├── dataloader_setup.py     # Creates DataLoaders with transforms
│   ├── engine.py               # Contains train_one_epoch and test_one_epoch
│   ├── model_builder.py        # Architects VGG16 and EfficientNet models
│   ├── organize_food101.py     # Downloads and organizes the dataset
│   ├── test.py                 # Script for evaluating the model
│   ├── train.py                # Main training script (default: ResNet-18)
│   └── utils.py                # Utility functions (e.g., save_model)
└── README.md
```

## 🚀 Getting Started

### Prerequisites

Make sure you have Python installed along with the following dependencies:

  * `torch`
  * `torchvision`
  * `tqdm`

You can install them via pip:

```bash
pip install torch torchvision tqdm
```

### 1\. Data Preparation

Before training, you need to download and organize the dataset. The provided script handles this automatically. It downloads the Food101 dataset and restructures it into `data/food101/train` and `data/food101/test` directories.

Run the organization script:

```bash
python src/organize_food101.py
```

  * **What this does**: It downloads the \~5GB dataset, reads the official train/test split text files, and copies images into class-specific folders (e.g., `data/food101/train/pizza/image123.jpg`).

### 2\. Training the Model

To train the model, run the `train.py` script. By default, this script uses **ResNet-18**.

```bash
python src/train.py
```

#### Configuration

You can adjust hyperparameters directly in `src/train.py`:

  * **Batch Size**: 32
  * **Epochs**: 1 (default for testing, increase this for better results)
  * **Learning Rate**: 0.001
  * **Device**: Automatically selects GPU if available.

The script applies data augmentation (RandomResizedCrop, RandomHorizontalFlip) to the training set to improve generalization.

## 🧠 Models Available

While `train.py` uses ResNet-18 by default, `src/model_builder.py` provides functions to build other architectures using transfer learning. You can modify `train.py` to import and use these:

1.  **VGG16**:
      * `build_vgg16(num_classes=101)`
      * Freezes feature layers and modifies the classifier.
2.  **EfficientNet-B0**:
      * `build_efficientnet_b0(num_classes=101)`
      * Highly efficient model architecture, also adapted for 101 classes.

## 📊 Results & Artifacts

  * **Console Output**: During training, the script prints the Train Loss, Train Accuracy, Test Loss, and Test Accuracy for every epoch.
  * **Saved Models**: The model state dictionary is saved automatically after every epoch as `model_epoch_{N}.pth`.

## 🛠 Usage Example

To modify the training to use **EfficientNet-B0** instead of ResNet-18:

1.  Open `src/train.py`.
2.  Import the builder: `from src.model_builder import build_efficientnet_b0`.
3.  Replace the model initialization:
    ```python
    # Old
    # model = models.resnet18()
    # model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))

    # New
    model = build_efficientnet_b0(num_classes=len(class_names))
    ```
