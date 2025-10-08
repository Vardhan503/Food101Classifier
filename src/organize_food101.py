"""
This module contains the code to download and organize the Food101 dataset into a specific directory structure.

The dataset is downloaded using the `torchvision.datasets.Food101` class, which is then restructured into the following format:

data/food101/
    ├── food101/
    │   ├── images/
    │   │   ├── class_name/
    │   │   │   ├── image1.jpg
    │   │   │   └── image2.jpg
    │   │   └── ...
    │   ├── meta/
    │   │   ├── classes.txt
    │   │   ├── labels.txt
    │   │   ├── test.json
    │   │   ├── test.txt
    │   │   ├── train.json
    │   │   └── train.txt
    ├── test/
    └── train/

The images will be organized into 'train' and 'test' subdirectories for each class. 
Each class will contain the images corresponding to that class from the original dataset.
"""

import os
import shutil
from torchvision.datasets import Food101
from tqdm import tqdm

# Paths
base_dir = "data/food101"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Download the dataset using torchvision
dataset = Food101(root=base_dir, download=True)

# The raw images are stored in: data/food101/food-101/images/{class_name}/{image_name}.jpg
image_root = os.path.join(base_dir, 'food-101', 'images')

# The training and testing image lists are stored in these .txt files
meta_root = os.path.join(base_dir, 'food-101', 'meta')
train_list_file = os.path.join(meta_root, 'train.txt')
test_list_file = os.path.join(meta_root, 'test.txt')

def organize_dataset(file_list, split_dir):
    """
    Organizes the Food101 dataset into train and test directories for each class.

    This function reads a list of image paths from a text file (`file_list`), 
    and copies the corresponding images into the appropriate class folder 
    inside the `split_dir` (either 'train' or 'test'). The images are copied
    from the raw source directory to the destination directory, preserving the 
    class structure.

    Args:
        file_list (str): Path to the text file containing a list of image paths.
        split_dir (str): Path to the directory where the images should be organized ('train' or 'test').

    Returns:
        None
    """
    with open(file_list, 'r') as f:
        image_paths = f.read().splitlines()

    for path in tqdm(image_paths, desc=f"Copying to {split_dir}"):
        # Extract class name and image name from the file path
        class_name = path.split('/')[0]
        img_name = path.split('/')[1] + '.jpg'  # Add the .jpg extension
        
        # Define source and destination paths
        src_path = os.path.join(image_root, class_name, img_name)
        dst_dir = os.path.join(split_dir, class_name)
        os.makedirs(dst_dir, exist_ok=True)  # Create class directory if it doesn't exist
        dst_path = os.path.join(dst_dir, img_name)

        # Copy the image to the corresponding destination folder
        shutil.copy(src_path, dst_path)

# Organize train and test datasets
organize_dataset(train_list_file, train_dir)
organize_dataset(test_list_file, test_dir)

print("Dataset organized into train/ and test/ folders.")

# Remove the .tar.gz file after the data is organized
tar_gz_path = os.path.join(base_dir, 'food-101.tar.gz')
if os.path.exists(tar_gz_path):
    os.remove(tar_gz_path)
    print(f"Removed the file: {tar_gz_path}")
else:
    print(f"The file {tar_gz_path} does not exist.")
