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

# Function to move images to train/test folder structures
def organize_dataset(file_list, split_dir):
    with open(file_list, 'r') as f:
        image_paths = f.read().splitlines()

    for path in tqdm(image_paths, desc=f"Copying to {split_dir}"):
        class_name = path.split('/')[0] # Taking down class names
        img_name = path.split('/')[1] + '.jpg' # image names of respective class
        src_path = os.path.join(image_root, class_name, img_name) # The original class name with its image
        dst_dir = os.path.join(split_dir, class_name)
        os.makedirs(dst_dir, exist_ok=True) # train and test directories created with class folders
        dst_path = os.path.join(dst_dir, img_name)
        shutil.copy(src_path, dst_path) # The original class name with its images copied to destination folder

# Organize train and test datasets
organize_dataset(train_list_file, train_dir)
organize_dataset(test_list_file, test_dir)

print("Dataset organized into train/ and test/ folders.")
