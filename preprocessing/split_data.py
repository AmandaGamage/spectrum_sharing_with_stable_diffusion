import os
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset
import torch
import csv

'''class DatasetMapper:
    def __init__(self, dataset_path, labels_file_path):
        self.dataset_path = dataset_path
        self.labels_file_path = labels_file_path
        self.class_to_idx = {}
        self.file_to_class = {}

        self.create_mappings()

    def read_labels_from_file(self):
        with open(self.labels_file_path, 'r') as file:
            labels = [line.split() for line in file.read().splitlines()]
        return labels

    def create_mappings(self):
        labels_info = self.read_labels_from_file()

        for file_name, class_name in labels_info:
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.class_to_idx)

            file_path = os.path.join(self.dataset_path, file_name)
            if os.path.isfile(file_path):
                self.file_to_class[file_name] = class_name'''
                
class DatasetMapper(DatasetFolder):
    def __init__(self, root, labels_file_path, transform=None):
        super(DatasetMapper, self).__init__(root, loader=self.loader, extensions='.jpg', transform=transform)
        self.labels_file_path = labels_file_path
        self.class_to_idx = {}
        self.file_to_class = {}
        self.create_mappings()

    def loader(self, path):
        # Load PNG image using Pillow (PIL)
        with open(path, 'rb') as img_file:
            img = Image.open(img_file).convert('RGB')
        return img

    def read_labels_from_file(self):
        with open(self.labels_file_path, 'r') as file:
            labels = [line.split() for line in file.read().splitlines()]
        return labels

    def create_mappings(self):
        labels_info = self.read_labels_from_file()

        for file_name, class_name in labels_info:
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.class_to_idx)

            file_path = os.path.join(self.root, file_name)
            if os.path.isfile(file_path):
                self.file_to_class[file_name] = class_name

class SplitData:
    def read_labels_from_file(labels_file_path):
        with open(labels_file_path, 'r') as file:
            labels = [line.split() for line in file.read().splitlines()]
        return labels

# Function to split the dataset into train and test sets
    def split_dataset(images_path, labels_file_path, test_size=0.2, random_seed=42):
        labels_info = SplitData.read_labels_from_file(labels_file_path)

    # Extracting file names and corresponding labels
        file_names, labels = zip(*labels_info)

    # Splitting into train and test sets
        train_file_names, test_file_names, train_labels, test_labels = train_test_split(
            file_names, labels, test_size=test_size, random_state=random_seed
        )

        return (train_file_names, train_labels), (test_file_names, test_labels)

class Preprocess():
    def preprocess(image):
        preprocessing_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize the image to a fixed size
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),           # Convert the image to a PyTorch tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the pixel values
        ])
        preprocessed_image = preprocessing_transform(image)

        return preprocessed_image
    
class Spectrogram(torch.utils.data.Dataset):

    def __init__(self, root_dir, csv_root,transform=None, target_transform=None):
        # Load the dataset files (images and labels) from root_dir
        self.root_dir = root_dir
        self.csv_root = csv_root
        self.transform = transform
        self.target_transform = target_transform
        self.images, self.labels = self.load_dataset()
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
    
    def load_dataset(self):
        images = []
        class_indices = []

        class_index_dict = {}
        with open(os.path.join(self.root_dir, self.csv_root), "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                class_index_dict[row['classname']] = int(row['classidx'])

        with open(os.path.join(self.root_dir, "labels.txt"), "r") as f:
            for line in f:
                label ,image_path = line.strip().split()
                image = Image.open(os.path.join(self.root_dir, image_path)).convert("RGB")
                images.append(image)
                class_indices.append(class_index_dict[label])  # Append class index instead of label

        return images, class_indices
        
         
         
              
# Example usage dataset mapper:
#dataset_path = "E:\\Msc\\Lab\\sd\\spectrogram_numpy\\spectrogram\\data_dir"
#labels_file_path = "E:\\Msc\\Lab\\sd\\spectrogram_numpy\\spectrogram\\labels_1.txt"
#mapper = DatasetMapper(dataset_path, labels_file_path)

# Access class index for a class name
#class_name = "empty"
#class_idx = mapper.class_to_idx.get(class_name, None)
#print(f"Class index for '{class_name}': {class_idx}")

# Access class label for a file name
#file_name = "chirp_image_27.png"
#file_class = mapper.file_to_class.get(file_name, None)
#print(f"Class label for '{file_name}': {file_class}")

########################################################################################################
# Example usage split data:
#dataset_path = "E:\\Msc\\Lab\\sd\\spectrogram_numpy\\spectrogram\\data_dir"
#labels_file_path = "E:\\Msc\\Lab\\sd\\spectrogram_numpy\\spectrogram\\labels_1.txt"

# Specify the desired test size and random seed
#test_size = 0.2
#random_seed = 42

# Split the dataset
#(train_files, train_labels), (test_files, test_labels) = SplitData.split_dataset(dataset_path, labels_file_path, test_size, random_seed)

# Display the number of samples in the train and test sets
#print(f"Number of samples in the training set: {len(train_files)}")
#print(f"Number of samples in the test set: {len(test_files)}")