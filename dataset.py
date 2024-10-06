import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
import torchvision
import torchvision.transforms as transforms


class RoadSignDataset(Dataset):
    def __init__(self, metadata_path, index_label=None, random_flip=False, random_crop=False):
        self.metadata = pd.read_csv(metadata_path)

        if index_label is not None:
            self.index_label = index_label
        else:
            index_label = {label: 0 for label in self.metadata["label"]}
            index_label = {label: i for i, label in enumerate(sorted(index_label.keys()))}
            self.index_label = index_label

        initial_process = []
        if random_flip:
            initial_process.append(transforms.RandomHorizontalFlip())  # Randomly flip images horizontally
        if random_crop:
            initial_process.append(transforms.RandomCrop(256, 256))  # Randomly flip images horizontally
        self.image_prep = transforms.Compose([
            transforms.ToTensor(),  # Convert image to PyTorch tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image with mean and std
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_file_path = "dataset/images/" + self.metadata['id'][idx]
        img = Image.open(image_file_path)

        label = self.index_label[self.metadata['label'][idx]]
        if self.image_prep is not None:
            img = self.image_prep(img)
        return img, label