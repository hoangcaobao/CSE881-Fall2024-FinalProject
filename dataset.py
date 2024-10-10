import torch
import torch.nn as nn
import pandas as pd
import torch.utils
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
import torchvision
import torchvision.transforms as transforms
import math


class RoadSignDataset(Dataset):
    def __init__(self, metadata_path, index_label=None, random_flip=False, random_crop=False,  return_raw_data=False):
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
        self.return_raw_data = return_raw_data

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_file_path = "dataset/images/" + self.metadata['id'][idx]
        img = Image.open(image_file_path)

        if self.return_raw_data:
            return img, self.metadata['label'][idx]
            
        label = self.index_label[self.metadata['label'][idx]]

        if self.image_prep is not None:
            img = self.image_prep(img)
        return img, label
    
    def get_image_id(self, idx):
        return self.metadata['id'][idx]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return map(self.__getitem__, range(self.__len__()))
        
        len_data = len(self)
        per_worker = int(math.ceil((len_data) / float(worker_info.num_workers)))
        worker_id = worker_info.id
        iter_start = worker_id * per_worker
        iter_end = min(iter_start + per_worker, len_data)
        return map(self.__getitem__, range(iter_start, iter_end))