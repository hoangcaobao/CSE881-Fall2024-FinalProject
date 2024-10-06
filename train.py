import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from PIL import Image
import cv2
from dataset import RoadSignDataset
from tqdm import tqdm
from model.SimpleCNN import SimpleCNN

if __name__ == "__main__":
    train_dataset = RoadSignDataset("dataset/metadata_train.csv")
    test_dataset = RoadSignDataset("dataset/metadata_test.csv")

    num_features = len(train_dataset.index_label)

    train_loader = DataLoader(train_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = SimpleCNN(num_classes=7).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in tqdm(range(0)):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

    preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, pred = torch.max(outputs.data, 1)
            preds.extend(pred.tolist())
    print(preds)


