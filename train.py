import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from PIL import Image
import cv2
from dataset import RoadSignDataset
from tqdm import tqdm
from torchvision import models
from model.SimpleCNN import SimpleCNN
import argparse
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # gpu, mps
    parser.add_argument("--device", type=str, default="gpu", required=False)
    parser.add_argument("--seed", type=int, default = 42, required=False)
    
    # CNN, VGG
    parser.add_argument("--model", type = str, default="VGG", required=False)

    args = parser.parse_args()
    train_dataset = RoadSignDataset("dataset/metadata_train.csv")
    test_dataset = RoadSignDataset("dataset/metadata_test.csv")

    num_features = len(train_dataset.index_label)

    train_loader = DataLoader(train_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    if args.device == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if args.device == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "CNN":
        model = SimpleCNN(num_classes=7).to(device)
    elif args.model == "VGG":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 7) 
    
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in tqdm(range(20)):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model, f"weights/{args.model}.pt")

    preds = []
    ground_truth = []
    with torch.no_grad():
        for images, labels in test_loader:
            model.eval()
            with torch.no_grad():
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
            ground_truth.extend(labels.tolist())
            _, pred = torch.max(outputs.data, 1)
            preds.extend(pred.tolist())

    print(accuracy_score(ground_truth, preds))


