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
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # gpu, mps
    parser.add_argument("--device", type=str, default="0", required=False)
    parser.add_argument("--seed", type=int, default = 42, required=False)
    parser.add_argument("--train", type=str, default = "all", required=False) # google, kaggle, all

    # CNN, VGG, RESNET
    parser.add_argument("--model", type = str, default="RESNET", required=False)

    args = parser.parse_args()
    
    set_seed(42)

    if args.train == "all":
        train_csv = "dataset/metadata_train.csv"
    else:
        train_csv = f"dataset/metadata_train_{args.train}.csv"

    train_dataset = RoadSignDataset(train_csv)

    test =  ["google", "kaggle", "all"]

    num_classes = len(train_dataset.index_label)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle = True)
    
    if args.device == "mps":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    if args.model == "VGG":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(4096, num_classes) 
    elif args.model == "RESNET":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = SimpleCNN(num_classes=num_classes)
    
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

    for t in test:
        
        if t == "all":
            test_csv = "dataset/metadata_test.csv"
        else:
            test_csv = f"dataset/metadata_test_{t}.csv"

        test_dataset = RoadSignDataset(test_csv)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle = False)

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

        print(f"Accuracy on {t} is {accuracy_score(ground_truth, preds)}")


