import numpy as np
import pandas as pd
import os
import torch
import argparse
from torch.utils.data import DataLoader
from dataset import RoadSignDataset
from torchvision import models
from model import SimpleCNN
import random
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

NUM_CLASSES = 4

WEIGHT_PATH = "weights/VGG.pt"
IMG_DIR = "dataset/images"
SELECT_IMG_DIR = "wrong_google_vgg"


def get_parser():
    parser = argparse.ArgumentParser()
    # gpu, mps
    parser.add_argument("--device", type=str, default="1", required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument("--train", type=str, default="all", required=False)  # google, kaggle, all
    # CNN, VGG, RESNET
    parser.add_argument("--model", type=str, default="VGG", required=False)
    parser.add_argument("--provide_incorrect", type=str, default=None, required=False)
    args = parser.parse_args()
    return args


def main(args):
    if args.device == "mps":
        device = torch.device(f"mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    if args.model == "VGG":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(4096, 4) 
    elif args.model == "RESNET":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features,4)
    else:
        model = SimpleCNN(num_classes=4)
    model = model.to(device)

    if os.path.exists(WEIGHT_PATH):
        model = torch.load(WEIGHT_PATH, weights_only=False, map_location=device)

    test_csv = f"dataset/metadata_test_google.csv"
    test_meta_data = pd.read_csv(test_csv)

    img_info = {}
    for idx, data in test_meta_data.iterrows():
        img_info[data["id"]] = data

    test_dataset = RoadSignDataset(test_csv)

    incorrect_img = []
    if not args.provide_incorrect:
        with torch.no_grad():

            for i in tqdm(range(len(test_dataset))):
                (img, label) = test_dataset[i]
                img = img.unsqueeze(0).to(device)
                pred = model(img).argmax(dim=1).cpu().numpy()[0]
                if label != pred:
                    incorrect_img.append((test_dataset.get_image_id(i), test_dataset.label_index[pred]))

        # Sample 100 images
        random.seed(0)
        print(len(incorrect_img))
        selected_img = random.sample(incorrect_img, 100)


        selected_img = pd.DataFrame(selected_img, columns=["img_id", "predict"])
    else:
        selected_img = pd.read_csv(args.provide_incorrect)
    new_data = []

    if not os.path.exists(SELECT_IMG_DIR):
        os.makedirs(SELECT_IMG_DIR)
    for i, data in selected_img.iterrows():
        img_id = data["img_id"]
        pred = data["predict"]
        img_data = img_info[img_id]

        shutil.copy(os.path.join(IMG_DIR, img_id), os.path.join(SELECT_IMG_DIR, img_id))
        new_data.append((img_data["id"], img_data["label"], pred))
    new_data_df = pd.DataFrame(new_data, columns=["id", "label", "pred"])
    new_data_df = new_data_df.sort_values(by=["id"])
    new_data_df.to_csv(os.path.join(SELECT_IMG_DIR, "metadata.csv"))


if __name__ == '__main__':
    args = get_parser()
    main(args)
