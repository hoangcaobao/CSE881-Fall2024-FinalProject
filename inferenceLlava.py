import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from PIL import Image
import cv2
from dataset import RoadSignDataset
from tqdm import tqdm
from model.LlavaNext import LlavaNext
import argparse

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Llava-Next experiment")
    parser.add_argument("--cuda", default="0", type=str, help="GPU to use")
    return parser.parse_args()


if __name__ == "__main__":
    # Do not need to pre-processing the image
    args = parse_arguments()

    device = "cpu"
    if torch.cuda.is_available():
        device = f"cuda:{args.cuda}"

    test_dataset = RoadSignDataset("dataset/metadata_test.csv", return_raw_data=True)


    test_loader = DataLoader(test_dataset)

    model = LlavaNext(cuda=device)
    results = []
    
    for i in tqdm(range(len(test_dataset))):
        (image, label) = test_dataset[i]
        image_id = test_dataset.get_image_id(i)
        pred = model(image)
        results.append([image_id, label, pred])
    
    df = pd.DataFrame(results, columns=["image_id", "label", "predict"])
    df.to_csv("result_llava_next.csv",index=False)

