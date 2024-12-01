import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import os
from dataset import RoadSignDataset
from tqdm import tqdm
from model.LlavaNext import LlavaNext
import argparse

DATASET_DIR = ""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Llava-Next experiment")
    parser.add_argument("--cuda", default="0", type=str, help="GPU to use")
    parser.add_argument("data_path", default="metadata_test", type=str, help="Test data path")
    parser.add_argument("use_BLIP", default=False, type=bool, help="Whether to use BLIP caption")
    parser.add_argument("--caption_dir", default="result_Bilp_caption_metadata_test.csv", type=str, help="Caption directory to augment")
    return parser.parse_args()


def inference_image(model, args):
    """
    Inference the answer from LLavaNext model
    :param model: LlavaNext model
    :param args: parameters
    :return:
    """
    #  Load test dataset
    test_dataset = RoadSignDataset(f"{DATASET_DIR}/{args.data_path}.csv", return_raw_data=True)

    results = []
    img_id_to_caption = {}

    if os.path.exists(args.caption_dir):
        caption_results = pd.read_csv(args.caption_dir)

        for idx, caption_result in caption_results.iterrows():
            img_id = caption_result["image_id"]
            caption = caption_result["caption"]

            img_id_to_caption[img_id] = caption

    for i in tqdm(range(len(test_dataset))):
        (image, label) = test_dataset[i]
        image_id = test_dataset.get_image_id(i)

        # Load caption if exist
        caption = caption=img_id_to_caption[image_id] if image_id in img_id_to_caption else None
        pred = model(image, caption=caption)
        results.append([image_id, label, pred])

    # Save predict result
    df = pd.DataFrame(results, columns=["image_id", "label", "predict"])
    df.to_csv(f"result_llava_next_{args.data_path}.csv", index=False)


if __name__ == "__main__":
    # Do not need to pre-processing the image
    args = parse_arguments()

    device = "cpu"
    if torch.cuda.is_available():
        cudas = args.cuda.split(",")
        if len(cudas) == 1:
            device = f"cuda:{args.cuda}"
        else:
            device = [cuda for cuda in cudas]
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device) # Set visible GPU to only

    model = LlavaNext(cuda=device)
    inference_image(model, args.data_path)# Inference all test_image from llama
