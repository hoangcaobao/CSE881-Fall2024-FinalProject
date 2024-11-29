import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from PIL import Image
import cv2
from dataset import RoadSignDataset
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import argparse


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Caption by BLIP")
    parser.add_argument("--cuda", default="0", type=str, help="GPU to use")
    parser.add_argument("--data_path", default="metadata_test", type=str, help="Path to testing dataset")
    return parser.parse_args()


def generate_caption(image, BLIP_process, BLIP_model):
    """
    Generate caption from image
    :param image: image to generate caption
    :param BLIP_process: BLIP pre-process the input
    :param BLIP_model: BLIP model
    :return: generated caption
    """
    inputs = BLIP_process(images=[image], return_tensors="pt").to(device, torch.float16)
    generated_ids = BLIP_model.generate(**inputs)
    generated_text = BLIP_process.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return generated_text


def inference_image(BLIP, processor, data_path):
    test_dataset = RoadSignDataset(f"dataset/{data_path}.csv", return_raw_data=True)

    results = []

    for i in tqdm(range(len(test_dataset))):
        (image, label) = test_dataset[i]
        image_id = test_dataset.get_image_id(i)

        pred = generate_caption(image, processor, BLIP)
        results.append([image_id, label, pred])

    # Save the generate caption for each image into csv file
    df = pd.DataFrame(results, columns=["image_id", "label", "caption"])
    df.to_csv(f"result_Bilp_caption_{data_path}.csv", index=False)


if __name__ == "__main__":
    # Do not need to pre-processing the image
    args = parse_arguments()

    device = "cpu"
    if torch.cuda.is_available():
        device = f"cuda:{args.cuda}"

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    BLIP = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map={"": device}).to(
        device)

    inference_image(BLIP, processor, args.data_path)
