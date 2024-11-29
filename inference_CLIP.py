from transformers import CLIPProcessor, CLIPModel
from dataset import *
import torch
import argparse
from tqdm import tqdm

label_text = ["a photo of stop sign", "a photo of cross walk sign", "a photo of speed limit sign",
              "a photo of traffic light sign"]
pred_label = ["stop", "crosswalk", "speedlimit", "trafficlight"]


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=1, required=False)
    parser.add_argument("--train_set", type=str, default="metadata_test", required=False)  # google, kaggle, all
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda}")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    results = []
    dataset = RoadSignDataset(f"dataset/{args.data_path}.csv", return_raw_data=True)
    for idx, (img, label) in tqdm(enumerate(dataset)):
        # Preprocess the image
        inputs = processor(text=label_text, images=img, return_tensors="pt", padding=True).to(device)
        # Predict the similarity score of image and text
        output = model(**inputs, return_dict=True)["logits_per_image"]
        ids = dataset.get_image_id(idx)

        pred = output.argmax(dim=-1).item()
        pred = pred_label[pred]

        results.append([ids, label, pred])

    df = pd.DataFrame(results, columns=["img_id", "label", "predict"])
    df.to_csv("CLIP.csv", index=False)


if __name__ == "__main__":
    args = parse_argument()
    main(args)