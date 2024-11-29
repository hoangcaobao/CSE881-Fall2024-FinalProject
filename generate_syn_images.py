import torch
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
import argparse
import pandas as pd
import json
from datasets import load_dataset
from tqdm import tqdm
from torchvision import transforms

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--output_dir", type=str, default="image_gen")
    parser.add_argument("--num_repeat", type=int, default=100, help="number of repeat image per label")
    parser.add_argument("--num_inference_steps", type=int, default=80, help="number of inference steps")

    return parser.parse_args()


def main(arg):
    cuda_number = arg.cuda

    if cuda_number == -1:
        cur_device = 'cpu'
    else:
        if torch.cuda.is_available():
            cur_device = "cuda:" + str(cuda_number)
        elif torch.backends.mps.is_available():
            cur_device = "mps"
        else:
            cur_device = "cpu"

    road_sign = ["speed limit", "stop", "cross walk", "no entry", "traffic light", "yield"]

    # Set generated image pipeline
    pipeline = AutoPipelineForText2Image.from_pretrained(arg.model, torch_dtype=torch.float16).to(cur_device)
    pipeline.set_progress_bar_config(disable=True)

    for sign in road_sign:
        context = f"a {sign} sign in the road."
        sign_name = "_".join(sign.split())
        for idx_img in tqdm(range(arg.num_repeat), sign_name):
            image = pipeline(context, num_inference_steps=arg.num_inference_steps).images[0]
            image.save(arg.output_dir + "/{:}_gen_{:}.png".format(sign_name, idx_img))


if __name__ == '__main__':
    arg = parse_argument()
    main(arg)
