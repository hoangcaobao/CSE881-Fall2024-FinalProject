import os
import cv2
import xml.etree.ElementTree as ET
import copy
import pandas as pd


def resize(img, size=(256, 256)):
    return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)


def add_images_from_scraped_images(scraped_path, dataset_path):
    image_metadata = {}

    # Checking the target dataset path
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    for img_dir in os.listdir(scraped_path):
        label = img_dir.lower()
        label_directory = os.path.join(scraped_path, img_dir)
        for img_path in os.listdir(label_directory):
            ids = len(image_metadata)

            image_name = f"sign{ids:04d}"
            image_metadata[image_name] = {}

            image_metadata[image_name]["id"] = image_name + ".png"
            image_metadata[image_name]["label"] = label
            image_metadata[image_name]["data_source"] = "https://images.google.com"

            img = cv2.imread(os.path.join(label_directory, img_path))
            img = resize(img)
            cv2.imwrite(f"{dataset_path}/{image_name}.png", img)

    return image_metadata


def merge_kaggle_images(kaggle_path, dataset_path, image_metadata):
    label_convert = {"speedlimit": "speedlimit", "stop": "stop"}

    if not os.path.exists(kaggle_path):
        return

    annotations_paths = os.path.join(kaggle_path, "annotations")
    for annotation in os.listdir(annotations_paths):

        tree = ET.parse(os.path.join(annotations_paths, annotation))
        root = tree.getroot()
        img_name = root.find("filename").text

        image_folder = os.path.join(kaggle_path, root.find("folder").text)

        # print(img_name)
        all_objects = root.findall("object")
        if len(all_objects) > 1:
            continue

        label = all_objects[0].find("name").text
        if label in label_convert:
            label = label_convert[label]
        ids = len(image_metadata)
        new_img_name = f"sign{ids:04d}"
        image_metadata[new_img_name] = {}

        image_metadata[new_img_name]["id"] = new_img_name + ".png"
        image_metadata[new_img_name]["label"] = label
        image_metadata[new_img_name]["data_source"] \
            = "https://www.kaggle.com/datasets/andrewmvd/road-sign-detection/data"

        img = cv2.imread(os.path.join(image_folder, img_name))
        img = resize(img)
        cv2.imwrite(f"{dataset_directory}/{new_img_name}.png", img)

    return metadata


if __name__ == '__main__':
    # creating meta data from scraped image and kaggle dataset
    scraped_images_directories = "../images"
    kaggle_path = "../kaggle"

    dataset_directory = "../dataset/"
    dataset_img_dir = os.path.join(dataset_directory, "images")

    metadata = add_images_from_scraped_images(scraped_images_directories, dataset_directory)
    metadata = merge_kaggle_images(kaggle_path, dataset_directory, metadata)

    # Create metadata.csv

    df = pd.DataFrame.from_dict(metadata, orient="index")
    df.to_csv(os.path.join(dataset_directory, "metadata.csv"), index=False)