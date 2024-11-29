from dataset import *
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AdamW
from PIL import Image
import argparse
from sentence_transformers import SentenceTransformer
from model.CLIPClassification import CLIPClassified
from tqdm import tqdm


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=1, required=False)
    parser.add_argument("--seed", type=int, default=42, required=False)
    parser.add_argument("--epoch", type=int, default=60, required=False)
    parser.add_argument("--train_set", type=str, default="google", required=False)  # google, kaggle, all
    args = parser.parse_args()

    return args


def evaluation(CLIPmodel, model, dataset):
    """
    Validation function for CLIP
    :param CLIPmodel: CLIP model
    :param model: classification model
    :param dataset: evaluation dataset
    :return: accuracy
    """
    model.eval()
    with torch.no_grad():
        total = len(dataset)
        acc = 0
        for img, label in tqdm(dataset, desc="Validation"):
            image_PLI = Image.open(img)
            image_embedding = torch.Tensor(CLIPmodel.encode(image_PLI)).to(model.device)
            label = label

            pred = model(image_embedding).argmax().item()

            acc += 1 if pred == label else 0

        return acc * 100 / total


def train(CLIP, train_dataloader, validation_data, lr_set, device="cpu"):
    """
    CLIP classification training and select the best model based on validation data
    :param CLIP: CLIP image encoder model
    :param train_dataloader: train dataloader
    :param validation_data: validation data
    :param lr_set: learning rate set (hyperparameters)
    :param device: device
    :return: best_acc, best_epoch, best_lr
    """
    best_acc = 0
    best_epoch = None
    best_lr = None

    for lr in lr_set:
        model = CLIPClassified().to(device)
        model.device = device
        # Optimizer
        optimizer = AdamW(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(args.epoch):  # Adjust the number of epochs as needed
            model.train()
            for images, labels in tqdm(train_dataloader, desc="Training Epoch {}".format(epoch + 1)):
                image_PLI = [Image.open(img) for img in images]
                image_embedding = torch.Tensor(CLIP.encode(image_PLI)).to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(image_embedding)

                # Clear gradient
                optimizer.zero_grad()

                # Compute the loss
                loss = loss_fn(outputs, labels)

                # Update gradient
                loss.backward()
                optimizer.step()

            if epoch % 10 != 0:
                continue
            acc_val = evaluation(CLIP, model, validation_data)

            if acc_val > best_acc:
                best_acc = acc_val
                torch.save(model, f"weights/CLIP_classifier_{args.train_set}_train.pt")
                best_epoch = epoch
                best_lr = lr

    return best_acc, best_epoch, best_lr


def main(args):
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    batch_size = 32

    if args.train_set == "google":
        dataset = RoadSignDataset("dataset/metadata_train_google.csv", return_image_id=True)
    elif args.train_set == "kaggle":
        dataset = RoadSignDataset("dataset/metadata_train_kaggle.csv", return_image_id=True)
    else:
        dataset = RoadSignDataset("dataset/metadata_train.csv", return_image_id=True)

    portion_train = int(0.9 * len(dataset))

    validation_data = Subset(dataset, range(portion_train, len(dataset)))
    dataset = Subset(dataset, range(portion_train))

    dataloader = DataLoader(dataset, batch_size=batch_size)
    # Loading the encoder model
    CLIPmodel = SentenceTransformer('clip-ViT-B-32', device=device)

    best_acc, best_epoch, best_lr = train(CLIPmodel, dataloader, validation_data, lr_set=[1e-6, 4e-6, 8e-6], device="cpu")

    with open("report.txt", 'a') as report_file:
        print("Best ACC {:.2f}% trained with {}".format(best_acc, args.train_set), file=report_file)
        print("Number of epoch: {}".format(best_epoch), file=report_file)
        print("Learning rate: {}".format(best_lr), file=report_file)


if __name__ == "__main__":
    args = parse_argument()
    main(args)
