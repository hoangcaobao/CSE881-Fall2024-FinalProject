import torch
from sentence_transformers import SentenceTransformer
from torch import nn



class CLIPClassified(nn.Module):
    """
    CILP classification model
    """
    def __init__(self, device, hidden_dim=512, output_dim=4):
        super(CLIPClassified, self).__init__()

        # CLIP image encoder
        self.fc = nn.Linear(hidden_dim, output_dim).to(device)

        self.softmax = nn.Softmax(dim=-1)

        self.CLIP_model = SentenceTransformer('clip-ViT-B-32', device=device)

        self.device = device

    def forward(self, image):
        images_embedding = self.pre_processing(image)
        logits = self.fc(images_embedding)
        return self.softmax(logits)

    def pre_processing(self, image):
        return torch.Tensor(self.CLIP_model.encode(image)).to(self.device)
