import streamlit as st
from PIL import Image
import random 
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import requests
import os

@st.cache(allow_output_mutation=True)
def download_models():
    models = ["VGG", "RESNET"]
    for model in models:
        if os.path.exists(f"{model}.pt"):
            continue
        url = f"https://storage.googleapis.com/model-weight/{model}.pt"
        response = requests.get(url)
        with open(f"{model}.pt", "wb") as f:
            f.write(response.content)

download_models()

resnet = torch.load("RESNET.pt", map_location=torch.device("cpu"))
vgg = torch.load("VGG.pt", map_location=torch.device("cpu"))

st.title("CSE 881 - Road Sign Detection Project")
st.markdown("""
    <div style="font-size:20px;">
        Authors: Bao Hoang and Tanawan Premsri
    </div>
""", unsafe_allow_html=True)

option = st.selectbox(
    "Which Computer Vision Architectures you want to use?",
    ("VGG", "ResNet"),
)

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
label_mapping = {0: 'Cross Walk', 1: 'Speed Limit', 2: 'Stop', 3: "Traffic Light"}

if uploaded_image is not None:
    # Open the uploaded image
    image = Image.open(uploaded_image)

    # Resize
    image_np = np.array(image)
    image_resized = cv2.resize(image_np, (256, 256), interpolation=cv2.INTER_CUBIC)
    image = Image.fromarray(image_resized)

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    st.image(image, caption="Uploaded Image")
        
    # Load the model
    if option == "VGG":
        model = vgg
    elif option == "ResNet":
        model = resnet

    model.eval() 
    
    # Transform to Tensor
    image_prep = transforms.Compose([
            transforms.ToTensor(),  # Convert image to PyTorch tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image with mean and std
        ])

    image_tensor = image_prep(image).unsqueeze(0)  

    with torch.no_grad():  
        output = model(image_tensor)
        _, pred = torch.max(output.data, 1)
    
    # Get label from prediction
    label = label_mapping[pred.item()]
    st.write(f"Uploaded Image Is {label} Sign")