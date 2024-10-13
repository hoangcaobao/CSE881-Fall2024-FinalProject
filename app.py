import streamlit as st
from PIL import Image
import random 
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np

st.title("Road Sign Detection Project")
option = st.selectbox(
    "Which Computer Vision Architectures you want to use?",
    ("VGG", "ResNet"),
)

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
label_mapping = {0: 'Cross Walk', 1: 'No Entry', 2: 'Pedestrian Crossing', 3: 'Speed Limit', 4: 'Stop', 5: 'Traffic Light', 6: 'Yield'}

if uploaded_image is not None:
    # Open the uploaded image
    image = Image.open(uploaded_image)

    # Resize
    image_np = np.array(image)
    image_resized = cv2.resize(image_np, (256, 256), interpolation=cv2.INTER_CUBIC)
    image = Image.fromarray(image_resized)

    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Load the model
    if option == "VGG":
        model = torch.load("weights/VGG.pt", map_location=torch.device("cpu"))
    elif option == "ResNet":
        model = torch.load("weights/RESNET.pt", map_location=torch.device("cpu"))
    else:
        model = torch.load("weights/VGG.pt", map_location=torch.device("cpu"))
        
    model.eval() 
    
    # Transform to Tensor
    image_prep = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image_tensor = image_prep(image).unsqueeze(0)  
    
    with torch.no_grad():  
        output = model(image_tensor)
        _, pred = torch.max(output.data, 1)
    
    # Get label from prediction
    label = label_mapping[pred.item()]
    st.write(f"Label: {label}")