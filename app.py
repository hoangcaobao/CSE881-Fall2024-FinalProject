import streamlit as st
from PIL import Image
import random 

st.title("Road Sign Detection Project")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)    
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # We will add model output's later
    label = random.choice(range(5))
    st.write(f"Label: {label}")
