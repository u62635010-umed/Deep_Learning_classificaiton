import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
import os

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Fresh vs Rotten", page_icon="🍎")
st.title("🍎 Fresh vs Rotten Image Classifier")

st.write("Upload an image and the model will predict whether it is **Fresh** or **Rotten**.")

# -----------------------------
# Load model (with download)
# -----------------------------
@st.cache_resource
def load_model():
    model_path = "model.keras"

    # Agar model local me nahi hai to download karo
    if not os.path.exists(model_path):
        url = "https://drive.google.com/open?id=1GIKAkWoimsjhjlf-crq9eJPkfXB1oswB&usp=drive_fs"
        gdown.download(url, model_path, quiet=False)

    return tf.keras.models.load_model(model_path)

model = load_model()

# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# File upload
# -----------------------------
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width="stretch")

    if st.button("Predict"):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)

        confidence = prediction[0][0]
        label = "Rotten 🔴" if confidence > 0.5 else "Fresh 🟢"

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence Score: **{confidence:.4f}**")

