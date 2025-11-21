import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ---------- CONFIG ----------
MODEL_PATH = "voter_id_model.h5"
IMG_SIZE = (224, 224)

# ---------- LOAD MODEL ONCE ----------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img)
    # same preprocessing used in training (MobileNetV2)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Ghana Voter ID Detector", page_icon="🪪")

st.title("🇬🇭 Ghana Voter ID Detector")
st.write(
    "Upload any image and the model will predict whether it is a "
    "**Ghana voter ID card** or **not**."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Check image"):
        x = preprocess_image(image)
        prob = float(model.predict(x)[0][0])  # 0–1

        st.write(f"Probability it is a **voter ID card**: `{prob:.2%}`")

        if prob >= 0.5:
            st.success("✅ Model prediction: **VOTER ID**")
        else:
            st.error("❌ Model prediction: **NOT A VOTER ID**")
