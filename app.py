import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------- CONFIG ----------
WEIGHTS_PATH = "voter_id_weights.h5"  # ensure this file is in the repo
IMG_SIZE = (224, 224)

# ---------- MODEL ARCHITECTURE (must match Colab) ----------
def build_model():
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # Binary classification

    model = tf.keras.Model(inputs, outputs)
    model.load_weights(WEIGHTS_PATH)  # Load weights
    return model

@st.cache_resource
def load_model():
    return build_model()

model = load_model()

# ---------- PREPROCESS IMAGE ----------
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Ghana Voter ID Detector", page_icon="🪪")
st.title("🇬🇭 Ghana Voter ID Detector")

st.write("Upload any image and the model will check if it is a **Ghana Voter ID** or not.")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Check Image"):
        x = preprocess_image(image)
        prob = float(model.predict(x)[0][0])  # between 0 and 1

        st.write(f"**Probability of being a Voter ID:** `{prob:.2%}`")

        if prob >= 0.5:
            st.success("🟢 Prediction: **VOTER ID**")
        else:
            st.error("🔴 Prediction: **NOT A VOTER ID**")
