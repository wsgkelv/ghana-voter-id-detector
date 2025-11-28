import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page Config
st.set_page_config(page_title="Ghana Voter ID Detector", page_icon="🇬🇭")

@st.cache_resource
def load_model():
    # Load the model we saved in Colab
    # Ensure 'final_voter_model.keras' is in the same folder as this script
    model = tf.keras.models.load_model('final_voter_model.keras')
    return model

st.title("🇬🇭 Ghana Voter ID Identifier")
st.write("Upload an image of an ID card to check if it is a valid Voters ID.")

# Load Model
try:
    with st.spinner("Loading Model..."):
        model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Choose an ID card image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded ID Card', use_column_width=True)

    # 2. Process the image for the model
    # Resize to the exact size the model expects (224x224)
    img_resized = image.resize((224, 224))
    
    # Convert to numpy array (values are 0-255)
    img_array = np.array(img_resized)
    
    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # 3. Make Prediction
    # Note: We do NOT use preprocess_input here because it is inside the model
    prediction = model.predict(img_array)
    probability = float(prediction[0][0])

    # 4. Display Results
    st.write("---")
    st.subheader("Analysis Result:")

    # Logic: 0 = 'other', 1 = 'voter_id'
    # Threshold is 0.5
    if probability > 0.5:
        confidence = probability * 100
        st.success(f"✅ **This is a VOTER ID CARD**")
        st.progress(int(confidence))
        st.write(f"Confidence: {confidence:.2f}%")
    else:
        confidence = (1 - probability) * 100
        st.error(f"❌ **This is NOT a Voter ID (Detected as 'Other')**")
        st.progress(int(confidence))
        st.write(f"Confidence: {confidence:.2f}%")