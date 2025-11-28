import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# 1. Page Configuration
st.set_page_config(page_title="Ghana Voter ID Detector", page_icon="🇬🇭")

# 2. Model Loading Function
@st.cache_resource
def load_learner():
    # Check if file exists relative to this script
    model_filename = 'voter_id_model.keras'
    
    if not os.path.exists(model_filename):
        # Fallback check for .h5 if you named it differently
        if os.path.exists('voter_id_model.h5'):
            model_filename = 'voter_id_model.h5'
        else:
            raise FileNotFoundError(f"Model file '{model_filename}' not found. Did you upload it to GitHub?")

    # Load the model
    # Note: We assume the model includes the preprocessing layer (rescaling) internally.
    model = tf.keras.models.load_model(model_filename)
    return model

# 3. App Title and Header
st.title("🇬🇭 Ghana Voter ID Identifier")
st.write("Upload an image of an ID card to verify if it is a valid Ghana Voters ID.")

# 4. Load Model
try:
    with st.spinner("Loading AI Model..."):
        model = load_learner()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("💡 **Fix:** Make sure you uploaded 'voter_id_model.keras' to your GitHub repository.")
    st.stop()

# 5. File Uploader
uploaded_file = st.file_uploader("Choose an ID card image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # --- A. Display Image ---
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded ID Card', use_column_width=True)

    # --- B. Preprocess Image ---
    # Resize to the exact size the model expects (224x224)
    img_resized = image.resize((224, 224))
    
    # Convert to numpy array
    img_array = np.array(img_resized)
    
    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # --- C. Make Prediction ---
    with st.spinner("Analyzing..."):
        prediction = model.predict(img_array)
        probability = float(prediction[0][0])
        confidence_score = probability * 100

    # --- D. Strict Logic (The Fix) ---
    st.write("---")
    st.subheader("Analysis Result:")

    # SETTING THE STRICT THRESHOLD
    # 0.85 means the model must be 85% sure. 
    # If it is 80% sure, we still say NO to prevent false positives.
    HIGH_THRESHOLD = 0.85

    if probability > HIGH_THRESHOLD:
        # SUCCESS CASE
        st.success(f"✅ **VERIFIED: This is a Ghana Voter ID**")
        st.progress(int(confidence_score))
        st.write(f"Confidence: {confidence_score:.2f}% (The model is very sure)")
    
    else:
        # REJECTION CASE (Covers both 'Other' and 'Unsure')
        st.error(f"❌ **REJECTED: NOT a Valid Voter ID**")
        st.progress(int(confidence_score))
        
        # Give a reason based on the score
        if probability > 0.5:
            st.warning(f"⚠️ **Reason:** The model sees some resemblance ({confidence_score:.2f}%), but it is not clear enough to be verified. Please upload a clearer photo.")
        else:
            st.write(f"**Reason:** The system does not recognize this as a Voter ID (Confidence: {confidence_score:.2f}%).")