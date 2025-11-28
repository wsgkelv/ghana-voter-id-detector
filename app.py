import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Page Config
st.set_page_config(page_title="Ghana Voter ID Detector", page_icon="🇬🇭")

@st.cache_resource
def load_learner():
    # 1. Check if file exists relative to this script
    model_filename = 'voter_id_model.keras'
    
    if not os.path.exists(model_filename):
        # Fallback check for .h5 if you named it differently
        if os.path.exists('voter_id_model.h5'):
            model_filename = 'voter_id_model.h5'
        else:
            raise FileNotFoundError(f"Model file '{model_filename}' not found in the repository. Did you upload it to GitHub?")

    # 2. Load the model
    model = tf.keras.models.load_model(model_filename)
    return model

st.title("🇬🇭 Ghana Voter ID Identifier")
st.write("Upload an image of an ID card to check if it is a valid Voters ID.")

# Load Model
try:
    with st.spinner("Loading Model..."):
        model = load_learner()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("💡 **Fix:** Make sure you uploaded 'voter_id_model.keras' to your GitHub repository folder.")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Choose an ID card image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded ID Card', use_column_width=True)

    # Preprocess
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    # (Note: We assume the model has built-in preprocessing. 
    # If your results are wrong, we may need to add /255.0 here depending on how you trained it)
    prediction = model.predict(img_array)
    probability = float(prediction[0][0])

    # Display Results
    st.write("---")
    if probability > 0.5:
        confidence = probability * 100
        st.success(f"✅ **This is a VOTER ID CARD** ({confidence:.2f}%)")
    else:
        confidence = (1 - probability) * 100
        st.error(f"❌ **This is NOT a Voter ID** ({confidence:.2f}%)")
# ... inside your app.py ...
    
    # Predict
    prediction = model.predict(img_array)
    probability = float(prediction[0][0])
    
    # Convert probability to percentage
    confidence_score = probability * 100

    st.write("---")
    st.subheader("Analysis Result:")

    # --- THE LOGIC FIX ---
    # We set a HIGH THRESHOLD (e.g., 0.85 or 85%)
    # Even if the model thinks it's 70% likely, we reject it to be safe.
    
    HIGH_THRESHOLD = 0.85  # 85% confidence required

    if probability > HIGH_THRESHOLD:
        st.success(f"✅ **Verified: This IS a Voter ID Card**")
        st.progress(int(confidence_score))
        st.write(f"Model Confidence: {confidence_score:.2f}%")
        st.write("The model is very sure this is a Ghana Voter ID.")
    else:
        # This catches "Other" (low probability) AND "Unsure" images
        st.error(f"❌ **Rejected: NOT a Voter ID Card**")
        
        # specific message depending on how low the score is
        if probability > 0.5:
             st.warning(f"⚠️ The model sees some resemblance ({confidence_score:.2f}%), but not enough to be verified. It might be a blurry image or a different type of card.")
        else:
             st.write(f"Confidence score is very low ({confidence_score:.2f}%). This looks like a random object or person.")