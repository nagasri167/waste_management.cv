import streamlit as st
from PIL import Image
import random
import io

# Set up the app title
st.set_page_config(page_title="Waste Classifier", page_icon="♻️")
st.title("🔋 Waste Management System 🎈")

# Capture image from webcam
cam = st.camera_input("Capture an image of the waste")

# Simulated prediction logic (can be replaced with real ML model)
def predict_waste(image: Image.Image) -> str:
    # This is a placeholder: randomly returns a class
    classes = ["Recyclable", "Organic", "Hazardous", "Electronic", "General Waste"]
    return random.choice(classes)

if cam:
    # Load image
    img = Image.open(io.BytesIO(cam.getvalue()))
    st.image(img, caption="Captured Image", use_column_width=True)

    # Simulate prediction
    predicted_class = predict_waste(img)

    # Display prediction
    st.success(f"Predicted Class: {predicted_class}")
