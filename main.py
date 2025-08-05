from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from io import BytesIO
import logging


logging.basicConfig(level=logging.INFO)


app = FastAPI()


try:
    model = load_model("waste_management.h5")
    logging.info("✅ Model loaded successfully.")
except Exception as e:
    logging.error("❌ Failed to load model.")
    raise RuntimeError(f"Model load failed: {e}")


class_names = ["Non-Recyclable", "Organic", "Recyclable"]


def read_image(file: bytes) -> Image.Image:
    try:
        img = Image.open(BytesIO(file)).convert("RGB")
        return img
    except Exception as e:
        raise ValueError("Invalid image format. Please upload a valid image.") from e

@app.get("/")
def read_root():
    return {"message": "Waste Classification API is running."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        
        image_bytes = await file.read()
        img = read_image(image_bytes)
        img = img.resize((224, 224))  
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  

        
        predictions = model.predict(img_array)
        predicted_class_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_class_index]

       
        logging.info(f"Predicted: {predicted_class}, Probabilities: {predictions.tolist()}")

        return {"predicted_type": predicted_class}

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})



