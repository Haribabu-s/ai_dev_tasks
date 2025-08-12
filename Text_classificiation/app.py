from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

# Load model
model = tf.keras.models.load_model("model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

max_length = 20  # Must match training

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TextRequest):
    # Convert text to padded sequence
    seq = tokenizer.texts_to_sequences([request.text])
    padded = pad_sequences(seq, maxlen=max_length, padding="post")
    
    # Predict
    prediction = model.predict(padded)
    label = "Sick" if prediction[0][0] > 0.5 else "Healthy"
    
    return {
        "input_text": request.text,
        "label": label,
        "confidence": float(prediction[0][0])
    }

@app.get("/")
def read_root():
    return {"message": "Medical Text Classifier API is running"}
