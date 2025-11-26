from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import uvicorn

# -------------------------------
# Initialize FastAPI app
# -------------------------------
app = FastAPI(title="Hybrid ONNX Sentiment API")

# -------------------------------
# Load Models
# -------------------------------
print("Loading SentenceTransformer...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading ONNX...")
ort_session = ort.InferenceSession("model/hybrid_sentiment.onnx")

id2label = {0: "Negative", 2: "Positive", 1: "Neutral"}

def clean_text(text):
    # remove html
    text = re.sub(r"<.*?>", "", text)
    # remove url
    text = re.sub(r"http\S+", "", text)
    # remove punctuation + emoji
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    # remove digits
    text = re.sub(r"\d+", "", text)
    # normalize spaces
    text = re.sub(r"\s+", " ", text)
    # lower case
    text = text.lower().strip()
    return text

# -------------------------------
# Request Body
# -------------------------------
class TextInput(BaseModel):
    text: str


# -------------------------------
# Prediction Function
# -------------------------------
def predict_sentiment(text: str):
    # Encode sentence → (1, 384)
    text = clean_text(text)

    emb = encoder.encode([text])
    emb = np.expand_dims(emb, axis=1).astype(np.float32)  # (1, 1, 384)

    # ONNX inference
    ort_inputs = {"embedding": emb}
    logits = ort_session.run(None, ort_inputs)[0]  # (1, num_classes)

    # Softmax
    probs = np.exp(logits) / np.exp(logits).sum()

    # Prediction
    pred = int(np.argmax(probs))
    return {
        "label": id2label[pred],
        "confidence": float(probs[0][pred])
    }


# -------------------------------
# API Routes
# -------------------------------
@app.post("/predict")
def predict(data: TextInput):
    return predict_sentiment(data.text)


@app.get("/")
def home():
    return {"message": "Hybrid ONNX Sentiment API is running!"}


# -------------------------------
# Run API
# -------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# run the app
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload