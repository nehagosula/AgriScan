import io
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------- CONFIG ---------------- #
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "agriscan_model.h5"
IMAGE_SIZE = (224, 224)
UNKNOWN_THRESHOLD = 0.60

# Static class names
class_names = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf Scald",
    "Sheath Blight",
    "Unknown"
]

app = Flask(__name__)

# ---------------- DISEASE INFO ---------------- #
DISEASE_INFO = {
    "Bacterial Leaf Blight": {
        "cause": "Bacterial infection spread via water and seeds.",
        "treatment": "Use disease-free seeds and proper drainage.",
    },
    "Brown Spot": {
        "cause": "Fungal disease due to nutrient deficiency.",
        "treatment": "Improve soil nutrients and apply fungicide.",
    },
    "Healthy Rice Leaf": {
        "cause": "No disease detected.",
        "treatment": "Maintain proper crop management.",
    },
    "Leaf Blast": {
        "cause": "Fungal infection under humid conditions.",
        "treatment": "Use resistant varieties and fungicide.",
    },
    "Leaf Scald": {
        "cause": "Fungal disease due to stress conditions.",
        "treatment": "Improve field hygiene.",
    },
    "Sheath Blight": {
        "cause": "Fungal infection in dense crops.",
        "treatment": "Reduce plant density and apply fungicide.",
    },
    "Unknown": {
        "cause": "Low confidence or unclear image.",
        "treatment": "Upload clearer image or consult expert.",
    },
}

# ---------------- MODEL LOADING ---------------- #
model = None

def load_prediction_model():
    try:
        print("Checking model path:", MODEL_PATH)

        if not MODEL_PATH.exists():
            print(" Model file NOT found!")
            return None

        print("Files in directory:", os.listdir(BASE_DIR))

        loaded_model = load_model(MODEL_PATH)
        print(" Model loaded successfully")

        return loaded_model

    except Exception as e:
        print(" Model loading failed:", e)
        return None


def get_model():
    global model
    if model is None:
        print(" Loading model now...")
        model = load_prediction_model()
    return model


# ---------------- PREPROCESS ---------------- #
def preprocess_image(uploaded_file):
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


# ---------------- PREDICTION ---------------- #
def classify_image(image_array):
    model_instance = get_model()

    if model_instance is None:
        print(" Model not loaded, returning Unknown")
        return "Unknown", 0.0

    probabilities = model_instance.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])

    prediction = class_names[predicted_index]

    if confidence < UNKNOWN_THRESHOLD:
        prediction = "Unknown"

    return prediction, confidence


# ---------------- ROUTES ---------------- #
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "model_loaded": get_model() is not None,
        "model_path": str(MODEL_PATH),
        "files_here": os.listdir(BASE_DIR),
        "classes": class_names
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    uploaded_file = request.files.get("file")

    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({"error": "No file uploaded"}), 400

    try:
        image_array = preprocess_image(uploaded_file)
        prediction, confidence = classify_image(image_array)

        info = DISEASE_INFO.get(prediction, DISEASE_INFO["Unknown"])

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "cause": info["cause"],
            "treatment": info["treatment"],
        })

    except Exception as exc:
        print("Prediction error:", exc)
        return jsonify({"error": str(exc)}), 500


# ---------------- RUN APP ---------------- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f" Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)
