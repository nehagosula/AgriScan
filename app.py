import io
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------- CONFIG ---------------- #
MODEL_PATH = Path("agriscan_model.h5")
IMAGE_SIZE = (224, 224)
UNKNOWN_THRESHOLD = 0.60

# Static class names (NO DATASET NEEDED)
class_names = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf Scald",
    "Sheath Blight"|
    "Unknown"
]

app = Flask(__name__)

# ---------------- DISEASE INFO ---------------- #
DISEASE_INFO = {
    "Bacterial Leaf Blight": {
        "cause": "Caused by the bacterium Xanthomonas oryzae, spread through infected seeds, water, and crop residue.",
        "treatment": "Use disease-free seed, avoid excess nitrogen, improve drainage, and apply copper-based treatments.",
    },
    "Brown Spot": {
        "cause": "A fungal disease linked with nutrient deficiency and drought stress.",
        "treatment": "Improve soil nutrition, maintain potassium levels, and apply fungicide if severe.",
    },
    "Healthy Rice Leaf": {
        "cause": "No disease symptoms detected.",
        "treatment": "Maintain proper irrigation and balanced fertilization.",
    },
    "Leaf Blast": {
        "cause": "Caused by fungus Magnaporthe oryzae under humid conditions.",
        "treatment": "Use resistant varieties and apply fungicides if needed.",
    },
    "Leaf Scald": {
        "cause": "Fungal disease under humid conditions and plant stress.",
        "treatment": "Use clean seeds and maintain proper field hygiene.",
    },
    "Sheath Blight": {
        "cause": "Caused by Rhizoctonia solani in dense planting.",
        "treatment": "Reduce canopy density and apply fungicides.",
    },
    "Unknown": {
        "cause": "Low confidence or unknown condition.",
        "treatment": "Upload a clearer image or consult an expert.",
    },
}

# ---------------- LOAD MODEL ---------------- #
def load_prediction_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file agriscan_model.h5 not found.")
    return load_model(MODEL_PATH)

model = load_prediction_model()

# ---------------- PREPROCESS ---------------- #
def preprocess_image(uploaded_file) -> np.ndarray:
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)

# ---------------- PREDICTION ---------------- #
def classify_image(image_array: np.ndarray):
    probabilities = model.predict(image_array, verbose=0)[0]
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
        "app": "AgriScan Flask Backend",
        "status": "running",
        "classes": class_names,
        "predict_endpoint": "/api/predict",
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
        return jsonify({"error": str(exc)}), 500

# ---------------- RUN APP ---------------- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
