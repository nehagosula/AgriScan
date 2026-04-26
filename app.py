import io
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from tensorflow.keras.models import load_model


MODEL_PATH = Path("agriscan_model.h5")
DATASET_PATH = Path("Rice_Leaf_AUG")
IMAGE_SIZE = (224, 224)
UNKNOWN_THRESHOLD = 0.60


app = Flask(__name__)


DISEASE_INFO = {
    "Bacterial Leaf Blight": {
        "cause": "Caused by the bacterium Xanthomonas oryzae, commonly spread through infected seeds, water, and crop residue.",
        "treatment": "Use disease-free seed, avoid excess nitrogen, improve drainage, and apply recommended copper-based treatments when needed.",
    },
    "Brown Spot": {
        "cause": "A fungal disease often linked with nutrient deficiency, drought stress, and poor field conditions.",
        "treatment": "Improve soil nutrition, maintain balanced potassium and phosphorus, remove infected debris, and use suitable fungicide if severe.",
    },
    "Healthy Rice Leaf": {
        "cause": "No visible disease symptoms were detected in the uploaded leaf image.",
        "treatment": "Continue regular monitoring, balanced fertilization, proper irrigation, and field sanitation.",
    },
    "Leaf Blast": {
        "cause": "Caused by the fungus Magnaporthe oryzae and favored by high humidity, cloudy weather, and dense crop growth.",
        "treatment": "Use resistant varieties, avoid excess nitrogen, improve spacing, and apply recommended blast-control fungicides.",
    },
    "Leaf Scald": {
        "cause": "A fungal disease that often appears under humid conditions and plant stress.",
        "treatment": "Use clean seed, remove infected residues, maintain balanced nutrition, and apply suitable fungicide in severe cases.",
    },
    "Leaf scald": {
        "cause": "A fungal disease that often appears under humid conditions and plant stress.",
        "treatment": "Use clean seed, remove infected residues, maintain balanced nutrition, and apply suitable fungicide in severe cases.",
    },
    "Sheath Blight": {
        "cause": "Caused by Rhizoctonia solani and favored by dense planting, high nitrogen, and warm humid conditions.",
        "treatment": "Reduce dense canopy conditions, avoid excess nitrogen, improve water management, and use recommended fungicides when needed.",
    },
    "Unknown": {
        "cause": "The model confidence is below the safe prediction threshold or the image does not clearly match the trained classes.",
        "treatment": "Upload a clear close-up rice leaf image with good lighting, or consult an agriculture expert for confirmation.",
    },
}


def load_class_names() -> list[str]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError("Rice_Leaf_AUG dataset folder was not found.")

    class_names = sorted(item.name for item in DATASET_PATH.iterdir() if item.is_dir())
    if not class_names:
        raise ValueError("No class folders were found inside Rice_Leaf_AUG.")

    return class_names


def load_prediction_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("agriscan_model.h5 was not found.")
    return load_model(MODEL_PATH)


model = load_prediction_model()
class_names = load_class_names()

if model.output_shape[-1] != len(class_names):
    raise ValueError(
        f"Model outputs {model.output_shape[-1]} classes, but dataset has "
        f"{len(class_names)} class folders."
    )


def preprocess_image(uploaded_file) -> np.ndarray:
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def classify_image(image_array: np.ndarray) -> tuple[str, float]:
    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])

    prediction = class_names[predicted_index]
    if confidence < UNKNOWN_THRESHOLD:
        prediction = "Unknown"

    return prediction, confidence


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "app": "AgriScan Flask Backend",
            "status": "running",
            "classes": class_names,
            "predict_endpoint": "/api/predict",
        }
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    uploaded_file = request.files.get("file")
    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({"error": "No file uploaded"}), 400

    try:
        image_array = preprocess_image(uploaded_file)
        prediction, confidence = classify_image(image_array)
        info = DISEASE_INFO.get(prediction, DISEASE_INFO["Unknown"])

        return jsonify(
            {
                "prediction": prediction,
                "confidence": round(confidence * 100, 2),
                "cause": info["cause"],
                "treatment": info["treatment"],
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port)
    #app.run(host="0.0.0.0", port=port, debug=debug)
