"""
Flask backend for the academic healthcare Federated Learning project.

This backend provides:
- Project information
- Federated-learning status
- HMAC-SHA256 authentication demonstration
- Chest X-ray image upload endpoint

Important:
The supplied project files do not contain a trained OSDSTran model file.
Therefore the prediction endpoint uses a local Keras model only when
models/healthcare_model.keras exists. Otherwise it returns a clear
"model not loaded" response instead of inventing a prediction.
"""

import os
import hmac
import hashlib
import secrets
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "models" / "healthcare_model.keras"

UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
HMAC_SECRET = os.environ.get("HMAC_SECRET_KEY", "")

app = Flask(__name__, static_folder=str(FRONTEND_DIR))
CORS(app)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def hmac_signature(message, secret):
    return hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "status": "online",
        "service": "Healthcare Federated Learning Backend",
        "model_available": MODEL_PATH.exists(),
    })


@app.get("/api/project")
def project():
    return jsonify({
        "title": "Privacy-Preserving Healthcare Device Authentication via Edge-AI and Federated Learning",
        "technologies": [
            "Python",
            "TensorFlow",
            "OpenCV",
            "Federated Learning",
            "HMAC-SHA256",
            "FedAvg",
            "Medical Image Classification",
            "Edge-AI",
        ],
        "reported_metrics": {
            "accuracy": "98.88%",
            "precision": "98.65%",
            "recall": "98.75%",
            "f1_score": "98.81%",
            "auc": "0.9883",
            "prc": "0.9881",
        },
    })


@app.get("/api/federated/status")
def federated_status():
    return jsonify({
        "server": "Central Federated Server",
        "status": "Ready",
        "clients": [
            {"id": "Hospital-01", "status": "Authenticated"},
            {"id": "Hospital-02", "status": "Authenticated"},
            {"id": "Hospital-03", "status": "Authenticated"},
        ],
        "aggregation": "FedAvg",
        "raw_data_shared": False,
    })


@app.post("/api/authenticate")
def authenticate():
    data = request.get_json(silent=True) or {}
    device_id = str(data.get("device_id", "")).strip()

    if not device_id:
        return jsonify({"authenticated": False, "message": "Device ID is required."}), 400

    if not HMAC_SECRET:
        return jsonify({
            "authenticated": False,
            "message": "HMAC_SECRET_KEY is not configured on the server."
        }), 503

    nonce = secrets.token_hex(16)
    message = f"{device_id}:{nonce}"
    signature = hmac_signature(message, HMAC_SECRET)

    return jsonify({
        "authenticated": True,
        "device_id": device_id,
        "token_type": "HMAC-SHA256",
        "signature": signature,
        "message": "Device authenticated for the demonstration workflow."
    })


@app.post("/api/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"success": False, "message": "Upload an image using the 'image' field."}), 400

    image = request.files["image"]

    if not image.filename:
        return jsonify({"success": False, "message": "No image selected."}), 400

    if not allowed_file(image.filename):
        return jsonify({
            "success": False,
            "message": "Only PNG, JPG and JPEG images are supported."
        }), 400

    filename = secure_filename(image.filename)
    saved_path = UPLOAD_DIR / filename
    image.save(saved_path)

    # No trained model is supplied with the source materials.
    if not MODEL_PATH.exists():
        return jsonify({
            "success": False,
            "model_loaded": False,
            "message": (
                "Image uploaded successfully, but no trained model file was "
                "supplied with the project repository. Add "
                "models/healthcare_model.keras after training to enable "
                "real predictions."
            ),
            "filename": filename,
        })

    # Lazy import so the API can start without TensorFlow installed.
    try:
        import cv2
        import numpy as np
        import tensorflow as tf

        model = tf.keras.models.load_model(MODEL_PATH)

        image_array = cv2.imread(str(saved_path))
        if image_array is None:
            raise ValueError("The uploaded image could not be decoded.")

        image_array = cv2.resize(image_array, (128, 128))
        image_array = image_array.astype("float32") / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        probability = float(model.predict(image_array, verbose=0)[0][0])
        label = "PNEUMONIA" if probability >= 0.5 else "NORMAL"

        return jsonify({
            "success": True,
            "model_loaded": True,
            "prediction": label,
            "probability": round(probability, 4),
            "message": "Prediction generated by the locally loaded Keras model.",
        })

    except Exception as exc:
        return jsonify({
            "success": False,
            "model_loaded": True,
            "message": f"Model inference failed: {exc}",
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
