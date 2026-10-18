
import json
import joblib
import numpy as np

from pathlib import Path
from flask import Flask, jsonify, request

# Menentukan lokasi folder proyek
BASE_DIR = Path(__file__).resolve().parent

# Lokasi file model dan metrik
MODEL_PATH = BASE_DIR / "models" / "best_regressor_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "model_metrics.json"

# Nama fitur sesuai urutan saat model dilatih
FEATURES_NAMES = [
    "age",
    "sex",
    "bmi",
    "bp",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6"
]

# Membuat aplikasi Flask
app = Flask(__name__)

# Memuat model
model = None
model_error = None

try:
    model = joblib.load(MODEL_PATH)
except Exception as error:
    model_error = str(error)


# Halaman utama
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Diabetes Regression API",
        "status": "running"
    })


# Memeriksa status API dan model
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })


# Informasi model
@app.route("/api/model", methods=["GET"])
def model_info():
    if not METRICS_PATH.exists():
        return jsonify({
            "error": "File model_metrics.json tidak ditemukan"
        }), 404

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        return jsonify(metrics)

    except (OSError, json.JSONDecodeError) as error:
        return jsonify({
            "error": str(error)
        }), 500


# Prediksi diabetes
@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "error": "Model gagal dimuat",
            "detail": model_error
        }), 500

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Kirim data menggunakan format JSON"
        }), 400

    # Memastikan semua fitur tersedia
    missing_features = [
        feature for feature in FEATURES_NAMES
        if feature not in data
    ]

    if missing_features:
        return jsonify({
            "error": "Ada fitur yang belum diisi",
            "missing_features": missing_features
        }), 400

    try:
        # Mengambil fitur sesuai urutan training
        features = [
            float(data[feature])
            for feature in FEATURES_NAMES
        ]

        # Memastikan input berupa angka yang valid
        if not np.all(np.isfinite(features)):
            return jsonify({
                "error": "Semua fitur harus berupa angka yang valid"
            }), 400

        # Mengubah data menjadi array 2 dimensi
        input_data = np.array([features])

        # Melakukan prediksi
        prediction = model.predict(input_data)

        result = float(prediction[0])

        return jsonify({
            "prediction": round(result, 2)
        })

    except (ValueError, TypeError) as error:
        return jsonify({
            "error": "Nilai fitur harus berupa angka",
            "detail": str(error)
        }), 400

    except Exception as error:
        app.logger.exception("Prediksi gagal")

        return jsonify({
            "error": "Terjadi kesalahan saat melakukan prediksi",
            "detail": str(error)
        }), 500


# Menjalankan aplikasi secara lokal
if __name__ == "__main__":
    app.run(
        debug=True,
        port=5001
    )