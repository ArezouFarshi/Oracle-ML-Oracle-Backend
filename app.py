from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from oracle1 import validate_payload

app = Flask(__name__)
CORS(app)

# In-memory model and data storage
model = None
X_train = []
y_train = []

ADMIN_KEY = "Admin_acsess_to_platform"

@app.route("/")
def home():
    return "Oracle backend is running."

@app.route("/ingest", methods=["POST"])
def ingest():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    passed, result = validate_payload(data)
    if not passed:
        return jsonify({"status": "rejected", **result}), 400

    if "warning" in result:
        return jsonify({"status": "warning", **result}), 200

    return jsonify({"status": "accepted", **result}), 200

@app.route("/train", methods=["POST"])
def train():
    if request.headers.get("X-Admin-Key") != ADMIN_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        payload = request.get_json()
        if not payload:
            return jsonify({"error": "Missing JSON payload"}), 400

        features = payload.get("features")
        labels = payload.get("labels")

        if not features or not labels or len(features) != len(labels):
            return jsonify({"error": "Invalid input format"}), 400

        global model, X_train, y_train
        X_train = features
        y_train = labels

        model = RandomForestClassifier(n_estimators=50)
        model.fit(X_train, y_train)
        joblib.dump(model, "trained_model.pkl")

        return jsonify({"status": "Model trained", "samples": len(X_train)}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    passed, result = validate_payload(data)
    if not passed:
        return jsonify({"status": "rejected", **result}), 400

    features = [[
        data["ambient_temp"],
        data["surface_temp"],
        data["accel_x"],
        data["accel_y"],
        data["accel_z"]
    ]]

    global model
    if not model:
        try:
            model = joblib.load("trained_model.pkl")
        except:
            return jsonify({"error": "Model not trained yet"}), 500

    prediction = model.predict(features)[0]
    result["prediction"] = int(prediction)

    return jsonify({"status": "predicted", **result}), 200

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=10000)
