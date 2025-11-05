from flask import Flask, request, jsonify
from oracle1_validation import validate_payload
from ml_model import predict_fault, load_model
from oracle2_finalize import finalize_event
import joblib
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import csv
from apscheduler.schedulers.background import BackgroundScheduler

MODEL_PATH = "fault_model.pkl"
DATA_PATH = "fault_cases.csv"

app = Flask(__name__)

# -- AUTORETRAIN LOGIC --
def retrain_model():
    if not os.path.exists(DATA_PATH):
        return
    X, y = [], []
    with open(DATA_PATH, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            X.append([float(row[0]), float(row[1]), float(row[2])])
            y.append(int(row[3]))
    if len(X) < 2:  # Not enough data yet
        return
    model = LogisticRegression()
    model.fit(np.array(X), np.array(y))
    joblib.dump(model, MODEL_PATH)
    load_model()
    print(f"Auto-retrained model on {len(X)} cases.")

# -- SCHEDULER --
scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(retrain_model, "interval", minutes=60)  # every hour
scheduler.start()

@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "status": "Oracle backend running"})

@app.route("/ingest", methods=["POST"])
def ingest():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"ok": False, "error": "Invalid JSON"}), 400

    valid, cleaned = validate_payload(data)
    if not valid:
        return jsonify({"ok": False, "error": cleaned.get("reason", "Validation failed")}), 400

    ml_ok, result = predict_fault(cleaned)
    if not ml_ok:
        return jsonify({"ok": False, "error": result.get("error", "ML model error")}), 500

    final_ok, status = finalize_event(cleaned.get("panel_id", "unknown"), result)

    # ONLY SAVE FAULTS/WARNINGS/SYSTEM FAILURES
    pred = result.get("prediction")
    if pred == 1:  # If 1 is 'fault' (adjust if needed!)
        with open(DATA_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                cleaned["temperature"],
                cleaned["humidity"],
                cleaned["tilt"],
                pred
            ])

    if final_ok:
        return jsonify({"ok": True, "status": status}), 200
    else:
        return jsonify({"ok": False, "status": status}), 500

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000)
