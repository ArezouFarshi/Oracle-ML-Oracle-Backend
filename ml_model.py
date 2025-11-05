import joblib
import numpy as np
import os

MODEL_PATH = "fault_model.pkl"
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model = None

def predict_fault(data: dict):
    if model is None:
        return False, {"error": "Model not trained yet."}
    try:
        features = np.array([
            data["temperature"],
            data["humidity"],
            data["tilt"]
        ]).reshape(1, -1)
        prediction = model.predict(features)[0]
        return True, {"prediction": int(prediction)}
    except Exception as e:
        return False, {"error": str(e)}
