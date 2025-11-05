import joblib
import numpy as np
import os

MODEL_PATH = "fault_model.pkl"

def is_model_available():
    return os.path.exists(MODEL_PATH)

def load_model():
    if is_model_available():
        return joblib.load(MODEL_PATH)
    return None

def predict_fault(data: dict):
    if not is_model_available():
        return False, {"error": "Model not trained yet."}
    try:
        model = load_model()
        features = np.array([
            data["surface_temp"],
            data["ambient_temp"]
        ]).reshape(1, -1)
        prediction = model.predict(features)[0]
        return True, {"prediction": int(prediction)}
    except Exception as e:
        return False, {"error": str(e)}
