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

# Load model at startup
load_model()

def reload_model():
    """Reload model after retraining."""
    load_model()

def predict_fault(data: dict):
    """
    Run ML model on validated data.
    """
    if model is None:
        return False, {"error": "Model not trained yet. Please POST to /retrain."}
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
