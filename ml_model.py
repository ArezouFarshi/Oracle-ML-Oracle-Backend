import joblib
import os

model_path = os.getenv("MODEL_PATH", "fault_model.pkl")
model = joblib.load(model_path)

def predict_fault(data):
    try:
        X = [[data["ambient_c"], data["object_c"], data["diff_c"]]]
        y_pred = model.predict(X)[0]
        y_score = model.predict_proba(X)[0][1]
        return True, {
            "fault": bool(y_pred),
            "score": round(y_score, 3),
            "decision": "FAULT" if y_pred else "NORMAL"
        }
    except Exception as e:
        return False, {"error": str(e)}
