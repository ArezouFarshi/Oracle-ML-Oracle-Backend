# Simulated ML prediction model for fault detection
THRESHOLD = 8.0  # °C difference

def predict_fault(data):
    diff = abs(data["diff_c"])
    fault = diff >= THRESHOLD
    return True, {
        "fault": fault,
        "score": diff,
        "threshold": THRESHOLD,
        "decision": "FAULT" if fault else "NORMAL"
    }
