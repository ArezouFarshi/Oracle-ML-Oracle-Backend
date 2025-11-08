from flask import Flask, request, jsonify, send_file, abort
from oracle1_validation import validate_payload
from ml_model import predict_fault, retrain_model
from oracle2_finalize import finalize_event

app = Flask(__name__)

panel_history = {}
ADMIN_API_KEY = "Admin_acsess_to_platform"

COLOR_CODES = {
    "not_installed":   ("Not installed yet", "gray"),
    "normal":          ("Installed and healthy (Normal operation)", "blue"),
    "warning":         ("Warning (abnormal values detected)", "yellow"),
    "fault":           ("Confirmed fault (urgent action needed)", "red"),
    "system_error":    ("Sensor/ML system/platform error (System error)", "purple"),
}

@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "status": "Oracle backend running"})

@app.route('/download_model', methods=['GET'])
def download_model():
    api_key = request.headers.get("X-API-KEY")
    if api_key != ADMIN_API_KEY:
        abort(403)
    return send_file('fault_model.pkl', as_attachment=True)

@app.route("/ingest", methods=["POST"])
def ingest():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({
            "ok": False,
            "color": COLOR_CODES['system_error'][1],
            "status": COLOR_CODES['system_error'][0],
            "reason": "PlatformError"
        }), 400

    panel_id = data.get("panel_id", "unknown")

    # Gray: Not installed yet (no data)
    if panel_id == "unknown" or not data:
        return jsonify({
            "ok": False,
            "color": COLOR_CODES['not_installed'][1],
            "status": COLOR_CODES['not_installed'][0]
        }), 200

    # Oracle 1: validate payload
    valid, cleaned = validate_payload(data)
    if not valid:
        reason = cleaned.get("reason", "Validation failed")
        error_type = "SensorError" if "sensor" in reason.lower() or "missing" in reason.lower() else "PlatformError"
        return jsonify({
            "ok": False,
            "color": COLOR_CODES['system_error'][1],
            "status": COLOR_CODES['system_error'][0],
            "reason": error_type
        }), 400

    # Oracle 2: ML prediction
    ml_ok, result = predict_fault(cleaned)
    if not ml_ok:
        return jsonify({
            "ok": False,
            "color": COLOR_CODES['system_error'][1],
            "status": COLOR_CODES['system_error'][0],
            "reason": "MLFailure"
        }), 500

    pred = result.get("prediction")
    # Default to normal (blue)
    color, status = COLOR_CODES['normal'][1], COLOR_CODES['normal'][0]
    cause = None

    # If warning/fault, analyze which sensor is most abnormal (distance from "typical" value)
    if pred == 2 or pred == 1:
        color = COLOR_CODES['warning'][1] if pred == 2 else COLOR_CODES['fault'][1]
        status = COLOR_CODES['warning'][0] if pred == 2 else COLOR_CODES['fault'][0]
        st, at, x, y, z = cleaned['surface_temp'], cleaned['ambient_temp'], cleaned['accel_x'], cleaned['accel_y'], cleaned['accel_z']
        deviations = {
            "surface_temp": abs(st - 23.5),   # Use your real healthy mean
            "ambient_temp": abs(at - 24.2),   # Use your real healthy mean
            "accel_x": abs(x - 1.03),
            "accel_y": abs(y - 0.00),
            "accel_z": abs(z - -0.08)
        }
        main_sensor = max(deviations, key=deviations.get)
        if main_sensor == "surface_temp":
            cause = "Surface temperature abnormal"
        elif main_sensor == "ambient_temp":
            cause = "Ambient temperature abnormal"
        elif main_sensor.startswith("accel"):
            cause = "Orientation/tilt abnormal"
        else:
            cause = "Unknown anomaly"

    response = {
        "ok": True,
        "color": color,
        "status": status
    }
    if cause:
        response["reason"] = cause

    return jsonify(response), 200

@app.route("/panel_history/<panel_id>", methods=["GET"])
def get_panel_history(panel_id):
    return jsonify({"panel_id": panel_id, "history": panel_history.get(panel_id, [])})

@app.route("/retrain", methods=["POST"])
def retrain():
    payload = request.get_json(force=True)
    features = payload.get("features")
    labels = payload.get("labels")
    if not features or not labels:
        return jsonify({"ok": False, "error": "Features and labels required"}), 400
    ok, msg = retrain_model(features, labels)
    if ok:
        return jsonify({"ok": True, "status": msg}), 200
    else:
        return jsonify({"ok": False, "error": msg}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
