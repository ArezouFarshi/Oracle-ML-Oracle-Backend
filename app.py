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

    # Oracle 2: ML prediction (THIS IS WHERE THE STATUS COMES FROM)
    ml_ok, result = predict_fault(cleaned)
    if not ml_ok:
        return jsonify({
            "ok": False,
            "color": COLOR_CODES['system_error'][1],
            "status": COLOR_CODES['system_error'][0],
            "reason": "MLFailure"
        }), 500

    # result["prediction"] should be 0 (normal), 1 (fault), or 2 (warning)
    pred = result.get("prediction")
    if pred == 0:
        color, status = COLOR_CODES['normal'][1], COLOR_CODES['normal'][0]
    elif pred == 2:
        color, status = COLOR_CODES['warning'][1], COLOR_CODES['warning'][0]
    elif pred == 1:
        color, status = COLOR_CODES['fault'][1], COLOR_CODES['fault'][0]
    else:
        color, status = COLOR_CODES['system_error'][1], COLOR_CODES['system_error'][0]

    response = {
        "ok": True,
        "color": color,
        "status": status
    }
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
