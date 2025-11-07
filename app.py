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

def diagnose_fault(data):
    messages = []
    warning = False
    # Surface temperature
    if data.get('surface_temp') is not None:
        st = data['surface_temp']
        if st > 35:
            messages.append("High surface temperature: possible solar load, thermal bridge, or insulation defect.")
            if st < 37:
                warning = True
        elif st < 20:
            messages.append("Low surface temperature: possible thermal bridge, insulation defect, or air leakage.")
            if st > 18:
                warning = True
    # Ambient temperature
    if data.get('ambient_temp') is not None:
        at = data['ambient_temp']
        if at < 20 or at > 28:
            messages.append("Ambient temperature out of comfort range: possible HVAC issue or open window.")
            if 18 < at < 20 or 28 < at < 30:
                warning = True
    # Panel moved/displaced
    accel_x, accel_y, accel_z = data.get('accel_x', 0), data.get('accel_y', 0), data.get('accel_z', 1)
    if abs(accel_x) > 1.2 or abs(accel_y) > 1.2 or abs(accel_z) < 0.8:
        messages.append("Panel orientation abnormal: possible displacement, fixing failure, or wind-induced movement.")
        if 1.0 < abs(accel_x) <= 1.2 or 1.0 < abs(accel_y) <= 1.2 or 0.7 < abs(accel_z) < 0.8:
            warning = True
    if warning and messages:
        return "; ".join(messages), "warning"
    if messages:
        return "; ".join(messages), "fault"
    return None, "normal"

def system_error_reason(data, error_type="SensorError"):
    if error_type == "SensorError":
        missing_fields = [k for k in ['surface_temp', 'ambient_temp', 'accel_x', 'accel_y', 'accel_z'] if data.get(k) is None]
        if missing_fields:
            return "SensorError"
    elif error_type == "MLFailure":
        return "MLFailure"
    else:
        return "PlatformError"
    return "SystemError"

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

    # Finalize Oracle: can be expanded later
    final_ok, status_text = finalize_event(panel_id, result)
    color, status = COLOR_CODES['normal'][1], COLOR_CODES['normal'][0]

    # Check for system/platform errors (missing data)
    if any(cleaned.get(k) is None for k in ['surface_temp', 'ambient_temp', 'accel_x', 'accel_y', 'accel_z']):
        return jsonify({
            "ok": False,
            "color": COLOR_CODES['system_error'][1],
            "status": COLOR_CODES['system_error'][0],
            "reason": "SensorError"
        }), 500

    # Analyze for warning or fault
    reason, detected_type = diagnose_fault(cleaned)
    if detected_type == "warning":
        color, status = COLOR_CODES['warning'][1], COLOR_CODES['warning'][0]
    elif detected_type == "fault":
        color, status = COLOR_CODES['fault'][1], COLOR_CODES['fault'][0]
    else:
        color, status = COLOR_CODES['normal'][1], COLOR_CODES['normal'][0]

    result_json = {
        "ok": True,
        "color": color,
        "status": status_text if detected_type == "normal" else status
    }
    if detected_type != "normal":
        result_json["reason"] = reason
    return jsonify(result_json), 200

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
