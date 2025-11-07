from flask import Flask, request, jsonify, send_file, abort
from oracle1_validation import validate_payload
from ml_model import predict_fault, retrain_model
from oracle2_finalize import finalize_event

app = Flask(__name__)

# In-memory store for per-panel records (cleared on restart)
panel_history = {}

# Set your admin API key here (change the value as you wish)
ADMIN_API_KEY = "Admin_acsess_to_platform"  # Or any value for testing

@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "status": "Oracle backend running"})

@app.route('/download_model', methods=['GET'])
def download_model():
    api_key = request.headers.get("X-API-KEY")
    if api_key != ADMIN_API_KEY:
        abort(403)  # Forbidden
    return send_file('fault_model.pkl', as_attachment=True)

@app.route("/ingest", methods=["POST"])
def ingest():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"ok": False, "error": "Invalid JSON"}), 400

    panel_id = data.get("panel_id", "unknown")

    # Oracle 1: validate payload
    valid, cleaned = validate_payload(data)
    if not valid:
        # Save fault to panel history
        panel_history.setdefault(panel_id, []).append({"input": data, "result": cleaned})
        return jsonify({"ok": False, "error": cleaned.get("reason", "Validation failed")}), 400

    # ML: predict fault
    ml_ok, result = predict_fault(cleaned)
    if not ml_ok:
        panel_history.setdefault(panel_id, []).append({"input": data, "result": result})
        return jsonify({"ok": False, "error": result.get("error", "ML model error")}), 500

    # Oracle 2: finalize event
    final_ok, status = finalize_event(panel_id, result)
    # Log every result per panel
    panel_history.setdefault(panel_id, []).append({"input": data, "result": status})

    if final_ok:
        return jsonify({"ok": True, "status": status}), 200
    else:
        return jsonify({"ok": False, "status": status}), 500

@app.route("/panel_history/<panel_id>", methods=["GET"])
def get_panel_history(panel_id):
    # Show all saved events for this panel (since last restart)
    return jsonify({"panel_id": panel_id, "history": panel_history.get(panel_id, [])})

@app.route("/retrain", methods=["POST"])
def retrain():
    # Expects JSON: { "features": [...], "labels": [...] }
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
