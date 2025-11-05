from flask import Flask, request, jsonify
from oracle1_validation import validate_payload
from ml_model import predict_fault
from oracle2_finalize import finalize_event

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "status": "Oracle backend running"})

@app.route("/ingest", methods=["POST"])
def ingest():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"ok": False, "error": "Invalid JSON"}), 400

    # Oracle 1: validate payload
    valid, cleaned = validate_payload(data)
    if not valid:
        return jsonify({"ok": False, "error": cleaned.get("reason", "Validation failed")}), 400

    # ML: predict fault
    ml_ok, result = predict_fault(cleaned)
    if not ml_ok:
        return jsonify({"ok": False, "error": result.get("error", "ML model error")}), 500

    # Oracle 2: finalize event
    final_ok, status = finalize_event(cleaned.get("panel_id", "unknown"), result)
    if final_ok:
        return jsonify({"ok": True, "status": status}), 200
    else:
        return jsonify({"ok": False, "status": status}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
