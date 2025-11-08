import os
import time
from flask import Flask, request, jsonify, send_file, abort
from web3 import Web3
from oracle1_validation import validate_payload
from ml_model import predict_fault, retrain_model
from oracle2_finalize import finalize_event

app = Flask(__name__)

# --- Access and per-panel state (last known status) ---
ADMIN_API_KEY = "Admin_acsess_to_platform"  # keep your existing wording
panel_history = {}  # { panel_id: "not_installed" | "normal" | "warning" | "fault" | "system_error" }

COLOR_CODES = {
    "not_installed":   ("Not installed yet", "gray"),
    "normal":          ("Installed and healthy (Normal operation)", "blue"),
    "warning":         ("Warning (abnormal values detected)", "yellow"),
    "fault":           ("Confirmed fault (urgent action needed)", "red"),
    "system_error":    ("Sensor/ML system/platform error (System error)", "purple"),
}

# --- Web3 / contract setup ---
INFURA_URL = os.environ["INFURA_URL"]
ORACLE_PRIVATE_KEY = os.environ["ORACLE_PRIVATE_KEY"]
CONTRACT_ADDRESS = os.environ["CONTRACT_ADDRESS"]  # deployed PanelEvents contract

w3 = Web3(Web3.HTTPProvider(INFURA_URL))
oracle_account = w3.eth.account.from_key(ORACLE_PRIVATE_KEY)

# Minimal ABI for addPanelEvent()
ABI = [
    {
        "inputs": [
            {"internalType": "string", "name": "panelId", "type": "string"},
            {"internalType": "bool", "name": "ok", "type": "bool"},
            {"internalType": "string", "name": "color", "type": "string"},
            {"internalType": "string", "name": "status", "type": "string"},
            {"internalType": "int256", "name": "prediction", "type": "int256"},
            {"internalType": "string", "name": "reason", "type": "string"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"}
        ],
        "name": "addPanelEvent",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]
contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=ABI)

def log_to_blockchain(panel_id: str, payload: dict) -> str:
    """
    Build and send addPanelEvent(panelId, ok, color, status, prediction, reason, timestamp).
    """
    tx = contract.functions.addPanelEvent(
        panel_id,
        bool(payload.get("ok", False)),
        str(payload.get("color", COLOR_CODES["system_error"][1])),
        str(payload.get("status", COLOR_CODES["system_error"][0])),
        int(payload.get("prediction", -1)),
        str(payload.get("reason", "")),
        int(time.time())
    ).build_transaction({
        "from": oracle_account.address,
        "nonce": w3.eth.get_transaction_count(oracle_account.address),
        "gas": 500000,
        "gasPrice": w3.to_wei("10", "gwei")
    })
    signed = w3.eth.account.sign_transaction(tx, ORACLE_PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
    return w3.to_hex(tx_hash)

def log_if_changed(panel_id: str, new_status: str, payload: dict) -> dict:
    """
    Event-driven logging helper:
    - Logs to blockchain only if status changes.
    - Updates panel_history.
    - Returns payload with tx_hash if logged.
    """
    last_status = panel_history.get(panel_id)
    if last_status != new_status:
        tx_hash = log_to_blockchain(panel_id, payload)
        payload["tx_hash"] = tx_hash
        panel_history[panel_id] = new_status
    return payload

@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "status": "Oracle backend running"})

# Admin-only model download (unchanged as requested)
@app.route("/download_model", methods=["GET"])
def download_model():
    api_key = request.headers.get("X-API-KEY")
    if api_key != ADMIN_API_KEY:
        abort(403)
    return send_file("fault_model.pkl", as_attachment=True)

@app.route("/ingest", methods=["POST"])
def ingest():
    # Parse JSON
    try:
        data = request.get_json(force=True)
    except Exception:
        response = {
            "ok": False,
            "color": COLOR_CODES['system_error'][1],
            "status": COLOR_CODES['system_error'][0],
            "reason": "PlatformError",
            "prediction": -1
        }
        # System/platform error is always logged
        tx_hash = log_to_blockchain("unknown", response)
        response["tx_hash"] = tx_hash
        return jsonify(response), 400

    panel_id = data.get("panel_id", "unknown")

    # Installation logic (one-time): if panel_id unknown treat as not_installed
    if panel_id == "unknown":
        payload = {
            "ok": False,
            "color": COLOR_CODES['not_installed'][1],
            "status": COLOR_CODES['not_installed'][0],
            "prediction": -1
        }
        response = log_if_changed(panel_id, "not_installed", payload)
        return jsonify(response), 200

    # Oracle 1: validate sensor payload (Trust Filter)
    valid, vresult = validate_payload(data)
    if not valid:
        # Purple system error (sensor error)
        payload = {
            "ok": False,
            "color": COLOR_CODES['system_error'][1],
            "status": COLOR_CODES['system_error'][0],
            "reason": vresult.get("reason", "SensorError"),
            "prediction": -1
        }
        response = log_if_changed(panel_id, "system_error", payload)
        return jsonify(response), 400

    # If Oracle 1 flagged a warning (pass-through), preserve cleaned data
    if isinstance(vresult, dict) and "warning" in vresult:
        cleaned = {k: vresult.get(k, data.get(k)) for k in ["surface_temp", "ambient_temp", "accel_x", "accel_y", "accel_z"]}
    else:
        cleaned = {
            "surface_temp": data["surface_temp"],
            "ambient_temp": data["ambient_temp"],
            "accel_x": data["accel_x"],
            "accel_y": data["accel_y"],
            "accel_z": data["accel_z"]
        }

    # Oracle 2: run ML and finalize event (Prediction Verifier)
    ml_ok, ml_result = predict_fault(cleaned)
    if not ml_ok:
        # Purple ML error / platform error
        payload = {
            "ok": False,
            "color": COLOR_CODES['system_error'][1],
            "status": COLOR_CODES['system_error'][0],
            "reason": "MLFailure",
            "prediction": -1
        }
        response = log_if_changed(panel_id, "system_error", payload)
        return jsonify(response), 500

    # Attach raw data to result for cause analysis in finalize_event
    ml_result["data"] = cleaned

    # Finalize event (status, color, reason), enforcing no duplicate normals
    last_status = panel_history.get(panel_id, None)
    ok2, final = finalize_event(panel_id, ml_result, last_status=last_status)

    # If finalize_event says skip (duplicate normal), return current state without logging
    if not ok2 and final.get("skip"):
        # Build a lightweight response showing current normal state
        response = {
            "ok": True,
            "color": COLOR_CODES['normal'][1],
            "status": COLOR_CODES['normal'][0],
            "prediction": 0
        }
        return jsonify(response), 200

    # Otherwise, log if status changed
    # We need to infer new_status from final["status"]
    # Map status string back to canonical key
    status_str = final["status"]
    if status_str.startswith("Installed and healthy"):
        new_status = "normal"
        prediction = 0
    elif status_str.startswith("Warning"):
        new_status = "warning"
        prediction = 2
    elif status_str.startswith("Confirmed fault"):
        new_status = "fault"
        prediction = 1
    else:
        new_status = "system_error"
        prediction = -1

    payload = {
        "ok": final.get("ok", True),
        "color": final.get("color", COLOR_CODES['system_error'][1]),
        "status": status_str,
        "reason": final.get("reason", ""),
        "prediction": prediction
    }
    response = log_if_changed(panel_id, new_status, payload)
    return jsonify(response), 200

# Admin retraining (unchanged as requested)
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
    return jsonify({"ok": False, "error": msg}), 500

if __name__ == "__main__":
    # In production, use Gunicorn or similar WSGI server
    app.run(host="0.0.0.0", port=5000)
