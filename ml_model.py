import os
import time
from flask import Flask, request, jsonify, send_file, abort
from web3 import Web3
from oracle1_validation import validate_payload
from ml_model import predict_fault, retrain_model

app = Flask(__name__)

# --- access and state ---
ADMIN_API_KEY = "Admin_acsess_to_platform"  # keep your existing wording
panel_history = {}  # { panel_id: "not_installed" | "normal" | "warning" | "fault" | "system_error" }

COLOR_CODES = {
    "not_installed":   ("Not installed yet", "gray"),
    "normal":          ("Installed and healthy (Normal operation)", "blue"),
    "warning":         ("Warning (abnormal values detected)", "yellow"),
    "fault":           ("Confirmed fault (urgent action needed)", "red"),
    "system_error":    ("Sensor/ML system/platform error (System error)", "purple"),
}

# --- Web3 setup ---
INFURA_URL = os.environ["INFURA_URL"]
ORACLE_PRIVATE_KEY = os.environ["ORACLE_PRIVATE_KEY"]
CONTRACT_ADDRESS = os.environ.get("CONTRACT_ADDRESS", "0xB0561d4580126DdF8DEEA9B7e356ee3F26A52e40")

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

@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "status": "Oracle backend running"})

@app.route("/download_model", methods=["GET"])
def download_model():
    api_key = request.headers.get("X-API-KEY")
    if api_key != ADMIN_API_KEY:
        abort(403)
    return send_file("fault_model.pkl", as_attachment=True)

@app.route("/ingest", methods=["POST"])
def ingest():
    # Parse request
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

    # Handle missing panel_id → treat as not installed (one-time)
    if panel_id == "unknown":
        response = {
            "ok": False,
            "color": COLOR_CODES['not_installed'][1],
            "status": COLOR_CODES['not_installed'][0],
            "prediction": -1
        }
        if panel_history.get(panel_id) != "not_installed":
            tx_hash = log_to_blockchain(panel_id, response)
            response["tx_hash"] = tx_hash
            panel_history[panel_id] = "not_installed"
        return jsonify(response), 200

    # Oracle 1: validate payload
    valid, cleaned = validate_payload(data)
    if not valid:
        response = {
            "ok": False,
            "color": COLOR_CODES['system_error'][1],
            "status": COLOR_CODES['system_error'][0],
            "reason": "SensorError",
            "prediction": -1
        }
        last = panel_history.get(panel_id)
        if last != "system_error":
            tx_hash = log_to_blockchain(panel_id, response)
            response["tx_hash"] = tx_hash
            panel_history[panel_id] = "system_error"
        return jsonify(response), 400

    # Oracle 2: predict + verify ML outcomes
    ml_ok, result = predict_fault(cleaned)
    if not ml_ok:
        response = {
            "ok": False,
            "color": COLOR_CODES['system_error'][1],
            "status": COLOR_CODES['system_error'][0],
            "reason": "MLFailure",
            "prediction": -1
        }
        last = panel_history.get(panel_id)
        if last != "system_error":
            tx_hash = log_to_blockchain(panel_id, response)
            response["tx_hash"] = tx_hash
            panel_history[panel_id] = "system_error"
        return jsonify(response), 500

    pred = int(result.get("prediction", -1))
    cause = result.get("reason", "")

    if pred == 0:
        new_status = "normal"
        color, status = COLOR_CODES['normal'][1], COLOR_CODES['normal'][0]
    elif pred == 2:
        new_status = "warning"
        color, status = COLOR_CODES['warning'][1], COLOR_CODES['warning'][0]
    elif pred == 1:
        new_status = "fault"
        color, status = COLOR_CODES['fault'][1], COLOR_CODES['fault'][0]
    else:
        new_status = "system_error"
        color, status = COLOR_CODES['system_error'][1], COLOR_CODES['system_error'][0]

    response = {
        "ok": (new_status != "system_error"),
        "color": color,
        "status": status,
        "prediction": pred
    }
    if cause and new_status in ("warning", "fault"):
        response["reason"] = cause

    # Event-driven logging: only on status change
    last_status = panel_history.get(panel_id)
    if last_status != new_status:
        tx_hash = log_to_blockchain(panel_id, response)
        response["tx_hash"] = tx_hash
        panel_history[panel_id] = new_status

    return jsonify(response), 200

@app.route("/retrain", methods=["POST"])
def retrain():
    payload = request.get_json(force=True)
    features = payload.get("features")
    labels = payload.get("labels")
    if not features or not labels:
        return jsonify({"ok": False, "error": "Features and labels required
