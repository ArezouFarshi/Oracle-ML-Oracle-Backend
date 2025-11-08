import os
import time
from flask import Flask, request, jsonify, send_file, abort
from web3 import Web3
from oracle1_validation import validate_payload
from ml_model import predict_fault, retrain_model
from oracle2_finalize import finalize_event

app = Flask(__name__)

# --- Access and per-panel state ---
ADMIN_API_KEY = "Admin_acsess_to_platform"
panel_history = {}       # { panel_id: last_status }
panel_last_seen = {}     # { panel_id: unix_timestamp }

COLOR_CODES = {
    "not_installed":   ("Not installed yet", "gray"),
    "normal":          ("Installed_Normal operation", "blue"),
    "warning":         ("Warning (abnormal values detected)", "yellow"),
    "fault":           ("Confirmed fault (urgent action needed)", "red"),
    "system_error":    ("Sensor/ML system/platform error (System error)", "purple"),
}

STALE_TIMEOUT = int(os.environ.get("STALE_TIMEOUT", "300"))  # default 5 minutes

# --- Web3 / contract setup ---
INFURA_URL = os.environ["INFURA_URL"]
ORACLE_PRIVATE_KEY = os.environ["ORACLE_PRIVATE_KEY"]
CONTRACT_ADDRESS = os.environ["CONTRACT_ADDRESS"]

w3 = Web3(Web3.HTTPProvider(INFURA_URL))
oracle_account = w3.eth.account.from_key(ORACLE_PRIVATE_KEY)

ABI = [
    {
        "inputs": [
            {"internalType": "string", "name": "panelId", "type": "string"},
            {"internalType": "bool", "name": "valid", "type": "bool"},
            {"internalType": "string", "name": "severity_color", "type": "string"},
            {"internalType": "string", "name": "state", "type": "string"},
            {"internalType": "int256", "name": "prediction", "type": "int256"},
            {"internalType": "string", "name": "details", "type": "string"},
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
    tx = contract.functions.addPanelEvent(
        panel_id,
        bool(payload.get("valid", False)),
        str(payload.get("severity_color", COLOR_CODES["system_error"][1])),
        str(payload.get("state", COLOR_CODES["system_error"][0])),
        int(payload.get("prediction", -1)),
        str(payload.get("details", "")),
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
    last_status = panel_history.get(panel_id)
    if last_status != new_status:
        tx_hash = log_to_blockchain(panel_id, payload)
        payload["tx_hash"] = tx_hash
        panel_history[panel_id] = new_status
    return payload


def mark_seen(panel_id: str):
    if panel_id and panel_id != "unknown":
        panel_last_seen[panel_id] = int(time.time())


@app.route("/", methods=["GET"])
def health():
    return jsonify({"valid": True, "state": "Oracle backend running", "timeout": STALE_TIMEOUT})


@app.route("/download_model", methods=["GET"])
def download_model():
    api_key = request.headers.get("X-API-KEY")
    if api_key != ADMIN_API_KEY:
        abort(403)
    return send_file("fault_model.pkl", as_attachment=True)


@app.route("/train", methods=["POST"])
def train():
    """
    Retrain the ML model with new data (admin only).
    """
    api_key = request.headers.get("X-API-KEY")
    if api_key != ADMIN_API_KEY:
        abort(403)

    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"valid": False, "details": "Invalid JSON payload"}), 400

    ok, result = retrain_model(data)
    return jsonify({"valid": ok, "details": result}), 200 if ok else 500


@app.route("/monitor", methods=["POST"])
def monitor():
    payload = request.get_json(silent=True) or {}
    now = int(time.time())
    scope = payload.get("panels")
    checked = scope if isinstance(scope, list) and scope else list(panel_last_seen.keys())

    results = []
    for pid in checked:
        last = panel_last_seen.get(pid)
        if not last:
            results.append({"panel_id": pid, "logged": False, "note": "Never seen"})
            continue

        is_stale = (now - last) >= STALE_TIMEOUT
        if is_stale:
            if panel_history.get(pid) != "system_error":
                purple = {
                    "valid": False,
                    "severity_color": COLOR_CODES["system_error"][1],
                    "state": COLOR_CODES["system_error"][0],
                    "details": "No data received (microcontroller/power/communication failure)",
                    "prediction": -1
                }
                logged = log_if_changed(pid, "system_error", purple)
                results.append({"panel_id": pid, "logged": True, "tx_hash": logged.get("tx_hash")})
            else:
                results.append({"panel_id": pid, "logged": False, "note": "Already in system_error"})
        else:
            results.append({"panel_id": pid, "logged": False, "note": "Active"})

    return jsonify({"valid": True, "checked": results, "timeout": STALE_TIMEOUT}), 200


@app.route("/ingest", methods=["POST"])
def ingest():
    try:
        data = request.get_json(force=True)
    except Exception:
        response = {
            "valid": False,
            "severity_color": COLOR_CODES['system_error'][1],
            "state": COLOR_CODES['system_error'][0],
            "details": "PlatformError",
            "prediction": -1
        }
        tx_hash = log_to_blockchain("unknown", response)
        response["tx_hash"] = tx_hash
        return jsonify(response), 400

    panel_id = data.get("panel_id", "unknown")
    mark_seen(panel_id)

    if panel_id == "unknown":
        payload = {
            "valid": False,
            "severity_color": COLOR_CODES['not_installed'][1],
            "state": COLOR_CODES['not_installed'][0],
            "prediction": -1
        }
        response = log_if_changed(panel_id, "not_installed", payload)
        return jsonify(response), 200

    valid, vresult = validate_payload(data)
    if not valid:
        payload = {
            "valid": False,
            "severity_color": COLOR_CODES['system_error'][1],
            "state": COLOR_CODES['system_error'][0],
            "details": vresult.get("reason", "SensorError"),
            "prediction": -1
        }
        response = log_if_changed(panel_id, "system_error", payload)
        return jsonify(response), 400

    cleaned = {
        "surface_temp": data["surface_temp"],
        "ambient_temp": data["ambient_temp"],
        "accel_x": data["accel_x"],
        "accel_y": data["accel_y"],
        "accel_z": data["accel_z"]
    }

    ml_ok, ml_result = predict_fault(cleaned)
    if not ml_ok:
        payload = {
            "valid": False,
            "severity_color": COLOR_CODES['system_error'][1],
            "state": COLOR_CODES['system_error'][0],
            "details": "MLFailure",
            "prediction": -1
        }
        response = log_if_changed(panel_id, "system_error", payload)
        return jsonify(response), 500

    ml_result["data"] = cleaned
    last_status = panel_history.get(panel_id, None)
    log_event, final = finalize_event(panel_id, ml_result, last_status=last_status)

    if not log_event and final.get("skip"):
        response = {
            "valid": True,
            "severity_color": COLOR_CODES['normal'][1],
            "state": COLOR_CODES['normal'][0],
            "prediction": 0
        }
        return jsonify(response), 200

    color = final.get("severity_color", COLOR_CODES['system_error'][1])
    if color == "blue":
        new_status = "normal"
        prediction = 0
    elif color == "yellow":
        new_status = "warning"
        prediction = 2
    elif color == "red":
        new_status = "fault"
        prediction = 1
