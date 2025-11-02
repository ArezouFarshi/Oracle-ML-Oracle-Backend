def validate_payload(data: dict):
    """
    Oracle 1: Validate incoming sensor payload.
    Checks required fields and plausible ranges.
    """
    required = ["panel_id", "temperature", "humidity", "tilt"]
    for field in required:
        if field not in data:
            return False, {"reason": f"Missing field: {field}"}

    if not (-40 <= data["temperature"] <= 80):
        return False, {"reason": "Temperature out of range"}
    if not (0 <= data["humidity"] <= 100):
        return False, {"reason": "Humidity out of range"}
    if not (-90 <= data["tilt"] <= 90):
        return False, {"reason": "Tilt out of range"}

    return True, data
