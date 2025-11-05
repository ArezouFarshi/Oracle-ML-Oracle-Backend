def validate_payload(data: dict):
    required = ["panel_id", "surface_temp", "ambient_temp"]
    for field in required:
        if field not in data:
            return False, {"reason": f"Missing field: {field}"}

    # Surface temperature checks
    st = data["surface_temp"]
    if st < -15 or st > 85:
        return False, {"reason": "Surface temperature FAULT"}
    if st < -10 or st > 75:
        return True, {"warning": "Surface temperature WARNING", **data}

    # Ambient temperature checks
    at = data["ambient_temp"]
    if at < -20 or at > 55:
        return False, {"reason": "Ambient temperature FAULT"}
    if at < -10 or at > 45:
        return True, {"warning": "Ambient temperature WARNING", **data}

    # You can add tilt/accel_x/accel_y/accel_z logic later

    return True, data
