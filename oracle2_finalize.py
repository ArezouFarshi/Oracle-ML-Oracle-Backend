def finalize_event(panel_id, result, last_status=None):
    """
    Oracle 2: Prediction Verifier
    - Interprets ML prediction
    - Detects specific cause
    - Returns final event dict with color, status, reason
    - Enforces event-driven logging (no duplicate normals)
    """
    try:
        prediction = int(result.get("prediction", -1))
        data = result.get("data", {})  # optional sensor values

        surface = data.get("surface_temp")
        ambient = data.get("ambient_temp")
        ax, ay, az = data.get("accel_x"), data.get("accel_y"), data.get("accel_z")

        # Thresholds for cause analysis
        tilt_limit = 0.25
        temp_diff_warn = 2.0
        temp_diff_fault = 3.5

        # Default
        cause = "Unknown"

        if prediction == 0:
            color = "blue"
            status = "Installed and healthy (Normal operation)"
            cause = "No issue detected"
            # Prevent duplicate normals: skip logging if already normal
            if last_status == "normal":
                return False, {"skip": True}
        elif prediction == 2:
            color = "yellow"
            status = "Warning (abnormal values detected)"
            if (surface is not None and ambient is not None and
                abs(surface - ambient) > temp_diff_warn):
                cause = "Temperature fluctuation (possible insulation issue)"
            elif ((ax is not None and abs(ax - 1.0) > tilt_limit) or
                  (ay is not None and abs(ay) > tilt_limit) or
                  (az is not None and abs(az + 0.08) > tilt_limit)):
                cause = "Panel orientation abnormal (check fixing or wind movement)"
            else:
                cause = "General warning"
        elif prediction == 1:
            color = "red"
            status = "Confirmed fault (urgent action needed)"
            if (surface is not None and ambient is not None and
                surface < ambient - temp_diff_fault):
                cause = "Fault due to surface temperature drop (possible thermal loss)"
            elif (surface is not None and ambient is not None and
                  surface > ambient + temp_diff_fault):
                cause = "Fault due to overheating (check exposure or insulation)"
            elif ((ax is not None and abs(ax - 1.0) > tilt_limit * 2) or
                  (ay is not None and abs(ay) > tilt_limit * 2) or
                  (az is not None and abs(az + 0.08) > tilt_limit * 2)):
                cause = "Fault due to panel tilt or displacement"
            else:
                cause = "Unidentified fault condition"
        else:
            color = "purple"
            status = "Sensor/ML system/platform error"
            cause = "System error (invalid model output)"

        return True, {
            "ok": (color != "purple"),
            "color": color,
            "status": status,
            "reason": cause
        }

    except Exception as e:
        return False, {
            "ok": False,
            "color": "purple",
            "status": "System error (exception)",
            "reason": str(e)
        }
