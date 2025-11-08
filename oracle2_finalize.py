def finalize_event(panel_id, result):
    try:
        prediction = int(result.get("prediction", -1))
        data = result.get("data", {})  # optional

        surface = data.get("surface_temp")
        ambient = data.get("ambient_temp")
        ax, ay, az = data.get("accel_x"), data.get("accel_y"), data.get("accel_z")

        # Define thresholds for fault causes
        tilt_limit = 0.25      # g deviation beyond ±0.25 = tilt issue
        temp_diff_warn = 2.0   # °C difference for warning
        temp_diff_fault = 3.5  # °C difference for fault

        cause = "Unknown"

        # --- Detect specific cause ---
        if prediction == 0:
            color = "blue"
            status = "Installed and healthy (Normal operation)"
            cause = "No issue detected"
        elif prediction == 2:
            color = "yellow"
            status = "Warning (abnormal values detected)"
            if abs(surface - ambient) > temp_diff_warn:
                cause = "Temperature fluctuation (possible insulation issue)"
            elif abs(ax - 1.0) > tilt_limit or abs(ay) > tilt_limit or abs(az + 0.08) > tilt_limit:
                cause = "Panel orientation abnormal (check fixing or wind movement)"
            else:
                cause = "General warning"
        elif prediction == 1:
            color = "red"
            status = "Confirmed fault (urgent action needed)"
            if surface < ambient - temp_diff_fault:
                cause = "Fault due to surface temperature drop (possible thermal loss)"
            elif surface > ambient + temp_diff_fault:
                cause = "Fault due to overheating (check exposure or insulation)"
            elif abs(ax - 1.0) > tilt_limit * 2 or abs(ay) > tilt_limit * 2 or abs(az + 0.08) > tilt_limit * 2:
                cause = "Fault due to panel tilt or displacement"
            else:
                cause = "Unidentified fault condition"
        else:
            color = "purple"
            status = "Sensor/ML system/platform error"
            cause = "System error (invalid model output)"

        return True, {
            "ok": True,
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
