def finalize_event(panel_id: str, result: dict, last_status: str = None):
    """
    Oracle 2: Prediction Verifier
    - Interprets ML prediction and sensor data
    - Returns (log_event, final) where:
        log_event: True if event should be logged, False if skipped
        final: dict with {valid, severity_color, state, details} or {"skip": True}
    """

    try:
        prediction = int(result.get("prediction", -1))
        data = result.get("data", {})

        surface = data.get("surface_temp")
        ambient = data.get("ambient_temp")
        ax, ay, az = data.get("accel_x"), data.get("accel_y"), data.get("accel_z")

        # --- Thresholds ---
        axis_warn = 0.10
        axis_fault = 0.25
        mag_warn = 0.10
        mag_fault = 0.25
        temp_diff_warn = 2.0
        temp_diff_fault = 3.5

        vec_mag = None
        if ax is not None and ay is not None and az is not None:
            vec_mag = (ax**2 + ay**2 + az**2) ** 0.5

        details = "Unknown"

        # --- Normal ---
        if prediction == 0:
            severity_color = "blue"
            state = "Installed_Normal operation"
            details = "No issue detected"
            if last_status == "normal":
                return False, {"skip": True}

        # --- Warning ---
        elif prediction == 2:
            severity_color = "yellow"
            state = "Warning (abnormal values detected)"

            tilt_warning = (
                (ax is not None and abs(ax - 1.0) > axis_warn) or
                (ay is not None and abs(ay) > axis_warn) or
                (az is not None and abs(az + 0.08) > axis_warn) or
                (vec_mag is not None and abs(vec_mag - 1.0) > mag_warn)
            )

            temp_warning = (
                surface is not None and ambient is not None and
                abs(surface - ambient) > temp_diff_warn
            )

            if tilt_warning and temp_warning:
                details = "Orientation and temperature fluctuations"
            elif tilt_warning:
                details = "Panel orientation abnormal (check fixing or wind movement)"
            elif temp_warning:
                details = "Temperature fluctuation (possible insulation issue)"
            else:
                details = "General warning"

        # --- Fault ---
        elif prediction == 1:
            severity_color = "red"
            state = "Confirmed fault (urgent action needed)"

            tilt_fault = (
                (ax is not None and abs(ax - 1.0) > axis_fault) or
                (ay is not None and abs(ay) > axis_fault) or
                (az is not None and abs(az + 0.08) > axis_fault) or
                (vec_mag is not None and abs(vec_mag - 1.0) > mag_fault)
            )

            temp_fault = False
            if surface is not None and ambient is not None:
                if surface < ambient - temp_diff_fault:
                    temp_fault = True
                    details = "Fault due to surface temperature drop (possible thermal loss)"
                elif surface > ambient + temp_diff_fault:
                    temp_fault = True
                    details = "Fault due to overheating (check exposure or insulation)"

            if tilt_fault and not temp_fault:
                details = "Fault due to panel tilt or displacement"
            elif tilt_fault and temp_fault:
                details = "Fault due to tilt and temperature anomaly"
            elif not tilt_fault and not temp_fault:
                details = "Unidentified fault condition"

        # --- System error ---
        else:
            severity_color = "purple"
            state = "System error"
            details = "System error (invalid model output)"

        return True, {
            "valid": (severity_color != "purple"),
            "severity_color": severity_color,
            "state": state,
            "details": details
        }

    except Exception as e:
        return False, {
            "valid": False,
            "severity_color": "purple",
            "state": "System error (exception)",
            "details": str(e)
        }
