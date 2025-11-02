def validate_payload(data):
    panel_id = data.get("panel_id", "")
    sensors = data.get("sensors", {})
    mlx = sensors.get("mlx90614", {})

    amb = mlx.get("ambient_c")
    obj = mlx.get("object_c")

    if panel_id != "ID_27_C_42":
        return False, {"reason": "unauthorized_panel"}

    if amb is None or obj is None:
        return False, {"reason": "missing_temperature_values"}

    try:
        amb = float(amb)
        obj = float(obj)
    except ValueError:
        return False, {"reason": "invalid_number_format"}

    if not (-40 <= amb <= 125) or not (-40 <= obj <= 380):
        return False, {"reason": "values_out_of_range"}

    return True, {
        "panel_id": panel_id,
        "ambient_c": amb,
        "object_c": obj,
        "diff_c": obj - amb
    }
