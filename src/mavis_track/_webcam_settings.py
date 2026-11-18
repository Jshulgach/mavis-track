"""Driver-aware USB webcam settings and portable, numeric camera profiles."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import cv2

CONTROLS = {
    "exposure": cv2.CAP_PROP_EXPOSURE,
    "focus": cv2.CAP_PROP_FOCUS,
    "zoom": cv2.CAP_PROP_ZOOM,
    "gain": cv2.CAP_PROP_GAIN,
    "white_balance": cv2.CAP_PROP_WB_TEMPERATURE,
    "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    "autofocus": cv2.CAP_PROP_AUTOFOCUS,
    "auto_white_balance": cv2.CAP_PROP_AUTO_WB,
}
MODE_PROPERTIES = {
    "width": cv2.CAP_PROP_FRAME_WIDTH, "height": cv2.CAP_PROP_FRAME_HEIGHT,
    "fps": cv2.CAP_PROP_FPS, "fourcc": cv2.CAP_PROP_FOURCC,
}
GEOMETRY_KEYS = {"width", "height", "fourcc", "focus", "zoom", "autofocus"}
AUTO_KEYS = {"auto_exposure", "autofocus", "auto_white_balance"}
ALL_KEYS = set(CONTROLS) | set(MODE_PROPERTIES)


class CameraSetupCancelled(RuntimeError):
    """The operator cancelled camera setup before calibration."""


def profile_path():
    from .calibration import default_calibration_dir
    return Path(os.environ.get("MAVIS_WEBCAM_PROFILE",
                               str(default_calibration_dir("webcam") / "webcam-settings.json")))


def validate_settings(settings):
    if not isinstance(settings, dict) or set(settings) - ALL_KEYS:
        raise ValueError("Unknown webcam setting")
    result = {}
    for key, value in settings.items():
        if key in AUTO_KEYS:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be true or false")
        elif key == "fourcc":
            if value not in ("MJPG", "YUY2", "YUYV"):
                raise ValueError("Format must be MJPG, YUY2 or YUYV")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
            if key in ("width", "height") and (value != int(value) or not 16 <= value <= 16384):
                raise ValueError("Resolution must be between 16 and 16384 pixels")
            if key == "fps" and not 0 < value <= 1000:
                raise ValueError("FPS must be between 0 and 1000")
            if key in CONTROLS and abs(value) > 1e6:
                raise ValueError("Driver control value is too large")
        result[key] = value
    if ("width" in result) != ("height" in result):
        raise ValueError("Set width and height together")
    for manual, auto in (("exposure", "auto_exposure"), ("focus", "autofocus"),
                         ("white_balance", "auto_white_balance")):
        if manual in result:
            if result.get(auto) is True:
                raise ValueError(f"Disable {auto} before setting {manual}")
            result[auto] = False
    return result


def load_profile(path=None):
    path = Path(path) if path is not None else profile_path()
    if not path.is_file():
        return {"schema": 1, "backend": "auto", "indices": [], "cameras": {}}
    if path.stat().st_size > 128 * 1024:
        raise ValueError("Webcam profile is too large")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Webcam profile must be a JSON object")  # noqa: TRY004 - Persisted-data validation.
    indices = data.get("indices")
    if (data.get("schema") != 1 or data.get("backend") not in ("auto", "dshow", "msmf")
            or not isinstance(indices, list) or not 1 <= len(indices) <= 16
            or any(type(i) is not int or not 0 <= i < 64 for i in indices)
            or len(set(indices)) != len(indices) or not isinstance(data.get("cameras"), dict)):
        raise ValueError("Invalid webcam profile")
    data["cameras"] = {str(i): validate_settings(data["cameras"].get(str(i), {})) for i in indices}
    identities = data.get("identities", {})
    if (not isinstance(identities, dict) or any(not isinstance(value, str) or len(value) > 4096
                                               for value in identities.values())):
        raise ValueError("Invalid camera identity metadata")
    data["identities"] = identities
    return data


def save_profile(indices, cameras, backend="auto", path=None, *, identities=None):
    path = Path(path) if path is not None else profile_path()
    data = {"schema": 1, "backend": backend, "indices": list(indices), "cameras": cameras,
            "identities": dict(identities or {})}
    # Validate the same structure accepted on the next process start.
    if backend not in ("auto", "dshow", "msmf") or not indices or len(set(indices)) != len(indices):
        raise ValueError("Choose unique cameras and a valid capture backend")
    if len(indices) > 16 or any(type(i) is not int or not 0 <= i < 64 for i in indices):
        raise ValueError("Invalid camera indices")
    data["cameras"] = {str(i): validate_settings(cameras.get(str(i), {})) for i in indices}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def open_webcam(index, backend="auto", factory=None):
    factory = factory or cv2.VideoCapture
    if backend not in ("auto", "dshow", "msmf"):
        raise ValueError("Unknown webcam backend")
    if os.name == "nt":
        apis = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if backend == "auto" else [
            cv2.CAP_DSHOW if backend == "dshow" else cv2.CAP_MSMF]
        for api in apis:
            capture = factory(index, api)
            if capture.isOpened():
                return capture
            capture.release()
        if backend != "auto":
            raise RuntimeError(f"Cannot open camera {index} with {backend}")
    return factory(index)


def backend_name(capture):
    try:
        return str(capture.getBackendName()).upper()
    except (AttributeError, cv2.error):
        return "UNKNOWN"


def read_settings(capture):
    values = {}
    for name, prop in {**CONTROLS, **MODE_PROPERTIES}.items():
        try:
            value = float(capture.get(prop))
            values[name] = value if math.isfinite(value) else None
        except (AttributeError, TypeError, ValueError, cv2.error):
            values[name] = None
    backend = backend_name(capture)
    if "DSHOW" in backend:
        # DirectShow returns native focus flags (1=auto, 2=manual). Its
        # CAP_PROP_AUTO_EXPOSURE getter is absent in some OpenCV versions.
        for key in AUTO_KEYS:
            if values[key] is not None and values[key] < 0:
                values[key] = None
        if values["autofocus"] is not None:
            values["autofocus"] = float(bool(int(values["autofocus"]) & 1))
    if values["fps"] is not None and values["fps"] <= 0:
        values["fps"] = None
    value = values["fourcc"]
    values["fourcc"] = ("".join(chr((int(value) >> (8 * i)) & 255) for i in range(4)).strip("\x00")
                        if value is not None and value > 0 else "")
    return values


def auto_value(name, enabled, backend):
    if name == "auto_exposure":
        if "V4L" in backend:
            return .75 if enabled else .25
        if not any(item in backend for item in ("DSHOW", "MSMF")):
            raise ValueError("Auto exposure encoding is unknown for this backend; use driver settings")
    return float(enabled)


def matches(name, requested, actual, backend):
    if actual is None:
        return False
    if name in AUTO_KEYS:
        return abs(actual - auto_value(name, requested, backend)) < .01
    if name == "fourcc":
        return actual == requested or {actual, requested} <= {"YUY2", "YUYV"}
    tolerance = .5 if name == "fps" else max(.01, abs(requested) * .001)
    return abs(actual - requested) <= tolerance


def apply_settings(capture, settings, *, verify_auto=None, native=None):
    """Apply native settings on the capture owner thread, checking readback.

    Size/format changes roll back together if rejected, including native errors.
    An unsupported FPS request does not discard a successfully negotiated size.
    """
    settings = validate_settings(settings)
    backend = backend_name(capture)
    before = read_settings(capture)
    accepted, errors, unverified = {}, {}, {}
    mode = {key: value for key, value in settings.items() if key in ("width", "height", "fourcc")}

    def restore_mode():
        for key in ("fourcc", "width", "height"):
            value = before[key]
            if value is not None and (len(value) == 4 if key == "fourcc" else value > 0):
                try:
                    capture.set(MODE_PROPERTIES[key], cv2.VideoWriter_fourcc(*value) if key == "fourcc" else value)
                except cv2.error:
                    errors["restore"] = "Previous video mode could not be restored; reopen the camera"

    if "fps" in settings:
        value = settings["fps"]
        try:
            supported = (matches("fps", value, before["fps"], backend)
                         or capture.set(cv2.CAP_PROP_FPS, value))
            actual = read_settings(capture)["fps"]
            if supported and matches("fps", value, actual, backend):
                accepted["fps"] = value
            else:
                errors["fps"] = f"FPS request {value} is unconfirmed (readback {actual}); use received FPS"
        except cv2.error as exc:
            errors["fps"] = str(exc)
    try:
        success = True
        for key in ("fourcc", "width", "height"):
            if key in mode and not matches(key, mode[key], before[key], backend):
                value = cv2.VideoWriter_fourcc(*mode[key]) if key == "fourcc" else mode[key]
                success = bool(capture.set(MODE_PROPERTIES[key], value)) and success
        actual = read_settings(capture)
        if mode and (not success or not all(matches(k, v, actual[k], backend) for k, v in mode.items())):
            errors["mode"] = "Requested video mode rejected/coerced; restoring previous mode"
            restore_mode()
        else:
            accepted.update(mode)
    except (TypeError, ValueError, cv2.error) as exc:
        errors["mode"] = str(exc)
        restore_mode()
    for key in ["auto_exposure", "autofocus", "auto_white_balance",
                "exposure", "focus", "zoom", "gain", "white_balance"]:
        if key not in settings:
            continue
        value = settings[key]
        required_auto = {"exposure": "auto_exposure", "focus": "autofocus",
                         "white_balance": "auto_white_balance"}.get(key)
        if required_auto in errors:
            errors[key] = f"{key}: manual mode could not be confirmed"
            continue
        try:
            native_value = auto_value(key, value, backend) if key in AUTO_KEYS else value
            success = bool(capture.set(CONTROLS[key], native_value))
            actual = read_settings(capture)[key]
            if actual is None and key in AUTO_KEYS and verify_auto is not None:
                actual = verify_auto(key)
            if success and actual is None and key in AUTO_KEYS:
                # Some DirectShow OpenCV builds implement the setter but no
                # getter. Keep the acknowledged request, never invent readback.
                accepted[key] = value
                unverified[key] = value
            elif not success or not matches(key, value, actual, backend):
                errors[key] = f"{key}: rejected or unconfirmed (requested {value}; readback {actual})"
            else:
                accepted[key] = value
        except (ValueError, TypeError, cv2.error) as exc:
            errors[key] = f"{key}: {exc}"
    native_modes = {}
    if native is not None and ("DSHOW" in backend) and (unverified or set(errors) & set(CONTROLS)):
        states = native(settings)
        current = read_settings(capture)
        for key, value in settings.items():
            if key not in CONTROLS:
                continue
            prop = {"auto_exposure": "exposure", "autofocus": "focus",
                    "auto_white_balance": "white_balance"}.get(key, key)
            state = states.get(prop, {})
            if not state.get("set_ok"):
                continue
            actual = (float(state["current_auto"]) if key in AUTO_KEYS
                      and state.get("current_auto") is not None else current.get(key))
            if matches(key, value, actual, backend):
                accepted[key] = value
                errors.pop(key, None)
                unverified.pop(key, None)
                if key in AUTO_KEYS:
                    native_modes[key] = actual
    actual = read_settings(capture)
    actual.update(native_modes)
    actual["_native_modes"] = native_modes
    actual["_unverified_controls"] = unverified
    return accepted, errors, actual


def geometry_signature(snapshot):
    """Record only lens properties that the Windows driver actually exposes."""
    actual, capabilities = snapshot.get("actual", {}), snapshot.get("capabilities", {})
    if (not isinstance(actual, dict) or not isinstance(capabilities, dict)
            or any(not isinstance(value, dict) for value in capabilities.values())):
        raise ValueError("Invalid webcam lens metadata")
    for key in ("focus", "zoom", "autofocus"):
        value = actual.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int))
                                  or not math.isfinite(value)):
            raise ValueError("Invalid webcam lens values")
    keys = [key for key in ("focus", "zoom") if key in capabilities
            and not capabilities[key].get("unsupported") and actual.get(key) is not None]
    result = {key: actual[key] for key in keys}
    if "focus" in keys and actual.get("autofocus") is not None:
        result["autofocus"] = actual["autofocus"]
    return result
