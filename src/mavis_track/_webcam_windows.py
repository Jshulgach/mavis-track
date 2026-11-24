"""Read DirectShow camera identity and control ranges using Windows COM only.

No stream is started here. All COM pointers and apartments are released on the
calling thread. Media Foundation indices are not treated as DirectShow indices.
"""
from __future__ import annotations

import ctypes as C
import os
import uuid


class GUID(C.Structure):
    _fields_ = [("data", C.c_ubyte * 16)]

    def __init__(self, text):
        super().__init__()
        self.data[:] = uuid.UUID(text).bytes_le


class Value(C.Union):
    _fields_ = [("bstr", C.c_void_p), ("number", C.c_double), ("padding", C.c_void_p * 2)]


class Variant(C.Structure):
    _anonymous_ = ("value",)
    _fields_ = [("vt", C.c_ushort), ("reserved", C.c_ushort * 3), ("value", Value)]


def _call(ptr, slot, result, args, *values):
    table = C.cast(ptr, C.POINTER(C.POINTER(C.c_void_p))).contents
    return C.WINFUNCTYPE(result, C.c_void_p, *args)(table[slot])(ptr, *values)


def _release(ptr):
    if ptr.value:
        _call(ptr, 2, C.c_ulong, ())
        ptr.value = None


def _query(ptr, iid):
    result = C.c_void_p()
    hr = _call(ptr, 0, C.c_long, (C.POINTER(GUID), C.POINTER(C.c_void_p)),
               C.byref(GUID(iid)), C.byref(result))
    return result if hr >= 0 else C.c_void_p()


def _property(moniker, key, automation):
    bag = C.c_void_p()
    iid = GUID("55272A00-42CB-11CE-8135-00AA004BB851")
    hr = _call(moniker, 9, C.c_long,
               (C.c_void_p, C.c_void_p, C.POINTER(GUID), C.POINTER(C.c_void_p)),
               None, None, C.byref(iid), C.byref(bag))
    if hr < 0:
        return ""
    value = Variant()
    try:
        hr = _call(bag, 3, C.c_long, (C.c_wchar_p, C.POINTER(Variant), C.c_void_p),
                   key, C.byref(value), None)
        return C.wstring_at(value.bstr) if hr >= 0 and value.vt == 8 and value.bstr else ""
    finally:
        automation.VariantClear(C.byref(value))
        _release(bag)


def _ranges(moniker, requested=None):
    capture = C.c_void_p()
    hr = _call(moniker, 8, C.c_long,
               (C.c_void_p, C.c_void_p, C.POINTER(GUID), C.POINTER(C.c_void_p)),
               None, None, C.byref(GUID("56A86895-0AD4-11CE-B03A-0020AF0BA770")), C.byref(capture))
    if hr < 0:
        return {}
    result = {}
    try:
        groups = [
            ("C6E13370-30AC-11D0-A18C-00A0C9118956", {"exposure": 4, "focus": 6, "zoom": 3}),
            ("C6E13360-30AC-11D0-A18C-00A0C9118956", {"white_balance": 7, "gain": 9}),
        ]
        for iid, properties in groups:
            interface = _query(capture, iid)
            if not interface.value:
                continue
            try:
                for name, prop in properties.items():
                    values = [C.c_long() for _ in range(5)]
                    hr = _call(interface, 3, C.c_long, (C.c_long, *([C.POINTER(C.c_long)] * 5)),
                               prop, *(C.byref(v) for v in values))
                    if hr >= 0:
                        low, high, step, default, flags = [v.value for v in values]
                        if low <= high:
                            current, mode = C.c_long(), C.c_long()
                            got = _call(interface, 5, C.c_long,
                                        (C.c_long, C.POINTER(C.c_long), C.POINTER(C.c_long)),
                                        prop, C.byref(current), C.byref(mode))
                            auto_name = {"exposure": "auto_exposure", "focus": "autofocus",
                                         "white_balance": "auto_white_balance"}.get(name)
                            update = requested or {}
                            set_ok = None
                            if name in update or auto_name in update:
                                automatic = bool(update.get(auto_name, False)) if auto_name else False
                                value = update.get(name, current.value)
                                if (value == int(value) and low <= value <= high
                                        and (int(value) - low) % max(step, 1) == 0
                                        and flags & (1 if automatic else 2)):
                                    status = _call(interface, 4, C.c_long, (C.c_long, C.c_long, C.c_long),
                                                   prop, int(value), 1 if automatic else 2)
                                    set_ok = status >= 0
                                    got = _call(interface, 5, C.c_long,
                                                (C.c_long, C.POINTER(C.c_long), C.POINTER(C.c_long)),
                                                prop, C.byref(current), C.byref(mode))
                                else:
                                    set_ok = False
                            result[name] = {"minimum": low, "maximum": high, "step": max(1, step),
                                                "default": default, "auto": bool(flags & 1), "manual": bool(flags & 2),
                                                "set_ok": set_ok, "current": current.value if got >= 0 else None,
                                                "current_auto": bool(mode.value & 1) if got >= 0 else None}
                    elif hr & 0xFFFFFFFF in (0x80070490, 0x80070057):
                        result[name] = {"unsupported": True}
            finally:
                _release(interface)
    finally:
        _release(capture)
    return result


def enumerate_devices(control_index=None, *, requested=None, expected_path=""):
    """Match the system enumerator order used by OpenCV's DirectShow backend."""
    if os.name != "nt":
        return []
    ole = C.WinDLL("ole32")
    automation = C.OleDLL("oleaut32")
    ole.CoInitializeEx.argtypes = [C.c_void_p, C.c_ulong]
    ole.CoInitializeEx.restype = C.c_long
    # STA works with native camera settings dialogs; an existing apartment is OK.
    initialized = ole.CoInitializeEx(None, 2)
    if initialized < 0 and initialized & 0xFFFFFFFF != 0x80010106:
        return []
    device_enum, monikers = C.c_void_p(), C.c_void_p()
    result = []
    try:
        ole.CoCreateInstance.argtypes = [C.POINTER(GUID), C.c_void_p, C.c_ulong,
                                        C.POINTER(GUID), C.POINTER(C.c_void_p)]
        hr = ole.CoCreateInstance(C.byref(GUID("62BE5D10-60EB-11D0-BD3B-00A0C911CE86")), None, 1,
                                 C.byref(GUID("29840822-5B84-11D0-BD3B-00A0C911CE86")), C.byref(device_enum))
        if hr < 0:
            return []
        hr = _call(device_enum, 3, C.c_long, (C.POINTER(GUID), C.POINTER(C.c_void_p), C.c_ulong),
                   C.byref(GUID("860BB310-5D01-11D0-BD3B-00A0C911CE86")), C.byref(monikers), 0)
        if hr != 0:
            return []
        while True:
            moniker, fetched = C.c_void_p(), C.c_ulong()
            hr = _call(monikers, 3, C.c_long, (C.c_ulong, C.POINTER(C.c_void_p), C.POINTER(C.c_ulong)),
                       1, C.byref(moniker), C.byref(fetched))
            if hr != 0:
                break
            try:
                index = len(result)
                path = _property(moniker, "DevicePath", automation)
                inspect = index == control_index and (
                    not expected_path or device_identity(path) == device_identity(expected_path))
                result.append({"index": index, "name": _property(moniker, "FriendlyName", automation),
                               "device_path": path,
                               "controls": _ranges(moniker, requested) if inspect else {}})
            finally:
                _release(moniker)
        return result
    except (OSError, ValueError, C.ArgumentError):
        return []
    finally:
        _release(monikers)
        _release(device_enum)
        if initialized >= 0:
            ole.CoUninitialize()


def enumerate_media_foundation():
    """Read the video device order used by OpenCV's Media Foundation backend."""
    if os.name != "nt":
        return []
    ole = C.WinDLL("ole32")
    try:
        platform = C.WinDLL("mfplat")
        mf = C.WinDLL("mf")
    except OSError:
        return []  # Media Foundation components are optional on some Windows editions.
    pointer = C.c_void_p
    device_array = C.POINTER(pointer)()
    attributes = pointer()
    count = C.c_ulong()
    ole.CoTaskMemFree.argtypes = (pointer,)
    ole.CoTaskMemFree.restype = None
    initialized = ole.CoInitializeEx(None, 0)
    if initialized < 0 and initialized & 0xFFFFFFFF != 0x80010106:
        return []
    started = False
    try:
        if platform.MFStartup(0x00020070, 0) < 0:
            return []
        started = True
        if platform.MFCreateAttributes(C.byref(attributes), 1) < 0:
            return []
        source_type = GUID("c60ac5fe-252a-478f-a0ef-bc8fa5f7cad3")
        video_type = GUID("8ac3587a-4ae7-42d8-99e0-0a6013eef90f")
        if _call(attributes, 24, C.c_long, (C.POINTER(GUID), C.POINTER(GUID)),
                 C.byref(source_type), C.byref(video_type)) < 0:
            return []
        if mf.MFEnumDeviceSources(attributes, C.byref(device_array), C.byref(count)) < 0:
            return []
        return [{"index": index, "controls": {},
                 "name": _mf_string(pointer(device_array[index]), "60d0e559-52f8-4fa2-bbce-acdb34a8ec01", ole),
                 "device_path": _mf_string(pointer(device_array[index]), "58f0aad8-22bf-4f8a-bb3d-d2c4978c6e2f", ole)}
                for index in range(count.value)]
    finally:
        if device_array:
            for index in range(count.value):
                _release(pointer(device_array[index]))
            ole.CoTaskMemFree(device_array)
        _release(attributes)
        if started:
            platform.MFShutdown()
        if initialized >= 0:
            ole.CoUninitialize()


def _mf_string(device, key, ole):
    value, length = C.c_void_p(), C.c_ulong()
    # IMFAttributes::GetAllocatedString
    try:
        hr = _call(device, 13, C.c_long, (C.POINTER(GUID), C.POINTER(C.c_void_p), C.POINTER(C.c_ulong)),
                   C.byref(GUID(key)), C.byref(value), C.byref(length))
        return C.wstring_at(value, length.value) if hr >= 0 and value else ""
    finally:
        ole.CoTaskMemFree(value)


def device_identity(path):
    """Compare Windows device interface paths across capture APIs."""
    return path.casefold().removesuffix(r"\global")


def apply_controls(index, requested, *, expected_path=""):
    """Fallback for drivers that require explicit IAM* manual/auto flags.

    OpenCV's DirectShow value setters pass flags=0, which some valid UVC drivers
    reject. Readback from the active capture still verifies manual values.
    """
    info = next((device for device in enumerate_devices(index, requested=requested, expected_path=expected_path)
                 if device["index"] == index), {})
    return info.get("controls", {})
