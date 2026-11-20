"""Thin Win32 helpers: DPI awareness, monitor enumeration, global hotkeys."""

from __future__ import annotations

import ctypes
from ctypes import wintypes as wt
from dataclasses import dataclass

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
user32.SetWindowDisplayAffinity.restype = wt.BOOL
user32.SetWindowDisplayAffinity.argtypes = [wt.HWND, wt.DWORD]

# ---------------------------------------------------------------- DPI


def enable_dpi_awareness() -> None:
    """Make all coordinates physical pixels so overlay/Tk agree with screen."""
    try:
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):  # PMv2
            return
    except AttributeError:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass


# ---------------------------------------------------------------- monitors


@dataclass(frozen=True)
class Monitor:
    x: int
    y: int
    width: int
    height: int
    primary: bool

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2

    def label(self) -> str:
        dpi = "プライマリ" if self.primary else ""
        return f"{self.width}x{self.height} @({self.x},{self.y}) {dpi}".strip()


class _MONITORINFOEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wt.DWORD),
        ("rcMonitor", wt.RECT),
        ("rcWork", wt.RECT),
        ("dwFlags", wt.DWORD),
        ("szDevice", ctypes.c_wchar * 32),
    ]


def list_monitors() -> list[Monitor]:
    found: list[Monitor] = []
    proc_type = ctypes.WINFUNCTYPE(
        ctypes.c_int, wt.HMONITOR, wt.HDC, ctypes.POINTER(wt.RECT), wt.LPARAM
    )

    def _cb(hmon, hdc, lprect, lparam):
        info = _MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(info)
        if user32.GetMonitorInfoW(hmon, ctypes.byref(info)):
            r = info.rcMonitor
            found.append(
                Monitor(
                    x=r.left,
                    y=r.top,
                    width=r.right - r.left,
                    height=r.bottom - r.top,
                    primary=bool(info.dwFlags & 1),
                )
            )
        return 1

    user32.EnumDisplayMonitors(None, None, proc_type(_cb), 0)
    found.sort(key=lambda m: (not m.primary, m.x, m.y))
    if not found:
        w = user32.GetSystemMetrics(0)
        h = user32.GetSystemMetrics(1)
        found.append(Monitor(0, 0, w, h, True))
    return found


# ---------------------------------------------------------------- hotkeys

VK_CODES = {
    **{f"F{i}": 0x6F + i for i in range(1, 13)},
    "ESC": 0x1B,
    "SPACE": 0x20,
    "TAB": 0x09,
    "HOME": 0x24,
    "END": 0x23,
    "INSERT": 0x2D,
    "DELETE": 0x2E,
    "PAUSE": 0x13,
    "SCROLL": 0x91,
    **{str(i): 0x30 + i for i in range(10)},
    **{chr(65 + i): 0x41 + i for i in range(26)},
}


def vk_code(name: str) -> int | None:
    name = str(name).strip().upper()
    if name in VK_CODES:
        return VK_CODES[name]
    if len(name) == 1 and name.isalnum():
        return ord(name)
    if name.startswith("VK_"):
        return getattr(wt, name, None)
    return None


class HotkeyWatcher:
    """Poll GetAsyncKeyState so hotkeys work while another app has focus."""

    def __init__(self, keys: dict[str, str]):
        self._codes = {}
        self._modifiers = {}
        for action, name in keys.items():
            tokens = [token.strip().upper() for token in str(name).split("+")]
            if not tokens or any(not token for token in tokens):
                continue
            modifiers = set(tokens[:-1])
            if not modifiers.issubset({"ALT", "CTRL", "SHIFT", "WIN"}):
                continue
            code = vk_code(tokens[-1])
            if code is not None:
                self._codes[action] = code
                self._modifiers[action] = modifiers
        self._down: dict[str, bool] = {
            action: bool(user32.GetAsyncKeyState(code) & 0x8000)
            for action, code in self._codes.items()
        }

    def poll(self) -> list[str]:
        pressed: list[str] = []
        modifier_codes = {
            "ALT": (0x12, 0xA4, 0xA5),
            "CTRL": (0x11, 0xA2, 0xA3),
            "SHIFT": (0x10, 0xA0, 0xA1),
            "WIN": (0x5B, 0x5C),
        }
        for action, code in self._codes.items():
            modifiers = self._modifiers[action]
            active_modifiers = {
                name for name, codes in modifier_codes.items()
                if any(user32.GetAsyncKeyState(modifier) & 0x8000 for modifier in codes)
            }
            state = (
                bool(user32.GetAsyncKeyState(code) & 0x8000)
                and active_modifiers == modifiers
            )
            if state and not self._down[action]:
                pressed.append(action)
            self._down[action] = state
        return pressed
