"""Click-through layered-window gaze dot (pure Win32, no GUI toolkit)."""

from __future__ import annotations

import ctypes
import math
from ctypes import wintypes as wt

import cv2
import numpy as np

from .win32 import gdi32, kernel32, user32

WS_POPUP = 0x80000000
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000

SW_SHOWNOACTIVATE = 4
SW_HIDE = 0
ULW_ALPHA = 0x00000002
AC_SRC_OVER = 0x00
AC_SRC_ALPHA = 0x01
HWND_TOPMOST = wt.HWND(-1)
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_NOACTIVATE = 0x0010
DIB_RGB_COLORS = 0


class _BLENDFUNCTION(ctypes.Structure):
    _fields_ = [
        ("BlendOp", ctypes.c_ubyte),
        ("BlendFlags", ctypes.c_ubyte),
        ("SourceConstantAlpha", ctypes.c_ubyte),
        ("AlphaFormat", ctypes.c_ubyte),
    ]


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wt.DWORD),
        ("biWidth", wt.LONG),
        ("biHeight", wt.LONG),
        ("biPlanes", wt.WORD),
        ("biBitCount", wt.WORD),
        ("biCompression", wt.DWORD),
        ("biSizeImage", wt.DWORD),
        ("biXPelsPerMeter", wt.LONG),
        ("biYPelsPerMeter", wt.LONG),
        ("biClrUsed", wt.DWORD),
        ("biClrImportant", wt.DWORD),
    ]


def _bind() -> None:
    user32.CreateWindowExW.restype = wt.HWND
    user32.CreateWindowExW.argtypes = [
        wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, wt.HWND, wt.HMENU, wt.HINSTANCE, wt.LPVOID,
    ]
    user32.UpdateLayeredWindow.restype = wt.BOOL
    user32.UpdateLayeredWindow.argtypes = [
        wt.HWND, wt.HDC, ctypes.POINTER(wt.POINT), ctypes.POINTER(wt.SIZE),
        wt.HDC, ctypes.POINTER(wt.POINT), wt.COLORREF,
        ctypes.POINTER(_BLENDFUNCTION), wt.DWORD,
    ]
    user32.GetDC.restype = wt.HDC
    user32.GetDC.argtypes = [wt.HWND]
    user32.ReleaseDC.restype = ctypes.c_int
    user32.ReleaseDC.argtypes = [wt.HWND, wt.HDC]
    user32.DestroyWindow.restype = wt.BOOL
    user32.DestroyWindow.argtypes = [wt.HWND]
    user32.ShowWindow.restype = wt.BOOL
    user32.ShowWindow.argtypes = [wt.HWND, ctypes.c_int]
    user32.SetWindowPos.restype = wt.BOOL
    user32.SetWindowPos.argtypes = [
        wt.HWND, wt.HWND, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, wt.UINT,
    ]
    gdi32.CreateDIBSection.restype = wt.HBITMAP
    gdi32.CreateDIBSection.argtypes = [
        wt.HDC, ctypes.POINTER(_BITMAPINFOHEADER), wt.UINT,
        ctypes.POINTER(ctypes.c_void_p), wt.HANDLE, wt.DWORD,
    ]
    gdi32.CreateCompatibleDC.restype = wt.HDC
    gdi32.CreateCompatibleDC.argtypes = [wt.HDC]
    gdi32.SelectObject.restype = wt.HGDIOBJ
    gdi32.SelectObject.argtypes = [wt.HDC, wt.HGDIOBJ]
    gdi32.DeleteObject.restype = wt.BOOL
    gdi32.DeleteObject.argtypes = [wt.HGDIOBJ]
    gdi32.DeleteDC.restype = wt.BOOL
    gdi32.DeleteDC.argtypes = [wt.HDC]
    kernel32.GetModuleHandleW.restype = wt.HMODULE
    kernel32.GetModuleHandleW.argtypes = [wt.LPCWSTR]


_bind()


def render_trail(
    width: int,
    points: list[tuple[float, float, float, float]],
    thickness: float,
    color: tuple[int, int, int],
    opacity: float,
    shadow: bool,
    glow_intensity: float = 0.72,
    glow_radius: float = 16.0,
    height: int | None = None,
) -> np.ndarray:
    """Render the outer contour of overlapping transparent trail circles."""
    width = max(1, int(width))
    height = width if height is None else max(1, int(height))
    if width * height > 400 * 400:
        scale = 0.5
        reduced = render_trail(
            max(1, int(round(width * scale))),
            [(x * scale, y * scale, radius * scale, alpha) for x, y, radius, alpha in points],
            thickness * scale, color, opacity, shadow, glow_intensity,
            glow_radius * scale, max(1, int(round(height * scale))),
        )
        return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_CUBIC)
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    thickness = min(max(1.0, float(thickness)), 30.0)
    op = min(1.0, max(0.0, float(opacity)))
    union = np.zeros((height, width), dtype=np.float32)
    point_alpha = np.zeros((height, width), dtype=np.float32)
    for x, y, radius, strength in points[-10:]:
        r = min(360.0, max(3.0, float(radius)))
        px, py = center_x + x, center_y + y
        x0, x1 = max(0, int(px - r - 1)), min(width, int(px + r + 2))
        y0, y1 = max(0, int(py - r - 1)), min(height, int(py + r + 2))
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        coverage = np.clip(r + 0.5 - np.hypot(xx - px, yy - py), 0.0, 1.0)
        region = union[y0:y1, x0:x1]
        np.maximum(region, coverage, out=region)
        np.maximum(
            point_alpha[y0:y1, x0:x1],
            coverage * min(1.0, max(0.0, float(strength))),
            out=point_alpha[y0:y1, x0:x1],
        )

    alpha = np.zeros((height, width), dtype=np.float32)

    # Blur a filled union once per glow scale; subtracting the shape keeps its
    # interior clear and avoids drawing glow seams where trail circles overlap.
    if shadow:
        glow = min(1.0, max(0.0, float(glow_intensity)))
        sigma = max(1.0, float(glow_radius))
        for blur_width, strength in ((sigma * 0.5, 0.30), (sigma * 1.6, 0.16)):
            blurred = cv2.GaussianBlur(
                point_alpha, (0, 0), sigmaX=blur_width, sigmaY=blur_width,
                borderType=cv2.BORDER_CONSTANT,
            )
            layer = np.maximum(blurred - point_alpha, 0.0)
            layer *= union < 0.01
            layer *= glow * strength * op
            alpha += layer * (1.0 - alpha)

    radius = max(1, int(math.ceil(thickness / 2.0)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    expanded = cv2.dilate(union, kernel)
    contracted = cv2.erode(union, kernel)
    rim = np.maximum(expanded - contracted, 0.0)
    rim *= cv2.dilate(point_alpha, kernel)
    alpha += rim * op * (1.0 - alpha)

    bgra = np.empty((height, width, 4), dtype=np.uint8)
    alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    bgra[..., 0] = (alpha * color[2]).astype(np.uint8)
    bgra[..., 1] = (alpha * color[1]).astype(np.uint8)
    bgra[..., 2] = (alpha * color[0]).astype(np.uint8)
    bgra[..., 3] = alpha_u8
    return bgra


class GazeOverlay:
    def __init__(
        self,
        radius: float = 30.0,
        thickness: float = 3.0,
        color: tuple[int, int, int] = (123, 97, 255),
        opacity: float = 1.0,
        shadow: bool = True,
        glow_intensity: float = 0.72,
        glow_radius: float = 16.0,
        trail_speed_scale: float = 0.65,
    ):
        self._hwnd: wt.HWND | None = None
        self._hdc_mem: wt.HDC | None = None
        self._hdc_screen: wt.HDC | None = None
        self._hbitmap: wt.HBITMAP | None = None
        self._bits = ctypes.c_void_p()
        self._bits_np: np.ndarray | None = None
        self._size = (0, 0)
        self._alpha = 1.0
        self._last_animation = 0.0
        self.exclude_from_capture = False
        self.visible = False
        self._pos = (-10000, -10000)
        self._trail: list[tuple[float, float, float, float]] | None = None
        self.configure(
            radius, thickness, color, opacity, shadow,
            glow_intensity, glow_radius, trail_speed_scale,
        )

    # ---------------------------------------------------------- pixels

    def configure(
        self,
        radius: float,
        thickness: float,
        color: tuple[int, int, int],
        opacity: float,
        shadow: bool,
        glow_intensity: float = 0.72,
        glow_radius: float = 16.0,
        trail_speed_scale: float = 0.65,
    ) -> None:
        self.radius = min(120.0, max(3.0, float(radius)))
        self.thickness = min(30.0, max(1.0, float(thickness)))
        self.color = tuple(int(c) for c in color)
        self.opacity = min(1.0, max(0.0, float(opacity)))
        self.shadow = bool(shadow)
        self.glow_intensity = min(1.0, max(0.0, float(glow_intensity)))
        self.glow_radius = min(100.0, max(0.0, float(glow_radius)))
        self.trail_speed_scale = min(2.0, max(0.0, float(trail_speed_scale)))
        if self._trail is not None:
            self.set_trail(self._trail)
            self._image = self._render_current_trail()
        else:
            self._image = self._render()
        size = (self._image.shape[1], self._image.shape[0])
        if self._hwnd is not None and size != self._size:
            self._rebuild(size)
        else:
            self._upload_image()
            if self.visible and self._hwnd is not None:
                self._blit()

    def _render(self) -> np.ndarray:
        pad = 6 + (self.glow_radius * 1.8 if self.shadow else 0)
        size = max(128, int(math.ceil(
            2.0 * (self.radius + self.glow_radius * 1.8 + self.thickness + pad)
        )))
        return render_trail(
            size, [(0.0, 0.0, self.radius, 1.0)], self.thickness,
            self.color, self.opacity, self.shadow, self.glow_intensity,
            self.glow_radius,
        )

    # ---------------------------------------------------------- window

    def _create_window(self, size: tuple[int, int]) -> None:
        width, height = size
        self._size = size
        ex = WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
        hinst = kernel32.GetModuleHandleW(None)
        hwnd = user32.CreateWindowExW(
            ex, "STATIC", "GazeDot", WS_POPUP,
            0, 0, width, height, None, None, hinst, None,
        )
        if not hwnd:
            raise RuntimeError(f"CreateWindowExW failed ({ctypes.get_last_error()})")
        self._hwnd = hwnd
        if self.exclude_from_capture:
            self.set_capture_excluded(True)

        self._hdc_screen = user32.GetDC(None)
        self._hdc_mem = gdi32.CreateCompatibleDC(self._hdc_screen)
        bmi = _BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        bmi.biWidth = width
        bmi.biHeight = -height  # top-down
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = 0
        hbm = gdi32.CreateDIBSection(
            self._hdc_screen, ctypes.byref(bmi), DIB_RGB_COLORS,
            ctypes.byref(self._bits), None, 0,
        )
        if not hbm:
            raise RuntimeError("CreateDIBSection failed")
        self._hbitmap = hbm
        gdi32.SelectObject(self._hdc_mem, hbm)
        self._bits_np = np.ctypeslib.as_array(
            ctypes.cast(self._bits, ctypes.POINTER(ctypes.c_ubyte)),
            shape=(width * height * 4,),
        )

        user32.SetWindowPos(
            hwnd, HWND_TOPMOST, 0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
        )

    def _rebuild(self, size: tuple[int, int]) -> None:
        was_visible = self.visible
        self._destroy_window()
        self._create_window(size)
        self._upload_image()
        if was_visible:
            self.show(*self._pos)

    def _destroy_window(self) -> None:
        if self._hwnd is not None:
            user32.DestroyWindow(self._hwnd)
            self._hwnd = None
        if self._hbitmap is not None:
            gdi32.DeleteObject(self._hbitmap)
            self._hbitmap = None
        if self._hdc_mem is not None:
            gdi32.DeleteDC(self._hdc_mem)
            self._hdc_mem = None
        if self._hdc_screen is not None:
            user32.ReleaseDC(None, self._hdc_screen)
            self._hdc_screen = None
        self._bits_np = None
        self._size = (0, 0)

    def _ensure_window(self) -> None:
        if self._hwnd is None:
            self._create_window((self._image.shape[1], self._image.shape[0]))

    def _upload_image(self) -> None:
        if self._hwnd is None:
            return
        if self._bits_np is None:
            return
        data = np.ascontiguousarray(self._image).reshape(-1)
        self._bits_np[:] = data

    # ---------------------------------------------------------- placement

    def move(self, x: float, y: float) -> None:
        if self._hwnd is None:
            return
        self._pos = (int(round(x)), int(round(y)))
        self._blit()

    def set_trail(self, points: list[tuple[float, float, float, float]]) -> None:
        self._trail = points[-10:]
        if not self._trail:
            return
        pad = self.radius * (1.0 + self.trail_speed_scale) + self.glow_radius * 2.0 + self.thickness
        xs = [p[0] for p in self._trail]
        ys = [p[1] for p in self._trail]
        center_x = (min(xs) + max(xs)) / 2.0
        center_y = (min(ys) + max(ys)) / 2.0
        required_width = max(128, int(math.ceil((max(xs) - min(xs) + 2 * pad) / 64.0) * 64))
        required_height = max(128, int(math.ceil((max(ys) - min(ys) + 2 * pad) / 64.0) * 64))
        current_width, current_height = self._size
        if current_width and required_width <= current_width and current_width - required_width < 128:
            required_width = current_width
        if current_height and required_height <= current_height and current_height - required_height < 128:
            required_height = current_height
        width = min(768, required_width)
        height = min(768, required_height)
        self._pos = (round(center_x), round(center_y))
        if (width, height) != self._size:
            self._image = self._render_current_trail(width, height)
            if self._hwnd is not None:
                self._rebuild((width, height))

    def clear_trail(self) -> None:
        self._trail = None
        self._image = self._render()
        if self._hwnd is not None:
            size = (self._image.shape[1], self._image.shape[0])
            if size != self._size:
                self._rebuild(size)
            elif self.visible:
                self._upload_image()
                self._blit()

    def _render_current_trail(
        self, width: int | None = None, height: int | None = None
    ) -> np.ndarray:
        if width is None:
            width = self._size[0] or max(128, self._image.shape[1])
        if height is None:
            height = self._size[1] or max(128, self._image.shape[0])
        if self._trail is None:
            return self._render()
        local = [
            (x - self._pos[0], y - self._pos[1], radius, alpha)
            for x, y, radius, alpha in self._trail
        ]
        return render_trail(
            width, local, self.thickness, self.color, self.opacity, self.shadow,
            self.glow_intensity, self.glow_radius, height,
        )

    def animate(self, now: float, max_hz: int = 30) -> None:
        interval = 1.0 / max(10, min(60, int(max_hz)))
        if not self.visible or now - self._last_animation < interval:
            return
        self._last_animation = now
        self._image = self._render_current_trail()
        self._upload_image()
        self._blit()

    def set_alpha(self, alpha: float) -> None:
        alpha = max(0.0, min(1.0, float(alpha)))
        if abs(alpha - self._alpha) < 0.004:
            return
        self._alpha = alpha
        if self._hwnd is not None and self.visible:
            self._blit()

    def set_capture_excluded(self, excluded: bool) -> None:
        excluded = bool(excluded)
        if self._hwnd is None:
            if not excluded:
                self.exclude_from_capture = False
                return
            self._ensure_window()
        if excluded == self.exclude_from_capture:
            return
        affinity = 0x00000011 if excluded else 0
        if not user32.SetWindowDisplayAffinity(self._hwnd, affinity):
            raise RuntimeError(
                "Windows が画面キャプチャ除外を設定できませんでした "
                f"({ctypes.get_last_error()})"
            )
        self.exclude_from_capture = excluded

    def _blit(self) -> None:
        assert self._hwnd is not None and self._hdc_mem is not None and self._hdc_screen is not None
        width, height = self._size
        dst = wt.POINT(self._pos[0] - width // 2, self._pos[1] - height // 2)
        src = wt.POINT(0, 0)
        sz = wt.SIZE(width, height)
        blend = _BLENDFUNCTION(AC_SRC_OVER, 0, int(self._alpha * 255), AC_SRC_ALPHA)
        ok = user32.UpdateLayeredWindow(
            self._hwnd, self._hdc_screen, ctypes.byref(dst), ctypes.byref(sz),
            self._hdc_mem, ctypes.byref(src), 0, ctypes.byref(blend), ULW_ALPHA,
        )
        if not ok:
            raise RuntimeError(f"UpdateLayeredWindow failed ({ctypes.get_last_error()})")

    def show(self, x: float, y: float) -> None:
        self._ensure_window()
        self._last_animation = 0.0
        self._upload_image()
        if self._trail is None:
            self._pos = (int(round(x)), int(round(y)))
        self._blit()
        user32.ShowWindow(self._hwnd, SW_SHOWNOACTIVATE)
        self.visible = True

    def hide(self) -> None:
        if self._hwnd is not None and self.visible:
            user32.ShowWindow(self._hwnd, SW_HIDE)
        self.visible = False

    def close(self) -> None:
        self._destroy_window()
