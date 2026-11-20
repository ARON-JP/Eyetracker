"""Configuration and profile storage (JSON, human editable)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = APP_ROOT / "config.json"
PROFILE_DIR = APP_ROOT / "profiles"
MODEL_PATH = APP_ROOT / "models" / "face_landmarker.task"


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    mirror: bool = True
    eye_mode: str = "both"


@dataclass
class OverlayConfig:
    radius: float = 30.0
    thickness: float = 3.0
    color: str = "#7B61FF"
    opacity: float = 1.0
    shadow: bool = True
    glow_intensity: float = 0.72
    glow_radius: float = 16.0
    trail_duration_ms: int = 360
    trail_speed_scale: float = 0.65
    exclude_from_capture: bool = False
    # look-away / blink behaviour
    dim_after_ms: int = 140
    hide_after_ms: int = 900


@dataclass
class FilterConfig:
    # 0 = raw / jittery, 100 = very smooth but laggy
    responsiveness: int = 35
    # ignore movements smaller than this (screen px); 0 disables
    deadzone_px: float = 1.8
    update_hz: int = 30
    head_responsiveness: int = 35
    # pause the dot when the head turns further than this (deg)
    max_yaw: float = 38.0
    max_pitch: float = 30.0
    max_roll: float = 35.0


@dataclass
class Config:
    monitor: int = 0
    profile: str = "default"
    calibration_points: int = 5
    show_overlay: bool = True
    show_preview: bool = False
    camera: CameraConfig = field(default_factory=CameraConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    hotkeys: dict[str, str] = field(
        default_factory=lambda: {
            "calibrate": "F9",
            "reset_frontal": "ALT+C",
            "edge_calibration": "F5",
            "toggle_overlay": "F10",
            "toggle_preview": "F8",
            "toggle_panel": "F7",
            "quit": "F12",
        }
    )

    # -------------------------------------------------- (de)serialisation

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        cfg = cls()
        for f in fields(cls):
            if f.name not in raw:
                continue
            value = raw[f.name]
            if f.name == "camera":
                value = CameraConfig(**{k: v for k, v in value.items() if k in _names(CameraConfig)})
            elif f.name == "overlay":
                value = OverlayConfig(**{k: v for k, v in value.items() if k in _names(OverlayConfig)})
            elif f.name == "filter":
                value = FilterConfig(**{k: v for k, v in value.items() if k in _names(FilterConfig)})
            elif f.name == "hotkeys":
                value = {**cfg.hotkeys, **value}
            setattr(cfg, f.name, value)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # -------------------------------------------------- file helpers

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Config":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return cls.from_dict(json.load(fh))
        except FileNotFoundError:
            return cls()
        except Exception:
            # a broken config must never stop the app
            return cls()

    def save(self, path: Path = CONFIG_PATH) -> None:
        try:
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception:
            pass


def _names(dc: type) -> set[str]:
    return {f.name for f in fields(dc)}


# ---------------------------------------------------------------- profiles


def profile_path(name: str) -> Path:
    safe = "".join(c for c in name if c.isalnum() or c in "-_ ").strip() or "default"
    return PROFILE_DIR / f"{safe}.json"


def list_profiles() -> list[str]:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(p.stem for p in PROFILE_DIR.glob("*.json"))


def read_profile(name: str) -> dict[str, Any] | None:
    try:
        with open(profile_path(name), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def write_profile(name: str, data: dict[str, Any]) -> None:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path = profile_path(name)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------- colour


def hex_to_rgb(
    value: str, fallback: tuple[int, int, int] = (123, 97, 255)
) -> tuple[int, int, int]:
    try:
        v = value.strip().lstrip("#")
        if len(v) == 3:
            v = "".join(c * 2 for c in v)
        if len(v) != 6:
            raise ValueError(value)
        return tuple(int(v[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except Exception:
        return fallback
