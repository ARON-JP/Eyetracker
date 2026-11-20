"""Landmark -> gaze feature extraction (iris ratios + head pose)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

# MediaPipe Face Landmarker indices.
# Group A = eye on the image-left side (subject's right eye), group B = other eye.
EYE_A = {"outer": 33, "inner": 133, "top": 159, "bottom": 145, "iris": 468}
EYE_B = {"outer": 362, "inner": 263, "top": 386, "bottom": 374, "iris": 473}

EYE_A_OUTLINE = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
EYE_B_OUTLINE = (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 388, 466)

PREVIEW_POINTS = (*EYE_A_OUTLINE, *EYE_B_OUTLINE, 468, 473, 1, 152, 61, 291, 199, 10)

# Rough generic face model (millimetres), y-down like the image plane, z forward.
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # 1   nose tip
        (0.0, 63.6, -12.5),  # 152 chin
        (-43.3, -32.7, -26.0),  # 33  right eye outer (image left)
        (43.3, -32.7, -26.0),  # 263 left eye outer
        (-28.9, 28.9, -24.1),  # 61  mouth right
        (28.9, 28.9, -24.1),  # 291 mouth left
        (0.0, -80.0, -18.0),  # 10  top of forehead
    ],
    dtype=np.float64,
)
_MODEL_INDEX = (1, 152, 33, 263, 61, 291, 10)

N_FEATURES = 17

BLINK_CLOSE = 0.135
BLINK_OPEN = 0.185


@dataclass
class FaceFeatures:
    vector: np.ndarray
    ear: float
    yaw: float
    pitch: float
    roll: float
    blink: bool = False
    points: np.ndarray | None = field(default=None, repr=False)


def _eye_ratios(lm: np.ndarray, eye: dict[str, int], w: int, h: int) -> tuple[float, float, float]:
    """Return (iris_x, iris_y, eye aspect ratio) in eye-local units."""
    cx = 0.5 * (lm[eye["outer"]][0] + lm[eye["inner"]][0])
    cy = 0.5 * (lm[eye["top"]][1] + lm[eye["bottom"]][1])
    ex = (lm[eye["inner"]][0] - lm[eye["outer"]][0]) * w
    ey = (lm[eye["bottom"]][1] - lm[eye["top"]][1]) * h
    width = abs(ex) + 1e-6
    height = abs(ey) + 1e-6
    ix = (lm[eye["iris"]][0] - cx) * w / width
    iy = (lm[eye["iris"]][1] - cy) * h / height
    ear = height / width
    return ix, iy, ear


def _expand(ix: float, iy: float) -> tuple[float, ...]:
    return (ix, iy, ix * iy, ix * ix, iy * iy)


def head_pose(lm: np.ndarray, w: int, h: int) -> tuple[float, float, float]:
    img = np.array(
        [(lm[i][0] * w, lm[i][1] * h) for i in _MODEL_INDEX], dtype=np.float64
    )
    focal = 0.87 * w
    cam = np.array([[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]])
    ok, rvec, _ = cv2.solvePnP(
        _MODEL_POINTS, img, cam, None, flags=cv2.SOLVEPNP_SQPNP
    )
    if not ok:
        ok, rvec, _ = cv2.solvePnP(_MODEL_POINTS, img, cam, None)
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.hypot(rmat[0, 0], rmat[1, 0])
    pitch = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
    yaw = math.degrees(math.atan2(-rmat[2, 0], sy))
    roll = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
    return yaw, pitch, roll


def extract(
    lm: np.ndarray, w: int, h: int, blink_state: bool, eye_mode: str = "both"
) -> FaceFeatures:
    """Build the regression feature vector from 478 normalised landmarks."""
    ax, ay, ear_a = _eye_ratios(lm, EYE_A, w, h)
    bx, by, ear_b = _eye_ratios(lm, EYE_B, w, h)
    ear = 0.5 * (ear_a + ear_b)
    blink = blink_state
    if ear < BLINK_CLOSE:
        blink = True
    elif ear > BLINK_OPEN:
        blink = False

    yaw, pitch, roll = head_pose(lm, w, h)

    cx, cy = lm[1][0], lm[1][1]
    scale = math.hypot(
        (lm[263][0] - lm[33][0]) * w, (lm[263][1] - lm[33][1]) * h
    ) / max(w, 1)

    eye_a, eye_b = _expand(ax, ay), _expand(bx, by)
    if eye_mode == "left":
        eye_a = eye_b
    elif eye_mode == "right":
        eye_b = eye_a

    vector = np.array(
        [
            1.0,
            *eye_a,
            *eye_b,
            yaw / 30.0,
            pitch / 30.0,
            roll / 30.0,
            cx,
            cy,
            scale,
        ],
        dtype=np.float32,
    )
    return FaceFeatures(
        vector=vector,
        ear=ear,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        blink=blink,
        points=lm[list(PREVIEW_POINTS), :2],
    )
