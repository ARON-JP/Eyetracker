"""MediaPipe face-landmark worker thread."""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .camera import Camera
from .features import FaceFeatures, extract


@dataclass
class Sample:
    t: float = 0.0
    features: np.ndarray | None = None
    face: bool = False
    blink: bool = False
    ear: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    points: np.ndarray | None = None
    fps: float = 0.0
    frame: np.ndarray | None = None


class Tracker(threading.Thread):
    def __init__(self, camera: Camera, model_path: Path):
        super().__init__(name="tracker", daemon=True)
        self.camera = camera
        self.model_path = Path(model_path)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._sample = Sample()
        self._blink = False
        self._ts_ms = 0
        self._last_frame_ts = -1.0
        self._fps = 0.0
        self.error: str | None = None
        self.frames = 0

    # ---------------------------------------------------------- public

    def snapshot(self) -> Sample:
        with self._lock:
            return self._sample

    @property
    def fps(self) -> float:
        return self._fps

    def stop(self) -> None:
        self._stop.set()

    def reset_blink(self) -> None:
        self._blink = False

    # ---------------------------------------------------------- worker

    def _make_landmarker(self) -> vision.FaceLandmarker:
        # MediaPipe cannot open non-ASCII paths on Windows, so feed it bytes.
        with open(self.model_path, "rb") as fh:
            model_bytes = fh.read()
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def run(self) -> None:
        landmarker = None
        try:
            landmarker = self._make_landmarker()
        except Exception as exc:  # missing / broken model file
            self.error = f"顔モデルを読み込めません: {exc}"
            return

        last_t = time.perf_counter()
        stamps: deque[float] = deque(maxlen=31)
        try:
            while not self._stop.is_set():
                frame, ts = self.camera.latest()
                if frame is None or ts == self._last_frame_ts:
                    time.sleep(0.002)
                    continue
                self._last_frame_ts = ts
                ok, result = self._infer(landmarker, frame, ts)
                if not ok:
                    # rebuild once, transients must not kill tracking
                    try:
                        landmarker.close()
                    except Exception:
                        pass
                    time.sleep(0.2)
                    try:
                        landmarker = self._make_landmarker()
                    except Exception as exc:
                        self.error = str(exc)
                        return
                    continue

                now = time.perf_counter()
                stamps.append(now)
                if len(stamps) >= 2:
                    span = stamps[-1] - stamps[0]
                    if span > 1e-3:
                        self._fps = (len(stamps) - 1) / span
                self.frames += 1
        finally:
            try:
                landmarker.close()
            except Exception:
                pass

    def _infer(self, landmarker, frame: np.ndarray, ts: float) -> tuple[bool, None]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        try:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._ts_ms = max(self._ts_ms + 1, int(ts * 1000.0))
            result = landmarker.detect_for_video(image, self._ts_ms)
        except Exception as exc:
            self.error = str(exc)
            return False, None

        if result.face_landmarks:
            lm = np.array(
                [(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype=np.float32
            )
            feats: FaceFeatures = extract(
                lm, w, h, self._blink, self.camera.cfg.eye_mode
            )
            self._blink = feats.blink
            sample = Sample(
                t=ts,
                features=feats.vector,
                face=True,
                blink=feats.blink,
                ear=feats.ear,
                yaw=feats.yaw,
                pitch=feats.pitch,
                roll=feats.roll,
                points=feats.points,
                fps=self._fps,
                frame=frame,
            )
        else:
            self._blink = False
            sample = Sample(t=ts, face=False, fps=self._fps, frame=frame)

        with self._lock:
            self._sample = sample
        return True, None
