"""Camera capture thread: always exposes the newest frame, never queues."""

from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import numpy as np

from .config import CameraConfig


class Camera:
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._ts = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self._fps = 0.0

    # ------------------------------------------------------------ lifecycle

    def open(self) -> None:
        last_err = ""
        # DirectShow first: virtual cameras (Iriun, OBS Virtual Camera, ...) often
        # open fine under MSMF but hand out solid black frames.
        for api, name in ((cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "MSMF")):
            cap = cv2.VideoCapture(self.cfg.index, api)
            if not cap.isOpened():
                cap.release()
                last_err = f"{name} で開けません"
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
            cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self.cfg.fps >= 50:
                # MJPG keeps 60fps webcams within USB bandwidth at 640x480
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            ok, frame = cap.read()
            if ok and frame is not None:
                self._cap = cap
                self._store(frame)
                return
            cap.release()
            last_err = f"{name} で映像を取得できません"
        raise RuntimeError(
            f"カメラ {self.cfg.index} を開けませんでした ({last_err}). "
            "他のアプリ(Beam / OBS / Teams など)がカメラを使用していないか確認してください。"
        )

    def start(self) -> None:
        if self._thread is not None:
            return
        if self._cap is None:
            self.open()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def close(self) -> None:
        self.stop()

    # ------------------------------------------------------------ data

    def latest(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            return self._frame, self._ts

    @property
    def fps(self) -> float:
        return self._fps

    # ------------------------------------------------------------ worker

    def _store(self, frame: np.ndarray) -> None:
        if self.cfg.mirror:
            frame = cv2.flip(frame, 1)
        with self._lock:
            self._frame = frame
            self._ts = time.perf_counter()

    def _loop(self) -> None:
        last = time.perf_counter()
        fails = 0
        stamps: deque[float] = deque(maxlen=31)
        while not self._stop.is_set():
            cap = self._cap
            if cap is None:
                break
            iter_start = time.perf_counter()
            ok, frame = cap.read()
            now = time.perf_counter()
            if not ok or frame is None:
                fails += 1
                if fails > 45:  # ~1.5s of dead camera: try to recover
                    fails = 0
                    try:
                        cap.release()
                        self._cap = None
                        time.sleep(0.4)
                        self.open()
                    except Exception as exc:  # keep the app alive
                        self.error = str(exc)
                        time.sleep(1.0)
                else:
                    time.sleep(0.01)
                continue
            fails = 0
            dt = iter_start - last
            last = iter_start

            # Rate over a sliding window: an EMA gets poisoned by the initial
            # burst of already-buffered frames some drivers hand out.
            stamps.append(now)
            if len(stamps) >= 2:
                span = stamps[-1] - stamps[0]
                if span > 1e-3:
                    self._fps = (len(stamps) - 1) / span
            self._store(frame)

            # Some UVC drivers deliver 120fps regardless of the request; cap it.
            budget = 1.0 / max(1, self.cfg.fps) - dt
            if budget > 0.001:
                time.sleep(budget)
