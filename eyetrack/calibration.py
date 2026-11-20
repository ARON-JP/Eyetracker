"""Full-screen calibration / validation UI (Tk, no extra dependencies).

Calibration is click driven: the user clicks the lit mark, holds their gaze
on it while the ring fills, and the next mark lights up automatically.
Centre first (that is where the head-pose baseline comes from), followed by
the four corners or the remaining eight positions.
"""

from __future__ import annotations

import math
import statistics
import time
import tkinter as tk
from dataclasses import dataclass, field

import numpy as np

from .model import GazeModel
from .tracker import Sample, Tracker

BG = "#0b0d10"
FG = "#dceaf5"
ACCENT = "#9b85ff"
OK = "#3ddc84"
DIM = "#243240"
SOFT = "#7f93a5"
WARN = "#ffb020"

TICK_MS = 20
SETTLE_S = 0.12
COLLECT_S = 0.80
MIN_SAMPLES = 6
GATE_YAW = 28.0
GATE_PITCH = 22.0
GATE_ROLL = 24.0

MARK_R = 30.0
MARK_HIT = 60.0

# Centre first, then the four corners: (fx, fy, label, short label)
CAL_POINTS_5 = [
    (0.5, 0.5, "中央", "まん中"),
    (0.025, 0.025, "左上", "左上"),
    (0.975, 0.025, "右上", "右上"),
    (0.025, 0.975, "左下", "左下"),
    (0.975, 0.975, "右下", "右下"),
]

VALID_POINTS = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85), (0.5, 0.5)]

VALID_MOVE_S = 0.35
VALID_COLLECT_S = 0.80
EDGE_POINTS = [
    (0.50, 0.04, "上端", "上"),
    (0.96, 0.50, "右端", "右"),
    (0.50, 0.96, "下端", "下"),
    (0.04, 0.50, "左端", "左"),
]
EDGE_TARGETS = [
    (0.50, 0.0),
    (1.0, 0.50),
    (0.50, 1.0),
    (0.0, 0.50),
]


def _points_for(count: int) -> list[tuple[float, float, str, str]]:
    """Return the click targets; the centre is always collected first."""
    if count <= 5:
        return list(CAL_POINTS_5)
    corners = [(fx, fy) for fy in (0.025, 0.5, 0.975) for fx in (0.025, 0.5, 0.975)]
    corners.remove((0.5, 0.5))
    ordered = [(0.5, 0.5)] + corners
    return [(fx, fy, f"{i + 1}", f"{i + 1}") for i, (fx, fy) in enumerate(ordered)]


def _targets_for(count: int) -> list[tuple[float, float]]:
    if count <= 5:
        return [(0.5, 0.5), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    corners = [(fx, fy) for fy in (0.0, 0.5, 1.0) for fx in (0.0, 0.5, 1.0)]
    corners.remove((0.5, 0.5))
    return [(0.5, 0.5)] + corners


@dataclass
class CalibrationData:
    features: np.ndarray
    targets: np.ndarray
    groups: np.ndarray
    baseline: tuple[float, float, float]
    baseline_ok: bool
    rect: tuple[int, int, int, int]
    counts: list[int] = field(default_factory=list)


@dataclass
class ValidationData:
    rmse: float
    per_point: list[tuple[tuple[float, float], float, int]]
    rect: tuple[int, int, int, int]


class _Screen:
    """Shared full-screen canvas plumbing."""

    def __init__(
        self, root: tk.Misc, tracker: Tracker, rect: tuple[int, int, int, int]
    ):
        self.tracker = tracker
        self.rect = rect
        self.result = None
        self._closed = False
        self._after_id = None

        x, y, w, h = rect
        self.w, self.h = w, h
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.geometry(f"{w}x{h}{x:+d}{y:+d}")
        self.win.configure(bg=BG)
        self.win.attributes("-topmost", True)
        self.canvas = tk.Canvas(
            self.win, width=w, height=h, bg=BG, highlightthickness=0, bd=0
        )
        self.canvas.pack(fill="both", expand=True)

        self._title = self.canvas.create_text(
            w // 2, 46, text="", fill=FG, font=("Meiryo UI", 17)
        )
        self._hint = self.canvas.create_text(
            w // 2,
            h - 40,
            text="点をクリック ・ 頭は動かさない ・ ESC でキャンセル",
            fill=SOFT,
            font=("Meiryo UI", 11),
        )
        self._count = self.canvas.create_text(
            w - 30, 28, text="", fill=SOFT, anchor="ne", font=("Meiryo UI", 11)
        )
        self._warn = self.canvas.create_text(
            w // 2, h // 2 + 120, text="", fill=WARN, font=("Meiryo UI", 16)
        )
        self._arc = self.canvas.create_arc(
            0, 0, 0, 0, start=90, extent=0, style="arc",
            outline=OK, width=5, state="hidden",
        )

        self._last_ok_t = time.perf_counter()
        self._black_check_t = 0.0
        self._black = False
        self._msg = ""
        self._msg_until = 0.0

        self.win.bind("<Escape>", lambda _e: self._cancel())
        self.win.protocol("WM_DELETE_WINDOW", self._cancel)

    # ------------------------------------------------------------ helpers

    def _note_status(self, s: Sample, now: float) -> None:
        """Face-lost / black-frame warning shared by both flows."""
        if s.face and not s.blink:
            self._last_ok_t = now

        if now - self._black_check_t > 0.4:
            self._black_check_t = now
            frame = s.frame
            self._black = (
                frame is not None
                and not s.face
                and float(frame.mean()) < 1.0
            )

        if now < self._msg_until:
            text = self._msg
        elif self._black and not s.face:
            text = "カメラ映像が真っ黒です（接続を確認）"
        elif now - self._last_ok_t > 2.0:
            text = "顔を検出できません"
        else:
            text = ""
        self.canvas.itemconfigure(self._warn, text=text)

    def _flash(self, text: str, seconds: float = 3.5) -> None:
        """Show a warning that survives the per-frame status refresh."""
        self._msg = text
        self._msg_until = time.perf_counter() + seconds
        self.canvas.itemconfigure(self._warn, text=text)

    def _clear_flash(self) -> None:
        self._msg = ""
        self._msg_until = 0.0
        self.canvas.itemconfigure(self._warn, text="")

    def _arc_progress(self, cx: float, cy: float, radius: float, progress: float) -> None:
        self.canvas.coords(self._arc, cx - radius, cy - radius, cx + radius, cy + radius)
        self.canvas.itemconfigure(
            self._arc,
            extent=-359.9 * max(0.0, min(1.0, progress)),
            state="normal",
        )

    def _hide_arc(self) -> None:
        self.canvas.itemconfigure(self._arc, state="hidden")

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._after_id is not None:
            try:
                self.win.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        try:
            self.win.grab_release()
        except Exception:
            pass
        try:
            self.win.destroy()
        except Exception:
            pass

    def _cancel(self) -> None:
        # result stays None unless the session already finished; this way a
        # stray ESC during the completion flash cannot discard good data.
        self._close()

    # ------------------------------------------------------------ loop

    def run(self):
        self.win.lift()
        self.win.focus_force()
        try:
            self.win.grab_set()
        except Exception:
            pass
        self._begin()
        self._after_id = self.win.after(TICK_MS, self._tick)
        self.win.wait_window()
        return self.result

    def _begin(self) -> None: ...

    def _tick(self) -> None:
        if self._closed:
            return
        try:
            self._step()
        except tk.TclError:
            self._closed = True
            return
        if not self._closed:
            self._after_id = self.win.after(TICK_MS, self._tick)

    def _step(self) -> None: ...


class CalibrationSession(_Screen):
    """Click each calibration mark, then hold gaze to collect samples."""

    def __init__(
        self, root: tk.Misc, tracker: Tracker, rect: tuple[int, int, int, int],
        points: int = 5,
        edge_only: bool = False,
        baseline: tuple[float, float, float] | None = None,
        baseline_ok: bool = False,
        target_points: list[tuple[float, float]] | None = None,
    ):
        super().__init__(root, tracker, rect)
        self.edge_only = edge_only
        self.points = list(EDGE_POINTS) if edge_only else _points_for(points)
        self.target_points = target_points
        self.win.attributes("-topmost", True)
        self.canvas.configure(cursor="hand2")
        if edge_only:
            self.canvas.itemconfigure(
                self._title, text="画面端を順にクリックして見つめてください"
            )
            self.canvas.itemconfigure(
                self._hint, text="正面を向いて頭を固定 ・ ESC でキャンセル"
            )

        self._marks: list[int] = []
        self._labels: list[int] = []
        self._done = [False] * len(self.points)
        self._idx = 0
        self._phase = "idle"  # idle | settle | collect | done
        self._t0 = 0.0
        self._samples: list[tuple[np.ndarray, float, float, float]] = []
        self._counts: list[int] = []
        self._gate_active = baseline_ok
        self.baseline = baseline or (0.0, 0.0, 0.0)
        self.baseline_ok = baseline_ok

        for i, (fx, fy, label, _short) in enumerate(self.points):
            ring = self.canvas.create_oval(
                0, 0, 0, 0, outline=DIM, width=2, state="normal"
            )
            text = self.canvas.create_text(
                0, 0, text=label, fill=SOFT, font=("Meiryo UI", 10)
            )
            self._marks.append(ring)
            self._labels.append(text)
            self._layout_mark(i)

        self.X: list[np.ndarray] = []
        self.Y: list[np.ndarray] = []
        self.G: list[int] = []

        self.canvas.bind("<Button-1>", self._on_click)
        self._refresh_marks()
        self.canvas.itemconfigure(
            self._title,
            text=(
                "画面端の点をクリックして開始してください"
                if self.edge_only else "まん中の点をクリックして開始してください"
            ),
        )
        self.canvas.itemconfigure(self._count, text=f"1 / {len(self.points)}")

    # ------------------------------------------------------------ drawing

    def _layout_mark(self, i: int) -> None:
        fx, fy, _label, _short = self.points[i]
        cx, cy = fx * self.w, fy * self.h
        if self._done[i]:
            r, outline, width = 7.0, OK, 2
        elif i == self._idx and self._phase != "done":
            r, outline, width = MARK_R, ACCENT, 4
        else:
            r, outline, width = 13.0, DIM, 2
        self.canvas.coords(
            self._marks[i], cx - r, cy - r, cx + r, cy + r
        )
        self.canvas.itemconfigure(
            self._marks[i], outline=outline, width=width,
            fill="" if i != self._idx or self._phase == "done" else "#123039",
        )
        label_x = min(max(cx, 48.0), self.w - 48.0)
        label_y = cy - MARK_R - 18 if fy > 0.85 else cy + MARK_R + 18
        self.canvas.coords(self._labels[i], label_x, label_y)

    def _refresh_marks(self) -> None:
        for i in range(len(self.points)):
            self.canvas.itemconfigure(
                self._labels[i],
                text="✓" if self._done[i] else self.points[i][3],
                fill=OK if self._done[i] else (
                    ACCENT if i == self._idx else SOFT
                ),
            )
            self._layout_mark(i)

    def _pulse(self, now: float) -> None:
        if self._phase != "idle" or self._idx >= len(self.points):
            return
        fx, fy, _label, _short = self.points[self._idx]
        cx, cy = fx * self.w, fy * self.h
        r = MARK_R + 5.0 * math.sin(now * 4.0)
        self.canvas.coords(
            self._marks[self._idx], cx - r, cy - r, cx + r, cy + r
        )

    # ------------------------------------------------------------ flow

    def _valid(self, s: Sample) -> bool:
        if not s.face or s.blink or s.features is None:
            return False
        if not self._gate_active or self._idx == 0:
            return True
        return (
            abs(s.yaw - self.baseline[0]) <= GATE_YAW
            and abs(s.pitch - self.baseline[1]) <= GATE_PITCH
            and abs(s.roll - self.baseline[2]) <= GATE_ROLL
        )

    def _on_click(self, event: tk.Event) -> None:
        if self._closed or self._phase != "idle" or self._idx >= len(self.points):
            return
        fx, fy, _label, _short = self.points[self._idx]
        cx, cy = fx * self.w, fy * self.h
        if math.hypot(event.x - cx, event.y - cy) > MARK_HIT:
            return
        self._samples = []
        self._phase = "settle"
        self._t0 = time.perf_counter()
        self._last_ok_t = self._t0
        self.canvas.itemconfigure(
            self._title, text="そのまま点を見つめてください"
        )
        self._clear_flash()
        self._hide_arc()

    def _begin(self) -> None:
        self._phase = "idle"
        self._refresh_marks()

    def _step(self) -> None:
        now = time.perf_counter()
        s = self.tracker.snapshot()
        self._note_status(s, now)

        if self._phase == "idle":
            self._pulse(now)
            return

        if self._phase == "settle":
            elapsed = now - self._t0
            fx, fy, _label, _short = self.points[self._idx]
            cx, cy = fx * self.w, fy * self.h
            self._arc_progress(cx, cy, MARK_R + 16, elapsed / SETTLE_S)
            if elapsed >= SETTLE_S:
                self._phase = "collect"
                self._t0 = now
                self._samples = []
            return

        if self._phase == "collect":
            elapsed = now - self._t0
            if self._valid(s):
                self._samples.append((s.features, s.yaw, s.pitch, s.roll))
            fx, fy, _label, _short = self.points[self._idx]
            cx, cy = fx * self.w, fy * self.h
            self._arc_progress(cx, cy, MARK_R + 16, elapsed / COLLECT_S)
            if elapsed >= COLLECT_S:
                self._complete_point()
            return

    def _complete_point(self) -> None:
        self._hide_arc()
        label = self.points[self._idx][2]
        if len(self._samples) < MIN_SAMPLES:
            self._phase = "idle"
            self._t0 = 0.0
            self.canvas.itemconfigure(
                self._title,
                text=f"{label}の点をもう一度クリックしてください",
            )
            self._flash(
                "サンプル不足：顔をはっきり映して、点を見たまま"
                "もう一度クリック"
            )
            return

        self._store_point()
        self._done[self._idx] = True
        self._idx += 1
        self._phase = "idle"
        self._t0 = 0.0
        self._clear_flash()

        if self._idx >= len(self.points):
            self._finish()
            return

        nxt = self.points[self._idx][2]
        self.canvas.itemconfigure(
            self._title, text=f"{nxt}の点をクリックしてください"
        )
        self.canvas.itemconfigure(
            self._count, text=f"{self._idx + 1} / {len(self.points)}"
        )
        self._refresh_marks()

    def _store_point(self) -> None:
        fx, fy, _label, _short = self.points[self._idx]
        if self.target_points is not None:
            fx, fy = self.target_points[self._idx]
        gid = self._idx
        for feats, _yaw, _pitch, _roll in self._samples:
            self.X.append(feats)
            self.Y.append(np.array((fx, fy), dtype=np.float64))
            self.G.append(gid)
        self._counts.append(len(self._samples))

        if gid == 0 and not self.edge_only:
            self.baseline = (
                statistics.median(v[1] for v in self._samples),
                statistics.median(v[2] for v in self._samples),
                statistics.median(v[3] for v in self._samples),
            )
            self.baseline_ok = len(self._samples) >= 5
            self._gate_active = self.baseline_ok

    def _finish(self) -> None:
        self._phase = "done"
        self._hide_arc()
        self._done = [True] * len(self.points)
        self._refresh_marks()
        self.canvas.itemconfigure(
            self._title, text="キャリブレーション完了"
        )
        self.canvas.itemconfigure(self._count, text="")
        self.canvas.itemconfigure(self._hint, text="")
        self._on_finished()
        self.win.after(420, self._close)

    def _on_finished(self) -> None:
        if not self.X:
            self.result = None
            return
        self.result = CalibrationData(
            features=np.array(self.X, dtype=np.float64),
            targets=np.array(self.Y, dtype=np.float64),
            groups=np.array(self.G, dtype=np.int32),
            baseline=self.baseline,
            baseline_ok=self.baseline_ok,
            rect=self.rect,
            counts=self._counts,
        )


class ValidationSession(_Screen):
    """Automatic five-point accuracy check for an existing model."""

    def __init__(
        self, root: tk.Misc, tracker: Tracker,
        rect: tuple[int, int, int, int], model: GazeModel,
        baseline: tuple[float, float, float] | None = None,
    ):
        super().__init__(root, tracker, rect)
        self.model = model
        self.baseline = baseline
        self._points = VALID_POINTS
        self._idx = 0
        self._phase = "move"
        self._t0 = time.perf_counter()
        self._samples: list[np.ndarray] = []
        self._errs: list[list[float]] = [[] for _ in self._points]
        self._counts: list[int] = []

        self._ring = self.canvas.create_oval(
            0, 0, 0, 0, outline=ACCENT, width=5, state="hidden"
        )
        self._dot = self.canvas.create_oval(
            0, 0, 0, 0, fill=ACCENT, outline="", state="hidden"
        )
        self.canvas.itemconfigure(
            self._title, text="精度チェック：点を見つめてください"
        )
        self.canvas.itemconfigure(self._count, text=f"1 / {len(self._points)}")

    def _target(self, radius: float, color: str, progress: float | None) -> None:
        fx, fy = self._points[self._idx]
        cx, cy = fx * self.w, fy * self.h
        self.canvas.coords(self._ring, cx - radius, cy - radius, cx + radius, cy + radius)
        self.canvas.itemconfigure(
            self._ring, outline=color, width=max(2, int(radius * 0.24)),
            state="normal",
        )
        r = max(2.0, radius * 0.16)
        self.canvas.coords(self._dot, cx - r, cy - r, cx + r, cy + r)
        self.canvas.itemconfigure(self._dot, fill=color, state="normal")
        if progress is None:
            self._hide_arc()
        else:
            self._arc_progress(cx, cy, radius + 16, progress)

    def _begin(self) -> None:
        self._phase = "move"
        self._t0 = time.perf_counter()

    def _step(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._t0
        s = self.tracker.snapshot()
        self._note_status(s, now)

        if self._phase == "move":
            p = min(1.0, elapsed / VALID_MOVE_S)
            self._target(30 - 12 * p, ACCENT, None)
            if elapsed >= VALID_MOVE_S:
                self._samples = []
                self._phase = "collect"
                self._t0 = now
            return

        if self._phase == "collect":
            if s.face and not s.blink and s.features is not None:
                self._samples.append(s.features)
            p = min(1.0, elapsed / VALID_COLLECT_S)
            self._target(18, ACCENT, p)
            if elapsed >= VALID_COLLECT_S:
                self._store_point()
                self._phase = "flash"
                self._t0 = now
            return

        if self._phase == "flash":
            if elapsed >= 0.14:
                self._idx += 1
                if self._idx >= len(self._points):
                    self._finish()
                else:
                    self._phase = "move"
                    self._t0 = now
                    self.canvas.itemconfigure(
                        self._count, text=f"{self._idx + 1} / {len(self._points)}"
                    )
            else:
                self._target(18, OK, 1.0)

    def _store_point(self) -> None:
        fx, fy = self._points[self._idx]
        tx, ty = fx * self.w, fy * self.h
        for feats in self._samples:
            adjusted = self.model.pose_adjusted(feats, self.baseline)
            nx, ny = self.model.apply_screen_correction(self.model.predict(adjusted))
            dx = nx * self.w - tx
            dy = ny * self.h - ty
            self._errs[self._idx].append(math.hypot(dx, dy))
        self._counts.append(len(self._samples))

    def _finish(self) -> None:
        self.canvas.itemconfigure(self._ring, state="hidden")
        self.canvas.itemconfigure(self._dot, state="hidden")
        self._hide_arc()
        self.canvas.itemconfigure(self._title, text="精度チェック完了")
        self.canvas.itemconfigure(self._count, text="")
        self.canvas.itemconfigure(self._hint, text="")

        per_point = []
        sq: list[float] = []
        for (fx, fy), errs in zip(self._points, self._errs):
            if not errs:
                continue
            per_point.append(((fx, fy), float(np.mean(errs)), len(errs)))
            sq.extend(errs)
        rmse = float(np.sqrt(np.mean(np.square(sq)))) if sq else 999.0
        self.result = ValidationData(rmse=rmse, per_point=per_point, rect=self.rect)
        self._phase = "done"
        self.win.after(420, self._close)
