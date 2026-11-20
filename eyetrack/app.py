"""Application wiring: camera -> tracker -> regression -> overlay + panel."""

from __future__ import annotations

import math
import time
import tkinter as tk
from collections import deque
from tkinter import messagebox

import numpy as np

from .calibration import EDGE_TARGETS, CalibrationSession, ValidationSession, _targets_for
from .camera import Camera
from .config import (
    MODEL_PATH,
    Config,
    hex_to_rgb,
    read_profile,
    write_profile,
)
from .model import GazeModel, OneEuro, reject_outliers
from .overlay import GazeOverlay
from .panel import Panel
from .tracker import Tracker
from .win32 import HotkeyWatcher, enable_dpi_awareness, list_monitors


class GazeApp:
    def __init__(self, root: tk.Tk, cfg: Config):
        self.root = root
        self.cfg = cfg
        self.config = cfg
        self.monitors = list_monitors()
        self.cfg.monitor = max(0, min(cfg.monitor, len(self.monitors) - 1))

        self.model: GazeModel | None = None
        self.baseline = (0.0, 0.0, 0.0)
        self.baseline_ok = False
        self.calibrating = False
        self._quitting = False

        self._last_usable = 0.0
        self._last_px = (0.0, 0.0)
        self._ov_px: tuple[float, float] | None = None
        self._tracking_position: tuple[float, float] | None = None
        self._tracking_pose: tuple[float, float, float] | None = None
        self._pose_samples: deque[tuple[float, float, float, float]] = deque(maxlen=90)
        self._last_pose_sample_t = -1.0
        self._last_panel = 0.0
        self._last_preview = 0.0
        self._reported_error = None
        self._test_until = 0.0
        self._last_gaze_sample_t = -1.0
        self._last_prediction_t = 0.0
        self._trail_samples: deque[tuple[float, float, float, float]] = deque(maxlen=64)

        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"顔ランドマークモデルが見つかりません:\n{MODEL_PATH}\n"
                "setup.ps1 を実行して再取得してください。"
            )

        # input -------------------------------------------------------
        self.camera = Camera(cfg.camera)
        self.camera.start()
        self.tracker = Tracker(self.camera, MODEL_PATH)
        self.tracker.start()

        # output ------------------------------------------------------
        ov = cfg.overlay
        self.overlay = GazeOverlay(
            radius=ov.radius,
            thickness=ov.thickness,
            color=hex_to_rgb(ov.color),
            opacity=ov.opacity,
            shadow=ov.shadow,
            glow_intensity=ov.glow_intensity,
            glow_radius=ov.glow_radius,
            trail_speed_scale=ov.trail_speed_scale,
        )
        self.overlay.set_capture_excluded(ov.exclude_from_capture)
        self.filter = OneEuro()
        self.filter.configure(cfg.filter.responsiveness)
        self.head_filter = OneEuro()
        self.head_filter.configure(cfg.filter.head_responsiveness)

        # stored calibration -----------------------------------------
        self._load_profile_quiet(cfg.profile)

        # ui ----------------------------------------------------------
        self.panel = Panel(root, self)
        self._place_panel()
        # Layout settles a moment after the first map; re-park once so the
        # panel really touches the right edge.
        root.after(250, self._place_panel)
        self.hotkeys = HotkeyWatcher(cfg.hotkeys)
        root.protocol("WM_DELETE_WINDOW", self.quit)
        root.after(16, self._tick)

    def _place_panel(self) -> None:
        """Park the panel in the top-right corner of the primary monitor."""
        try:
            self.root.update_idletasks()
            w = self.root.winfo_width()
            h = self.root.winfo_height()
            if w <= 1 or h <= 1:
                w = self.root.winfo_reqwidth()
                h = self.root.winfo_reqheight()
            mon = next((m for m in self.monitors if m.primary), self.monitors[0])
            x = max(mon.x + 8, mon.x + mon.width - w - 24)
            y = mon.y + 24
            self.root.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            pass

    # ================================================================ loop

    def run(self) -> None:
        self.root.mainloop()

    def _tick(self) -> None:
        if self._quitting:
            return
        try:
            self._poll_hotkeys()
            if not self.calibrating:
                self._update_overlay()
                self.overlay.animate(time.perf_counter(), self.cfg.filter.update_hz)
            now = time.perf_counter()
            if now - self._last_panel > 0.15:
                self._last_panel = now
                self._refresh_panel()
            if (
                self.cfg.show_preview
                and now - self._last_preview > 0.1
                and not self.calibrating
            ):
                self._last_preview = now
                self.panel.update_preview(self.tracker.snapshot())
        except tk.TclError:
            self._quitting = True
            return
        except Exception:
            # never let the UI loop die
            import traceback

            traceback.print_exc()
        self.root.after(16, self._tick)

    def _refresh_panel(self) -> None:
        s = self.tracker.snapshot()
        if self.model is None:
            info = "未キャリブレーション（F9）"
        else:
            info = f"精度 {self.model.rmse_px:.0f} px"
        if (
            not s.face
            and s.frame is not None
            and float(s.frame.mean()) < 1.0
        ):
            info = "カメラ映像が真っ黒です（接続を確認）"
        if self.tracker.error and self.tracker.error != self._reported_error:
            self._reported_error = self.tracker.error
            info = "トラッカーエラー"
            messagebox.showerror(
                "GazeDot",
                f"顔トラッカーでエラーが発生しました:\n{self.tracker.error}",
                parent=self.root,
            )
        tracking = (
            f"視線: {self._tracking_position[0]:.0f}, {self._tracking_position[1]:.0f}"
            + (
                f" / 顔角度(Y/P/R): {self._tracking_pose[0]:.0f}°,"
                f" {self._tracking_pose[1]:.0f}°, {self._tracking_pose[2]:.0f}°"
                if self._tracking_pose is not None else ""
            )
            if self._tracking_position is not None
            else ("追跡待機中" if self.model is not None else "未キャリブレーション")
        )
        self.panel.refresh(bool(s.face), s.fps, info, tracking)

    def _poll_hotkeys(self) -> None:
        for action in self.hotkeys.poll():
            if action == "quit":
                self.quit()
                return
            if self.calibrating:
                continue
            if action == "calibrate":
                self.start_calibration(self.cfg.calibration_points)
            elif action == "reset_frontal":
                self.reset_frontal_baseline()
            elif action == "edge_calibration":
                self.start_edge_calibration()
            elif action == "toggle_overlay":
                self.cfg.show_overlay = not self.cfg.show_overlay
                self.on_config_changed()
                self.panel.sync_toggles()
            elif action == "toggle_preview":
                self.cfg.show_preview = not self.cfg.show_preview
                self.on_config_changed()
                self.panel.sync_toggles()
            elif action == "toggle_panel":
                self.panel.toggle()

    # ================================================================ gaze

    def _monitor(self):
        return self.monitors[min(self.cfg.monitor, len(self.monitors) - 1)]

    def _update_overlay(self) -> None:
        now = time.perf_counter()

        # "テスト表示" pins the bubble at screen centre for a few seconds so the
        # look can be checked in OBS before calibrating.
        if now < self._test_until:
            mon = self._monitor()
            self._place(
                mon.x + (mon.width - 1) / 2.0,
                mon.y + (mon.height - 1) / 2.0,
                1.0,
            )
            return
        if self._test_until:
            self._test_until = 0.0
            self._last_usable = now
            self._ov_px = None
            self.overlay.hide()

        s = self.tracker.snapshot()
        if s.face and not s.blink and s.t != self._last_pose_sample_t:
            self._pose_samples.append((s.t, s.yaw, s.pitch, s.roll))
            self._last_pose_sample_t = s.t

        if not self.cfg.show_overlay or self.model is None:
            self.overlay.hide()
            self._ov_px = None
            self._tracking_position = None
            self._tracking_pose = None
            return

        usable = bool(
            s.face and not s.blink and s.features is not None
            and now - s.t <= 0.25
        )

        if usable and self.baseline_ok:
            if (
                abs(s.yaw - self.baseline[0]) > self.cfg.filter.max_yaw
                or abs(s.pitch - self.baseline[1]) > self.cfg.filter.max_pitch
                or abs(s.roll - self.baseline[2]) > self.cfg.filter.max_roll
            ):
                usable = False

        if usable:
            min_interval = 1.0 / max(10, min(60, self.cfg.filter.update_hz))
            if (
                s.t <= self._last_gaze_sample_t
                or now - self._last_prediction_t < min_interval
            ):
                self._refresh_trail(now)
                return
            adjusted = self.model.pose_adjusted(s.features, self.baseline)
            adjusted = adjusted.copy()
            adjusted[11:14] = self.head_filter.filter(
                adjusted[11:14], s.t
            )
            predicted = self.model.predict(adjusted)
            nx, ny = self.model.apply_screen_correction(predicted)
            sm = self.filter.filter(np.array([nx, ny]), now)
            mon = self._monitor()
            px = mon.x + float(sm[0]) * (mon.width - 1)
            py = mon.y + float(sm[1]) * (mon.height - 1)
            px = min(max(px, mon.x + 1.0), mon.x + mon.width - 2.0)
            py = min(max(py, mon.y + 1.0), mon.y + mon.height - 2.0)

            dz = self.cfg.filter.deadzone_px
            if self._ov_px is not None and dz > 0 and math.hypot(
                px - self._ov_px[0], py - self._ov_px[1]
            ) < dz:
                px, py = self._ov_px

            self._last_px = (px, py)
            self._tracking_position = (px, py)
            self._tracking_pose = (s.yaw, s.pitch, s.roll)
            self._last_usable = now
            self._last_gaze_sample_t = s.t
            self._last_prediction_t = now
            previous = self._trail_samples[-1] if self._trail_samples else None
            speed = (
                0.0 if previous is None
                else math.hypot(px - previous[1], py - previous[2])
                / max(0.001, now - previous[0])
            )
            self._trail_samples.append((now, px, py, speed))
            self._refresh_trail(now)
            self._place(px, py, 1.0)
            return

        self._refresh_trail(now)
        self._tracking_position = None
        self._tracking_pose = None
        # blink / face lost: hold, dim, then hide
        silent = (now - self._last_usable) * 1000.0
        if self._ov_px is None or silent > self.cfg.overlay.hide_after_ms:
            self.overlay.hide()
            self._ov_px = None
            self._clear_trail()
            return
        alpha = 0.32 if silent > self.cfg.overlay.dim_after_ms else 1.0
        if not self._trail_samples:
            self.overlay.set_trail([
                (self._last_px[0], self._last_px[1], self.overlay.radius, 1.0)
            ])
        self._place(self._last_px[0], self._last_px[1], alpha)

    def _place(self, px: float, py: float, alpha: float) -> None:
        if not self.overlay.visible:
            self.overlay.show(px, py)
            self.overlay.set_alpha(alpha)
            self._ov_px = (round(px, 1), round(py, 1))
            return
        self._ov_px = (round(px, 1), round(py, 1))
        self.overlay.set_alpha(alpha)

    def _refresh_trail(self, now: float) -> None:
        lifetime = max(80, min(1500, self.cfg.overlay.trail_duration_ms)) / 1000.0
        while self._trail_samples and now - self._trail_samples[0][0] >= lifetime:
            self._trail_samples.popleft()
        scale = self.overlay.trail_speed_scale
        points = []
        for t, x, y, speed in self._trail_samples:
            age = max(0.0, now - t)
            fade = max(0.0, 1.0 - age / lifetime)
            speed_factor = min(1.0, speed / 1800.0)
            radius = self.overlay.radius * (1.0 + scale * speed_factor)
            points.append((x, y, radius, fade))
        self.overlay.set_trail(points)

    def _clear_trail(self) -> None:
        self._trail_samples.clear()
        self.overlay.clear_trail()

    # ================================================================ config

    def on_config_changed(self, reset_filter: bool = False) -> None:
        ov = self.cfg.overlay
        self.overlay.configure(
            ov.radius, ov.thickness, hex_to_rgb(ov.color),
            ov.opacity, ov.shadow, ov.glow_intensity,
            ov.glow_radius, ov.trail_speed_scale,
        )
        self.overlay.set_capture_excluded(ov.exclude_from_capture)
        self.filter.configure(self.cfg.filter.responsiveness)
        self.head_filter.configure(self.cfg.filter.head_responsiveness)
        if reset_filter:
            self.filter.reset()
            self.head_filter.reset()
        if not self.cfg.show_overlay:
            self.overlay.hide()
            self._ov_px = None
        self.cfg.save()

    def apply_camera_settings(
        self, index: int, width: int, height: int, fps: int, mirror: bool, eye_mode: str
    ) -> bool:
        camera_cfg = self.cfg.camera
        previous = (
            camera_cfg.index, camera_cfg.width, camera_cfg.height,
            camera_cfg.fps, camera_cfg.mirror, camera_cfg.eye_mode,
        )
        updated = (index, width, height, fps, mirror, eye_mode)
        if previous == updated:
            return True
        try:
            self.camera.stop()
            (
                camera_cfg.index, camera_cfg.width, camera_cfg.height,
                camera_cfg.fps, camera_cfg.mirror, camera_cfg.eye_mode,
            ) = updated
            self.camera.start()
        except Exception as exc:
            self.camera.stop()
            (
                camera_cfg.index, camera_cfg.width, camera_cfg.height,
                camera_cfg.fps, camera_cfg.mirror, camera_cfg.eye_mode,
            ) = previous
            try:
                self.camera.start()
            except Exception as restore_exc:
                raise RuntimeError(
                    f"カメラ設定に失敗しました: {exc}\n"
                    f"元のカメラも再接続できません: {restore_exc}"
                ) from restore_exc
            raise RuntimeError(f"カメラ設定を適用できませんでした: {exc}") from exc

        self.model = None
        self.filter.reset()
        self.head_filter.reset()
        self._ov_px = None
        self._clear_trail()
        self.overlay.hide()
        self.cfg.save()
        return True

    # ================================================================ calibration

    def start_calibration(self, points: int = 5) -> None:
        if self.calibrating:
            return
        self.calibrating = True
        self.overlay.hide()
        self._clear_trail()
        self._ov_px = None
        try:
            mon = self._monitor()
            rect = (mon.x, mon.y, mon.width, mon.height)
            data = CalibrationSession(
                self.root, self.tracker, rect, points,
                target_points=_targets_for(points),
            ).run()
            if data is None:
                return
            if len(np.unique(data.groups)) < 5:
                messagebox.showwarning(
                    "GazeDot",
                    "キャリブレーションが失敗しました。\n"
                    "すべてのマークをクリックしたか確認して、"
                    "もう一度お試しください。",
                    parent=self.root,
                )
                return

            X, Y, G = data.features, data.targets, data.groups
            keep = reject_outliers(X, G)
            X, Y, G = X[keep], Y[keep], G[keep]
            if X.shape[0] < 25 or len(np.unique(G)) < 5:
                messagebox.showwarning(
                    "GazeDot",
                    "有効なサンプルが足りませんでした。\n"
                    "明るい場所で、顔がはっきり映るようにして再試行してください。",
                    parent=self.root,
                )
                return

            model = GazeModel.fit(
                X, Y, G, self.cfg.monitor, mon.width, mon.height,
                pose_anchor=data.baseline if data.baseline_ok else None,
            )

            self.model = model
            self.baseline = data.baseline
            self.baseline_ok = data.baseline_ok
            self.filter.reset()
            self.head_filter.reset()
            self._ov_px = None
            self.save_profile(self.cfg.profile)

            diag = float(np.hypot(mon.width, mon.height))
            pct = model.rmse_px / diag * 100.0
            if pct < 2.0:
                verdict = "とても良好です。"
            elif pct < 4.0:
                verdict = "良好です。気になる場合は再キャリブレーションしてください。"
            else:
                verdict = (
                    "やや大きめです。明るい場所で、頭を動かさずに"
                    "再キャリブレーションしてください。"
                )
            messagebox.showinfo(
                "キャリブレーション完了",
                f"推定誤差（点ごとの交差検証）: {model.rmse_px:.0f} px"
                f"（画面対角の {pct:.1f}%）\n"
                f"{verdict}\n\n"
                "「精度チェック」で実際の追従を確認できます。",
                parent=self.root,
            )
        except Exception as exc:
            messagebox.showerror(
                "GazeDot", f"キャリブレーション中にエラー:\n{exc}", parent=self.root
            )
        finally:
            self.calibrating = False
            self._last_usable = time.perf_counter()
            self._ov_px = None

    def reset_frontal_baseline(self) -> None:
        if self.calibrating:
            return
        if self.model is None:
            messagebox.showwarning(
                "正面リセット",
                "先にキャリブレーションを完了してください。",
                parent=self.root,
            )
            return
        now = time.perf_counter()
        recent = [row for row in self._pose_samples if now - row[0] <= 1.5]
        pose_values = np.asarray([row[1:] for row in recent], dtype=np.float64)
        if (
            len(recent) < 8
            or now - recent[-1][0] > 0.35
            or not self.tracker.snapshot().face
            or np.any(
                np.percentile(pose_values, 90, axis=0)
                - np.percentile(pose_values, 10, axis=0)
                > (10.0, 10.0, 12.0)
            )
        ):
            messagebox.showwarning(
                "正面リセット",
                "顔を検出できていないか、安定した姿勢データが不足しています。\n"
                "カメラを見て正面を向き、少し待ってから再度実行してください。",
                parent=self.root,
            )
            return
        self.baseline = tuple(
            float(np.median([row[i] for row in recent])) for i in (1, 2, 3)
        )
        self.baseline_ok = True
        self.filter.reset()
        self.head_filter.reset()
        self._ov_px = None
        self._tracking_position = None
        self._tracking_pose = None
        if not self.save_profile(self.cfg.profile):
            messagebox.showerror(
                "正面リセット",
                "姿勢基準は更新しましたが、プロファイルを保存できませんでした。",
                parent=self.root,
            )
            return
        messagebox.showinfo(
            "正面リセット",
            "現在の顔の向きを新しい基準にしました。\n"
            "画面位置がずれている場合は「画面端を簡易補正」も実行してください。",
            parent=self.root,
        )

    def start_edge_calibration(self) -> None:
        if self.calibrating:
            return
        if self.model is None:
            messagebox.showwarning(
                "画面端を簡易補正",
                "先に通常のキャリブレーションを完了してください。",
                parent=self.root,
            )
            return
        self.calibrating = True
        self.overlay.hide()
        self._clear_trail()
        self._ov_px = None
        try:
            mon = self._monitor()
            rect = (mon.x, mon.y, mon.width, mon.height)
            data = CalibrationSession(
                self.root, self.tracker, rect, edge_only=True,
                baseline=self.baseline, baseline_ok=self.baseline_ok,
                target_points=EDGE_TARGETS,
            ).run()
            if data is None or len(np.unique(data.groups)) != 4:
                return

            measured: list[np.ndarray] = []
            targets: list[np.ndarray] = []
            for group in range(4):
                mask = data.groups == group
                if int(mask.sum()) < 6:
                    raise ValueError("各端点で十分な視線データを取得できませんでした。")
                values = []
                for features in data.features[mask]:
                    adjusted = self.model.pose_adjusted(features, self.baseline)
                    values.append(self.model.predict(adjusted))
                measured.append(np.median(np.asarray(values), axis=0))
                targets.append(np.median(data.targets[mask], axis=0))

            measured_array = np.asarray(measured)
            target_array = np.asarray(targets)
            if np.ptp(measured_array[:, 0]) < 0.15 or np.ptp(measured_array[:, 1]) < 0.15:
                raise ValueError(
                    "画面端の視線位置を十分に区別できませんでした。\n"
                    "頭を固定し、明るい場所で再度お試しください。"
                )

            correction = np.empty((2, 2), dtype=np.float64)
            for axis in range(2):
                design = np.column_stack((measured_array[:, axis], np.ones(4)))
                slope = float(
                    np.linalg.lstsq(design, target_array[:, axis], rcond=None)[0][0]
                )
                slope = min(2.0, max(0.5, slope))
                intercept = float(
                    np.mean(target_array[:, axis] - slope * measured_array[:, axis])
                )
                correction[axis] = (slope, intercept)

            self.model.screen_correction = correction
            self.filter.reset()
            self.head_filter.reset()
            self._ov_px = None
            if not self.save_profile(self.cfg.profile):
                messagebox.showerror(
                    "画面端を簡易補正",
                    "補正は適用しましたが、プロファイルを保存できませんでした。",
                    parent=self.root,
                )
                return
            messagebox.showinfo(
                "画面端を簡易補正",
                "画面端4点の補正を保存しました。",
                parent=self.root,
            )
        except Exception as exc:
            messagebox.showerror(
                "画面端を簡易補正",
                f"補正に失敗しました:\n{exc}",
                parent=self.root,
            )
        finally:
            self.calibrating = False
            self._last_usable = time.perf_counter()
            self._ov_px = None

    def start_validation(self) -> None:
        if self.calibrating or self.model is None:
            return
        self.calibrating = True
        self.overlay.hide()
        self._clear_trail()
        self._ov_px = None
        try:
            mon = self._monitor()
            rect = (mon.x, mon.y, mon.width, mon.height)
            result = ValidationSession(
                self.root, self.tracker, rect, self.model, self.baseline
            ).run()
            if result is None or not result.per_point:
                return
            diag = float(np.hypot(mon.width, mon.height))
            pct = result.rmse / diag * 100.0
            self.model.rmse_px = result.rmse
            self.save_profile(self.cfg.profile)
            messagebox.showinfo(
                "精度チェック",
                f"平均誤差: {result.rmse:.0f} px（画面対角の {pct:.1f}%）",
                parent=self.root,
            )
        finally:
            self.calibrating = False
            self._last_usable = time.perf_counter()
            self._ov_px = None

    def test_overlay(self) -> None:
        """Pin the bubble at screen centre for a few seconds (OBS check)."""
        if self.calibrating:
            return
        self.on_config_changed()
        self._clear_trail()
        mon = self._monitor()
        cx = mon.x + (mon.width - 1) / 2.0
        cy = mon.y + (mon.height - 1) / 2.0
        try:
            self.overlay.show(cx, cy)
            self.overlay.set_alpha(1.0)
        except Exception as exc:
            messagebox.showerror(
                "GazeDot", f"オーバーレイを表示できません:\n{exc}",
                parent=self.root,
            )
            return
        self._ov_px = (round(cx, 1), round(cy, 1))
        self._test_until = time.perf_counter() + 5.0

    # ================================================================ profiles

    def _load_profile_quiet(self, name: str) -> None:
        raw = read_profile(name)
        if not raw or "model" not in raw:
            return
        if raw.get("eye_mode", "both") != self.cfg.camera.eye_mode:
            return
        try:
            self.model = GazeModel.from_dict(raw["model"])
            self.baseline = tuple(raw.get("baseline", (0.0, 0.0, 0.0)))  # type: ignore[assignment]
            self.baseline_ok = bool(raw.get("baseline_ok", False))
            if self.model.pose_anchor is None:
                self.model.pose_anchor = self.baseline
        except Exception:
            self.model = None

    def load_profile(self, name: str) -> bool:
        raw = read_profile(name)
        if not raw or "model" not in raw:
            return False
        if raw.get("eye_mode", "both") != self.cfg.camera.eye_mode:
            return False
        try:
            model = GazeModel.from_dict(raw["model"])
        except Exception:
            return False
        self.model = model
        self.baseline = tuple(raw.get("baseline", (0.0, 0.0, 0.0)))  # type: ignore[assignment]
        self.baseline_ok = bool(raw.get("baseline_ok", False))
        if model.pose_anchor is None:
            model.pose_anchor = self.baseline
        self.cfg.profile = name
        self.filter.reset()
        self.head_filter.reset()
        self._ov_px = None
        return True

    def set_hotkey(self, action: str, key: str) -> None:
        self.cfg.hotkeys[action] = key
        self.hotkeys = HotkeyWatcher(self.cfg.hotkeys)
        self.cfg.save()

    def save_profile(self, name: str) -> bool:
        if self.model is None:
            return False
        try:
            write_profile(
                name,
                {
                    "version": 1,
                    "model": self.model.to_dict(),
                    "baseline": list(self.baseline),
                    "baseline_ok": self.baseline_ok,
                    "eye_mode": self.cfg.camera.eye_mode,
                },
            )
            self.cfg.profile = name
            self.cfg.save()
            return True
        except Exception:
            return False

    # ================================================================ teardown

    def quit(self) -> None:
        if self._quitting:
            return
        self._quitting = True
        try:
            self.cfg.save()
        except Exception:
            pass
        try:
            self.tracker.stop()
        except Exception:
            pass
        try:
            self.camera.stop()
        except Exception:
            pass
        try:
            self.overlay.close()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass


def main() -> None:
    enable_dpi_awareness()
    cfg = Config.load()
    root = tk.Tk()
    try:
        app = GazeApp(root, cfg)
    except Exception as exc:
        try:
            root.withdraw()
            messagebox.showerror("GazeDot", str(exc), parent=root)
        finally:
            root.destroy()
        return
    app.run()
