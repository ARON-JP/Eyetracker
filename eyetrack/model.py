"""Per-user gaze regression + One Euro smoothing."""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass

import numpy as np

ALPHA_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def _solve(design: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    reg = alpha * np.eye(design.shape[1])
    reg[-1, -1] = 0.0  # never penalise the bias
    return np.linalg.solve(design.T @ design + reg, design.T @ target)


def _pixel_rmse(
    design: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    coef: np.ndarray,
    alpha: float,
    screen_w: int,
    screen_h: int,
) -> float:
    """Leave-one-point-out error, converted to screen pixels."""
    scale = np.array([float(screen_w), float(screen_h)])
    sq: list[float] = []
    for g in np.unique(groups):
        test = groups == g
        if test.sum() < 2 or (~test).sum() <= design.shape[1]:
            continue
        w = _solve(design[~test], target[~test], alpha)
        pred = design[test] @ w
        d = (pred - target[test]) * scale
        sq.extend(np.sum(d * d, axis=1).tolist())

    if sq:
        return float(np.sqrt(np.mean(sq)))

    # Too few groups to cross-validate: fall back to the training residual.
    pred = (design @ coef) * scale
    truth = target * scale
    return float(np.sqrt(np.mean(np.sum((pred - truth) ** 2, axis=1))))


@dataclass
class GazeModel:
    """Ridge regression: feature vector -> normalised screen position."""

    coef: np.ndarray  # (n_features + 1, 2)
    mu: np.ndarray
    sd: np.ndarray
    alpha: float = 1.0
    rmse_px: float = 0.0
    monitor: int = 0
    samples: int = 0
    created: str = ""
    pose_anchor: tuple[float, float, float] | None = None
    screen_correction: np.ndarray | None = None

    # ---------------------------------------------------------- fitting

    @staticmethod
    def fit(
        X: np.ndarray,
        Y: np.ndarray,
        groups: np.ndarray,
        monitor: int = 0,
        screen_w: int = 1920,
        screen_h: int = 1080,
        pose_anchor: tuple[float, float, float] | None = None,
    ) -> "GazeModel":
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-6] = 1.0
        Xs = (X - mu) / sd
        ones = np.ones((Xs.shape[0], 1))

        # Pick the regularisation with the lowest leave-one-target-out error.
        alpha = 1.0
        uniq = np.unique(groups)
        if len(uniq) >= 3:
            best = math.inf
            for cand in ALPHA_GRID:
                err = 0.0
                count = 0
                for g in uniq:
                    test = groups == g
                    if test.sum() < 2 or (~test).sum() <= Xs.shape[1]:
                        continue
                    w = _solve(
                        np.hstack([Xs[~test], ones[~test]]), Y[~test], cand
                    )
                    pred = np.hstack([Xs[test], ones[test]]) @ w
                    err += float(np.mean((pred - Y[test]) ** 2))
                    count += 1
                if count:
                    err /= count
                    if err < best:
                        best = err
                        alpha = cand

        coef = _solve(np.hstack([Xs, ones]), Y, alpha)

        # Honest pixel error: leave one calibration point out at a time, so the
        # number is what the dot will actually do on unseen gaze positions.
        # (Training residual would look far too good.)
        rmse = _pixel_rmse(
            np.hstack([Xs, ones]), Y, groups, coef, alpha, screen_w, screen_h
        )

        return GazeModel(
            coef=coef,
            mu=mu,
            sd=sd,
            alpha=alpha,
            rmse_px=rmse,
            monitor=monitor,
            samples=int(X.shape[0]),
            created=_dt.datetime.now().isoformat(timespec="seconds"),
            pose_anchor=pose_anchor,
        )

    # ---------------------------------------------------------- inference

    def predict(self, vector: np.ndarray) -> np.ndarray:
        z = (vector - self.mu) / self.sd
        out = float(np.dot(z, self.coef[:-1, 0]) + self.coef[-1, 0]), float(
            np.dot(z, self.coef[:-1, 1]) + self.coef[-1, 1]
        )
        return np.clip(np.array(out, dtype=np.float64), 0.0, 1.0)

    def pose_adjusted(
        self, vector: np.ndarray, current_baseline: tuple[float, float, float] | None
    ) -> np.ndarray:
        """Express pose features relative to the calibration posture."""
        if self.pose_anchor is None or current_baseline is None:
            return vector
        adjusted = np.asarray(vector, dtype=np.float64).copy()
        adjusted[11:14] += (
            np.asarray(self.pose_anchor) - np.asarray(current_baseline)
        ) / 30.0
        return adjusted

    def apply_screen_correction(self, position: np.ndarray) -> np.ndarray:
        if self.screen_correction is None:
            return position
        corrected = np.asarray(position, dtype=np.float64).copy()
        corrected[0] = corrected[0] * self.screen_correction[0, 0] + self.screen_correction[0, 1]
        corrected[1] = corrected[1] * self.screen_correction[1, 0] + self.screen_correction[1, 1]
        return np.clip(corrected, 0.0, 1.0)

    # ---------------------------------------------------------- storage

    def to_dict(self) -> dict:
        raw = {
            "version": 1,
            "created": self.created,
            "alpha": self.alpha,
            "coef": self.coef.tolist(),
            "mu": self.mu.tolist(),
            "sd": self.sd.tolist(),
            "rmse_px": self.rmse_px,
            "monitor": self.monitor,
            "samples": self.samples,
        }
        if self.pose_anchor is not None:
            raw["pose_anchor"] = list(self.pose_anchor)
        if self.screen_correction is not None:
            raw["screen_correction"] = self.screen_correction.tolist()
        return raw

    @classmethod
    def from_dict(cls, raw: dict) -> "GazeModel":
        anchor = raw.get("pose_anchor")
        correction = raw.get("screen_correction")
        correction_array = (
            None if correction is None else np.asarray(correction, dtype=np.float64)
        )
        anchor_tuple = None if anchor is None else tuple(float(v) for v in anchor)
        if anchor_tuple is not None and (
            len(anchor_tuple) != 3 or not np.isfinite(anchor_tuple).all()
        ):
            raise ValueError("invalid pose anchor")
        if correction_array is not None and (
            correction_array.shape != (2, 2)
            or not np.isfinite(correction_array).all()
        ):
            raise ValueError("invalid screen correction")
        return cls(
            coef=np.array(raw["coef"], dtype=np.float64),
            mu=np.array(raw["mu"], dtype=np.float64),
            sd=np.array(raw["sd"], dtype=np.float64),
            alpha=float(raw.get("alpha", 1.0)),
            rmse_px=float(raw.get("rmse_px", 0.0)),
            monitor=int(raw.get("monitor", 0)),
            samples=int(raw.get("samples", 0)),
            created=str(raw.get("created", "")),
            pose_anchor=anchor_tuple,
            screen_correction=correction_array,
        )


def reject_outliers(X: np.ndarray, groups: np.ndarray, k: float = 4.5) -> np.ndarray:
    """Drop blink/saccade contaminated samples inside each calibration point."""
    keep = np.ones(X.shape[0], dtype=bool)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        if idx.size < 6:
            continue
        sub = X[idx].astype(np.float64)
        med = np.median(sub, axis=0)
        mad = np.median(np.abs(sub - med), axis=0)
        dims = mad > 1e-6
        if not dims.any():
            continue
        dev = np.abs(sub[:, dims] - med[dims]) / (1.4826 * mad[dims])
        keep[idx[np.max(dev, axis=1) > k]] = False
    return keep


# ---------------------------------------------------------------- smoothing


class _LowPass:
    def __init__(self) -> None:
        self.value: np.ndarray | None = None

    def __call__(self, x: np.ndarray, alpha: float) -> np.ndarray:
        if self.value is None:
            self.value = x.copy()
        else:
            self.value += alpha * (x - self.value)
        return self.value


class OneEuro:
    """Jitter-free when still, snappy when the eye moves fast."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last: np.ndarray | None = None
        self._last_t: float | None = None

    def reset(self) -> None:
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last = None
        self._last_t = None

    def configure(self, responsiveness: int) -> None:
        """responsiveness: 0 (raw) .. 100 (heavy smoothing)."""
        s = min(100.0, max(0.0, float(responsiveness))) / 100.0
        self.min_cutoff = math.exp(math.log(4.0) * (1 - s) + math.log(0.4) * s)
        self.beta = 2.5 * (1 - s) + 0.05 * s

    def filter(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._last_t is None:
            dt = 1.0 / 60.0
        else:
            dt = t - self._last_t
            if dt <= 1e-4:
                dt = 1e-4
        freq = 1.0 / dt

        rate = 2.0 * math.pi * self.d_cutoff
        a_d = 1.0 / (1.0 + rate / freq)
        raw_d = np.zeros_like(x) if self._last is None else (x - self._last) * freq
        d = self._dx(raw_d, a_d)
        cutoff = self.min_cutoff + self.beta * np.abs(d)
        a = 1.0 / (1.0 + (2.0 * math.pi * cutoff) / freq)
        out = self._x(x, a).copy()

        self._last = x
        self._last_t = t
        return out
