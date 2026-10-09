"""Grid pose <-> model-world mapping and the rodent-eye camera model.

Everything here mirrors blairlab-fusion-capture/CaptureView/CaptureView.py so
that a MuJoCo render of pose (x, y, dir) lines up with the Fusion render that
produced corner-maze-rl's pose_visual lookup.

Conventions
-----------
* Grid coordinates are MiniGrid coordinates (x right, y down, 13x13 grid).
* Model world frame is the Fusion design frame, converted to meters, Z up.
* Model X comes from grid y, model Y from grid x (CaptureView's anchor table).
* MiniGrid heading: 0 = east (+grid x), 1 = south (+grid y), 2 = west, 3 = north.
  In the capture script east = model +Y and north = model -X, hence
  dir 0 -> +Y, 1 -> +X, 2 -> -Y, 3 -> -X.

Where the grid sits in the world is a `GridFrame`: cell-centre knots plus the
track-top height. `CAPTUREVIEW` is the frame of the Rendering Model (maze v2)
and the default for the module-level helpers. A maze built from another Fusion
design carries its own frame in its metadata, and the renderer uses that one,
so the eyes follow the track when mazes are swapped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

IN = 0.0254

# CaptureView anchors (inches). Linear interpolation between knots.
X_FROM_GRID_Y_IN = {1: 8.039, 2: 16.336, 6: 55.519, 10: 94.702, 11: 102.999}
Y_FROM_GRID_X_IN = {1: 6.02, 2: 14.317, 6: 53.50, 10: 92.683, 11: 100.98}

TRACK_TOP_IN = 33.577
EYE_RAISE_IN = 2.5
EYE_Z_M = (TRACK_TOP_IN + EYE_RAISE_IN) * IN
EYE_SEPARATION_M = 0.5 * IN
EYE_YAW_DEG = 50.0
EYE_PITCH_DEG = 15.0
EYE_VFOV_NOMINAL_DEG = 150.0  # value written into Fusion's Camera.viewAngle by CaptureView
# Effective vertical FOV of the Fusion renders, fitted 2026-10-07 by matching
# monitor silhouettes over six open-maze poses (mask IoU 0.88 at 121-122 deg,
# 0.07 at 150 deg). Fusion evidently does not honour 150 deg as the vertical FOV.
EYE_VFOV_DEG = 121.5
EYE_IMG = 128

# MiniGrid dir -> unit heading in model XY
HEADING_XY = {0: (0.0, 1.0), 1: (1.0, 0.0), 2: (0.0, -1.0), 3: (-1.0, 0.0)}
DIR_NAMES = {0: "e", 1: "s", 2: "w", 3: "n"}


def _interp(index: float, knots: dict) -> float:
    """Piecewise-linear through the knots, clamped outside them (as CaptureView does)."""
    ks = sorted(knots)
    if index <= ks[0]:
        return knots[ks[0]]
    if index >= ks[-1]:
        return knots[ks[-1]]
    for a, b in zip(ks, ks[1:]):
        if a <= index <= b:
            t = (index - a) / (b - a)
            return knots[a] * (1 - t) + knots[b] * t
    raise ValueError(index)


def _inv_interp(v: float, knots: dict) -> float:
    """Inverse of _interp; extrapolates linearly from the end segments outside the knots."""
    ks = sorted(knots)
    for a, b in zip(ks, ks[1:]):
        lo, hi = sorted((knots[a], knots[b]))
        if lo - 1e-6 <= v <= hi + 1e-6:
            return a + (v - knots[a]) / (knots[b] - knots[a]) * (b - a)
    a, b = (ks[0], ks[1]) if abs(v - knots[ks[0]]) < abs(v - knots[ks[-1]]) else (ks[-2], ks[-1])
    return a + (v - knots[a]) / (knots[b] - knots[a]) * (b - a)


def heading_angle(direction: int) -> float:
    hx, hy = HEADING_XY[direction]
    return math.atan2(hy, hx)


def quat_from_z_angle(theta: float) -> np.ndarray:
    return np.array([math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)])


@dataclass(frozen=True)
class GridFrame:
    """Where the 13x13 grid sits in the model world, in meters.

    `x_from_gy_m[gy]` is the model X of any cell with grid y = gy and
    `y_from_gx_m[gx]` the model Y of any cell with grid x = gx (CaptureView's
    axis swap). Cells between knots are linearly interpolated.
    """

    x_from_gy_m: dict
    y_from_gx_m: dict
    track_top_m: float
    eye_raise_m: float = EYE_RAISE_IN * IN

    @classmethod
    def from_inches(cls, x_from_gy_in: dict, y_from_gx_in: dict, track_top_in: float, eye_raise_in: float = EYE_RAISE_IN) -> "GridFrame":
        return cls({int(k): float(v) * IN for k, v in x_from_gy_in.items()},
                   {int(k): float(v) * IN for k, v in y_from_gx_in.items()},
                   float(track_top_in) * IN, float(eye_raise_in) * IN)

    @classmethod
    def from_meta(cls, d: dict) -> "GridFrame":
        return cls({int(k): float(v) for k, v in d["x_from_grid_y_m"].items()},
                   {int(k): float(v) for k, v in d["y_from_grid_x_m"].items()},
                   float(d["track_top_m"]), float(d.get("eye_raise_m", EYE_RAISE_IN * IN)))

    def to_meta(self) -> dict:
        return {"x_from_grid_y_m": {str(k): round(v, 6) for k, v in sorted(self.x_from_gy_m.items())},
                "y_from_grid_x_m": {str(k): round(v, 6) for k, v in sorted(self.y_from_gx_m.items())},
                "track_top_m": round(self.track_top_m, 6), "eye_raise_m": round(self.eye_raise_m, 6)}

    @property
    def eye_z_m(self) -> float:
        return self.track_top_m + self.eye_raise_m

    def grid_to_model_m(self, gx: float, gy: float) -> np.ndarray:
        """Grid cell centre -> model XY in meters."""
        return np.array([_interp(gy, self.x_from_gy_m), _interp(gx, self.y_from_gx_m)])

    def model_to_grid(self, mx_m: float, my_m: float) -> tuple[float, float]:
        """Inverse of grid_to_model_m (piecewise linear, extrapolated outside the knots)."""
        return _inv_interp(my_m, self.y_from_gx_m), _inv_interp(mx_m, self.x_from_gy_m)

    def center_xy_m(self) -> np.ndarray:
        return self.grid_to_model_m(6, 6)

    def translated(self, d) -> "GridFrame":
        d = np.asarray(d, dtype=float)
        return GridFrame({k: v + d[0] for k, v in self.x_from_gy_m.items()},
                         {k: v + d[1] for k, v in self.y_from_gx_m.items()},
                         self.track_top_m + d[2], self.eye_raise_m)

    def head_pose(self, gx: float, gy: float, direction: int) -> tuple[np.ndarray, np.ndarray]:
        """World position and quaternion of the head (mocap) body for a grid pose."""
        xy = self.grid_to_model_m(gx, gy)
        return np.array([xy[0], xy[1], self.eye_z_m]), quat_from_z_angle(heading_angle(direction))


# Frame of the Rendering Model (maze v2), as used by CaptureView for the Fusion renders.
CAPTUREVIEW = GridFrame.from_inches(X_FROM_GRID_Y_IN, Y_FROM_GRID_X_IN, TRACK_TOP_IN)


def grid_to_model_m(gx: float, gy: float, frame: GridFrame = CAPTUREVIEW) -> np.ndarray:
    """Grid cell centre -> model XY in meters (same table as CaptureView by default)."""
    return frame.grid_to_model_m(gx, gy)


def model_to_grid(mx_m: float, my_m: float, frame: GridFrame = CAPTUREVIEW) -> tuple[float, float]:
    """Inverse of grid_to_model_m."""
    return frame.model_to_grid(mx_m, my_m)


def rotmat_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> (w, x, y, z), MuJoCo order."""
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def camera_quat_from_forward(forward: np.ndarray, up_hint=(0.0, 0.0, 1.0)) -> np.ndarray:
    """MuJoCo camera looks along its -Z with +Y up. Build quat from a forward vector."""
    f = np.asarray(forward, float)
    f = f / np.linalg.norm(f)
    u = np.asarray(up_hint, float)
    u = u - np.dot(u, f) * f
    u = u / np.linalg.norm(u)
    r = np.cross(f, u)
    R = np.column_stack([r, u, -f])
    return rotmat_to_quat(R)


def eye_local_frames() -> dict:
    """Left/right eye position + orientation in the head frame (x forward, y left, z up)."""
    out = {}
    cp, sp = math.cos(math.radians(EYE_PITCH_DEG)), math.sin(math.radians(EYE_PITCH_DEG))
    for name, side, yaw_deg in (("eye_left", +1.0, +EYE_YAW_DEG), ("eye_right", -1.0, -EYE_YAW_DEG)):
        yaw = math.radians(yaw_deg)
        fwd = np.array([cp * math.cos(yaw), cp * math.sin(yaw), sp])
        pos = np.array([0.0, side * EYE_SEPARATION_M / 2, 0.0])
        out[name] = {"pos": pos, "quat": camera_quat_from_forward(fwd)}
    return out


def head_pose(gx: float, gy: float, direction: int, frame: GridFrame = CAPTUREVIEW) -> tuple[np.ndarray, np.ndarray]:
    """World position and quaternion of the head (mocap) body for a grid pose."""
    return frame.head_pose(gx, gy, direction)
