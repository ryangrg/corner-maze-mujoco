"""Stereo rodent-eye renderer with the corner-maze-rl pose-label contract.

    r = MazeRenderer("model/corner_maze.xml")
    left, right = r.render_pose(x, y, direction, cue=None, raised_barriers=())

Returns two (128, 128) uint8 grayscale images, left and right eye, matching
the layout of corner-maze-rl's pose_visual lookup. `cue` is a grid cue cell
such as (6, 1) or None; `raised_barriers` is an iterable of (x, y) grid cells
matching the env's BARRIER_LOCATIONS convention.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")
import mujoco  # noqa: E402

from . import pose as P  # noqa: E402


class MazeRenderer:
    def __init__(self, xml_path: str | Path, size: int = P.EYE_IMG):
        xml_path = Path(xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        meta_path = xml_path.with_name(xml_path.stem + "_meta.json")
        if not meta_path.exists():  # pre-2026-10-09 single-file builds
            meta_path = xml_path.parent / "model_meta.json"
        self.meta = json.loads(meta_path.read_text())
        # grid frame of the maze this model was composed with (eyes follow the track)
        self.frame = P.GridFrame.from_meta(self.meta["frame"]) if "frame" in self.meta else P.CAPTUREVIEW
        self.size = size
        self._renderer = mujoco.Renderer(self.model, height=size, width=size)
        self._head = self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "rat_head")]
        self._screen_geom = {w: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) for w, g in self.meta["screens"].items()}
        self._light = {w: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, n) for w, n in self.meta.get("lights", {}).items()}
        self._mat_blank = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "screen_blank")
        self._mat_cue = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, "screen_cue")
        self._barrier_q = {}
        for bm in self.meta["barriers"]:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, bm["name"])
            self._barrier_q[tuple(bm["grid"])] = self.model.jnt_qposadr[jid]
        self.travel = float(self.meta["barrier_travel_m"])

    # ---- state ---------------------------------------------------------------
    def set_barriers(self, raised: set[tuple[int, int]]):
        for cell, adr in self._barrier_q.items():
            self.data.qpos[adr] = self.travel if cell in raised else 0.0

    def set_cue(self, cue: tuple[int, int] | None):
        for wall, gid in self._screen_geom.items():
            on = cue is not None and wall == f"cue_x{cue[0]}_y{cue[1]}"
            self.model.geom_matid[gid] = self._mat_cue if on else self._mat_blank
            if wall in self._light:
                self.model.light_active[self._light[wall]] = 1 if on else 0

    def set_head(self, x: float, y: float, direction: int):
        pos, quat = self.frame.head_pose(x, y, direction)
        self.data.mocap_pos[self._head] = pos
        self.data.mocap_quat[self._head] = quat

    # ---- rendering ----------------------------------------------------------
    def render_eyes(self) -> tuple[np.ndarray, np.ndarray]:
        mujoco.mj_forward(self.model, self.data)
        out = []
        for cam in ("eye_left", "eye_right"):
            self._renderer.update_scene(self.data, camera=cam)
            rgb = self._renderer.render()
            gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).round().astype(np.uint8)
            out.append(gray)
        return out[0], out[1]

    def render_pose(self, x, y, direction, cue=None, raised_barriers=()):
        self.set_barriers(set(tuple(c) for c in raised_barriers))
        self.set_cue(cue)
        self.set_head(x, y, direction)
        return self.render_eyes()

    def render_camera(self, camera: str, size: int = 512) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        r = mujoco.Renderer(self.model, height=size, width=size)
        r.update_scene(self.data, camera=camera)
        return r.render()
