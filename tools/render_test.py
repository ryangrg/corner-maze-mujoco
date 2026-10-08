"""Compile the generated model, verify eye-camera orientation, render test views.

    PYTHONPATH=src python tools/render_test.py --model model/corner_maze.xml --out model/test_renders --pose 6 6 0
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")
import mujoco  # noqa: E402
from PIL import Image  # noqa: E402

from corner_maze_mujoco import pose as P  # noqa: E402


def expected_eye_forward(direction: int, eye: str) -> np.ndarray:
    th = P.heading_angle(direction) + math.radians(P.EYE_YAW_DEG if eye == "eye_left" else -P.EYE_YAW_DEG)
    cp, sp = math.cos(math.radians(P.EYE_PITCH_DEG)), math.sin(math.radians(P.EYE_PITCH_DEG))
    return np.array([cp * math.cos(th), cp * math.sin(th), sp])


def set_pose(model, data, gx, gy, direction):
    pos, quat = P.head_pose(gx, gy, direction)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rat_head")
    mid = model.body_mocapid[bid]
    data.mocap_pos[mid] = pos
    data.mocap_quat[mid] = quat
    mujoco.mj_forward(model, data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/corner_maze.xml")
    ap.add_argument("--out", default="model/test_renders")
    ap.add_argument("--pose", nargs=3, type=int, default=[6, 6, 0])
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--raise", dest="raise_", default="", help="comma-separated barrier names to raise")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model = mujoco.MjModel.from_xml_path(a.model)
    data = mujoco.MjData(model)
    print(f"compiled in {time.time()-t0:.1f}s: nbody={model.nbody} ngeom={model.ngeom} nmesh={model.nmesh} "
          f"nmeshvert={model.nmeshvert} nmeshface={model.nmeshface} ncam={model.ncam} nq={model.nq}")

    for name in [s for s in a.raise_.split(",") if s]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = 16 * P.IN
    gx, gy, d = a.pose
    set_pose(model, data, gx, gy, d)

    # camera orientation check
    for eye in ("eye_left", "eye_right"):
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, eye)
        R = data.cam_xmat[cid].reshape(3, 3)
        fwd = -R[:, 2]
        up = R[:, 1]
        exp = expected_eye_forward(d, eye)
        print(f"{eye}: pos={np.round(data.cam_xpos[cid],4)} forward={np.round(fwd,3)} expected={np.round(exp,3)} "
              f"err={np.linalg.norm(fwd-exp):.2e} up_z={up[2]:.3f}")

    r = mujoco.Renderer(model, height=a.size, width=a.size)
    for eye in ("eye_left", "eye_right"):
        r.update_scene(data, camera=eye)
        img = r.render()
        Image.fromarray(img).save(out / f"{eye}_x{gx}_y{gy}_d{d}.png")
        Image.fromarray(img).convert("L").save(out / f"{eye}_x{gx}_y{gy}_d{d}_gray.png")
    r2 = mujoco.Renderer(model, height=768, width=768)
    r2.update_scene(data, camera="overhead")
    Image.fromarray(r2.render()).save(out / "overhead.png")
    # oblique view of the head position from behind/above
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = data.mocap_pos[0]
    cam.distance = 1.2
    cam.azimuth = math.degrees(P.heading_angle(d)) + 180
    cam.elevation = -25
    r2.update_scene(data, camera=cam)
    Image.fromarray(r2.render()).save(out / "oblique.png")
    t1 = time.time()
    n = 50
    for _ in range(n):
        r.update_scene(data, camera="eye_left"); r.render()
    print(f"eye render: {(time.time()-t1)/n*1000:.1f} ms/frame at {a.size}px")
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
