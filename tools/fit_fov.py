"""Fit the effective Fusion field of view and check for stray light.

Open-maze (expa) Fusion renders contain only the blank monitors, so a bright
mask (>30/255) isolates them. We sweep MuJoCo fovy and score mask IoU per pose.
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image, ImageFilter

from corner_maze_mujoco.layouts import parse_pose_label
from corner_maze_mujoco.renderer import MazeRenderer


def mask(img, thr=30, blur=1.0):
    g = np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(blur)), float) if blur else img.astype(float)
    return g > thr


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/corner_maze.xml")
    ap.add_argument("--npz", default="../corner-maze-rl/data/lookups/pose_visual.npz")
    ap.add_argument("--labels", nargs="+", default=["expa_x_x_xx_6_6_0", "expa_x_x_xx_6_3_3", "expa_x_x_xx_2_6_2", "expa_x_x_xx_6_8_1", "expa_x_x_xx_10_10_0", "expa_x_x_xx_2_2_3"])
    ap.add_argument("--fovs", nargs="+", type=float, default=[90, 100, 110, 115, 120, 125, 130, 140, 150])
    a = ap.parse_args()
    d = np.load(a.npz, allow_pickle=False)
    labels = list(d["pose_labels"]); idx = d["pose_to_idx"]; L, R = d["left_eyes"], d["right_eyes"]
    r = MazeRenderer(a.model)
    m = r.model

    # --- stray light test: force all non-screen materials black, render expa centre
    keep = {r._mat_blank, r._mat_cue}
    saved = m.mat_rgba.copy()
    for i in range(m.nmat):
        if i not in keep:
            m.mat_rgba[i, :3] = 0
    ml, mr = r.render_pose(6, 6, 0)
    print(f"all-black materials, expa centre: mean={ml.mean():.2f} nonzero={np.count_nonzero(ml)} of {ml.size}")
    m.mat_rgba[:] = saved
    ml, mr = r.render_pose(6, 6, 0)
    nm = ~mask(ml, thr=30, blur=0)
    print(f"normal materials, expa centre: mean={ml.mean():.2f}; mean outside bright mask={ml[nm].mean():.2f}; headlight active={m.vis.headlight.active} ambient={m.vis.headlight.ambient}")

    # --- FOV sweep
    cams = [i for i in range(m.ncam) if m.camera(i).name in ("eye_left", "eye_right")]
    fov0 = m.cam_fovy.copy()
    rows = []
    for fov in a.fovs:
        for ci in cams:
            m.cam_fovy[ci] = fov
        scores = []
        for lab in a.labels:
            if lab not in labels:
                continue
            pl = parse_pose_label(lab)
            ml, mr = r.render_pose(pl["x"], pl["y"], pl["dir"], cue=pl["cue"], raised_barriers=pl["raised"])
            i = int(idx[labels.index(lab)])
            scores.append((iou(mask(ml), mask(L[i])) + iou(mask(mr), mask(R[i]))) / 2)
        rows.append((fov, np.nanmean(scores), [round(s, 2) for s in scores]))
        print(f"fovy={fov:5.1f}  mean IoU={np.nanmean(scores):.3f}  per-pose={[round(s,2) for s in scores]}")
    m.cam_fovy[:] = fov0
    best = max(rows, key=lambda t: t[1])
    print(f"best fovy={best[0]} (mean IoU {best[1]:.3f})  labels={a.labels}")


if __name__ == "__main__":
    main()
