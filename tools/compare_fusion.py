"""Render pose labels in MuJoCo and compare with the Fusion renders in pose_visual.npz.

    PYTHONPATH=src python tools/compare_fusion.py --model model/corner_maze.xml \
        --npz ../corner-maze-rl/data/lookups/pose_visual.npz --labels expa_x_x_xx_6_6_0 expa_x_x_xx_6_3_3

Barriers and cue per label come from corner_maze_mujoco.layouts, which mirrors
the env's layout tuples. --raise adds extra barrier cells on top.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from corner_maze_mujoco.layouts import parse_pose_label
from corner_maze_mujoco.renderer import MazeRenderer


def score(a: np.ndarray, b: np.ndarray, blur: float) -> dict:
    fa = np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(blur)), float) if blur else a.astype(float)
    fb = np.asarray(Image.fromarray(b).filter(ImageFilter.GaussianBlur(blur)), float) if blur else b.astype(float)
    mae = np.abs(fa - fb).mean()
    corr = float(np.corrcoef(fa.ravel(), fb.ravel())[0, 1]) if fa.std() > 0 and fb.std() > 0 else float("nan")
    return {"mae": round(float(mae), 2), "corr": round(corr, 3), "mean_mj": round(float(a.mean()), 1), "mean_fusion": round(float(b.mean()), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model/corner_maze.xml")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", default="model/compare")
    ap.add_argument("--blur", type=float, default=1.5)
    ap.add_argument("--raise", dest="raise_", default="", help="x:y,x:y barrier cells to raise")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(a.npz, allow_pickle=False)
    labels = list(d["pose_labels"])
    idx = d["pose_to_idx"]
    L, R = d["left_eyes"], d["right_eyes"]
    raised = [tuple(int(v) for v in s.split(":")) for s in a.raise_.split(",") if s]
    r = MazeRenderer(a.model)
    for lab in a.labels:
        pl = parse_pose_label(lab)
        ml, mr = r.render_pose(pl["x"], pl["y"], pl["dir"], cue=pl["cue"], raised_barriers=pl["raised"] | set(raised))
        if lab in labels:
            i = int(idx[labels.index(lab)])
            fl, fr = L[i], R[i]
            sl, sr = score(ml, fl, a.blur), score(mr, fr, a.blur)
            print(f"{lab}: L {sl}  R {sr}")
        else:
            fl = fr = np.zeros_like(ml)
            print(f"{lab}: not in npz (rendered only)")
        sep = np.full((128, 4), 255, np.uint8)
        top = np.concatenate([ml, sep, mr], 1)
        bot = np.concatenate([fl, sep, fr], 1)
        panel = np.concatenate([top, np.full((4, top.shape[1]), 255, np.uint8), bot], 0)
        Image.fromarray(panel).resize((panel.shape[1] * 3, panel.shape[0] * 3), Image.NEAREST).save(out / f"{lab}.png")
    print("panels (MuJoCo top, Fusion bottom) in", out)


if __name__ == "__main__":
    main()
