"""Score MuJoCo renders against every raw Fusion base capture.

Raw captures live in corner-maze-encoder-pretrain/data/images/corner-maze-base-images/
<prefix>/<prefix>_<x>_<y>_<letter>_<eye>.png with letter in e/s/w/n (-> dir 0..3)
or ne/se/sw/nw at corner cells (skipped: the env has no diagonal headings).
Writes a CSV and prints a per-prefix summary plus outliers.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from corner_maze_mujoco.layouts import parse_pose_label
from corner_maze_mujoco.renderer import MazeRenderer

LET = {"e": 0, "s": 1, "w": 2, "n": 3}
RX = re.compile(r"^(?P<prefix>.+)_(?P<x>\d+)_(?P<y>\d+)_(?P<let>[nesw]{1,2})_(?P<eye>[lr])\.png$")


def corr(a, b, blur=1.5):
    fa = np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(blur)), float)
    fb = np.asarray(Image.fromarray(b).filter(ImageFilter.GaussianBlur(blur)), float)
    if fa.std() == 0 or fb.std() == 0:
        return 1.0 if np.abs(fa - fb).mean() < 1.0 else 0.0
    return float(np.corrcoef(fa.ravel(), fb.ravel())[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../corner-maze-encoder-pretrain/data/images/corner-maze-base-images")
    ap.add_argument("--model", default="model/corner_maze.xml")
    ap.add_argument("--out", default="model/compare/raw_eval.csv")
    ap.add_argument("--thresh", type=float, default=0.8)
    a = ap.parse_args()
    r = MazeRenderer(a.model)
    pairs = defaultdict(dict)
    for p in Path(a.raw).rglob("*.png"):
        m = RX.match(p.name)
        if not m or m.group("let") not in LET:
            continue
        key = (m.group("prefix"), int(m.group("x")), int(m.group("y")), LET[m.group("let")])
        pairs[key][m.group("eye")] = p
    rows = []
    skipped = 0
    for (prefix, x, y, d), eyes in sorted(pairs.items()):
        if "l" not in eyes or "r" not in eyes:
            continue
        try:
            pl = parse_pose_label(f"{prefix}_{x}_{y}_{d}")
        except KeyError:
            skipped += 1
            continue
        ml, mr = r.render_pose(x, y, d, cue=pl["cue"], raised_barriers=pl["raised"])
        fl = np.asarray(Image.open(eyes["l"]).convert("L").resize((128, 128), Image.BILINEAR), np.uint8)
        fr = np.asarray(Image.open(eyes["r"]).convert("L").resize((128, 128), Image.BILINEAR), np.uint8)
        rows.append(dict(prefix=prefix, x=x, y=y, dir=d, corr_l=corr(ml, fl), corr_r=corr(mr, fr),
                         mae_l=float(np.abs(ml.astype(float) - fl).mean()), mae_r=float(np.abs(mr.astype(float) - fr).mean()),
                         mean_mj=float((ml.mean() + mr.mean()) / 2), mean_fusion=float((fl.mean() + fr.mean()) / 2)))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"scored {len(rows)} stereo pairs ({skipped} skipped: unknown prefix); csv -> {a.out}")
    by = defaultdict(list)
    for row in rows:
        by[row["prefix"]].append(min(row["corr_l"], row["corr_r"]))
    print(f"{'prefix':18s} {'n':>4s} {'median':>7s} {'p10':>6s} {'min':>6s}  {'<thr':>5s}")
    for pfx, v in sorted(by.items()):
        v = np.array(v)
        print(f"{pfx:18s} {len(v):4d} {np.median(v):7.3f} {np.percentile(v,10):6.3f} {v.min():6.3f}  {(v<a.thresh).sum():5d}")
    allv = np.array([min(r_["corr_l"], r_["corr_r"]) for r_ in rows])
    print(f"ALL: n={len(allv)} median={np.median(allv):.3f} p10={np.percentile(allv,10):.3f} frac>=0.9={np.mean(allv>=0.9):.3f} frac<{a.thresh}={np.mean(allv<a.thresh):.3f}")
    bad = sorted([r_ for r_ in rows if min(r_["corr_l"], r_["corr_r"]) < a.thresh], key=lambda r_: min(r_["corr_l"], r_["corr_r"]))
    print(f"\nworst {min(25, len(bad))} of {len(bad)} below {a.thresh}:")
    for r_ in bad[:25]:
        print(f"  {r_['prefix']}_{r_['x']}_{r_['y']}_{r_['dir']}  L={r_['corr_l']:.2f} R={r_['corr_r']:.2f}  mean mj/fusion={r_['mean_mj']:.1f}/{r_['mean_fusion']:.1f}")


if __name__ == "__main__":
    main()
