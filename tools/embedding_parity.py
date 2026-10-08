"""Acceptance test: does the pretrained stereo CNN see MuJoCo renders like Fusion renders?

1. Identify the TorchScript model + preprocessing that reproduce the 60-D
   embeddings stored in corner-maze-rl's pose_visual.npz.
2. Embed MuJoCo renders of every lookup row with that pipeline and compare:
   cosine similarity to the Fusion embedding of the same row, nearest-neighbour
   retrieval accuracy over all rows, and classifier top-1 agreement.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import re
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from corner_maze_mujoco.layouts import parse_pose_label
from corner_maze_mujoco.renderer import MazeRenderer


def prep(imgs, size, flip):
    out = []
    for im in imgs:
        a = np.asarray(Image.fromarray(im).resize((size, size), Image.BILINEAR), np.float32) / 255.0
        if flip:
            a = a[:, ::-1]
        out.append(np.ascontiguousarray(a))
    return np.stack(out)


def embed(model, L, R, size, flipL, flipR, order, relu):
    l, r = prep(L, size, flipL), prep(R, size, flipR)
    x = np.stack([l, r] if order == "lr" else [r, l], axis=1)
    with torch.no_grad():
        out = model(torch.from_numpy(x))
    logits, emb = out[0].numpy(), out[1].numpy()
    if relu:
        emb = np.maximum(emb, 0)
    return logits, emb


def cosine_rows(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return (A * B).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="../corner-maze-rl/data/lookups/pose_visual.npz")
    ap.add_argument("--models", default="../corner-maze-encoder-pretrain/experiments/evaluate/*/*.ts")
    ap.add_argument("--model-xml", default="model/corner_maze.xml")
    a = ap.parse_args()
    d = np.load(a.npz, allow_pickle=False)
    E, L, R, names = d["embeddings"], d["left_eyes"], d["right_eyes"], list(d["label_names"])
    print(f"stored embeddings: {E.shape} min={E.min():.2f} max={E.max():.1f} (min>=0 means ReLU applied)")
    torch.set_grad_enabled(False)

    # ---- 1. identify model + preprocessing
    best = None
    for mp in sorted(glob.glob(a.models)):
        try:
            m = torch.jit.load(mp, map_location="cpu").eval()
        except Exception:
            continue
        for size, flipL, flipR, order, relu in itertools.product((96, 128), (False, True), (False, True), ("lr", "rl"), (False, True)):
            try:
                _, emb = embed(m, L, R, size, flipL, flipR, order, relu)
            except Exception:
                break
            if emb.shape[1] != E.shape[1]:
                break
            cos = cosine_rows(emb, E).mean()
            mae = np.abs(emb - E).mean()
            if best is None or cos > best[0]:
                best = (cos, mae, mp, size, flipL, flipR, order, relu)
    cos, mae, mp, size, flipL, flipR, order, relu = best
    print(f"best match: {mp.split('/')[-1]}  size={size} flipL={flipL} flipR={flipR} order={order} relu={relu}  mean cos={cos:.5f} mae={mae:.3f}")
    model = torch.jit.load(mp, map_location="cpu").eval()
    logits_f, emb_f = embed(model, L, R, size, flipL, flipR, order, relu)

    # ---- 2. MuJoCo renders of every row
    r = MazeRenderer(a.model_xml)
    ML, MR = [], []
    for n in names:
        pl = parse_pose_label(n)
        ml, mr = r.render_pose(pl["x"], pl["y"], pl["dir"], cue=pl["cue"], raised_barriers=pl["raised"])
        ML.append(ml); MR.append(mr)
    ML, MR = np.stack(ML), np.stack(MR)
    logits_m, emb_m = embed(model, ML, MR, size, flipL, flipR, order, relu)

    cos = cosine_rows(emb_m, emb_f)
    # nearest neighbour: MuJoCo embedding -> closest Fusion embedding (cosine)
    Fn = emb_f / (np.linalg.norm(emb_f, axis=1, keepdims=True) + 1e-9)
    Mn = emb_m / (np.linalg.norm(emb_m, axis=1, keepdims=True) + 1e-9)
    nn = (Mn @ Fn.T).argmax(1)
    nn_ok = nn == np.arange(len(names))
    top1_same = logits_m.argmax(1) == logits_f.argmax(1)
    cued = np.array([bool(re.match(r"^(trl|pre)_[nesw]_[nesw]_", n)) for n in names])
    def rep(mask, label):
        print(f"  {label:22s} n={mask.sum():3d}  cos mean={cos[mask].mean():.3f} median={np.median(cos[mask]):.3f} p10={np.percentile(cos[mask],10):.3f}  NN-retrieval={nn_ok[mask].mean():.3f}  top1-agree={top1_same[mask].mean():.3f}")
    print("\nMuJoCo-vs-Fusion embedding parity (same pretrained CNN):")
    rep(np.ones(len(names), bool), "all rows")
    rep(~cued, "uncued layouts")
    rep(cued, "cued layouts")
    by = defaultdict(list)
    for n, c, ok in zip(names, cos, nn_ok):
        by[re.sub(r"_\d+_\d+_\d$", "", n)].append((c, ok))
    print("\nper prefix: n, mean cos, NN-retrieval")
    for p, v in sorted(by.items()):
        v = np.array(v)
        print(f"  {p:18s} {len(v):4d}  {v[:,0].mean():.3f}  {v[:,1].mean():.3f}")
    worst = np.argsort(cos)[:12]
    print("\nlowest-cosine rows:")
    for i in worst:
        print(f"  {names[i]:22s} cos={cos[i]:.3f}  NN->{names[nn[i]]}")
    # Fusion self-consistency reference: NN of Fusion embedding against itself excluding self is not meaningful;
    # instead report how separable Fusion rows are: mean cosine to the nearest *other* Fusion row.
    S = Fn @ Fn.T
    np.fill_diagonal(S, -1)
    print(f"\nreference: Fusion rows' mean cosine to their nearest OTHER row = {S.max(1).mean():.3f} (MuJoCo-vs-Fusion same-row cosine should exceed this)")


if __name__ == "__main__":
    main()
