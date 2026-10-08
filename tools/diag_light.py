"""Per-material stray-light diagnostic and bounce-light sweep for cued poses."""
from __future__ import annotations

import itertools

import mujoco
import numpy as np
from PIL import Image, ImageFilter

from corner_maze_mujoco.layouts import parse_pose_label
from corner_maze_mujoco.renderer import MazeRenderer

NPZ = "../corner-maze-rl/data/lookups/pose_visual.npz"
d = np.load(NPZ, allow_pickle=False)
labels = list(d["pose_labels"]); idx = d["pose_to_idx"]; L, R = d["left_eyes"], d["right_eyes"]
r = MazeRenderer("model/corner_maze.xml")
m = r.model
for ci in range(m.ncam):
    if m.camera(ci).name.startswith("eye"):
        m.cam_fovy[ci] = 120.0

# ---- 1. which materials are visible/bright with no light (expa centre, left eye)
r.set_barriers(set()); r.set_cue(None); r.set_head(6, 6, 0)
mujoco.mj_forward(m, r.data)
gray, _ = r.render_eyes()
seg = mujoco.Renderer(m, height=128, width=128)
seg.enable_segmentation_rendering()
seg.update_scene(r.data, camera="eye_left")
s = seg.render()
geom_id = s[..., 0].astype(int)
per_mat = {}
for gid in np.unique(geom_id):
    if gid < 0:
        continue
    mat = int(m.geom_matid[gid])
    name = m.material(mat).name if mat >= 0 else "(none)"
    px = gray[geom_id == gid]
    e = per_mat.setdefault(name, [0, 0.0])
    e[0] += px.size; e[1] += px.sum()
print("headlight active:", m.vis.headlight.active, "ambient:", m.vis.headlight.ambient)
print("per-material pixel count / mean brightness (expa centre, left eye, fovy 120):")
for name, (n, tot) in sorted(per_mat.items(), key=lambda kv: -kv[1][1]):
    print(f"   {name:34s} px={n:6d} mean={tot/max(n,1):6.2f} rgba={m.mat_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, name)] if name!='(none)' else '-'} emission={m.mat_emission[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, name)] if name!='(none)' else '-'}")
# headlight active with zeros?
m.vis.headlight.active = 1
g2, _ = r.render_eyes()
print(f"headlight active=1 with zero ambient/diffuse: mean outside screens = {g2[gray<30].mean():.2f} (was {gray[gray<30].mean():.2f})")
m.vis.headlight.active = 0

# ---- 2. bounce-light proxy sweep on cued poses
def score(a, b, blur=1.5):
    fa = np.asarray(Image.fromarray(a).filter(ImageFilter.GaussianBlur(blur)), float)
    fb = np.asarray(Image.fromarray(b).filter(ImageFilter.GaussianBlur(blur)), float)
    corr = float(np.corrcoef(fa.ravel(), fb.ravel())[0, 1]) if fa.std() > 0 and fb.std() > 0 else float("nan")
    return corr, float(np.abs(fa - fb).mean())

cued = ["trl_e_n_xx_6_6_0", "trl_n_n_xx_6_3_3", "trl_s_e_xx_6_8_1", "trl_w_w_xx_3_6_3", "trl_n_s_xx_6_4_0"]
cued = [c for c in cued if c in labels]
light_ids = list(r._light.values())
diff0 = m.light_diffuse.copy(); att0 = m.light_attenuation.copy(); cut0 = m.light_cutoff.copy()
best = None
print("\nbounce sweep (fovy 120): ambient | spot diffuse | cutoff | quad atten -> mean corr, mean MAE over", cued)
for amb, diff, cut, quad in itertools.product([0.0, 0.03, 0.06, 0.12], [0.6, 1.0, 1.6], [80.0, 180.0], [0.15, 0.6]):
    m.vis.headlight.active = 1 if amb > 0 else 0
    m.vis.headlight.ambient[:] = amb
    m.vis.headlight.diffuse[:] = 0; m.vis.headlight.specular[:] = 0
    for li in light_ids:
        m.light_diffuse[li] = diff; m.light_cutoff[li] = cut; m.light_attenuation[li] = [1, 0, quad]
    cs, ms = [], []
    for lab in cued:
        pl = parse_pose_label(lab)
        ml, mr = r.render_pose(pl["x"], pl["y"], pl["dir"], cue=pl["cue"], raised_barriers=pl["raised"])
        i = int(idx[labels.index(lab)])
        for a, b in ((ml, L[i]), (mr, R[i])):
            c, e = score(a, b); cs.append(c); ms.append(e)
    res = (np.nanmean(cs), np.mean(ms))
    if best is None or res[0] > best[0][0]:
        best = (res, (amb, diff, cut, quad))
    print(f"   amb={amb:.2f} diff={diff:.1f} cut={cut:3.0f} quad={quad:.2f} -> corr={res[0]:.3f} mae={res[1]:.1f}")
print("BEST:", best)
