# corner-maze-mujoco

MuJoCo model of the Blair Lab corner maze, generated from the Fusion 360 rig
("Rendering Model") that produced the stereo rodent-eye views used by
`corner-maze-rl`. The goal is a renderer that reproduces those views for any
pose and layout, with the geometry, barrier travel, monitor placement and
camera model taken from the CAD rather than hand-measured.

## Layout

```
tools/fusion/DumpTree/      Fusion script (read-only): occurrence tree -> JSON
tools/fusion/ExportMeshes/  Fusion script (read-only): every body -> OBJ + manifest
tools/render_test.py        compile model, verify eye cameras, render test views
tools/compare_fusion.py     render pose labels and score them against pose_visual.npz
src/corner_maze_mujoco/
  pose.py                   grid <-> world mapping, eye camera model (mirrors CaptureView)
  fusion_manifest.py        manifest loader, dedup key, MTL colours
  layouts.py                barrier/cue state per pose-label prefix (mirrors the env tables)
  build_mjcf.py             manifest -> model/corner_maze.xml + deduplicated meshes
  renderer.py               MazeRenderer: (x, y, dir, cue, barriers) -> left/right 128x128
data/tree_dump_rendering_model_v43.json
assets/fusion_export/       manifest.json + meshes/ (meshes gitignored, regenerate)
model/                      generated MJCF + meshes (meshes gitignored)
```

## Pipeline

1. In Fusion, open the rig and run `DumpTree` (optional, inspection) and
   `ExportMeshes`. Install a script by copying its folder into
   `~/Library/Application Support/Autodesk/Autodesk Fusion 360/API/Scripts/`
   and launching it from Utilities > Add-Ins > Scripts and Add-Ins. For
   `ExportMeshes`, copy `export_config.example.json` to `export_config.json`
   next to the installed script and set `out_dir` to `assets/fusion_export`.
2. Build the model:
   `PYTHONPATH=src python -m corner_maze_mujoco.build_mjcf --manifest assets/fusion_export/manifest.json --out model`
3. Check it: `PYTHONPATH=src python tools/render_test.py`
4. Compare with the Fusion renders:
   `PYTHONPATH=src python tools/compare_fusion.py --npz ../corner-maze-rl/data/lookups/pose_visual.npz --labels expa_x_x_xx_6_6_0 trl_e_n_xx_6_6_0`

## Model facts (from the CAD)

* Units: meters, Z up, Fusion design frame. Grid cell pitch 0.249 m.
* Track top at 0.853 m; eyes 0.0635 m above it; eye separation 0.0127 m;
  eye yaw +/-50 deg, pitch +15 deg, vertical FOV 150 deg.
* 24 barriers, vertical slide joints, travel 0.4064 m (16 in). Sixteen match
  the env's `BARRIER_LOCATIONS`; eight more sit beside the corner platforms.
* Four monitors tilted 45 deg down, one per wall. The room is unlit; the
  monitors are the only light sources. A blank monitor is a dim emitter, the
  cue is a full-white emitter plus a spotlight (`light_cue_x*_y*`).
* OBJ files from Fusion are in component-local frames; the manifest's
  per-occurrence transform places them (verified against world bounds).
  Identical parts share one mesh asset.

## Calibration and parity (2026-10-07)

* Effective vertical FOV of the Fusion renders is 121.5 deg, not the nominal
  150 deg written to Fusion's camera (fitted on monitor silhouettes, mask IoU
  0.88; see `tools/fit_fov.py`). `pose.EYE_VFOV_DEG` carries the fitted value.
* Against all 1,244 raw stereo captures (`tools/eval_raw_captures.py`):
  every uncued layout (exposure, ITI, no-cue trial/pretrial) matches at
  correlation >= 0.99; cued layouts match at the median but differ where the
  lit monitor's indirect light, halo, track reflections, or a lit acrylic
  panel dominate the frame.
* One raw Fusion capture is bad: `trl_e_n_xx_8_6_e_l.png` (left eye, pose
  (8,6) facing east). The lookup row and its three rotated labels inherit it.
* Encoder behind `pose_visual.npz`: encoder-pretrain
  `experiments/evaluate/dual-indep-20260319-222411/stereo_cnn-dual-indep-20260319-222411.ts`,
  input (N, 2, 96, 96) float in [0, 1], channel order left/right, no flips,
  embedding = second tuple output (pre-ReLU). Reproduces the stored
  embeddings at cosine 1.0000 (`tools/embedding_parity.py`).
* MuJoCo renders through that encoder vs the Fusion embeddings: uncued rows
  cosine 0.996 (NN retrieval 0.87, top-1 agreement 0.85); cued rows cosine
  0.918 (NN retrieval 0.33, top-1 agreement 0.26). For reference, Fusion rows
  are only 0.96 apart from their nearest other row, so retrieval is a hard
  metric on this lookup.

Open items: indirect-light proxy for the lit cue (omni light + tessellated
curtains, or a bloom post-process), acrylic appearance when back-lit, and
whether to keep the pretrained encoder or retrain it on MuJoCo renders.
