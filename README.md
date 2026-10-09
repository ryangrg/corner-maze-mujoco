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
tools/fit_fov.py            fit the effective Fusion FOV by monitor-silhouette IoU
tools/eval_raw_captures.py  score renders against all 1,244 raw Fusion captures
tools/embedding_parity.py   push renders through the pretrained encoder, compare embeddings
tools/diag_light.py         stray-light / bounce-light diagnostic for cued poses
tests/test_pose.py          pins the coordinate and camera conventions
tests/test_context_constants.py  the vendored door/zone registry vs layouts.py and the model
src/corner_maze_mujoco/context/  doors.json, zones.json (copies of corner-maze-analysis/docs/context/
                            constants/, hashes in context_constants.json): the rig's 16 doors, the
                            zone edges they close, the permutation to BARRIER_LOCATIONS, the MuJoCo
                            body names; the rig order is the ground truth
src/corner_maze_mujoco/
  pose.py                   grid <-> world mapping (GridFrame), eye camera model (mirrors CaptureView)
  fusion_manifest.py        manifest loader, dedup key, MTL colours
  layouts.py                barrier/cue state per pose-label prefix (mirrors the env tables)
  build_mjcf.py             exports -> room.xml + maze_<tag>.xml, composed into corner_maze.xml
  renderer.py               MazeRenderer: (x, y, dir, cue, barriers) -> left/right 128x128
data/tree_dump_<source>.json      Fusion assembly structure, one per exported design
assets/fusion_export/<source>/   manifest.json + meshes/ per Fusion design (meshes gitignored)
model/                      room.xml, maze_<tag>.xml, corner_maze.xml and their *_meta.json;
                            meshes/<source>/ gitignored
```

## Pipeline

1. In Fusion, open the rig and run `DumpTree` (optional, inspection) and
   `ExportMeshes`. Install a script by copying its folder into
   `~/Library/Application Support/Autodesk/Autodesk Fusion 360/API/Scripts/`
   and launching it from Utilities > Add-Ins > Scripts and Add-Ins. For
   `ExportMeshes`, copy `export_config.example.json` to `export_config.json`
   next to the installed script and set `out_dir` to
   `assets/fusion_export/<source>` (one folder per Fusion design).
2. Build the model. The room and the maze are separate files so maze versions
   can be swapped under the same room:
   `PYTHONPATH=src python -m corner_maze_mujoco.build_mjcf all`
   builds `model/room.xml` (monitors, curtains, frame) and `model/maze_v2.xml`
   (the maze subtree of the Rendering Model) and composes them into
   `model/corner_maze.xml`. To add another maze design from its own export:
   `PYTHONPATH=src python -m corner_maze_mujoco.build_mjcf maze --manifest assets/fusion_export/<source>/manifest.json --tag <tag> --profile <profile>`
   then `PYTHONPATH=src python -m corner_maze_mujoco.build_mjcf compose --maze <tag>`.
   A maze is placed with its grid centre on the room's reference point (stored
   in `room_meta.json`) and its floor on the room floor, so it keeps its real
   height; its own grid frame travels with it in `maze_<tag>_meta.json`, and
   the renderer puts the eyes there.
3. Check it: `PYTHONPATH=src python tools/render_test.py`
4. Compare with the Fusion renders:
   `PYTHONPATH=src python tools/compare_fusion.py --npz ../corner-maze-rl/data/lookups/pose_visual.npz --labels expa_x_x_xx_6_6_0 trl_e_n_xx_6_6_0`

## Model facts (from the CAD)

* Units: meters, Z up, Fusion design frame. Grid cell pitch 0.249 m.
* Track top at 0.853 m; eyes 0.0635 m above it; eye separation 0.0127 m;
  eye yaw +/-50 deg, pitch +15 deg, vertical FOV 121.5 deg effective (150 deg
  nominal; see Calibration below).
* 24 barriers, vertical slide joints, travel 0.4064 m (16 in). Sixteen match
  the env's `BARRIER_LOCATIONS`; eight more sit beside the corner platforms.
* Four monitors tilted 45 deg down, one per wall. The room is unlit; the
  monitors are the only light sources. A blank monitor is a dim emitter, the
  cue is a full-white emitter plus a spotlight (`light_cue_x*_y*`).
* OBJ files from Fusion are in component-local frames; the manifest's
  per-occurrence transform places them (verified against world bounds).
  Identical parts share one mesh asset.

## Maze versions

| tag | Fusion source | what it is | profile |
|---|---|---|---|
| `v2` | Rendering Model v43, subtree `Full Assembly v32:1` | the 2.0 redesign, never built; the maze behind the Fusion stereo renders | `v2`, frame = CaptureView table |
| `behavior` | Complete Maze Assembly II v16 (`assets/fusion_export/behavior_maze/`) | the maze the behavior data was collected on | `behavior`, frame from geometry |

Build and compose the behavioral maze under the same room:

```
PYTHONPATH=src python -m corner_maze_mujoco.build_mjcf maze --manifest assets/fusion_export/behavior_maze/manifest.json --tag behavior --profile behavior
PYTHONPATH=src python -m corner_maze_mujoco.build_mjcf compose --maze behavior --name corner_maze_behavior
```

`MazeRenderer("model/corner_maze_behavior.xml")` then renders the same pose labels
on the behavioral maze. `model/compare/v2_vs_behavior_*.png` show both side by side.

How a maze's grid is read (`build_mjcf.calibrate_frame`): cell (6,6) is the centre
platform, the perimeter straight tracks form the ring at cells 2 and 10, and cells 1
and 11 are the reward wells beyond the ring corners (CaptureView's cell-1 point is the
2.0 maze's syringe-pump axis to within 3 mm). Applied to the 2.0 maze this reproduces
the CaptureView table within 3 mm at every knot (`tests/test_build.py`); on the
behavioral maze the 16 barrier panels land exactly on the env's 16 `BARRIER_LOCATIONS`.

Both mazes have a 2.0 m ring (cells 2 to 10) and a 25 cm cell pitch. Each stands on
the room floor at its real height: walking surface 0.853 m for the 2.0 design and
1.003 m for the behavioral maze, so the eyes sit 15 cm higher on the latter and the
monitors appear correspondingly lower in its views.

Barrier travel is 16 in on the 2.0 maze and 12 in on the behavioral maze. Track and
wall colours are still Fusion's exported appearances (a generic "Surface - Matte"
grey on the behavioral maze); they will be tuned against the real rig in a later
phase, by navigating the rendered environment (`APPEARANCE_RGBA` overrides).

## Calibration and parity (2026-10-07)

* Effective vertical FOV of the Fusion renders is 121.5 deg, not the nominal
  150 deg written to Fusion's camera (fitted on monitor silhouettes, mask IoU
  0.88; see `tools/fit_fov.py`). `pose.EYE_VFOV_DEG` carries the fitted value.
* Against all 1,244 raw stereo captures (`tools/eval_raw_captures.py`):
  every uncued layout (exposure, ITI, no-cue trial/pretrial) matches at
  correlation >= 0.99; cued layouts match at the median but differ where the
  lit monitor's indirect light, halo, track reflections, or a lit acrylic
  panel dominate the frame.
* One raw Fusion capture was bad: `trl_e_n_xx_8_6_e_l.png` (left eye, pose
  (8,6) facing east) was rendered rolled 90 deg. Fixed at the source on
  2026-10-07; `pose_visual.npz` was rebuilt (294 rows) with a retrained encoder
  of the same architecture, `dual-indep-20261007-204751`.
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

Decision (2026-10-08): no further parity tuning and no encoder pretraining.
End-to-end agents trained on raw stereo frames do as well or better, so the
MuJoCo path feeds rendered frames directly. The parity tools stay as
regression checks. Next: an interactive arrow-key simulator.
