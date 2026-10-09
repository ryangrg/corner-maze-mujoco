"""Grid-frame plumbing and the geometry-based calibration (validated on the 2.0 maze)."""
from pathlib import Path

import numpy as np
import pytest

from corner_maze_mujoco import pose as P
from corner_maze_mujoco.build_mjcf import PROFILES, RENDERING_MODEL_MAZE_ROOT, calibrate_frame, select_bodies
from corner_maze_mujoco.fusion_manifest import load_manifest

V2_MANIFEST = Path(__file__).resolve().parents[1] / "assets/fusion_export/rendering_model_v43/manifest.json"


def test_gridframe_meta_roundtrip():
    f = P.CAPTUREVIEW
    g = P.GridFrame.from_meta(f.to_meta())
    assert np.allclose(g.grid_to_model_m(3, 9), f.grid_to_model_m(3, 9))
    assert np.isclose(g.eye_z_m, f.eye_z_m)


def test_gridframe_translated_moves_everything():
    f = P.CAPTUREVIEW.translated([0.1, -0.2, 0.3])
    assert np.allclose(f.center_xy_m(), P.CAPTUREVIEW.center_xy_m() + [0.1, -0.2])
    assert np.isclose(f.track_top_m, P.CAPTUREVIEW.track_top_m + 0.3)
    pos, _ = f.head_pose(6, 6, 0)
    assert np.isclose(pos[2], P.EYE_Z_M + 0.3)


def test_model_to_grid_extrapolates_beyond_knots():
    f = P.CAPTUREVIEW
    far = f.grid_to_model_m(6, 11) + [0.3, 0.0]  # 30 cm beyond the grid-y=11 line (model X follows grid y)
    gx, gy = f.model_to_grid(*far)
    assert gy > 11.5 and np.isclose(gx, 6)


def test_v2_geometry_frame_matches_captureview():
    """The same reading of the track (centre, ring, wells) that we apply to the behavioral maze
    must reproduce the CaptureView table on the 2.0 maze, where the table is ground truth."""
    if not V2_MANIFEST.exists():
        pytest.skip("Fusion export not present")
    _, bodies = load_manifest(V2_MANIFEST)
    sel, _ = select_bodies(bodies, exclude=set(), include_hidden=False, root=RENDERING_MODEL_MAZE_ROOT)
    if len(sel) < 100:
        pytest.skip("Fusion meshes not present (gitignored); nothing to calibrate on")
    g = calibrate_frame(sel, PROFILES["v2"])
    cv = P.CAPTUREVIEW
    for k in (1, 2, 6, 10, 11):
        assert abs(g.x_from_gy_m[k] - cv.x_from_gy_m[k]) < 0.005, k
        assert abs(g.y_from_gx_m[k] - cv.y_from_gx_m[k]) < 0.005, k
    assert abs(g.track_top_m - cv.track_top_m) < 0.001
