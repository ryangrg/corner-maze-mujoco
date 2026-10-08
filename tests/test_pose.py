"""Pin down the CaptureView conventions reproduced in corner_maze_mujoco.pose."""
import math

import numpy as np
import pytest

from corner_maze_mujoco import pose as P
from corner_maze_mujoco.layouts import parse_pose_label, BARRIER_LOCATIONS


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def test_anchor_table_matches_captureview():
    # CaptureView: model X from grid y, model Y from grid x, inches
    assert np.allclose(P.grid_to_model_m(6, 6) / P.IN, [55.519, 53.50])
    assert np.allclose(P.grid_to_model_m(11, 6) / P.IN, [55.519, 100.98])
    assert np.allclose(P.grid_to_model_m(6, 1) / P.IN, [8.039, 53.50])


@pytest.mark.parametrize("gx,gy", [(2, 2), (10, 5), (6, 6), (1, 11), (11, 1)])
def test_grid_roundtrip(gx, gy):
    mx, my = P.grid_to_model_m(gx, gy)
    assert np.allclose(P.model_to_grid(mx, my), (gx, gy))


def test_headings_match_captureview():
    # east = (11,6)-(6,6) -> +Y ; north = (6,1)-(6,6) -> -X
    assert P.HEADING_XY[0] == (0.0, 1.0)
    assert P.HEADING_XY[3] == (-1.0, 0.0)
    assert P.HEADING_XY[1] == tuple(-np.array(P.HEADING_XY[3]))
    assert P.HEADING_XY[2] == tuple(-np.array(P.HEADING_XY[0]))


@pytest.mark.parametrize("direction", [0, 1, 2, 3])
def test_eye_forward_vectors(direction):
    pos, hq = P.head_pose(6, 6, direction)
    Rh = quat_to_mat(hq)
    th = P.heading_angle(direction)
    for name, yaw in (("eye_left", +P.EYE_YAW_DEG), ("eye_right", -P.EYE_YAW_DEG)):
        fr = P.eye_local_frames()[name]
        Rc = Rh @ quat_to_mat(fr["quat"])
        fwd = -Rc[:, 2]
        cp, sp = math.cos(math.radians(P.EYE_PITCH_DEG)), math.sin(math.radians(P.EYE_PITCH_DEG))
        exp = [cp * math.cos(th + math.radians(yaw)), cp * math.sin(th + math.radians(yaw)), sp]
        assert np.allclose(fwd, exp, atol=1e-9)
        assert Rc[2, 1] > 0.9  # camera up is roughly world up
        # eye offset along the head's right (-y) axis, left eye on the +y side
        side = 1.0 if name == "eye_left" else -1.0
        assert np.allclose(fr["pos"], [0, side * P.EYE_SEPARATION_M / 2, 0])


def test_eye_height():
    assert math.isclose(P.EYE_Z_M, (33.577 + 2.5) * 0.0254)


def test_rotmat_to_quat_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = quat_to_mat(q)
        q2 = P.rotmat_to_quat(R)
        assert np.allclose(quat_to_mat(q2), R, atol=1e-9)


def test_pose_label_layouts():
    p = parse_pose_label("trl_e_n_xx_6_6_0")
    assert p["cue"] == (6, 1)
    assert p["raised"] == {(10, 5), (10, 7), (2, 7), (2, 5), (5, 6)}
    assert parse_pose_label("expa_x_x_xx_6_6_0")["raised"] == set()
    assert len(BARRIER_LOCATIONS) == 16
