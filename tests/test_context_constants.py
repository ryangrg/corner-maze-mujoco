"""The vendored task-context constants and this repo's layouts and model.

src/corner_maze_mujoco/context/ holds copies of
corner-maze/docs/context/constants/ with their SHA-256 in
context_constants.json, so a copy cannot drift silently. The rig's door order
is the ground truth: layouts.py's tables (which mirror the RL env) must equal
the rig's pretrial, trial and Exposure 2e layouts after the registry's
permutation, and the behavioral maze model must name its 16 barriers after the
registry's grid cells.
"""
import hashlib
import json
from pathlib import Path

import pytest

from corner_maze_mujoco import layouts as L

HERE = Path(__file__).resolve().parents[1]
CTX = HERE / "src" / "corner_maze_mujoco" / "context"
MANIFEST = json.loads((CTX / "context_constants.json").read_text())
REG = json.loads((CTX / "doors.json").read_text())
RIG_LAYOUTS = REG["layouts_rig_order"]
ARM = {"n": "North", "e": "East", "s": "South", "w": "West"}


def rig_to_env(vec):
    by_id = {d["id"]: vec[d["rig_column"] - 3] for d in REG["doors"]}
    return [by_id[did] for did in REG["env_index_to_rig_door"]]


def test_vendored_files_match_their_recorded_hashes():
    for name, entry in MANIFEST.items():
        digest = hashlib.sha256((CTX / name).read_bytes()).hexdigest()
        assert digest == entry["sha256"], (
            f"{name} differs from the recorded copy; update it from {entry['source']} "
            "and refresh context_constants.json deliberately"
        )


def test_barrier_locations_follow_the_registry():
    assert [list(b) for b in L.BARRIER_LOCATIONS] == REG["env_barrier_locations"]


@pytest.mark.parametrize("arm", list(ARM))
def test_pretrial_and_trial_layouts_equal_the_rig(arm):
    assert L._PRE[arm] == rig_to_env(RIG_LAYOUTS["pretrial"][ARM[arm]])
    assert L._TRL[arm] == rig_to_env(RIG_LAYOUTS["trial"][ARM[arm]])


def test_exposure_2e_sequence_equals_the_rig():
    env_seq = [L._FIXED[k] for k in ("expb_xxxx_x_xx", "expb_exxx_x_xx", "expb_enxx_x_xx", "expb_enwx_x_xx", "expb_enws_x_xx", "expb_x_x_xx")]
    assert env_seq == [rig_to_env(s["doors"]) for s in RIG_LAYOUTS["exposure_2e_sequence"]]


def test_iti_layouts_equal_the_rig():
    """Since 2026-10-09 the env (and this mirror) raise both centre doors on the
    start arm's axis during the ITI, as the rig does."""
    corner = {"nw": "Northwest", "ne": "Northeast", "se": "Southeast", "sw": "Southwest"}
    for name, env in L._FIXED.items():
        if not name.startswith("iti_"):
            continue
        _, arm, _, var = name.split("_")
        if var == "xx":
            rig = RIG_LAYOUTS["iti_base"][ARM[arm]]
        else:
            rig = RIG_LAYOUTS["iti_close"][f"{corner[var]}->{ARM[arm]}"]
        assert env == rig_to_env(rig), name


def test_behavioral_maze_barriers_are_named_after_the_registry():
    meta = HERE / "model" / "maze_behavior_meta.json"
    if not meta.exists():
        pytest.skip("behavioral maze model not built here")
    names = {b["name"] for b in json.loads(meta.read_text())["barriers"]}
    assert names == {d["mujoco_body"] for d in REG["doors"]}
