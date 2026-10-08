"""Barrier and cue state per corner-maze-rl pose-label prefix.

Mirrors the layout tuples built in corner_maze_rl.env.corner_maze_env
(CornerMazeEnv.__init__): tuple index 0 = state type, 1..16 = barriers in
BARRIER_LOCATIONS order, 17..20 = cues (E, S, W, N). Only the parts needed to
reproduce a pose label's visual state are kept here. In the corner-maze-rl
adapter the env's own tuple is passed instead, so this table is for
stand-alone tools and tests.
"""
from __future__ import annotations

import re

# corner_maze_rl.env.constants.BARRIER_LOCATIONS, same order
BARRIER_LOCATIONS = [
    (10, 5), (9, 6), (10, 7),      # E arm
    (7, 10), (6, 9), (5, 10),      # S arm
    (2, 7), (3, 6), (2, 5),        # W arm
    (5, 2), (6, 3), (7, 2),        # N arm
    (7, 6), (6, 7), (5, 6), (6, 5),  # center E, S, W, N
]
CUE_CELLS = {"e": (11, 6), "s": (6, 11), "w": (1, 6), "n": (6, 1)}

_TRL = {
    "n": [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
    "e": [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    "s": [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    "w": [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
}
_PRE = {
    "n": [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
    "e": [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    "s": [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
    "w": [1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
}
_FIXED = {
    "expa_x_x_xx": [0] * 16,
    "expb_xxxx_x_xx": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1],
    "expb_exxx_x_xx": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1],
    "expb_enxx_x_xx": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
    "expb_enwx_x_xx": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    "expb_enws_x_xx": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    "expb_x_x_xx": [0] * 16,
    "iti_e_x_xx": [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    "iti_e_x_ne": [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],
    "iti_e_x_se": [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    "iti_s_x_xx": [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    "iti_s_x_se": [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    "iti_s_x_sw": [0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0],
    "iti_w_x_xx": [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    "iti_w_x_sw": [0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
    "iti_w_x_nw": [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0],
    "iti_n_x_xx": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    "iti_n_x_nw": [0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    "iti_n_x_ne": [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
}

_LABEL_RX = re.compile(r"^(?P<prefix>.+)_(?P<x>\d+)_(?P<y>\d+)_(?P<d>\d)$")


def barrier_bits(prefix: str) -> list[int]:
    if prefix in _FIXED:
        return list(_FIXED[prefix])
    m = re.match(r"^(trl|pre)_(?P<arm>[nesw])_(?P<cue>[nesw]|x)_", prefix)
    if m:
        table = _TRL if prefix.startswith("trl") else _PRE
        return list(table[m.group("arm")])
    raise KeyError(f"unknown layout prefix: {prefix}")


def cue_cell(prefix: str) -> tuple[int, int] | None:
    m = re.match(r"^(trl|pre)_(?P<arm>[nesw])_(?P<cue>[nesw]|x)_", prefix)
    if m and m.group("cue") != "x":
        return CUE_CELLS[m.group("cue")]
    return None


def raised_barriers(prefix: str) -> set[tuple[int, int]]:
    return {BARRIER_LOCATIONS[i] for i, v in enumerate(barrier_bits(prefix)) if v}


def parse_pose_label(label: str) -> dict:
    m = _LABEL_RX.match(label)
    if not m:
        raise ValueError(label)
    prefix = m.group("prefix")
    return {
        "prefix": prefix,
        "x": int(m.group("x")),
        "y": int(m.group("y")),
        "dir": int(m.group("d")),
        "cue": cue_cell(prefix),
        "raised": raised_barriers(prefix),
    }


def from_layout_tuple(layout: tuple) -> tuple[set[tuple[int, int]], tuple[int, int] | None]:
    """Env layout tuple -> (raised barrier cells, cue cell)."""
    raised = {BARRIER_LOCATIONS[i] for i in range(16) if layout[i + 1]}
    cue = None
    for i, key in enumerate("eswn"):
        if layout[17 + i]:
            cue = CUE_CELLS[key]
    return raised, cue
