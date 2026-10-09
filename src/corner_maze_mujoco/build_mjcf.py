"""Generate the MuJoCo model from Fusion exports: one room file plus swappable maze files.

    # everything for the Rendering Model export: room + maze "v2" + the composed model
    python -m corner_maze_mujoco.build_mjcf all

    # or piece by piece
    python -m corner_maze_mujoco.build_mjcf room    --manifest assets/fusion_export/rendering_model_v43/manifest.json
    python -m corner_maze_mujoco.build_mjcf maze    --manifest <manifest> --tag v2 --root "Full Assembly v32:1" --profile v2
    python -m corner_maze_mujoco.build_mjcf compose --maze v2

Output (in --out, default model/)
---------------------------------
room.xml, room_meta.json             monitors (screens + cue lights), curtains, frame, floor
maze_<tag>.xml, maze_<tag>_meta.json one per maze version: track, collisions, barriers and
                                     actuators, all inside body "maze_<tag>", positioned with the
                                     maze's grid centre on the room's reference point and its floor
                                     on the room floor (each maze keeps its real height)
meshes/room/, meshes/<tag>/          deduplicated OBJ copies (gitignored)
corner_maze.xml, corner_maze_meta.json  composed model: settings + <include> room + <include>
                                     one maze + the mocap rat head carrying the eye cameras

Design
------
* Every exported body becomes a visual mesh geom (contype/conaffinity 0), placed
  with its occurrence transform. Identical parts share one mesh asset.
* Collision geometry is boxes from the world bounding boxes of track pieces and
  barrier panels (all axis-aligned in this rig).
* Barrier mechanisms become bodies with a vertical slide joint, joint 0 = lowered.
  Which components form a barrier, which part of it moves, and what counts as
  track is a per-maze `MazeProfile`. Barriers raised in the Fusion snapshot are
  normalised to lowered.
* The room is the Rendering Model export minus its maze subtree. The blank
  monitors' screen geoms are named so a cue can be shown by swapping the geom
  material at runtime; hidden "Cued" monitor assemblies are not instanced (they
  duplicate the blank ones in place).
* A maze's grid frame (cell centres, track top) is the CaptureView table for
  profile v2 or is derived from the maze's own geometry. The composed model's
  rat head and the renderer use the frame stored in the metadata, so swapping
  mazes moves the eyes with the track.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from . import pose as P
from .fusion_manifest import BodyInstance, asset_name, group_by_key, load_manifest, read_mtl_kd, sanitize

DEFAULT_MANIFEST = Path("assets/fusion_export/rendering_model_v43/manifest.json")
RENDERING_MODEL_MAZE_ROOT = "Full Assembly v32:1"

MONITOR_BLANK_RX = re.compile(r"^Monitor Moount Assembly Angel v\d+:\d+$")
MONITOR_CUED_RX = re.compile(r"^Cued Monitor")
SCREEN_COMPONENT_RX = re.compile(r"^24” Monitor v\d+$|^24\" Monitor v\d+$")

# appearance name -> rgba override (Fusion appearances that the MTL Kd misrepresents)
APPEARANCE_RGBA = {
    "Fabric (Black)": (0.02, 0.02, 0.02, 1.0),
    "Track Black": (0.04, 0.04, 0.04, 1.0),
    "Acrylic (Clear)": (0.85, 0.9, 0.95, 0.12),
}


@dataclass(frozen=True)
class MazeProfile:
    """How to read one maze design: which components form barriers, move, and carry the track."""

    barrier_rx: re.Pattern    # matched on the body path; group "prefix" names one barrier mechanism
    moving_rx: re.Pattern     # body path inside a mechanism that rides the slide
    panel_rx: re.Pattern      # component name of the acrylic panel (collision box, raised detection)
    track_slab_rx: re.Pattern
    track_wall_rx: re.Pattern
    travel_m: float
    frame: str                # "captureview": pose.CAPTUREVIEW; "geometry": derive from the track
    floor_z_m: float = 0.0    # the design's floor plane in its own frame; it is placed on the room floor
    # frame="geometry" only: component regexes (matched on the component name) that locate the grid
    center_rx: re.Pattern | None = None   # the centre platform slab -> cell (6, 6)
    ring_rx: re.Pattern | None = None     # straight track slabs; the perimeter ones give cells 2 and 10
    well_rx: re.Pattern | None = None     # the reward well where the rat's head is -> cells 1 and 11
    surface_rx: re.Pattern | None = None  # walking-surface slabs; their top is the track top


PROFILES = {
    # The unbuilt 2.0 redesign (Rendering Model's "Full Assembly"). Frame = CaptureView's table.
    "v2": MazeProfile(
        barrier_rx=re.compile(r"^(?P<prefix>.*\+New Barrier Assembly v\d+:\d+)"),
        moving_rx=re.compile(r"Saddle and Plexiglass Assembly"),
        panel_rx=re.compile(r"New Plexiglass Barrier"),
        track_slab_rx=re.compile(r"^(Straight Track|Track Corner|Side Intersection|Center Intersection) v\d+$"),
        track_wall_rx=re.compile(r"Track Wall|Track: Wall|Straight Track Wall"),
        travel_m=16.0 * P.IN,
        frame="captureview",
        center_rx=re.compile(r"^Center Intersection v\d+$"),
        ring_rx=re.compile(r"^Straight Track v\d+$"),
        well_rx=re.compile(r"^Syringe_Pump_Slide v\d+$"),  # pump axis = CaptureView's cell (1,1) point
        surface_rx=re.compile(r"^(Straight Track|Track Corner|Side Intersection|Center Intersection) v\d+$"),
    ),
    # The maze actually used for behavior ("Complete Maze Assembly II"). 16 barriers, one per env
    # barrier cell, each a panel + light bracket + actuator piston riding in two channels.
    # Travel: the actuator is a PA-15-12-11 (12 in stroke); confirm against the real rig.
    "behavior": MazeProfile(
        barrier_rx=re.compile(r"^(?P<prefix>.*\+Support Assembly Accuated Light v\d+:\d+)"),
        moving_rx=re.compile(r"Light Barrier Assembly"),
        panel_rx=re.compile(r"^Barrier II v\d+$"),
        track_slab_rx=re.compile(r"^Maze Segment: (Straight|Four Way Intersection|Three Way Intersection|Corner) v\d+$|^Track Reward v\d+$"),
        track_wall_rx=re.compile(r"Wall|^Guard:"),
        travel_m=12.0 * P.IN,
        frame="geometry",
        center_rx=re.compile(r"^Maze Segment: Four Way Intersection v\d+$"),
        ring_rx=re.compile(r"^Maze Segment: Straight v\d+$"),
        well_rx=re.compile(r"^Reward Recepticle"),
        surface_rx=re.compile(r"^Maze Segment: (Straight|Four Way Intersection|Three Way Intersection|Corner) v\d+$|^Track Reward v\d+$"),
    ),
}


# ---- small helpers ------------------------------------------------------------------------------

def fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in np.atleast_1d(v))


def is_axis_aligned(R: np.ndarray, tol=1e-6) -> bool:
    return np.all(np.isclose(np.abs(R), np.round(np.abs(R)), atol=tol))


def write_xml(root: ET.Element, path: Path):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=False)


class MeshLib:
    """Deduplicated mesh assets for one source, copied to meshes/<tag>/ and named <tag>_<part>."""

    def __init__(self, tag: str, out: Path, copy: bool):
        self.tag, self.dir, self.copy = tag, out / "meshes" / tag, copy
        self.name: dict[tuple, str] = {}
        self.file: dict[tuple, str] = {}

    def register(self, bodies: list[BodyInstance]):
        self.dir.mkdir(parents=True, exist_ok=True)
        for key, insts in group_by_key(bodies).items():
            base = asset_name(key)
            self.name[key] = f"{self.tag}_{base}"
            self.file[key] = f"{self.tag}/{base}.obj"
            dst = self.dir / f"{base}.obj"
            if self.copy and not dst.exists():
                with open(insts[0].file, "r", errors="ignore") as fi, open(dst, "w") as fo:
                    for line in fi:
                        if line.startswith(("mtllib", "usemtl")):
                            continue
                        fo.write(line)

    def write(self, asset: ET.Element):
        for key, name in sorted(self.name.items(), key=lambda kv: kv[1]):
            ET.SubElement(asset, "mesh", name=name, file=self.file[key])

    def __len__(self):
        return len(self.name)


class Materials:
    """Per-source material registry, names prefixed so room and maze files never collide."""

    def __init__(self, tag: str):
        self.tag = tag
        self.rgba: dict[str, tuple] = {}

    def for_body(self, b: BodyInstance) -> str:
        name = b.appearance or "default"
        rgba = APPEARANCE_RGBA.get(name)
        if rgba is None:
            kd, alpha, _ = read_mtl_kd(b.file)
            if kd is None:
                kd = [0.5, 0.5, 0.5]
            rgba = (kd[0], kd[1], kd[2], 1.0 if alpha is None else alpha)
        mat = f"mat_{self.tag}_{sanitize(name, 40)}"
        self.rgba.setdefault(mat, rgba)
        return mat

    def write(self, asset: ET.Element):
        for mat, rgba in sorted(self.rgba.items()):
            ET.SubElement(asset, "material", name=mat, rgba=fmt(rgba))


def visual_geom(parent, b: BodyInstance, mesh: str, material: str, dz: float = 0.0, name: str | None = None):
    pos = b.pos.copy()
    pos[2] += dz
    attrs = dict(type="mesh", mesh=mesh, pos=fmt(pos), quat=fmt(P.rotmat_to_quat(b.R)), material=material)
    if name:
        attrs["name"] = name
    return ET.SubElement(parent, "geom", **attrs)


def collision_box(parent, center: np.ndarray, size: np.ndarray, name: str):
    ET.SubElement(parent, "geom", {"class": "collision"}, name=name, pos=fmt(center), size=fmt(size / 2))


def select_bodies(bodies, *, exclude, include_hidden, root=None, not_root=None, skip_top_rx=None):
    """Filter manifest bodies by top-level occurrence, category and visibility."""
    sel, skipped = [], {"other_root": 0, "skipped_top": 0, "category": 0, "hidden": 0}
    for b in bodies:
        if root is not None and b.top_level != root:
            skipped["other_root"] += 1
            continue
        if not_root is not None and b.top_level == not_root:
            skipped["other_root"] += 1
            continue
        if skip_top_rx is not None and skip_top_rx.match(b.top_level):
            skipped["skipped_top"] += 1
            continue
        if b.category in exclude:
            skipped["category"] += 1
            continue
        if not (b.body_visible and b.occ_visible) and not include_hidden:
            skipped["hidden"] += 1
            continue
        sel.append(b)
    return sel, skipped


def wall_from_center(c: np.ndarray, frame: P.GridFrame) -> str:
    """Which wall a monitor sits at, named by the grid cue cell it faces."""
    gx, gy = frame.model_to_grid(c[0], c[1])
    # cue cells in the env: (11,6) +Y wall, (6,11) +X wall, (1,6) -Y wall, (6,1) -X wall
    cands = [(11, 6), (6, 11), (1, 6), (6, 1)]
    best = min(cands, key=lambda k: (k[0] - gx) ** 2 + (k[1] - gy) ** 2)
    return f"cue_x{best[0]}_y{best[1]}"


def add_screen_light(world, wall: str, screen_center: np.ndarray, maze_center: np.ndarray) -> str:
    """Spotlight standing in for the emissive cue screen (off by default)."""
    horiz = maze_center[:2] - screen_center[:2]
    horiz = horiz / np.linalg.norm(horiz)
    c45, s45 = math.cos(math.radians(45)), math.sin(math.radians(45))
    d = np.array([horiz[0] * c45, horiz[1] * c45, -s45])
    name = f"light_{wall}"
    ET.SubElement(world, "light", name=name, pos=fmt(screen_center + d * 0.01), dir=fmt(d), cutoff="80", exponent="1",
                  attenuation="1 0 0.6", diffuse="0.6 0.6 0.6", specular="0.2 0.2 0.2", ambient="0 0 0",
                  castshadow="true", active="false")
    return name


def calibrate_frame(bodies: list[BodyInstance], prof: MazeProfile) -> P.GridFrame:
    """Derive the grid frame from the track geometry, in the maze's own export coordinates.

    Reading of the maze (same as CaptureView's table for the 2.0 design, verified on it):
    cell 6 is the centre platform; the perimeter straight tracks form the ring at cells 2 and
    10; cells 3-5 and 7-9 are spaced evenly in between; cells 1 and 11 are the reward wells
    beyond the ring corners (also used as nominal points beyond the T-junctions, where the cue
    monitors stand). Model X follows grid y and model Y follows grid x.
    """
    def pick(rx):
        return [b for b in bodies if rx is not None and rx.match(b.component)]

    centers = pick(prof.center_rx)
    if len(centers) != 1:
        raise SystemExit(f"calibrate_frame: expected one centre slab, found {len(centers)} for {prof.center_rx.pattern!r}")
    c = centers[0].center
    ring = pick(prof.ring_rx)
    if len(ring) < 8:
        raise SystemExit(f"calibrate_frame: expected >= 8 straight slabs, found {len(ring)}")
    dx = np.array([abs(b.center[0] - c[0]) for b in ring])
    dy = np.array([abs(b.center[1] - c[1]) for b in ring])
    # perimeter slabs sit at the ring distance on one axis; cross-arm slabs are at about half of it,
    # so keep only the outermost cluster (within 10% of the farthest slab)
    half_x = float(dx[dx > 0.9 * dx.max()].mean())
    half_y = float(dy[dy > 0.9 * dy.max()].mean())
    wells = pick(prof.well_rx)
    if len(wells) >= 4:
        well_x = float(np.mean([abs(b.center[0] - c[0]) for b in wells]))
        well_y = float(np.mean([abs(b.center[1] - c[1]) for b in wells]))
    else:  # no well bodies: fall back to CaptureView's overhang beyond the ring (8.3 in)
        well_x = half_x + 8.297 * P.IN
        well_y = half_y + 8.297 * P.IN
    surface = pick(prof.surface_rx)
    track_top = float(max(b.bbox_max[2] for b in surface)) if surface else float(centers[0].bbox_max[2])
    return P.GridFrame(
        {1: c[0] - well_x, 2: c[0] - half_x, 6: c[0], 10: c[0] + half_x, 11: c[0] + well_x},
        {1: c[1] - well_y, 2: c[1] - half_y, 6: c[1], 10: c[1] + half_y, 11: c[1] + well_y},
        track_top,
    )


# ---- room -----------------------------------------------------------------------------------------

def build_room(manifest: Path, out: Path, *, exclude=frozenset(), include_hidden=False, copy_meshes=True,
               maze_root: str = RENDERING_MODEL_MAZE_ROOT, name: str = "room") -> dict:
    """Everything in the Rendering Model export except the maze subtree -> <name>.xml (+ meta)."""
    meta_in, bodies = load_manifest(manifest)
    out.mkdir(parents=True, exist_ok=True)
    sel, skipped = select_bodies(bodies, exclude=exclude, include_hidden=include_hidden, not_root=maze_root, skip_top_rx=MONITOR_CUED_RX)
    ref = P.CAPTUREVIEW  # the Rendering Model room was arranged around the CaptureView grid
    lib = MeshLib(name, out, copy_meshes)
    lib.register(sel)
    mats = Materials(name)

    root = ET.Element("mujocoinclude")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="flat", rgb1="0 0 0", rgb2="0 0 0", width="32", height="32")
    ET.SubElement(asset, "material", name="mat_floor", rgba="0.03 0.03 0.03 1")
    # blank monitor: dim gray emitter (~75/255 in the Fusion renders); cue: full white emitter
    ET.SubElement(asset, "material", name="screen_blank", rgba="0.3 0.3 0.3 1", emission="1", specular="0", shininess="0")
    ET.SubElement(asset, "material", name="screen_cue", rgba="1 1 1 1", emission="1", specular="0", shininess="0")

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="6 6 0.1", pos="1.4 1.35 0", material="mat_floor",
                  contype="1", conaffinity="1", group="0")
    screens: dict[str, str] = {}
    lights: dict[str, str] = {}
    maze_center = np.array([*ref.center_xy_m(), ref.track_top_m])
    for b in sel:
        if SCREEN_COMPONENT_RX.match(b.component) and MONITOR_BLANK_RX.match(b.top_level) and b.body == "Body2":
            wall = wall_from_center(b.center, ref)
            gname = f"screen_{wall}"
            screens[wall] = gname
            visual_geom(world, b, lib.name[b.key], "screen_blank", name=gname)
            lights[wall] = add_screen_light(world, wall, b.center, maze_center)
            continue
        visual_geom(world, b, lib.name[b.key], mats.for_body(b))
    lib.write(asset)
    mats.write(asset)
    write_xml(root, out / f"{name}.xml")

    meta = {
        "name": name,
        "source_document": meta_in.get("document"),
        "manifest": str(manifest),
        "maze_root_excluded": maze_root,
        "bodies_selected": len(sel),
        "skipped": skipped,
        "mesh_assets": len(lib),
        "screens": screens,
        "lights": lights,
        # grid centre the room was arranged around (mazes are centred on it) and the floor plane height
        "reference": ref.to_meta(),
        "floor_z_m": 0.0,
    }
    (out / f"{name}_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


# ---- maze -----------------------------------------------------------------------------------------

def build_maze(manifest: Path, out: Path, *, tag: str, root: str | None = None, profile: str = "v2",
               exclude=frozenset(), include_hidden=False, copy_meshes=True, room_meta: Path | None = None) -> dict:
    """One maze design -> maze_<tag>.xml (+ meta), placed on the room's reference point."""
    prof = PROFILES[profile]
    meta_in, bodies = load_manifest(manifest)
    out.mkdir(parents=True, exist_ok=True)
    sel, skipped = select_bodies(bodies, exclude=exclude, include_hidden=include_hidden, root=root, skip_top_rx=MONITOR_CUED_RX)
    if not sel:
        raise SystemExit(f"no bodies selected from {manifest} (root={root!r})")

    # the maze's grid frame in its own export coordinates, then the shift onto the room's reference
    frame = P.CAPTUREVIEW if prof.frame == "captureview" else calibrate_frame(sel, prof)
    ref, ref_floor = P.CAPTUREVIEW, 0.0
    if room_meta is not None and Path(room_meta).exists():
        rm = json.loads(Path(room_meta).read_text())
        ref, ref_floor = P.GridFrame.from_meta(rm["reference"]), float(rm.get("floor_z_m", 0.0))
    # XY: grid centre onto the room's reference point. Z: the design's floor onto the room floor, so each
    # maze keeps its real height (decision 2026-10-09; the 2.0 and behavioral surfaces differ by 15 cm).
    offset = np.array([*(ref.center_xy_m() - frame.center_xy_m()), ref_floor - prof.floor_z_m])

    lib = MeshLib(tag, out, copy_meshes)
    lib.register(sel)
    mats = Materials(tag)

    barriers: dict[str, dict] = {}
    static: list[BodyInstance] = []
    for b in sel:
        mm = prof.barrier_rx.match(b.path)
        if mm:
            bar = barriers.setdefault(mm.group("prefix"), {"moving": [], "static": [], "panel": None})
            if prof.moving_rx.search(b.path):
                bar["moving"].append(b)
                if prof.panel_rx.search(b.component):
                    bar["panel"] = b
            else:
                bar["static"].append(b)
        else:
            static.append(b)

    root_el = ET.Element("mujocoinclude")
    asset = ET.SubElement(root_el, "asset")
    world = ET.SubElement(root_el, "worldbody")
    wrap = ET.SubElement(world, "body", name=f"maze_{tag}", pos=fmt(offset))
    n_col = 0

    for b in static:
        visual_geom(wrap, b, lib.name[b.key], mats.for_body(b))
        if prof.track_slab_rx.match(b.component) or prof.track_wall_rx.search(b.component):
            if is_axis_aligned(b.R):
                collision_box(wrap, b.center, b.size, f"col_{sanitize(b.component, 24)}_{b.id}")
                n_col += 1

    barrier_meta = []
    for prefix, bar in sorted(barriers.items()):
        panel = bar["panel"]
        if panel is None:
            for b in bar["moving"] + bar["static"]:
                visual_geom(wrap, b, lib.name[b.key], mats.for_body(b))
            continue
        raised = panel.bbox_max[2] > frame.track_top_m + 0.02
        dz = -prof.travel_m if raised else 0.0
        gx, gy = frame.model_to_grid(panel.center[0], panel.center[1])
        bname = f"barrier_x{int(round(gx))}_y{int(round(gy))}"
        body = ET.SubElement(wrap, "body", name=bname, pos="0 0 0")
        ET.SubElement(body, "joint", name=bname, type="slide", axis="0 0 1", range=f"0 {prof.travel_m:.6g}", damping="20")
        ET.SubElement(body, "inertial", pos=fmt(panel.center + [0, 0, dz]), mass="2.0", diaginertia="0.02 0.02 0.02")
        for b in bar["moving"]:
            visual_geom(body, b, lib.name[b.key], mats.for_body(b), dz=dz)
        pc = panel.center.copy()
        pc[2] += dz
        collision_box(body, pc, panel.size, f"col_{bname}")
        n_col += 1
        for b in bar["static"]:
            visual_geom(wrap, b, lib.name[b.key], mats.for_body(b))
        barrier_meta.append({"name": bname, "grid": [int(round(gx)), int(round(gy))], "grid_exact": [round(gx, 3), round(gy, 3)],
                             "snapshot_raised": bool(raised), "panel_center_lowered_m": [round(float(v), 5) for v in pc + offset]})

    lib.write(asset)
    mats.write(asset)
    act = ET.SubElement(root_el, "actuator")
    for bm in barrier_meta:
        ET.SubElement(act, "position", name=f"act_{bm['name']}", joint=bm["name"], kp="2000", kv="100", ctrlrange=f"0 {prof.travel_m:.6g}")
    write_xml(root_el, out / f"maze_{tag}.xml")

    placed = frame.translated(offset)
    meta = {
        "tag": tag,
        "source_document": meta_in.get("document"),
        "manifest": str(manifest),
        "root": root,
        "profile": profile,
        "offset_m": [round(float(v), 6) for v in offset],
        "placement": "grid centre on the room reference; design floor on the room floor",
        "lowest_point_m": round(float(min(b.bbox_min[2] for b in sel) + offset[2]), 4),
        "frame": placed.to_meta(),
        "track_top_m": placed.track_top_m,
        "eye_z_m": placed.eye_z_m,
        "bodies_selected": len(sel),
        "skipped": skipped,
        "mesh_assets": len(lib),
        "collision_boxes": n_col,
        "barriers": barrier_meta,
        "barrier_travel_m": prof.travel_m,
    }
    (out / f"maze_{tag}_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


# ---- compose --------------------------------------------------------------------------------------

def compose(out: Path, *, maze: str, room: str = "room", name: str = "corner_maze") -> dict:
    """Settings + <include> room + <include> maze_<tag> + rat head -> <name>.xml and <name>_meta.json."""
    room_meta = json.loads((out / f"{room}_meta.json").read_text())
    maze_meta = json.loads((out / f"maze_{maze}_meta.json").read_text())
    frame = P.GridFrame.from_meta(maze_meta["frame"])

    root = ET.Element("mujoco", model=name)
    ET.SubElement(root, "compiler", angle="radian", meshdir="meshes", autolimits="true")
    ET.SubElement(root, "option", gravity="0 0 -9.81", timestep="0.002")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth="1024", offheight="1024")
    # The real room is dark; the monitors are the only light sources. Keep the headlight *active*
    # with zero terms: an inactive headlight leaves OpenGL's default global ambient in place.
    ET.SubElement(vis, "headlight", ambient="0 0 0", diffuse="0 0 0", specular="0 0 0", active="1")
    ET.SubElement(vis, "quality", shadowsize="4096")
    ET.SubElement(vis, "map", znear="0.005", zfar="50")
    dflt = ET.SubElement(root, "default")
    ET.SubElement(dflt, "geom", contype="0", conaffinity="0", group="1")
    dcol = ET.SubElement(dflt, "default", {"class": "collision"})
    ET.SubElement(dcol, "geom", contype="1", conaffinity="1", group="3", rgba="1 0 0 0.3", type="box")
    dcam = ET.SubElement(dflt, "default", {"class": "eye"})
    ET.SubElement(dcam, "camera", fovy=f"{P.EYE_VFOV_DEG:g}")

    ET.SubElement(root, "include", file=f"{room}.xml")
    ET.SubElement(root, "include", file=f"maze_{maze}.xml")

    world = ET.SubElement(root, "worldbody")
    hp, hq = frame.head_pose(6, 6, 0)
    head = ET.SubElement(world, "body", name="rat_head", mocap="true", pos=fmt(hp), quat=fmt(hq))
    ET.SubElement(head, "site", name="eye_center", size="0.005", rgba="0 1 0 1", group="4")
    for cname, fr in P.eye_local_frames().items():
        ET.SubElement(head, "camera", {"class": "eye"}, name=cname, pos=fmt(fr["pos"]), quat=fmt(fr["quat"]))
    # plan view from inside the curtain enclosure (ceiling cloth at 2.24 m, floor at 0): wide enough
    # to frame the 2.6 m ring from 1.25 m above the track
    c = frame.center_xy_m()
    ET.SubElement(world, "camera", name="overhead", pos=fmt([c[0], c[1], frame.track_top_m + 1.25]), quat="1 0 0 0", fovy="100")
    write_xml(root, out / f"{name}.xml")

    meta = {
        "name": name,
        "room": room,
        "maze": maze,
        "source_documents": {"room": room_meta.get("source_document"), "maze": maze_meta.get("source_document")},
        "frame": maze_meta["frame"],
        "eye_z_m": frame.eye_z_m,
        "track_top_m": frame.track_top_m,
        "barriers": maze_meta["barriers"],
        "barrier_travel_m": maze_meta["barrier_travel_m"],
        "screens": room_meta["screens"],
        "lights": room_meta["lights"],
        "cameras": ["eye_left", "eye_right", "overhead"],
        "head_body": "rat_head",
        "mesh_assets": {"room": room_meta["mesh_assets"], "maze": maze_meta["mesh_assets"]},
        "collision_boxes": maze_meta["collision_boxes"],
    }
    (out / f"{name}_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


# ---- CLI --------------------------------------------------------------------------------------------

def _summary(meta: dict) -> str:
    return json.dumps({k: v for k, v in meta.items() if k not in ("barriers", "frame")}, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--out", default="model")
        p.add_argument("--exclude", default="", help="comma-separated categories to drop, e.g. pump,fastener")
        p.add_argument("--include-hidden", action="store_true", help="also instance bodies hidden in Fusion")
        p.add_argument("--no-copy", action="store_true", help="do not (re)copy mesh files")

    p = sub.add_parser("room", help="room = Rendering Model export minus the maze subtree")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--maze-root", default=RENDERING_MODEL_MAZE_ROOT, help="top-level occurrence to leave out")
    common(p)

    p = sub.add_parser("maze", help="one maze version from an export")
    p.add_argument("--manifest", required=True)
    p.add_argument("--tag", required=True, help="short name, e.g. v2; files become maze_<tag>.xml")
    p.add_argument("--root", default=None, help="top-level occurrence holding the maze; omit when the export is the maze alone")
    p.add_argument("--profile", default="v2", choices=sorted(PROFILES))
    common(p)

    p = sub.add_parser("compose", help="room + one maze + rat head -> loadable model")
    p.add_argument("--maze", required=True, help="tag of a built maze")
    p.add_argument("--room", default="room")
    p.add_argument("--name", default="corner_maze", help="output model name (<name>.xml, <name>_meta.json)")
    p.add_argument("--out", default="model")

    p = sub.add_parser("all", help="room + maze v2 + compose, from the Rendering Model export")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    common(p)

    a = ap.parse_args(argv)
    out = Path(a.out)
    exclude = {s.strip() for s in getattr(a, "exclude", "").split(",") if s.strip()}
    kw = dict(exclude=exclude, include_hidden=getattr(a, "include_hidden", False), copy_meshes=not getattr(a, "no_copy", False))

    if a.cmd == "room":
        print(_summary(build_room(Path(a.manifest), out, maze_root=a.maze_root, **kw)))
    elif a.cmd == "maze":
        m = build_maze(Path(a.manifest), out, tag=a.tag, root=a.root, profile=a.profile, room_meta=out / "room_meta.json", **kw)
        print(_summary(m))
        print(f"barriers: {len(m['barriers'])}  raised in snapshot: {sum(b['snapshot_raised'] for b in m['barriers'])}")
    elif a.cmd == "compose":
        print(_summary(compose(out, maze=a.maze, room=a.room, name=a.name)))
    elif a.cmd == "all":
        r = build_room(Path(a.manifest), out, **kw)
        m = build_maze(Path(a.manifest), out, tag="v2", root=RENDERING_MODEL_MAZE_ROOT, profile="v2", room_meta=out / "room_meta.json", **kw)
        c = compose(out, maze="v2")
        print(f"room: {r['bodies_selected']} bodies, {r['mesh_assets']} meshes, screens {list(r['screens'])}")
        print(f"maze v2: {m['bodies_selected']} bodies, {m['mesh_assets']} meshes, {len(m['barriers'])} barriers, offset {m['offset_m']}")
        print(f"composed: {out / 'corner_maze.xml'}  eye_z={c['eye_z_m']:.4f}")


if __name__ == "__main__":
    main()
