"""Generate the MuJoCo model (MJCF + deduplicated meshes) from the Fusion export.

    python -m corner_maze_mujoco.build_mjcf \
        --manifest assets/fusion_export/manifest.json --out model

Design
------
* Every exported body becomes a visual mesh geom (contype/conaffinity 0),
  placed with its occurrence transform. Identical parts share one mesh asset.
* Collision geometry is boxes derived from world bounding boxes of track
  pieces and barrier panels (all axis-aligned in this rig).
* The 24 barrier mechanisms become bodies with a vertical slide joint; the
  moving subtree is the "Saddle and Plexiglass Assembly". Barriers that the
  Fusion snapshot has raised are normalised to lowered, so joint 0 = lowered
  and joint 0.4064 m (16 in) = raised.
* The four blank monitors are kept; their screen geoms are named so a cue can
  be shown by swapping the geom's material at runtime. Hidden "Cued" monitor
  assemblies are not instanced (they duplicate the blank ones in place).
* A mocap body "rat_head" carries the two eye cameras from pose.py.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from . import pose as P
from .fusion_manifest import BodyInstance, asset_name, group_by_key, load_manifest, read_mtl_kd, sanitize

BARRIER_RX = re.compile(r"^(?P<prefix>.*\+New Barrier Assembly v\d+:\d+)")
MOVING_RX = re.compile(r"Saddle and Plexiglass Assembly")
PANEL_RX = re.compile(r"New Plexiglass Barrier")
TRACK_SLAB_RX = re.compile(r"^(Straight Track|Track Corner|Side Intersection|Center Intersection) v\d+$")
TRACK_WALL_RX = re.compile(r"Track Wall|Track: Wall|Straight Track Wall")
MONITOR_BLANK_RX = re.compile(r"^Monitor Moount Assembly Angel v\d+:\d+$")
MONITOR_CUED_RX = re.compile(r"^Cued Monitor")
SCREEN_COMPONENT_RX = re.compile(r"^24” Monitor v\d+$|^24\" Monitor v\d+$")

BARRIER_TRAVEL_M = 16.0 * P.IN
TRACK_TOP_M = P.TRACK_TOP_IN * P.IN

# appearance name -> rgba override (Fusion appearances that the MTL Kd misrepresents)
APPEARANCE_RGBA = {
    "Fabric (Black)": (0.02, 0.02, 0.02, 1.0),
    "Track Black": (0.04, 0.04, 0.04, 1.0),
    "Acrylic (Clear)": (0.85, 0.9, 0.95, 0.12),
}


def fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in np.atleast_1d(v))


def is_axis_aligned(R: np.ndarray, tol=1e-6) -> bool:
    return np.all(np.isclose(np.abs(R), np.round(np.abs(R)), atol=tol))


def material_for(b: BodyInstance, materials: dict) -> str:
    """Return material name for a body, registering it if new."""
    name = b.appearance or "default"
    kd, alpha, _ = read_mtl_kd(b.file)
    rgba = APPEARANCE_RGBA.get(name)
    if rgba is None:
        if kd is None:
            kd = [0.5, 0.5, 0.5]
        rgba = (kd[0], kd[1], kd[2], 1.0 if alpha is None else alpha)
    mat = "mat_" + sanitize(name, 40)
    materials.setdefault(mat, rgba)
    return mat


def wall_from_center(c: np.ndarray) -> str:
    """Which wall a monitor sits at, named by the grid cue cell it faces."""
    gx, gy = P.model_to_grid(c[0], c[1])
    # cue cells in the env: (11,6) +Y wall, (6,11) +X wall, (1,6) -Y wall, (6,1) -X wall
    cands = {(11, 6): None, (6, 11): None, (1, 6): None, (6, 1): None}
    best = min(cands, key=lambda k: (k[0] - gx) ** 2 + (k[1] - gy) ** 2)
    return f"cue_x{best[0]}_y{best[1]}"


def build(manifest: Path, out: Path, exclude: set[str], include_hidden: bool, copy_meshes: bool) -> dict:
    meta_in, bodies = load_manifest(manifest)
    out.mkdir(parents=True, exist_ok=True)
    mesh_out = out / "meshes"
    mesh_out.mkdir(exist_ok=True)

    # ---- selection -------------------------------------------------------
    selected: list[BodyInstance] = []
    skipped = {"category": 0, "hidden": 0, "cued_monitor": 0}
    for b in bodies:
        if b.category in exclude:
            skipped["category"] += 1
            continue
        if MONITOR_CUED_RX.match(b.top_level):
            skipped["cued_monitor"] += 1
            continue
        if not (b.body_visible and b.occ_visible) and not include_hidden:
            skipped["hidden"] += 1
            continue
        selected.append(b)

    # ---- mesh assets -----------------------------------------------------
    groups = group_by_key(selected)
    assets: dict[tuple, str] = {}
    for key, insts in groups.items():
        name = asset_name(key)
        assets[key] = name
        dst = mesh_out / f"{name}.obj"
        if copy_meshes and not dst.exists():
            with open(insts[0].file, "r", errors="ignore") as fi, open(dst, "w") as fo:
                for line in fi:
                    if line.startswith("mtllib") or line.startswith("usemtl"):
                        continue
                    fo.write(line)

    # ---- barrier grouping --------------------------------------------------
    barriers: dict[str, dict] = {}
    static: list[BodyInstance] = []
    for b in selected:
        mm = BARRIER_RX.match(b.path)
        if mm:
            bar = barriers.setdefault(mm.group("prefix"), {"moving": [], "static": [], "panel": None})
            if MOVING_RX.search(b.path):
                bar["moving"].append(b)
                if PANEL_RX.search(b.component):
                    bar["panel"] = b
            else:
                bar["static"].append(b)
        else:
            static.append(b)

    # ---- XML ----------------------------------------------------------------
    root = ET.Element("mujoco", model="corner_maze")
    ET.SubElement(root, "compiler", angle="radian", meshdir="meshes", autolimits="true")
    ET.SubElement(root, "option", gravity="0 0 -9.81", timestep="0.002")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth="1024", offheight="1024")
    # The real room is dark; the monitors are the only light sources.
    # Keep the headlight *active* with zero terms: an inactive headlight leaves
    # OpenGL's default global ambient in place and faintly lights everything.
    ET.SubElement(vis, "headlight", ambient="0 0 0", diffuse="0 0 0", specular="0 0 0", active="1")
    ET.SubElement(vis, "quality", shadowsize="4096")
    ET.SubElement(vis, "map", znear="0.005", zfar="50")

    dflt = ET.SubElement(root, "default")
    ET.SubElement(dflt, "geom", contype="0", conaffinity="0", group="1")
    dcol = ET.SubElement(dflt, "default", {"class": "collision"})
    ET.SubElement(dcol, "geom", contype="1", conaffinity="1", group="3", rgba="1 0 0 0.3", type="box")
    dcam = ET.SubElement(dflt, "default", {"class": "eye"})
    ET.SubElement(dcam, "camera", fovy=f"{P.EYE_VFOV_DEG:g}")

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="flat", rgb1="0 0 0", rgb2="0 0 0", width="32", height="32")
    materials: dict[str, tuple] = {}
    ET.SubElement(asset, "material", name="mat_floor", rgba="0.03 0.03 0.03 1")
    # blank monitor: dim gray emitter (~75/255 in the Fusion renders); cue: full white emitter
    ET.SubElement(asset, "material", name="screen_blank", rgba="0.3 0.3 0.3 1", emission="1", specular="0", shininess="0")
    ET.SubElement(asset, "material", name="screen_cue", rgba="1 1 1 1", emission="1", specular="0", shininess="0")
    for key, name in sorted(assets.items(), key=lambda kv: kv[1]):
        ET.SubElement(asset, "mesh", name=name, file=f"{name}.obj")

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="6 6 0.1", pos="1.4 1.35 0", material="mat_floor", contype="1", conaffinity="1", group="0")

    screens: dict[str, str] = {}
    lights: dict[str, str] = {}
    n_collision = 0
    maze_center = np.array([*P.grid_to_model_m(6, 6), TRACK_TOP_M])

    def add_screen_light(wall: str, screen_center: np.ndarray):
        """Spotlight standing in for the emissive cue screen (off by default)."""
        horiz = maze_center[:2] - screen_center[:2]
        horiz = horiz / np.linalg.norm(horiz)
        d = np.array([horiz[0] * math.cos(math.radians(45)), horiz[1] * math.cos(math.radians(45)), -math.sin(math.radians(45))])
        name = f"light_{wall}"
        ET.SubElement(world, "light", name=name, pos=fmt(screen_center + d * 0.01), dir=fmt(d), cutoff="80", exponent="1",
                      attenuation="1 0 0.6", diffuse="0.6 0.6 0.6", specular="0.2 0.2 0.2", ambient="0 0 0",
                      castshadow="true", active="false")
        lights[wall] = name

    def add_visual(parent, b: BodyInstance, dz: float = 0.0, name: str | None = None):
        mat = material_for(b, materials)
        pos = b.pos.copy()
        pos[2] += dz
        attrs = dict(type="mesh", mesh=assets[b.key], pos=fmt(pos), quat=fmt(P.rotmat_to_quat(b.R)), material=mat)
        if name:
            attrs["name"] = name
        return ET.SubElement(parent, "geom", **attrs)

    def add_box(parent, center: np.ndarray, size: np.ndarray, name: str):
        nonlocal n_collision
        n_collision += 1
        ET.SubElement(parent, "geom", {"class": "collision"}, name=name, pos=fmt(center), size=fmt(size / 2))

    # static visuals + track collisions
    for b in static:
        gname = None
        if SCREEN_COMPONENT_RX.match(b.component) and MONITOR_BLANK_RX.match(b.top_level) and b.body == "Body2":
            wall = wall_from_center(b.center)
            gname = f"screen_{wall}"
            screens[wall] = gname
            g = add_visual(world, b, name=gname)
            g.set("material", "screen_blank")
            add_screen_light(wall, b.center)
            continue
        add_visual(world, b)
        if TRACK_SLAB_RX.match(b.component) or TRACK_WALL_RX.search(b.component):
            if not is_axis_aligned(b.R):
                continue
            add_box(world, b.center, b.size, f"col_{sanitize(b.component, 24)}_{b.id}")

    # barriers
    barrier_meta = []
    for prefix, bar in sorted(barriers.items()):
        panel = bar["panel"]
        if panel is None:
            for b in bar["moving"] + bar["static"]:
                add_visual(world, b)
            continue
        raised = panel.bbox_max[2] > TRACK_TOP_M + 0.02
        dz = -BARRIER_TRAVEL_M if raised else 0.0
        gx, gy = P.model_to_grid(panel.center[0], panel.center[1])
        bname = f"barrier_x{int(round(gx))}_y{int(round(gy))}"
        body = ET.SubElement(world, "body", name=bname, pos="0 0 0")
        ET.SubElement(body, "joint", name=bname, type="slide", axis="0 0 1", range=f"0 {BARRIER_TRAVEL_M:.6g}", damping="20")
        ET.SubElement(body, "inertial", pos=fmt(panel.center + [0, 0, dz]), mass="2.0", diaginertia="0.02 0.02 0.02")
        for b in bar["moving"]:
            add_visual(body, b, dz=dz)
        pc = panel.center.copy()
        pc[2] += dz
        add_box(body, pc, panel.size, f"col_{bname}")
        for b in bar["static"]:
            add_visual(world, b)
        barrier_meta.append({"name": bname, "grid": [int(round(gx)), int(round(gy))], "grid_exact": [round(gx, 3), round(gy, 3)],
                             "snapshot_raised": bool(raised), "panel_center_lowered_m": [round(v, 5) for v in pc]})

    # rat head with eye cameras
    hp, hq = P.head_pose(6, 6, 0)
    head = ET.SubElement(world, "body", name="rat_head", mocap="true", pos=fmt(hp), quat=fmt(hq))
    ET.SubElement(head, "site", name="eye_center", size="0.005", rgba="0 1 0 1", group="4")
    for cname, fr in P.eye_local_frames().items():
        ET.SubElement(head, "camera", {"class": "eye"}, name=cname, pos=fmt(fr["pos"]), quat=fmt(fr["quat"]))
    ET.SubElement(world, "camera", name="overhead", pos="1.4 1.35 4.0", quat="1 0 0 0", fovy="60")

    # materials collected during geom creation
    for mat, rgba in sorted(materials.items()):
        ET.SubElement(asset, "material", name=mat, rgba=fmt(rgba))

    # actuators: hold barriers at a commanded height
    act = ET.SubElement(root, "actuator")
    for bm in barrier_meta:
        ET.SubElement(act, "position", name=f"act_{bm['name']}", joint=bm["name"], kp="2000", kv="100", ctrlrange=f"0 {BARRIER_TRAVEL_M:.6g}")

    ET.indent(root, space="  ")
    xml_path = out / "corner_maze.xml"
    ET.ElementTree(root).write(xml_path, encoding="unicode", xml_declaration=False)

    meta = {
        "source_document": meta_in.get("document"),
        "bodies_in_manifest": len(bodies),
        "bodies_selected": len(selected),
        "skipped": skipped,
        "mesh_assets": len(assets),
        "collision_boxes": n_collision,
        "barriers": barrier_meta,
        "barrier_travel_m": BARRIER_TRAVEL_M,
        "screens": screens,
        "lights": lights,
        "cameras": ["eye_left", "eye_right", "overhead"],
        "head_body": "rat_head",
        "eye_z_m": P.EYE_Z_M,
        "track_top_m": TRACK_TOP_M,
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="assets/fusion_export/manifest.json")
    ap.add_argument("--out", default="model")
    ap.add_argument("--exclude", default="", help="comma-separated categories to drop, e.g. pump,fastener")
    ap.add_argument("--include-hidden", action="store_true", help="also instance bodies hidden in Fusion")
    ap.add_argument("--no-copy", action="store_true", help="do not (re)copy mesh files")
    a = ap.parse_args(argv)
    exclude = {s.strip() for s in a.exclude.split(",") if s.strip()}
    meta = build(Path(a.manifest), Path(a.out), exclude, a.include_hidden, not a.no_copy)
    print(json.dumps({k: v for k, v in meta.items() if k != "barriers"}, indent=1))
    print(f"barriers: {len(meta['barriers'])}  raised in snapshot: {sum(b['snapshot_raised'] for b in meta['barriers'])}")


if __name__ == "__main__":
    main()
