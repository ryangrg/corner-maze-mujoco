"""Fusion script: export every body of the active design to OBJ + manifest.

Read-only with respect to the design. For each BRepBody (in assembly context,
i.e. world frame) it writes one OBJ file and records, in manifest.json:
occurrence path, component, source document, body name, visibility, world
bounding box, occurrence transform, appearance name/colour, a category hint,
and the bbox centre mapped to CornerMaze grid coordinates.

Configuration: an optional export_config.json next to this script, see
export_config.example.json. Defaults write to <script_dir>/export_out.

Progress is appended to <out_dir>/export_log.txt so it can be tailed from a
shell while Fusion's UI is blocked.
"""
import adsk.core
import adsk.fusion
import json
import os
import re
import time
import traceback

CM_PER_IN = 2.54
TRACK_TOP_IN = 33.577  # track surface height used by CaptureView

# CaptureView grid anchors (inches). Model X comes from grid y, model Y from grid x.
X_FROM_GRID_Y_IN = {1: 8.039, 2: 16.336, 6: 55.519, 10: 94.702, 11: 102.999}
Y_FROM_GRID_X_IN = {1: 6.02, 2: 14.317, 6: 53.50, 10: 92.683, 11: 100.98}

CATEGORY_PATTERNS = [
    ("track", r"Straight Track|Track Corner|Side Intersection|Center Intersection|Track Wall|Wall Corner|Connecting Plate"),
    ("barrier_panel", r"New Plexiglass Barrier"),
    ("barrier_mech", r"Barrier: |Stand Assembly|Lead Screw|Guide Shaft|LM6UU|KFL08|coupling|Stepper|Electrical Box|608RS"),
    ("monitor", r"Monitor|OverheadPlatform|Platform with"),
    ("curtain", r"Curtain"),
    ("structure", r"Cross Beam|Corner Support|Corner Post|Shim|Post Base"),
    ("pump", r"Syringe|Pump|0402|0805|SOT|DIODO|CAPC|LEDC|bornera|SOD128|header|capacitor|Tube"),
    ("fastener", r"Screw|Washer|BRACKET|Nut"),
]


def _cfg(script_dir):
    cfg = {
        "out_dir": os.path.join(script_dir, "export_out"),
        "refinement": "medium",
        "units": "meter",
        "include_hidden": True,
        "skip_existing": True,
        "max_bodies": 0,
    }
    p = os.path.join(script_dir, "export_config.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def _inv(v, knots):
    ks = sorted(knots)
    for a, b in zip(ks, ks[1:]):
        if knots[a] - 1e-6 <= v <= knots[b] + 1e-6:
            return a + (v - knots[a]) / (knots[b] - knots[a]) * (b - a)
    return None


def _bbox(ent):
    try:
        bb = ent.boundingBox
        return {
            "min_cm": [bb.minPoint.x, bb.minPoint.y, bb.minPoint.z],
            "max_cm": [bb.maxPoint.x, bb.maxPoint.y, bb.maxPoint.z],
        }
    except Exception:
        return None


def _grid_center(bb):
    if not bb:
        return None
    cx = (bb["min_cm"][0] + bb["max_cm"][0]) / 2 / CM_PER_IN
    cy = (bb["min_cm"][1] + bb["max_cm"][1]) / 2 / CM_PER_IN
    gx = _inv(cy, Y_FROM_GRID_X_IN)
    gy = _inv(cx, X_FROM_GRID_Y_IN)
    return [gx, gy]


def _appearance(body):
    out = {"name": None, "rgba": None, "source": None}
    try:
        ap = body.appearance
        if ap is None:
            return out
        out["name"] = ap.name
        try:
            out["source"] = int(body.appearanceSourceType)
        except Exception:
            pass
        for pid in ("opaque_albedo", "surface_albedo", "transparent_color", "generic_diffuse", "metal_f0"):
            try:
                prop = ap.appearanceProperties.itemById(pid)
                if prop is not None and hasattr(prop, "value") and prop.value is not None:
                    c = prop.value
                    ok, r, g, b, a = c.getColor()
                    if ok:
                        out["rgba"] = [r, g, b, a]
                        out["rgba_prop"] = pid
                        break
            except Exception:
                continue
    except Exception:
        pass
    return out


def _category(text):
    for name, rx in CATEGORY_PATTERNS:
        if re.search(rx, text, re.I):
            return name
    return "other"


def _sanitize(s, n=60):
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s[:n] if s else "x"


def run(context):
    app = adsk.core.Application.get()
    ui = app.userInterface
    try:
        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            ui.messageBox("No active design.")
            return
        script_dir = os.path.dirname(os.path.realpath(__file__))
        cfg = _cfg(script_dir)
        out_dir = cfg["out_dir"]
        mesh_dir = os.path.join(out_dir, "meshes")
        os.makedirs(mesh_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "export_log.txt")

        def log(msg):
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(time.strftime("%H:%M:%S ") + msg + "\n")

        refinement = {
            "high": adsk.fusion.MeshRefinementSettings.MeshRefinementHigh,
            "medium": adsk.fusion.MeshRefinementSettings.MeshRefinementMedium,
            "low": adsk.fusion.MeshRefinementSettings.MeshRefinementLow,
        }[cfg["refinement"]]
        units = {
            "meter": adsk.fusion.DistanceUnits.MeterDistanceUnits,
            "centimeter": adsk.fusion.DistanceUnits.CentimeterDistanceUnits,
            "millimeter": adsk.fusion.DistanceUnits.MillimeterDistanceUnits,
            "inch": adsk.fusion.DistanceUnits.InchDistanceUnits,
        }[cfg["units"]]

        root = design.rootComponent
        em = design.exportManager
        manifest = {
            "document": app.activeDocument.name,
            "design_units": design.unitsManager.defaultLengthUnits,
            "export_units": cfg["units"],
            "refinement": cfg["refinement"],
            "track_top_in": TRACK_TOP_IN,
            "grid_anchors_in": {"x_from_grid_y": X_FROM_GRID_Y_IN, "y_from_grid_x": Y_FROM_GRID_X_IN},
            "frame_note": "OBJ vertices are in the design's world frame (assembly context), converted to export_units.",
            "bodies": [],
            "failures": [],
        }
        log(f"=== export start: {manifest['document']} -> {out_dir}")

        idx = 0
        n_done = 0
        n_skip = 0
        t0 = time.time()
        for occ in root.allOccurrences:
            comp = occ.component
            try:
                bodies = occ.bRepBodies
            except Exception:
                continue
            try:
                tf = list(occ.transform2.asArray())
            except Exception:
                tf = None
            src = None
            try:
                src = comp.parentDesign.parentDocument.name
            except Exception:
                pass
            for i in range(bodies.count):
                body = bodies.item(i)
                idx += 1
                if cfg["max_bodies"] and idx > cfg["max_bodies"]:
                    break
                visible = bool(body.isVisible)
                if not visible and not cfg["include_hidden"]:
                    n_skip += 1
                    continue
                fname = f"{idx:04d}_{_sanitize(comp.name, 40)}_{_sanitize(body.name, 20)}.obj"
                fpath = os.path.join(mesh_dir, fname)
                entry = {
                    "id": idx,
                    "file": "meshes/" + fname,
                    "path": occ.fullPathName,
                    "component": comp.name,
                    "source_document": src,
                    "body": body.name,
                    "body_visible": visible,
                    "occ_visible": bool(occ.isVisible),
                    "occ_light_bulb_on": bool(occ.isLightBulbOn),
                    "category": _category(occ.fullPathName + " " + body.name),
                    "bbox_world": _bbox(body),
                    "occ_transform_rowmajor_cm": tf,
                    "appearance": _appearance(body),
                }
                entry["grid_center"] = _grid_center(entry["bbox_world"])
                if cfg["skip_existing"] and os.path.exists(fpath) and os.path.getsize(fpath) > 0:
                    entry["exported"] = "existing"
                    manifest["bodies"].append(entry)
                    continue
                try:
                    opts = em.createOBJExportOptions(body, fpath)
                    opts.meshRefinement = refinement
                    opts.unitType = units
                    opts.isOneFilePerBody = False
                    opts.isIncludingInvisibleBodies = True
                    opts.isIncludingInvisibleComponents = True
                    ok = em.execute(opts)
                    entry["exported"] = bool(ok) and os.path.exists(fpath)
                    if not entry["exported"]:
                        manifest["failures"].append({"id": idx, "path": occ.fullPathName, "body": body.name, "reason": "execute returned False or no file"})
                except Exception:
                    entry["exported"] = False
                    manifest["failures"].append({"id": idx, "path": occ.fullPathName, "body": body.name, "reason": traceback.format_exc().splitlines()[-1]})
                manifest["bodies"].append(entry)
                n_done += 1
                if n_done % 25 == 0:
                    log(f"{n_done} exported ({idx} seen, {len(manifest['failures'])} failures, {time.time()-t0:.0f}s)")
                    with open(os.path.join(out_dir, "manifest.partial.json"), "w", encoding="utf-8") as f:
                        json.dump(manifest, f)
                    adsk.doEvents()
            if cfg["max_bodies"] and idx > cfg["max_bodies"]:
                break

        manifest["summary"] = {
            "bodies_seen": idx,
            "exported_now": n_done,
            "skipped_hidden": n_skip,
            "failures": len(manifest["failures"]),
            "seconds": round(time.time() - t0, 1),
        }
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1)
        try:
            os.remove(os.path.join(out_dir, "manifest.partial.json"))
        except Exception:
            pass
        log(f"=== export done: {manifest['summary']}")
        ui.messageBox(f"Export finished.\n{json.dumps(manifest['summary'], indent=1)}\n{out_dir}")
    except Exception:
        if ui:
            ui.messageBox("Failed:\n{}".format(traceback.format_exc()))
