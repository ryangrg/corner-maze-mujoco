"""Read-only: dump the active Fusion design's structure to JSON.

Writes <script_dir>/tree_dump.json. Does not modify the design.

Captured:
  - document name, root component, default length unit, user parameters
  - every occurrence: full path (assembly hierarchy), component name,
    source document (linked/XRef), referenced flag, light-bulb + effective
    visibility, grounded flag, world 4x4 transform, world bounding box (cm),
    per-body names / visibility / world bbox (capped per occurrence)
  - joints and as-built joints: type, connected occurrences, slide/rotation
    axis vectors, slide/rotation limits
  - rigid groups: member occurrence paths
"""
import adsk.core
import adsk.fusion
import json
import os
import traceback

MAX_BODIES_PER_OCC = 40


def _bbox(ent):
    try:
        bb = ent.boundingBox
        if bb is None:
            return None
        return {
            "min_cm": [bb.minPoint.x, bb.minPoint.y, bb.minPoint.z],
            "max_cm": [bb.maxPoint.x, bb.maxPoint.y, bb.maxPoint.z],
        }
    except Exception:
        return None


def _vec(v):
    try:
        return [v.x, v.y, v.z]
    except Exception:
        return None


def _source_doc(comp):
    try:
        pd = comp.parentDesign
        if pd and pd.parentDocument:
            return pd.parentDocument.name
    except Exception:
        pass
    return None


def _motion(jm):
    d = {"motion_type": int(jm.jointType)}
    for attr, key in (
        ("slideDirectionVector", "slide_axis"),
        ("rotationAxisVector", "rotation_axis"),
        ("primarySlideDirectionVector", "slide_axis_primary"),
        ("secondarySlideDirectionVector", "slide_axis_secondary"),
    ):
        v = getattr(jm, attr, None)
        if v is not None:
            d[key] = _vec(v)
    for attr, key in (
        ("slideLimits", "slide_limits_cm"),
        ("rotationLimits", "rotation_limits_rad"),
        ("primarySlideLimits", "slide_limits_primary_cm"),
        ("secondarySlideLimits", "slide_limits_secondary_cm"),
    ):
        lim = getattr(jm, attr, None)
        if lim is not None:
            try:
                d[key] = {
                    "min_enabled": bool(lim.isMinimumValueEnabled),
                    "min": lim.minimumValue,
                    "max_enabled": bool(lim.isMaximumValueEnabled),
                    "max": lim.maximumValue,
                    "rest_enabled": bool(lim.isRestValueEnabled),
                    "rest": lim.restValue,
                }
            except Exception:
                pass
    for attr, key in (("slideValue", "slide_value_cm"), ("rotationValue", "rotation_value_rad")):
        v = getattr(jm, attr, None)
        if v is not None:
            d[key] = v
    return d


def _joint_entry(j):
    try:
        e = {
            "name": j.name,
            "occ1": j.occurrenceOne.fullPathName if j.occurrenceOne else None,
            "occ2": j.occurrenceTwo.fullPathName if j.occurrenceTwo else None,
            "is_suppressed": bool(getattr(j, "isSuppressed", False)),
        }
        e.update(_motion(j.jointMotion))
        try:
            o = j.geometryOrOriginOne
            if o is not None and getattr(o, "origin", None) is not None:
                e["origin1_cm"] = _vec(o.origin)
        except Exception:
            pass
        return e
    except Exception:
        return {"name": getattr(j, "name", "?"), "error": traceback.format_exc()}


def run(context):
    app = adsk.core.Application.get()
    ui = app.userInterface
    try:
        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            ui.messageBox("No active design.")
            return
        root = design.rootComponent
        out = {
            "document": app.activeDocument.name,
            "root_component": root.name,
            "default_length_units": design.unitsManager.defaultLengthUnits,
            "design_type": int(design.designType),
            "user_parameters": [],
            "occurrences": [],
            "joints": [],
            "as_built_joints": [],
            "rigid_groups": [],
        }

        try:
            for p in design.userParameters:
                out["user_parameters"].append({
                    "name": p.name,
                    "expression": p.expression,
                    "value_internal": p.value,
                    "unit": p.unit,
                    "comment": p.comment,
                })
        except Exception:
            out["user_parameters_error"] = traceback.format_exc()

        for occ in root.allOccurrences:
            comp = occ.component
            entry = {
                "path": occ.fullPathName,
                "component": comp.name,
                "source_document": _source_doc(comp),
                "is_referenced": bool(occ.isReferencedComponent),
                "light_bulb_on": bool(occ.isLightBulbOn),
                "is_visible": bool(occ.isVisible),
                "is_grounded": bool(occ.isGrounded),
                "bbox": _bbox(occ),
            }
            try:
                entry["transform_rowmajor"] = list(occ.transform2.asArray())
            except Exception:
                pass
            bodies = []
            try:
                occ_bodies = occ.bRepBodies
                entry["n_brep_bodies"] = occ_bodies.count
                for i in range(min(occ_bodies.count, MAX_BODIES_PER_OCC)):
                    b = occ_bodies.item(i)
                    bodies.append({
                        "name": b.name,
                        "light_bulb_on": bool(b.isLightBulbOn),
                        "is_visible": bool(b.isVisible),
                        "bbox": _bbox(b),
                    })
            except Exception:
                entry["n_brep_bodies"] = None
            try:
                entry["n_mesh_bodies"] = comp.meshBodies.count
            except Exception:
                entry["n_mesh_bodies"] = None
            entry["bodies"] = bodies
            out["occurrences"].append(entry)

        try:
            for j in root.allJoints:
                out["joints"].append(_joint_entry(j))
        except Exception:
            out["joints_error"] = traceback.format_exc()
        try:
            for j in root.allAsBuiltJoints:
                out["as_built_joints"].append(_joint_entry(j))
        except Exception:
            out["as_built_joints_error"] = traceback.format_exc()
        try:
            for rg in root.allRigidGroups:
                members = []
                try:
                    for i in range(rg.occurrences.count):
                        members.append(rg.occurrences.item(i).fullPathName)
                except Exception:
                    pass
                out["rigid_groups"].append({"name": rg.name, "members": members})
        except Exception:
            out["rigid_groups_error"] = traceback.format_exc()

        script_dir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(script_dir, "tree_dump.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        ui.messageBox(
            f"Dumped {len(out['occurrences'])} occurrences, "
            f"{len(out['joints'])} joints, {len(out['as_built_joints'])} as-built joints, "
            f"{len(out['rigid_groups'])} rigid groups, {len(out['user_parameters'])} user params to\n{path}"
        )
    except Exception:
        if ui:
            ui.messageBox("Failed:\n{}".format(traceback.format_exc()))
