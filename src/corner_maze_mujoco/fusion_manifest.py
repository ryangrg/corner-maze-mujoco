"""Load the ExportMeshes manifest into typed body instances.

OBJ files from the Fusion export are in each component's *local* frame, in
meters. The manifest's per-occurrence transform (row-major 4x4, translation in
cm) maps local -> world, verified against the world bounding boxes.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import json
import numpy as np

CM = 0.01


@dataclass
class BodyInstance:
    id: int
    file: Path
    path: str
    component: str
    source_document: str | None
    body: str
    category: str
    body_visible: bool
    occ_visible: bool
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    T: np.ndarray
    appearance: str | None
    grid_center: list | None

    @property
    def key(self) -> tuple:
        return (self.source_document, self.component, self.body)

    @property
    def pos(self) -> np.ndarray:
        return self.T[:3, 3]

    @property
    def R(self) -> np.ndarray:
        return self.T[:3, :3]

    @property
    def top_level(self) -> str:
        return self.path.split("+")[0]

    @property
    def center(self) -> np.ndarray:
        return (self.bbox_min + self.bbox_max) / 2

    @property
    def size(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min


def sanitize(s: str, n: int = 48) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return (s[:n] if s else "x")


def asset_name(key: tuple) -> str:
    doc, comp, body = key
    h = hashlib.md5(f"{doc}|{comp}|{body}".encode()).hexdigest()[:6]
    return f"{sanitize(comp, 36)}_{sanitize(body, 10)}_{h}"


def load_manifest(manifest_path: str | Path) -> tuple[dict, list[BodyInstance]]:
    manifest_path = Path(manifest_path)
    m = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    bodies: list[BodyInstance] = []
    for b in m["bodies"]:
        if not b.get("exported"):
            continue
        f = root / b["file"]
        if not f.exists() or f.stat().st_size == 0:
            continue
        bw = b.get("bbox_world")
        tf = b.get("occ_transform_rowmajor_cm")
        if bw is None or tf is None:
            continue
        T = np.array(tf, dtype=float).reshape(4, 4)
        T[:3, 3] *= CM
        bodies.append(
            BodyInstance(
                id=b["id"],
                file=f,
                path=b["path"],
                component=b["component"],
                source_document=b.get("source_document"),
                body=b["body"],
                category=b.get("category", "other"),
                body_visible=bool(b.get("body_visible", True)),
                occ_visible=bool(b.get("occ_visible", True)),
                bbox_min=np.array(bw["min_cm"]) * CM,
                bbox_max=np.array(bw["max_cm"]) * CM,
                T=T,
                appearance=(b.get("appearance") or {}).get("name"),
                grid_center=b.get("grid_center"),
            )
        )
    return m, bodies


def read_mtl_kd(obj_path: Path) -> tuple[list[float] | None, float | None, str | None]:
    """Return (Kd rgb, alpha, material name) from the .mtl next to an OBJ, if any."""
    mtl = obj_path.with_suffix(".mtl")
    if not mtl.exists():
        return None, None, None
    kd = None
    alpha = None
    name = None
    for line in mtl.read_text(errors="ignore").splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "newmtl":
            name = " ".join(t[1:])
        elif t[0] == "Kd" and kd is None:
            kd = [float(v) for v in t[1:4]]
        elif t[0] in ("d",) and alpha is None:
            alpha = float(t[1])
        elif t[0] == "Tr" and alpha is None:
            alpha = 1.0 - float(t[1])
    return kd, alpha, name


def group_by_key(bodies: list[BodyInstance]) -> dict[tuple, list[BodyInstance]]:
    out: dict[tuple, list[BodyInstance]] = {}
    for b in bodies:
        out.setdefault(b.key, []).append(b)
    return out
