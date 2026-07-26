"""Geometry-aware UV projection and a keyword material catalog for the texture
layer.

Two responsibilities, both pure/offline:

* **Projection** -- wrap a *tiling* PBR material onto a shape by the projection
  that suits its form: cylindrical for surfaces of revolution (barrels, tubes,
  posts), planar for flat panels, box as a fallback. Atlas unwrapping is
  deliberately avoided: it randomises chart orientation, which looks wrong for a
  tiling material. Each projection unwelds to independent triangles and returns
  ``(points, uvs)``; the caller pairs them with smooth vertex normals so shading
  stays continuous.

* **Catalog** -- map a shape's name to an ambientCG material category (or
  ``None`` to leave it parametric, e.g. glass and glowing bulbs, which no tiling
  texture models well).
"""

from __future__ import annotations

import numpy as np

# category -> ambientCG asset id (all CC0, verified downloadable @ 1K).
CATEGORY_ASSETS = {
    "aluminum": "Metal050A",   # clean smooth aluminum -- shells, tubes, panels
    "steel": "Metal009",       # brushed steel -- blades, rotors, springs, chuck
    "dark_metal": "Metal046A", # dark metal -- exhaust plugs, cores
    "plastic": "Plastic010",   # molded plastic -- housings, covers, knobs
    "rubber": "Rubber004",     # rubber -- grips, feet, seals
}

# Which categories are conductors (metalness 1) vs dielectrics (metalness 0).
# Getting this wrong renders plastic/rubber as shiny metal.
METAL_CATEGORIES = frozenset({"aluminum", "steel", "dark_metal"})

# First match wins. Rubber and glass/bulb are checked before the metals so a
# rubber *grip* or a glass *lens* is not captured by a generic metal word.
_RUBBER = ("rubber", "grip", "foot", "pad", "seal", "eyecup", "gasket",
           "bumper", "tire", "boot")
_GLASS = ("glass", "window", "screen", "lcd", "bulb", "filament", "acrylic")
# "lens" is glass only as the optical element, not lens hardware.
_LENS_HARDWARE = ("mount", "ring", "button", "release", "cap", "dot", "mark",
                  "line", "thread", "barrel", "collar", "flange", "bayonet")
_PARAMETRIC = ("ceramic", "porcelain", "pointer", "index")
_STEEL = ("blade", "rotor", "vane", "stator", "spring", "chuck", "jaw", "hinge",
          "pin", "shaft", "screw", "bolt", "spindle", "bearing", "gear", "knurl",
          "drive", "steel")
_DARK = ("exhaust", "plug", "core", "dark")
_PLASTIC = ("plastic", "case", "housing", "cover", "knob", "button", "nose",
            "guard", "slider", "key", "cap", "hump", "shade", "motor", "battery",
            "vent", "panel")
_ALUMINUM = ("nacelle", "cowl", "shell", "body", "tube", "fairing", "band",
             "spigot", "post", "turntable", "barrel", "dial", "rim", "mount",
             "flange", "lip", "hub", "frame", "arm", "bracket", "collar", "rail",
             "neck", "socket", "weighted", "metal", "alumin", "spinner", "plate",
             "ring", "boss", "clamp", "pivot", "elbow", "shoulder", "lug")


def classify(name: str) -> str | None:
    """Return a ``CATEGORY_ASSETS`` key, or ``None`` to keep the shape parametric."""

    text = name.lower()
    if _hit(text, _RUBBER):
        return "rubber"
    if _hit(text, _GLASS) or ("lens" in text and not _hit(text, _LENS_HARDWARE)):
        return None
    if _hit(text, _PARAMETRIC):
        return None
    if _hit(text, _STEEL):
        return "steel"
    if _hit(text, _DARK):
        return "dark_metal"
    if _hit(text, _PLASTIC):
        return "plastic"
    if _hit(text, _ALUMINUM):
        return "aluminum"
    return "steel"  # unknown machined part -- brushed steel reads neutrally


def pick_projection(vertices: np.ndarray) -> str:
    """Choose 'cylinder' / 'planar' / 'box' from a shape's principal extents."""

    centered = np.asarray(vertices, dtype=float)
    centered = centered - centered.mean(axis=0)
    # spread along each principal axis (sqrt eigenvalues of the covariance)
    eigvals = np.linalg.eigvalsh(centered.T @ centered)
    e3, e2, e1 = np.sqrt(np.maximum(eigvals, 0.0))  # ascending -> e1 largest
    if e2 < 1e-9:
        return "box"
    if e1 >= 1.5 * e2 and e2 <= 1.6 * e3:
        return "cylinder"   # elongated with a round-ish cross-section
    if e3 <= 0.35 * e2:
        return "planar"     # flat slab
    return "box"


def project(mode: str, vertices: np.ndarray, faces: np.ndarray,
            face_normals: np.ndarray, tile: float) -> tuple[np.ndarray, np.ndarray]:
    if mode == "cylinder":
        return _cylindrical(vertices, faces, tile)
    if mode == "planar":
        return _planar(vertices, faces, tile)
    return _box(vertices, faces, face_normals, tile)


def unwelded_normals(vertex_normals: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Smooth per-vertex normals carried onto the unwelded triangles."""

    return np.asarray(vertex_normals)[np.asarray(faces)].reshape(-1, 3).astype(np.float32)


def adaptive_tile(
    trimesh_obj,
    base_tile: float,
    *,
    target_repeats: float = 2.5,
    min_tile: float = 0.012,
) -> float:
    """Scale the texture tile to the geometry so every part reads coherently.

    A tile fixed in meters looks right on a big surface but turns tiny parts into
    a random speck of texture. Here the tile is sized from the *feature* rather
    than the whole shape: the mesh is split into connected components (so a mesh
    of 40 little vanes measures one vane, not the 2 m ring), and the tile targets
    a few repeats across a component's characteristic width. Big smooth surfaces
    keep the real-world ``base_tile``; small features shrink to stay legible.
    """

    try:
        components = trimesh_obj.split(only_watertight=False)
    except Exception:
        components = []
    meshes = list(components) if len(components) else [trimesh_obj]

    widths = []
    for mesh in meshes:
        verts = np.asarray(mesh.vertices)
        if verts.size == 0:
            continue
        extents = np.sort(verts.max(axis=0) - verts.min(axis=0))  # ascending
        widths.append(extents[1])  # middle extent ~ the surface's characteristic width
    if not widths:
        return base_tile
    characteristic = float(np.median(widths))
    return float(np.clip(characteristic / target_repeats, min_tile, base_tile))


def textured_mesh(trimesh_obj, base_tile: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return unwelded (points, uvs, smooth_normals, projection_mode) for a mesh.

    One call produces everything the textured USD export needs: a geometry-aware
    projection's UVs (at a tile scaled to the geometry) plus matching smooth
    normals on independent triangles.
    """

    verts = np.asarray(trimesh_obj.vertices)
    faces = np.asarray(trimesh_obj.faces)
    tile = adaptive_tile(trimesh_obj, base_tile)
    mode = pick_projection(verts)
    points, uvs = project(mode, verts, faces, np.asarray(trimesh_obj.face_normals), tile)
    normals = unwelded_normals(trimesh_obj.vertex_normals, faces)
    return points, uvs, normals, mode


def _box(vertices, faces, face_normals, tile):
    plane = np.array([[1, 2], [0, 2], [0, 1]])
    dominant = np.argmax(np.abs(face_normals), axis=1)
    points = vertices[faces].reshape(-1, 3)
    axes = np.repeat(plane[dominant], 3, axis=0)
    u = np.take_along_axis(points, axes[:, :1], axis=1)[:, 0]
    v = np.take_along_axis(points, axes[:, 1:], axis=1)[:, 0]
    return points.astype(np.float32), (np.stack([u, v], axis=1) / tile).astype(np.float32)


def _planar(vertices, faces, tile):
    verts = np.asarray(vertices, dtype=float)
    _, eigvecs = np.linalg.eigh((verts - verts.mean(0)).T @ (verts - verts.mean(0)))
    u_hat, v_hat = eigvecs[:, -1], eigvecs[:, -2]  # two largest-spread axes
    points = verts[faces].reshape(-1, 3)
    uvs = np.stack([points @ u_hat, points @ v_hat], axis=1) / tile
    return points.astype(np.float32), uvs.astype(np.float32)


def _cylindrical(vertices, faces, tile):
    verts = np.asarray(vertices, dtype=float)
    center = verts.mean(axis=0)
    _, eigvecs = np.linalg.eigh((verts - center).T @ (verts - center))
    axis = eigvecs[:, -1] / np.linalg.norm(eigvecs[:, -1])
    ref = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    u_hat = np.cross(axis, ref)
    u_hat /= np.linalg.norm(u_hat)
    v_hat = np.cross(axis, u_hat)

    points = verts[faces].reshape(-1, 3)
    rel = points - center
    along = rel @ axis
    theta = np.arctan2(rel @ v_hat, rel @ u_hat)
    radius_mean = float(np.hypot(rel @ u_hat, rel @ v_hat).mean())
    # unwrap each triangle onto one branch so none straddles the +/-pi seam
    theta = theta.reshape(-1, 3)
    base = theta[:, :1]
    theta = (base + (np.mod(theta - base + np.pi, 2 * np.pi) - np.pi)).reshape(-1)
    uvs = np.stack([theta * radius_mean, along], axis=1) / tile
    return points.astype(np.float32), uvs.astype(np.float32)


def _hit(text: str, keywords: tuple[str, ...]) -> bool:
    return any(word in text for word in keywords)
