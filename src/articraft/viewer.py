from __future__ import annotations

import contextlib
import json
import re
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from pxr import Usd

from articraft import package_dir


@dataclass(frozen=True)
class ViewerRun:
    versions: tuple[dict[str, object], ...]
    files: dict[str, Path]

    def bootstrap(self) -> dict[str, object]:
        return {"versions": self.versions}


def load_viewer_run(run_dir: Path | str) -> ViewerRun:
    run_dir = Path(run_dir).resolve()
    usdz_dir = run_dir / "result" / "usdz"
    paths = sorted(
        (path for path in usdz_dir.glob("*.usdz") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
        reverse=True,
    )
    if not paths:
        raise ValueError(f"no numbered USDZ files found in {usdz_dir}")

    versions = tuple(_read_version(path) | _read_trajectory(run_dir, path) for path in paths)
    files = {str(version["id"]): path for version, path in zip(versions, paths, strict=True)}
    return ViewerRun(versions=versions, files=files)


def _read_trajectory(run_dir: Path, usdz: Path) -> dict[str, object]:
    """The recorded simulation for this USDZ, if `articraft simulate` has run."""

    record = run_dir / "result" / "simulation" / f"{usdz.stem}.trajectory.json"
    if not record.is_file():
        return {}
    try:
        return {"trajectory": json.loads(record.read_text(encoding="utf-8"))}
    except (OSError, json.JSONDecodeError):
        return {}


def serve_viewer(run_dir: Path | str, *, open_browser: bool = True) -> None:
    viewer_run = load_viewer_run(run_dir)
    page = (package_dir / "viewer.html").read_bytes()
    bootstrap = json.dumps(viewer_run.bootstrap(), separators=(",", ":")).encode()
    handler = _handler(page, bootstrap, viewer_run.files)

    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Viewer URL: {url}")
        if open_browser:
            webbrowser.open(url)
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()


def _read_version(path: Path) -> dict[str, object]:
    try:
        stage = Usd.Stage.Open(str(path.resolve()))
    except Exception as exc:
        raise ValueError(f"could not open USDZ file: {path}") from exc
    if stage is None:
        raise ValueError(f"could not open USDZ file: {path}")

    world = stage.GetDefaultPrim()
    object_prims = [
        prim
        for prim in world.GetChildren()
        if prim.GetChild("rigid_bodies") or prim.GetChild("parts")
    ]
    if len(object_prims) != 1:
        raise ValueError(f"expected one articulated object in {path}")
    object_prim = object_prims[0]

    parts = [
        {
            "name": _attribute(part, "name", part.GetName()),
            "usd_name": part.GetName(),
            "shapes": _read_shapes(part),
            "mass": _read_mass(part),
        }
        for part in (
            object_prim.GetChild("rigid_bodies") or object_prim.GetChild("parts")
        ).GetChildren()
    ]

    articulations = [_read_joint(joint) for joint in object_prim.GetChild("joints").GetChildren()]
    _orient_joints(articulations)

    return {
        "id": path.stem,
        "filename": path.name,
        "model": {
            "name": _attribute(object_prim, "name", object_prim.GetName()),
            "parts": parts,
            "articulations": articulations,
        },
    }


_AXIS_VECTORS = {
    "transX": [1.0, 0.0, 0.0],
    "transY": [0.0, 1.0, 0.0],
    "transZ": [0.0, 0.0, 1.0],
    "rotX": [1.0, 0.0, 0.0],
    "rotY": [0.0, 1.0, 0.0],
    "rotZ": [0.0, 0.0, 1.0],
}


def _read_joint(joint: Usd.Prim) -> dict[str, object]:
    joint_type = _attribute(joint, "jointType", "fixed")
    if not joint.GetAttribute("articraft:jointType"):
        return _read_legacy_joint(joint)
    dofs = json.loads(str(_attribute(joint, "dofs", "[]")))
    # Only a single axis moves in the viewer; a d6 joint previews as fixed.
    dof = dofs[0] if joint_type in ("revolute", "prismatic") and dofs else None
    limits = dof["limits"] if dof else None
    return {
        "name": _attribute(joint, "name", joint.GetName()),
        "type": joint_type if dof else "fixed",
        "parent": _attribute(joint, "body0"),
        "child": _attribute(joint, "body1"),
        "origin": {
            "xyz": _attribute(joint, "frame0:xyz", [0.0, 0.0, 0.0]),
            "rpy": _attribute(joint, "frame0:rpy", [0.0, 0.0, 0.0]),
        },
        "child_origin": {
            "xyz": _attribute(joint, "frame1:xyz", [0.0, 0.0, 0.0]),
            "rpy": _attribute(joint, "frame1:rpy", [0.0, 0.0, 0.0]),
        },
        "axis": _AXIS_VECTORS[dof["axis"]] if dof else [0.0, 0.0, 1.0],
        "motion_limits": None if limits is None else {"lower": limits[0], "upper": limits[1]},
        # A loop closing joint is a constraint, not a place to hang the child;
        # the viewer must not reparent or pose along it.
        "closes_loop": bool(_usd_attribute(joint, "physics:excludeFromArticulation", False)),
        # A closer free to translate is a cylinder: the two ends stay in line and
        # the distance between them is the extension. That lets the viewer hold
        # the loop shut by aiming, rather than letting the parts drift apart.
        "slide_axis": _slide_axis(dofs),
        "driven": False,
    }


def _slide_axis(dofs: list[dict[str, object]]) -> list[float] | None:
    for dof in dofs:
        axis = str(dof["axis"])
        if axis.startswith("trans"):
            return _AXIS_VECTORS[axis]
    return None


def _read_legacy_joint(joint: Usd.Prim) -> dict[str, object]:
    """A joint written before the graph model.

    Runs already on disk cannot be regenerated, so the viewer keeps reading them.
    Their metadata names the joint kind and a single parent-side origin outright.
    """

    joint_type = _attribute(joint, "articulationType", "fixed")
    limits = None
    if joint_type != "fixed":
        limits = {
            "lower": _attribute(joint, "limits:lower"),
            "upper": _attribute(joint, "limits:upper"),
        }
    return {
        "name": _attribute(joint, "name", joint.GetName()),
        "type": joint_type,
        "parent": _attribute(joint, "parent"),
        "child": _attribute(joint, "child"),
        "origin": {
            "xyz": _attribute(joint, "origin:xyz", [0.0, 0.0, 0.0]),
            "rpy": _attribute(joint, "origin:rpy", [0.0, 0.0, 0.0]),
        },
        "child_origin": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
        "axis": _attribute(joint, "axis", [0.0, 0.0, 1.0]),
        "motion_limits": limits,
        "closes_loop": bool(_usd_attribute(joint, "physics:excludeFromArticulation", False)),
        "slide_axis": None,  # the old model had no multi-axis closers
        "driven": _attribute(joint, "driven", "false") == "true",
    }


def _orient_joints(joints: list[dict[str, object]]) -> None:
    """Point every tree joint away from the root.

    ``body0``/``body1`` are symmetric in the assembly, but the viewer hangs the
    child off the parent, so a joint authored the other way round would build a
    cycle. Walk the tree and swap the ends that point backwards.
    """
    pending = [joint for joint in joints if not joint["closes_loop"]]
    seen: set[object] = set()
    while pending:
        reachable = [j for j in pending if j["parent"] in seen or j["child"] in seen]
        if not reachable:  # first joint, or the start of a separate component
            reachable = [pending[0]]
            seen.add(pending[0]["parent"])
        for joint in reachable:
            if joint["parent"] not in seen:
                joint["parent"], joint["child"] = joint["child"], joint["parent"]
                joint["origin"], joint["child_origin"] = joint["child_origin"], joint["origin"]
                joint["axis"] = [-value for value in cast(list[float], joint["axis"])]
            seen.update((joint["parent"], joint["child"]))
        pending = [joint for joint in pending if joint not in reachable]


def _read_shapes(part: Usd.Prim) -> list[dict[str, object]]:
    shapes_scope = part.GetChild("shapes")
    if not shapes_scope:
        return []
    return [
        {
            "usd_name": shape.GetName(),
            "appearance": _read_appearance(shape),
            "material": _attribute(shape, "material"),
            "coating": _attribute(shape, "coating"),
        }
        for shape in shapes_scope.GetChildren()
    ]


def _read_mass(part: Usd.Prim) -> dict[str, object] | None:
    """The part's exported mass, for the viewer's parts panel.

    Read from the stage rather than the manifest, because that is the only thing
    the viewer is given.
    """

    # UsdPhysics attributes are standard, not articraft-namespaced.
    mass_attr = part.GetAttribute("physics:mass")
    mass = mass_attr.Get() if mass_attr else None
    if mass is None:
        return None
    center_attr = part.GetAttribute("physics:centerOfMass")
    center = center_attr.Get() if center_attr else None
    shapes = part.GetChild("shapes")
    names = {
        str(value)
        for shape in (shapes.GetChildren() if shapes else [])
        if (value := _attribute(shape, "material")) is not None
    }
    materials = sorted(names)
    return {
        "kilograms": float(mass),
        "materials": materials,
        "center_of_mass": [float(value) for value in (center or (0.0, 0.0, 0.0))],
    }


def _read_appearance(shape: Usd.Prim) -> dict[str, object] | None:
    metallic = _attribute(shape, "material:metallic")
    if metallic is None:
        return None
    return {
        "base_color": _attribute(shape, "material:baseColor", [0.8, 0.8, 0.8]),
        "metallic": metallic,
        "roughness": _attribute(shape, "material:roughness", 0.6),
        "opacity": _attribute(shape, "material:opacity", 1.0),
        "emissive": _attribute(shape, "material:emissive"),
        # When set, USDLoader has already applied ambientCG texture maps; the
        # viewer keeps those and only layers on the authored tint + metalness.
        "textured": _attribute(shape, "material:textured") is not None,
    }


def _usd_attribute(prim: Usd.Prim, name: str, default=None):
    """Read a schema attribute by its full USD name, no articraft prefix."""
    attribute = prim.GetAttribute(name)
    value = attribute.Get() if attribute else None
    return default if value is None else value


def _attribute(prim: Usd.Prim, name: str, default=None):
    attribute = prim.GetAttribute(f"articraft:{name}")
    value = attribute.Get() if attribute else None
    if value is None:
        return default
    if hasattr(value, "__len__") and hasattr(value, "__getitem__") and not isinstance(value, str):
        return [float(value[index]) for index in range(len(value))]
    return value


def _handler(
    page: bytes,
    bootstrap: bytes,
    files: dict[str, Path],
) -> type[BaseHTTPRequestHandler]:
    class ViewerHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            route = urlsplit(self.path).path
            if route in {"/", "/index.html"}:
                self._send(200, "text/html; charset=utf-8", page)
                return
            if route == "/api/bootstrap":
                self._send(200, "application/json", bootstrap)
                return
            match = re.fullmatch(r"/models/(\d+)\.usdz", route)
            path = files.get(match.group(1)) if match else None
            if path is not None:
                self._send(200, "model/vnd.usdz+zip", path.read_bytes())
                return
            if route == "/favicon.ico":
                self._send(204, "image/x-icon", b"")
                return
            self._send(404, "text/plain; charset=utf-8", b"Not found\n")

        def log_message(self, format: str, *args: object) -> None:
            del format, args
            return

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ViewerHandler
