"""Run an exported USDZ in a physics engine and report whether it behaves.

OpenUSD's validators say a stage is well formed. They do not say a lid rests on a
box when you press play. This loads what we export -- rigid bodies, mass, colliders,
contact materials, joints -- into MuJoCo, drops it on a floor, and reports what
happened.

MuJoCo has no USD importer, so the stage is translated to MJCF here. That
translation is the part most likely to be wrong, so it is deliberately literal:
every value comes from the schema we authored, and units are converted in exactly
one place.

MuJoCo is an optional dependency. Install it with ``uv sync --group sim``.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade  # pyright: ignore[reportAttributeAccessIssue]

# USD joint prim type -> MJCF joint type. A fixed joint welds the bodies, which
# MuJoCo expresses by nesting them with no joint element at all.
_JOINT_TYPES: dict[str, str | None] = {
    "PhysicsRevoluteJoint": "hinge",
    "PhysicsPrismaticJoint": "slide",
    "PhysicsFixedJoint": None,
}
_AXES = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}

# The usd-core stubs omit these schemas; bind them once rather than suppressing
# at every call site.
_MassAPI = UsdPhysics.MassAPI  # pyright: ignore[reportAttributeAccessIssue]
_CollisionAPI = UsdPhysics.CollisionAPI  # pyright: ignore[reportAttributeAccessIssue]
_MaterialAPI = UsdPhysics.MaterialAPI  # pyright: ignore[reportAttributeAccessIssue]
_XformCache = UsdGeom.XformCache  # pyright: ignore[reportAttributeAccessIssue]
_Mesh = UsdGeom.Mesh  # pyright: ignore[reportAttributeAccessIssue]

DROP_HEIGHT = 0.02
"""Metres above the floor to release the object, so contact is exercised."""


class SimulationUnavailable(RuntimeError):
    """MuJoCo is not installed."""


@dataclass(frozen=True)
class SimulationResult:
    """What happened when the exported object was dropped on a floor."""

    bodies: tuple[str, ...]
    total_mass: float
    start_heights: tuple[float, float]
    end_heights: tuple[float, float]
    contacts: int
    deepest_penetration: float
    largest_separation_change: float
    residual_velocity: float
    diverged: bool = False

    @property
    def fell_through_floor(self) -> bool:
        return self.end_heights[0] < -0.05

    @property
    def parts_stayed_together(self) -> bool:
        return self.largest_separation_change < 0.005

    @property
    def stood_up(self) -> bool:
        return (
            not self.diverged
            and not self.fell_through_floor
            and self.parts_stayed_together
            and self.deepest_penetration > -0.01
        )

    def summary(self) -> str:
        lines = [
            f"{len(self.bodies)} bodies, {self.total_mass:.3f} kg total",
            f"  heights: {self.start_heights[0]:+.4f} -> {self.end_heights[0]:+.4f} m (lowest body)",
            f"  contacts at rest: {self.contacts}",
            f"  deepest penetration: {self.deepest_penetration * 1000:+.2f} mm",
            f"  largest part separation change: {self.largest_separation_change * 1000:+.2f} mm",
            f"  residual velocity: {self.residual_velocity:.4f}",
        ]
        if self.diverged:
            lines.append("  DIVERGED: the solver produced non-finite state")
        lines.append(f"  verdict: {'stands up' if self.stood_up else 'FAILED'}")
        return "\n".join(lines)


@dataclass
class _Joint:
    name: str
    kind: str | None
    parent: str
    child: str
    anchor: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower: float | None
    upper: float | None


@dataclass
class _Scene:
    """The parts of an exported stage that a simulator needs."""

    parts: dict[str, Usd.Prim] = field(default_factory=dict)
    joints: list[_Joint] = field(default_factory=list)

    def root(self) -> str:
        children = {joint.child for joint in self.joints}
        roots = [name for name in self.parts if name not in children]
        if len(roots) != 1:
            raise ValueError(f"expected exactly one root part, found {roots}")
        return roots[0]


def simulate_usdz(usdz: Path, work_dir: Path, *, seconds: float = 3.0) -> SimulationResult:
    """Drop an exported USDZ on a floor and report what happened."""

    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise SimulationUnavailable(
            "MuJoCo is not installed; run `uv sync --group sim` to enable simulation"
        ) from exc

    model_path = write_mjcf(usdz, work_dir)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index) for index in range(1, model.nbody)
    )
    start = data.xpos[1:].copy()
    separations = {
        (a, b): float(np.linalg.norm(data.xpos[a] - data.xpos[b]))
        for a in range(1, model.nbody)
        for b in range(a + 1, model.nbody)
    }

    deepest = 0.0
    diverged = False
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if data.ncon:
            deepest = min(deepest, float(data.contact.dist[: data.ncon].min()))
        if not np.all(np.isfinite(data.qpos)):
            diverged = True
            break

    end = data.xpos[1:].copy()
    drift = max(
        (
            abs(float(np.linalg.norm(data.xpos[a] - data.xpos[b])) - was)
            for (a, b), was in separations.items()
        ),
        default=0.0,
    )
    return SimulationResult(
        bodies=names,
        total_mass=float(sum(model.body_mass)),
        start_heights=(float(start[:, 2].min()), float(start[:, 2].max())),
        end_heights=(float(end[:, 2].min()), float(end[:, 2].max())),
        contacts=int(data.ncon),
        deepest_penetration=deepest,
        largest_separation_change=drift,
        residual_velocity=float(np.abs(data.qvel).max()) if model.nv else 0.0,
        diverged=diverged,
    )


def write_mjcf(usdz: Path, out_dir: Path) -> Path:
    """Translate an exported stage into an MJCF model beside its meshes."""

    out_dir.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(str(usdz))
    if stage is None:
        raise ValueError(f"could not open {usdz}")
    scene = _read_scene(stage)
    root = scene.root()

    lowest = _lowest_point(scene)
    lift = np.eye(4)
    lift[2, 3] = -lowest + DROP_HEIGHT

    mujoco_el = ET.Element("mujoco", model=usdz.stem or "object")
    ET.SubElement(mujoco_el, "compiler", angle="radian", meshdir=".")
    ET.SubElement(mujoco_el, "option", timestep="0.002", integrator="implicitfast")
    asset = ET.SubElement(mujoco_el, "asset")
    world = ET.SubElement(mujoco_el, "worldbody")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="5 5 0.1",
        pos="0 0 0",
        friction="1 0.005 0.0001",
    )

    _add_body(world, asset, scene, root, np.linalg.inv(lift), stage, out_dir)

    path = out_dir / "model.xml"
    ET.indent(mujoco_el)
    path.write_text(ET.tostring(mujoco_el, encoding="unicode"), encoding="utf-8")
    return path


def _read_scene(stage: Usd.Stage) -> _Scene:
    world = stage.GetDefaultPrim()
    objects = [prim for prim in world.GetChildren() if prim.GetChild("parts")]
    if len(objects) != 1:
        raise ValueError("expected one articulated object on the stage")
    obj = objects[0]

    scene = _Scene(parts={prim.GetName(): prim for prim in obj.GetChild("parts").GetChildren()})
    joints_scope = obj.GetChild("joints")
    for prim in joints_scope.GetChildren() if joints_scope else []:
        kind = _JOINT_TYPES.get(str(prim.GetTypeName()))
        if str(prim.GetTypeName()) not in _JOINT_TYPES:
            continue
        bodies = [prim.GetRelationship(f"physics:body{index}").GetTargets() for index in (0, 1)]
        if not all(bodies):
            continue
        scene.joints.append(
            _Joint(
                name=prim.GetName(),
                kind=kind,
                parent=bodies[0][0].name,
                child=bodies[1][0].name,
                anchor=_triple(_attr(prim, "physics:localPos1", (0.0, 0.0, 0.0))),
                axis=_AXES.get(str(_attr(prim, "physics:axis", "Z")), (0.0, 0.0, 1.0)),
                lower=_number(_attr(prim, "physics:lowerLimit")),
                upper=_number(_attr(prim, "physics:upperLimit")),
            )
        )
    return scene


def _add_body(
    parent_el: ET.Element,
    asset: ET.Element,
    scene: _Scene,
    part_name: str,
    parent_world: np.ndarray,
    stage: Usd.Stage,
    out_dir: Path,
) -> None:
    prim = scene.parts[part_name]
    world = _world_transform(prim)
    relative = np.linalg.inv(parent_world) @ world
    body = ET.SubElement(parent_el, "body", name=part_name, pos=_vector(relative[:3, 3]))

    joint = next((item for item in scene.joints if item.child == part_name), None)
    if joint is None:
        ET.SubElement(body, "freejoint")
    elif joint.kind is not None:
        attributes = {
            "name": joint.name,
            "type": joint.kind,
            "pos": _vector(joint.anchor),
            "axis": _vector(joint.axis),
        }
        if joint.lower is not None and joint.upper is not None:
            # UsdPhysics states revolute limits in degrees and prismatic limits in
            # stage units. MJCF is written here in radians and metres.
            scale = math.pi / 180.0 if joint.kind == "hinge" else 1.0
            attributes["range"] = f"{joint.lower * scale:.6f} {joint.upper * scale:.6f}"
            attributes["limited"] = "true"
        ET.SubElement(body, "joint", attributes)

    mass_api = _MassAPI(prim)
    mass = _number(mass_api.GetMassAttr().Get())
    if mass:
        ET.SubElement(
            body,
            "inertial",
            pos=_vector(mass_api.GetCenterOfMassAttr().Get() or (0.0, 0.0, 0.0)),
            mass=f"{mass:.6f}",
            diaginertia=" ".join(
                f"{max(float(value), 1e-9):.9f}"
                for value in (mass_api.GetDiagonalInertiaAttr().Get() or (1e-4, 1e-4, 1e-4))
            ),
        )

    shapes = prim.GetChild("shapes")
    for shape in shapes.GetChildren() if shapes else []:
        if not shape.HasAPI(_CollisionAPI):
            continue
        mesh_name = f"{part_name}_{shape.GetName()}"
        _write_obj(out_dir / f"{mesh_name}.obj", *_mesh_arrays(shape))
        ET.SubElement(asset, "mesh", name=mesh_name, file=f"{mesh_name}.obj")
        geom: dict[str, str] = {"type": "mesh", "mesh": mesh_name, "name": mesh_name}
        friction = _contact_friction(shape, stage)
        if friction is not None:
            geom["friction"] = f"{friction:.4f} 0.005 0.0001"
        ET.SubElement(body, "geom", geom)

    for child in (item.child for item in scene.joints if item.parent == part_name):
        _add_body(body, asset, scene, child, world, stage, out_dir)


def _contact_friction(shape: Usd.Prim, stage: Usd.Stage) -> float | None:
    """The dynamic friction bound to this collider, if any."""

    targets = UsdShade.MaterialBindingAPI(shape).GetDirectBindingRel("physics").GetTargets()
    if not targets:
        return None
    material = _MaterialAPI(stage.GetPrimAtPath(targets[0]))
    return _number(material.GetDynamicFrictionAttr().Get())


def _lowest_point(scene: _Scene) -> float:
    lowest = math.inf
    for prim in scene.parts.values():
        transform = _world_transform(prim)
        shapes = prim.GetChild("shapes")
        for shape in shapes.GetChildren() if shapes else []:
            points, _ = _mesh_arrays(shape)
            homogeneous = np.hstack([points, np.ones((len(points), 1))])
            lowest = min(lowest, float((transform @ homogeneous.T).T[:, 2].min()))
    if not math.isfinite(lowest):
        raise ValueError("stage has no collidable geometry to place on the floor")
    return lowest


def _world_transform(prim: Usd.Prim) -> np.ndarray:
    matrix = _XformCache().GetLocalToWorldTransform(prim)
    return np.array(matrix, dtype=float).T  # USD stores row-vector matrices


def _mesh_arrays(prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray]:
    mesh = _Mesh(prim)
    points = np.array(mesh.GetPointsAttr().Get(), dtype=float)
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
    if not np.all(counts == 3):
        raise ValueError(f"{prim.GetPath()} is not triangulated")
    return points, indices.reshape(-1, 3)


def _write_obj(path: Path, points: np.ndarray, faces: np.ndarray) -> None:
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in points]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces]
    path.write_text("\n".join(lines), encoding="utf-8")


def _attr(prim: Usd.Prim, name: str, default: Any = None) -> Any:
    attribute = prim.GetAttribute(name)
    value = attribute.Get() if attribute else None
    return default if value is None else value


def _number(value: Any) -> float | None:
    return None if value is None else float(value)


def _triple(value: Any) -> tuple[float, float, float]:
    x, y, z = (float(component) for component in value)
    return (x, y, z)


def _vector(value: Any) -> str:
    return " ".join(f"{float(component):.6f}" for component in value)
