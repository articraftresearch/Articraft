"""Run an exported USDZ in a physics engine and report whether it behaves.

OpenUSD's validators say a stage is well formed. They do not say a lid rests on a
box when you press play. This loads what we export -- rigid bodies, mass, colliders,
contact materials, joints -- into MuJoCo, drops it on a floor, and reports what
happened.

MuJoCo has no USD importer, so the stage is translated to MJCF here. That
translation is the part most likely to be wrong, so it is deliberately literal:
every value comes from the schema we authored, and units are converted in exactly
one place.

What this does **not** cover. A material authors static friction, dynamic
friction, and restitution; only dynamic friction reaches the simulation. MuJoCo
carries a single sliding coefficient, so static friction has nowhere to go, and
it has no restitution parameter at all -- bounce comes from contact stiffness
(``solref``), which is not the same quantity. So a passing run says the geometry,
mass, joints, and sliding friction behave. It says nothing about how the object
would bounce, and slip onset is measured against the dynamic coefficient rather
than the static one that governs it.

MuJoCo is an optional dependency. Install it with ``uv sync --group sim``.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pxr import (
    Gf,
    Usd,
    UsdGeom,
    UsdPhysics,
    UsdShade,  # pyright: ignore[reportAttributeAccessIssue]
)
from scipy.spatial.transform import Rotation  # pyright: ignore[reportMissingTypeStubs]

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

TILT_RATE = 12.0
"""Degrees per second the floor tilts in the ``tilt`` scenario."""

SLIP_DISTANCE = 0.02
"""Lateral metres that counts as sliding rather than settling."""

MAX_TILT = 50.0
"""Degrees to stop tilting at. Past this an object topples rather than slides."""


class SimulationUnavailable(RuntimeError):
    """MuJoCo is not installed."""


class SimulationCapabilityError(ValueError):
    """The exported graph uses constraints this MuJoCo translator cannot preserve."""


@dataclass(frozen=True)
class Trajectory:
    """Authoritative world-space body poses recorded from a simulation."""

    fps: float
    frames: tuple[dict[str, Any], ...]

    def to_payload(self) -> dict[str, Any]:
        return {"fps": self.fps, "frames": list(self.frames)}


@dataclass(frozen=True)
class SimulationResult:
    """What happened when the exported object was dropped on a floor."""

    bodies: tuple[str, ...]
    total_mass: float
    start_height: float
    end_height: float
    contacts: int
    deepest_penetration: float
    """The worst instant, which for a drop is the landing impact."""
    resting_penetration: float
    """How far the object sinks once it has settled, which is what a verdict wants."""
    largest_separation_change: float
    residual_velocity: float
    diverged: bool = False
    trajectory: Trajectory | None = None
    scenario: str = "drop"
    slip_angle: float | None = None
    """Degrees of tilt at which the object started sliding, in the tilt scenario."""
    measured_friction: float | None = None
    """The coefficient implied by that slip angle: tan(slip)."""
    peak_joint_speed: float | None = None
    """Fastest joint motion seen, in rad/s or m/s. How hard a released joint slams."""
    expected_friction: float | None = None
    """The lowest friction authored on a shape that touches the floor."""

    @property
    def fell_through_floor(self) -> bool:
        """Whether the object passed through the floor instead of landing on it.

        Height alone cannot say: ``end_height`` is the lowest body *origin*, and
        a part's origin sits wherever its author put it -- a tripod leg's frame
        is up at its hinge while the leg hangs below. An object resting on the
        floor is in contact with it, so contact is what settles the question.
        """

        return self.end_height < -0.05 and self.contacts == 0

    @property
    def parts_stayed_together(self) -> bool:
        return self.largest_separation_change < 0.005

    @property
    def stood_up(self) -> bool:
        """Whether the object behaved.

        Only ``drop`` is judged on resting quietly. ``tilt`` and ``release`` are
        deliberately violent, so they are judged on staying whole.
        """

        if self.diverged or not self.parts_stayed_together:
            return False
        if self.scenario != "drop":
            return True
        return not self.fell_through_floor and self.resting_penetration > -0.01

    def summary(self) -> str:
        lines = [
            f"{len(self.bodies)} bodies, {self.total_mass:.3f} kg total",
            _headline(self.scenario, self.start_height, self.end_height),
            f"  contacts at rest: {self.contacts}",
            f"  deepest penetration: {self.deepest_penetration * 1000:+.2f} mm"
            f" on impact, {self.resting_penetration * 1000:+.2f} mm at rest",
            f"  largest part separation change: {self.largest_separation_change * 1000:+.2f} mm",
            f"  residual velocity: {self.residual_velocity:.4f}",
        ]
        if self.scenario == "release" and self.peak_joint_speed is not None:
            lines.append(f"  peak joint speed: {self.peak_joint_speed:.2f} per second")
        if self.scenario == "tilt":
            if self.slip_angle is None:
                lines.append("  never slipped: it held to the end of the tilt")
            else:
                lines.append(f"  slipped at: {self.slip_angle:.1f} deg of tilt")
                lines.append(
                    f"  friction: measured {self.measured_friction:.2f}"
                    + (
                        f", authored {self.expected_friction:.2f}"
                        if self.expected_friction is not None
                        else ""
                    )
                )
        if self.diverged:
            lines.append("  DIVERGED: the solver produced non-finite state")
        lines.append(f"  verdict: {'stands up' if self.stood_up else 'FAILED'}")
        return "\n".join(lines)


def _headline(scenario: str, start: float, end: float) -> str:
    if scenario == "drop":
        return f"  lowest body: {start:+.4f} -> {end:+.4f} m"
    if scenario == "tilt":
        return "  settled, then tilted until it moved"
    return "  joints released from mid-travel"


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
    excluded: bool = False
    """Marked ``physics:excludeFromArticulation``: a loop closing constraint."""
    parent_anchor: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis_in_parent: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """The same joint axis expressed in body0's frame, for flipped joints."""
    extra_axes: tuple[
        tuple[
            str, tuple[float, float, float], tuple[float, float, float], float | None, float | None
        ],
        ...,
    ] = ()
    """Further free axes on a D6 joint, as (kind, axis, axis_in_parent, lower, upper)."""


@dataclass
class _Scene:
    """The parts of an exported stage that a simulator needs."""

    parts: dict[str, Usd.Prim] = field(default_factory=dict)
    joints: list[_Joint] = field(default_factory=list)
    articulation_root: str | None = None
    world_joints: list[str] = field(default_factory=list)
    """Joints anchored to WORLD, which this translator cannot express yet."""

    @property
    def tree_joints(self) -> list[_Joint]:
        return [joint for joint in self.joints if not joint.excluded]

    def orient_tree(self) -> None:
        """Point every tree joint away from the root.

        ``body0``/``body1`` are symmetric in the assembly, so a joint may be
        authored child-first. MuJoCo nests bodies, so one pointing backwards
        makes its own child a second root. Walk out from the articulation root
        and swap the ones facing the wrong way.
        """

        pending = self.tree_joints
        seen = {self.articulation_root} if self.articulation_root in self.parts else set()
        while pending:
            reachable = [j for j in pending if j.parent in seen or j.child in seen]
            if not reachable:  # no root marked, or a separate component
                reachable = [pending[0]]
                seen.add(pending[0].parent)
            for joint in reachable:
                if joint.parent not in seen:
                    joint.parent, joint.child = joint.child, joint.parent
                    joint.anchor, joint.parent_anchor = joint.parent_anchor, joint.anchor
                    # MJCF wants the axis in the child body's frame. After the
                    # swap the child is body0, so negating the body1-frame
                    # axis is only right when both bodies rest unrotated; use
                    # the axis as body0 expresses it instead.
                    joint.axis = _negated(joint.axis_in_parent)
                    joint.lower, joint.upper = (
                        (None if joint.upper is None else -joint.upper),
                        (None if joint.lower is None else -joint.lower),
                    )
                    joint.extra_axes = tuple(
                        (
                            kind,
                            _negated(axis_in_parent),
                            _negated(axis),
                            (None if high is None else -high),
                            (None if low is None else -low),
                        )
                        for kind, axis, axis_in_parent, low, high in joint.extra_axes
                    )
                seen.update((joint.parent, joint.child))
            pending = [joint for joint in pending if joint not in reachable]

    def root(self) -> str:
        children = {joint.child for joint in self.tree_joints}
        roots = [name for name in self.parts if name not in children]
        if len(roots) != 1:
            raise SimulationCapabilityError(
                "MuJoCo simulation requires one spanning articulation tree; "
                f"found root bodies {roots!r}"
            )
        return roots[0]


def simulate_usdz(
    usdz: Path,
    work_dir: Path,
    *,
    seconds: float = 3.0,
    fps: float = 30.0,
    scenario: str = "drop",
) -> SimulationResult:
    """Run an exported USDZ on a floor and report what happened.

    ``drop`` releases it just above the floor and watches it settle. ``tilt``
    settles it first, then tips the floor until it slides, which measures the
    friction the materials authored. ``release`` opens every joint to its limit
    and lets go, which is the motion an articulated object is actually for --
    and, until joints have drives, the motion that shows they hold nothing.

    The motion is recorded at ``fps`` so the viewer can play it back.
    """

    if scenario not in {"drop", "tilt", "release"}:
        raise ValueError(f"unknown scenario {scenario!r}; expected 'drop', 'tilt', or 'release'")

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

    # Trajectory frames are keyed by authored names: the viewer looks bodies
    # and joints up by articraft:name, while MJCF names are the sanitized USD
    # prim names. New exports keep the two equal, but legacy stages may not.
    authored = _authored_names(Usd.Stage.Open(str(usdz)))
    names = tuple(
        authored.get(raw, raw)
        for raw in (
            str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index))
            for index in range(1, model.nbody)
        )
    )
    start = data.xpos[1:].copy()
    separations = _tracked_body_separations(
        model,
        data.xpos,
        # Hinge and slide are the only kinds this MJCF writer emits: a joint
        # with several degrees of freedom becomes one axis per freedom.
        movable_joint_types=(mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE),
    )
    # A tree joint holds its bodies together exactly, so the only thing that can
    # genuinely come apart is a loop pin: it is a constraint the solver pulls
    # shut rather than a coordinate it cannot violate.
    pins = _loop_pin_anchors(
        model,
        data,
        # The mujoco stubs omit mjtEq, though the enum is there at runtime.
        connect_type=mujoco.mjtEq.mjEQ_CONNECT,  # pyright: ignore[reportAttributeAccessIssue]
    )
    worst_pin = 0.0

    root_body = 1  # the free body; MuJoCo orders bodies from the world outward
    movable = [
        index
        for index in range(model.njnt)
        if model.jnt_type[index] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
    ]
    joint_names = tuple(
        authored.get(raw, raw)
        for raw in (
            str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)) for index in movable
        )
    )

    if scenario == "release":
        for index in range(model.njnt):
            # Placing each joint at mid-travel independently is an impossible
            # configuration for a closed loop, and the stiff pin would snap the
            # linkage at the first step. Looped mechanisms release from the
            # assembled rest pose instead.
            if model.neq:
                break
            if model.jnt_type[index] not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                continue
            lower, upper = (float(value) for value in model.jnt_range[index])
            # Halfway through travel, which is where letting go of a door leaves it.
            # Releasing at a limit can rest against the stop and never move: a lid
            # opened past vertical is held open by its own weight.
            data.qpos[model.jnt_qposadr[index]] = (lower + upper) / 2.0
        mujoco.mj_forward(model, data)

    def frame(time: float) -> dict[str, Any]:
        return {
            "t": round(time, 4),
            "bodies": {
                name: {
                    "pos": [round(float(value), 6) for value in data.xpos[index]],
                    # MuJoCo quaternions are (w, x, y, z).
                    "quat": [round(float(value), 6) for value in data.xquat[index]],
                }
                for index, name in enumerate(names, start=1)
            },
            "dofs": {
                name: round(float(data.qpos[model.jnt_qposadr[index]]), 6)
                for name, index in zip(joint_names, movable, strict=True)
            },
        }

    deepest = 0.0
    # Landing is not resting. A dropped object squashes its contacts for an
    # instant on impact, and judging it on that instant fails every object that
    # lands at all, however quietly it then sits.
    settled_from = int(int(seconds / model.opt.timestep) * 0.6)
    resting_depth = 0.0
    diverged = False
    peak_joint_speed = 0.0
    frames = [frame(0.0)]
    every = max(1, round(1.0 / (fps * model.opt.timestep)))
    gravity = float(np.linalg.norm(model.opt.gravity))
    settle_steps = int(1.0 / model.opt.timestep) if scenario == "tilt" else 0
    slip_angle: float | None = None
    resting = None
    touching: float | None = None

    for step in range(int(seconds / model.opt.timestep)):
        if scenario == "tilt" and slip_angle is not None:
            break  # the question is answered; tilting further only topples it
        if scenario == "tilt" and step >= settle_steps:
            # Tipping gravity is equivalent to tipping the floor, and leaves the
            # contact geometry untouched.
            degrees = min(MAX_TILT, TILT_RATE * (step - settle_steps) * model.opt.timestep)
            angle = math.radians(degrees)
            model.opt.gravity[:] = (gravity * math.sin(angle), 0.0, -gravity * math.cos(angle))
            if resting is None:
                resting = data.xpos[root_body].copy()
                touching = _floor_friction(model, data)
            elif slip_angle is None:
                travelled = float(np.linalg.norm((data.xpos[root_body] - resting)[:2]))
                if travelled > SLIP_DISTANCE:
                    slip_angle = degrees

        mujoco.mj_step(model, data)
        worst_pin = max(worst_pin, _loop_pin_gap(data, pins))
        floor_depth = _floor_contact_depth(data)
        if floor_depth is not None:
            deepest = min(deepest, floor_depth)
            if step >= settled_from:
                resting_depth = min(resting_depth, floor_depth)
        for index in movable:
            speed = abs(float(data.qvel[model.jnt_dofadr[index]]))
            peak_joint_speed = max(peak_joint_speed, speed)
        if not np.all(np.isfinite(data.qpos)):
            diverged = True
            break
        if (step + 1) % every == 0:
            frames.append(frame((step + 1) * model.opt.timestep))

    end = data.xpos[1:].copy()
    drift = max(
        (
            abs(float(np.linalg.norm(data.xpos[a] - data.xpos[b])) - was)
            for (a, b), was in separations.items()
        ),
        default=0.0,
    )
    drift = max(drift, worst_pin)
    return SimulationResult(
        bodies=names,
        total_mass=float(sum(model.body_mass)),
        start_height=float(start[:, 2].min()),
        end_height=float(end[:, 2].min()),
        contacts=int(data.ncon),
        deepest_penetration=deepest,
        resting_penetration=resting_depth,
        largest_separation_change=drift,
        residual_velocity=float(np.abs(data.qvel).max()) if model.nv else 0.0,
        diverged=diverged,
        scenario=scenario,
        slip_angle=slip_angle,
        measured_friction=None if slip_angle is None else math.tan(math.radians(slip_angle)),
        peak_joint_speed=peak_joint_speed if movable else None,
        expected_friction=touching,
        trajectory=Trajectory(
            fps=fps,
            frames=tuple(frames),
        ),
    )


def _tracked_body_separations(
    model: Any,
    positions: Any,
    *,
    movable_joint_types: tuple[Any, ...],
) -> dict[tuple[int, int], float]:
    """Body distances that should stay fixed while joints move.

    Any movable joint moves its whole child subtree relative to the rest of the
    object, so a distance that crosses one is expected to change: a slider
    translates what hangs below it, and a hinge swings it through an arc just as
    legitimately. Only pairs with no movable joint between them are rigidly
    related, and those are the ones worth watching -- a distance that changes
    there means parts really did come apart.

    Watching only sliders used to fail every multi link object: releasing a two
    hinge arm read as 24 mm of "separation" and an excavator boom as 448 mm,
    purely because rotating a joint moves everything downstream of it.
    """

    movable_roots = {
        int(model.jnt_bodyid[index])
        for index in range(model.njnt)
        if model.jnt_type[index] in movable_joint_types
    }
    parents = model.body_parentid

    def crosses_a_joint(first: int, second: int) -> bool:
        return any(
            _is_descendant(first, root, parents) != _is_descendant(second, root, parents)
            for root in movable_roots
        )

    return {
        (first, second): float(np.linalg.norm(positions[first] - positions[second]))
        for first in range(1, model.nbody)
        for second in range(first + 1, model.nbody)
        if not crosses_a_joint(first, second)
    }


def _loop_pin_anchors(
    model: Any, data: Any, *, connect_type: Any
) -> list[tuple[int, int, Any, Any]]:
    """Where each loop pin sits in both of the bodies it holds together.

    Read once at the assembled start pose: the pin is a single physical point,
    so its coordinates in either body are fixed facts. Watching those two points
    drift apart is what "did the mechanism stay assembled" actually means, now
    that ordinary joint motion no longer counts.

    Only ``connect`` constraints are read. A weld closure stores a relative pose
    rather than a single anchor in the same slots, so a shared point is not the
    right question to ask of it.
    """

    anchors: list[tuple[int, int, Any, Any]] = []
    for index in range(model.neq):
        if model.eq_type[index] != connect_type:
            continue
        first, second = int(model.eq_obj1id[index]), int(model.eq_obj2id[index])
        if first == 0 or second == 0:
            continue
        local_first = np.asarray(model.eq_data[index][:3], dtype=float)
        world = data.xmat[first].reshape(3, 3) @ local_first + data.xpos[first]
        local_second = data.xmat[second].reshape(3, 3).T @ (world - data.xpos[second])
        anchors.append((first, second, local_first, local_second))
    return anchors


def _loop_pin_gap(data: Any, anchors: list[tuple[int, int, Any, Any]]) -> float:
    """How far the worst loop pin is currently pulled open."""

    worst = 0.0
    for first, second, local_first, local_second in anchors:
        here = data.xmat[first].reshape(3, 3) @ local_first + data.xpos[first]
        there = data.xmat[second].reshape(3, 3) @ local_second + data.xpos[second]
        worst = max(worst, float(np.linalg.norm(here - there)))
    return worst


def _floor_contact_depth(data: Any) -> float | None:
    """The deepest contact against the floor, ignoring the object's own parts.

    Geom 0 is the floor plane, so a contact naming it is the object meeting the
    ground. Everything else is the object touching itself, which is ordinary
    construction -- a rod inside its barrel, a pin inside its bore -- rather
    than a defect.
    """

    depths = [
        float(data.contact.dist[index])
        for index in range(data.ncon)
        if 0 in (int(data.contact.geom1[index]), int(data.contact.geom2[index]))
    ]
    return min(depths) if depths else None


def _is_descendant(body: int, root: int, parents: Any) -> bool:
    while body and body != root:
        body = int(parents[body])
    return body == root


def _floor_friction(model: Any, data: Any) -> float | None:
    """Friction of the geoms actually resting on the floor.

    A crate on rubber feet slides on rubber, whatever its body is made of, so the
    object's minimum friction is the wrong thing to compare a slip angle against.
    """

    values = [
        float(model.geom_friction[geom][0])
        for index in range(data.ncon)
        for geom in (data.contact.geom1[index], data.contact.geom2[index])
        if geom != 0  # geom 0 is the floor
    ]
    return min(values) if values else None


def write_mjcf(usdz: Path, out_dir: Path) -> Path:
    """Translate an exported stage into an MJCF model beside its meshes."""

    out_dir.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.Open(str(usdz))
    if stage is None:
        raise ValueError(f"could not open {usdz}")
    scene = _read_scene(stage)
    if scene.world_joints:
        raise SimulationCapabilityError(
            "MuJoCo translation cannot yet anchor joints to WORLD; dropping them "
            "would simulate the mechanism free-floating: "
            + ", ".join(repr(name) for name in scene.world_joints)
        )
    unsupported_closures = [
        joint.name
        for joint in scene.joints
        if joint.excluded and (joint.kind == "slide" or joint.extra_axes)
    ]
    if unsupported_closures:
        raise SimulationCapabilityError(
            "MuJoCo translation cannot preserve prismatic or multi-DOF loop constraints: "
            + ", ".join(repr(name) for name in unsupported_closures)
        )
    root = scene.root()

    lowest = _lowest_point(scene)
    lift = np.eye(4)
    lift[2, 3] = -lowest + DROP_HEIGHT

    mujoco_el = ET.Element("mujoco", model=usdz.stem or "object")
    ET.SubElement(mujoco_el, "compiler", angle="radian", meshdir=".")
    ET.SubElement(mujoco_el, "option", timestep="0.002", integrator="implicitfast")
    asset = ET.SubElement(mujoco_el, "asset")
    world = ET.SubElement(mujoco_el, "worldbody")
    # MuJoCo takes the elementwise MAX of the two geoms' friction, so a floor with
    # any friction of its own would mask the material's. Zero here means the
    # contact uses exactly what the shape's material authored.
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="5 5 0.1",
        pos="0 0 0",
        friction="0 0.005 0.0001",
    )

    _add_body(world, asset, scene, root, np.linalg.inv(lift), stage, out_dir)

    closures = [joint for joint in scene.joints if joint.excluded]
    if closures:
        equality = ET.SubElement(mujoco_el, "equality")
        for joint in closures:
            # MuJoCo's default equality impedance is sized for light objects;
            # a multi tonne linkage sags visibly on it. A stiff pin is the
            # honest reading of a physical pin.
            stiffness = {"solref": "0.004 1", "solimp": "0.99 0.999 0.0001 0.5 2"}
            if joint.kind is None:
                ET.SubElement(
                    equality,
                    "weld",
                    {"name": joint.name, "body1": joint.parent, "body2": joint.child, **stiffness},
                )
            else:
                # A hinge pin holds a shared point; the connect constraint is
                # exactly that, anchored on the parent side of the pin.
                ET.SubElement(
                    equality,
                    "connect",
                    {
                        "name": joint.name,
                        "body1": joint.parent,
                        "body2": joint.child,
                        "anchor": _vector(joint.parent_anchor),
                        **stiffness,
                    },
                )

    path = out_dir / "model.xml"
    ET.indent(mujoco_el)
    path.write_text(ET.tostring(mujoco_el, encoding="unicode"), encoding="utf-8")
    return path


def _bodies_scope(prim: Usd.Prim) -> Usd.Prim | None:
    """The scope holding an object's rigid bodies, under either schema.

    A v1 stage calls it ``parts``; a rigid-body graph calls it
    ``rigid_bodies``. Both hold one prim per body, which is all a reader needs.
    """

    return prim.GetChild("rigid_bodies") or prim.GetChild("parts") or None


_D6_AXES = {
    "transX": ("slide", (1.0, 0.0, 0.0)),
    "transY": ("slide", (0.0, 1.0, 0.0)),
    "transZ": ("slide", (0.0, 0.0, 1.0)),
    "rotX": ("hinge", (1.0, 0.0, 0.0)),
    "rotY": ("hinge", (0.0, 1.0, 0.0)),
    "rotZ": ("hinge", (0.0, 0.0, 1.0)),
}


def _general_joint_axes(
    prim: Usd.Prim,
) -> list[
    tuple[str, tuple[float, float, float], tuple[float, float, float], float | None, float | None]
]:
    """The free axes of a UsdPhysics.Joint, in USD's own order.

    A D6 joint locks an axis with a LimitAPI whose low exceeds its high, and
    leaves an axis free by carrying no LimitAPI for it at all. Anything else is
    a limited axis. MuJoCo has no six-axis joint, so each free axis becomes a
    sibling hinge or slide on the same body, which composes to the same motion.
    Each entry carries the axis in both endpoint frames, so a joint authored
    child-first can be flipped without leaving the axis in the wrong body.
    """

    free: list[
        tuple[
            str, tuple[float, float, float], tuple[float, float, float], float | None, float | None
        ]
    ] = []
    for token, (kind, axis) in _D6_AXES.items():
        low = _number(_attr(prim, f"limit:{token}:physics:low"))
        high = _number(_attr(prim, f"limit:{token}:physics:high"))
        if low is not None and high is not None and low > high:
            continue  # locked
        child_axis = _rotated_axis(prim, axis, "physics:localRot1")
        parent_axis = _rotated_axis(prim, axis, "physics:localRot0")
        if low is None and high is None:
            free.append((kind, child_axis, parent_axis, None, None))  # free, unlimited
            continue
        free.append((kind, child_axis, parent_axis, low, high))
    return free


def _negated(axis: tuple[float, float, float]) -> tuple[float, float, float]:
    return (-axis[0], -axis[1], -axis[2])


def _authored_names(stage: Usd.Stage | None) -> dict[str, str]:
    """Map sanitized prim names to authored ``articraft:name`` values."""

    if stage is None:
        return {}
    world = stage.GetDefaultPrim()
    objects = [prim for prim in world.GetChildren() if _bodies_scope(prim)]
    if len(objects) != 1:
        return {}
    scopes = [_bodies_scope(objects[0]), objects[0].GetChild("joints")]
    mapping: dict[str, str] = {}
    for scope in scopes:
        for prim in scope.GetChildren() if scope else []:
            value = prim.GetAttribute("articraft:name").Get()
            if value:
                mapping[prim.GetName()] = str(value)
    return mapping


def _read_scene(stage: Usd.Stage) -> _Scene:
    world = stage.GetDefaultPrim()
    objects = [prim for prim in world.GetChildren() if _bodies_scope(prim)]
    if len(objects) != 1:
        raise ValueError("expected one articulated object on the stage")
    obj = objects[0]

    bodies = _bodies_scope(obj)
    assert bodies is not None
    scene = _Scene(parts={prim.GetName(): prim for prim in bodies.GetChildren()})
    scene.articulation_root = next(
        (name for name, prim in scene.parts.items() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)),
        None,
    )
    joints_scope = obj.GetChild("joints")
    if scene.articulation_root is None and joints_scope:
        # A world-anchored assembly puts ArticulationRootAPI on the anchoring
        # joint prim, not on a body; the viewer already reads both.
        for prim in joints_scope.GetChildren():
            if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                continue
            anchored = [
                target.name
                for index in (0, 1)
                for target in prim.GetRelationship(f"physics:body{index}").GetTargets()
                if target.name in scene.parts
            ]
            if anchored:
                scene.articulation_root = anchored[0]
                break
    for prim in joints_scope.GetChildren() if joints_scope else []:
        type_name = str(prim.GetTypeName())
        general = type_name == "PhysicsJoint"
        kind = _JOINT_TYPES.get(type_name)
        if not general and type_name not in _JOINT_TYPES:
            continue
        free: list[
            tuple[
                str,
                tuple[float, float, float],
                tuple[float, float, float],
                float | None,
                float | None,
            ]
        ] = []
        if general:
            free = _general_joint_axes(prim)
            # every axis locked is a fixed joint by another name
            kind = free[0][0] if free else None
        bodies = [prim.GetRelationship(f"physics:body{index}").GetTargets() for index in (0, 1)]
        if not all(bodies):
            # A joint to WORLD has one empty body rel. MuJoCo could anchor
            # it, but this translator does not yet; dropping it silently
            # would simulate a wall-mounted mechanism free-floating.
            scene.world_joints.append(prim.GetName())
            continue
        scene.joints.append(
            _Joint(
                name=prim.GetName(),
                kind=kind,
                parent=bodies[0][0].name,
                child=bodies[1][0].name,
                anchor=_triple(_attr(prim, "physics:localPos1", (0.0, 0.0, 0.0))),
                axis=free[0][1] if free else _joint_axis(prim),
                lower=free[0][3] if free else _number(_attr(prim, "physics:lowerLimit")),
                upper=free[0][4] if free else _number(_attr(prim, "physics:upperLimit")),
                excluded=bool(_attr(prim, "physics:excludeFromArticulation", False)),
                parent_anchor=_triple(_attr(prim, "physics:localPos0", (0.0, 0.0, 0.0))),
                axis_in_parent=(free[0][2] if free else _joint_axis(prim, "physics:localRot0")),
                extra_axes=tuple(free[1:]),
            )
        )
    scene.orient_tree()
    return scene


def _joint_axis(
    prim: Usd.Prim, rotation_attr: str = "physics:localRot1"
) -> tuple[float, float, float]:
    """Return the USD joint axis in one endpoint body's frame.

    MJCF wants the axis in the frame of whichever body ends up as the MuJoCo
    child, so callers read it through ``physics:localRot1`` for body1 and
    ``physics:localRot0`` for a joint flipped to hang from body0.
    """

    token = str(_attr(prim, "physics:axis", "X"))
    axis = _AXES.get(token)
    if axis is None:
        raise ValueError(f"unsupported USD joint axis: {token!r}")
    return _rotated_axis(prim, axis, rotation_attr)


def _rotated_axis(
    prim: Usd.Prim,
    axis: tuple[float, float, float],
    rotation_attr: str,
) -> tuple[float, float, float]:
    rotation = _attr(prim, rotation_attr)
    if rotation is None:
        return axis
    frame = Gf.Matrix4d(1.0)
    frame.SetRotate(rotation)
    transformed = frame.TransformDir(Gf.Vec3d(*axis)).GetNormalized()
    return (float(transformed[0]), float(transformed[1]), float(transformed[2]))


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
    placement = {"name": part_name, "pos": _vector(relative[:3, 3])}
    orientation = _quaternion(relative[:3, :3])
    if orientation is not None:
        placement["quat"] = _vector(orientation)
    body = ET.SubElement(parent_el, "body", placement)

    joint = next((item for item in scene.tree_joints if item.child == part_name), None)
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
            lower, upper = joint.lower * scale, joint.upper * scale
            if not _representable_range(lower, upper):
                raise ValueError(
                    f"joint {joint.name!r} has a range too narrow to simulate "
                    f"({lower} to {upper}); a joint that cannot move is a fixed joint"
                )
            attributes["range"] = f"{lower:.9g} {upper:.9g}"
            attributes["limited"] = "true"
        ET.SubElement(body, "joint", attributes)
        for index, (kind, axis, _parent_axis, low, high) in enumerate(joint.extra_axes, start=2):
            extra = {
                "name": f"{joint.name}_{index}",
                "type": kind,
                "pos": _vector(joint.anchor),
                "axis": _vector(axis),
            }
            if low is not None and high is not None:
                scale = math.pi / 180.0 if kind == "hinge" else 1.0
                extra["range"] = f"{low * scale:.9g} {high * scale:.9g}"
                extra["limited"] = "true"
            ET.SubElement(body, "joint", extra)

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

    for child in (item.child for item in scene.tree_joints if item.parent == part_name):
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


def _quaternion(rotation: np.ndarray) -> tuple[float, float, float, float] | None:
    """A rest rotation as an MJCF ``(w, x, y, z)`` quaternion, or None if upright.

    Dropping this silently misplaces any part whose joint carries an ``rpy``.
    """

    if np.allclose(rotation, np.eye(3), atol=1e-9):
        return None
    x, y, z, w = Rotation.from_matrix(rotation).as_quat()
    return (float(w), float(x), float(y), float(z))


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


def _representable_range(lower: float, upper: float) -> bool:
    """Whether the two ends still differ once written into the MJCF text."""

    return f"{lower:.9g}" != f"{upper:.9g}"


def _vector(value: Any) -> str:
    # Nine significant figures, not six decimals: a joint range or an anchor
    # offset can legitimately be far smaller than a micron, and rounding it to
    # zero here turns a real range into a degenerate one MuJoCo refuses to load.
    return " ".join(f"{float(component):.9g}" for component in value)
