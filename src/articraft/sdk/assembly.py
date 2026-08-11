from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TypeAlias, cast

import numpy as np
import trimesh

from articraft.sdk._values import _as_name
from articraft.sdk.bodies import RigidBody, RigidBodyRef
from articraft.sdk.errors import ValidationError
from articraft.sdk.mass import MassProperties
from articraft.sdk.physics import BodyState, PhysicsScene

Vec3: TypeAlias = tuple[float, float, float]
Mat4: TypeAlias = np.ndarray
Matrix4: TypeAlias = tuple[tuple[float, float, float, float], ...]


class JointAxis(StrEnum):
    """The six independent freedoms of a USD D6 joint."""

    TRANS_X = "transX"
    TRANS_Y = "transY"
    TRANS_Z = "transZ"
    ROT_X = "rotX"
    ROT_Y = "rotY"
    ROT_Z = "rotZ"

    @property
    def is_rotational(self) -> bool:
        return self.value.startswith("rot")

    @property
    def component(self) -> int:
        return {"X": 0, "Y": 1, "Z": 2}[self.value[-1]]


@dataclass(frozen=True, slots=True)
class JointFrame:
    """A joint endpoint pose in its rigid body, in metres and radians."""

    xyz: Vec3 = (0.0, 0.0, 0.0)
    rpy: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "xyz", _as_vec3(self.xyz, field_name="joint frame xyz"))
        object.__setattr__(self, "rpy", _as_vec3(self.rpy, field_name="joint frame rpy"))


@dataclass(frozen=True, slots=True)
class JointDOF:
    """One free or limited axis on a joint; axes not listed are locked."""

    axis: JointAxis | str
    limits: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        try:
            axis = self.axis if isinstance(self.axis, JointAxis) else JointAxis(str(self.axis))
        except ValueError as exc:
            raise ValidationError(f"unknown joint axis: {self.axis!r}") from exc
        object.__setattr__(self, "axis", axis)
        if self.limits is None:
            return
        if isinstance(self.limits, (str, bytes)):
            raise ValidationError("joint DOF limits must be (lower, upper)")
        try:
            values = tuple(self.limits)
        except TypeError as exc:
            raise ValidationError("joint DOF limits must be (lower, upper)") from exc
        if len(values) != 2:
            raise ValidationError("joint DOF limits must be (lower, upper)")
        lower = _finite(values[0], field_name="joint DOF lower limit")
        upper = _finite(values[1], field_name="joint DOF upper limit")
        if lower > upper:
            raise ValidationError("joint DOF lower limit cannot exceed its upper limit")
        if not lower <= 0.0 <= upper:
            raise ValidationError("joint DOF limits must contain the zero configuration")
        object.__setattr__(self, "limits", (lower, upper))


@dataclass(frozen=True, slots=True)
class _WorldEndpoint:
    def __repr__(self) -> str:
        return "WORLD"


WORLD = _WorldEndpoint()
"""The USD world endpoint. A joint may connect one rigid body to ``WORLD``."""

JointEndpoint: TypeAlias = RigidBody | _WorldEndpoint
JointEndpointRef: TypeAlias = RigidBodyRef | _WorldEndpoint


@dataclass(frozen=True, slots=True, eq=False)
class Joint:
    """A physical constraint between two bodies, or one body and the world."""

    name: str
    body0: JointEndpoint
    frame0: JointFrame
    body1: JointEndpoint
    frame1: JointFrame
    dofs: tuple[JointDOF, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _as_name(self.name, field_name="joint name"))
        if not isinstance(self.frame0, JointFrame) or not isinstance(self.frame1, JointFrame):
            raise ValidationError(f"joint {self.name!r} frames must be JointFrame values")
        try:
            dofs = tuple(self.dofs)
        except TypeError as exc:
            raise ValidationError(f"joint {self.name!r} dofs must be JointDOF values") from exc
        if any(not isinstance(dof, JointDOF) for dof in dofs):
            raise ValidationError(f"joint {self.name!r} dofs must be JointDOF values")
        axes = [cast(JointAxis, dof.axis) for dof in dofs]
        if len(set(axes)) != len(axes):
            raise ValidationError(f"joint {self.name!r} has duplicate DOF axes")
        order = {axis: index for index, axis in enumerate(JointAxis)}
        object.__setattr__(
            self,
            "dofs",
            tuple(sorted(dofs, key=lambda dof: order[cast(JointAxis, dof.axis)])),
        )
        if self.body0 is WORLD and self.body1 is WORLD:
            raise ValidationError(f"joint {self.name!r} cannot connect WORLD to WORLD")
        if self.body0 is self.body1:
            raise ValidationError(f"joint {self.name!r} endpoints cannot be the same")

    @property
    def is_fixed(self) -> bool:
        return not self.dofs

    @property
    def is_revolute(self) -> bool:
        return len(self.dofs) == 1 and cast(JointAxis, self.dofs[0].axis).is_rotational

    @property
    def is_prismatic(self) -> bool:
        return len(self.dofs) == 1 and not cast(JointAxis, self.dofs[0].axis).is_rotational

    def dof_id(self, dof: JointDOF) -> str:
        return f"{self.name}.{cast(JointAxis, dof.axis).value}"


JointRef: TypeAlias = str | Joint
ArticulationRootRef: TypeAlias = RigidBodyRef | Joint


@dataclass(frozen=True, slots=True, eq=False)
class Articulation:
    """A reduced-coordinate spanning tree within the physical joint graph."""

    name: str
    root: RigidBody | Joint
    joints: tuple[Joint, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _as_name(self.name, field_name="articulation name"))
        if self.joints is not None:
            try:
                joints = tuple(self.joints)
            except TypeError as exc:
                raise ValidationError(
                    f"articulation {self.name!r} joints must be Joint values"
                ) from exc
            if any(not isinstance(joint, Joint) for joint in joints):
                raise ValidationError(f"articulation {self.name!r} joints must be Joint values")
            object.__setattr__(self, "joints", joints)


@dataclass(frozen=True, slots=True, init=False)
class PhysicsState:
    """Authoritative world transforms for every rigid body in an assembly."""

    body_poses: Mapping[str, Matrix4]
    dof_positions: Mapping[str, float]

    def __init__(
        self,
        body_poses: Mapping[str, JointFrame | Sequence[Sequence[float]] | Mat4]
        | Mapping[RigidBodyRef, JointFrame | Sequence[Sequence[float]] | Mat4],
        *,
        dof_positions: Mapping[str, float] | None = None,
    ) -> None:
        poses: dict[str, Matrix4] = {}
        for key, value in body_poses.items():
            name = _body_name(key, field_name="physics state rigid body")
            if name in poses:
                raise ValidationError(f"physics state contains duplicate rigid body {name!r}")
            poses[name] = _matrix_tuple(
                _as_matrix4(value, field_name=f"physics state pose for {name!r}")
            )
        positions: dict[str, float] = {}
        for key, value in dict(dof_positions or {}).items():
            name = _as_name(key, field_name="physics state DOF")
            positions[name] = _finite(value, field_name=f"physics state DOF {name!r}")
        object.__setattr__(self, "body_poses", MappingProxyType(poses))
        object.__setattr__(self, "dof_positions", MappingProxyType(positions))

    def matrix(self, body: RigidBodyRef) -> Mat4:
        name = _body_name(body, field_name="rigid body")
        try:
            return np.asarray(self.body_poses[name], dtype=np.float64)
        except KeyError as exc:
            raise ValidationError(f"physics state has no pose for rigid body {name!r}") from exc


@dataclass(frozen=True, slots=True)
class ResolvedJoint:
    joint: Joint
    articulation: str | None
    exclude_from_articulation: bool


@dataclass(frozen=True, slots=True)
class ResolvedArticulation:
    articulation: Articulation
    rigid_bodies: tuple[RigidBody, ...]
    joints: tuple[Joint, ...]


@dataclass(frozen=True, slots=True)
class ResolvedRigidBodyAssembly:
    """The validated, immutable view consumed by compilers and geometry tools."""

    name: str
    rigid_bodies: tuple[RigidBody, ...]
    joints: tuple[ResolvedJoint, ...]
    articulations: tuple[ResolvedArticulation, ...]
    reference_state: PhysicsState
    has_closed_loops: bool
    scene: PhysicsScene

    def get_rigid_body(self, body: RigidBodyRef) -> RigidBody:
        name = _body_name(body, field_name="rigid body")
        for candidate in self.rigid_bodies:
            if candidate.name == name:
                return candidate
        raise ValidationError(f"unknown rigid body: {name!r}")

    def get_joint(self, joint: JointRef) -> ResolvedJoint:
        name = joint.name if isinstance(joint, Joint) else _as_name(joint, field_name="joint")
        for candidate in self.joints:
            if candidate.joint.name == name:
                return candidate
        raise ValidationError(f"unknown joint: {name!r}")

    def world_transforms(self, state: PhysicsState | None = None) -> dict[str, Mat4]:
        checked = self.validate_state(state or self.reference_state)
        return {
            name: np.asarray(matrix, dtype=np.float64)
            for name, matrix in checked.body_poses.items()
        }

    def validate_state(
        self,
        state: PhysicsState,
        *,
        linear_tolerance: float = 1e-6,
        angular_tolerance: float = 1e-6,
    ) -> PhysicsState:
        if not isinstance(state, PhysicsState):
            raise ValidationError("state must be a PhysicsState")
        expected = {body.name for body in self.rigid_bodies}
        found = set(state.body_poses)
        if found != expected:
            missing = sorted(expected - found)
            unexpected = sorted(found - expected)
            raise ValidationError(
                f"physics state rigid bodies do not match the assembly: "
                f"missing={missing!r} unexpected={unexpected!r}"
            )
        linear_tolerance = _positive_tolerance(linear_tolerance, "linear tolerance")
        angular_tolerance = _positive_tolerance(angular_tolerance, "angular tolerance")
        matrices = {
            body.name: _as_matrix4(
                state.body_poses[body.name], field_name=f"physics state pose for {body.name!r}"
            )
            for body in self.rigid_bodies
        }
        valid_dof_ids = {
            joint.joint.dof_id(dof) for joint in self.joints for dof in joint.joint.dofs
        }
        unknown_dofs = sorted(set(state.dof_positions) - valid_dof_ids)
        if unknown_dofs:
            raise ValidationError(f"physics state contains unknown DOFs: {unknown_dofs!r}")

        derived: dict[str, float] = {}
        for resolved_joint in self.joints:
            joint = resolved_joint.joint
            values = _joint_values(joint, matrices)
            authored = {cast(JointAxis, dof.axis): dof for dof in joint.dofs}
            # Euler angles are ambiguous at gimbal lock: with pitch at +-90 the
            # roll and yaw terms describe the same rotation, so a decomposition
            # can hand roll's share to yaw, or to a locked axis that never
            # moved. When the joint's own values rebuild the pose exactly, they
            # are the honest reading and the decomposition is just one of the
            # several that happen to fit.
            supplied_free = {
                joint.dof_id(dof): state.dof_positions[joint.dof_id(dof)]
                for dof in joint.dofs
                if joint.dof_id(dof) in state.dof_positions
            }
            if supplied_free:
                candidate = _values_from(joint, supplied_free)
                if _reproduces_pose(
                    joint, candidate, matrices, angular_tolerance, linear_tolerance
                ):
                    values = candidate
            for axis, value in values.items():
                dof = authored.get(axis)
                tolerance = angular_tolerance if axis.is_rotational else linear_tolerance
                if dof is None:
                    if abs(value) > tolerance:
                        raise ValidationError(
                            f"physics state violates locked axis {axis.value!r} on joint "
                            f"{joint.name!r}: value={value:.9g}"
                        )
                    continue
                dof_id = joint.dof_id(dof)
                if dof.limits is not None:
                    lower, upper = dof.limits
                    if value < lower - tolerance or value > upper + tolerance:
                        raise ValidationError(
                            f"physics state violates limits on {dof_id!r}: "
                            f"value={value:.9g} limits={dof.limits!r}"
                        )
                supplied = state.dof_positions.get(dof_id)
                if supplied is not None and not math.isclose(
                    supplied, value, rel_tol=0.0, abs_tol=tolerance
                ):
                    raise ValidationError(
                        f"physics state DOF {dof_id!r} disagrees with body poses: "
                        f"supplied={supplied:.9g} derived={value:.9g}"
                    )
                derived[dof_id] = value
        return PhysicsState(matrices, dof_positions=derived)

    def forward_kinematics(self, dof_positions: Mapping[str, float] | None = None) -> PhysicsState:
        """Pose an acyclic physical graph; closed loops require a solver or body poses."""

        if self.has_closed_loops:
            raise ValidationError(
                "closed-loop assemblies cannot be posed from joint positions; "
                "provide a complete PhysicsState or use a physics solver"
            )
        positions = {
            _as_name(name, field_name="joint position"): _finite(
                value, field_name=f"joint position {name!r}"
            )
            for name, value in dict(dof_positions or {}).items()
        }
        valid = {joint.joint.dof_id(dof) for joint in self.joints for dof in joint.joint.dofs}
        unknown = sorted(set(positions) - valid)
        if unknown:
            raise ValidationError(f"unknown joint positions: {unknown!r}")
        for resolved_joint in self.joints:
            for dof in resolved_joint.joint.dofs:
                value = positions.get(resolved_joint.joint.dof_id(dof), 0.0)
                if dof.limits is not None and not dof.limits[0] <= value <= dof.limits[1]:
                    raise ValidationError(
                        f"joint position {resolved_joint.joint.dof_id(dof)!r} is outside "
                        f"limits {dof.limits!r}"
                    )

        transforms = _propagate_transforms(
            self.rigid_bodies,
            tuple(item.joint for item in self.joints),
            self.articulations,
            positions,
        )
        return self.validate_state(PhysicsState(transforms, dof_positions=positions))


@dataclass
class RigidBodyAssembly:
    """A connected graph of USD rigid bodies, joints, and solver articulations."""

    name: str
    scene: PhysicsScene = field(default=PhysicsScene(), kw_only=True)
    rigid_bodies: list[RigidBody] = field(default_factory=list, init=False)
    joints: list[Joint] = field(default_factory=list, init=False)
    articulations: list[Articulation] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.name = _as_name(self.name, field_name="assembly name")
        if not isinstance(self.scene, PhysicsScene):
            raise ValidationError(f"assembly {self.name!r} scene must be a PhysicsScene")

    @property
    def meters_per_unit(self) -> float:
        return 1.0

    def rigid_body(
        self,
        name: str,
        *,
        mass_properties: MassProperties | None = None,
        body_state: BodyState | None = None,
    ) -> RigidBody:
        body = RigidBody(
            name,
            mass_properties=mass_properties,
            body_state=BodyState() if body_state is None else body_state,
        )
        if any(existing.name == body.name for existing in self.rigid_bodies):
            raise ValidationError(f"duplicate rigid body name: {body.name!r}")
        self.rigid_bodies.append(body)
        return body

    def joint(
        self,
        name: str,
        *,
        body0: JointEndpointRef,
        frame0: JointFrame | None = None,
        body1: JointEndpointRef,
        frame1: JointFrame | None = None,
        dofs: Iterable[JointDOF] = (),
    ) -> Joint:
        joint = Joint(
            name=name,
            body0=self._endpoint(body0, field_name="body0"),
            frame0=JointFrame() if frame0 is None else frame0,
            body1=self._endpoint(body1, field_name="body1"),
            frame1=JointFrame() if frame1 is None else frame1,
            dofs=tuple(dofs),
        )
        if any(existing.name == joint.name for existing in self.joints):
            raise ValidationError(f"duplicate joint name: {joint.name!r}")
        self.joints.append(joint)
        return joint

    def articulation(
        self,
        name: str,
        *,
        root: ArticulationRootRef,
        joints: Iterable[JointRef] | None = None,
    ) -> Articulation:
        resolved_root = self._root(root)
        selected = None if joints is None else tuple(self.get_joint(joint) for joint in joints)
        articulation = Articulation(name=name, root=resolved_root, joints=selected)
        if any(existing.name == articulation.name for existing in self.articulations):
            raise ValidationError(f"duplicate articulation name: {articulation.name!r}")
        self.articulations.append(articulation)
        return articulation

    def get_rigid_body(self, body: RigidBodyRef) -> RigidBody:
        name = _body_name(body, field_name="rigid body")
        for existing in self.rigid_bodies:
            if existing.name == name:
                return existing
        raise ValidationError(f"unknown rigid body: {name!r}")

    def get_joint(self, joint: JointRef) -> Joint:
        name = joint.name if isinstance(joint, Joint) else _as_name(joint, field_name="joint")
        for existing in self.joints:
            if existing.name == name:
                return existing
        raise ValidationError(f"unknown joint: {name!r}")

    def get_articulation(self, articulation: str | Articulation) -> Articulation:
        name = (
            articulation.name
            if isinstance(articulation, Articulation)
            else _as_name(articulation, field_name="articulation")
        )
        for existing in self.articulations:
            if existing.name == name:
                return existing
        raise ValidationError(f"unknown articulation: {name!r}")

    def validate(self) -> None:
        self.resolve()

    def resolve(self) -> ResolvedRigidBodyAssembly:
        self.name = _as_name(self.name, field_name="assembly name")
        _validate_members(self)
        resolved_articulations, selected_by_joint, articulated_bodies = _resolve_articulations(self)
        resolved_joints = tuple(
            ResolvedJoint(
                joint=joint,
                articulation=selected_by_joint.get(joint),
                exclude_from_articulation=(
                    joint not in selected_by_joint
                    and any(
                        endpoint in articulated_bodies
                        for endpoint in (joint.body0, joint.body1)
                        if endpoint is not WORLD
                    )
                ),
            )
            for joint in self.joints
        )
        has_closed_loops = _graph_has_cycle(tuple(self.rigid_bodies), tuple(self.joints))
        transforms = _propagate_transforms(
            tuple(self.rigid_bodies),
            tuple(self.joints),
            resolved_articulations,
            {},
        )
        unresolved = ResolvedRigidBodyAssembly(
            name=self.name,
            rigid_bodies=tuple(self.rigid_bodies),
            joints=resolved_joints,
            articulations=resolved_articulations,
            reference_state=PhysicsState(transforms),
            has_closed_loops=has_closed_loops,
            scene=self.scene,
        )
        reference = unresolved.validate_state(unresolved.reference_state)
        return ResolvedRigidBodyAssembly(
            name=unresolved.name,
            rigid_bodies=unresolved.rigid_bodies,
            joints=unresolved.joints,
            articulations=unresolved.articulations,
            reference_state=reference,
            has_closed_loops=unresolved.has_closed_loops,
            scene=unresolved.scene,
        )

    def physics_state(
        self,
        body_poses: Mapping[str, JointFrame | Sequence[Sequence[float]] | Mat4]
        | Mapping[RigidBodyRef, JointFrame | Sequence[Sequence[float]] | Mat4],
        *,
        dof_positions: Mapping[str, float] | None = None,
    ) -> PhysicsState:
        return self.resolve().validate_state(PhysicsState(body_poses, dof_positions=dof_positions))

    def _endpoint(self, endpoint: JointEndpointRef, *, field_name: str) -> JointEndpoint:
        if endpoint is WORLD:
            return WORLD
        try:
            return self.get_rigid_body(cast(RigidBodyRef, endpoint))
        except ValidationError as exc:
            raise ValidationError(f"unknown {field_name} rigid body: {endpoint!r}") from exc

    def _root(self, root: ArticulationRootRef) -> RigidBody | Joint:
        if isinstance(root, RigidBody):
            return self.get_rigid_body(root)
        if isinstance(root, Joint):
            return self.get_joint(root)
        name = _as_name(root, field_name="articulation root")
        bodies = [body for body in self.rigid_bodies if body.name == name]
        joints = [joint for joint in self.joints if joint.name == name]
        if bodies and joints:
            raise ValidationError(
                f"ambiguous articulation root {name!r}; pass the rigid body or joint object"
            )
        if bodies:
            return bodies[0]
        if joints:
            return joints[0]
        raise ValidationError(f"unknown articulation root: {name!r}")


def _validate_members(assembly: RigidBodyAssembly) -> None:
    if not assembly.rigid_bodies:
        raise ValidationError("assembly must contain at least one rigid body")
    if any(not isinstance(body, RigidBody) for body in assembly.rigid_bodies):
        raise ValidationError("assembly rigid_bodies must contain RigidBody values")
    for body in assembly.rigid_bodies:
        body.validate()
    body_names = [body.name for body in assembly.rigid_bodies]
    if len(set(body_names)) != len(body_names):
        raise ValidationError("rigid body names must be unique")
    body_ids = {id(body) for body in assembly.rigid_bodies}

    if any(not isinstance(joint, Joint) for joint in assembly.joints):
        raise ValidationError("assembly joints must contain Joint values")
    joint_names = [joint.name for joint in assembly.joints]
    if len(set(joint_names)) != len(joint_names):
        raise ValidationError("joint names must be unique")
    for joint in assembly.joints:
        for field_name, endpoint in (("body0", joint.body0), ("body1", joint.body1)):
            if endpoint is not WORLD and id(endpoint) not in body_ids:
                raise ValidationError(
                    f"joint {joint.name!r} references a {field_name} outside this assembly"
                )

    if any(not isinstance(item, Articulation) for item in assembly.articulations):
        raise ValidationError("assembly articulations must contain Articulation values")
    names = [item.name for item in assembly.articulations]
    if len(set(names)) != len(names):
        raise ValidationError("articulation names must be unique")
    joint_ids = {id(joint) for joint in assembly.joints}
    for articulation in assembly.articulations:
        root = articulation.root
        if isinstance(root, RigidBody) and id(root) not in body_ids:
            raise ValidationError(
                f"articulation {articulation.name!r} root body is outside this assembly"
            )
        if isinstance(root, Joint) and id(root) not in joint_ids:
            raise ValidationError(
                f"articulation {articulation.name!r} root joint is outside this assembly"
            )
    if not _graph_connected(tuple(assembly.rigid_bodies), tuple(assembly.joints)):
        raise ValidationError("assembly rigid bodies must form one connected joint graph")


def _resolve_articulations(
    assembly: RigidBodyAssembly,
) -> tuple[tuple[ResolvedArticulation, ...], dict[Joint, str], set[RigidBody]]:
    selected_by_joint: dict[Joint, str] = {}
    articulated_bodies: set[RigidBody] = set()
    resolved: list[ResolvedArticulation] = []
    for articulation in assembly.articulations:
        selected = articulation.joints
        if selected is None:
            if len(assembly.articulations) != 1 or _graph_has_cycle(
                tuple(assembly.rigid_bodies), tuple(assembly.joints)
            ):
                raise ValidationError(
                    f"articulation {articulation.name!r} requires explicit joints because "
                    "the assembly topology is not one unambiguous tree"
                )
            selected = tuple(assembly.joints)
        if len(set(selected)) != len(selected):
            raise ValidationError(f"articulation {articulation.name!r} repeats a joint")
        if any(joint not in assembly.joints for joint in selected):
            raise ValidationError(
                f"articulation {articulation.name!r} contains a joint outside this assembly"
            )
        bodies = _validate_articulation_tree(articulation, selected)
        overlap = sorted(body.name for body in bodies if body in articulated_bodies)
        if overlap:
            raise ValidationError(
                f"rigid bodies may belong to only one articulation; repeated={overlap!r}"
            )
        for joint in selected:
            previous = selected_by_joint.get(joint)
            if previous is not None:
                raise ValidationError(
                    f"joint {joint.name!r} belongs to articulations {previous!r} and "
                    f"{articulation.name!r}"
                )
            selected_by_joint[joint] = articulation.name
        articulated_bodies.update(bodies)
        resolved.append(
            ResolvedArticulation(
                articulation=articulation,
                rigid_bodies=tuple(sorted(bodies, key=lambda body: body.name)),
                joints=selected,
            )
        )
    return tuple(resolved), selected_by_joint, articulated_bodies


def _validate_articulation_tree(
    articulation: Articulation, selected: tuple[Joint, ...]
) -> set[RigidBody]:
    nodes: set[JointEndpoint] = set()
    for joint in selected:
        nodes.update((joint.body0, joint.body1))

    root = articulation.root
    if isinstance(root, Joint):
        if root not in selected:
            raise ValidationError(
                f"fixed articulation {articulation.name!r} root joint must be selected"
            )
        if (root.body0 is WORLD) == (root.body1 is WORLD):
            raise ValidationError(
                f"articulation {articulation.name!r} joint root must connect one body to WORLD"
            )
        if WORLD not in nodes:
            raise ValidationError(f"articulation {articulation.name!r} fixed root is disconnected")
    elif isinstance(root, RigidBody):
        if not selected:
            nodes.add(root)
        if root not in nodes:
            raise ValidationError(
                f"articulation {articulation.name!r} root body is not in its selected tree"
            )
        if WORLD in nodes:
            raise ValidationError(
                f"floating articulation {articulation.name!r} cannot select a WORLD joint; "
                "use that joint as the articulation root"
            )
    else:
        raise ValidationError(
            f"articulation {articulation.name!r} root must be a RigidBody or Joint"
        )

    if len(selected) != max(0, len(nodes) - 1) or _edge_graph_has_cycle(selected):
        raise ValidationError(f"articulation {articulation.name!r} joints must form a tree")
    if nodes and not _nodes_connected(nodes, selected):
        raise ValidationError(f"articulation {articulation.name!r} joints must be connected")
    return {cast(RigidBody, node) for node in nodes if node is not WORLD}


def _propagate_transforms(
    bodies: tuple[RigidBody, ...],
    joints: tuple[Joint, ...],
    articulations: tuple[ResolvedArticulation, ...],
    positions: dict[str, float],
) -> dict[str, Mat4]:
    transforms: dict[JointEndpoint, Mat4] = {}
    if any(WORLD in (joint.body0, joint.body1) for joint in joints):
        transforms[WORLD] = np.identity(4, dtype=np.float64)
    else:
        root: RigidBody | None = None
        if articulations:
            authored_root = articulations[0].articulation.root
            if isinstance(authored_root, RigidBody):
                root = authored_root
        transforms[root or bodies[0]] = np.identity(4, dtype=np.float64)

    # Authored order, not a set: Joint hashes by identity, so set iteration
    # follows memory addresses. In a tree that only changes how many passes
    # this takes, but a closed loop offers two paths to the same body, and
    # the winner decides the low bits of its transform. That reached the file.
    remaining = list(joints)
    while remaining:
        progressed = False
        for joint in tuple(remaining):
            known0 = joint.body0 in transforms
            known1 = joint.body1 in transforms
            if not known0 and not known1:
                continue
            frame0 = _frame_matrix(joint.frame0)
            frame1 = _frame_matrix(joint.frame1)
            motion = _motion_matrix(joint, positions)
            if known0 and not known1:
                transforms[joint.body1] = (
                    transforms[joint.body0] @ frame0 @ motion @ np.linalg.inv(frame1)
                )
            elif known1 and not known0:
                transforms[joint.body0] = (
                    transforms[joint.body1] @ frame1 @ np.linalg.inv(motion) @ np.linalg.inv(frame0)
                )
            remaining.remove(joint)
            progressed = True
        if not progressed:
            raise ValidationError("assembly graph could not be placed from its root")
    return {body.name: transforms[body] for body in bodies}


def _values_from(joint: Joint, positions: Mapping[str, float]) -> dict[JointAxis, float]:
    """The six axis values implied by the joint values a caller supplied."""

    values = dict.fromkeys(JointAxis, 0.0)
    for dof in joint.dofs:
        values[cast(JointAxis, dof.axis)] = positions.get(joint.dof_id(dof), 0.0)
    return values


def _reproduces_pose(
    joint: Joint,
    values: Mapping[JointAxis, float],
    transforms: Mapping[str, Mat4],
    angular_tolerance: float,
    linear_tolerance: float,
) -> bool:
    """Whether these axis values rebuild the joint's actual relative transform."""

    positions = {joint.dof_id(dof): values[cast(JointAxis, dof.axis)] for dof in joint.dofs}
    body0 = (
        np.identity(4, dtype=np.float64)
        if joint.body0 is WORLD
        else transforms[cast(RigidBody, joint.body0).name]
    )
    body1 = (
        np.identity(4, dtype=np.float64)
        if joint.body1 is WORLD
        else transforms[cast(RigidBody, joint.body1).name]
    )
    actual = np.linalg.inv(body0 @ _frame_matrix(joint.frame0)) @ (
        body1 @ _frame_matrix(joint.frame1)
    )
    rebuilt = _motion_matrix(joint, positions)
    tolerance = max(angular_tolerance, linear_tolerance)
    return bool(np.allclose(actual, rebuilt, rtol=0.0, atol=tolerance * 10.0))


def _joint_values(joint: Joint, transforms: Mapping[str, Mat4]) -> dict[JointAxis, float]:
    body0 = (
        np.identity(4, dtype=np.float64)
        if joint.body0 is WORLD
        else transforms[cast(RigidBody, joint.body0).name]
    )
    body1 = (
        np.identity(4, dtype=np.float64)
        if joint.body1 is WORLD
        else transforms[cast(RigidBody, joint.body1).name]
    )
    relative = np.linalg.inv(body0 @ _frame_matrix(joint.frame0)) @ (
        body1 @ _frame_matrix(joint.frame1)
    )
    rotation = trimesh.transformations.euler_from_matrix(relative, axes="sxyz")
    return {
        JointAxis.TRANS_X: float(relative[0, 3]),
        JointAxis.TRANS_Y: float(relative[1, 3]),
        JointAxis.TRANS_Z: float(relative[2, 3]),
        JointAxis.ROT_X: float(rotation[0]),
        JointAxis.ROT_Y: float(rotation[1]),
        JointAxis.ROT_Z: float(rotation[2]),
    }


def _motion_matrix(joint: Joint, positions: Mapping[str, float]) -> Mat4:
    translation = np.zeros(3, dtype=np.float64)
    rotation = np.zeros(3, dtype=np.float64)
    for dof in joint.dofs:
        axis = cast(JointAxis, dof.axis)
        value = positions.get(joint.dof_id(dof), 0.0)
        if axis.is_rotational:
            rotation[axis.component] = value
        else:
            translation[axis.component] = value
    matrix = np.identity(4, dtype=np.float64)
    matrix[:3, 3] = translation
    return matrix @ np.asarray(
        trimesh.transformations.euler_matrix(*rotation, axes="sxyz"), dtype=np.float64
    )


def _frame_matrix(frame: JointFrame) -> Mat4:
    matrix = np.asarray(
        trimesh.transformations.euler_matrix(*frame.rpy, axes="sxyz"), dtype=np.float64
    )
    matrix[:3, 3] = np.asarray(frame.xyz, dtype=np.float64)
    return matrix


def _graph_connected(bodies: tuple[RigidBody, ...], joints: tuple[Joint, ...]) -> bool:
    if len(bodies) <= 1:
        return True
    nodes: set[JointEndpoint] = set(bodies)
    if any(WORLD in (joint.body0, joint.body1) for joint in joints):
        nodes.add(WORLD)
    return _nodes_connected(nodes, joints)


def _nodes_connected(nodes: set[JointEndpoint], joints: Iterable[Joint]) -> bool:
    if not nodes:
        return True
    adjacency: dict[JointEndpoint, set[JointEndpoint]] = {node: set() for node in nodes}
    for joint in joints:
        adjacency.setdefault(joint.body0, set()).add(joint.body1)
        adjacency.setdefault(joint.body1, set()).add(joint.body0)
    visited: set[JointEndpoint] = set()
    stack = [next(iter(nodes))]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adjacency.get(node, set()) - visited)
    return nodes <= visited


def _graph_has_cycle(bodies: tuple[RigidBody, ...], joints: tuple[Joint, ...]) -> bool:
    del bodies
    return _edge_graph_has_cycle(joints)


def _edge_graph_has_cycle(joints: Iterable[Joint]) -> bool:
    parent: dict[JointEndpoint, JointEndpoint] = {}

    def find(node: JointEndpoint) -> JointEndpoint:
        parent.setdefault(node, node)
        while parent[node] is not node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for joint in joints:
        root0 = find(joint.body0)
        root1 = find(joint.body1)
        if root0 is root1:
            return True
        parent[root1] = root0
    return False


def _body_name(value: RigidBodyRef, *, field_name: str) -> str:
    return _as_name(value if isinstance(value, str) else value.name, field_name=field_name)


def _as_vec3(value: Sequence[float], *, field_name: str) -> Vec3:
    if isinstance(value, (str, bytes)):
        raise ValidationError(f"{field_name} must have 3 numeric values")
    try:
        values = tuple(float(component) for component in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{field_name} must have 3 numeric values") from exc
    if len(values) != 3 or any(not math.isfinite(component) for component in values):
        raise ValidationError(f"{field_name} must have 3 finite numeric values")
    return cast(Vec3, values)


def _finite(value: object, *, field_name: str) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{field_name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValidationError(f"{field_name} must be finite")
    return result


def _positive_tolerance(value: object, field_name: str) -> float:
    result = _finite(value, field_name=field_name)
    if result <= 0.0:
        raise ValidationError(f"{field_name} must be positive")
    return result


def _as_matrix4(value: object, *, field_name: str) -> Mat4:
    if isinstance(value, JointFrame):
        return _frame_matrix(value)
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{field_name} must be a JointFrame or 4x4 matrix") from exc
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValidationError(f"{field_name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-9):
        raise ValidationError(f"{field_name} must be a rigid homogeneous transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.identity(3), rtol=0.0, atol=1e-7):
        raise ValidationError(f"{field_name} rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1e-7):
        raise ValidationError(f"{field_name} rotation must be right-handed")
    return matrix.copy()


def _matrix_tuple(matrix: Mat4) -> Matrix4:
    return cast(Matrix4, tuple(tuple(float(value) for value in row) for row in matrix))
