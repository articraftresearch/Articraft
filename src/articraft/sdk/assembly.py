from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TypeAlias, cast

import numpy as np
from scipy.optimize import least_squares  # pyright: ignore[reportMissingTypeStubs]
from scipy.spatial.transform import Rotation  # pyright: ignore[reportMissingTypeStubs]

from articraft.sdk._values import _as_identifier, _as_name, _finite, _positive
from articraft.sdk.bodies import RigidBody, RigidBodyRef
from articraft.sdk.errors import LoopClosureError, ValidationError
from articraft.sdk.frames import WORLD, BodyFrame, JointFrame, _matrix_rpy, _WorldEndpoint
from articraft.sdk.mass import MassProperties
from articraft.sdk.physics import BodyState, PhysicsScene

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


JointEndpoint: TypeAlias = RigidBody | _WorldEndpoint


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
        object.__setattr__(self, "name", _as_identifier(self.name, field_name="joint name"))
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

    def get_dof(self, dof: JointAxis | str) -> JointDOF:
        """Return one authored degree of freedom by axis or qualified id."""
        value = str(dof).strip()
        if "." in value:
            joint_name, value = value.rsplit(".", 1)
            if joint_name != self.name:
                raise ValidationError(
                    f"DOF {dof!r} belongs to joint {joint_name!r}, not {self.name!r}"
                )
        try:
            axis = JointAxis(value)
        except ValueError as exc:
            raise ValidationError(f"unknown joint axis: {value!r}") from exc
        for authored in self.dofs:
            if authored.axis == axis:
                return authored
        available = [cast(JointAxis, authored.axis).value for authored in self.dofs]
        raise ValidationError(
            f"joint {self.name!r} has no {axis.value!r} DOF; available axes are {available!r}"
        )


JointRef: TypeAlias = str | Joint
ArticulationRootRef: TypeAlias = RigidBodyRef | Joint


@dataclass(frozen=True, slots=True, eq=False)
class Articulation:
    """A reduced-coordinate spanning tree within the physical joint graph."""

    name: str
    root: RigidBody | Joint
    joints: tuple[Joint, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _as_identifier(self.name, field_name="articulation name"))
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

    def frame_in(
        self,
        source: BodyFrame,
        body: RigidBody | _WorldEndpoint = WORLD,
    ) -> BodyFrame:
        """Express one body-bound frame in another body's coordinates.

        A state stores poses by body *name*, so bodies here resolve by name
        alone. ``RigidBodyAssembly.frame_in`` additionally checks identity;
        prefer it while authoring.
        """

        source_world = self._body_matrix(source.body) @ _frame_matrix(source.frame)
        local = np.linalg.inv(self._body_matrix(body)) @ source_world
        return BodyFrame(body, _matrix_frame(local))

    def _body_matrix(self, body: RigidBody | _WorldEndpoint) -> Mat4:
        if body is WORLD:
            return np.identity(4, dtype=np.float64)
        return self.matrix(cast(RigidBody, body))


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
        linear_tolerance = _positive(linear_tolerance, "linear tolerance")
        angular_tolerance = _positive(angular_tolerance, "angular tolerance")
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
        """Pose the articulation tree and solve unspecified coordinates that close loops."""

        positions, transforms = self._kinematics(dof_positions)
        return self.validate_state(PhysicsState(transforms, dof_positions=positions))

    def _kinematics(
        self,
        dof_positions: Mapping[str, float] | None = None,
        *,
        relax_limits: bool = False,
        loop_start: Mapping[str, float] | None = None,
    ) -> tuple[dict[str, float], dict[str, Mat4]]:
        """Solve the tree and its loops, before the state is validated.

        With ``relax_limits`` the solver ignores the limits on the coordinates
        it derives, so a caller can ask where the mechanism reaches instead of
        where the authored limits let it reach. Those coordinates can then land
        outside their own limits, which is why this stays private:
        ``forward_kinematics`` validates everything it hands out. ``loop_start``
        seeds the loop solver with a neighbouring solution, so a sweep can
        continue along one assembly branch instead of solving every pose from
        rest.
        """

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

        tree = tuple(item.joint for item in self.joints if not item.exclude_from_articulation)
        closures = tuple(item.joint for item in self.joints if item.exclude_from_articulation)
        if self.has_closed_loops and _edge_graph_has_cycle(tree):
            raise ValidationError(
                "closed-loop forward kinematics requires an explicit articulation tree; "
                "provide a complete PhysicsState for a maximal-coordinate assembly"
            )
        closure_dofs = {joint.dof_id(dof) for joint in closures for dof in joint.dofs}
        supplied_closure_dofs = sorted(set(positions) & closure_dofs)
        if supplied_closure_dofs:
            raise ValidationError(
                "closure joint positions are derived from body poses and cannot be supplied: "
                f"{supplied_closure_dofs!r}"
            )
        if closures:
            positions = _solve_closed_loops(
                self.rigid_bodies,
                tree,
                closures,
                self.articulations,
                positions,
                relax_limits=relax_limits,
                start=loop_start,
            )
        try:
            transforms = _propagate_transforms(
                self.rigid_bodies,
                tree,
                self.articulations,
                positions,
            )
        except ValidationError as exc:
            raise ValidationError(
                "forward kinematics requires one spanning articulation tree; "
                "provide a complete PhysicsState for this assembly"
            ) from exc
        return positions, transforms


@dataclass
class RigidBodyAssembly:
    """A connected graph of USD rigid bodies, joints, and solver articulations."""

    name: str
    scene: PhysicsScene = field(default=PhysicsScene(), kw_only=True)
    rigid_bodies: list[RigidBody] = field(default_factory=list, init=False)
    joints: list[Joint] = field(default_factory=list, init=False)
    articulations: list[Articulation] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.name = _as_identifier(self.name, field_name="assembly name")
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
        endpoint0: BodyFrame,
        endpoint1: BodyFrame,
        *,
        dofs: Iterable[JointDOF] = (),
    ) -> Joint:
        if not isinstance(endpoint0, BodyFrame) or not isinstance(endpoint1, BodyFrame):
            raise ValidationError("joint endpoints must be body.at(...) or WORLD.at(...) frames")
        joint = Joint(
            name=name,
            body0=self._endpoint(endpoint0.body, field_name="endpoint0"),
            frame0=endpoint0.frame,
            body1=self._endpoint(endpoint1.body, field_name="endpoint1"),
            frame1=endpoint1.frame,
            dofs=tuple(dofs),
        )
        if any(existing.name == joint.name for existing in self.joints):
            raise ValidationError(f"duplicate joint name: {joint.name!r}")
        self.joints.append(joint)
        return joint

    def frame_in(
        self,
        source: BodyFrame,
        body: RigidBody | _WorldEndpoint = WORLD,
    ) -> BodyFrame:
        """Express a frame in another body at the assembly's reference state."""

        if not isinstance(source, BodyFrame):
            raise ValidationError("source must be a body.at(...) or WORLD.at(...) frame")
        if source.body is not WORLD:
            source_body = self.get_rigid_body(cast(RigidBody, source.body))
            if source_body is not source.body:
                raise ValidationError("source frame belongs to another assembly")
        if body is not WORLD:
            resolved_body = self.get_rigid_body(cast(RigidBody, body))
            if resolved_body is not body:
                raise ValidationError("target body belongs to another assembly")
        try:
            reference = self.resolve().reference_state
        except ValidationError as exc:
            raise ValidationError(
                "frame_in reads the assembly's reference state, so the bodies and "
                "joints that place both frames must be authored first; "
                f"resolving failed: {exc}"
            ) from exc
        return reference.frame_in(source, body)

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
        self.name = _as_identifier(self.name, field_name="assembly name")
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
        has_closed_loops = _graph_has_cycle(tuple(self.joints))
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

    def _endpoint(self, endpoint: JointEndpoint, *, field_name: str) -> JointEndpoint:
        if endpoint is WORLD:
            return WORLD
        try:
            resolved = self.get_rigid_body(cast(RigidBodyRef, endpoint))
        except ValidationError as exc:
            raise ValidationError(f"unknown {field_name} rigid body: {endpoint!r}") from exc
        if resolved is not endpoint:
            raise ValidationError(f"{field_name} frame belongs to another assembly")
        return resolved

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
            if len(assembly.articulations) != 1 or _graph_has_cycle(tuple(assembly.joints)):
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


_LOOP_STEPS = 5
# The acceptance gate, equal to validate_state's per-axis tolerance so a
# solve is accepted exactly when validation would pass it. The least-squares
# solve itself polishes well past this so accepted solves clear the gate
# with margin rather than landing on it.
_LOOP_TOLERANCE = 1e-6


def _solve_closed_loops(
    bodies: tuple[RigidBody, ...],
    tree: tuple[Joint, ...],
    closures: tuple[Joint, ...],
    articulations: tuple[ResolvedArticulation, ...],
    positions: dict[str, float],
    *,
    relax_limits: bool = False,
    start: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Solve unspecified tree DOFs so every excluded constraint stays closed.

    The articulation tree supplies coordinates; excluded joints supply residuals.
    Walking from the zero configuration to the requested pose keeps linkages on
    the assembly branch they were authored in instead of snapping through a
    singular pose.

    ``start`` maps dof ids to a warm start. Coordinates it names begin there,
    projected into their bounds, missing ones begin at zero, and the homotopy
    from rest is skipped: a caller walking a sweep hands in the neighbouring
    solution and stays on the branch it was already on.
    """

    active_closures: list[Joint] = []
    candidates: list[tuple[Joint, JointDOF]] = []
    seen: set[str] = set()
    for closure in closures:
        path = _joint_path(tree, closure.body0, closure.body1)
        if path is None:
            raise ValidationError(
                f"closure joint {closure.name!r} is not spanned by one articulation tree; "
                "provide a complete PhysicsState for this assembly"
            )
        active_closures.append(closure)
        for joint in path:
            for dof in joint.dofs:
                dof_id = joint.dof_id(dof)
                if dof_id not in positions and dof_id not in seen:
                    seen.add(dof_id)
                    candidates.append((joint, dof))

    locked = {
        closure: tuple(axis for axis in JointAxis if axis not in {dof.axis for dof in closure.dofs})
        for closure in active_closures
    }
    if not any(locked.values()):
        return positions

    # Relaxing takes the walls away from the coordinates the solver derives,
    # so the ring reports where the mechanism reaches instead of where the
    # authored limits allow it to reach. Supplied coordinates keep their own
    # limits either way -- the caller is driving those.
    limits = tuple(None if relax_limits else dof.limits for _, dof in candidates)
    periodic = tuple(
        _is_periodic(dof, bound) for (_, dof), bound in zip(candidates, limits, strict=True)
    )

    def assemble(scale: float, values: np.ndarray) -> dict[str, float]:
        result = {name: value * scale for name, value in positions.items()}
        for (joint, dof), value in zip(candidates, values, strict=True):
            result[joint.dof_id(dof)] = float(value)
        return result

    def project(values: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                _project_value(float(value), bound, wraps)
                for value, bound, wraps in zip(values, limits, periodic, strict=True)
            ],
            dtype=np.float64,
        )

    def residual(scale: float, values: np.ndarray) -> np.ndarray:
        placed = _propagate_transforms(
            bodies,
            tree,
            articulations,
            assemble(scale, values),
        )
        rows: list[float] = []
        for closure in active_closures:
            joint_values = _joint_values(closure, placed)
            rows.extend(joint_values[axis] for axis in locked[closure])
        return np.asarray(rows, dtype=np.float64)

    # Limits become solver bounds, except on a full-circle hinge: its
    # coordinate is periodic, so it solves unbounded and wraps afterwards
    # rather than stopping at a seam that is not a physical wall.
    lower_bounds = np.asarray(
        [
            -np.inf if bound is None or wraps else bound[0]
            for bound, wraps in zip(limits, periodic, strict=True)
        ],
        dtype=np.float64,
    )
    upper_bounds = np.asarray(
        [
            np.inf if bound is None or wraps else bound[1]
            for bound, wraps in zip(limits, periodic, strict=True)
        ],
        dtype=np.float64,
    )

    def solve_toward(scale: float, start: np.ndarray) -> np.ndarray:
        # A bounded trust-region solve keeps every iterate inside the limits
        # and differentiates into the feasible interval at a bound -- the
        # properties the hand-rolled Gauss-Newton loop had to be taught.
        solution = least_squares(
            lambda unknowns: residual(scale, unknowns),
            start,
            jac="2-point",
            bounds=(lower_bounds, upper_bounds),
            method="dogbox",
            ftol=1e-14,
            xtol=1e-14,
            gtol=1e-14,
        )
        return project(solution.x)

    if start is not None and candidates:
        # A warm start is already on the branch the caller is walking, so the
        # homotopy from rest is not needed: one full-scale solve tracks it.
        values = project(
            np.asarray(
                [start.get(joint.dof_id(dof), 0.0) for joint, dof in candidates],
                dtype=np.float64,
            )
        )
        values = solve_toward(1.0, values)
    else:
        values = np.zeros(len(candidates), dtype=np.float64)
        # With no unknowns left -- every ring coordinate supplied by the caller --
        # there is nothing to solve, but the closure residual still decides
        # whether the supplied pose keeps the loop assembled.
        for step in range(1, (_LOOP_STEPS if candidates else 0) + 1):
            values = solve_toward(step / _LOOP_STEPS, values)

    # One gate, equal to validate_state's per-axis tolerance: anything
    # accepted here passes validation, anything rejected names the loop.
    worst = float(np.max(np.abs(residual(1.0, values)), initial=0.0))
    if worst > _LOOP_TOLERANCE:
        names = ", ".join(repr(closure.name) for closure in active_closures)
        pinned = sorted(
            joint.dof_id(dof)
            for ((joint, dof), value, bound, wraps) in zip(
                candidates, values, limits, periodic, strict=True
            )
            if bound is not None and not wraps and (value <= bound[0] or value >= bound[1])
        )
        if pinned:
            raise LoopClosureError(
                f"pose leaves loop closure {names} open (constraint residual {worst:.3g}) "
                f"with solved joint positions pinned at their limits: {pinned!r}; "
                "widen those limits if the mechanism should reach this pose"
            )
        if not candidates:
            raise LoopClosureError(
                f"pose leaves loop closure {names} open (constraint residual {worst:.3g}); "
                "every ring coordinate was supplied -- adjust them so the loop closes, "
                "or leave one unspecified for the solver to derive"
            )
        raise LoopClosureError(
            f"pose leaves loop closure {names} open (constraint residual {worst:.3g}); "
            "the mechanism cannot reach this pose"
        )
    return assemble(1.0, values)


def _is_periodic(dof: JointDOF, limits: tuple[float, float] | None) -> bool:
    """True when the coordinate turns full circle, so its limits are not a wall.

    A hinge authored -pi..pi says "unconstrained", not "must reach every angle":
    there is no seam at the ends, and no range claim for a mechanism to
    contradict.
    """

    if limits is None or not cast(JointAxis, dof.axis).is_rotational:
        return False
    return limits[1] - limits[0] >= 2.0 * math.pi - 1e-9


def _joint_path(
    joints: tuple[Joint, ...],
    start: JointEndpoint,
    end: JointEndpoint,
) -> tuple[Joint, ...] | None:
    adjacency: dict[JointEndpoint, list[tuple[JointEndpoint, Joint]]] = {}
    for joint in joints:
        adjacency.setdefault(joint.body0, []).append((joint.body1, joint))
        adjacency.setdefault(joint.body1, []).append((joint.body0, joint))
    previous: dict[JointEndpoint, tuple[JointEndpoint, Joint] | None] = {start: None}
    pending = [start]
    while pending and end not in previous:
        node = pending.pop(0)
        for neighbour, joint in adjacency.get(node, []):
            if neighbour in previous:
                continue
            previous[neighbour] = (node, joint)
            pending.append(neighbour)
    if end not in previous:
        return None
    path: list[Joint] = []
    node = end
    while (step := previous[node]) is not None:
        node, joint = step
        path.append(joint)
    path.reverse()
    return tuple(path)


def _clamp(value: float, limits: tuple[float, float] | None) -> float:
    if limits is None:
        return value
    return min(max(value, limits[0]), limits[1])


def _project_value(value: float, limits: tuple[float, float] | None, periodic: bool) -> float:
    """Return the nearest in-range coordinate: wrapped if periodic, clamped if not.

    A full-circle hinge has no wall at +-pi: the coordinate is periodic, and
    clamping it would strand a solution that continues on the other side of
    the seam. The period is the physical turn, 2*pi -- limits merely wide
    enough to contain one (say -3.15..3.15) span slightly more, and wrapping
    by that span would land on a genuinely different angle.
    """

    if limits is None or not periodic:
        return _clamp(value, limits)
    turn = 2.0 * math.pi
    wrapped = limits[0] + math.fmod(value - limits[0], turn)
    return wrapped + turn if wrapped < limits[0] else wrapped


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
    return {
        JointAxis.TRANS_X: float(relative[0, 3]),
        JointAxis.TRANS_Y: float(relative[1, 3]),
        JointAxis.TRANS_Z: float(relative[2, 3]),
        **_rotation_values(joint, relative),
    }


def _rotation_values(joint: Joint, relative: Mat4) -> dict[JointAxis, float]:
    """Per-axis rotation readings that stay honest over the full circle.

    An Euler decomposition confines its middle angle to [-pi/2, pi/2]; past
    that, a pure Y rotation of 1.7 rad reads as (-pi, 1.44, -pi) and a healthy
    hinge appears to violate its locked axes. Joints with at most one free
    rotational axis -- fixed, prismatic, revolute, cylindrical -- are measured
    against that axis directly instead, which has no branch to fall off. Only
    joints with two or more free rotational axes still read through the Euler
    decomposition, whose angles are what their motion model composes.
    """

    free = [
        cast(JointAxis, dof.axis) for dof in joint.dofs if cast(JointAxis, dof.axis).is_rotational
    ]
    rotation = np.asarray(relative[:3, :3], dtype=np.float64)
    if len(free) > 1:
        angles = _matrix_rpy(rotation)
        return {
            JointAxis.ROT_X: float(angles[0]),
            JointAxis.ROT_Y: float(angles[1]),
            JointAxis.ROT_Z: float(angles[2]),
        }
    values = {JointAxis.ROT_X: 0.0, JointAxis.ROT_Y: 0.0, JointAxis.ROT_Z: 0.0}
    if free:
        direction = np.zeros(3, dtype=np.float64)
        direction[free[0].component] = 1.0
        # The best-fit angle about a known axis: for R = Rot(a, angle),
        # vee((R - R^T)/2) = sin(angle) * a and tr(R) - a.R.a = 2 cos(angle).
        skew = (rotation - rotation.T) * 0.5
        sine = float(direction @ (skew[2, 1], skew[0, 2], skew[1, 0]))
        cosine = float(np.trace(rotation) - direction @ rotation @ direction) * 0.5
        angle = math.atan2(sine, cosine)
        values[free[0]] = angle
        rotation = Rotation.from_rotvec(-angle * direction).as_matrix() @ rotation
    leftover = _rotation_vector(rotation)
    for axis in (JointAxis.ROT_X, JointAxis.ROT_Y, JointAxis.ROT_Z):
        if axis not in free:
            values[axis] = float(leftover[axis.component])
    return values


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """The axis-angle vector of a rotation matrix; zero for the identity."""

    return Rotation.from_matrix(rotation).as_rotvec()


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
    matrix[:3, :3] = Rotation.from_euler("xyz", rotation).as_matrix()
    motion = np.identity(4, dtype=np.float64)
    motion[:3, 3] = translation
    return motion @ matrix


def _frame_matrix(frame: JointFrame) -> Mat4:
    matrix = np.identity(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_euler("xyz", frame.rpy).as_matrix()
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


def _graph_has_cycle(joints: tuple[Joint, ...]) -> bool:
    """A cycle is a property of the edges; the body list never mattered."""

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


def _matrix_frame(matrix: Mat4) -> JointFrame:
    rpy = _matrix_rpy(np.asarray(matrix)[:3, :3])
    return JointFrame(
        xyz=(float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3])),
        rpy=(float(rpy[0]), float(rpy[1]), float(rpy[2])),
    )
