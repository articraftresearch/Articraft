"""Driven joints keep a linkage assembled when only its driving joint moves."""

from __future__ import annotations

import math

import numpy as np
import pytest
from build123d import Box, Cylinder, Pos, Rot

from articraft.sdk import ArticulatedObject, ArticulationType, MotionLimits, Origin, Part
from articraft.sdk._collision import MeshCollisionKernel
from articraft.sdk.errors import ValidationError
from articraft.sdk.joints import AimAt, SpanTo

# The ram's far eye rides on the arm, 0.6 m out from the arm's own pivot.
ARM_EYE = (0.6, 0.0, 0.0)
BARREL_PIVOT = (0.35, 0.0, -0.25)
REST_LENGTH = 0.30


def _ram_model(*, driven: bool) -> ArticulatedObject:
    """A base, an arm on a hinge, and a two piece ram reaching between them."""

    model = ArticulatedObject("ram")
    base = model.part("base")
    base.add(Box(0.3, 0.2, 0.1), name="block")
    arm = model.part("arm")
    arm.add(Pos(X=0.35) * Box(0.7, 0.08, 0.08), name="beam")
    barrel = model.part("barrel")
    barrel.add(Rot(Y=90) * Cylinder(0.04, REST_LENGTH), name="tube")
    rod = model.part("rod")
    rod.add(Rot(Y=90) * Cylinder(0.02, 0.3), name="shaft")

    model.articulation(
        "arm_hinge",
        ArticulationType.REVOLUTE,
        base,
        arm,
        origin=Origin(xyz=(0.0, 0.0, 0.0)),
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-0.6, upper=0.6),
    )
    model.articulation(
        "barrel_swivel",
        ArticulationType.REVOLUTE,
        base,
        barrel,
        origin=Origin(xyz=BARREL_PIVOT),
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-1.6, upper=1.6),
        drive=AimAt("arm", ARM_EYE) if driven else None,
    )
    model.articulation(
        "ram_extension",
        ArticulationType.PRISMATIC,
        barrel,
        rod,
        origin=Origin(xyz=(0.0, 0.0, 0.0)),
        axis=(1.0, 0.0, 0.0),
        motion_limits=MotionLimits(lower=-0.4, upper=0.4),
        drive=SpanTo("arm", ARM_EYE, rest_length=REST_LENGTH) if driven else None,
    )
    return model


def _eye_gap(model: ArticulatedObject, angle: float) -> float:
    """Distance between the rod's tip and the eye it is supposed to hold."""

    kernel = MeshCollisionKernel(model, mesh_tolerance=0.001)
    transforms = kernel.world_transforms({"arm_hinge": angle})
    rod_tip = (
        transforms["rod"][:3, :3] @ np.array([REST_LENGTH, 0.0, 0.0]) + transforms["rod"][:3, 3]
    )
    eye = transforms["arm"][:3, :3] @ np.array(ARM_EYE) + transforms["arm"][:3, 3]
    return float(np.linalg.norm(rod_tip - eye))


def test_driven_ram_stays_on_its_eye_through_the_swing() -> None:
    model = _ram_model(driven=True)
    gaps = [_eye_gap(model, angle) for angle in np.linspace(-0.6, 0.6, 13)]
    assert max(gaps) < 1e-6, f"ram came off its eye: worst gap {max(gaps) * 1000:.3f} mm"


def test_undriven_ram_comes_apart() -> None:
    """The failure this feature exists to fix, pinned so it cannot come back."""

    model = _ram_model(driven=False)
    gaps = [_eye_gap(model, angle) for angle in np.linspace(-0.6, 0.6, 13)]
    assert max(gaps) > 0.1, "expected the undriven ram to detach"


def test_drive_follows_the_driving_joint_not_the_pose_dict() -> None:
    """A driven joint ignores any value handed to it directly."""

    model = _ram_model(driven=True)
    kernel = MeshCollisionKernel(model, mesh_tolerance=0.001)
    honest = kernel.world_transforms({"arm_hinge": 0.4})
    meddled = kernel.world_transforms({"arm_hinge": 0.4, "ram_extension": 5.0})
    assert np.allclose(honest["rod"], meddled["rod"])


def test_drive_type_must_match_the_joint() -> None:
    model = ArticulatedObject("mismatch")
    base = model.part("base")
    base.add(Box(0.1, 0.1, 0.1), name="block")
    arm = model.part("arm")
    arm.add(Box(0.1, 0.1, 0.1), name="block")
    other = model.part("other")
    other.add(Box(0.1, 0.1, 0.1), name="block")
    with pytest.raises(ValidationError, match="must be SpanTo"):
        model.articulation(
            "slide",
            ArticulationType.PRISMATIC,
            base,
            arm,
            axis=(1.0, 0.0, 0.0),
            motion_limits=MotionLimits(lower=0.0, upper=0.1),
            drive=AimAt("other"),
        )


def test_drive_cannot_read_a_part_the_joint_moves() -> None:
    model = ArticulatedObject("circular")
    base = model.part("base")
    base.add(Box(0.1, 0.1, 0.1), name="block")
    arm = model.part("arm")
    arm.add(Box(0.1, 0.1, 0.1), name="block")
    with pytest.raises(ValidationError, match="parent or child"):
        model.articulation(
            "slide",
            ArticulationType.PRISMATIC,
            base,
            arm,
            axis=(1.0, 0.0, 0.0),
            motion_limits=MotionLimits(lower=0.0, upper=0.1),
            drive=SpanTo("arm"),
        )


def test_aim_reference_must_be_non_zero() -> None:
    with pytest.raises(ValidationError, match="aim reference"):
        AimAt("arm", (0.0, 0.0, 0.0), reference=(0.0, 0.0, 0.0))


def test_angle_matches_the_geometry_by_hand() -> None:
    """Check the solved swivel against the angle worked out from the triangle."""

    model = _ram_model(driven=True)
    kernel = MeshCollisionKernel(model, mesh_tolerance=0.001)
    angle = 0.35
    resolved = kernel._resolve_drives({"arm_hinge": angle})
    eye = np.array(
        [
            ARM_EYE[0] * math.cos(angle) + ARM_EYE[2] * math.sin(angle),
            0.0,
            -ARM_EYE[0] * math.sin(angle) + ARM_EYE[2] * math.cos(angle),
        ]
    )
    offset = eye - np.array(BARREL_PIVOT)
    assert resolved["ram_extension"] == pytest.approx(
        float(np.linalg.norm(offset)) - REST_LENGTH, abs=1e-9
    )
    assert resolved["barrel_swivel"] == pytest.approx(math.atan2(-offset[2], offset[0]), abs=1e-9)


def _slider(model: ArticulatedObject, name: str) -> Part:
    part = model.part(name)
    part.add(Box(0.05, 0.05, 0.05), name="block")
    return part


def test_drive_cannot_chain_through_another_driven_joint() -> None:
    """One-pass resolution would hand the second drive a stale anchor."""

    model = ArticulatedObject("chained")
    base = _slider(model, "base")
    arm = _slider(model, "arm")
    carriage = _slider(model, "carriage")
    shuttle = _slider(model, "shuttle")
    model.articulation(
        "arm_hinge",
        ArticulationType.REVOLUTE,
        base,
        arm,
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-0.5, upper=0.5),
    )
    model.articulation(
        "carriage_slide",
        ArticulationType.PRISMATIC,
        base,
        carriage,
        axis=(1.0, 0.0, 0.0),
        motion_limits=MotionLimits(lower=-1.0, upper=1.0),
        drive=SpanTo("arm", (0.3, 0.0, 0.0)),
    )
    model.articulation(
        "shuttle_slide",
        ArticulationType.PRISMATIC,
        base,
        shuttle,
        axis=(1.0, 0.0, 0.0),
        motion_limits=MotionLimits(lower=-1.0, upper=1.0),
        drive=SpanTo("carriage", (0.0, 0.0, 0.0)),
    )
    with pytest.raises(ValidationError, match="drives resolve in one pass"):
        model.validate()


def test_drive_cannot_read_its_own_subtree() -> None:
    """Reading a grandchild is circular: the joint moves its own anchor."""

    model = ArticulatedObject("circular_deep")
    base = _slider(model, "base")
    carriage = _slider(model, "carriage")
    tool = _slider(model, "tool")
    model.articulation(
        "carriage_slide",
        ArticulationType.PRISMATIC,
        base,
        carriage,
        axis=(1.0, 0.0, 0.0),
        motion_limits=MotionLimits(lower=-1.0, upper=1.0),
        drive=SpanTo("tool", (0.0, 0.0, 0.0)),
    )
    model.articulation(
        "tool_mount",
        ArticulationType.FIXED,
        carriage,
        tool,
    )
    with pytest.raises(ValidationError, match="own joint moves"):
        model.validate()
