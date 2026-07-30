"""The exported stage has to survive a real physics engine, not just validation."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from mini_articraft.sdk import (
    ArticulatedObject,
    ArticulationType,
    BoxGeometry,
    Material,
    MotionLimits,
    Origin,
)
from mini_articraft.sdk.export import export_object
from mini_articraft.simulate import simulate_usdz, write_mjcf

mujoco = pytest.importorskip("mujoco", reason="simulation needs the sim dependency group")

LOWER, UPPER = 0.0, 1.5  # radians, as authored


def _hinged_box() -> ArticulatedObject:
    model = ArticulatedObject("crate")
    base = model.part("base")
    base.add(
        BoxGeometry((0.30, 0.20, 0.10)).translate(0.0, 0.0, 0.05),
        name="body",
        material=Material.HARDWOOD,
    )
    lid = model.part("lid")
    lid.add(
        BoxGeometry((0.30, 0.20, 0.01)).translate(0.0, 0.10, 0.0),
        name="panel",
        material=Material.STEEL,
    )
    model.articulation(
        "lid_hinge",
        ArticulationType.REVOLUTE,
        base,
        lid,
        origin=Origin(xyz=(0.0, -0.10, 0.10)),
        axis=(1.0, 0.0, 0.0),
        motion_limits=MotionLimits(effort=5.0, velocity=2.0, lower=LOWER, upper=UPPER),
    )
    return model


def _export(model: ArticulatedObject, tmp_path: Path) -> Path:
    return Path(export_object(model, tmp_path).usdz)


def test_an_exported_object_stands_up_on_a_floor(tmp_path: Path) -> None:
    result = simulate_usdz(_export(_hinged_box(), tmp_path), tmp_path / "sim")

    assert result.stood_up, result.summary()
    assert not result.fell_through_floor
    assert result.parts_stayed_together
    assert not result.diverged


def test_the_simulated_mass_is_the_mass_we_exported(tmp_path: Path) -> None:
    result = simulate_usdz(_export(_hinged_box(), tmp_path), tmp_path / "sim")

    wood = 0.30 * 0.20 * 0.10 * Material.HARDWOOD.density
    steel = 0.30 * 0.20 * 0.01 * Material.STEEL.density
    assert result.total_mass == pytest.approx(wood + steel, rel=1e-3)


def test_revolute_limits_convert_from_degrees_to_radians(tmp_path: Path) -> None:
    """UsdPhysics states revolute limits in degrees; MJCF is written in radians.

    Reading them straight through turns a 1.5 rad hinge into an 86 rad one, which
    silently removes the limit rather than failing.
    """
    write_mjcf(_export(_hinged_box(), tmp_path), tmp_path / "sim")
    joint = ET.parse(tmp_path / "sim" / "model.xml").getroot().find(".//joint")

    assert joint is not None
    lower, upper = (float(value) for value in str(joint.get("range")).split())
    assert lower == pytest.approx(LOWER, abs=1e-4)
    assert upper == pytest.approx(UPPER, abs=1e-4)
    assert upper < math.pi  # not degrees


def test_a_part_with_no_joint_gets_a_free_body(tmp_path: Path) -> None:
    model = ArticulatedObject("lone")
    model.part("body").add(BoxGeometry((0.1, 0.1, 0.1)), name="cube", material=Material.STEEL)

    write_mjcf(_export(model, tmp_path), tmp_path / "sim")
    root = ET.parse(tmp_path / "sim" / "model.xml").getroot()

    assert root.find(".//freejoint") is not None


def test_contact_friction_reaches_the_geom(tmp_path: Path) -> None:
    model = ArticulatedObject("gripped")
    part = model.part("body")
    part.add(BoxGeometry((0.1, 0.1, 0.1)), name="pad", material=Material.RUBBER)

    write_mjcf(_export(model, tmp_path), tmp_path / "sim")
    geom = ET.parse(tmp_path / "sim" / "model.xml").getroot().find(".//body/geom")

    assert geom is not None
    friction = float(str(geom.get("friction")).split()[0])
    assert friction == pytest.approx(Material.RUBBER.dynamic_friction or 0.0, abs=1e-3)


def test_material_friction_reaches_the_contact_not_the_floors(tmp_path: Path) -> None:
    """MuJoCo combines friction with max, so a floor with any of its own masks ours."""
    import mujoco

    model = ArticulatedObject("block")
    model.part("body").add(BoxGeometry((0.1, 0.1, 0.1)), name="cube", material=Material.STEEL)

    path = write_mjcf(_export(model, tmp_path), tmp_path / "sim")
    compiled = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(compiled)
    for _ in range(200):
        mujoco.mj_step(compiled, data)

    assert data.ncon
    assert data.contact.friction[0][0] == pytest.approx(Material.STEEL.dynamic_friction, abs=1e-3)


def test_tilting_measures_the_friction_that_was_authored(tmp_path: Path) -> None:
    """A grippier material has to hold to a steeper angle than a slippery one."""
    angles = {}
    for material in (Material.RUBBER, Material.STEEL):
        model = ArticulatedObject("block")
        model.part("body").add(BoxGeometry((0.12, 0.12, 0.06)), name="block", material=material)
        result = simulate_usdz(
            _export(model, tmp_path / material.name),
            tmp_path / material.name / "sim",
            seconds=6.0,
            scenario="tilt",
        )
        assert result.slip_angle is not None, result.summary()
        angles[material.name] = result.slip_angle

    assert angles["rubber"] > angles["steel"]


def test_a_tilt_run_stops_once_it_slips(tmp_path: Path) -> None:
    """Tilting past the slip angle topples the object and tells us nothing more."""
    model = ArticulatedObject("block")
    model.part("body").add(BoxGeometry((0.12, 0.12, 0.06)), name="block", material=Material.STEEL)

    result = simulate_usdz(
        _export(model, tmp_path), tmp_path / "sim", seconds=20.0, scenario="tilt"
    )

    assert result.slip_angle is not None
    assert result.slip_angle < 50.0
    # It is still whole and not flying: the run ended at the answer.
    assert result.residual_velocity < 5.0
    assert result.parts_stayed_together


def test_an_unknown_scenario_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown scenario"):
        simulate_usdz(_export(_hinged_box(), tmp_path), tmp_path / "sim", scenario="wobble")


def test_a_rotated_rest_pose_survives_the_translation(tmp_path: Path) -> None:
    """A joint with an rpy rotates its child; dropping that misplaces the part."""
    model = ArticulatedObject("tilted")
    base = model.part("base")
    base.add(BoxGeometry((0.2, 0.2, 0.05)), name="plate", material=Material.STEEL)
    arm = model.part("arm")
    arm.add(BoxGeometry((0.02, 0.02, 0.20)), name="post", material=Material.STEEL)
    model.articulation(
        "mount",
        ArticulationType.FIXED,
        base,
        arm,
        origin=Origin(xyz=(0.0, 0.0, 0.05), rpy=(0.0, math.pi / 4, 0.0)),
    )

    write_mjcf(_export(model, tmp_path), tmp_path / "sim")
    arm_body = ET.parse(tmp_path / "sim" / "model.xml").getroot().find(".//body/body")

    assert arm_body is not None
    quaternion = arm_body.get("quat")
    assert quaternion is not None, "a rotated part must carry its orientation"
    w, _, y, _ = (float(value) for value in quaternion.split())
    # 45 degrees about Y: w = cos(22.5 deg), y = sin(22.5 deg).
    assert w == pytest.approx(math.cos(math.pi / 8), abs=1e-4)
    assert abs(y) == pytest.approx(math.sin(math.pi / 8), abs=1e-4)


def test_releasing_a_joint_produces_motion_worth_watching(tmp_path: Path) -> None:
    """Released at a limit a lid can rest against the stop; from mid-travel it swings."""
    result = simulate_usdz(
        _export(_hinged_box(), tmp_path), tmp_path / "sim", seconds=2.0, scenario="release"
    )

    assert result.trajectory is not None
    angles = [frame["joints"][0] for frame in result.trajectory.frames]
    assert max(angles) - min(angles) > 0.5  # radians
    assert result.peak_joint_speed is not None and result.peak_joint_speed > 1.0
