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
