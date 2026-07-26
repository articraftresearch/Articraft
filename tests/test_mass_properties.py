from __future__ import annotations

import math

import pytest
import trimesh
from pxr import Usd, UsdPhysics

from mini_articraft.sdk import (
    ArticulatedObject,
    BoxGeometry,
    CylinderGeometry,
    MassProperties,
    MaterialDensity,
    TestContext,
)
from mini_articraft.sdk._mass_solver import resolve_mass
from mini_articraft.sdk.errors import ValidationError
from mini_articraft.sdk.export import export_object


def _box(size=(0.2, 0.1, 0.05)) -> trimesh.Trimesh:
    return trimesh.creation.box(size)


def test_material_density_lookup() -> None:
    assert MaterialDensity.STEEL.density == 7850.0
    assert MassProperties(material=MaterialDensity.STEEL).resolved_density == 7850.0
    assert MassProperties(density=1234.0).resolved_density == 1234.0
    assert MassProperties(mass=2.0).resolved_density is None


def test_mass_properties_reject_conflicting_and_empty_values() -> None:
    with pytest.raises(ValidationError, match="only one of"):
        MassProperties(material=MaterialDensity.STEEL, density=1000.0)
    with pytest.raises(ValidationError, match="only one of"):
        MassProperties(mass=1.0, density=1000.0)
    with pytest.raises(ValidationError, match="need one of"):
        MassProperties()
    with pytest.raises(ValidationError, match="positive"):
        MassProperties(density=-5.0)
    with pytest.raises(ValidationError, match="positive"):
        MassProperties(mass=0.0)
    with pytest.raises(ValidationError, match="diagonal_inertia"):
        MassProperties(mass=1.0, diagonal_inertia=(1.0, 0.0, 1.0))
    with pytest.raises(ValidationError, match="principal_axes"):
        MassProperties(mass=1.0, principal_axes=(0.0, 0.0, 0.0, 0.0))


def test_computed_mass_and_inertia_match_the_analytic_box() -> None:
    width, depth, height = 0.2, 0.1, 0.05
    resolved = resolve_mass(
        MassProperties(material=MaterialDensity.STEEL),
        [_box((width, depth, height))],
        part_name="slab",
    )
    expected_mass = width * depth * height * 7850.0
    assert resolved.mass == pytest.approx(expected_mass)

    # A solid box's principal moments are the textbook values.
    expected = sorted(
        (
            expected_mass * (depth**2 + height**2) / 12.0,
            expected_mass * (width**2 + height**2) / 12.0,
            expected_mass * (width**2 + depth**2) / 12.0,
        )
    )
    assert sorted(resolved.diagonal_inertia) == pytest.approx(expected, rel=1e-6)
    assert resolved.center_of_mass == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_explicit_values_win_over_measurement() -> None:
    resolved = resolve_mass(
        MassProperties(
            mass=0.5,
            center_of_mass=(0.0, 0.0, 0.25),
            diagonal_inertia=(1.0, 2.0, 3.0),
            principal_axes=(0.0, 1.0, 0.0, 0.0),
        ),
        [_box()],
        part_name="slab",
    )
    assert resolved.mass == 0.5
    assert resolved.center_of_mass == (0.0, 0.0, 0.25)
    assert resolved.diagonal_inertia == (1.0, 2.0, 3.0)
    assert resolved.principal_axes == (0.0, 1.0, 0.0, 0.0)


def test_center_of_mass_follows_offset_geometry() -> None:
    offset = _box((0.1, 0.1, 0.1))
    offset.apply_translation((0.3, 0.0, 0.0))
    resolved = resolve_mass(
        MassProperties(material=MaterialDensity.ABS_PLASTIC), [offset], part_name="nub"
    )
    assert resolved.center_of_mass[0] == pytest.approx(0.3, abs=1e-6)


def test_overlapping_shapes_do_not_double_count_mass() -> None:
    first = _box((0.1, 0.1, 0.1))
    second = _box((0.1, 0.1, 0.1))
    second.apply_translation((0.05, 0.0, 0.0))  # half of it overlaps the first
    resolved = resolve_mass(MassProperties(density=1000.0), [first, second], part_name="pair")
    union_volume = 0.1 * 0.1 * 0.15
    assert resolved.mass == pytest.approx(union_volume * 1000.0, rel=1e-3)


def test_export_writes_mass_api_attributes(tmp_path) -> None:
    model = ArticulatedObject("massed")
    part = model.part("body", mass=MassProperties(material=MaterialDensity.HARDWOOD))
    part.add(BoxGeometry((0.2, 0.2, 0.05)), name="slab")

    result = export_object(model, tmp_path)
    stage = Usd.Stage.Open(str(result.usdz))
    mass_api = UsdPhysics.MassAPI  # pyright: ignore[reportAttributeAccessIssue]
    prims = [prim for prim in stage.Traverse() if prim.HasAPI(mass_api)]
    assert len(prims) == 1

    api = mass_api(prims[0])
    assert api.GetMassAttr().Get() == pytest.approx(0.2 * 0.2 * 0.05 * 700.0)
    assert api.GetDensityAttr().Get() == pytest.approx(700.0)
    assert api.GetCenterOfMassAttr().Get() is not None
    inertia = api.GetDiagonalInertiaAttr().Get()
    assert all(value > 0.0 for value in inertia)
    axes = api.GetPrincipalAxesAttr().Get()
    assert math.isclose(
        axes.GetReal() ** 2 + sum(value**2 for value in axes.GetImaginary()), 1.0, rel_tol=1e-5
    )


def test_export_omits_mass_api_when_the_part_has_no_properties(tmp_path) -> None:
    model = ArticulatedObject("plain")
    model.part("body").add(BoxGeometry((0.1, 0.1, 0.1)), name="cube")

    result = export_object(model, tmp_path)
    stage = Usd.Stage.Open(str(result.usdz))
    mass_api = UsdPhysics.MassAPI  # pyright: ignore[reportAttributeAccessIssue]
    assert not [prim for prim in stage.Traverse() if prim.HasAPI(mass_api)]


def test_missing_mass_check_reports_every_part_without_properties() -> None:
    model = ArticulatedObject("mixed")
    heavy = model.part("heavy", mass=MassProperties(material=MaterialDensity.STEEL))
    heavy.add(BoxGeometry((0.1, 0.1, 0.1)), name="block")
    light = model.part("light")
    light.add(CylinderGeometry(0.05, 0.02).translate(0.0, 0.0, 0.06), name="disc")

    ctx = TestContext(model)
    assert ctx.fail_if_parts_have_no_mass() is False
    report = ctx.report()
    details = report.failures[0].details
    assert "light" in details and "heavy" not in details
    assert "steel" in details  # the message lists the material library


def test_missing_mass_check_passes_when_every_part_has_properties() -> None:
    model = ArticulatedObject("complete")
    part = model.part("body", mass=MassProperties(density=900.0))
    part.add(BoxGeometry((0.1, 0.1, 0.1)), name="block")

    ctx = TestContext(model)
    assert ctx.fail_if_parts_have_no_mass() is True
    assert ctx.report().passed
