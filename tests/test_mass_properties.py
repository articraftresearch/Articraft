from __future__ import annotations

import math
from pathlib import Path

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
    part = model.part("body", mass_properties=MassProperties(material=MaterialDensity.HARDWOOD))
    part.add(BoxGeometry((0.2, 0.2, 0.05)), name="slab")

    result = export_object(model, tmp_path)
    stage = Usd.Stage.Open(str(result.usdz))
    mass_api = UsdPhysics.MassAPI  # pyright: ignore[reportAttributeAccessIssue]
    prims = [prim for prim in stage.Traverse() if prim.HasAPI(mass_api)]
    assert len(prims) == 1

    api = mass_api(prims[0])
    assert api.GetMassAttr().Get() == pytest.approx(0.2 * 0.2 * 0.05 * 700.0)
    # Density is intentionally not authored: mass already resolves it, and the two
    # can disagree when a part's shapes are combined by concatenation.
    assert not api.GetDensityAttr().Get()
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
    heavy = model.part("heavy", mass_properties=MassProperties(material=MaterialDensity.STEEL))
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
    part = model.part("body", mass_properties=MassProperties(density=900.0))
    part.add(BoxGeometry((0.1, 0.1, 0.1)), name="block")

    ctx = TestContext(model)
    assert ctx.fail_if_parts_have_no_mass() is True
    assert ctx.report().passed


def test_authored_center_of_mass_shifts_the_measured_inertia() -> None:
    box = _box((0.1, 0.1, 0.1))
    measured = resolve_mass(MassProperties(density=1000.0), [box], part_name="p")
    shifted = resolve_mass(
        MassProperties(density=1000.0, center_of_mass=(0.0, 0.0, 0.5)), [box], part_name="p"
    )
    # Parallel axis adds m*d^2 about the two axes perpendicular to the offset.
    expected = measured.mass * 0.5**2
    moved = sorted(shifted.diagonal_inertia)[1:]
    assert all(
        value == pytest.approx(sorted(measured.diagonal_inertia)[0] + expected) for value in moved
    )


def test_inverted_winding_still_measures_a_positive_mass() -> None:
    inverted = _box((0.1, 0.1, 0.1))
    inverted.invert()
    resolved = resolve_mass(MassProperties(density=1000.0), [inverted], part_name="p")
    assert resolved.mass == pytest.approx(1.0)


def test_open_shapes_fail_instead_of_being_dropped() -> None:
    solid = _box((0.1, 0.1, 0.1))
    open_shell = _box((0.1, 0.1, 0.1))
    open_shell.faces = open_shell.faces[:-2]
    with pytest.raises(ValidationError, match="not closed solids"):
        resolve_mass(MassProperties(density=1000.0), [solid, open_shell], part_name="mixed")


def test_physics_lane_blocks_a_compile_when_a_part_has_no_mass(tmp_path: Path) -> None:
    # The gate is only useful if it stops the compile; most baseline checks report as
    # non-blocking diagnostics, so this guards that missing mass is not one of them.
    from mini_articraft.environments.local import LocalEnvironment

    source = (
        "from mini_articraft.sdk import ArticulatedObject, BoxGeometry, TestContext, TestReport\n"
        "\n"
        "def build_object_model() -> ArticulatedObject:\n"
        "    model = ArticulatedObject('plain')\n"
        "    model.part('body').add(BoxGeometry((0.1, 0.1, 0.1)), name='cube')\n"
        "    return model\n"
        "\n"
        "object_model = build_object_model()\n"
        "\n"
        "def run_tests() -> TestReport:\n"
        "    return TestContext(object_model).report()\n"
    )

    statuses = {}
    for physics in (False, True):
        env = LocalEnvironment(output_dir=tmp_path / str(physics), physics_enabled=physics)
        run_dir = env.create_run(f"run_{physics}")
        (run_dir / "workspace" / "main.py").write_text(source)
        statuses[physics] = env.compile_path(run_dir)["status"]

    assert statuses[False] == "success"
    assert statuses[True] == "error"
