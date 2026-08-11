"""Closed loop joint graphs: a second pin on a part closes a loop.

The first articulation declared for a child owns the tree edge; later ones are
loop closures. They validate, export as regular USD joints marked
``physics:excludeFromArticulation``, and never move a part twice.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from build123d import Box
from pxr import Usd

from articraft import package_dir
from articraft.sdk import ArticulatedObject, ArticulationType, MotionLimits, Origin
from articraft.sdk._collision import MeshCollisionKernel
from articraft.sdk.errors import ValidationError
from articraft.sdk.export import export_object
from articraft.sdk.joints import partition_articulations
from articraft.viewer import load_viewer_run


def _bar(model: ArticulatedObject, name: str):
    part = model.part(name)
    part.add(Box(0.3, 0.04, 0.04), name="bar")
    return part


def _hinge(model, name, parent, child, x, z=0.0):
    return model.articulation(
        name,
        ArticulationType.REVOLUTE,
        parent,
        child,
        origin=Origin(xyz=(x, 0.0, z)),
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-1.0, upper=1.0),
    )


def _four_bar() -> ArticulatedObject:
    model = ArticulatedObject("four_bar")
    ground = _bar(model, "ground")
    crank = _bar(model, "crank")
    coupler = _bar(model, "coupler")
    rocker = _bar(model, "rocker")
    _hinge(model, "crank_pin", ground, crank, -0.15)
    _hinge(model, "coupler_pin", crank, coupler, 0.15)
    _hinge(model, "rocker_pin", ground, rocker, 0.15)
    # rocker already has a parent, so this pin closes the loop.
    _hinge(model, "closing_pin", coupler, rocker, 0.15)
    return model


def test_four_bar_validates_and_partitions() -> None:
    model = _four_bar()
    model.validate()
    tree, loops = partition_articulations(model.articulations)
    assert [item.name for item in tree] == ["crank_pin", "coupler_pin", "rocker_pin"]
    assert [item.name for item in loops] == ["closing_pin"]


def test_two_loops_validate() -> None:
    model = _four_bar()
    brace = _bar(model, "brace")
    _hinge(model, "brace_pin", model.get_part("ground"), brace, -0.15, z=0.1)
    _hinge(model, "brace_closing_pin", model.get_part("coupler"), brace, 0.1)
    model.validate()
    _tree, loops = partition_articulations(model.articulations)
    assert [item.name for item in loops] == ["closing_pin", "brace_closing_pin"]


def test_ordinary_tree_has_no_loops() -> None:
    model = ArticulatedObject("chain")
    base = _bar(model, "base")
    arm = _bar(model, "arm")
    _hinge(model, "pin", base, arm, 0.1)
    model.validate()
    _tree, loops = partition_articulations(model.articulations)
    assert loops == []


def test_loop_joint_does_not_move_its_child() -> None:
    """The closure is a constraint, not a second parent: placement follows the tree."""

    model = _four_bar()
    kernel = MeshCollisionKernel(model, mesh_tolerance=0.001)
    with_value = kernel.world_transforms({"closing_pin": 0.9})
    without = kernel.world_transforms({})
    assert np.allclose(with_value["rocker"], without["rocker"])


def test_missing_part_still_fails() -> None:
    model = ArticulatedObject("broken")
    base = _bar(model, "base")
    arm = _bar(model, "arm")
    _hinge(model, "pin", base, arm, 0.1)
    model.articulations[0].child = "ghost"
    with pytest.raises(ValidationError, match="missing child part"):
        model.validate()


def test_unreachable_parts_still_fail() -> None:
    model = ArticulatedObject("islands")
    _bar(model, "base")
    _bar(model, "adrift")
    with pytest.raises(ValidationError, match="exactly one root"):
        model.validate()


def test_export_marks_only_loop_joints_excluded(tmp_path) -> None:
    result = export_object(_four_bar(), tmp_path / "result")
    stage = Usd.Stage.Open(str(result.usdz))
    excluded = {}
    for prim in stage.Traverse():
        attribute = prim.GetAttribute("articraft:name")
        if not attribute or prim.GetTypeName() not in (
            "PhysicsRevoluteJoint",
            "PhysicsPrismaticJoint",
            "PhysicsFixedJoint",
        ):
            continue
        flag = prim.GetAttribute("physics:excludeFromArticulation")
        excluded[attribute.Get()] = bool(flag.Get()) if flag else False
    assert excluded == {
        "crank_pin": False,
        "coupler_pin": False,
        "rocker_pin": False,
        "closing_pin": True,
    }


def test_manifest_and_viewer_record_loop_closure(tmp_path) -> None:
    run_dir = tmp_path / "run"
    export_object(_four_bar(), run_dir / "result")
    version = load_viewer_run(run_dir).versions[0]
    articulations = cast(
        "list[dict[str, Any]]", cast("dict[str, Any]", version["model"])["articulations"]
    )
    flags = {item["name"]: item["closes_loop"] for item in articulations}
    assert flags == {
        "crank_pin": False,
        "coupler_pin": False,
        "rocker_pin": False,
        "closing_pin": True,
    }


def test_mujoco_enforces_the_closing_pin(tmp_path) -> None:
    """The exported loop survives physics: the pin holds while the linkage falls."""

    mujoco = pytest.importorskip("mujoco", reason="loop constraints need the sim group")
    from articraft.simulate import write_mjcf

    result = export_object(_four_bar(), tmp_path / "result")
    mjcf = write_mjcf(result.usdz, tmp_path / "sim")
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    assert model.neq == 1, "the closing pin should become one equality constraint"

    data = mujoco.MjData(model)
    worst = 0.0
    for _ in range(3000):
        mujoco.mj_step(model, data)
        coupler = data.xmat[model.body("coupler").id].reshape(3, 3)
        rocker_position = data.xpos[model.body("rocker").id]
        pin = coupler @ np.array([0.15, 0.0, 0.0]) + data.xpos[model.body("coupler").id]
        worst = max(worst, float(np.linalg.norm(pin - rocker_position)))
    assert worst < 0.02, f"closing pin drifted {worst * 1000:.1f} mm"


def test_loop_closure_authors_the_true_child_frame(tmp_path) -> None:
    """localPos1 = 0 is only right for tree edges; a loop pin must be located
    in the child's frame or USD consumers snap the child's origin onto it."""

    import runpy

    example = package_dir / "sdk" / "docs" / "examples" / "hydraulic_ram_loop.py"
    values = runpy.run_path(str(example))
    result = export_object(values["object_model"], tmp_path / "result")
    stage = Usd.Stage.Open(str(result.usdz))
    for prim in stage.Traverse():
        name_attr = prim.GetAttribute("articraft:name")
        if not name_attr or name_attr.Get() != "rod_eye_pin":
            continue
        local_pos_1 = tuple(prim.GetAttribute("physics:localPos1").Get())
        # The rod's eye sits at x = REST = 0.20 in the rod's frame.
        assert local_pos_1 == pytest.approx((0.20, 0.0, 0.0), abs=1e-5), local_pos_1
        break
    else:
        pytest.fail("rod_eye_pin not found in the exported stage")


def test_prismatic_loop_closure_warns_in_mjcf(tmp_path) -> None:
    pytest.importorskip("mujoco", reason="loop constraints need the sim group")
    from articraft.simulate import write_mjcf

    model = _four_bar()
    # replace the closing pin with a sliding closure
    model.articulations.pop()
    model.articulation(
        "closing_slide",
        ArticulationType.PRISMATIC,
        model.get_part("coupler"),
        model.get_part("rocker"),
        axis=(1.0, 0.0, 0.0),
        motion_limits=MotionLimits(lower=-0.2, upper=0.2),
    )
    result = export_object(model, tmp_path / "result")
    with pytest.warns(UserWarning, match="prismatic loop closure"):
        write_mjcf(result.usdz, tmp_path / "sim")


def test_release_keeps_the_assembled_pose_for_looped_models(tmp_path) -> None:
    pytest.importorskip("mujoco", reason="loop constraints need the sim group")
    import runpy

    from articraft.simulate import simulate_usdz

    example = package_dir / "sdk" / "docs" / "examples" / "hydraulic_ram_loop.py"
    values = runpy.run_path(str(example))
    result = export_object(values["object_model"], tmp_path / "result")
    outcome = simulate_usdz(result.usdz, tmp_path / "sim", scenario="release", seconds=2.0)
    # Released from the assembled rest pose, the stiff pin must not snap the
    # linkage: the reviewer measured 35/s when mid-travel displacement fought it.
    assert outcome.peak_joint_speed is not None and outcome.peak_joint_speed < 8.0
    assert outcome.stood_up
