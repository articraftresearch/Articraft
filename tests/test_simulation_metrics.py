from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from articraft.simulate import _representable_range, _tracked_body_separations


def test_joint_motion_is_not_counted_as_part_separation() -> None:
    """Any movable joint moves its subtree, so those pairs are not tracked.

    Watching hinges made every multi link object fail: the shipped four bar
    example read as 45.8 mm of "separation" on 120 mm bars, purely because
    rotating a joint moves everything downstream of it.
    """

    slide = 7
    hinge = 8
    model = SimpleNamespace(
        nbody=4,
        njnt=2,
        jnt_type=np.array([slide, hinge]),
        jnt_bodyid=np.array([2, 3]),
        body_parentid=np.array([0, 0, 1, 2]),
    )
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2]])

    separations = _tracked_body_separations(model, positions, movable_joint_types=(slide, hinge))

    assert separations == {}


def test_rigidly_welded_pairs_are_still_tracked() -> None:
    """Bodies with no movable joint between them must stay put relative to each other."""

    hinge = 8
    model = SimpleNamespace(
        nbody=4,
        njnt=1,
        jnt_type=np.array([hinge]),
        jnt_bodyid=np.array([3]),
        body_parentid=np.array([0, 0, 1, 2]),
    )
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2]])

    separations = _tracked_body_separations(model, positions, movable_joint_types=(hinge,))

    # 1 and 2 are welded to each other; everything across the hinge is exempt.
    assert set(separations) == {(1, 2)}


def test_a_range_that_rounds_to_a_point_is_rejected() -> None:
    """MuJoCo refuses a joint whose two limits print identically."""

    assert _representable_range(-0.5, 0.5)
    assert _representable_range(1e-9, 2e-9)  # tiny, but still two distinct numbers
    assert not _representable_range(0.0, 0.0)
    assert not _representable_range(0.1, 0.1 + 1e-15)


def test_the_shipped_loop_example_stands_up() -> None:
    """End to end, on the example the SDK ships: it used to fail its own check."""

    pytest.importorskip("mujoco", reason="simulation needs the sim dependency group")
    import runpy
    import tempfile
    from pathlib import Path

    from articraft import package_dir
    from articraft.sdk.export import export_assembly
    from articraft.simulate import _loop_pin_anchors, _loop_pin_gap, simulate_usdz, write_mjcf

    values = runpy.run_path(
        str(package_dir / "sdk" / "docs" / "examples" / "closed_loop_linkage.py")
    )
    with tempfile.TemporaryDirectory() as work:
        result = export_assembly(values["object_model"], Path(work) / "result")

        import mujoco

        model = mujoco.MjModel.from_xml_path(str(write_mjcf(result.usdz, Path(work) / "mjcf")))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        # The loop closure became an equality constraint, shut at the start pose.
        anchors = _loop_pin_anchors(
            model,
            data,
            connect_type=mujoco.mjtEq.mjEQ_CONNECT,  # pyright: ignore[reportAttributeAccessIssue]
        )
        assert anchors
        assert _loop_pin_gap(data, anchors) < 1e-9

        outcome = simulate_usdz(result.usdz, Path(work) / "sim")

    assert outcome.parts_stayed_together, outcome.largest_separation_change
    assert outcome.stood_up, outcome.summary()
    # Impact is harsher than rest; the verdict must read the settled number.
    assert outcome.resting_penetration > outcome.deepest_penetration
