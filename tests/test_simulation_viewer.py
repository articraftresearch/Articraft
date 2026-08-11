"""The viewer plays back what the simulation recorded."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from articraft.sdk import (
    BoxGeometry,
    JointAxis,
    JointDOF,
    JointFrame,
    Material,
    RigidBodyAssembly,
)
from articraft.sdk.export import export_assembly
from articraft.viewer import load_viewer_run

pytest.importorskip("mujoco", reason="recording motion needs the sim dependency group")

from articraft.simulate import simulate_usdz


def _run_dir(tmp_path: Path) -> tuple[Path, Path]:
    """A run directory laid out the way the viewer expects."""
    model = RigidBodyAssembly("crate")
    base = model.rigid_body("base")
    base.add(
        BoxGeometry((0.30, 0.20, 0.10)).translate(0.0, 0.0, 0.05),
        name="body",
        material=Material.HARDWOOD,
    )
    lid = model.rigid_body("lid")
    lid.add(
        BoxGeometry((0.30, 0.20, 0.01)).translate(0.0, 0.10, 0.0),
        name="panel",
        material=Material.STEEL,
    )
    model.joint(
        "lid_hinge",
        body0=base,
        frame0=JointFrame(),
        body1=lid,
        frame1=JointFrame(),
        dofs=(JointDOF(JointAxis.ROT_X, limits=(0.0, 1.5)),),
    )

    run_dir = tmp_path / "run"
    result_dir = run_dir / "result"
    export_assembly(model, result_dir)
    usdz = next((result_dir / "usdz").glob("*.usdz"))
    return run_dir, usdz


def test_a_recorded_trajectory_reaches_the_viewer(tmp_path: Path) -> None:
    run_dir, usdz = _run_dir(tmp_path)
    simulation_dir = run_dir / "result" / "simulation"
    result = simulate_usdz(usdz, simulation_dir, seconds=1.0)
    assert result.trajectory is not None
    (simulation_dir / f"{usdz.stem}.trajectory.json").write_text(
        json.dumps(result.trajectory.to_payload())
    )

    version = load_viewer_run(run_dir).versions[0]
    trajectory = version["trajectory"]

    assert isinstance(trajectory, dict)
    assert trajectory["root"] == "base"
    assert trajectory["joints"] == ["lid_hinge"]
    assert len(trajectory["frames"]) > 1


def test_the_viewer_sees_no_trajectory_until_the_run_is_simulated(tmp_path: Path) -> None:
    run_dir, _ = _run_dir(tmp_path)

    version = load_viewer_run(run_dir).versions[0]

    assert "trajectory" not in version


def test_recorded_quaternions_are_unit_and_wxyz_ordered(tmp_path: Path) -> None:
    """MuJoCo orders quaternions (w, x, y, z); three.js takes (x, y, z, w).

    The viewer reorders on playback, so this pins the order it is reordering
    from. An object dropped flat barely rotates, so w dominates.
    """
    run_dir, usdz = _run_dir(tmp_path)
    result = simulate_usdz(usdz, run_dir / "result" / "simulation", seconds=1.0)

    assert result.trajectory is not None
    for frame in result.trajectory.frames:
        quaternion = frame["root"]["quat"]
        assert len(quaternion) == 4
        assert sum(value * value for value in quaternion) == pytest.approx(1.0, abs=1e-3)
        assert abs(quaternion[0]) > 0.9  # w first, and near-upright throughout


def test_the_object_falls_to_the_floor_over_the_recording(tmp_path: Path) -> None:
    run_dir, usdz = _run_dir(tmp_path)
    result = simulate_usdz(usdz, run_dir / "result" / "simulation", seconds=1.0)

    assert result.trajectory is not None
    frames = result.trajectory.frames
    assert frames[-1]["root"]["pos"][2] < frames[0]["root"]["pos"][2]
    assert frames[-1]["t"] > frames[0]["t"]
