from __future__ import annotations

import json
import math
import runpy
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from typing import Any, cast

import pytest
from build123d import Box

from articraft import package_dir
from articraft.sdk import (
    JointAxis,
    JointDOF,
    JointFrame,
    RigidBodyAssembly,
)
from articraft.sdk.export import export_assembly
from articraft.viewer import _handler, load_viewer_run


def test_load_viewer_run_reads_each_usdz_version(tmp_path) -> None:
    run_dir = tmp_path / "run-demo"
    result_dir = run_dir / "result"
    export_assembly(_revolute_model(), result_dir)
    export_assembly(_prismatic_model(), result_dir)

    viewer_run = load_viewer_run(run_dir)

    assert [version["id"] for version in viewer_run.versions] == ["0001", "0000"]
    latest = cast(dict[str, Any], viewer_run.versions[0]["model"])
    assert latest["name"] == "slider"
    assert latest["can_pose"] is True
    assert latest["parts"] == [
        {
            "name": "base_plate",
            "usd_name": "base_plate",
            "shapes": [
                {
                    "usd_name": "base_shape",
                    "appearance": None,
                    "material": None,
                    "coating": None,
                }
            ],
            "mass": None,
        },
        {
            "name": "carriage",
            "usd_name": "carriage",
            "shapes": [
                {"usd_name": "payload", "appearance": None, "material": None, "coating": None}
            ],
            "mass": None,
        },
    ]
    joint = cast(list[dict[str, Any]], latest["articulations"])[0]
    assert joint["name"] == "linear_travel"
    assert joint["type"] == "prismatic"
    assert joint["parent"] == "base_plate"
    assert joint["child"] == "carriage"
    # A named axis plus a rotated frame: the diagonal lives in origin.rpy.
    assert joint["axis"] == [1.0, 0.0, 0.0]
    assert joint["origin"]["xyz"] == pytest.approx([0.1, 0.2, 0.3])
    assert joint["origin"]["rpy"] == pytest.approx([0.0, 0.1, math.pi / 4.0])
    assert joint["child_origin"] == {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]}
    limits = cast(dict[str, float], joint["motion_limits"])
    assert limits["lower"] == pytest.approx(-0.1)
    assert limits["upper"] == pytest.approx(0.2)
    prior = cast(dict[str, Any], viewer_run.versions[1]["model"])
    assert cast(list[dict[str, Any]], prior["articulations"])[0]["type"] == "revolute"


def test_load_viewer_run_rejects_empty_and_invalid_runs(tmp_path) -> None:
    empty_run = tmp_path / "empty"
    empty_run.mkdir()
    with pytest.raises(ValueError, match="no numbered USDZ files"):
        load_viewer_run(empty_run)

    invalid_run = tmp_path / "invalid"
    usdz_dir = invalid_run / "result" / "usdz"
    usdz_dir.mkdir(parents=True)
    usdz_dir.joinpath("0000.usdz").write_text("not usd", encoding="utf-8")
    with pytest.raises(ValueError, match="could not open USDZ"):
        load_viewer_run(invalid_run)


def test_a_closed_loop_is_never_posed_along_the_tree(tmp_path) -> None:
    """The loop is routed to the solver rather than walked.

    This used to disable posing outright, which left a linkage frozen. The
    contract that mattered is unchanged -- a closure is not a tree edge -- but
    it is now met by asking the SDK instead of by refusing.
    """

    model = RigidBodyAssembly("loop")
    base = model.rigid_body("base")
    base.add(Box(0.2, 0.2, 0.1), name="body")
    link = model.rigid_body("link")
    link.add(Box(0.1, 0.02, 0.2), name="body")
    tree = model.joint(
        "tree",
        base.at(),
        link.at(),
        dofs=(JointDOF(JointAxis.ROT_Y),),
    )
    model.joint(
        "closure",
        base.at(),
        link.at(),
        dofs=(JointDOF(JointAxis.ROT_Y),),
    )
    model.articulation("main", root=base, joints=(tree,))
    run_dir = tmp_path / "run"
    export_assembly(model, run_dir / "result")

    graph = cast(dict[str, Any], load_viewer_run(run_dir).versions[0]["model"])

    assert graph["solver"] == "server"
    assert [joint["name"] for joint in graph["articulations"] if joint["closes_loop"]] == [
        "closure"
    ]


def test_viewer_orients_symmetric_joint_from_the_articulation_root(tmp_path) -> None:
    model = RigidBodyAssembly("child_first")
    base = model.rigid_body("base")
    base.add(Box(0.2, 0.2, 0.1), name="body")
    arm = model.rigid_body("arm")
    arm.add(Box(0.1, 0.02, 0.2), name="body")
    hinge = model.joint(
        "hinge",
        arm.at(JointFrame(xyz=(0.2, 0.0, 0.0))),
        base.at(JointFrame(xyz=(0.1, 0.0, 0.0))),
        dofs=(JointDOF(JointAxis.ROT_Y, limits=(-0.5, 0.75)),),
    )
    model.articulation("main", root=base, joints=(hinge,))
    run_dir = tmp_path / "run"
    export_assembly(model, run_dir / "result")

    graph = cast(dict[str, Any], load_viewer_run(run_dir).versions[0]["model"])
    joint = cast(list[dict[str, Any]], graph["articulations"])[0]

    assert joint["parent"] == "base"
    assert joint["child"] == "arm"
    assert joint["origin"]["xyz"] == pytest.approx([0.1, 0.0, 0.0])
    assert joint["child_origin"]["xyz"] == pytest.approx([0.2, 0.0, 0.0])
    assert joint["axis"] == [0.0, -1.0, 0.0]
    assert joint["motion_limits"] == {"lower": -0.75, "upper": 0.5}


def test_viewer_handler_serves_only_known_routes(tmp_path) -> None:
    run_dir = tmp_path / "run-demo"
    export_assembly(_revolute_model(), run_dir / "result")
    viewer_run = load_viewer_run(run_dir)
    bootstrap = json.dumps(viewer_run.bootstrap()).encode()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _handler(b"<h1>viewer</h1>", bootstrap, viewer_run.files),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert urllib.request.urlopen(f"{base}/").read() == b"<h1>viewer</h1>"
        bootstrap_payload = json.load(urllib.request.urlopen(f"{base}/api/bootstrap"))
        assert bootstrap_payload["versions"][0]["id"] == "0000"
        response = urllib.request.urlopen(f"{base}/models/0000.usdz")
        assert response.headers.get_content_type() == "model/vnd.usdz+zip"
        assert response.read() == viewer_run.files["0000"].read_bytes()
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{base}/models/9999.usdz")
        assert exc_info.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{base}/record.json")
        assert exc_info.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_viewer_page_exposes_only_the_minimal_view_options() -> None:
    page = (package_dir / "viewer.html").read_text(encoding="utf-8")

    assert 'id="part-colors"' in page
    assert 'id="preview-motion"' in page
    # Only shown once a run has been simulated; the option list stays enumerated
    # here so it cannot creep.
    assert 'id="play-simulation"' in page
    assert page.count('role="switch"') == 3
    assert "contrastingPalette(version.model.parts.length)" in page
    assert "index%palette.length" not in page


def _revolute_model() -> RigidBodyAssembly:
    model = RigidBodyAssembly("hinge")
    base = model.rigid_body("base")
    base.add(Box(0.2, 0.2, 0.1), name="body")
    door = model.rigid_body("door")
    door.add(Box(0.1, 0.02, 0.2), name="panel")
    model.joint(
        "hinge_joint",
        base.at(JointFrame()),
        door.at(JointFrame()),
        dofs=(JointDOF(JointAxis.ROT_Y, limits=(-0.5, 0.75)),),
    )
    return model


def _prismatic_model() -> RigidBodyAssembly:
    model = RigidBodyAssembly("slider")
    base = model.rigid_body("base_plate")
    base.add(Box(0.3, 0.2, 0.05), name="base shape")
    carriage = model.rigid_body("carriage")
    carriage.add(Box(0.05, 0.05, 0.05), name="payload")
    # A diagonal travel direction rides in the frame's rotation.
    model.joint(
        "linear_travel",
        base.at(JointFrame(xyz=(0.1, 0.2, 0.3), rpy=(0.0, 0.1, math.pi / 4.0))),
        carriage.at(JointFrame()),
        dofs=(JointDOF(JointAxis.TRANS_X, limits=(-0.1, 0.2)),),
    )
    model.articulation("main", root=base, joints=["linear_travel"])
    return model


def test_viewer_composes_joint_frames_in_sdk_order() -> None:
    """The page must build ``origin.rpy`` the way the SDK does.

    The SDK composes rpy as an extrinsic ``sxyz`` matrix, which is Rz*Ry*Rx and
    therefore three.js Euler order ``ZYX``. Order only shows up once two
    components are nonzero, so a tripod leg yawed 120 degrees off a pitched
    frame silently swung around the wrong axis while single axis hinges looked
    perfect.
    """

    page = (package_dir / "viewer.html").read_text(encoding="utf-8")
    assert 'new THREE.Euler(...frame.rpy,"ZYX")' in page


def test_viewer_keeps_body_nodes_flat_and_plays_complete_body_poses() -> None:
    page = (package_dir / "viewer.html").read_text(encoding="utf-8")

    assert "root.add(node)" in page
    assert "parent.add(frame)" not in page
    assert "Object.entries(frame.bodies??{})" in page
    assert "previewButton.disabled=!state.version?.model.can_pose" in page


def test_sdk_rpy_matrix_is_extrinsic_xyz() -> None:
    """Pin the convention the viewer is matched against."""

    import numpy as np

    from articraft.sdk.assembly import _frame_matrix

    rpy = (0.0, -0.35, 2.09)
    hinge = _frame_matrix(JointFrame(rpy=rpy))[:3, :3] @ np.array([0.0, 1.0, 0.0])
    # A leg yawed off a pitched frame keeps a level hinge axis; the reversed
    # order tilts it out of plane, which is the bug this guards.
    assert abs(float(hinge[2])) < 1e-9


def _loop_model() -> RigidBodyAssembly:
    """The shipped four-bar, which the viewer could not pose at all before."""

    values = runpy.run_path(
        str(package_dir / "sdk" / "docs" / "examples" / "closed_loop_linkage.py")
    )
    return cast(RigidBodyAssembly, values["object_model"])


def _serve(run):
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        _handler(b"", json.dumps(run.bootstrap()).encode(), run.files),
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _post(server, path: str, payload: dict) -> tuple[int, Any]:
    url = f"http://127.0.0.1:{server.server_port}{path}"
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def test_a_looped_model_is_posed_by_the_solver_not_the_tree(tmp_path) -> None:
    """A loop makes tree posing wrong, so those models are marked for the server."""

    run_dir = tmp_path / "loop-run"
    export_assembly(_loop_model(), run_dir / "result")

    model = cast(dict[str, Any], load_viewer_run(run_dir).versions[0]["model"])

    # Before this, any closure forced can_pose False and the model was frozen.
    assert model["can_pose"] is True
    assert model["solver"] == "server"


def test_posing_a_loop_matches_the_sdk_exactly(tmp_path) -> None:
    """The viewer must agree with compile checks and render_view, to the bit."""

    run_dir = tmp_path / "loop-run"
    assembly = _loop_model()
    export_assembly(assembly, run_dir / "result")
    run = load_viewer_run(run_dir)
    resolved = assembly.resolve()
    server = _serve(run)
    try:
        for angle in (-0.5, 0.0, 0.5):
            status, solved = _post(server, "/api/pose/0000", {"ground_left": angle})
            assert status == 200
            truth = resolved.forward_kinematics({"ground_left.rotZ": angle}).body_poses
            for name, matrix in truth.items():
                served = [round(value, 9) for value in solved["bodies"][name]]
                expected = [round(float(value), 9) for row in matrix for value in row]
                assert served == expected, name
            # The followers moved, so the sliders must be told.
            assert solved["dofs"]["coupler_right.rotZ"] == pytest.approx(angle, abs=1e-6)
    finally:
        server.shutdown()


def test_an_unreachable_pose_is_reported_not_raised(tmp_path) -> None:
    run_dir = tmp_path / "loop-run"
    export_assembly(_loop_model(), run_dir / "result")
    server = _serve(load_viewer_run(run_dir))
    try:
        status, body = _post(server, "/api/pose/0000", {"ground_left": 99.0})
        assert status == 409
        assert "outside limits" in body
    finally:
        server.shutdown()


def test_a_tree_model_still_poses_in_the_browser(tmp_path) -> None:
    """No loop means no round trip: the tree walk is exact and free."""

    run_dir = tmp_path / "tree-run"
    export_assembly(_revolute_model(), run_dir / "result")

    model = cast(dict[str, Any], load_viewer_run(run_dir).versions[0]["model"])

    assert model["can_pose"] is True
    assert model["solver"] == "tree"
