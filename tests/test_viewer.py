from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from typing import Any, cast

import pytest
from build123d import Box

from mini_articraft import package_dir
from mini_articraft.sdk import ArticulatedObject, ArticulationType, MotionLimits, Origin
from mini_articraft.sdk.export import export_object
from mini_articraft.viewer import _handler, load_viewer_run


def test_load_viewer_run_reads_each_usdz_version(tmp_path) -> None:
    run_dir = tmp_path / "run-demo"
    result_dir = run_dir / "result"
    export_object(_revolute_model(), result_dir)
    export_object(_prismatic_model(), result_dir)

    viewer_run = load_viewer_run(run_dir)

    assert [version["id"] for version in viewer_run.versions] == ["0001", "0000"]
    latest = cast(dict[str, Any], viewer_run.versions[0]["model"])
    assert latest["name"] == "slider"
    assert latest["parts"] == [
        {
            "name": "base plate",
            "usd_name": "base_plate",
            "shapes": [{"usd_name": "base_shape", "material": None}],
        },
        {
            "name": "carriage",
            "usd_name": "carriage",
            "shapes": [{"usd_name": "payload", "material": None}],
        },
    ]
    joint = cast(list[dict[str, Any]], latest["articulations"])[0]
    assert joint["name"] == "linear travel"
    assert joint["type"] == "prismatic"
    assert joint["parent"] == "base plate"
    assert joint["child"] == "carriage"
    assert joint["axis"] == [1.0, 1.0, 0.0]
    assert joint["origin"] == {"xyz": [0.1, 0.2, 0.3], "rpy": [0.0, 0.1, 0.0]}
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


def test_viewer_handler_serves_only_known_routes(tmp_path) -> None:
    run_dir = tmp_path / "run-demo"
    export_object(_revolute_model(), run_dir / "result")
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
    assert page.count('role="switch"') == 2
    assert "contrastingPalette(version.model.parts.length)" in page
    assert "index%palette.length" not in page


def _revolute_model() -> ArticulatedObject:
    model = ArticulatedObject("hinge")
    base = model.part("base")
    base.add(Box(0.2, 0.2, 0.1), name="body")
    door = model.part("door")
    door.add(Box(0.1, 0.02, 0.2), name="panel")
    model.articulation(
        "hinge joint",
        ArticulationType.REVOLUTE,
        base,
        door,
        axis=(0.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-0.5, upper=0.75),
    )
    return model


def _prismatic_model() -> ArticulatedObject:
    model = ArticulatedObject("slider")
    base = model.part("base plate")
    base.add(Box(0.3, 0.2, 0.05), name="base shape")
    carriage = model.part("carriage")
    carriage.add(Box(0.05, 0.05, 0.05), name="payload")
    model.articulation(
        "linear travel",
        ArticulationType.PRISMATIC,
        base,
        carriage,
        origin=Origin(xyz=(0.1, 0.2, 0.3), rpy=(0.0, 0.1, 0.0)),
        axis=(1.0, 1.0, 0.0),
        motion_limits=MotionLimits(lower=-0.1, upper=0.2),
    )
    return model
