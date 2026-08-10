from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pxr import Usd, UsdGeom  # pyright: ignore[reportAttributeAccessIssue]

from articraft.sdk import (
    ArticulatedObject,
    ArticulationType,
    BoxGeometry,
    ImagePoint,
    LatheGeometry,
    LineOverlay,
    MeridionalSectionView,
    ModelView,
    MotionLimits,
    MotionStripView,
    Origin,
    PointOverlay,
    Reticle,
    SectionView,
    TestContext,
    ValidationError,
    annotate_image,
    render_view,
)
from articraft.sdk.export import export_object


def test_partial_lathe_is_capped_watertight_and_axis_aware() -> None:
    profile = [(0.2, 0.0), (0.4, 0.0), (0.4, 1.0), (0.2, 1.0)]

    meshes = [
        LatheGeometry(profile, angle=math.pi, axis=axis, cap_ends=True) for axis in ("x", "y", "z")
    ]

    assert all(mesh.is_watertight for mesh in meshes)
    assert all(mesh.to_trimesh().volume > 0.0 for mesh in meshes)
    assert meshes[0].bounds[1][0] - meshes[0].bounds[0][0] == pytest.approx(1.0)
    assert meshes[1].bounds[1][1] - meshes[1].bounds[0][1] == pytest.approx(1.0)
    assert meshes[2].bounds[1][2] - meshes[2].bounds[0][2] == pytest.approx(1.0)


def test_mesh_normal_controls_are_exported_for_every_mesh(tmp_path: Path) -> None:
    model = ArticulatedObject("normals")
    part = model.part("body")
    part.add(BoxGeometry((1.0, 1.0, 1.0)).use_hard_normals(), name="hard")
    part.add(
        BoxGeometry((0.5, 0.5, 0.5)).translate(1.5, 0.0, 0.0).use_smooth_normals(),
        name="smooth",
    )

    result = export_object(model, tmp_path)
    stage = Usd.Stage.Open(str(result.usdz))
    assert stage is not None
    meshes = [
        UsdGeom.Mesh(prim)  # pyright: ignore[reportCallIssue]
        for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Mesh)
    ]

    assert len(meshes) == 2
    by_name = {str(mesh.GetPrim().GetAttribute("articraft:name").Get()): mesh for mesh in meshes}
    assert by_name["hard"].GetNormalsInterpolation() == UsdGeom.Tokens.faceVarying
    assert by_name["smooth"].GetNormalsInterpolation() == UsdGeom.Tokens.vertex
    assert len(by_name["hard"].GetNormalsAttr().Get()) == 36
    assert len(by_name["smooth"].GetNormalsAttr().Get()) == 8
    assert result.audit.meshes_with_normals == 2


def test_geometry_metrics_cover_components_orientation_and_symmetry() -> None:
    model = ArticulatedObject("metrics")
    part = model.part("body")
    joined = BoxGeometry((1.0, 2.0, 3.0))
    joined.merge(BoxGeometry((0.2, 0.2, 0.2)).translate(2.0, 0.0, 0.0))
    part.add(joined, name="joined")
    ctx = TestContext(model)

    metrics = ctx.measure_geometry("body", shape="joined")

    assert metrics.triangle_count == 24
    assert metrics.component_count == 2
    assert metrics.watertight
    assert metrics.orientation == "outward"
    assert metrics.signed_volume > 6.0
    assert ctx.expect_component_count(2, "body", shape="joined")
    assert ctx.expect_positive_volume("body", shape="joined")

    symmetric = ArticulatedObject("symmetric")
    symmetric.part("body").add(BoxGeometry((2.0, 1.0, 1.0)), name="box")
    assert TestContext(symmetric).expect_symmetry(tolerance=1e-9)

    reversed_model = ArticulatedObject("reversed")
    reversed_mesh = BoxGeometry((1.0, 1.0, 1.0))
    reversed_mesh.faces = [(a, c, b) for a, b, c in reversed_mesh.faces]
    reversed_model.part("body").add(reversed_mesh, name="inside_out")
    reversed_metrics = TestContext(reversed_model).measure_geometry()
    assert reversed_metrics.orientation == "inward"
    assert reversed_metrics.signed_volume < 0.0


def test_metrics_and_artifacts_are_recorded(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    model = ArticulatedObject("evidence")
    model.part("body").add(BoxGeometry((1.0, 1.0, 1.0)), name="box")
    ctx = TestContext(model)
    Path("qa").mkdir()
    Path("qa/data.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    ctx.record_metric("triangle budget", 12, unit="triangles")
    assert ctx.expect_metric("width", 1.0, minimum=0.9, maximum=1.1, unit="m")
    artifact = ctx.attach_artifact("qa/data.json", name="sample data")
    report = ctx.report()

    assert report.metrics[0].name == "triangle budget"
    assert report.metrics[1].passed is True
    assert artifact.kind == "json"
    assert report.artifacts == (artifact,)
    with pytest.raises(ValidationError):
        ctx.attach_artifact("../escape.json")
    with pytest.raises(ValidationError):
        ctx.attach_artifact("qa/missing.json")


def _slider_model() -> ArticulatedObject:
    model = ArticulatedObject("slider")
    model.part("base").add(BoxGeometry((0.5, 0.5, 0.5)), name="base")
    model.part("slider").add(BoxGeometry((0.2, 0.2, 0.2)), name="block")
    model.articulation(
        "slide",
        ArticulationType.PRISMATIC,
        "base",
        "slider",
        origin=Origin(xyz=(1.0, 0.0, 0.0)),
        axis=(1.0, 0.0, 0.0),
        motion_limits=MotionLimits(lower=0.0, upper=1.0),
    )
    return model


def test_motion_samples_track_points_and_check_clearance() -> None:
    model = _slider_model()
    ctx = TestContext(model)
    poses = ctx.sample_joint("slide", samples=3)

    points = ctx.track_point("slider", (0.0, 0.0, 0.0), poses)

    assert [point[0] for point in points] == pytest.approx([1.0, 1.5, 2.0])
    assert ctx.expect_no_collision_at_poses("base", "slider", poses)
    assert ctx.expect_distance_at_poses("base", "slider", poses, minimum=0.6)


def test_model_section_meridional_and_motion_views_render(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    model = _slider_model()
    outputs = [
        render_view(
            model,
            ModelView.three_quarter(
                width=240,
                height=180,
                points=(PointOverlay((1.0, 0.0, 0.0), label="tracked"),),
                lines=(LineOverlay((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),),
            ),
            "model.png",
        ),
        render_view(
            model,
            SectionView(width=240, height=180, plane_normal=(0.0, 1.0, 0.0)),
            "section.png",
        ),
        render_view(
            model,
            MeridionalSectionView(width=240, height=180),
            "meridional.png",
        ),
        render_view(
            model,
            MotionStripView(
                "slide",
                positions=(0.0, 0.5, 1.0),
                view=ModelView.side(width=160, height=140),
            ),
            "motion.png",
        ),
    ]

    for output in outputs:
        image = Image.open(output)
        colors = np.asarray(image).reshape((-1, 3))
        assert len(np.unique(colors, axis=0)) > 2
    model_pixels = np.asarray(Image.open("model.png"))
    assert np.any(np.all(model_pixels == np.asarray((220, 45, 45)), axis=2))
    assert Image.open("motion.png").width == 160 * 3 + 12 * 2

    ctx = TestContext(model)
    artifact = ctx.attach_artifact("motion.png", name="slider motion")
    assert artifact.path == "motion.png"
    assert Path(artifact.path).is_file()


def test_reference_reticles_use_normalized_coordinates(tmp_path: Path) -> None:
    source = tmp_path / "reference.webp"
    Image.new("RGB", (200, 100), (240, 240, 240)).save(source)

    output = annotate_image(
        source,
        (Reticle(ImagePoint(0.25, 0.75), "joint"),),
        tmp_path / "marked.png",
    )

    image = np.asarray(Image.open(output))
    assert output == tmp_path / "marked.png"
    assert np.any(np.all(image[70:81, 45:56] == np.asarray((235, 55, 170)), axis=2))
    with pytest.raises(ValidationError, match="between 0 and 1"):
        ImagePoint(1.1, 0.5)
    with pytest.raises(ValidationError, match=r"\.png"):
        annotate_image(source, (), tmp_path / "marked.jpg")
