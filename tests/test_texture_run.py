from __future__ import annotations

import json
from types import SimpleNamespace

import mini_articraft.compiler.worker as worker
from mini_articraft.sdk import ArticulatedObject, BoxGeometry
from mini_articraft.sdk.export import TextureExportReport, export_object


def _model() -> ArticulatedObject:
    model = ArticulatedObject("plain")
    model.part("base").add(
        BoxGeometry([0.1, 0.1, 0.1]),
        name="body",
        color=(0.5, 0.5, 0.5),
    )
    return model


def _run_dir(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "workspace").mkdir(parents=True)
    (run_dir / "workspace" / "main.py").touch()
    return run_dir


def test_texture_run_preserves_prior_result_when_export_fails(monkeypatch, tmp_path) -> None:
    run_dir = _run_dir(tmp_path)
    prior = export_object(_model(), run_dir / "result")
    manifest_before = prior.manifest.read_bytes()
    monkeypatch.setattr(
        worker.runpy, "run_path", lambda *_args, **_kwargs: {"object_model": _model()}
    )

    def fail_export(*_args, **_kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(worker, "export_object", fail_export)
    outcome = worker.texture_run(run_dir)

    assert not outcome.succeeded
    assert not outcome.applied
    assert prior.usdz.exists()
    assert prior.manifest.read_bytes() == manifest_before


def test_texture_run_keeps_parametric_result_when_no_textures_apply(monkeypatch, tmp_path) -> None:
    run_dir = _run_dir(tmp_path)
    prior = export_object(_model(), run_dir / "result")
    manifest_before = prior.manifest.read_bytes()
    monkeypatch.setattr(
        worker.runpy, "run_path", lambda *_args, **_kwargs: {"object_model": _model()}
    )

    outcome = worker.texture_run(run_dir)

    assert outcome.succeeded
    assert not outcome.applied
    assert prior.manifest.read_bytes() == manifest_before
    assert json.loads(prior.manifest.read_text())["files"] == {"usdz": "usdz/0000.usdz"}
    assert [path.name for path in prior.usdz.parent.glob("*.usdz")] == ["0000.usdz"]


def test_texture_run_reports_the_final_usdz(monkeypatch, tmp_path) -> None:
    run_dir = _run_dir(tmp_path)
    prior = export_object(_model(), run_dir / "result")
    monkeypatch.setattr(
        worker.runpy, "run_path", lambda *_args, **_kwargs: {"object_model": _model()}
    )
    final_usdz = prior.usdz.parent / "0001.usdz"

    def textured_export(*_args, **_kwargs):
        final_usdz.write_bytes(b"usdz")
        return SimpleNamespace(
            usdz=final_usdz,
            textures=TextureExportReport(requested_shapes=1, textured_shapes=1),
        )

    monkeypatch.setattr(worker, "export_object", textured_export)

    outcome = worker.texture_run(run_dir)

    assert outcome.applied
    assert outcome.usdz == final_usdz
    assert not prior.usdz.exists()
