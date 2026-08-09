"""Seeding a run from an existing object script, so a run can modify it."""

from __future__ import annotations

import pytest

from mini_articraft.agent.harness import _read_prompt
from mini_articraft.agent.workspace.local import DEFAULT_MAIN_PY, LocalWorkspace

_EDITED = DEFAULT_MAIN_PY.replace('ArticulatedObject("object")', 'ArticulatedObject("crank")')


def test_a_seeded_run_starts_from_the_source_script(tmp_path) -> None:
    source = tmp_path / "main.py"
    source.write_text(_EDITED, encoding="utf-8")
    env = LocalWorkspace(output_dir=tmp_path / "runs")

    run_dir = env.create_run("modify", source=source)

    assert run_dir.joinpath("workspace", "main.py").read_text() == _EDITED
    # The rest of the scaffold is unchanged, so the run compiles like any other.
    assert run_dir.joinpath("workspace", "docs", "sdk").is_symlink()


def test_an_unseeded_run_still_gets_the_default_scaffold(tmp_path) -> None:
    env = LocalWorkspace(output_dir=tmp_path / "runs")

    run_dir = env.create_run("fresh")

    assert run_dir.joinpath("workspace", "main.py").read_text() == DEFAULT_MAIN_PY


def test_a_seeded_run_compiles(tmp_path) -> None:
    source = tmp_path / "main.py"
    source.write_text(_EDITED, encoding="utf-8")
    env = LocalWorkspace(output_dir=tmp_path / "runs")

    result = env.compile_path(env.create_run("modify", source=source))

    assert result["status"] == "success"


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("def build_object_model():\n    pass\n", "missing run_tests"),
        ("x = (", "not valid Python"),
    ],
)
def test_a_source_the_worker_cannot_use_is_rejected_before_the_run_exists(
    tmp_path, text: str, match: str
) -> None:
    source = tmp_path / "main.py"
    source.write_text(text, encoding="utf-8")
    env = LocalWorkspace(output_dir=tmp_path / "runs")

    with pytest.raises(ValueError, match=match):
        env.create_run("modify", source=source)

    # Nothing half-built is left behind for the next run id to trip over.
    assert not (tmp_path / "runs" / "modify").exists()


def test_a_missing_source_is_rejected(tmp_path) -> None:
    env = LocalWorkspace(output_dir=tmp_path / "runs")

    with pytest.raises(ValueError, match="cannot be read"):
        env.create_run("modify", source=tmp_path / "absent.py")


def test_the_task_prompt_explains_modifying_only_when_seeded() -> None:
    fresh = _read_prompt("task.md")
    seeded = _read_prompt("task.md", include_source=True)

    assert "already contains a working object" in seeded
    assert "already contains a working object" not in fresh
    assert "<source_prompt>" not in seeded
