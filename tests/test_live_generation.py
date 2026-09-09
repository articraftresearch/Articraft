"""Real model generation. Run with ``uv run pytest -q -m live``."""

from __future__ import annotations

from pathlib import Path

import pytest
from harness import WarmEnvironment, run_scenario

from articraft.agent import Model
from articraft.agent.provider.openai import OpenAIModel
from articraft.settings import DEFAULT_MAX_TURNS, Settings


@pytest.fixture
def live_model(monkeypatch: pytest.MonkeyPatch) -> Model:
    monkeypatch.setitem(Settings.model_config, "env_file", ".env")
    return OpenAIModel(Settings())


@pytest.mark.live
def test_box_generation(live_model: Model, tmp_path: Path) -> None:
    artifacts = run_scenario(
        "a simple box",
        model=live_model,
        env=WarmEnvironment(output_dir=tmp_path),
        max_turns=DEFAULT_MAX_TURNS,
    )
    assert artifacts.record.status == "success"
    assert artifacts.record.result.endswith(".usdz")
    assert (artifacts.run_dir / artifacts.record.result).is_file()
