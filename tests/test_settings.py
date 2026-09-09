from __future__ import annotations

import pytest
from pydantic import ValidationError

from articraft.agent.compaction import KEEP_RECENT_TOKENS, RESERVE_TOKENS
from articraft.settings import DEFAULT_MAX_TURNS, Settings


def test_openai_defaults_to_astra(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ARTICRAFT_PROVIDER", "ARTICRAFT_MODEL", "ARTICRAFT_REASONING_EFFORT"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings()  # pyright: ignore[reportCallIssue]

    assert settings.provider == "openai"
    assert settings.selected_model == "gpt-6-astra"
    assert settings.selected_reasoning_effort == "high"


@pytest.mark.parametrize("model_name", ["gpt-5.6-sol", "gpt-5.6"])
def test_openai_model_environment_override(
    monkeypatch: pytest.MonkeyPatch, model_name: str
) -> None:
    monkeypatch.setenv("ARTICRAFT_MODEL", model_name)

    assert Settings().openai_model == model_name  # pyright: ignore[reportCallIssue]


def test_settings_ignore_local_dotenv_by_default_in_tests(tmp_path, monkeypatch) -> None:
    tmp_path.joinpath(".env").write_text("ARTICRAFT_MAX_TURNS=999\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert Settings().max_turns == DEFAULT_MAX_TURNS  # pyright: ignore[reportCallIssue]


def test_openrouter_context_window_rejects_values_too_small_to_protect() -> None:
    with pytest.raises(ValidationError, match="36384"):
        Settings(openrouter_context_window_tokens=32_768)  # pyright: ignore[reportCallIssue]


def test_openrouter_context_window_accepts_zero_and_the_minimum() -> None:
    assert RESERVE_TOKENS + KEEP_RECENT_TOKENS == 36_384
    disabled = Settings(openrouter_context_window_tokens=0)  # pyright: ignore[reportCallIssue]
    assert disabled.openrouter_context_window_tokens == 0
    minimum = Settings(openrouter_context_window_tokens=36_384)  # pyright: ignore[reportCallIssue]
    assert minimum.openrouter_context_window_tokens == 36_384


@pytest.mark.parametrize("limit", [0, -1])
def test_openrouter_summary_output_limit_must_be_positive(limit: int) -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        Settings(openrouter_summary_max_output_tokens=limit)  # pyright: ignore[reportCallIssue]
