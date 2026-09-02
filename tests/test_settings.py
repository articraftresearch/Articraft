from __future__ import annotations

import pytest
from pydantic import ValidationError

from articraft.agent.compaction import KEEP_RECENT_TOKENS, RESERVE_TOKENS
from articraft.settings import DEFAULT_MAX_TURNS, Settings


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
