from __future__ import annotations

from mini_articraft.settings import DEFAULT_MAX_TURNS, Settings


def test_settings_ignore_local_dotenv_by_default_in_tests(tmp_path, monkeypatch) -> None:
    tmp_path.joinpath(".env").write_text("MINI_ARTICRAFT_MAX_TURNS=999\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert Settings().max_turns == DEFAULT_MAX_TURNS  # pyright: ignore[reportCallIssue]
