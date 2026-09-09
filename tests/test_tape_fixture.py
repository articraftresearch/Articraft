"""Tests for the tape_model fixture's record/replay mode matrix.

The contract: no flag means live -- the real model runs and no tape is read
or written. Recording is the only opt-in that writes (``--record``), and
``--replay`` is the offline path (exit when the tape is missing; what CI
runs).
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import conftest
import pytest
import test_live_generation
from harness import (
    RecordingModel,
    ReplayHarness,
    ReplayModel,
    ScriptedModel,
    calls,
    run,
    text,
    tool_call,
)


class _FakeConfig:
    def __init__(self, *, record: bool = False, replay: bool = False):
        self._options = {"--record": record, "--replay": replay}

    def getoption(self, name: str) -> bool:
        return self._options.get(name, False)


@dataclass
class _FakeNode:
    name: str


@dataclass
class _FakeRequest:
    config: _FakeConfig
    node: _FakeNode


def _opener(request: _FakeRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """The fixture's open_tape, bound to a fake request and a tmp library."""
    monkeypatch.setattr(conftest, "TAPE_ROOT", tmp_path / "tapes")
    return conftest._tape_model_opener(cast(pytest.FixtureRequest, request))


def _fake_live_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conftest, "_live_model", lambda: ScriptedModel([text("live")]))


def _forbid_tape_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(self: ReplayHarness, name: str, *, strict: bool = True):
        raise AssertionError("default mode must not load tapes")

    monkeypatch.setattr(ReplayHarness, "replay", forbidden)


def test_default_runs_live_and_never_loads_tapes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_live_model(monkeypatch)
    _forbid_tape_loads(monkeypatch)
    ReplayHarness(tmp_path / "tapes").set("existing", [text("tape")])  # must stay unread
    opener = _opener(_FakeRequest(_FakeConfig(), _FakeNode("existing")), monkeypatch, tmp_path)

    with opener() as model:
        assert not isinstance(model, ReplayModel)
        assert run(model.query([]))["text"] == "live"


def test_default_runs_live_when_no_tape_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_live_model(monkeypatch)
    _forbid_tape_loads(monkeypatch)
    opener = _opener(_FakeRequest(_FakeConfig(), _FakeNode("missing")), monkeypatch, tmp_path)

    with opener() as model:
        assert run(model.query([]))["text"] == "live"


def test_live_generation_can_finish_after_a_tenth_turn_compile(tmp_path: Path) -> None:
    model = ScriptedModel(
        [*[text("Still working.") for _ in range(9)], calls(tool_call("compile")), text("Done.")]
    )

    test_live_generation.test_box_generation(lambda: nullcontext(model), tmp_path)


def test_replay_replays_an_existing_tape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_live_model(monkeypatch)  # live must not be consulted in replay mode
    ReplayHarness(tmp_path / "tapes").set("happy", [text("tape")])
    opener = _opener(
        _FakeRequest(_FakeConfig(replay=True), _FakeNode("happy")), monkeypatch, tmp_path
    )

    with opener() as model:
        assert isinstance(model, ReplayModel)
        assert run(model.query([]))["text"] == "tape"


def test_replay_exits_on_a_missing_tape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    opener = _opener(
        _FakeRequest(_FakeConfig(replay=True), _FakeNode("missing")), monkeypatch, tmp_path
    )
    with pytest.raises(pytest.exit.Exception) as exc_info, opener():
        pass
    assert exc_info.value.returncode == 1


def test_record_runs_live_and_writes_the_tape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_live_model(monkeypatch)
    opener = _opener(
        _FakeRequest(_FakeConfig(record=True), _FakeNode("rec")), monkeypatch, tmp_path
    )
    with opener() as model:
        assert isinstance(model, RecordingModel)
        run(model.query([]))

    library = ReplayHarness(tmp_path / "tapes")
    assert library.has("rec")
    assert library.entries("rec")[0]["response"]["text"] == "live"


def test_record_and_replay_together_are_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(pytest.exit.Exception) as exc_info:
        _opener(
            _FakeRequest(_FakeConfig(record=True, replay=True), _FakeNode("x")),
            monkeypatch,
            tmp_path,
        )
    assert exc_info.value.returncode == 2
