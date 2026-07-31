from __future__ import annotations

import json

import pytest

from mini_articraft.agent.record import Record, append_conversation, read_conversation


def test_record_saves_slim_run_summary(tmp_path) -> None:
    path = tmp_path / "record.json"

    Record().save(path)

    # Pin the serialized field set: the record stays a slim summary. Values and
    # load round-tripping are covered by test_record_round_trips_all_fields.
    assert set(json.loads(path.read_text(encoding="utf-8"))) == {
        "run_id",
        "status",
        "attempts",
        "error",
        "result",
        "cost",
        "token_usage",
    }


def test_record_round_trips_all_fields(tmp_path) -> None:
    path = tmp_path / "record.json"
    record = Record(
        run_id="run_9",
        status="error",
        attempts=3,
        error="boom",
        result="result/model.usdz",
        cost=1.25,
        token_usage={"input": 10, "output": 4},
    )

    record.save(path)

    assert Record.load(path) == record


def test_record_load_missing_file_returns_default(tmp_path) -> None:
    assert Record.load(tmp_path / "absent.json") == Record()


def test_record_load_ignores_unknown_keys(tmp_path) -> None:
    path = tmp_path / "record.json"
    path.write_text(json.dumps({"run_id": "run_1", "legacy_field": "x"}), encoding="utf-8")

    assert Record.load(path) == Record(run_id="run_1")


def test_record_load_rejects_non_object_json(tmp_path) -> None:
    path = tmp_path / "record.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        Record.load(path)


def test_conversation_jsonl_is_append_only(tmp_path) -> None:
    path = tmp_path / "conversation.jsonl"

    append_conversation(path, {"role": "user", "content": "make a drawer"})
    append_conversation(path, {"role": "compiler", "status": "success"})

    events = read_conversation(path)
    assert [event["role"] for event in events] == ["user", "compiler"]
    assert events[0]["content"] == "make a drawer"
    assert events[1]["status"] == "success"
