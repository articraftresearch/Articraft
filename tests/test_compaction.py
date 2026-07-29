from __future__ import annotations

import json

from mini_articraft.agent.compaction import (
    KEEP_RECENT_TOKENS,
    RESERVE_TOKENS,
    compacted_messages,
    estimate_context_tokens,
    prepare_compaction,
    serialize_messages,
)


def _static_messages() -> list[dict]:
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "quickstart"},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "build the reference"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,REFERENCE",
                    "detail": "original",
                },
            ],
        },
    ]


def _long_history(total_tokens: int) -> list[dict]:
    return [
        *_static_messages(),
        {
            "role": "assistant",
            "content": "old work",
            "tool_calls": [
                {
                    "id": "call_write",
                    "name": "write",
                    "arguments": json.dumps({"path": "main.py", "content": "old file body"}),
                },
                {
                    "id": "call_view",
                    "name": "view_image",
                    "arguments": json.dumps({"path": "preview-old.png"}),
                },
            ],
        },
        {
            "type": "function_call_output",
            "call_id": "call_write",
            "output": json.dumps({"result": {"path": "main.py"}}),
        },
        {
            "role": "assistant",
            "content": "r" * (KEEP_RECENT_TOKENS * 4),
            "tool_calls": [],
            "token_usage": {"total_tokens": total_tokens},
        },
    ]


def test_compaction_triggers_only_above_working_threshold() -> None:
    threshold = 272_000 - RESERVE_TOKENS

    assert prepare_compaction(_long_history(threshold), 272_000) is None

    plan = prepare_compaction(_long_history(threshold + 1), 272_000)

    assert plan is not None
    assert plan.tokens_before == threshold + 1
    assert len(plan.messages_to_summarize) == 2
    assert plan.recent_messages[0]["role"] == "assistant"


def test_compaction_keeps_static_messages_and_reference_image() -> None:
    messages = _long_history(260_000)
    plan = prepare_compaction(messages, 272_000)

    assert plan is not None
    compacted = compacted_messages(messages, plan, "checkpoint")

    assert compacted[:3] == messages[:3]
    assert compacted[2]["content"][1]["image_url"].endswith("REFERENCE")
    assert compacted[3]["compaction"]["summary"] == "checkpoint"
    assert compacted[4:] == plan.recent_messages


def test_context_estimate_adds_messages_after_latest_usage() -> None:
    messages = [
        *_static_messages(),
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [],
            "token_usage": {"total_tokens": 250_000},
        },
        {
            "type": "function_call_output",
            "call_id": "call_read",
            "output": "x" * 4_000,
        },
    ]

    assert estimate_context_tokens(messages) == 251_000


def test_summary_serialization_omits_file_and_image_payloads() -> None:
    serialized = serialize_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_write",
                        "name": "write",
                        "arguments": json.dumps(
                            {"path": "main.py", "content": "SECRET_FILE_BODY" * 200}
                        ),
                    }
                ],
            },
            {
                "type": "function_call_output",
                "call_id": "call_image",
                "output": [
                    {"type": "input_text", "text": "x" * 3_000},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,SECRET_IMAGE_DATA",
                    },
                ],
            },
        ]
    )

    assert "SECRET_FILE_BODY" not in serialized
    assert "SECRET_IMAGE_DATA" not in serialized
    assert "[file content omitted]" in serialized
    assert "more characters omitted" in serialized
    assert "image payload omitted" in serialized


def test_repeated_compaction_updates_previous_checkpoint_and_tracks_paths() -> None:
    messages = _long_history(260_000)
    first_plan = prepare_compaction(messages, 272_000)
    assert first_plan is not None
    compacted = compacted_messages(messages, first_plan, "first checkpoint")
    compacted.extend(
        [
            {
                "type": "function_call_output",
                "call_id": "call_view",
                "output": json.dumps({"result": {"path": "preview-old.png"}}),
            },
            {
                "role": "assistant",
                "content": "n" * (KEEP_RECENT_TOKENS * 4),
                "tool_calls": [
                    {
                        "id": "call_edit",
                        "name": "edit",
                        "arguments": json.dumps({"path": "previews.py"}),
                    }
                ],
                "token_usage": {"total_tokens": 260_000},
            },
        ]
    )

    second_plan = prepare_compaction(compacted, 272_000)

    assert second_plan is not None
    assert second_plan.previous_summary == "first checkpoint"
    summary_input = second_plan.summary_messages[1]["content"]
    assert "<previous-checkpoint>\nfirst checkpoint" in summary_input
    assert "<compaction-checkpoint>" not in summary_input
    assert second_plan.modified_files == ["main.py", "previews.py"]
    assert second_plan.inspected_images == ["preview-old.png"]
