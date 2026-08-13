from __future__ import annotations

import json

from articraft.agent.compaction import (
    KEEP_RECENT_TOKENS,
    RESERVE_TOKENS,
    prepare_compaction,
)


def _history(total_tokens: int) -> list[dict]:
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "quickstart"},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "build the reference"},
                {"type": "input_image", "image_url": "data:image/png;base64,REFERENCE"},
            ],
        },
        {
            "role": "assistant",
            "content": "old work",
            "tool_calls": [
                {
                    "name": "write",
                    "arguments": json.dumps({"path": "main.py", "content": "file body"}),
                },
                {
                    "name": "view_image",
                    "arguments": json.dumps({"path": "preview.png"}),
                },
            ],
        },
        {"type": "function_call_output", "call_id": "write", "output": "written"},
        {
            "role": "assistant",
            "content": "r" * (KEEP_RECENT_TOKENS * 4),
            "tool_calls": [],
            "token_usage": {"total_tokens": total_tokens},
        },
    ]


def test_compaction_threshold_and_message_replacement() -> None:
    threshold = 272_000 - RESERVE_TOKENS
    assert prepare_compaction(_history(threshold), 272_000) is None

    messages = _history(threshold + 1)
    plan = prepare_compaction(messages, 272_000)
    assert plan is not None

    compacted = plan.apply(messages, "checkpoint")
    assert plan.tokens_before == threshold + 1
    assert compacted[:3] == messages[:3]
    assert compacted[2]["content"][1]["image_url"].endswith("REFERENCE")
    assert compacted[3]["compaction"]["summary"] == "checkpoint"
    assert compacted[4:] == plan.recent_messages


def test_summary_input_omits_large_payloads() -> None:
    messages = _history(260_000)
    messages[3]["tool_calls"][0]["arguments"] = json.dumps(
        {"path": "main.py", "content": "SECRET_FILE_BODY" * 200}
    )
    messages[3]["tool_calls"].append(
        {
            "name": "edit",
            "arguments": json.dumps(
                {
                    "path": "main.py",
                    "edits": [
                        {
                            "old_text": "SECRET_OLD_TEXT" * 200,
                            "new_text": "SECRET_NEW_TEXT" * 200,
                        }
                    ],
                }
            ),
        }
    )
    messages[4]["output"] = [
        {"type": "input_text", "text": "x" * 3_000},
        {
            "type": "input_image",
            "image_url": "data:image/png;base64,SECRET_IMAGE_DATA",
        },
    ]

    plan = prepare_compaction(messages, 272_000)
    assert plan is not None
    summary_input = plan.summary_messages[1]["content"]
    assert "SECRET_FILE_BODY" not in summary_input
    assert "SECRET_OLD_TEXT" not in summary_input
    assert "SECRET_NEW_TEXT" not in summary_input
    assert "SECRET_IMAGE_DATA" not in summary_input
    assert "[file content omitted]" in summary_input
    assert "[edit text omitted]" in summary_input
    assert "more characters omitted" in summary_input
    assert "image payload omitted" in summary_input


def test_repeated_compaction_updates_checkpoint_and_paths() -> None:
    messages = _history(260_000)
    first = prepare_compaction(messages, 272_000)
    assert first is not None

    compacted = first.apply(messages, "first checkpoint")
    compacted.extend(
        [
            {"type": "function_call_output", "call_id": "view", "output": "viewed"},
            {
                "role": "assistant",
                "content": "n" * (KEEP_RECENT_TOKENS * 4),
                "tool_calls": [
                    {
                        "name": "edit",
                        "arguments": json.dumps({"path": "previews.py"}),
                    }
                ],
                "token_usage": {"total_tokens": 260_000},
            },
        ]
    )

    second = prepare_compaction(compacted, 272_000)
    assert second is not None
    summary_input = second.summary_messages[1]["content"]
    assert "<previous-checkpoint>\nfirst checkpoint" in summary_input
    assert "<compaction-checkpoint>" not in summary_input
    assert second.modified_files == ["main.py", "previews.py"]
    assert second.inspected_images == ["preview.png"]
