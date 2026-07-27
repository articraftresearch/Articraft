from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import shlex
import sys
import time

import pytest
from PIL import Image

from mini_articraft.agent.images import LIMITS, prepare_image
from mini_articraft.agent.tools import ToolContext, get, schemas
from mini_articraft.agent.tools._core import workspace_digest
from mini_articraft.agent.tools._exec import ExecSessions
from mini_articraft.compile_feedback import build_compile_report_from_payload
from mini_articraft.environments.local import LocalEnvironment


def run(awaitable):
    return asyncio.get_event_loop().run_until_complete(awaitable)


def context(tmp_path) -> ToolContext:
    env = LocalEnvironment(output_dir=tmp_path)
    run_dir = env.create_run("tools")
    return ToolContext(env, run_dir, run_dir / "workspace")


def test_tool_schemas_include_prompting_guidance() -> None:
    by_name = {tool["name"]: tool for tool in schemas()}
    exec_props = by_name["exec_command"]["parameters"]["properties"]
    stdin_props = by_name["write_stdin"]["parameters"]["properties"]
    edit_props = by_name["edit"]["parameters"]["properties"]

    assert "session ID for ongoing interaction" in by_name["exec_command"]["description"]
    assert "$MINI_ARTICRAFT_PYTHON" in by_name["exec_command"]["description"]
    assert "does not render, copy, or attach images" in by_name["compile"]["description"]
    assert exec_props["command"]["description"] == "Shell command to execute."
    assert (
        exec_props["max_output_tokens"]["description"]
        == "Output token budget. Defaults to 10000 tokens; larger requests may be capped by policy."
    )
    assert (
        stdin_props["chars"]["description"]
        == "Bytes to write to stdin. Defaults to empty, which polls without writing."
    )
    assert "appears exactly once" in edit_props["old_text"]["description"]
    assert get("read").supports_parallel is True
    assert get("view_image").supports_parallel is False
    assert get("exec_command").supports_parallel is False


def test_read_rejects_path_escape(tmp_path) -> None:
    ctx = context(tmp_path)

    with pytest.raises(ValueError, match="inside"):
        run(get("read").run(ctx, {"path": "../outside.txt"}))


def test_read_text_with_offset_and_limit(tmp_path) -> None:
    ctx = context(tmp_path)
    ctx.workspace.joinpath("notes.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")

    result = run(get("read").run(ctx, {"path": "notes.txt", "offset": 2, "limit": 1}))

    assert result == {"path": "notes.txt", "text": "L2: two"}


def test_workspace_tools_handle_default_relative_run_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    env = LocalEnvironment()
    run_dir = env.create_run("relative")
    ctx = ToolContext(env, run_dir, run_dir / "workspace")

    write_result = run(get("write").run(ctx, {"path": "main.py", "content": "old\n"}))
    read_result = run(get("read").run(ctx, {"path": "main.py"}))
    edit_result = run(
        get("edit").run(ctx, {"path": "main.py", "old_text": "old", "new_text": "new"})
    )

    assert write_result == {"path": "main.py", "bytes": 4}
    assert read_result == {"path": "main.py", "text": "L1: old"}
    assert edit_result == {"path": "main.py", "replaced": 1}


def test_read_can_open_symlinked_sdk_docs(tmp_path) -> None:
    ctx = context(tmp_path)

    result = run(
        get("read").run(
            ctx,
            {"path": "docs/sdk/common/20_core_types.md", "offset": 1, "limit": 1},
        )
    )

    assert result["path"] == "docs/sdk/common/20_core_types.md"
    assert result["text"] == "L1: # Shared units and types"


def test_build123d_docs_support_code_and_images(tmp_path) -> None:
    ctx = context(tmp_path)

    page = run(
        get("read").run(
            ctx,
            {"path": "docs/sdk/build123d/tttt.md"},
        )
    )
    snippet = run(
        get("read").run(
            ctx,
            {
                "path": "docs/sdk/build123d/assets/ttt/ttt-23-t-24-curved_support.py",
                "offset": 1,
                "limit": 1,
            },
        )
    )
    assert page["path"] == "docs/sdk/build123d/tttt.md"
    assert "ttt-23-t-24-curved_support.py" in page["text"]
    assert snippet["path"] == "docs/sdk/build123d/assets/ttt/ttt-23-t-24-curved_support.py"
    assert snippet["text"].startswith("L1: ")

    image = run(
        get("view_image").run(
            ctx,
            {"path": "docs/sdk/build123d/assets/AngleIron.png"},
        )
    )
    assert image.output["path"] == "docs/sdk/build123d/assets/AngleIron.png"
    assert image.output["mime_type"] == "image/png"
    assert image.content_items[0]["type"] == "input_image"


def test_read_rejects_images_and_view_image_returns_typed_content(tmp_path) -> None:
    ctx = context(tmp_path)
    image_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    ctx.workspace.joinpath("image.png").write_bytes(image_bytes)

    with pytest.raises(ValueError, match="use view_image"):
        run(get("read").run(ctx, {"path": "image.png"}))

    result = run(get("view_image").run(ctx, {"path": "image.png"}))

    assert result.output["path"] == "image.png"
    assert result.output["mime_type"] == "image/png"
    assert result.output["width"] == 1
    assert result.output["height"] == 1
    assert result.output["detail"] == "original"
    image_url = result.content_items[0]["image_url"]
    assert image_url.startswith("data:image/png;base64,")
    assert base64.b64decode(image_url.split(",", 1)[1]) == image_bytes


def test_view_image_uses_packaged_webp_companion_for_svg(tmp_path) -> None:
    ctx = context(tmp_path)

    result = run(
        get("view_image").run(
            ctx,
            {"path": "docs/sdk/build123d/assets/align.svg"},
        )
    )

    assert result.output["path"] == "docs/sdk/build123d/assets/align.svg"
    assert result.output["raster_path"] == "docs/sdk/build123d/assets/align.svg.webp"
    assert result.output["mime_type"] == "image/webp"
    assert result.content_items[0]["image_url"].startswith("data:image/webp;base64,")


def test_view_image_never_enlarges_or_exceeds_the_patch_limit(tmp_path) -> None:
    ctx = context(tmp_path)
    source_size = (1_249, 1_985)
    image = Image.new("RGB", source_size, color="white")
    image.save(ctx.workspace / "near-limit.png")

    result = run(get("view_image").run(ctx, {"path": "near-limit.png"}))

    output_size = (result.output["width"], result.output["height"])
    assert output_size[0] <= source_size[0]
    assert output_size[1] <= source_size[1]
    assert (
        math.ceil(output_size[0] / 32) * math.ceil(output_size[1] / 32)
        <= LIMITS["original"].max_patches
    )
    high = run(
        get("view_image").run(
            ctx,
            {"path": "near-limit.png", "detail": "high"},
        )
    )
    assert (
        math.ceil(high.output["width"] / 32) * math.ceil(high.output["height"] / 32)
        <= LIMITS["high"].max_patches
    )


def test_prepare_image_rotates_converts_and_bounds_images(tmp_path) -> None:
    jpeg_path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (40, 20), color="white").save(jpeg_path, exif=exif)
    rotated = prepare_image(jpeg_path)
    assert (rotated.width, rotated.height) == (20, 40)

    gif_path = tmp_path / "still.gif"
    Image.new("RGB", (20, 10), color="white").save(gif_path)
    converted = prepare_image(gif_path)
    assert converted.mime_type == "image/png"

    large_path = tmp_path / "large.png"
    Image.new("RGB", (6_100, 100), color="white").save(large_path)
    bounded = prepare_image(large_path)
    assert bounded.width == 6_000
    assert bounded.height < 100


def test_prepare_image_rejects_animated_and_oversized_files(monkeypatch, tmp_path) -> None:
    animated_path = tmp_path / "animated.gif"
    frames = [Image.new("RGB", (2, 2), color=color) for color in ("white", "black")]
    frames[0].save(animated_path, save_all=True, append_images=frames[1:])
    with pytest.raises(ValueError, match="animated GIF"):
        prepare_image(animated_path)

    image_path = tmp_path / "image.png"
    Image.new("RGB", (1, 1), color="white").save(image_path)
    monkeypatch.setattr("mini_articraft.agent.images.MAX_IMAGE_BYTES", 1)
    with pytest.raises(ValueError, match="image exceeds"):
        prepare_image(image_path)


def test_edit_replaces_unique_text(tmp_path) -> None:
    ctx = context(tmp_path)
    path = ctx.workspace / "main.py"
    path.write_text("old\n", encoding="utf-8")

    result = run(get("edit").run(ctx, {"path": "main.py", "old_text": "old", "new_text": "new"}))

    assert result == {"path": "main.py", "replaced": 1}
    assert path.read_text(encoding="utf-8") == "new\n"


def test_edit_fails_when_text_is_not_unique(tmp_path) -> None:
    ctx = context(tmp_path)
    ctx.workspace.joinpath("main.py").write_text("x\nx\n", encoding="utf-8")

    with pytest.raises(ValueError, match="2 times"):
        run(get("edit").run(ctx, {"path": "main.py", "old_text": "x", "new_text": "y"}))


def test_edit_rejects_symlinked_sdk_docs(tmp_path) -> None:
    ctx = context(tmp_path)

    with pytest.raises(ValueError, match="inside"):
        run(
            get("edit").run(
                ctx,
                {
                    "path": "docs/sdk/common/20_core_types.md",
                    "old_text": "Core Types",
                    "new_text": "Core Values",
                },
            )
        )


def test_write_creates_parent_dirs(tmp_path) -> None:
    ctx = context(tmp_path)

    result = run(get("write").run(ctx, {"path": "parts/main.py", "content": "x"}))

    assert result == {"path": "parts/main.py", "bytes": 1}
    assert ctx.workspace.joinpath("parts", "main.py").read_text(encoding="utf-8") == "x"


def test_write_rejects_symlinked_sdk_docs(tmp_path) -> None:
    ctx = context(tmp_path)

    with pytest.raises(ValueError, match="inside"):
        run(get("write").run(ctx, {"path": "docs/sdk/common/new.md", "content": "x"}))


def test_create_run_links_sdk_docs(tmp_path) -> None:
    ctx = context(tmp_path)

    link = ctx.workspace / "docs" / "sdk"
    assert link.is_symlink()
    assert link.joinpath("common", "00_quickstart.md").is_file()


def test_workspace_digest_ignores_docs_caches_and_temp_files(tmp_path) -> None:
    ctx = context(tmp_path)
    baseline = workspace_digest(ctx.workspace)

    for directory in (
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".git",
    ):
        path = ctx.workspace / directory
        path.mkdir()
        path.joinpath("generated.txt").write_text("ignored", encoding="utf-8")
    for name in (
        "helper.pyc",
        "main.py~",
        ".main.py.swp",
        ".#main.py",
        "#main.py#",
        ".~lock.main.py#",
        ".DS_Store",
    ):
        ctx.workspace.joinpath(name).write_text("ignored", encoding="utf-8")
    docs = ctx.workspace / "docs"
    docs.joinpath("sdk").unlink()
    docs.rmdir()
    external_docs = tmp_path / "external-docs"
    external_docs.mkdir()
    external_docs.joinpath("changed.txt").write_text("external", encoding="utf-8")
    docs.symlink_to(external_docs, target_is_directory=True)

    assert workspace_digest(ctx.workspace) == baseline

    ctx.workspace.joinpath("build").mkdir()
    ctx.workspace.joinpath("build", "generated.py").write_text("authored", encoding="utf-8")
    assert workspace_digest(ctx.workspace) != baseline


def test_workspace_digest_hashes_non_doc_symlink_targets(tmp_path) -> None:
    ctx = context(tmp_path)
    target = tmp_path / "shared.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    ctx.workspace.joinpath("shared.py").symlink_to(target)
    baseline = workspace_digest(ctx.workspace)

    target.write_text("VALUE = 2\n", encoding="utf-8")

    assert workspace_digest(ctx.workspace) != baseline


def test_workspace_digest_is_independent_of_workspace_location(tmp_path) -> None:
    left = context(tmp_path / "left")
    right = context(tmp_path / "right")
    left.workspace.joinpath("helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    right.workspace.joinpath("helper.py").write_text("VALUE = 1\n", encoding="utf-8")

    assert workspace_digest(left.workspace) == workspace_digest(right.workspace)


def test_exec_command_reports_output_and_return_code(tmp_path) -> None:
    ctx = context(tmp_path)

    result = run(get("exec_command").run(ctx, {"command": "printf hi; exit 3"}))

    assert result["stdout"] == "hi"
    assert result["stderr"] == ""
    assert result["returncode"] == 3
    assert result["session_id"] is None
    assert result["running"] is False
    assert not ctx.run_dir.joinpath(".tmp").exists()


def test_exec_command_does_not_receive_api_credentials(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("EXAMPLE_API_KEY", "example-secret")
    monkeypatch.setenv("VISIBLE_SETTING", "visible")
    ctx = context(tmp_path)
    command = (
        f"{shlex.quote(sys.executable)} -c "
        "'import json, os; print(json.dumps({key: os.environ.get(key) for key in "
        '["OPENAI_API_KEY", "EXAMPLE_API_KEY", "VISIBLE_SETTING", '
        '"MINI_ARTICRAFT_PYTHON"]}))\''
    )

    result = run(get("exec_command").run(ctx, {"command": command}))

    assert json.loads(result["stdout"]) == {
        "OPENAI_API_KEY": None,
        "EXAMPLE_API_KEY": None,
        "VISIBLE_SETTING": "visible",
        "MINI_ARTICRAFT_PYTHON": sys.executable,
    }


def test_exec_command_renders_public_sdk_previews_before_compile(tmp_path) -> None:
    ctx = context(tmp_path)
    ctx.workspace.joinpath("main.py").write_text(
        """from mini_articraft.sdk import (
    ArticulatedObject,
    ArticulationType,
    BoxGeometry,
    MotionLimits,
    Origin,
    TestContext,
    TestReport,
)

object_model = ArticulatedObject("slider")
object_model.part("base").add(BoxGeometry((0.5, 0.5, 0.5)), name="base")
object_model.part("slider").add(BoxGeometry((0.2, 0.2, 0.2)), name="block")
object_model.articulation(
    "slide",
    ArticulationType.PRISMATIC,
    "base",
    "slider",
    origin=Origin(xyz=(1.0, 0.0, 0.0)),
    axis=(1.0, 0.0, 0.0),
    motion_limits=MotionLimits(lower=0.0, upper=1.0),
)

def run_tests() -> TestReport:
    return TestContext(object_model).report()
""",
        encoding="utf-8",
    )
    ctx.workspace.joinpath("previews.py").write_text(
        """from main import object_model
from mini_articraft.sdk import (
    MeridionalSectionView,
    ModelView,
    MotionStripView,
    SectionView,
    render_view,
)

render_view(
    object_model,
    ModelView.three_quarter(
        width=240,
        height=180,
        selected_parts=("slider",),
    ),
    "qa/previews/selected.png",
)
render_view(
    object_model,
    SectionView(width=240, height=180, plane_normal=(0.0, 1.0, 0.0)),
    "qa/previews/section.png",
)
render_view(
    object_model,
    MeridionalSectionView(width=240, height=180),
    "qa/previews/meridional.png",
)
render_view(
    object_model,
    MotionStripView(
        "slide",
        positions=(0.0, 0.5, 1.0),
        view=ModelView.side(width=160, height=140),
    ),
    "qa/previews/motion.png",
)
""",
        encoding="utf-8",
    )

    result = run(
        get("exec_command").run(
            ctx,
            {"command": '"$MINI_ARTICRAFT_PYTHON" previews.py'},
        )
    )

    assert result["returncode"] == 0, result
    assert ctx.compile_result is None
    for name in ("selected.png", "section.png", "meridional.png", "motion.png"):
        image_path = ctx.workspace / "qa" / "previews" / name
        assert image_path.is_file()
        with Image.open(image_path) as image:
            assert image.width > 0
            assert image.height > 0


def test_preview_output_invalidates_compile_freshness(tmp_path) -> None:
    ctx = context(tmp_path)
    ctx.successful_compile_result = {"status": "success"}
    ctx.successful_compile_digest = workspace_digest(ctx.workspace)
    assert ctx.refresh_compile_freshness()

    result = run(
        get("exec_command").run(
            ctx,
            {"command": "mkdir -p qa/previews && printf preview > qa/previews/detail.txt"},
        )
    )

    assert result["returncode"] == 0
    assert not ctx.refresh_compile_freshness()


def test_exec_command_reports_timeout(tmp_path) -> None:
    ctx = context(tmp_path)
    command = f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(2)'"

    result = run(get("exec_command").run(ctx, {"command": command, "timeout": 0.1}))

    assert result["timed_out"] is True
    assert result["running"] is False


def test_exec_command_rejects_cwd_outside_run_dir(tmp_path) -> None:
    ctx = context(tmp_path)

    with pytest.raises(ValueError, match="run workspace"):
        run(get("exec_command").run(ctx, {"command": "pwd", "cwd": str(tmp_path)}))


def test_exec_command_returns_session_and_write_stdin_polls(tmp_path) -> None:
    ctx = context(tmp_path)
    command = f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(.4); print(\"late\")'"

    async def exercise() -> tuple[dict[str, object], dict[str, object]]:
        first = await get("exec_command").run(ctx, {"command": command, "yield_time_ms": 50})
        second = await get("write_stdin").run(
            ctx,
            {"session_id": first["session_id"], "yield_time_ms": 1000},
        )
        return first, second

    first, second = run(exercise())

    assert first["running"] is True
    assert first["session_id"] is not None
    assert second["stdout"] == "late\n"
    assert second["returncode"] == 0


def test_exec_command_rejects_a_second_live_session(tmp_path) -> None:
    ctx = context(tmp_path)
    command = f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(60)'"

    async def exercise() -> None:
        first = await get("exec_command").run(ctx, {"command": command, "yield_time_ms": 10})
        assert first["running"] is True
        with pytest.raises(ValueError, match="already running"):
            await get("exec_command").run(ctx, {"command": "printf second"})
        await get("write_stdin").run(
            ctx,
            {"session_id": first["session_id"], "chars": "\x03", "yield_time_ms": 1000},
        )

    run(exercise())


def test_write_stdin_sends_input(tmp_path) -> None:
    ctx = context(tmp_path)
    command = f"{shlex.quote(sys.executable)} -c 'import sys; print(sys.stdin.readline().strip().upper())'"

    async def exercise() -> tuple[dict[str, object], dict[str, object]]:
        first = await get("exec_command").run(ctx, {"command": command, "yield_time_ms": 50})
        second = await get("write_stdin").run(
            ctx,
            {"session_id": first["session_id"], "chars": "hello\n", "yield_time_ms": 1000},
        )
        return first, second

    _first, second = run(exercise())

    assert second["stdout"] == "HELLO\n"
    assert second["returncode"] == 0


def test_exec_command_truncates_output_middle(tmp_path) -> None:
    ctx = context(tmp_path)

    result = run(
        get("exec_command").run(
            ctx,
            {"command": "printf 0123456789abcdefghij", "max_output_tokens": 3},
        )
    )

    assert result["stdout"] == "012345…8 chars truncated…efghij"


def test_exec_sessions_aclose_logs_cleanup_errors(caplog) -> None:
    class BrokenSession:
        session_id = 7

        async def aclose(self) -> None:
            raise RuntimeError("close failed")

    sessions = ExecSessions()
    sessions._sessions[7] = BrokenSession()  # type: ignore[assignment]
    caplog.set_level(logging.WARNING, logger="mini_articraft.agent.tools._exec")

    run(sessions.aclose())

    assert sessions._sessions == {}
    assert "failed to close exec session 7 during run cleanup" in caplog.text


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_exec_reaps_detached_descendants_after_exit(tmp_path) -> None:
    """A command's detached children die with the tracked shell."""
    ctx = context(tmp_path)

    result = run(
        get("exec_command").run(
            ctx,
            {
                "command": "nohup sleep 60 >/dev/null 2>&1 & echo $!",
                "login": False,
                "yield_time_ms": 2000,
            },
        )
    )

    pid = int(str(result["stdout"]).strip())
    deadline = time.monotonic() + 5
    while _process_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _process_exists(pid)


def test_exec_command_zero_output_budget_returns_only_a_truncation_marker(tmp_path) -> None:
    ctx = context(tmp_path)

    result = run(
        get("exec_command").run(
            ctx,
            {"command": "printf hello", "max_output_tokens": 0},
        )
    )

    assert result["stdout"] == "…5 chars truncated…"


def test_compile_tool_compiles_workspace(tmp_path) -> None:
    ctx = context(tmp_path)
    ctx.workspace.joinpath("main.py").write_text(
        """
from build123d import *

from mini_articraft.sdk import ArticulatedObject, TestContext, TestReport


def build_object_model() -> ArticulatedObject:
    model = ArticulatedObject("box")
    base = model.part("base")
    base.add(Box(1, 1, 1), name="body")
    return model


object_model = build_object_model()


def run_tests() -> TestReport:
    return TestContext(object_model).report()
""",
        encoding="utf-8",
    )

    result = run(get("compile").run(ctx, {}))

    assert result["status"] == "success"
    assert set(result) == {"status", "compile_signals"}
    assert result["compile_signals"].count("<compile_signals>") == 1
    serialized = json.dumps(result)
    assert "input_image" not in serialized
    assert "data:image" not in serialized
    assert ctx.compile_result is not None
    assert ctx.compile_result["compile_report"]["status"] == "success"
    assert ctx.refresh_compile_freshness()


def test_compile_tool_escalates_repeated_failures(tmp_path) -> None:
    ctx = ToolContext(FakeCompileEnv("error", "error", "error"), tmp_path, tmp_path)

    first = run(get("compile").run(ctx, {}))
    second = run(get("compile").run(ctx, {}))
    third = run(get("compile").run(ctx, {}))

    assert "matches the previous compile attempt" not in first["compile_signals"]
    assert "matches the previous compile attempt" in second["compile_signals"]
    assert "This is compile failure 3 in a row." in third["compile_signals"]
    assert "`exec_command` inspection" in third["compile_signals"]
    assert not ctx.refresh_compile_freshness()


def test_compile_tool_resets_failure_streak_on_success(tmp_path) -> None:
    ctx = ToolContext(FakeCompileEnv("error", "success"), tmp_path, tmp_path)

    run(get("compile").run(ctx, {}))
    result = run(get("compile").run(ctx, {}))

    assert result["status"] == "success"
    assert ctx.last_compile_failure_signature is None
    assert ctx.consecutive_compile_failures == 0


def test_cached_compile_success_resets_failure_streak(tmp_path) -> None:
    ctx = context(tmp_path)
    ctx.successful_compile_result = {
        "status": "success",
        "compile_report": build_compile_report_from_payload(
            {"status": "success", "test_report": None}
        ),
    }
    ctx.successful_compile_digest = workspace_digest(ctx.workspace)
    ctx.last_compile_failure_signature = "previous-failure"
    ctx.consecutive_compile_failures = 2

    result = run(get("compile").run(ctx, {}))

    assert result["status"] == "success"
    assert ctx.last_compile_failure_signature is None
    assert ctx.consecutive_compile_failures == 0


class FakeCompileEnv:
    def __init__(self, *statuses: str) -> None:
        self.statuses = list(statuses)

    def compile_path(self, _run_dir):
        status = self.statuses.pop(0)
        payload = {
            "status": status,
            "error": "ValueError: bad loft" if status == "error" else "",
            "stdout": "",
            "stderr": "",
            "traceback": "",
            "returncode": 1 if status == "error" else 0,
            "test_report": None,
        }
        payload["compile_report"] = build_compile_report_from_payload(payload)
        return payload
