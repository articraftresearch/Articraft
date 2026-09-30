from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_agent_tools_and_workspace_do_not_load_the_cad_compiler(tmp_path: Path) -> None:
    code = """
import sys
from pathlib import Path
from articraft.agent.tools import schemas
from articraft.agent.workspace.local import LocalWorkspace

names = {schema["name"] for schema in schemas()}
assert {"read", "write", "edit", "compile"} <= names
workspace = LocalWorkspace(output_dir=Path(sys.argv[1]))
run = workspace.create_run("import-test")
assert (run / "workspace" / "main.py").is_file()
heavy = {"articraft.compiler.feedback", "articraft.sdk", "build123d", "OCP"}
assert not heavy.intersection(sys.modules), heavy.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], check=True)


def test_compile_results_still_reach_the_agent_as_feedback(tmp_path: Path) -> None:
    from articraft.agent.tools import ToolContext
    from articraft.agent.tools.compile import _compact_result, _internal_result
    from articraft.agent.workspace.local import LocalWorkspace, _finalize_result
    from articraft.compiler.result import CompileResult

    workspace = LocalWorkspace(output_dir=tmp_path)
    run = workspace.create_run("feedback-test")
    context = ToolContext(workspace, run, run / "workspace")
    payload = _finalize_result(CompileResult(status="error", error="test failure"), 1)
    feedback = _compact_result(_internal_result(context, payload))
    assert feedback["status"] == "error"
    assert "test failure" in feedback["compile_signals"]
    assert context.consecutive_compile_failures == 1
