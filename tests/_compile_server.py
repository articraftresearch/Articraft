"""Warm compile worker for the test suite.

Reads ``{"run_dir": ...}`` requests as newline-delimited JSON on stdin and
writes one compile payload line per request on stdout -- the same payload
shape ``mini_articraft.compiler.worker`` prints for a one-shot compile.
Keeping one worker alive lets a test session pay the geometry import cost
once instead of once per compile (~3s -> ~0.1s).

This is the tests-only half of the worker contract. If a persistent worker
ever becomes a product feature, promote this loop into
``mini_articraft/compiler/worker.py``; the wire format is already shared.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mini_articraft.compiler.result import CompileResult
from mini_articraft.compiler.worker import compile_run


def main() -> int:
    _emit({"ready": True})  # startup handshake: imports are done, requests are served
    for line in sys.stdin:
        run_dir = _parse_request(line)
        if run_dir is None:
            _emit(CompileResult(error=f"bad compile request: {line.strip()!r}").to_payload())
            continue
        try:
            _emit(compile_run(run_dir, include_report=False))
        except BaseException as exc:  # keep serving after a broken compile
            _emit(CompileResult(error=f"{type(exc).__name__}: {exc}").to_payload())
        _evict_workspace_modules(run_dir / "workspace")
    return 0


def _parse_request(line: str) -> Path | None:
    try:
        request = json.loads(line)
        return Path(request["run_dir"]).resolve()
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _emit(payload: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, default=str) + "\n")
    sys.stdout.flush()


def _evict_workspace_modules(workspace: Path) -> None:
    """Forget modules imported from the workspace so the next run re-imports.

    Two runs may both define ``helper.py``; without eviction the second
    compile would silently see the first run's module from ``sys.modules``.
    """
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if file is None:
            continue
        try:
            if Path(file).resolve().is_relative_to(workspace):
                del sys.modules[name]
        except OSError:
            continue


if __name__ == "__main__":
    raise SystemExit(main())
