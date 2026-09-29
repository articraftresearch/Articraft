from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_viewer_motion_javascript() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("viewer motion tests require Node.js (installed in CI)")
    result = subprocess.run(
        [node, "--test", str(Path(__file__).with_name("viewer_motion.test.mjs"))],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
