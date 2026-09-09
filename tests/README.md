# Test environment

This suite checks the generation loop with scripted tests and real model calls.
`tests/harness.py` is the shared kit; prefer it over per-file fakes.

## The four lanes

| Lane | Cost | Use for |
| --- | --- | --- |
| Unit | ~0s | pure functions: SDK checks, `compile_feedback`, signals |
| Warm compile | ~0.1s per compile | compile behavior via `WarmEnvironment` |
| Scripted agent | ~0.5s per run | the full agent loop via `ScriptedModel` + `run_scenario` |
| Live | paid | real model generation and compile in `test_live_generation.py` |

```python
from harness import WarmEnvironment, run_scenario, calls, text, tool_call

artifacts = run_scenario(
    "a box",
    [
        calls(tool_call("write", {"path": "main.py", "content": GOOD_MAIN_PY})),
        calls(tool_call("compile")),
        text("done"),
    ],
    env=WarmEnvironment(output_dir=tmp_path),
)
assert artifacts.record.status == "success"
```

A scripted step can also be a callable `(ModelQuery) -> Response`, so the
"model" can assert on what the agent sent and react to earlier tool outputs.
See `tests/test_agent_scenarios.py` for end-to-end examples (repair loops,
repeat-failure guidance, allowances).

## WarmEnvironment vs LocalWorkspace

Both lanes run every compile in a worker subprocess with the same timeout,
cleanup, and result-assembly contract (shared in
`src/articraft/agent/workspace/local.py`). They differ only in the worker
lifecycle:

- `LocalWorkspace` (cold) spawns a fresh interpreter per compile (~3s).
  It owns the fresh-interpreter, process-cleanup, and installed-wheel
  contracts (`test_compile.py`).
- `WarmEnvironment` (warm) keeps one worker (`tests/_compile_server.py`)
  alive for the whole test session, so compiles cost ~0.1s. A compile that
  times out or kills the worker (e.g. `os._exit` in workspace code) gets an
  error result; the next compile lazily starts a fresh worker. Compiles are
  serialized through the shared worker.

Agent-loop tests that monkeypatch the compile tool (`compile_success_tool()`)
never compile at all and can use either environment.

## Live tests

A bare `uv run pytest -q` includes live generation. It reads `OPENAI_API_KEY`
from the environment or `.env`; missing credentials fail the live test.
The live test uses the production turn budget. Scripted scenarios keep a
shorter default so an unexpected extra turn fails quickly.

```bash
uv run pytest -q                              # full suite, including paid calls
uv run pytest -q -m live                      # real generation only
uv run pytest -q -m 'not live and not volume'  # no credentials needed
```

## CI

`ci.yml` runs the tests that need no credentials on pull requests and main.
`live-tests.yml` runs real model calls weekly and through `workflow_dispatch`
using the `OPENAI_API_KEY` repository secret.

The `dist` job also builds the sdist/wheel on 3.11 and 3.12, runs
`twine check`, installs the wheel into a clean venv, and smoke-tests the
compile worker against `examples/mesh_knob` -- the distribution, not just
the checkout.

## Known platform flakes

On macOS, a few exec-output timing tests can fail with empty captured output
even on a clean tree; they pass on Linux CI. Reproduce on Linux before
touching exec semantics, and do not weaken the assertions to make them pass
locally.
