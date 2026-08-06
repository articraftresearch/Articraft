# Test environment

This suite verifies the generation loop without paid model calls.
`tests/harness.py` is the shared kit; prefer it over per-file fakes.

## The four lanes

| Lane | Cost | Use for |
| --- | --- | --- |
| Unit | ~0s | pure functions: SDK checks, `compile_feedback`, signals |
| Warm compile | ~0.1s per compile | compile behavior via `WarmEnvironment` |
| Scripted agent | ~0.5s per run | the full agent loop via `ScriptedModel` + `run_scenario` |
| Tape replay | free | recorded real runs via `ReplayHarness` |

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
`src/mini_articraft/compiler/runner.py`). They differ only in the worker
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

## Tapes: pay once, replay forever

`ReplayHarness` manages any number of named recordings as
`<name>.jsonl` files under one root. The `replay_harness` pytest fixture
gives each test a scratch library.

```python
# record a real run once (or a scripted run, for free authoring)
with replay_harness.record("hinged-box", OpenAIModel()) as model:
    run(Agent(model, env).run("a hinged box"))

# set: install or transform a tape without running anything
replay_harness.set("plain-box", [calls(tool_call("compile")), text("done")])

# replay: plug any recording into a run; strict mode fails on trajectory drift
artifacts = run_scenario("a box", model=replay_harness.replay("plain-box"))

# erase / clear
replay_harness.erase("plain-box")
```

Strict replay compares a structural fingerprint (roles, tool names, call
ids), not payload text, because tool outputs embed machine-specific run
paths. Rows installed by `set()` carry no fingerprint and match any request.

Curated regression tapes belong in `tests/tapes/` (git-ignored by
default; force-add the ones worth keeping). Scratch tapes belong under
`tmp_path`.

## Live tests: `--record` / `--replay`

Tests using the `tape_model` fixture **replay offline by default** -- a bare
`pytest` run never calls a paid model. Live recording is a deliberate
opt-in via `--record`. Named tapes (`tests/tapes/<name>.jsonl`; default
name = test function, or e.g. `tape_model("latest")`) drive the replay:

```bash
# default: replay the tape offline; skip when no tape exists
uv run pytest tests/test_live_generation.py

# offline and strict: replay the tape; exit(1) if missing (the CI contract)
uv run pytest tests/test_live_generation.py --replay

# live + (re)record the tape (paid, deliberate opt-in; needs OPENAI_API_KEY)
OPENAI_API_KEY=... uv run pytest tests/test_live_generation.py --record
```

Commit a tape (`git add -f`) and CI replays it for free with `--replay`.
Recording is only ever a conscious, flagged act.

## The tape CLI

`scripts/tape.py` manages the recording library by hand:

```bash
uv run python scripts/tape.py list            # recordings + prompts
uv run python scripts/tape.py show box        # one recording's exchanges
uv run python scripts/tape.py replay box      # replay offline, end to end
uv run python scripts/tape.py record box "a small box"   # live, pays once
uv run python scripts/tape.py erase box
```

`record` stores the prompt as tape metadata, which is what `replay`
replays against. `--root` points at another library (the default is
`tests/tapes/`).

## CI: record once, replay everywhere

Two workflows close the loop:

- `record-tapes.yml` (weekly + `workflow_dispatch`, needs the
  `OPENAI_API_KEY` secret) runs the live generation tests with `--record`
  and uploads `tests/tapes/` as the `tapes` artifact.
- `ci.yml` downloads the latest `tapes` artifact before running
  `pytest --replay`, so replay coverage tracks the real model for free;
  with no artifact present it falls back to the committed tapes.

The `dist` job also builds the sdist/wheel on 3.11 and 3.12, runs
`twine check`, installs the wheel into a clean venv, and smoke-tests the
compile worker against `examples/mesh_knob` -- the distribution, not just
the checkout.

## Known platform flakes

On macOS, a few exec-output timing tests can fail with empty captured output
even on a clean tree; they pass on Linux CI. Reproduce on Linux before
touching exec semantics, and do not weaken the assertions to make them pass
locally.
