# Repository guidelines

## Project purpose

mini-articraft is a small reference version of
[articraft](https://github.com/mattzh72/articraft).

Keep this repo simple. The goal is to preserve the useful core idea from
articraft while leaving behind the larger repo's messy code, broad feature set,
viewer, data library, provenance system, and heavy storage flows.

The core loop is:

```text
prompt -> model -> environment -> record
```

When you need behavior from articraft, read the source first and bring over
only the smallest idea needed for this repo. Prefer writing a clear new version
over copying a large module.

## Project shape

The Python package lives in `src/mini_articraft/`.

Use these areas as the main boundaries:

- `agent/` owns the generate and compile loop.
- `models/` owns model adapters.
- `agent/workspace/` owns run creation and compile execution: where the agent works.
- `compiler/` owns the compile itself, plus its result and the signals the agent reads.
- `sdk/` owns the small build123d object API, joints, and export helpers.
- `prompts/` owns the agent prompts.
- `settings.py` owns default runtime settings.
- `record.py` owns the small JSON record and conversation helpers.
- `tests/` owns pytest coverage for the package.

Do not add a viewer, full local library, category system, record manifest,
paper tooling, or broad provider matrix unless the user asks for it directly.

## Development commands

Use `uv` for local work.

```bash
uv sync --group dev
uv run pytest -q
uv run ruff check .
uv run ruff format .
```

Use `OPENAI_API_KEY` in `.env` when testing the OpenAI model adapter.

The CLI entry point is:

```bash
uv run mini-articraft
```

The compile worker entry point is:

```bash
uv run python -m mini_articraft.compiler.worker <run_dir>
```

Prefer calling the compile worker through `LocalWorkspace` unless you are
debugging the worker itself.

## Coding style

Target Python 3.11 and keep the current style.

Use `from __future__ import annotations` in Python modules. Use explicit type
hints for public functions and helpers. Keep dataclasses and Pydantic models
small. Prefer plain functions and simple classes over new frameworks.

Write code in the mini-articraft style: small, direct, and easy to fork. Favor
clear data shapes, compact helpers, and obvious control flow over defensive
frameworks, plugin systems, policy objects, registries, and broad fallback
machinery. Extensible should mean that a reader can understand the core idea and
edit it by hand, not that the repo grows a configurable abstraction layer.

When porting an Articraft idea, keep the useful behavior and drop the ceremony.
Prefer one readable module with a few plain dataclasses and functions over a
large subsystem split across many files. Avoid over-engineering for inputs this
repo does not produce. Keep tests focused on behavior and avoid repeating the
same assertions at every integration layer.

Ruff is configured with a line length of 100, Python 3.11 syntax, import
sorting, and double quotes.

## Change policy

Make narrow changes. Keep each module easy to read on its own.

Avoid hidden global state. Avoid background services. Avoid adding caches,
registries, database layers, or file layouts that are not needed for the small
reference flow.

Generated runs, result files, local secrets, virtual environments, and caches
should stay out of commits. If a new workflow writes generated files, either
write them under a clearly ignored folder or update `.gitignore` in the same
change.

## Testing

Add or update tests when behavior changes. Keep tests close to the code they
cover and name new files `test_<feature>.py`.

Prefer fast pytest tests that exercise the package directly. Use temporary
directories for compile and record tests. Do not require real model calls unless
the test is explicitly about a live adapter.

`tests/harness.py` is the modular test environment for agent-loop behavior:
scripted models, warm-worker compiles (`WarmEnvironment`), and tape replay
(`ReplayHarness`) cover the full loop without paid model calls. See
`tests/README.md` for the four lanes and when to use each. Keep the subprocess
worker contract covered in `test_compile.py`. A few exec-output timing tests
are known to flake on macOS; do not weaken them locally.

Run this before handing off a code change:

```bash
uv run pytest -q
uv run ruff check .
```
