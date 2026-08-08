<h1 align="center">mini-articraft</h1>

mini-articraft is a small agent that turns a prompt into an articulated 3D object.

> [!NOTE]
> mini-articraft is the supported successor to the
> [original Articraft harness](https://github.com/mattzh72/articraft).
> Researchers and engineers from academia and industry maintain and support this project.

> [!IMPORTANT]
> mini-articraft is under active development. Expect changes before version 1.0.

<p align="center">
<img width="500" height="490" alt="Screen Recording 2026-07-27 at 1 57 32 AM" src="https://github.com/user-attachments/assets/683b8c3f-32c7-4b95-98ee-0cec2300ddbd" />
</p>

The agent writes a Python model. It compiles and checks the geometry. It exports a posable USDZ
file. To make objects directly, read about the [mesh authoring SDK](docs/sdk.md). To understand
the generation loop, read the [agent design](docs/agent.md).

---

## Quickstart

### Install and run mini-articraft

Install the package and the development tools:

```shell
uv sync --group dev
```

Add your OpenAI API key to `.env`:

```shell
OPENAI_API_KEY=your_key_here
```

Generate an object:

```shell
uv run mini-articraft "a jet engine"
```

Add one local reference image when you want to reconstruct an object:

```shell
uv run mini-articraft --image reference.png "reconstruct this desk lamp"
```

OpenAI is the default provider. To use Gemini, pass a Gemini API key and select the provider:

```shell
GEMINI_API_KEY=your_key_here uv run mini-articraft \
  --provider gemini --model gemini-3.6-flash "make a folding chair"
```

To use Anthropic, pass an Anthropic API key and select the provider:

```shell
ANTHROPIC_API_KEY=your_key_here uv run mini-articraft \
  --provider anthropic --model claude-sonnet-5 \
  --image reference.png "reconstruct this folding chair"
```

The local model catalog tracks current pricing and context metadata for:

- OpenAI GPT-5.6: `gpt-5.6` / `gpt-5.6-sol`, `gpt-5.6-terra`, and `gpt-5.6-luna`.
- Anthropic Claude 5: `claude-fable-5`, `claude-mythos-5`, `claude-opus-5`, and
  `claude-sonnet-5` (`claude-mythos-5` is invitation-only).
- Google production models: `gemini-3.6-flash` and `gemini-3.5-flash-lite`.

These families expose roughly 1M-token API context windows. Mini-articraft keeps its existing
272k working budget for compaction and TUI context tracking.

Other model slugs are passed through to the selected provider. The live TUI warns when a slug
has no local metadata, so context tracking and cost estimates are unavailable. If the provider
rejects the slug, the run fails with the provider error.

Each run keeps the complete Anthropic response blocks in `conversation.jsonl`.

To use OpenRouter, provide an API key:

```shell
OPENROUTER_API_KEY=your_key_here uv run mini-articraft \
  --provider openrouter "make a folding chair"
```

OpenRouter defaults to `nvidia/nemotron-3-ultra-550b-a55b:free`. You can select another model
with `--model` or `MINI_ARTICRAFT_OPENROUTER_MODEL`.
The OpenRouter lane is text-only, so it omits image tools and related agent instructions.
Optional `OPENROUTER_HTTP_REFERER` and `OPENROUTER_APP_TITLE` values enable OpenRouter app
attribution. OpenRouter reports token usage and request cost when available. Because arbitrary
model identifiers are accepted without a local model catalog, their context window is reported as
unknown and the TUI omits its context-percentage display.

Each run is in the `runs/` directory. Open a completed run in the browser viewer:

```shell
uv run mini-articraft view runs/<run-id>
```

Use the viewer to examine each generated version and move its joints.

### Use it from Python

Call the same generation loop directly from Python:

```python
import mini_articraft

result = mini_articraft.generate(
    "reconstruct this desk lamp",
    image="reference.png",
    provider="anthropic",
    model="claude-sonnet-5",
    on_event=print,
)

print(result.status, result.run_dir, result.artifact)
```

`generate()` blocks until the run finishes. Pass `on_event` to receive progress
events as they happen.

Async applications use the native coroutine:

```python
async def main():
    result = await mini_articraft.generate_async(
        "reconstruct this desk lamp",
        image="reference.png",
        on_event=print,
    )
    print(result.status, result.artifact)
```

`generate_async()` works with normal asyncio tasks, cancellation, and timeouts.
Cancellation takes effect at the next await point. A compile already in progress
finishes before cancellation completes so it is not abandoned in the background.

### Simulate a run

Export validation says the USD is well formed. Simulation says whether the object
stands up. Drop a run on a floor and see what happens:

```shell
uv sync --group sim
uv run mini-articraft simulate runs/<run-id>
```

```
2 bodies, 29.306 kg total
  lowest body: +0.0370 -> +0.0169 m
  contacts at rest: 8
  deepest penetration: -4.15 mm
  largest part separation change: +0.00 mm
  residual velocity: 0.0000
  verdict: stands up
```

Tilt the floor until it slides, which measures the friction its materials
declared instead of taking it on faith:

```shell
uv run mini-articraft simulate runs/<run-id> --scenario tilt --seconds 8
```

```
  slipped at: 42.3 deg of tilt
  friction: measured 0.91, authored 0.85
```

Let the joints fall from mid-travel, which is the motion an articulated object is
actually for:

```shell
uv run mini-articraft simulate runs/<run-id> --scenario release
```

```
  joints released from mid-travel
  peak joint speed: 6.20 per second
```

Every run records its motion, so `mini-articraft view` gains a **Play
simulation** switch that replays it in the same viewer used to pose joints.
MuJoCo is optional, so the `sim` group is not installed by default.

A passing run covers geometry, mass, joints, and sliding friction. It does not
cover restitution: MuJoCo has no such parameter, and static friction has nowhere
to go in its single sliding coefficient. Those values still export for engines
that read them.

### Run the checks

```shell
uv run pytest -q
uv run ruff check .
```

## Preview releases

Preview releases attach the exact wheel and sdist that the release workflow verified. Install a
wheel from the [releases page](https://github.com/mattzh72/mini-articraft/releases):

```shell
uv pip install https://github.com/mattzh72/mini-articraft/releases/download/<tag>/<wheel-file>
```

To record a preview as a project dependency instead, pin its release tag:

```shell
uv add "mini-articraft @ git+https://github.com/mattzh72/mini-articraft.git@<tag>"
```

Maintainers create releases with a manual workflow. Read the [release guide](docs/releasing.md).

## Docs

- [**Mesh authoring SDK**](docs/sdk.md)
- [**Agent design**](docs/agent.md)
- [**Release guide**](docs/releasing.md)
- [**Examples**](examples)
- [**Repository guide**](AGENTS.md)

This repository has an [Apache 2.0 License](LICENSE).

<sub>This project is based on the [Articraft paper](https://arxiv.org/abs/2605.15187).</sub>
