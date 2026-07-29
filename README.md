<h1 align="center">mini-articraft</h1>

mini-articraft is a small agent that turns a prompt into an articulated 3D object.

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

The Anthropic provider supports `claude-sonnet-5` and `claude-opus-5`.
Each run keeps the complete Anthropic response blocks in `conversation.jsonl`.

Each run is in the `runs/` directory. Open a completed run in the browser viewer:

```shell
uv run mini-articraft view runs/<run-id>
```

Use the viewer to examine each generated version and move its joints.

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

Every run records its motion, so `mini-articraft view` gains a **Play
simulation** switch that replays it in the same viewer used to pose joints.
MuJoCo is optional, so the `sim` group is not installed by default.

### Run the checks

```shell
uv run pytest -q
uv run ruff check .
```

## Docs

- [**Mesh authoring SDK**](docs/sdk.md)
- [**Agent design**](docs/agent.md)
- [**Examples**](examples)
- [**Repository guide**](AGENTS.md)

This repository has an [Apache 2.0 License](LICENSE).

<sub>This project is based on the [Articraft paper](https://arxiv.org/abs/2605.15187).</sub>
