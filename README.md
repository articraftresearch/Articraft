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
