# Model providers

Articraft supports OpenAI, Anthropic, Gemini, and OpenRouter. OpenAI is the default.

## Configure a provider

Set the API key for the provider that you want to use:

| Provider | API key | Default model | Reference images |
| --- | --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | `gpt-6-astra` | Yes |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-5` | Yes |
| Gemini | `GEMINI_API_KEY` | `gemini-3.6-flash` | Yes |
| OpenRouter | `OPENROUTER_API_KEY` | `nvidia/nemotron-3-ultra-550b-a55b:free` | No |

You can put the key in `.env` or set it for one command. Do not commit API keys.

Select a provider with `--provider`:

```shell
ANTHROPIC_API_KEY=your_key_here uv run articraft \
  --provider anthropic "a folding chair"
```

## Select a model

Use `--model` to replace the default model:

```shell
GEMINI_API_KEY=your_key_here uv run articraft \
  --provider gemini --model gemini-3.6-flash "a folding chair"
```

Articraft passes an unknown model name to the selected provider. The provider returns an
error if it does not accept the name.

The live interface cannot estimate cost or context use for an unknown model. The run can
still continue if the provider accepts the model.

## Use OpenRouter

OpenRouter accepts text prompts only. Do not pass `--image` with this provider.

Set `OPENROUTER_HTTP_REFERER` and `OPENROUTER_APP_TITLE` if you want OpenRouter attribution.
These values are optional.

OpenRouter reports token use and request cost when its API returns them. Articraft does not
keep a context window catalog for OpenRouter models.

Set `ARTICRAFT_OPENROUTER_CONTEXT_WINDOW_TOKENS` to the selected model's context window to
enable conversation compaction — the number is on the model's OpenRouter page, or in
`GET /api/v1/models` as `context_length`. Articraft replaces older turns with a compact
summary when the conversation approaches that budget, and the live interface shows context
use against it. When the value is unset, Articraft does not know the window and never
compacts. Values between 1 and 36383 are rejected.

The model's maximum output size is separate from its context window. If it is below
8192, set `ARTICRAFT_OPENROUTER_SUMMARY_MAX_OUTPUT_TOKENS` to that limit or lower.
This setting defaults to 8192 and must be positive. It caps summary requests only.
A smaller limit requested by the agent is still respected. It does not change ordinary
generation requests.

## Use the Python API

Pass the same provider and model names to `generate()` or `generate_async()`:

```python
result = articraft.generate(
    "a folding chair",
    provider="anthropic",
    model="claude-sonnet-5",
)
```

The function checks the required API key before it starts the run.
