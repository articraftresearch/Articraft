from __future__ import annotations

from mini_articraft import Model
from mini_articraft.models.anthropic import AnthropicModel
from mini_articraft.models.anthropic import (
    context_window_tokens_for as anthropic_context_window_tokens_for,
)
from mini_articraft.models.gemini import GeminiModel
from mini_articraft.models.gemini import (
    context_window_tokens_for as gemini_context_window_tokens_for,
)
from mini_articraft.models.openai import OpenAIModel
from mini_articraft.models.openai import (
    context_window_tokens_for as openai_context_window_tokens_for,
)
from mini_articraft.settings import Settings


def create_model(settings: Settings) -> Model:
    if settings.provider == "anthropic":
        return AnthropicModel(settings)
    if settings.provider == "gemini":
        return GeminiModel(settings)
    return OpenAIModel(settings)


def context_window_tokens_for(model: str) -> int | None:
    return (
        openai_context_window_tokens_for(model)
        or gemini_context_window_tokens_for(model)
        or anthropic_context_window_tokens_for(model)
    )


__all__ = [
    "AnthropicModel",
    "GeminiModel",
    "OpenAIModel",
    "context_window_tokens_for",
    "create_model",
]
