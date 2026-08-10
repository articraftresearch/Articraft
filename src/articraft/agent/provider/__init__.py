from __future__ import annotations

from articraft.agent.protocols import Model
from articraft.agent.provider.anthropic import AnthropicModel
from articraft.agent.provider.anthropic import (
    context_window_tokens_for as anthropic_context_window_tokens_for,
)
from articraft.agent.provider.gemini import GeminiModel
from articraft.agent.provider.gemini import (
    context_window_tokens_for as gemini_context_window_tokens_for,
)
from articraft.agent.provider.openai import OpenAIModel
from articraft.agent.provider.openai import (
    context_window_tokens_for as openai_context_window_tokens_for,
)
from articraft.agent.provider.openrouter import OpenRouterModel
from articraft.settings import Settings


def create_model(settings: Settings) -> Model:
    if settings.provider == "openrouter":
        return OpenRouterModel(settings)
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
    "OpenRouterModel",
    "context_window_tokens_for",
    "create_model",
]
