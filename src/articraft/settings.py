from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_OUTPUT_DIR = Path("runs")
DEFAULT_MAX_TURNS = 100
DEFAULT_COMPILE_TIMEOUT_SECONDS = 900.0
DEFAULT_OPENAI_MODEL = "gpt-6-astra"
DEFAULT_OPENAI_MAX_ATTEMPTS = 4
DEFAULT_OPENAI_REQUEST_TIMEOUT_SECONDS = 900.0
DEFAULT_PROVIDER = "openai"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_ANTHROPIC_MAX_ATTEMPTS = 4
DEFAULT_ANTHROPIC_REQUEST_TIMEOUT_SECONDS = 3600.0
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_GEMINI_MAX_ATTEMPTS = 4
DEFAULT_GEMINI_REQUEST_TIMEOUT_SECONDS = 900.0
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_OPENROUTER_MAX_ATTEMPTS = 4
DEFAULT_OPENROUTER_REQUEST_TIMEOUT_SECONDS = 900.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ARTICRAFT_",
        extra="ignore",
        populate_by_name=True,
    )

    output_dir: Path = Field(
        default=DEFAULT_OUTPUT_DIR,
        validation_alias="ARTICRAFT_OUTPUT_DIR",
    )
    provider: Literal["openai", "gemini", "anthropic", "openrouter"] = Field(
        default=DEFAULT_PROVIDER,
        validation_alias="ARTICRAFT_PROVIDER",
    )
    openai_model: str = Field(
        default=DEFAULT_OPENAI_MODEL,
        validation_alias="ARTICRAFT_MODEL",
    )
    openai_reasoning_effort: str = Field(
        default="high",
        validation_alias="ARTICRAFT_REASONING_EFFORT",
    )
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_max_attempts: int = Field(
        default=DEFAULT_OPENAI_MAX_ATTEMPTS,
        ge=1,
        validation_alias="ARTICRAFT_OPENAI_MAX_ATTEMPTS",
    )
    openai_request_timeout_seconds: float = Field(
        default=DEFAULT_OPENAI_REQUEST_TIMEOUT_SECONDS,
        gt=0.0,
        validation_alias="ARTICRAFT_OPENAI_REQUEST_TIMEOUT_SECONDS",
    )
    anthropic_model: str = Field(
        default=DEFAULT_ANTHROPIC_MODEL,
        validation_alias="ARTICRAFT_ANTHROPIC_MODEL",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias="ANTHROPIC_API_KEY",
    )
    anthropic_max_attempts: int = Field(
        default=DEFAULT_ANTHROPIC_MAX_ATTEMPTS,
        ge=1,
        validation_alias="ARTICRAFT_ANTHROPIC_MAX_ATTEMPTS",
    )
    anthropic_request_timeout_seconds: float = Field(
        default=DEFAULT_ANTHROPIC_REQUEST_TIMEOUT_SECONDS,
        gt=0.0,
        validation_alias="ARTICRAFT_ANTHROPIC_REQUEST_TIMEOUT_SECONDS",
    )
    gemini_model: str = Field(
        default=DEFAULT_GEMINI_MODEL,
        validation_alias="ARTICRAFT_GEMINI_MODEL",
    )
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias="GEMINI_API_KEY",
    )
    gemini_max_attempts: int = Field(
        default=DEFAULT_GEMINI_MAX_ATTEMPTS,
        ge=1,
        validation_alias="ARTICRAFT_GEMINI_MAX_ATTEMPTS",
    )
    gemini_request_timeout_seconds: float = Field(
        default=DEFAULT_GEMINI_REQUEST_TIMEOUT_SECONDS,
        gt=0.0,
        validation_alias="ARTICRAFT_GEMINI_REQUEST_TIMEOUT_SECONDS",
    )
    openrouter_model: str = Field(
        default=DEFAULT_OPENROUTER_MODEL,
        validation_alias="ARTICRAFT_OPENROUTER_MODEL",
    )
    openrouter_api_key: str | None = Field(
        default=None,
        validation_alias="OPENROUTER_API_KEY",
    )
    openrouter_max_attempts: int = Field(
        default=DEFAULT_OPENROUTER_MAX_ATTEMPTS,
        ge=1,
        validation_alias="ARTICRAFT_OPENROUTER_MAX_ATTEMPTS",
    )
    openrouter_request_timeout_seconds: float = Field(
        default=DEFAULT_OPENROUTER_REQUEST_TIMEOUT_SECONDS,
        gt=0.0,
        validation_alias="ARTICRAFT_OPENROUTER_REQUEST_TIMEOUT_SECONDS",
    )
    openrouter_context_window_tokens: int = Field(
        default=0,
        ge=0,
        validation_alias="ARTICRAFT_OPENROUTER_CONTEXT_WINDOW_TOKENS",
    )
    openrouter_summary_max_output_tokens: int = Field(
        default=8_192,
        ge=1,
        validation_alias="ARTICRAFT_OPENROUTER_SUMMARY_MAX_OUTPUT_TOKENS",
    )

    @field_validator("openrouter_context_window_tokens")
    @classmethod
    def _openrouter_window_floor(cls, value: int) -> int:
        """Reject windows below RESERVE_TOKENS + KEEP_RECENT_TOKENS: they look set but never protect."""
        if 0 < value < 36_384:
            raise ValueError(
                "ARTICRAFT_OPENROUTER_CONTEXT_WINDOW_TOKENS must be 0 (disabled) or at least 36384"
            )
        return value

    openrouter_http_referer: str | None = Field(
        default=None,
        validation_alias="OPENROUTER_HTTP_REFERER",
    )
    openrouter_app_title: str | None = Field(
        default=None,
        validation_alias="OPENROUTER_APP_TITLE",
    )
    max_turns: int = Field(default=DEFAULT_MAX_TURNS, validation_alias="ARTICRAFT_MAX_TURNS")
    physics_enabled: bool = Field(
        default=False,
        validation_alias="ARTICRAFT_PHYSICS",
    )
    compile_timeout_seconds: float = Field(
        default=DEFAULT_COMPILE_TIMEOUT_SECONDS,
        gt=0.0,
        validation_alias="ARTICRAFT_COMPILE_TIMEOUT_SECONDS",
    )

    @property
    def selected_model(self) -> str:
        if self.provider == "openrouter":
            return self.openrouter_model
        if self.provider == "anthropic":
            return self.anthropic_model
        if self.provider == "gemini":
            return self.gemini_model
        return self.openai_model

    @property
    def selected_reasoning_effort(self) -> str:
        if self.provider == "openai":
            return self.openai_reasoning_effort
        return ""


@cache
def get_settings() -> Settings:
    return Settings()  # pyright: ignore[reportCallIssue]
