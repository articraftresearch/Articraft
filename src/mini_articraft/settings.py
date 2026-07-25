from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_OUTPUT_DIR = Path("runs")
DEFAULT_MAX_TURNS = 100
DEFAULT_COMPILE_TIMEOUT_SECONDS = 900.0
DEFAULT_OPENAI_MODEL = "gpt-5.5-2026-04-23"
DEFAULT_OPENAI_MAX_ATTEMPTS = 4
DEFAULT_OPENAI_REQUEST_TIMEOUT_SECONDS = 900.0
DEFAULT_PROVIDER = "openai"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_ANTHROPIC_MAX_ATTEMPTS = 4
DEFAULT_ANTHROPIC_REQUEST_TIMEOUT_SECONDS = 900.0
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_GEMINI_MAX_ATTEMPTS = 4
DEFAULT_GEMINI_REQUEST_TIMEOUT_SECONDS = 900.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MINI_ARTICRAFT_",
        extra="ignore",
        populate_by_name=True,
    )

    output_dir: Path = Field(
        default=DEFAULT_OUTPUT_DIR,
        validation_alias="MINI_ARTICRAFT_OUTPUT_DIR",
    )
    provider: Literal["openai", "gemini", "anthropic"] = Field(
        default=DEFAULT_PROVIDER,
        validation_alias="MINI_ARTICRAFT_PROVIDER",
    )
    openai_model: str = Field(
        default=DEFAULT_OPENAI_MODEL,
        validation_alias="MINI_ARTICRAFT_MODEL",
    )
    openai_reasoning_effort: str = Field(
        default="high",
        validation_alias="MINI_ARTICRAFT_REASONING_EFFORT",
    )
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_max_attempts: int = Field(
        default=DEFAULT_OPENAI_MAX_ATTEMPTS,
        ge=1,
        validation_alias="MINI_ARTICRAFT_OPENAI_MAX_ATTEMPTS",
    )
    openai_request_timeout_seconds: float = Field(
        default=DEFAULT_OPENAI_REQUEST_TIMEOUT_SECONDS,
        gt=0.0,
        validation_alias="MINI_ARTICRAFT_OPENAI_REQUEST_TIMEOUT_SECONDS",
    )
    anthropic_model: str = Field(
        default=DEFAULT_ANTHROPIC_MODEL,
        validation_alias="MINI_ARTICRAFT_ANTHROPIC_MODEL",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias="ANTHROPIC_API_KEY",
    )
    anthropic_max_attempts: int = Field(
        default=DEFAULT_ANTHROPIC_MAX_ATTEMPTS,
        ge=1,
        validation_alias="MINI_ARTICRAFT_ANTHROPIC_MAX_ATTEMPTS",
    )
    anthropic_request_timeout_seconds: float = Field(
        default=DEFAULT_ANTHROPIC_REQUEST_TIMEOUT_SECONDS,
        gt=0.0,
        validation_alias="MINI_ARTICRAFT_ANTHROPIC_REQUEST_TIMEOUT_SECONDS",
    )
    gemini_model: str = Field(
        default=DEFAULT_GEMINI_MODEL,
        validation_alias="MINI_ARTICRAFT_GEMINI_MODEL",
    )
    gemini_api_key: str | None = Field(
        default=None,
        validation_alias="GEMINI_API_KEY",
    )
    gemini_max_attempts: int = Field(
        default=DEFAULT_GEMINI_MAX_ATTEMPTS,
        ge=1,
        validation_alias="MINI_ARTICRAFT_GEMINI_MAX_ATTEMPTS",
    )
    gemini_request_timeout_seconds: float = Field(
        default=DEFAULT_GEMINI_REQUEST_TIMEOUT_SECONDS,
        gt=0.0,
        validation_alias="MINI_ARTICRAFT_GEMINI_REQUEST_TIMEOUT_SECONDS",
    )
    max_turns: int = Field(default=DEFAULT_MAX_TURNS, validation_alias="MINI_ARTICRAFT_MAX_TURNS")
    compile_timeout_seconds: float = Field(
        default=DEFAULT_COMPILE_TIMEOUT_SECONDS,
        gt=0.0,
        validation_alias="MINI_ARTICRAFT_COMPILE_TIMEOUT_SECONDS",
    )

    @property
    def selected_model(self) -> str:
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
