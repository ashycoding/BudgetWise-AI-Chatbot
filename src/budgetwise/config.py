"""
BudgetWise AI - Centralized Configuration Module.

Manages environment configuration, path resolutions, operational thresholds,
and currency defaults using Pydantic Settings and python-dotenv.
Fails gracefully with clear actionable guidance when credentials are missing.
"""

from functools import lru_cache
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Explicitly load .env file from the project directory if present
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class OpenAIKeyMissingError(RuntimeError):
    """Raised when an operation requiring OpenAI is invoked without an API key."""

    def __init__(
        self,
        message: str = (
            "OpenAI API key is missing. Please set OPENAI_API_KEY in your .env file "
            "or environment variables to enable conversational and AI features."
        ),
    ) -> None:
        super().__init__(message)


class Settings(BaseSettings):
    """
    Central application settings and environment variable validation.
    """

    # OpenAI API Configuration
    openai_api_key: Optional[str] = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
        description="OpenAI API key for language model orchestration.",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        validation_alias="OPENAI_MODEL",
        description="Configurable OpenAI model identifier.",
    )

    # Persistence / Database
    database_path: Path = Field(
        default=_PROJECT_ROOT / "data" / "budgetwise.db",
        validation_alias="DATABASE_PATH",
        description="Path to persistent SQLite database file.",
    )

    # Application Environment
    app_env: str = Field(
        default="development",
        validation_alias="APP_ENV",
        description="Application lifecycle environment (e.g. development, test, production).",
    )
    log_level: str = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
        description="Logging level for standard output handlers.",
    )

    # Currency Setting
    default_currency: str = Field(
        default="INR",
        validation_alias="DEFAULT_CURRENCY",
        description="Default currency code for financial transactions and budgets.",
    )

    # Expense Categorizer Configuration
    categorizer_confidence_threshold: float = Field(
        default=0.70,
        validation_alias="CATEGORIZER_CONFIDENCE_THRESHOLD",
        description="Minimum score (0.0 to 1.0) before categorizer falls back to requesting confirmation.",
    )

    # Budget Warning Thresholds
    budget_warning_threshold: float = Field(
        default=0.80,
        validation_alias="BUDGET_WARNING_THRESHOLD",
        description="Utilization percentage (e.g. 0.80 = 80%) that triggers a budget warning.",
    )
    budget_critical_threshold: float = Field(
        default=1.00,
        validation_alias="BUDGET_CRITICAL_THRESHOLD",
        description="Utilization percentage (e.g. 1.00 = 100%) that triggers an over-budget alert.",
    )

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def is_openai_configured(self) -> bool:
        """Verify whether an OpenAI API key is present and non-empty."""
        return bool(self.openai_api_key and self.openai_api_key.strip())

    def get_openai_client(self):
        """
        Return an authenticated OpenAI client or raise OpenAIKeyMissingError.
        Fails gracefully without crashing startup when key is not yet provided.
        """
        if not self.is_openai_configured():
            raise OpenAIKeyMissingError()
        from openai import OpenAI

        return OpenAI(api_key=self.openai_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
