"""
Automated Tests for Configuration, Models, and Database Initialization (Stage 1).
"""

import os
import sqlite3
import pytest
from budgetwise.config import Settings, OpenAIKeyMissingError
from budgetwise.database import DatabaseManager
from budgetwise.models import (
    TransactionCreate,
    TransactionType,
    StandardCategory,
    rupees_to_paise,
    paise_to_rupees,
    BudgetCreate,
)


def test_rupees_to_paise_conversion():
    """Verify that rupees are accurately converted to integer paise without float inaccuracies."""
    assert rupees_to_paise(450) == 45000
    assert rupees_to_paise(450.50) == 45050
    assert rupees_to_paise("99.99") == 9999
    assert rupees_to_paise(0.05) == 5
    assert paise_to_rupees(45050) == 450.50
    assert paise_to_rupees(9999) == 99.99


def test_config_graceful_missing_openai_key(monkeypatch):
    """Verify that accessing OpenAI client without an API key raises a clear OpenAIKeyMissingError."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(openai_api_key=None)
    assert not settings.is_openai_configured()
    with pytest.raises(OpenAIKeyMissingError) as exc_info:
        settings.get_openai_client()
    assert "OpenAI API key is missing" in str(exc_info.value)


def test_config_default_currency_and_model(monkeypatch):
    """Verify default configuration values."""
    settings = Settings()
    assert settings.default_currency == "INR"
    assert settings.openai_model == "gpt-4o-mini"
    assert settings.categorizer_confidence_threshold == 0.70


def test_database_initialization_and_validation(tmp_path):
    """Verify that DatabaseManager initializes and validates the SQLite schema."""
    db_file = tmp_path / "test_budgetwise.db"
    db_mgr = DatabaseManager(db_path=db_file)
    db_mgr.init_schema()

    assert db_mgr.validate_schema() is True
    assert db_file.exists()


def test_database_enforces_positive_amount_constraint(tmp_path):
    """Verify that CHECK constraint prevents negative or zero transaction amounts."""
    db_file = tmp_path / "test_constraint.db"
    db_mgr = DatabaseManager(db_path=db_file)
    db_mgr.init_schema()

    with pytest.raises(sqlite3.IntegrityError):
        with db_mgr.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO transactions (date, description, amount_paise, currency, transaction_type, category)
                VALUES ('2026-10-15', 'Bad Amount', -500, 'INR', 'expense', 'Other');
                """
            )


def test_transaction_model_validation():
    """Verify Pydantic model validation on transactions."""
    valid_txn = TransactionCreate(
        date="2026-10-15",
        description="Swiggy Order",
        amount_paise=45000,
        currency="INR",
        transaction_type=TransactionType.EXPENSE,
        category=StandardCategory.FOOD_AND_DINING.value,
        notes="Dinner",
    )
    assert valid_txn.amount_paise == 45000
    assert valid_txn.currency == "INR"

    # Invalid ISO date
    with pytest.raises(ValueError):
        TransactionCreate(
            date="15-10-2026",
            description="Bad Date",
            amount_paise=1000,
            category="Other",
        )

    # Invalid amount <= 0
    with pytest.raises(ValueError):
        TransactionCreate(
            date="2026-10-15",
            description="Negative Amount",
            amount_paise=0,
            category="Other",
        )


def test_budget_model_validation():
    """Verify Pydantic model validation on budgets."""
    valid_budget = BudgetCreate(
        category=StandardCategory.GROCERIES.value,
        monthly_limit_paise=1000000,
        month="2026-10",
        currency="INR",
    )
    assert valid_budget.month == "2026-10"

    # Invalid month format
    with pytest.raises(ValueError):
        BudgetCreate(
            category="Groceries",
            monthly_limit_paise=100000,
            month="2026-13",
        )
