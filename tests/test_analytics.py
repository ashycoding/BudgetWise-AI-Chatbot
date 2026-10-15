"""
Automated Tests for Financial Analytics Module (Stage 3).
"""

import pandas as pd
import pytest
from budgetwise.analytics import FinancialAnalytics
from budgetwise.models import rupees_to_paise


@pytest.fixture
def sample_transactions_df():
    """Fixture providing a known multi-month synthetic transactions DataFrame."""
    records = [
        # September 2026
        {"date": "2026-09-01", "description": "Salary Sept", "amount_paise": rupees_to_paise(60000), "currency": "INR", "transaction_type": "income", "category": "Income"},
        {"date": "2026-09-05", "description": "House Rent", "amount_paise": rupees_to_paise(15000), "currency": "INR", "transaction_type": "expense", "category": "Housing and Rent"},
        {"date": "2026-09-10", "description": "Groceries DMart", "amount_paise": rupees_to_paise(6000), "currency": "INR", "transaction_type": "expense", "category": "Groceries"},
        {"date": "2026-09-15", "description": "Swiggy", "amount_paise": rupees_to_paise(2000), "currency": "INR", "transaction_type": "expense", "category": "Food and Dining"},
        {"date": "2026-09-20", "description": "Netflix", "amount_paise": rupees_to_paise(649), "currency": "INR", "transaction_type": "expense", "category": "Subscriptions"},

        # October 2026
        {"date": "2026-10-01", "description": "Salary Oct", "amount_paise": rupees_to_paise(60000), "currency": "INR", "transaction_type": "income", "category": "Income"},
        {"date": "2026-10-05", "description": "House Rent", "amount_paise": rupees_to_paise(15000), "currency": "INR", "transaction_type": "expense", "category": "Housing and Rent"},
        {"date": "2026-10-10", "description": "Groceries BigBasket", "amount_paise": rupees_to_paise(7500), "currency": "INR", "transaction_type": "expense", "category": "Groceries"},
        {"date": "2026-10-12", "description": "Shopping Festival Surge", "amount_paise": rupees_to_paise(14000), "currency": "INR", "transaction_type": "expense", "category": "Shopping"},
        {"date": "2026-10-15", "description": "Swiggy", "amount_paise": rupees_to_paise(3000), "currency": "INR", "transaction_type": "expense", "category": "Food and Dining"},
        {"date": "2026-10-20", "description": "Netflix", "amount_paise": rupees_to_paise(649), "currency": "INR", "transaction_type": "expense", "category": "Subscriptions"},
        {"date": "2026-10-25", "description": "Shopping Return", "amount_paise": rupees_to_paise(2000), "currency": "INR", "transaction_type": "refund", "category": "Shopping"},
    ]
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_calculate_summary_totals(sample_transactions_df):
    """Verify exact income, expense, and savings calculations on sample DataFrame."""
    # October only
    oct_df = sample_transactions_df[sample_transactions_df["date"] >= "2026-10-01"]
    summary = FinancialAnalytics.calculate_summary(oct_df, period_label="October 2026", currency="INR")

    # Income = 60,000
    assert summary["total_income_rupees"] == 60000.0

    # Gross Expenses = 15000 (Rent) + 7500 (Groceries) + 14000 (Shopping) + 3000 (Dining) + 649 (Netflix) = 40149
    # Refund = 2000
    # Net Expenses = 40149 - 2000 = 38149
    assert summary["total_expense_rupees"] == 38149.0

    # Net Savings = 60000 - 38149 = 21851
    assert summary["net_savings_rupees"] == 21851.0
    assert summary["savings_rate_percentage"] == round((21851 / 60000) * 100, 2)


def test_category_breakdown(sample_transactions_df):
    """Verify category totals and percentages."""
    sept_df = sample_transactions_df[sample_transactions_df["date"] < "2026-10-01"]
    breakdown = FinancialAnalytics.category_breakdown(sept_df, currency="INR")

    # Total Sept expenses = 15000 + 6000 + 2000 + 649 = 23649
    assert breakdown["total_spent_rupees"] == 23649.0
    cats = {c["category"]: c["total_rupees"] for c in breakdown["categories"]}
    assert cats["Housing and Rent"] == 15000.0
    assert cats["Groceries"] == 6000.0
    assert cats["Food and Dining"] == 2000.0
    assert cats["Subscriptions"] == 649.0


def test_compare_months(sample_transactions_df):
    """Verify month-over-month differences and top increasing category."""
    comparison = FinancialAnalytics.compare_months(
        sample_transactions_df,
        month_a="2026-09",
        month_b="2026-10",
        currency="INR",
        month_b_is_partial=False,
    )
    assert comparison["month_a"] == "2026-09"
    assert comparison["month_b"] == "2026-10"
    # Net expense: Sept was 23,649, Oct was 38,149. Diff = 14,500
    assert comparison["expense_diff_rupees"] == 14500.0
    assert comparison["highest_increase_category"] is not None
    # Shopping had 0 in Sept and 14,000 in Oct (highest increase)
    assert comparison["highest_increase_category"]["category"] == "Shopping"


def test_compare_months_flags_partial_period(sample_transactions_df):
    """Verify unequal period warning is issued when month_b is partial."""
    comparison = FinancialAnalytics.compare_months(
        sample_transactions_df,
        month_a="2026-09",
        month_b="2026-10",
        month_b_is_partial=True,
    )
    assert comparison["month_b_is_partial"] is True
    assert "partial billing period" in comparison["period_warning"]


def test_detect_spending_anomalies_neutral_language(sample_transactions_df):
    """Verify anomaly detection flags spikes using neutral, objective terminology."""
    anomalies = FinancialAnalytics.detect_spending_anomalies(
        sample_transactions_df,
        current_month="2026-10",
        currency="INR",
        multiplier_threshold=1.5,
    )
    detected = anomalies["anomalies_detected"]
    assert len(detected) > 0

    # Verify neutral terminology
    for a in detected:
        assert "Unusual increase worth reviewing" in a["status"]
        assert "fraud" not in a["status"].lower()


def test_find_recurring_candidates(sample_transactions_df):
    """Verify detection of subscriptions and recurring monthly payments."""
    recurring = FinancialAnalytics.find_recurring_candidates(sample_transactions_df)
    descriptions = [r["description"] for r in recurring]
    assert "Netflix" in descriptions
    assert "House Rent" in descriptions
