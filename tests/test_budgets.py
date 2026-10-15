"""
Automated Tests for Budget Service Module (Stage 3).
"""

import pytest
from budgetwise.budget_service import BudgetService
from budgetwise.database import DatabaseManager
from budgetwise.models import (
    BudgetCreate,
    StandardCategory,
    TransactionCreate,
    TransactionType,
    rupees_to_paise,
)
from budgetwise.transaction_service import TransactionService


@pytest.fixture
def services(tmp_path):
    """Fixture providing BudgetService and TransactionService connected to an isolated test DB."""
    db_file = tmp_path / "test_budgets.db"
    db_mgr = DatabaseManager(db_path=db_file)
    db_mgr.init_schema()
    txn_service = TransactionService(db_manager=db_mgr)
    bgt_service = BudgetService(db_manager=db_mgr)
    return bgt_service, txn_service


def test_create_and_update_budget(services):
    """Test creating and upserting a category budget limit."""
    bgt_service, _ = services
    budget = bgt_service.set_budget(
        BudgetCreate(
            category=StandardCategory.GROCERIES.value,
            monthly_limit_paise=rupees_to_paise(10000),
            month="2026-10",
            currency="INR",
        )
    )
    assert budget.id is not None
    assert budget.monthly_limit_rupees == 10000.0

    # Modify limit for same category and month
    updated = bgt_service.set_budget(
        BudgetCreate(
            category=StandardCategory.GROCERIES.value,
            monthly_limit_paise=rupees_to_paise(12000),
            month="2026-10",
            currency="INR",
        )
    )
    assert updated.id == budget.id
    assert updated.monthly_limit_rupees == 12000.0


def test_budget_utilization_and_warning_thresholds(services):
    """Test spending calculation, remaining budget, and warning/critical alerts."""
    bgt_service, txn_service = services
    # Set Dining budget of ₹5,000 for 2026-10
    bgt_service.set_budget(
        BudgetCreate(
            category=StandardCategory.FOOD_AND_DINING.value,
            monthly_limit_paise=rupees_to_paise(5000),
            month="2026-10",
        )
    )

    # 1. Normal state: Spend ₹2,000 (40% utilization)
    txn_service.add_transaction(
        TransactionCreate(
            date="2026-10-05",
            description="Dinner",
            amount_paise=rupees_to_paise(2000),
            category=StandardCategory.FOOD_AND_DINING.value,
        )
    )
    status = bgt_service.get_budget_status(StandardCategory.FOOD_AND_DINING.value, "2026-10")
    assert status.actual_spent_rupees == 2000.0
    assert status.remaining_rupees == 3000.0
    assert status.utilization_percentage == 40.0
    assert status.alert_level == "NORMAL"
    assert status.is_warning is False
    assert status.is_over_budget is False

    # 2. Warning state: Spend ₹2,200 more -> Total ₹4,200 (84% utilization >= 80%)
    txn_service.add_transaction(
        TransactionCreate(
            date="2026-10-10",
            description="Party Dineout",
            amount_paise=rupees_to_paise(2200),
            category=StandardCategory.FOOD_AND_DINING.value,
        )
    )
    status_warn = bgt_service.get_budget_status(StandardCategory.FOOD_AND_DINING.value, "2026-10")
    assert status_warn.actual_spent_rupees == 4200.0
    assert status_warn.remaining_rupees == 800.0
    assert status_warn.utilization_percentage == 84.0
    assert status_warn.alert_level == "WARNING"
    assert status_warn.is_warning is True
    assert status_warn.is_over_budget is False

    # 3. Critical state: Spend ₹1,000 more -> Total ₹5,200 (104% utilization > 100%)
    txn_service.add_transaction(
        TransactionCreate(
            date="2026-10-14",
            description="Cafe snacks",
            amount_paise=rupees_to_paise(1000),
            category=StandardCategory.FOOD_AND_DINING.value,
        )
    )
    status_crit = bgt_service.get_budget_status(StandardCategory.FOOD_AND_DINING.value, "2026-10")
    assert status_crit.actual_spent_rupees == 5200.0
    assert status_crit.remaining_rupees == -200.0
    assert status_crit.utilization_percentage == 104.0
    assert status_crit.alert_level == "CRITICAL"
    assert status_crit.is_over_budget is True


def test_refund_reduces_budget_spending(services):
    """Test that refunds reduce net spent amount in budget tracking."""
    bgt_service, txn_service = services
    bgt_service.set_budget(
        BudgetCreate(
            category=StandardCategory.SHOPPING.value,
            monthly_limit_paise=rupees_to_paise(10000),
            month="2026-10",
        )
    )

    # Initial purchase: ₹3,000
    txn_service.add_transaction(
        TransactionCreate(
            date="2026-10-02",
            description="Amazon Jacket",
            amount_paise=rupees_to_paise(3000),
            transaction_type=TransactionType.EXPENSE,
            category=StandardCategory.SHOPPING.value,
        )
    )
    # Return / Refund: ₹1,000
    txn_service.add_transaction(
        TransactionCreate(
            date="2026-10-04",
            description="Amazon Jacket Return Refund",
            amount_paise=rupees_to_paise(1000),
            transaction_type=TransactionType.REFUND,
            category=StandardCategory.SHOPPING.value,
        )
    )

    status = bgt_service.get_budget_status(StandardCategory.SHOPPING.value, "2026-10")
    assert status.actual_spent_rupees == 2000.0
    assert status.remaining_rupees == 8000.0
    assert status.utilization_percentage == 20.0


def test_overall_monthly_budget_summary(services):
    """Test aggregate multi-category monthly summary."""
    bgt_service, txn_service = services
    bgt_service.set_budget(
        BudgetCreate(category=StandardCategory.GROCERIES.value, monthly_limit_paise=rupees_to_paise(8000), month="2026-10")
    )
    bgt_service.set_budget(
        BudgetCreate(category=StandardCategory.UTILITIES.value, monthly_limit_paise=rupees_to_paise(2000), month="2026-10")
    )

    txn_service.add_transaction(
        TransactionCreate(date="2026-10-01", description="Groceries", amount_paise=rupees_to_paise(4000), category=StandardCategory.GROCERIES.value)
    )

    summary = bgt_service.get_monthly_budget_summary("2026-10")
    assert summary["total_budget_limit_rupees"] == 10000.0
    assert summary["total_actual_spent_rupees"] == 4000.0
    assert summary["total_remaining_rupees"] == 6000.0
    assert summary["overall_utilization_percentage"] == 40.0
    assert summary["category_count"] == 2
