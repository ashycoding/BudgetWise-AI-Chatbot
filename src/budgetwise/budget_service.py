"""
BudgetWise AI - Budget Service Module.

Manages monthly category budgets, tracks actual expenses against limits,
and generates utilization percentages and threshold warnings.
"""

from typing import Any, Dict, List, Optional
from budgetwise.config import get_settings
from budgetwise.database import DatabaseManager
from budgetwise.models import (
    BudgetCreate,
    BudgetRecord,
    BudgetStatus,
    BudgetUpdate,
    paise_to_rupees,
)


class BudgetService:
    """
    Coordinates CRUD operations and deterministic spending tracking for category budgets.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None) -> None:
        self.settings = get_settings()
        self.db = db_manager or DatabaseManager()

    def set_budget(self, budget: BudgetCreate) -> BudgetRecord:
        """
        Create or update a category budget limit for a specific month (YYYY-MM).
        Uses SQLite upsert on (category, month, currency).
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO budgets (category, monthly_limit_paise, currency, month, created_at, updated_at)
                VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))
                ON CONFLICT(category, month, currency) DO UPDATE SET
                    monthly_limit_paise = excluded.monthly_limit_paise,
                    updated_at = datetime('now')
                RETURNING id, category, monthly_limit_paise, currency, month, created_at, updated_at;
                """,
                (
                    budget.category,
                    budget.monthly_limit_paise,
                    budget.currency.upper(),
                    budget.month,
                ),
            )
            row = cursor.fetchone()
            return BudgetRecord(
                id=row["id"],
                category=row["category"],
                monthly_limit_paise=row["monthly_limit_paise"],
                currency=row["currency"],
                month=row["month"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    def get_budget(self, budget_id: int) -> Optional[BudgetRecord]:
        """Retrieve a budget by its primary key ID."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, category, monthly_limit_paise, currency, month, created_at, updated_at
                FROM budgets WHERE id = ?;
                """,
                (budget_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return BudgetRecord(
                id=row["id"],
                category=row["category"],
                monthly_limit_paise=row["monthly_limit_paise"],
                currency=row["currency"],
                month=row["month"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    def get_budget_by_category_and_month(
        self, category: str, month: str, currency: str = "INR"
    ) -> Optional[BudgetRecord]:
        """Find budget by category and YYYY-MM month."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, category, monthly_limit_paise, currency, month, created_at, updated_at
                FROM budgets
                WHERE category = ? AND month = ? AND currency = ?;
                """,
                (category, month, currency.upper()),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return BudgetRecord(
                id=row["id"],
                category=row["category"],
                monthly_limit_paise=row["monthly_limit_paise"],
                currency=row["currency"],
                month=row["month"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    def list_budgets(
        self, month: Optional[str] = None, currency: Optional[str] = None
    ) -> List[BudgetRecord]:
        """List all budgets, optionally filtered by month or currency."""
        query = "SELECT id, category, monthly_limit_paise, currency, month, created_at, updated_at FROM budgets WHERE 1=1"
        params: List[Any] = []
        if month:
            query += " AND month = ?"
            params.append(month)
        if currency:
            query += " AND currency = ?"
            params.append(currency.upper())
        query += " ORDER BY month DESC, category ASC"

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [
                BudgetRecord(
                    id=r["id"],
                    category=r["category"],
                    monthly_limit_paise=r["monthly_limit_paise"],
                    currency=r["currency"],
                    month=r["month"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                )
                for r in rows
            ]

    def delete_budget(self, budget_id: int) -> bool:
        """Delete a budget limit by ID. Returns True if deleted."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM budgets WHERE id = ?;", (budget_id,))
            return cursor.rowcount > 0

    def get_budget_status(
        self, category: str, month: str, currency: str = "INR"
    ) -> Optional[BudgetStatus]:
        """
        Calculate actual net expenditure against configured limit for a category and month.
        Net expense = expense - refund. Income and transfers are excluded.
        """
        budget = self.get_budget_by_category_and_month(category, month, currency)
        if not budget:
            return None

        # Query net actual expense in this month
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    COALESCE(SUM(CASE WHEN transaction_type = 'expense' THEN amount_paise ELSE 0 END), 0) as total_expenses,
                    COALESCE(SUM(CASE WHEN transaction_type = 'refund' THEN amount_paise ELSE 0 END), 0) as total_refunds
                FROM transactions
                WHERE category = ?
                  AND date LIKE ?
                  AND currency = ?;
                """,
                (category, f"{month}%", currency.upper()),
            )
            row = cursor.fetchone()
            total_expenses = row["total_expenses"]
            total_refunds = row["total_refunds"]
            net_spent = max(0, total_expenses - total_refunds)

        limit_paise = budget.monthly_limit_paise
        remaining_paise = limit_paise - net_spent
        utilization_pct = round((net_spent / limit_paise) * 100.0, 2) if limit_paise > 0 else 0.0

        is_over = net_spent > limit_paise
        is_warn = (net_spent / limit_paise) >= self.settings.budget_warning_threshold if limit_paise > 0 else False

        if is_over:
            alert = "CRITICAL"
        elif is_warn:
            alert = "WARNING"
        else:
            alert = "NORMAL"

        return BudgetStatus(
            category=category,
            month=month,
            currency=currency.upper(),
            limit_paise=limit_paise,
            actual_spent_paise=net_spent,
            remaining_paise=remaining_paise,
            utilization_percentage=utilization_pct,
            is_warning=is_warn,
            is_over_budget=is_over,
            alert_level=alert,
        )

    def get_monthly_budget_summary(self, month: str, currency: str = "INR") -> Dict[str, Any]:
        """
        Compile complete budget status across all configured categories for the month.
        """
        budgets = self.list_budgets(month=month, currency=currency)
        statuses: List[BudgetStatus] = []
        total_limit_paise = 0
        total_spent_paise = 0

        for b in budgets:
            st = self.get_budget_status(b.category, month, currency)
            if st:
                statuses.append(st)
                total_limit_paise += st.limit_paise
                total_spent_paise += st.actual_spent_paise

        total_remaining = total_limit_paise - total_spent_paise
        overall_utilization = (
            round((total_spent_paise / total_limit_paise) * 100.0, 2)
            if total_limit_paise > 0
            else 0.0
        )

        warning_count = sum(1 for s in statuses if s.is_warning and not s.is_over_budget)
        over_budget_count = sum(1 for s in statuses if s.is_over_budget)

        return {
            "month": month,
            "currency": currency.upper(),
            "total_budget_limit_paise": total_limit_paise,
            "total_budget_limit_rupees": paise_to_rupees(total_limit_paise),
            "total_actual_spent_paise": total_spent_paise,
            "total_actual_spent_rupees": paise_to_rupees(total_spent_paise),
            "total_remaining_paise": total_remaining,
            "total_remaining_rupees": paise_to_rupees(total_remaining),
            "overall_utilization_percentage": overall_utilization,
            "category_statuses": statuses,
            "category_count": len(statuses),
            "warning_count": warning_count,
            "over_budget_count": over_budget_count,
        }
