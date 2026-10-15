"""
BudgetWise AI - Financial Analytics Module.

Executes deterministic calculations and statistical aggregations using pandas.
Calculates income, expenses, net savings, category distributions, monthly comparisons,
and detects unusual spending surges using neutral, objective terminology.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from budgetwise.models import paise_to_rupees, rupees_to_paise


class FinancialAnalytics:
    """
    Deterministic financial engine powered by pandas.
    Guarantees arithmetic precision without language-model extrapolation.
    """

    @staticmethod
    def calculate_summary(
        df: pd.DataFrame, period_label: str = "All Time", currency: str = "INR"
    ) -> Dict[str, Any]:
        """
        Calculate total income, total expenses, net cash flow, and savings rate.
        Refunds offset expenses; transfers and investments are tracked separately.
        """
        if df.empty:
            return {
                "period": period_label,
                "currency": currency.upper(),
                "total_income_paise": 0,
                "total_income_rupees": 0.0,
                "total_expense_paise": 0,
                "total_expense_rupees": 0.0,
                "net_savings_paise": 0,
                "net_savings_rupees": 0.0,
                "savings_rate_percentage": 0.0,
                "transaction_count": 0,
                "income_transaction_count": 0,
                "expense_transaction_count": 0,
            }

        # Filter currency to prevent mixing
        curr_df = df[df["currency"].str.upper() == currency.upper()].copy()
        if curr_df.empty:
            return {
                "period": period_label,
                "currency": currency.upper(),
                "total_income_paise": 0,
                "total_income_rupees": 0.0,
                "total_expense_paise": 0,
                "total_expense_rupees": 0.0,
                "net_savings_paise": 0,
                "net_savings_rupees": 0.0,
                "savings_rate_percentage": 0.0,
                "transaction_count": 0,
                "income_transaction_count": 0,
                "expense_transaction_count": 0,
            }

        income_mask = curr_df["transaction_type"] == "income"
        expense_mask = curr_df["transaction_type"] == "expense"
        refund_mask = curr_df["transaction_type"] == "refund"

        total_income_paise = int(curr_df.loc[income_mask, "amount_paise"].sum())
        gross_expense_paise = int(curr_df.loc[expense_mask, "amount_paise"].sum())
        total_refund_paise = int(curr_df.loc[refund_mask, "amount_paise"].sum())

        net_expense_paise = max(0, gross_expense_paise - total_refund_paise)
        net_savings_paise = total_income_paise - net_expense_paise

        savings_rate = (
            round((net_savings_paise / total_income_paise) * 100.0, 2)
            if total_income_paise > 0
            else 0.0
        )

        return {
            "period": period_label,
            "currency": currency.upper(),
            "total_income_paise": total_income_paise,
            "total_income_rupees": paise_to_rupees(total_income_paise),
            "total_expense_paise": net_expense_paise,
            "total_expense_rupees": paise_to_rupees(net_expense_paise),
            "net_savings_paise": net_savings_paise,
            "net_savings_rupees": paise_to_rupees(net_savings_paise),
            "savings_rate_percentage": savings_rate,
            "transaction_count": len(curr_df),
            "income_transaction_count": int(income_mask.sum()),
            "expense_transaction_count": int(expense_mask.sum()),
        }

    @staticmethod
    def category_breakdown(df: pd.DataFrame, currency: str = "INR") -> Dict[str, Any]:
        """
        Aggregate spending by category, computing totals and percentages.
        """
        if df.empty:
            return {"currency": currency.upper(), "categories": [], "total_spent_rupees": 0.0}

        curr_df = df[df["currency"].str.upper() == currency.upper()].copy()
        expense_df = curr_df[curr_df["transaction_type"] == "expense"]

        if expense_df.empty:
            return {"currency": currency.upper(), "categories": [], "total_spent_rupees": 0.0}

        grouped = (
            expense_df.groupby("category")["amount_paise"]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": "total_paise", "count": "transaction_count"})
        )

        total_expense_paise = grouped["total_paise"].sum()
        grouped["total_rupees"] = grouped["total_paise"].apply(paise_to_rupees)
        grouped["percentage"] = (
            (grouped["total_paise"] / total_expense_paise * 100.0).round(2)
            if total_expense_paise > 0
            else 0.0
        )

        grouped = grouped.sort_values(by="total_paise", ascending=False)
        categories = grouped.to_dict(orient="records")

        return {
            "currency": currency.upper(),
            "total_spent_paise": int(total_expense_paise),
            "total_spent_rupees": paise_to_rupees(int(total_expense_paise)),
            "category_count": len(categories),
            "categories": categories,
        }

    @staticmethod
    def compare_months(
        df: pd.DataFrame,
        month_a: str,
        month_b: str,
        currency: str = "INR",
        month_b_is_partial: bool = False,
    ) -> Dict[str, Any]:
        """
        Compare spending and income across two specific months (YYYY-MM).
        Distinguishes full from partial periods to avoid misleading comparisons.
        """
        if df.empty:
            raise ValueError("Transaction dataset is empty.")

        df_copy = df[df["currency"].str.upper() == currency.upper()].copy()
        df_copy["month_str"] = pd.to_datetime(df_copy["date"]).dt.strftime("%Y-%m")

        df_a = df_copy[df_copy["month_str"] == month_a]
        df_b = df_copy[df_copy["month_str"] == month_b]

        sum_a = FinancialAnalytics.calculate_summary(df_a, period_label=month_a, currency=currency)
        sum_b = FinancialAnalytics.calculate_summary(df_b, period_label=month_b, currency=currency)

        expense_diff_paise = sum_b["total_expense_paise"] - sum_a["total_expense_paise"]
        expense_pct_change = (
            round((expense_diff_paise / sum_a["total_expense_paise"]) * 100.0, 2)
            if sum_a["total_expense_paise"] > 0
            else 0.0
        )

        # Category level changes
        cat_a = {
            item["category"]: item["total_paise"]
            for item in FinancialAnalytics.category_breakdown(df_a, currency)["categories"]
        }
        cat_b = {
            item["category"]: item["total_paise"]
            for item in FinancialAnalytics.category_breakdown(df_b, currency)["categories"]
        }

        all_cats = set(cat_a.keys()).union(set(cat_b.keys()))
        cat_shifts = []
        for c in all_cats:
            val_a = cat_a.get(c, 0)
            val_b = cat_b.get(c, 0)
            diff = val_b - val_a
            pct = round((diff / val_a * 100.0), 2) if val_a > 0 else (100.0 if val_b > 0 else 0.0)
            cat_shifts.append({
                "category": c,
                "month_a_rupees": paise_to_rupees(val_a),
                "month_b_rupees": paise_to_rupees(val_b),
                "diff_rupees": paise_to_rupees(diff),
                "pct_change": pct,
            })

        cat_shifts.sort(key=lambda x: x["diff_rupees"], reverse=True)
        top_increase = cat_shifts[0] if cat_shifts and cat_shifts[0]["diff_rupees"] > 0 else None
        top_decrease = cat_shifts[-1] if cat_shifts and cat_shifts[-1]["diff_rupees"] < 0 else None

        return {
            "currency": currency.upper(),
            "month_a": month_a,
            "month_b": month_b,
            "month_b_is_partial": month_b_is_partial,
            "period_warning": (
                f"Note: {month_b} is a partial billing period. Unequal period comparisons should be interpreted conservatively."
                if month_b_is_partial
                else None
            ),
            "month_a_summary": sum_a,
            "month_b_summary": sum_b,
            "expense_diff_rupees": paise_to_rupees(expense_diff_paise),
            "expense_pct_change": expense_pct_change,
            "category_shifts": cat_shifts,
            "highest_increase_category": top_increase,
            "highest_decrease_category": top_decrease,
        }

    @staticmethod
    def detect_spending_anomalies(
        df: pd.DataFrame,
        current_month: Optional[str] = None,
        currency: str = "INR",
        multiplier_threshold: float = 1.5,
    ) -> Dict[str, Any]:
        """
        Identify categories or transactions experiencing unusual spending increases.
        Uses neutral, professional language (e.g., 'unusual increase', 'worth reviewing').
        """
        if df.empty:
            return {"anomalies_detected": [], "currency": currency.upper()}

        curr_df = df[df["currency"].str.upper() == currency.upper()].copy()
        curr_df["month_str"] = pd.to_datetime(curr_df["date"]).dt.strftime("%Y-%m")

        months = sorted(curr_df["month_str"].unique())
        if len(months) < 2:
            return {
                "currency": currency.upper(),
                "anomalies_detected": [],
                "note": "Insufficient historical months to establish baseline spending patterns.",
            }

        target_month = current_month or months[-1]
        baseline_months = [m for m in months if m < target_month]

        if not baseline_months:
            return {"currency": currency.upper(), "anomalies_detected": []}

        baseline_df = curr_df[curr_df["month_str"].isin(baseline_months) & (curr_df["transaction_type"] == "expense")]
        target_df = curr_df[(curr_df["month_str"] == target_month) & (curr_df["transaction_type"] == "expense")]

        baseline_cat_avg = (
            baseline_df.groupby(["category", "month_str"])["amount_paise"]
            .sum()
            .groupby("category")
            .mean()
        )

        target_cat_sum = target_df.groupby("category")["amount_paise"].sum()

        anomalies = []
        for cat, target_spend in target_cat_sum.items():
            base_avg = baseline_cat_avg.get(cat, 0)
            if base_avg > 0 and target_spend >= (base_avg * multiplier_threshold):
                increase_factor = round(target_spend / base_avg, 2)
                diff_rupees = paise_to_rupees(int(target_spend - base_avg))
                anomalies.append({
                    "category": cat,
                    "target_month": target_month,
                    "target_spend_rupees": paise_to_rupees(int(target_spend)),
                    "baseline_average_rupees": paise_to_rupees(int(base_avg)),
                    "increase_rupees": diff_rupees,
                    "increase_factor": f"{increase_factor}x",
                    "status": "Unusual increase worth reviewing",
                })

        return {
            "currency": currency.upper(),
            "target_month": target_month,
            "baseline_months": baseline_months,
            "anomalies_detected": anomalies,
            "total_anomalies": len(anomalies),
        }

    @staticmethod
    def find_recurring_candidates(df: pd.DataFrame, currency: str = "INR") -> List[Dict[str, Any]]:
        """
        Identify recurring transactions (e.g. monthly subscriptions, rent, utility bills)
        sharing merchant and consistent amounts across distinct dates.
        """
        if df.empty:
            return []

        curr_df = df[df["currency"].str.upper() == currency.upper()].copy()
        expense_df = curr_df[curr_df["transaction_type"] == "expense"].copy()
        if expense_df.empty:
            return []

        expense_df["month_str"] = pd.to_datetime(expense_df["date"]).dt.strftime("%Y-%m")
        grouped = expense_df.groupby(["description", "amount_paise", "category"])

        recurring_candidates = []
        for (desc, amount_paise, cat), group in grouped:
            unique_months = group["month_str"].nunique()
            if unique_months >= 2:
                recurring_candidates.append({
                    "description": desc,
                    "category": cat,
                    "amount_rupees": paise_to_rupees(amount_paise),
                    "amount_paise": amount_paise,
                    "occurrences": len(group),
                    "distinct_months": unique_months,
                    "frequency": "Monthly candidate",
                })

        recurring_candidates.sort(key=lambda x: x["amount_paise"], reverse=True)
        return recurring_candidates
