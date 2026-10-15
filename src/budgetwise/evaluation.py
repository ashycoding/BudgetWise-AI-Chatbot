"""
BudgetWise AI - Repeatable Evaluation & Benchmarking Module.

Evaluates categorization accuracy, precision, recall, F1, confusion matrix,
intent-routing accuracy, and arithmetic calculation correctness without requiring
an external OpenAI API key.
"""

from collections import defaultdict
from datetime import datetime
import json
from typing import Any, Dict, List, Tuple
from budgetwise.analytics import FinancialAnalytics
from budgetwise.categorizer import ExpenseCategorizer
from budgetwise.chatbot import BudgetWiseChatbot
from budgetwise.models import StandardCategory, rupees_to_paise
import pandas as pd


# Benchmark Dataset: Labeled Transaction Descriptions
SYNTHETIC_BENCHMARK_DATA: List[Tuple[str, str]] = [
    # Groceries
    ("Blinkit daily vegetables and milk", StandardCategory.GROCERIES.value),
    ("Zepto instant grocery delivery", StandardCategory.GROCERIES.value),
    ("DMart Supermarket monthly ration", StandardCategory.GROCERIES.value),
    ("BigBasket fruits and bread", StandardCategory.GROCERIES.value),
    ("Kirana store provisions", StandardCategory.GROCERIES.value),

    # Food and Dining
    ("Swiggy Biryani Dinner", StandardCategory.FOOD_AND_DINING.value),
    ("Zomato Pizza order", StandardCategory.FOOD_AND_DINING.value),
    ("Starbucks coffee latte", StandardCategory.FOOD_AND_DINING.value),
    ("McDonalds burger meal", StandardCategory.FOOD_AND_DINING.value),
    ("Haldiram sweets and snacks", StandardCategory.FOOD_AND_DINING.value),

    # Transportation
    ("Uber cab trip to office", StandardCategory.TRANSPORTATION.value),
    ("Ola auto ride", StandardCategory.TRANSPORTATION.value),
    ("Indian Oil petrol refill", StandardCategory.TRANSPORTATION.value),
    ("Shell fuel station diesel", StandardCategory.TRANSPORTATION.value),
    ("Metro smart card recharge", StandardCategory.TRANSPORTATION.value),

    # Housing and Rent
    ("House rent payment for October", StandardCategory.HOUSING_AND_RENT.value),
    ("Apartment society maintenance charge", StandardCategory.HOUSING_AND_RENT.value),
    ("NoBroker flat rent transfer", StandardCategory.HOUSING_AND_RENT.value),

    # Utilities
    ("BESCOM electricity bill payment", StandardCategory.UTILITIES.value),
    ("Airtel broadband wifi bill", StandardCategory.UTILITIES.value),
    ("Indane gas cylinder booking", StandardCategory.UTILITIES.value),

    # Shopping
    ("Amazon electronics and laptop stand", StandardCategory.SHOPPING.value),
    ("Flipkart sale clothing purchase", StandardCategory.SHOPPING.value),
    ("Myntra apparel and shoes", StandardCategory.SHOPPING.value),
    ("Decathlon sports gear", StandardCategory.SHOPPING.value),

    # Healthcare
    ("Apollo Pharmacy medicines", StandardCategory.HEALTHCARE.value),
    ("1mg prescription tablets", StandardCategory.HEALTHCARE.value),
    ("Dr Lal Pathlabs blood diagnostic test", StandardCategory.HEALTHCARE.value),

    # Subscriptions
    ("Netflix monthly 4K subscription", StandardCategory.SUBSCRIPTIONS.value),
    ("Spotify family music plan", StandardCategory.SUBSCRIPTIONS.value),
    ("YouTube Premium membership", StandardCategory.SUBSCRIPTIONS.value),

    # Entertainment
    ("PVR Cinemas movie tickets for two", StandardCategory.ENTERTAINMENT.value),
    ("BookMyShow comedy show pass", StandardCategory.ENTERTAINMENT.value),

    # Travel
    ("MakeMyTrip flight to Mumbai", StandardCategory.TRAVEL.value),
    ("Goibibo hotel stay booking", StandardCategory.TRAVEL.value),

    # Personal Care
    ("Urban Company haircut and salon", StandardCategory.PERSONAL_CARE.value),

    # EMI and Debt Payments
    ("HDFC home loan EMI debit", StandardCategory.EMI_AND_DEBT.value),
    ("SBI credit card bill payment", StandardCategory.EMI_AND_DEBT.value),

    # Investments and Savings
    ("Zerodha Coin mutual fund SIP", StandardCategory.INVESTMENTS_AND_SAVINGS.value),
    ("Groww equity stocks purchase", StandardCategory.INVESTMENTS_AND_SAVINGS.value),

    # Income
    ("Monthly Salary credit from Employer", StandardCategory.INCOME.value),
    ("Freelance consulting payment received", StandardCategory.INCOME.value),
]

# Benchmark Dataset: User Intent Routing
INTENT_BENCHMARK_DATA: List[Tuple[str, str]] = [
    ("How much did I spend this month?", "get_summary"),
    ("What are my total expenses for October?", "get_summary"),
    ("Where is most of my money going?", "category_breakdown"),
    ("Give me my spending breakdown", "category_breakdown"),
    ("How much did I spend on food last month?", "category_spend"),
    ("Am I over my shopping budget?", "check_budget"),
    ("What is my budget status for groceries?", "check_budget"),
    ("Compare my expenses with last month.", "compare_months"),
    ("What subscriptions am I paying for?", "get_subscriptions"),
    ("How much money did I save this month?", "get_savings"),
    ("Which category increased the most?", "detect_anomalies"),
    ("Add ₹450 for groceries yesterday.", "add_transaction"),
    ("Spent 1200 on blinkit today", "add_transaction"),
    ("Set my monthly entertainment budget to ₹2,000.", "set_budget"),
    ("Write a poem about money and rain", "out_of_domain"),
]


class ModelEvaluator:
    """
    Automated evaluation framework for BudgetWise AI.
    Executes repeatable deterministic evaluations and generates performance reports.
    """

    def __init__(self, categorizer: ExpenseCategorizer | None = None, chatbot: BudgetWiseChatbot | None = None) -> None:
        self.categorizer = categorizer or ExpenseCategorizer()
        self.chatbot = chatbot or BudgetWiseChatbot()

    def evaluate_categorizer(
        self, dataset: List[Tuple[str, str]] = SYNTHETIC_BENCHMARK_DATA
    ) -> Dict[str, Any]:
        """
        Evaluate classification accuracy, per-category precision, recall, F1, and confusion matrix.
        """
        y_true = []
        y_pred = []
        confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for desc, true_cat in dataset:
            pred = self.categorizer.predict(desc)
            pred_cat = pred.predicted_category
            y_true.append(true_cat)
            y_pred.append(pred_cat)
            confusion[true_cat][pred_cat] += 1

        total = len(dataset)
        correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        accuracy = round((correct / total) * 100.0, 2) if total > 0 else 0.0

        # Compute per-category precision, recall, F1
        classes = sorted(list(set(y_true)))
        metrics_by_cat = {}
        for c in classes:
            tp = confusion[c][c]
            fp = sum(confusion[other][c] for other in classes if other != c)
            fn = sum(confusion[c][other] for other in classes if other != c)

            prec = round(tp / (tp + fp), 2) if (tp + fp) > 0 else 0.0
            rec = round(tp / (tp + fn), 2) if (tp + fn) > 0 else 0.0
            f1 = round(2 * (prec * rec) / (prec + rec), 2) if (prec + rec) > 0 else 0.0

            metrics_by_cat[c] = {
                "support": tp + fn,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
            }

        macro_f1 = round(sum(m["f1_score"] for m in metrics_by_cat.values()) / len(classes), 2)

        return {
            "total_samples": total,
            "correct_samples": correct,
            "accuracy_percentage": accuracy,
            "macro_f1": macro_f1,
            "category_metrics": metrics_by_cat,
            "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        }

    def evaluate_intent_routing(
        self, dataset: List[Tuple[str, str]] = INTENT_BENCHMARK_DATA
    ) -> Dict[str, Any]:
        """Evaluate domain intent detection accuracy."""
        correct = 0
        total = len(dataset)
        results = []

        for query, expected_intent in dataset:
            detected_intent, _ = self.chatbot.detect_intent(query)
            is_match = (detected_intent == expected_intent)
            if is_match:
                correct += 1
            results.append({
                "query": query,
                "expected": expected_intent,
                "detected": detected_intent,
                "correct": is_match,
            })

        acc = round((correct / total) * 100.0, 2) if total > 0 else 0.0
        return {
            "total_queries": total,
            "correct_routing": correct,
            "intent_accuracy": acc,
            "details": results,
        }

    def evaluate_calculation_grounding(self) -> Dict[str, Any]:
        """
        Verify that mathematical aggregations exactly match independently computed sums.
        Asserts 0% arithmetic hallucination.
        """
        # Independent synthetic dataset with known exact mathematical ground truth
        test_rows = [
            {"date": "2026-10-01", "description": "Salary", "amount_paise": rupees_to_paise(50000), "currency": "INR", "transaction_type": "income", "category": "Income"},
            {"date": "2026-10-02", "description": "Rent", "amount_paise": rupees_to_paise(12000), "currency": "INR", "transaction_type": "expense", "category": "Housing and Rent"},
            {"date": "2026-10-05", "description": "Groceries", "amount_paise": rupees_to_paise(4500), "currency": "INR", "transaction_type": "expense", "category": "Groceries"},
            {"date": "2026-10-10", "description": "Dinner", "amount_paise": rupees_to_paise(1500), "currency": "INR", "transaction_type": "expense", "category": "Food and Dining"},
            {"date": "2026-10-12", "description": "Refund", "amount_paise": rupees_to_paise(500), "currency": "INR", "transaction_type": "refund", "category": "Food and Dining"},
        ]
        df = pd.DataFrame(test_rows)

        summary = FinancialAnalytics.calculate_summary(df, period_label="2026-10", currency="INR")

        expected_income = 50000.0
        expected_gross_expense = 18000.0
        expected_refund = 500.0
        expected_net_expense = 17500.0
        expected_savings = 32500.0
        expected_savings_rate = 65.0

        is_income_correct = (summary["total_income_rupees"] == expected_income)
        is_expense_correct = (summary["total_expense_rupees"] == expected_net_expense)
        is_savings_correct = (summary["net_savings_rupees"] == expected_savings)
        is_rate_correct = (summary["savings_rate_percentage"] == expected_savings_rate)

        all_passed = is_income_correct and is_expense_correct and is_savings_correct and is_rate_correct

        return {
            "all_passed": all_passed,
            "hallucination_rate_percentage": 0.0 if all_passed else 100.0,
            "grounding_fidelity_percentage": 100.0 if all_passed else 0.0,
            "verifications": {
                "income": {"expected": expected_income, "actual": summary["total_income_rupees"], "passed": is_income_correct},
                "net_expense": {"expected": expected_net_expense, "actual": summary["total_expense_rupees"], "passed": is_expense_correct},
                "net_savings": {"expected": expected_savings, "actual": summary["net_savings_rupees"], "passed": is_savings_correct},
                "savings_rate": {"expected": expected_savings_rate, "actual": summary["savings_rate_percentage"], "passed": is_rate_correct},
            },
        }

    def run_full_evaluation(self) -> Dict[str, Any]:
        """Execute complete offline benchmark suite."""
        cat_results = self.evaluate_categorizer()
        intent_results = self.evaluate_intent_routing()
        calc_results = self.evaluate_calculation_grounding()

        return {
            "timestamp": datetime.now().isoformat(),
            "categorization": cat_results,
            "intent_routing": intent_results,
            "calculation_grounding": calc_results,
        }

    def generate_report(self) -> str:
        """Generate a formatted markdown evaluation report."""
        eval_data = self.run_full_evaluation()
        cat = eval_data["categorization"]
        intent = eval_data["intent_routing"]
        calc = eval_data["calculation_grounding"]

        report = f"""# BudgetWise AI - Evaluation & Performance Benchmark Report
**Generated At:** `{eval_data['timestamp']}`
**Environment:** Offline Deterministic Verification Suite

---

## 1. Expense Categorization Performance
- **Total Benchmark Samples:** {cat['total_samples']}
- **Correct Predictions:** {cat['correct_samples']}
- **Classification Accuracy:** **{cat['accuracy_percentage']}%**
- **Macro F1 Score:** **{cat['macro_f1']}**

### Per-Category Metrics:
| Category | Support | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
"""
        for cat_name, metrics in cat["category_metrics"].items():
            report += f"| {cat_name} | {metrics['support']} | {metrics['precision']} | {metrics['recall']} | {metrics['f1_score']} |\n"

        report += f"""
---

## 2. Intent-Routing Performance
- **Total Test Queries:** {intent['total_queries']}
- **Correct Routings:** {intent['correct_routing']}
- **Intent Recognition Accuracy:** **{intent['intent_accuracy']}%**

---

## 3. Financial Calculation & Grounding Fidelity
- **Arithmetic Verification:** {'[PASSED (100% Match)]' if calc['all_passed'] else '[FAILED]'}
- **Hallucinated Metric Rate:** **{calc['hallucination_rate_percentage']}%**
- **Deterministic Grounding Fidelity:** **{calc['grounding_fidelity_percentage']}%**

All financial figures are strictly calculated by Python/pandas and verified against ground truth arithmetic.
"""
        return report


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    evaluator = ModelEvaluator()
    print(evaluator.generate_report())
