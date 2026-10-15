"""
Automated Tests for Evaluation Module (Stage 6).
"""

import pytest
from budgetwise.evaluation import ModelEvaluator


@pytest.fixture
def evaluator():
    """Fixture providing a fresh ModelEvaluator instance."""
    return ModelEvaluator()


def test_evaluation_categorizer_accuracy(evaluator):
    """Verify that categorization benchmark achieves high accuracy and valid F1 metrics."""
    res = evaluator.evaluate_categorizer()
    assert res["total_samples"] > 0
    assert res["accuracy_percentage"] >= 90.0
    assert res["macro_f1"] >= 0.85
    assert len(res["category_metrics"]) > 0


def test_evaluation_intent_routing_accuracy(evaluator):
    """Verify that domain intent routing benchmark achieves high accuracy."""
    res = evaluator.evaluate_intent_routing()
    assert res["total_queries"] > 0
    assert res["intent_accuracy"] >= 90.0


def test_evaluation_calculation_fidelity(evaluator):
    """Verify that calculation grounding achieves 0% hallucination and 100% fidelity."""
    res = evaluator.evaluate_calculation_grounding()
    assert res["all_passed"] is True
    assert res["hallucination_rate_percentage"] == 0.0
    assert res["grounding_fidelity_percentage"] == 100.0


def test_generate_evaluation_report(evaluator):
    """Verify markdown evaluation report formatting."""
    report = evaluator.generate_report()
    assert "Expense Categorization Performance" in report
    assert "Intent-Routing Performance" in report
    assert "Financial Calculation & Grounding Fidelity" in report
