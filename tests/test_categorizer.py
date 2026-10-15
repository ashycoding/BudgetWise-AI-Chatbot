"""
Automated Tests for Expense Categorizer Module (Stage 2).
"""

import pytest
from budgetwise.categorizer import ExpenseCategorizer
from budgetwise.database import DatabaseManager
from budgetwise.models import ClassificationMethod, ConfidenceBand, StandardCategory


@pytest.fixture
def categorizer(tmp_path):
    """Fixture providing an isolated ExpenseCategorizer backed by a temp DB."""
    db_file = tmp_path / "test_cat.db"
    db_mgr = DatabaseManager(db_path=db_file)
    db_mgr.init_schema()
    return ExpenseCategorizer(db_manager=db_mgr)


def test_predict_known_grocery_merchant(categorizer):
    """Verify standard grocery merchants map to Groceries with high confidence."""
    for desc in ["Blinkit Order #1234", "BigBasket daily veggies", "DMart Supermarket purchase", "Zepto quick delivery"]:
        pred = categorizer.predict(desc)
        assert pred.predicted_category == StandardCategory.GROCERIES.value
        assert pred.confidence_band == ConfidenceBand.HIGH
        assert pred.classification_method == ClassificationMethod.KEYWORD_RULE
        assert pred.needs_confirmation is False


def test_predict_food_and_dining(categorizer):
    """Verify food delivery and dining keywords map accurately."""
    for desc in ["Swiggy food delivery", "Zomato Biryani", "Starbucks Coffee", "McDonalds burger meal"]:
        pred = categorizer.predict(desc)
        assert pred.predicted_category == StandardCategory.FOOD_AND_DINING.value
        assert pred.confidence_band == ConfidenceBand.HIGH
        assert pred.needs_confirmation is False


def test_predict_transportation(categorizer):
    """Verify transportation and transit merchants."""
    for desc in ["Uber trip to airport", "Ola cab ride", "Petrol at Indian Oil", "Shell fuel station"]:
        pred = categorizer.predict(desc)
        assert pred.predicted_category == StandardCategory.TRANSPORTATION.value
        assert pred.confidence_band == ConfidenceBand.HIGH


def test_predict_utilities_and_subscriptions(categorizer):
    """Verify utility bills and digital recurring subscriptions."""
    pred_util = categorizer.predict("BESCOM electricity bill payment")
    assert pred_util.predicted_category == StandardCategory.UTILITIES.value

    pred_sub = categorizer.predict("Netflix monthly subscription")
    assert pred_sub.predicted_category == StandardCategory.SUBSCRIPTIONS.value


def test_predict_explicit_income(categorizer):
    """Verify explicit income keywords and cash-flow tags."""
    pred = categorizer.predict("Monthly Salary from Acme Corp", transaction_type="income")
    assert pred.predicted_category == StandardCategory.INCOME.value
    assert pred.confidence_band == ConfidenceBand.HIGH


def test_predict_user_override_precedence(categorizer):
    """Verify that a persisted user override overrides keyword and default rules."""
    # Custom business "Namma Chai Adda" that is usually unmapped or could be misclassified
    pattern = "Namma Chai Adda"
    categorizer.save_user_override(pattern, StandardCategory.FOOD_AND_DINING.value)

    pred = categorizer.predict("Payment at Namma Chai Adda Indiranagar")
    assert pred.predicted_category == StandardCategory.FOOD_AND_DINING.value
    assert pred.classification_method == ClassificationMethod.USER_OVERRIDE
    assert pred.confidence_band == ConfidenceBand.HIGH
    assert pred.needs_confirmation is False


def test_predict_ambiguous_and_empty_descriptions(categorizer):
    """Verify that unmapped descriptions trigger low confidence or clarification."""
    pred_empty = categorizer.predict("")
    assert pred_empty.confidence_band == ConfidenceBand.UNKNOWN
    assert pred_empty.needs_confirmation is True

    pred_random = categorizer.predict("XYZ987654321 Transaction 882")
    # Low confidence or unknown should flag need for confirmation
    assert pred_random.needs_confirmation is True
