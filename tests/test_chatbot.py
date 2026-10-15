"""
Automated Tests for Chatbot Orchestrator Module (Stage 4).
"""

from unittest.mock import MagicMock, patch
import pytest
from budgetwise.analytics import FinancialAnalytics
from budgetwise.budget_service import BudgetService
from budgetwise.categorizer import ExpenseCategorizer
from budgetwise.chatbot import BudgetWiseChatbot
from budgetwise.database import DatabaseManager
from budgetwise.models import (
    BudgetCreate,
    ChatQuery,
    StandardCategory,
    TransactionCreate,
    rupees_to_paise,
)
from budgetwise.transaction_service import TransactionService


@pytest.fixture
def chatbot(tmp_path):
    """Fixture providing an isolated BudgetWiseChatbot instance with test database."""
    db_file = tmp_path / "test_chatbot.db"
    db_mgr = DatabaseManager(db_path=db_file)
    db_mgr.init_schema()

    txn_service = TransactionService(db_manager=db_mgr)
    bgt_service = BudgetService(db_manager=db_mgr)
    analytics = FinancialAnalytics()
    categorizer = ExpenseCategorizer(db_manager=db_mgr)

    return BudgetWiseChatbot(
        transaction_service=txn_service,
        budget_service=bgt_service,
        analytics=analytics,
        categorizer=categorizer,
        db_manager=db_mgr,
    )


def test_intent_detection_queries(chatbot):
    """Verify accurate intent detection across all required natural-language queries."""
    cases = [
        ("How much did I spend this month?", "get_summary"),
        ("Where is most of my money going?", "category_breakdown"),
        ("How much did I spend on food last month?", "category_spend"),
        ("Am I over my shopping budget?", "check_budget"),
        ("Compare my expenses with last month.", "compare_months"),
        ("What subscriptions am I paying for?", "get_subscriptions"),
        ("How much money did I save this month?", "get_savings"),
        ("Which category increased the most?", "detect_anomalies"),
        ("Add ₹450 for groceries yesterday.", "add_transaction"),
        ("Set my monthly entertainment budget to ₹2,000.", "set_budget"),
        ("Write a python script to scrape a website", "out_of_domain"),
    ]
    for query, expected_intent in cases:
        intent, params = chatbot.detect_intent(query)
        assert intent == expected_intent, f"Query '{query}' expected {expected_intent}, got {intent}"


def test_chatbot_add_transaction_via_natural_language(chatbot):
    """Verify that user can add transactions using natural language."""
    resp = chatbot.process_message(ChatQuery(message="Add ₹450 for groceries yesterday."))
    assert resp.action_executed == "add_transaction"
    assert resp.data["amount"] == 450.0
    assert "Recorded expense: **₹450.00**" in resp.reply

    # Verify transaction in database
    txns = chatbot.transactions.list_transactions()
    assert len(txns) == 1
    assert txns[0].amount_rupees == 450.0
    assert txns[0].category == StandardCategory.GROCERIES.value


def test_chatbot_set_budget_via_natural_language(chatbot):
    """Verify setting category budget limits via natural language."""
    resp = chatbot.process_message(
        ChatQuery(message="Set my monthly entertainment budget to ₹2,000.")
    )
    assert resp.action_executed == "set_budget"
    assert "Configured monthly budget for **Entertainment**: **₹2,000.00**" in resp.reply

    # Verify budget in database
    budgets = chatbot.budgets.list_budgets()
    assert len(budgets) == 1
    assert budgets[0].category == StandardCategory.ENTERTAINMENT.value
    assert budgets[0].monthly_limit_rupees == 2000.0


def test_chatbot_budget_status_inquiry(chatbot):
    """Verify chatbot answers budget utilization questions using verified figures."""
    month = chatbot._get_current_month()
    chatbot.budgets.set_budget(
        BudgetCreate(
            category=StandardCategory.SHOPPING.value,
            monthly_limit_paise=rupees_to_paise(5000),
            month=month,
        )
    )
    chatbot.transactions.add_transaction(
        TransactionCreate(
            date=f"{month}-05",
            description="Zara Jeans",
            amount_paise=rupees_to_paise(6000),
            category=StandardCategory.SHOPPING.value,
        )
    )

    resp = chatbot.process_message(ChatQuery(message="Am I over my shopping budget?"))
    assert resp.intent == "check_budget"
    assert "Over Budget" in resp.reply
    assert "120.0%" in resp.reply or "120%" in resp.reply


def test_chatbot_out_of_domain_polite_clarification(chatbot):
    """Verify polite clarification for non-finance requests."""
    resp = chatbot.process_message(ChatQuery(message="Write a poem about rain."))
    assert resp.intent == "out_of_domain"
    assert "specialized personal budgeting assistant" in resp.reply


def test_chatbot_large_transaction_confirmation(chatbot):
    """Verify large transaction requests prompt user confirmation."""
    resp = chatbot.process_message(ChatQuery(message="Spent 250000 on car down payment."))
    assert resp.requires_confirmation is True
    assert "Please confirm" in resp.reply


def test_chatbot_mocked_openai_grounding(chatbot):
    """Verify that when OpenAI is configured, verified calculations are passed in context."""
    fake_client = MagicMock()
    fake_completion = MagicMock()
    fake_completion.choices = [
        MagicMock(message=MagicMock(content="You have spent ₹3,500 on groceries this month."))
    ]
    fake_client.chat.completions.create.return_value = fake_completion

    with patch("budgetwise.config.Settings.is_openai_configured", return_value=True), \
         patch("budgetwise.config.Settings.get_openai_client", return_value=fake_client):
        resp = chatbot.process_message(ChatQuery(message="How much did I spend this month?"))
        assert resp.reply == "You have spent ₹3,500 on groceries this month."
        assert fake_client.chat.completions.create.called
