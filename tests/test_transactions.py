"""
Automated Tests for Transaction Service Module (Stage 2).
"""

import pytest
from budgetwise.categorizer import ExpenseCategorizer
from budgetwise.database import DatabaseManager
from budgetwise.models import (
    StandardCategory,
    TransactionCreate,
    TransactionType,
    TransactionUpdate,
    rupees_to_paise,
)
from budgetwise.transaction_service import TransactionService


@pytest.fixture
def service(tmp_path):
    """Fixture providing a fresh TransactionService backed by a temporary SQLite database."""
    db_file = tmp_path / "test_transactions.db"
    db_mgr = DatabaseManager(db_path=db_file)
    db_mgr.init_schema()
    categorizer = ExpenseCategorizer(db_manager=db_mgr)
    return TransactionService(db_manager=db_mgr, categorizer=categorizer)


def test_add_and_get_transaction(service):
    """Test adding a single transaction and retrieving it by ID."""
    txn = TransactionCreate(
        date="2026-10-15",
        description="Swiggy Biryani Order",
        amount_paise=rupees_to_paise(450.00),
        currency="INR",
        transaction_type=TransactionType.EXPENSE,
        category="Other",  # Service will auto-categorize via categorizer
        notes="Dinner with family",
    )
    record = service.add_transaction(txn)
    assert record.id is not None
    assert record.amount_paise == 45000
    assert record.amount_rupees == 450.00
    assert record.category == StandardCategory.FOOD_AND_DINING.value

    fetched = service.get_transaction(record.id)
    assert fetched is not None
    assert fetched.id == record.id
    assert fetched.description == "Swiggy Biryani Order"


def test_list_and_filter_transactions(service):
    """Test filtering transactions by category and date ranges."""
    service.add_transaction(
        TransactionCreate(
            date="2026-10-01",
            description="Blinkit Groceries",
            amount_paise=rupees_to_paise(1200),
            category=StandardCategory.GROCERIES.value,
        )
    )
    service.add_transaction(
        TransactionCreate(
            date="2026-10-10",
            description="Uber Ride",
            amount_paise=rupees_to_paise(350),
            category=StandardCategory.TRANSPORTATION.value,
        )
    )
    service.add_transaction(
        TransactionCreate(
            date="2026-09-28",
            description="Past Electricity Bill",
            amount_paise=rupees_to_paise(1800),
            category=StandardCategory.UTILITIES.value,
        )
    )

    all_txns = service.list_transactions()
    assert len(all_txns) == 3

    oct_txns = service.list_transactions(start_date="2026-10-01", end_date="2026-10-31")
    assert len(oct_txns) == 2

    grocery_txns = service.list_transactions(category=StandardCategory.GROCERIES.value)
    assert len(grocery_txns) == 1
    assert grocery_txns[0].description == "Blinkit Groceries"


def test_update_and_delete_transaction(service):
    """Test editing transaction details and deletion."""
    record = service.add_transaction(
        TransactionCreate(
            date="2026-10-14",
            description="Coffee Shop",
            amount_paise=rupees_to_paise(150),
            category=StandardCategory.FOOD_AND_DINING.value,
        )
    )

    # Update amount and notes
    updated = service.update_transaction(
        record.id,
        TransactionUpdate(amount_paise=rupees_to_paise(200), notes="Extra pastry"),
    )
    assert updated is not None
    assert updated.amount_rupees == 200.00
    assert updated.notes == "Extra pastry"

    # Delete
    deleted = service.delete_transaction(record.id)
    assert deleted is True
    assert service.get_transaction(record.id) is None


def test_import_from_csv_with_duplicate_detection(service):
    """Test CSV ingestion, amount normalization, and duplicate handling."""
    csv_content = """Date,Description,Amount,Type,Category,Notes
12/10/2026,Apollo Pharmacy,₹650.00,Debit,,Medicines
13/10/2026,Salary October,50000.00,Credit,Income,Monthly Payroll
14/10/2026,Netflix,649.00,Debit,Subscriptions,Premium Plan
12/10/2026,Apollo Pharmacy,650.00,Debit,Healthcare,Medicines Duplicate
"""
    result = service.import_from_csv(csv_content, currency="INR", skip_duplicates=True)
    assert result["imported_count"] == 3
    assert result["skipped_duplicates"] == 1
    assert result["invalid_rows"] == 0

    txns = service.list_transactions()
    assert len(txns) == 3


def test_export_to_dataframe(service):
    """Test exporting transaction rows to a pandas DataFrame for analytics."""
    service.add_transaction(
        TransactionCreate(
            date="2026-10-15",
            description="DMart",
            amount_paise=rupees_to_paise(2500),
            category=StandardCategory.GROCERIES.value,
        )
    )
    df = service.export_to_dataframe()
    assert not df.empty
    assert len(df) == 1
    assert df.iloc[0]["description"] == "DMart"
    assert df.iloc[0]["amount_rupees"] == 2500.00
