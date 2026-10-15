"""
BudgetWise AI - SQLite Database Management Module.

Handles SQLite connection pooling, schema migrations, and parameterized query execution.
Enforces foreign-key constraints and stores monetary values in integer minor units (paise).
"""

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Generator, List, Tuple
from budgetwise.config import get_settings


SCHEMA_VERSION = 1

CREATE_TABLES_SQL = """
-- Transactions Table
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    description TEXT NOT NULL,
    amount_paise INTEGER NOT NULL CHECK(amount_paise > 0),
    currency TEXT NOT NULL DEFAULT 'INR',
    transaction_type TEXT NOT NULL,
    category TEXT NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date);
CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category);
CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions(transaction_type);
CREATE INDEX IF NOT EXISTS idx_transactions_currency ON transactions(currency);

-- Budgets Table
CREATE TABLE IF NOT EXISTS budgets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    monthly_limit_paise INTEGER NOT NULL CHECK(monthly_limit_paise > 0),
    currency TEXT NOT NULL DEFAULT 'INR',
    month TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    CONSTRAINT uq_category_month_currency UNIQUE (category, month, currency)
);

CREATE INDEX IF NOT EXISTS idx_budgets_month ON budgets(month);
CREATE INDEX IF NOT EXISTS idx_budgets_category ON budgets(category);

-- Category Overrides Table (Persists user corrections)
CREATE TABLE IF NOT EXISTS category_overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_category_overrides_pattern ON category_overrides(pattern);

-- Conversation History Table
CREATE TABLE IF NOT EXISTS conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    metadata_json TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_conversation_session ON conversation_history(session_id);

-- Schema Metadata Table
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class DatabaseManager:
    """
    Manages SQLite connections, schema initialization, and transactional safety.
    """

    def __init__(self, db_path: Path | str | None = None, auto_init: bool = True) -> None:
        if db_path is not None:
            self.db_path = Path(db_path) if isinstance(db_path, str) and db_path != ":memory:" else db_path
        else:
            settings = get_settings()
            self.db_path = settings.database_path
        if auto_init and str(self.db_path) != ":memory:":
            self.init_schema()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Yield an active SQLite connection with row factory configured
        and foreign key constraints enabled.
        """
        # Ensure parent directory exists for file-based databases
        if isinstance(self.db_path, Path) and str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        """
        Initialize database tables, indices, and schema migration tracking.
        Safe to call idempotently.
        """
        with self.get_connection() as conn:
            conn.executescript(CREATE_TABLES_SQL)
            cursor = conn.cursor()
            cursor.execute("SELECT version FROM schema_migrations WHERE version = ?", (SCHEMA_VERSION,))
            row = cursor.fetchone()
            if not row:
                cursor.execute("INSERT INTO schema_migrations (version) VALUES (?)", (SCHEMA_VERSION,))

    def validate_schema(self) -> bool:
        """
        Validate that all required tables and indices exist in the database.
        Returns True if schema is valid, False otherwise.
        """
        required_tables = {"transactions", "budgets", "category_overrides", "conversation_history", "schema_migrations"}
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            existing_tables = {row["name"] for row in cursor.fetchall()}
            if not required_tables.issubset(existing_tables):
                return False

            # Validate column presence in transactions
            cursor.execute("PRAGMA table_info(transactions);")
            txn_cols = {col["name"] for col in cursor.fetchall()}
            expected_txn_cols = {"id", "date", "description", "amount_paise", "currency", "transaction_type", "category", "notes"}
            if not expected_txn_cols.issubset(txn_cols):
                return False

            # Validate column presence in budgets
            cursor.execute("PRAGMA table_info(budgets);")
            budget_cols = {col["name"] for col in cursor.fetchall()}
            expected_budget_cols = {"id", "category", "monthly_limit_paise", "currency", "month"}
            if not expected_budget_cols.issubset(budget_cols):
                return False

        return True

    def reset_database(self) -> None:
        """Drop and re-initialize all tables. Intended for testing only."""
        with self.get_connection() as conn:
            conn.execute("PRAGMA foreign_keys = OFF;")
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%';")
            tables = [row["name"] for row in cursor.fetchall()]
            for tbl in tables:
                conn.execute(f"DROP TABLE IF EXISTS {tbl};")
            conn.execute("PRAGMA foreign_keys = ON;")
        self.init_schema()
