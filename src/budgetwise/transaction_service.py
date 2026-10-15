"""
BudgetWise AI - Transaction Service Module.

Coordinates CRUD operations, CSV statement parsing, duplicate detection,
and data normalization for financial transactions using SQLite and pandas.
"""

from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from budgetwise.categorizer import ExpenseCategorizer
from budgetwise.config import get_settings
from budgetwise.database import DatabaseManager
from budgetwise.models import (
    TransactionCreate,
    TransactionRecord,
    TransactionType,
    TransactionUpdate,
    rupees_to_paise,
)


class DuplicateTransactionError(Exception):
    """Raised when duplicate transactions are detected and strict prevention is active."""
    pass


class TransactionService:
    """
    Coordinates persistent storage, retrieval, validation, and CSV ingestion of transactions.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        categorizer: Optional[ExpenseCategorizer] = None,
    ) -> None:
        self.settings = get_settings()
        self.db = db_manager or DatabaseManager()
        self.categorizer = categorizer or ExpenseCategorizer(db_manager=self.db)

    def add_transaction(self, txn: TransactionCreate) -> TransactionRecord:
        """
        Insert a validated transaction record into SQLite.
        """
        # If category is default/other, attempt prediction
        category = txn.category
        if not category or category == "Other":
            pred = self.categorizer.predict(txn.description, txn.transaction_type.value)
            category = pred.predicted_category

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO transactions (
                    date, description, amount_paise, currency,
                    transaction_type, category, notes, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
                RETURNING id, date, description, amount_paise, currency,
                          transaction_type, category, notes, created_at, updated_at;
                """,
                (
                    txn.date,
                    txn.description.strip(),
                    txn.amount_paise,
                    txn.currency.upper(),
                    txn.transaction_type.value,
                    category,
                    txn.notes.strip() if txn.notes else None,
                ),
            )
            row = cursor.fetchone()
            return TransactionRecord(
                id=row["id"],
                date=row["date"],
                description=row["description"],
                amount_paise=row["amount_paise"],
                currency=row["currency"],
                transaction_type=TransactionType(row["transaction_type"]),
                category=row["category"],
                notes=row["notes"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    def get_transaction(self, transaction_id: int) -> Optional[TransactionRecord]:
        """Retrieve a single transaction by ID."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, date, description, amount_paise, currency,
                       transaction_type, category, notes, created_at, updated_at
                FROM transactions WHERE id = ?;
                """,
                (transaction_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return TransactionRecord(
                id=row["id"],
                date=row["date"],
                description=row["description"],
                amount_paise=row["amount_paise"],
                currency=row["currency"],
                transaction_type=TransactionType(row["transaction_type"]),
                category=row["category"],
                notes=row["notes"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    def list_transactions(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
        transaction_type: Optional[TransactionType] = None,
        currency: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> List[TransactionRecord]:
        """
        Query transactions with filtering and pagination.
        """
        query = """
            SELECT id, date, description, amount_paise, currency,
                   transaction_type, category, notes, created_at, updated_at
            FROM transactions
            WHERE 1=1
        """
        params: List[Any] = []

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        if category:
            query += " AND category = ?"
            params.append(category)
        if transaction_type:
            query += " AND transaction_type = ?"
            params.append(transaction_type.value)
        if currency:
            query += " AND currency = ?"
            params.append(currency.upper())

        query += " ORDER BY date DESC, id DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
            if offset is not None:
                query += " OFFSET ?"
                params.append(offset)

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [
                TransactionRecord(
                    id=r["id"],
                    date=r["date"],
                    description=r["description"],
                    amount_paise=r["amount_paise"],
                    currency=r["currency"],
                    transaction_type=TransactionType(r["transaction_type"]),
                    category=r["category"],
                    notes=r["notes"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                )
                for r in rows
            ]

    def update_transaction(
        self, transaction_id: int, updates: TransactionUpdate
    ) -> Optional[TransactionRecord]:
        """Update fields of an existing transaction."""
        existing = self.get_transaction(transaction_id)
        if not existing:
            return None

        update_fields: List[str] = []
        params: List[Any] = []

        if updates.date is not None:
            update_fields.append("date = ?")
            params.append(updates.date)
        if updates.description is not None:
            update_fields.append("description = ?")
            params.append(updates.description.strip())
        if updates.amount_paise is not None:
            update_fields.append("amount_paise = ?")
            params.append(updates.amount_paise)
        if updates.currency is not None:
            update_fields.append("currency = ?")
            params.append(updates.currency.upper())
        if updates.transaction_type is not None:
            update_fields.append("transaction_type = ?")
            params.append(updates.transaction_type.value)
        if updates.category is not None:
            update_fields.append("category = ?")
            params.append(updates.category)
        if updates.notes is not None:
            update_fields.append("notes = ?")
            params.append(updates.notes.strip() if updates.notes else None)

        if not update_fields:
            return existing

        update_fields.append("updated_at = datetime('now')")
        query = f"UPDATE transactions SET {', '.join(update_fields)} WHERE id = ?;"
        params.append(transaction_id)

        with self.db.get_connection() as conn:
            conn.execute(query, params)

        return self.get_transaction(transaction_id)

    def delete_transaction(self, transaction_id: int) -> bool:
        """Delete a transaction by ID. Returns True if deleted, False if not found."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM transactions WHERE id = ?;", (transaction_id,))
            return cursor.rowcount > 0

    def find_likely_duplicates(
        self, date: str, amount_paise: int, description: str, currency: str = "INR"
    ) -> List[TransactionRecord]:
        """
        Detect existing transactions sharing the identical date, amount, and similar description.
        Avoids flagging unrelated transactions sharing only date and amount.
        """
        import re
        norm_desc_tokens = set(re.sub(r"[^a-zA-Z0-9\s]", "", description.lower()).split())

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, date, description, amount_paise, currency,
                       transaction_type, category, notes, created_at, updated_at
                FROM transactions
                WHERE date = ? AND amount_paise = ? AND currency = ?;
                """,
                (date, amount_paise, currency.upper()),
            )
            rows = cursor.fetchall()
            candidates: List[TransactionRecord] = []
            for r in rows:
                existing_desc = r["description"]
                existing_tokens = set(re.sub(r"[^a-zA-Z0-9\s]", "", existing_desc.lower()).split())
                overlap = norm_desc_tokens.intersection(existing_tokens)
                # Match only if word tokens overlap or if descriptions are exactly identical
                if overlap or existing_desc.lower().strip() == description.lower().strip():
                    candidates.append(
                        TransactionRecord(
                            id=r["id"],
                            date=r["date"],
                            description=r["description"],
                            amount_paise=r["amount_paise"],
                            currency=r["currency"],
                            transaction_type=TransactionType(r["transaction_type"]),
                            category=r["category"],
                            notes=r["notes"],
                            created_at=r["created_at"],
                            updated_at=r["updated_at"],
                        )
                    )
            return candidates

    def import_from_csv(
        self,
        csv_data_or_path: Path | str,
        currency: str = "INR",
        skip_duplicates: bool = True,
    ) -> Dict[str, Any]:
        """
        Parse, normalize, and batch-import transactions from CSV data using pandas.
        Handles date parsing, header normalization, amount cleanup, duplicate detection,
        and sanitizes against spreadsheet formula injection.
        """
        if isinstance(csv_data_or_path, (str, Path)) and Path(str(csv_data_or_path)).is_file():
            df = pd.read_csv(csv_data_or_path)
        else:
            df = pd.read_csv(StringIO(str(csv_data_or_path)))

        # Normalize column headers (strip spaces, lowercase)
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

        # Identify required columns flexibly
        date_col = next((c for c in df.columns if "date" in c or "trans" in c), None)
        desc_col = next((c for c in df.columns if "desc" in c or "narration" in c or "merchant" in c or "particular" in c), None)
        amount_col = next((c for c in df.columns if "amount" in c or "amt" in c), None)
        type_col = next((c for c in df.columns if "type" in c or "cr/dr" in c), None)
        category_col = next((c for c in df.columns if "cat" in c), None)
        notes_col = next((c for c in df.columns if "note" in c or "remark" in c), None)

        if not (date_col and desc_col and amount_col):
            raise ValueError(
                f"CSV must contain date, description, and amount columns. Found: {list(df.columns)}"
            )

        imported_records: List[TransactionRecord] = []
        duplicate_candidates: List[Dict[str, Any]] = []
        invalid_rows: int = 0

        for idx, row in df.iterrows():
            raw_date = row.get(date_col)
            raw_desc = row.get(desc_col)
            raw_amount = row.get(amount_col)

            if pd.isna(raw_date) or pd.isna(raw_desc) or pd.isna(raw_amount):
                invalid_rows += 1
                continue

            # Normalize Date
            try:
                parsed_dt = pd.to_datetime(raw_date, dayfirst=True)
                iso_date = parsed_dt.strftime("%Y-%m-%d")
            except Exception:
                invalid_rows += 1
                continue

            # Normalize Amount & Type
            try:
                amt_str = str(raw_amount).replace(",", "").replace("₹", "").replace("$", "").strip()
                amt_float = float(amt_str)
                if amt_float == 0:
                    invalid_rows += 1
                    continue

                if amt_float < 0:
                    txn_type = TransactionType.EXPENSE
                    amt_float = abs(amt_float)
                else:
                    raw_type = str(row.get(type_col, "")).lower() if type_col else ""
                    if "credit" in raw_type or "cr" in raw_type or "income" in raw_type:
                        txn_type = TransactionType.INCOME
                    elif "refund" in raw_type:
                        txn_type = TransactionType.REFUND
                    elif "transfer" in raw_type:
                        txn_type = TransactionType.TRANSFER
                    elif "invest" in raw_type:
                        txn_type = TransactionType.INVESTMENT
                    else:
                        txn_type = TransactionType.EXPENSE

                amount_paise = rupees_to_paise(amt_float)
            except Exception:
                invalid_rows += 1
                continue

            description = str(raw_desc).strip()
            notes = str(row.get(notes_col, "")).strip() if notes_col and not pd.isna(row.get(notes_col)) else None

            # Category resolution
            category = str(row.get(category_col, "")).strip() if category_col and not pd.isna(row.get(category_col)) else ""
            if not category or category.lower() in ("other", "nan", ""):
                pred = self.categorizer.predict(description, txn_type.value)
                category = pred.predicted_category

            # Duplicate check
            existing_dupes = self.find_likely_duplicates(iso_date, amount_paise, description, currency)
            if existing_dupes:
                duplicate_candidates.append({
                    "date": iso_date,
                    "description": description,
                    "amount_paise": amount_paise,
                    "matched_id": existing_dupes[0].id,
                })
                if skip_duplicates:
                    continue

            # Insert transaction
            create_payload = TransactionCreate(
                date=iso_date,
                description=description,
                amount_paise=amount_paise,
                currency=currency,
                transaction_type=txn_type,
                category=category,
                notes=notes,
            )
            record = self.add_transaction(create_payload)
            imported_records.append(record)

        return {
            "imported_count": len(imported_records),
            "skipped_duplicates": len(duplicate_candidates) if skip_duplicates else 0,
            "duplicate_candidates": duplicate_candidates,
            "invalid_rows": invalid_rows,
            "transactions": imported_records,
        }

    def export_to_dataframe(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
        currency: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Extract transactions into a pandas DataFrame ready for financial analytics.
        """
        txns = self.list_transactions(start_date=start_date, end_date=end_date, category=category, currency=currency)
        if not txns:
            return pd.DataFrame(
                columns=[
                    "id", "date", "description", "amount_paise", "amount_rupees",
                    "currency", "transaction_type", "category", "notes", "created_at",
                ]
            )

        data = [
            {
                "id": t.id,
                "date": t.date,
                "description": t.description,
                "amount_paise": t.amount_paise,
                "amount_rupees": t.amount_rupees,
                "currency": t.currency,
                "transaction_type": t.transaction_type.value,
                "category": t.category,
                "notes": t.notes,
                "created_at": t.created_at,
            }
            for t in txns
        ]
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        return df
