"""
BudgetWise AI - Domain Models & Data Schemas.

Defines validated Pydantic models, enums, and conversion utilities.
Monetary amounts are strictly handled in integer minor units (e.g. paise for INR)
to eliminate floating-point rounding errors.
"""

from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator


def rupees_to_paise(rupees: float | int | str | Decimal) -> int:
    """
    Convert a rupee value to integer paise minor units using exact decimal rounding.
    Example: 450.50 -> 45050 paise.
    """
    dec_val = Decimal(str(rupees))
    return int((dec_val * Decimal(100)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def paise_to_rupees(paise: int) -> float:
    """
    Convert integer paise minor units to floating-point rupees for presentation.
    Example: 45050 -> 450.50.
    """
    return round(paise / 100.0, 2)


class TransactionType(str, Enum):
    """Supported cash-flow transaction classifications."""
    INCOME = "income"
    EXPENSE = "expense"
    TRANSFER = "transfer"
    REFUND = "refund"
    INVESTMENT = "investment"


class StandardCategory(str, Enum):
    """
    Standard financial categories.
    Income is kept separate from expense categories.
    Transfers, refunds, and investments are treated distinctly to avoid double counting.
    """
    FOOD_AND_DINING = "Food and Dining"
    GROCERIES = "Groceries"
    TRANSPORTATION = "Transportation"
    HOUSING_AND_RENT = "Housing and Rent"
    UTILITIES = "Utilities"
    SHOPPING = "Shopping"
    HEALTHCARE = "Healthcare"
    EDUCATION = "Education"
    ENTERTAINMENT = "Entertainment"
    SUBSCRIPTIONS = "Subscriptions"
    TRAVEL = "Travel"
    PERSONAL_CARE = "Personal Care"
    EMI_AND_DEBT = "EMI and Debt Payments"
    INVESTMENTS_AND_SAVINGS = "Investments and Savings"
    INCOME = "Income"
    OTHER = "Other"

    @classmethod
    def get_expense_categories(cls) -> list[str]:
        """Return categories that represent direct monthly spending."""
        excluded = {cls.INCOME.value, cls.INVESTMENTS_AND_SAVINGS.value}
        return [c.value for c in cls if c.value not in excluded]


class ConfidenceBand(str, Enum):
    """Explicit confidence classification for categorization transparency."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class ClassificationMethod(str, Enum):
    """Mechanism utilized to categorize an item."""
    USER_OVERRIDE = "user_override"
    EXACT_RULE = "exact_rule"
    KEYWORD_RULE = "keyword_rule"
    ML_MODEL = "ml_model"
    FALLBACK = "fallback"


class CategoryPrediction(BaseModel):
    """Structured result of an expense categorization attempt."""
    predicted_category: str
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence_band: ConfidenceBand
    classification_method: ClassificationMethod
    explanation: str
    needs_confirmation: bool


class TransactionBase(BaseModel):
    """Base fields for financial transactions."""
    date: str = Field(description="Transaction date in ISO format YYYY-MM-DD.")
    description: str = Field(min_length=1, max_length=255, description="Merchant or transaction summary.")
    amount_paise: int = Field(gt=0, description="Amount in integer minor units (e.g. paise).")
    currency: str = Field(default="INR", max_length=3, description="Three-letter currency code (ISO 4217).")
    transaction_type: TransactionType = Field(default=TransactionType.EXPENSE, description="Transaction type.")
    category: str = Field(default=StandardCategory.OTHER.value, description="Assigned category.")
    notes: Optional[str] = Field(default=None, max_length=1000, description="Optional metadata or notes.")

    @field_validator("date")
    @classmethod
    def validate_iso_date(cls, v: str) -> str:
        """Ensure date conforms strictly to YYYY-MM-DD."""
        try:
            parsed = date.fromisoformat(v)
            return parsed.isoformat()
        except ValueError as exc:
            raise ValueError(f"Invalid date format '{v}'. Expected YYYY-MM-DD.") from exc

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, v: str) -> str:
        return v.strip().upper()


class TransactionCreate(TransactionBase):
    """Schema for creating a new transaction record."""
    pass


class TransactionUpdate(BaseModel):
    """Schema for updating an existing transaction."""
    date: Optional[str] = None
    description: Optional[str] = None
    amount_paise: Optional[int] = Field(default=None, gt=0)
    currency: Optional[str] = None
    transaction_type: Optional[TransactionType] = None
    category: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("date")
    @classmethod
    def validate_iso_date(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        try:
            return date.fromisoformat(v).isoformat()
        except ValueError as exc:
            raise ValueError(f"Invalid date format '{v}'. Expected YYYY-MM-DD.") from exc


class TransactionRecord(TransactionBase):
    """Schema for transactions retrieved from SQLite."""
    id: int
    created_at: str
    updated_at: str

    @property
    def amount_rupees(self) -> float:
        """Helper property to access amount in rupees."""
        return paise_to_rupees(self.amount_paise)


class BudgetBase(BaseModel):
    """Base fields for monthly category budgets."""
    category: str = Field(description="Category name to cap spending on.")
    monthly_limit_paise: int = Field(gt=0, description="Monthly spending limit in integer paise.")
    month: str = Field(description="Target month in YYYY-MM format.")
    currency: str = Field(default="INR", max_length=3, description="Currency code.")

    @field_validator("month")
    @classmethod
    def validate_year_month(cls, v: str) -> str:
        parts = v.strip().split("-")
        if len(parts) != 2 or len(parts[0]) != 4 or len(parts[1]) != 2:
            raise ValueError(f"Invalid month format '{v}'. Expected YYYY-MM.")
        try:
            y, m = int(parts[0]), int(parts[1])
            if not (1 <= m <= 12):
                raise ValueError("Month must be between 01 and 12.")
            return f"{y:04d}-{m:02d}"
        except ValueError as exc:
            raise ValueError(f"Invalid month '{v}'. Expected YYYY-MM.") from exc


class BudgetCreate(BudgetBase):
    """Schema for creating a budget limit."""
    pass


class BudgetUpdate(BaseModel):
    """Schema for modifying an existing budget limit."""
    category: Optional[str] = None
    monthly_limit_paise: Optional[int] = Field(default=None, gt=0)
    month: Optional[str] = None
    currency: Optional[str] = None


class BudgetRecord(BudgetBase):
    """Persisted budget retrieved from SQLite."""
    id: int
    created_at: str
    updated_at: str

    @property
    def monthly_limit_rupees(self) -> float:
        return paise_to_rupees(self.monthly_limit_paise)


class BudgetStatus(BaseModel):
    """Detailed evaluation of spending against a budget limit."""
    category: str
    month: str
    currency: str
    limit_paise: int
    actual_spent_paise: int
    remaining_paise: int
    utilization_percentage: float
    is_warning: bool
    is_over_budget: bool
    alert_level: str  # NORMAL, WARNING (>= 80%), CRITICAL (>= 100%)

    @property
    def limit_rupees(self) -> float:
        return paise_to_rupees(self.limit_paise)

    @property
    def actual_spent_rupees(self) -> float:
        return paise_to_rupees(self.actual_spent_paise)

    @property
    def remaining_rupees(self) -> float:
        return paise_to_rupees(self.remaining_paise)


class ChatQuery(BaseModel):
    """User input prompt or command for the chatbot."""
    message: str = Field(min_length=1, description="Raw natural-language user query.")
    session_id: str = Field(default="default_session", description="Conversation session ID.")


class ChatResponse(BaseModel):
    """Structured response from the BudgetWise chatbot orchestrator."""
    reply: str
    intent: Optional[str] = None
    action_executed: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    requires_confirmation: bool = False
    pending_action: Optional[Dict[str, Any]] = None
