# BudgetWise AI - Domain-Specific Personal Budgeting Assistant

BudgetWise AI is a domain-specific personal finance and budgeting chatbot designed to provide rigorous, verifiable insights into personal finances. It pairs OpenAI models for conversational natural language understanding with deterministic engines: **pandas** for transaction analysis and statistical aggregations, and **SQLite** for ACID-compliant persistent local storage.

---

## 1. Architecture & Deterministic Data Flow

BudgetWise AI enforces strict separation between numerical computation and conversational synthesis. The language model is never permitted to invent balances, extrapolate spending trends, or calculate sums.

```
                           User Request (Chat or Web UI)
                                         │
                                         ▼
                             Input Validation (Pydantic)
                                         │
                                         ▼
                     Domain Intent Classification & Entity Extraction
                                         │
                                         ▼
                        Service Routing & Deterministic DB/Pandas
           ┌─────────────────────────────┼─────────────────────────────┐
           ▼                             ▼                             ▼
   Transaction Service             Budget Service              Analytics Engine
   (SQLite CRUD, Ingestion)    (Category Limits, 80%/100%)    (Totals, Variances, Surges)
           │                             │                             │
           └─────────────────────────────┼─────────────────────────────┘
                                         ▼
                          Verified Financial Calculation
                                         │
                                         ▼
                    OpenAI Synthesis with Strict Policy Guardrails
                  (Prompts prohibit hallucination, mandate evidence)
                                         │
                                         ▼
                           User-Facing Grounded Response
```

---

## 2. Core Capabilities

### A. Transaction & Statement Management
- **Persistent Storage:** Stored in SQLite with integer minor units (`amount_paise` for INR) to eliminate floating-point rounding errors.
- **Cash Flow Types:** Categorizes transactions into `income`, `expense`, `transfer`, `refund`, and `investment` to avoid double-counting.
- **CSV Statement Ingestion:** Ingests bank and card statements using pandas with flexible column normalization (`date`, `description`, `amount`).
- **Duplicate Detection:** Detects existing transactions matching the same date and amount, offering automatic deduplication.

### B. Domain-Specific Expense Categorization
- Predefined financial categories:
  - *Food and Dining*, *Groceries*, *Transportation*, *Housing and Rent*, *Utilities*, *Shopping*, *Healthcare*, *Education*, *Entertainment*, *Subscriptions*, *Travel*, *Personal Care*, *EMI and Debt Payments*, *Investments and Savings*, *Income*, *Other*.
- **Classification Pipeline:**
  1. **Persistent User Overrides:** User corrections are prioritized above all rules.
  2. **Keyword Rules:** Direct mapping for Indian and international merchants (Swiggy, Blinkit, DMart, BESCOM, Netflix, Uber, etc.).
  3. **ML Baseline Model:** Scikit-learn TF-IDF + Naive Bayes classifier for ambiguous merchant descriptions.
  4. **Confidence Bands:** High, Medium, Low, Unknown. Flags need for confirmation if confidence is below threshold.

### C. Monthly Category Budgeting
- Monthly category limits (e.g., ₹5,000 for Food and Dining in 2026-10).
- Real-time limit tracking: calculates actual spending, remaining allowance, and utilization percentage.
- **Threshold Warnings:** Automatic alerts at 80% (Warning) and 100% (Critical / Over Budget).
- **Refund Policy:** Category refunds directly offset gross expenditure, preventing skewed budget utilization.

### D. Pandas Financial Analytics
- Monthly income vs. expense totals, net savings, and savings rate percentage.
- Category expenditure distributions and rankings.
- Month-over-month comparisons with explicit warnings when comparing partial billing periods.
- **Spending Surge Detection:** Statistical anomaly detection using neutral terminology ("unusual increase worth reviewing", never "fraudulent").
- **Subscription Candidates:** Automatic discovery of monthly recurring debits.

### E. Conversational Chatbot
- Handles natural queries such as:
  - *"How much did I spend this month?"*
  - *"Where is most of my money going?"*
  - *"How much did I spend on food last month?"*
  - *"Am I over my shopping budget?"*
  - *"Compare my expenses with last month."*
  - *"What subscriptions am I paying for?"*
  - *"How much money did I save this month?"*
  - *"Add ₹450 for groceries yesterday."*
  - *"Set my monthly entertainment budget to ₹2,000."*
- **Offline / Mock Resilience:** Operates with 100% accuracy in deterministic mode even when an OpenAI API key is not present.
- **OpenAI Polishing:** When `OPENAI_API_KEY` is provided, enhances conversational naturalness while enforcing strict factual grounding.

---

## 3. Installation & Setup

### Prerequisites
- Python 3.11 or newer.
- Git.

### Step 1: Navigate to Project Directory
```bash
cd budgetwise-ai
```

### Step 2: Set Up Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Environment Configuration

1. Copy the template:
   ```bash
   cp .env.example .env
   ```
2. Configure settings in `.env`:
   ```ini
   OPENAI_API_KEY=your_actual_key_here  # Optional: deterministic mode runs without key
   OPENAI_MODEL=gpt-4o-mini
   DATABASE_PATH=data/budgetwise.db
   DEFAULT_CURRENCY=INR
   CATEGORIZER_CONFIDENCE_THRESHOLD=0.70
   BUDGET_WARNING_THRESHOLD=0.80
   BUDGET_CRITICAL_THRESHOLD=1.00
   ```
   > **Note:** The application starts and functions completely even if `OPENAI_API_KEY` is empty. You can also provide an API key temporarily in the Streamlit UI sidebar for active sessions.

---

## 5. Running the Application

Launch the Streamlit web interface:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

**Quick Start in UI:**
- Click **"📥 Load Synthetic Sample CSV"** in the sidebar to populate the system with realistic sample transactions.
- Switch to the **Conversational Chat** tab to interact with the assistant.
- Explore the **Financial Dashboard** and **Monthly Budgets** tabs.

---

## 6. Running Tests & Benchmarks

### Automated Test Suite
Run all unit and integration tests across database, categorizer, transactions, budgets, analytics, chatbot, and evaluation:
```bash
pytest -v
```

### Evaluation Benchmark
Run the repeatable evaluation benchmark:
```bash
python evaluate.py
```
This generates a markdown benchmark report measuring:
- Expense categorization accuracy and per-category Precision / Recall / F1.
- Domain intent-routing accuracy across typical user prompts.
- Deterministic calculation correctness and 0% arithmetic hallucination rate.

---

## 7. Security & Privacy Considerations

- **Zero Secret Exposure:** `.env` and `.streamlit/secrets.toml` are strictly ignored by `.gitignore`.
- **Database Privacy:** SQLite databases (`data/*.db`) and personal CSV exports are never committed to version control.
- **SQL Injection Prevention:** 100% of SQLite queries use parameterized placeholders (`?`).
- **Safe Monetary Math:** All monetary balances are stored as 64-bit integer minor units (paise) to guarantee arithmetic precision.
