"""

BudgetWise AI - Streamlit Web Application.



A domain-specific personal budgeting and financial assistant interface.

Provides conversational AI, deterministic analytics, budget limit tracking,

and CSV statement management.

"""



import sys

from datetime import date, datetime

from pathlib import Path

import pandas as pd

import streamlit as st



# Ensure src/ is on sys.path for local Streamlit execution

_SRC_DIR = Path(__file__).resolve().parent / "src"

if str(_SRC_DIR) not in sys.path:

          sys.path.insert(0, str(_SRC_DIR))



from budgetwise.analytics import FinancialAnalytics

from budgetwise.budget_service import BudgetService

from budgetwise.categorizer import ExpenseCategorizer

from budgetwise.chatbot import BudgetWiseChatbot

from budgetwise.config import get_settings

from budgetwise.database import DatabaseManager

from budgetwise.models import (

          BudgetCreate,

          ChatQuery,

          StandardCategory,

          TransactionCreate,

          TransactionType,

          paise_to_rupees,

          rupees_to_paise,

)

from budgetwise.transaction_service import TransactionService





# ---------------------------------------------------------------------------

# Visual theme (presentation only)

# ---------------------------------------------------------------------------

ACCENT = "#0D6E46"
CUSTOM_CSS = f"""
<style>
:root {{
    --page: #D5EBDC;
    --surface: #FFFFFF;
    --surface-muted: #C5E4CE;
    --text: #082618;
    --muted: #244C36;
    --border: #A3D1B3;
    --accent: {ACCENT};
    --accent-hover: #084E31;
    --accent-soft: #BFE4CE;
    --success-bg: #C7ECD3;
    --success-text: #0E562F;
    --warning-bg: #FEF3C7;
    --warning-text: #85520B;
    --danger-bg: #FDE8E8;
    --danger-text: #9B1C1C;
}}

html, body, [class*="css"], .stApp {{
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: var(--text);
}}
.stApp {{ background: var(--page); background-color: var(--page); }}
.block-container {{ max-width: 1440px; padding: 2rem clamp(1.25rem, 3vw, 3rem) 3.25rem; }}
header[data-testid="stHeader"] {{ background: rgba(213, 235, 220, 0.95); }}
#MainMenu, footer {{ visibility: hidden; }}

h1, h2, h3, h4 {{ color: var(--text); letter-spacing: -.028em; font-weight: 680; }}
h1 {{ font-size: 2rem; }}
h2 {{ font-size: 1.45rem; margin-bottom: .25rem; }}
h3 {{ font-size: 1.08rem; }}
p, label, [data-testid="stCaptionContainer"] {{ color: var(--muted); }}
.bw-hero {{ margin: .2rem 0 1.4rem; padding: .4rem 0 1.1rem; }}
.bw-hero-start {{ height: .15rem; }}
[data-testid="stAppViewContainer"] {{ background: var(--page); background-color: var(--page); }}
[data-testid="stSidebar"] {{ box-shadow: 1px 0 0 rgba(20,37,29,.05); }}
.bw-hero h1 {{ font-size: clamp(1.85rem, 3vw, 2.35rem); margin: 0; color: var(--text); letter-spacing: -.045em; }}
.bw-hero p {{ color: var(--muted); margin: .45rem 0 0; font-size: 1rem; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background: #C2E2CC; background-color: #C2E2CC; border-right: 1px solid var(--border); }}
section[data-testid="stSidebar"] > div {{ padding-top: 1.35rem; }}
section[data-testid="stSidebar"] h1 {{ color: var(--text); font-size: 1.35rem; letter-spacing: -.035em; }}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ color: var(--muted); }}

/* Navigation */
.stTabs [data-baseweb="tab-list"] {{ gap: .35rem; padding: .32rem; background: #BDE0C7; border: 1px solid var(--border); border-radius: 13px; }}
.stTabs [data-baseweb="tab"] {{ min-height: 2.75rem; padding: .15rem 1rem; color: #1D4B34; border-radius: 9px; font-weight: 600; background: transparent; }}
.stTabs [data-baseweb="tab"]:hover {{ color: var(--accent); background: #E5F4EA; }}
.stTabs [aria-selected="true"] {{ color: var(--accent) !important; background: var(--surface) !important; box-shadow: 0 1px 4px rgba(8,38,24,.12); }}
.stTabs [data-baseweb="tab-highlight"] {{ background: transparent; }}

/* Metric panels */
[data-testid="stMetric"] {{ min-height: 118px; background: var(--surface); border: 1px solid var(--border); border-radius: 15px; padding: 1.05rem 1.15rem; box-shadow: 0 3px 10px rgba(8,38,24,.035); }}
[data-testid="stMetricLabel"] p {{ color: var(--muted); font-size: .76rem; font-weight: 700; text-transform: uppercase; letter-spacing: .055em; }}
[data-testid="stMetricValue"] {{ color: var(--text); font-weight: 750; letter-spacing: -.04em; }}
[data-testid="stMetricDelta"] {{ font-weight: 650; }}

/* Interactive controls */
.stButton > button, .stFormSubmitButton > button {{ min-height: 2.65rem; border: 1px solid #A3D1B3; border-radius: 10px; background: var(--surface); color: var(--text); font-weight: 620; transition: background 120ms ease, border-color 120ms ease; box-shadow: 0 1px 2px rgba(8,38,24,.03); }}
.stButton > button:hover, .stFormSubmitButton > button:hover {{ border-color: var(--accent); color: var(--accent); background: var(--accent-soft); }}
.stButton > button:focus-visible, .stFormSubmitButton > button:focus-visible {{ outline: 3px solid #75C296; outline-offset: 2px; }}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{ background: var(--accent); border-color: var(--accent); color: #FFF; }}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {{ background: var(--accent-hover); border-color: var(--accent-hover); color: #FFF; }}
[data-testid="stForm"] {{ background: var(--surface); border: 1px solid var(--border); border-radius: 15px; padding: 1.2rem; box-shadow: 0 3px 10px rgba(8,38,24,.03); }}
.stTextInput input, .stNumberInput input, .stDateInput input, [data-baseweb="input"] input, [data-baseweb="textarea"] textarea {{ color: var(--text); border-radius: 9px; }}
[data-baseweb="select"] > div {{ border-radius: 9px; }}
[data-baseweb="input"]:focus-within, [data-baseweb="select"]:focus-within, [data-baseweb="textarea"]:focus-within {{ border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }}
[data-testid="stFileUploader"] section {{ background: #E7F5EC; border: 1px dashed #87C59E; border-radius: 12px; }}

/* Chat */
[data-testid="stChatMessage"] {{ background: var(--surface); border: 1px solid var(--border); border-radius: 15px; padding: .9rem 1rem; margin-bottom: .65rem; }}
[data-testid="stChatInput"] {{ border-color: #A3D1B3; border-radius: 13px; background: var(--surface); }}
[data-testid="stChatInput"]:focus-within {{ border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }}

/* Data, feedback and budget cards */
[data-testid="stDataFrame"] {{ border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
[data-testid="stAlert"] {{ border-radius: 11px; }}
.stProgress > div > div > div > div {{ background-color: var(--accent); }}
hr {{ border-color: var(--border); }}
[data-testid="stExpander"] {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; }}
.bw-budget-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 1rem 1.1rem .8rem; margin-bottom: .7rem; box-shadow: 0 3px 9px rgba(31,45,38,.025); }}
.bw-budget-head {{ display: flex; justify-content: space-between; align-items: center; gap: .75rem; margin-bottom: .55rem; }}
.bw-budget-title {{ font-weight: 680; color: var(--text); }}
.bw-pill {{ white-space: nowrap; font-size: .72rem; font-weight: 750; padding: .22rem .62rem; border-radius: 999px; letter-spacing: .01em; }}
.bw-pill-ok {{ background: var(--success-bg); color: var(--success-text); }}
.bw-pill-warn {{ background: var(--warning-bg); color: var(--warning-text); }}
.bw-pill-bad {{ background: var(--danger-bg); color: var(--danger-text); }}
.bw-budget-meta {{ color: var(--muted); font-size: .86rem; line-height: 1.8; }}
.bw-budget-meta b {{ color: var(--text); font-weight: 680; }}

@media (max-width: 768px) {{
    .block-container {{ padding-top: 1rem; padding-left: 1rem; padding-right: 1rem; }}
    .stTabs [data-baseweb="tab"] {{ padding: .1rem .55rem; font-size: .83rem; }}
    [data-testid="stMetric"] {{ padding: .8rem; min-height: 100px; }}
}}
</style>
"""


@st.cache_resource

def get_services():

          """Initialize and cache database and core domain services."""

          db_mgr = DatabaseManager()

          db_mgr.init_schema()

          categorizer = ExpenseCategorizer(db_manager=db_mgr)

          txn_service = TransactionService(db_manager=db_mgr, categorizer=categorizer)

          bgt_service = BudgetService(db_manager=db_mgr)

          analytics = FinancialAnalytics()

          chatbot = BudgetWiseChatbot(

                    transaction_service=txn_service,

                    budget_service=bgt_service,

                    analytics=analytics,

                    categorizer=categorizer,

                    db_manager=db_mgr,

          )

          return db_mgr, txn_service, bgt_service, analytics, categorizer, chatbot





def main() -> None:

          st.set_page_config(

                    page_title="BudgetWise AI - Personal Budgeting",

                    page_icon="💰",

                    layout="wide",

                    initial_sidebar_state="expanded",

          )

          st.markdown(CUSTOM_CSS, unsafe_allow_html=True)



          db_mgr, txn_service, bgt_service, analytics, categorizer, chatbot = get_services()

          settings = get_settings()



          # --- Sidebar Configuration ---

          st.sidebar.title("BudgetWise AI")

          st.sidebar.caption("Your personal finance companion")



          # Currency & Period Controls

          currency = settings.default_currency

          all_txns_df = txn_service.export_to_dataframe(currency=currency)



          available_months = [datetime.now().strftime("%Y-%m")]

          if not all_txns_df.empty:

                    df_months = pd.to_datetime(all_txns_df["date"]).dt.strftime("%Y-%m").unique().tolist()

                    for m in df_months:

                              if m not in available_months:

                                        available_months.append(m)

          available_months.sort(reverse=True)



          st.sidebar.markdown("##### Period")

          selected_month = st.sidebar.selectbox(

                    "Billing month", options=available_months, index=0, label_visibility="collapsed"

          )

          st.sidebar.caption(f"Currency: {currency} (₹)")



          # API Key Configuration in Sidebar

          st.sidebar.divider()

          st.sidebar.markdown("##### AI model")

          openai_ready = settings.is_openai_configured()



          if openai_ready:

                    st.sidebar.success(f"OpenAI active (`{settings.openai_model}`)")

          else:

                    st.sidebar.info("Running on the built-in engine. Add an OpenAI key below for richer replies.")

                    session_key = st.sidebar.text_input(

                              "OpenAI key (this session only)",

                              type="password",

                              help="Your key remains in local session memory and is never committed.",

                    )

                    if session_key.strip():

                              settings.openai_api_key = session_key.strip()

                              st.sidebar.success("Session key applied")



          # Demo Dataset Ingestion Helper

          st.sidebar.divider()

          st.sidebar.markdown("##### Try it out")

          sample_csv_path = Path("data/samples/sample_transactions.csv")

          if sample_csv_path.exists():

                    if st.sidebar.button("Load sample data", help="Imports anonymized Indian transactions", use_container_width=True):

                              result = txn_service.import_from_csv(sample_csv_path, currency=currency, skip_duplicates=True)

                              st.sidebar.success(f"Imported {result['imported_count']} sample transactions")

                              st.rerun()



          # --- Page header ---
          # Use native Streamlit text components here instead of an HTML block.
          # This avoids raw HTML being displayed as code in environments where
          # markdown sanitization or indentation handling differs.
          with st.container():
                    st.markdown('<div class="bw-hero-start"></div>', unsafe_allow_html=True)
                    st.title("Welcome back")
                    st.caption("Here's a clear view of your money, one month at a time.")



          # --- Main Navigation Tabs ---

          tab_chat, tab_dashboard, tab_transactions, tab_budgets = st.tabs([

                    "Assistant",

                    "Dashboard",

                    "Transactions",

                    "Budgets",

          ])



          # =========================================================================

          # TAB 1: Conversational Chat

          # =========================================================================

          with tab_chat:

                    st.header("Financial assistant")

                    st.caption("Ask about your spending, check your limits, or log an expense in plain words.")



                    # Quick action prompt buttons

                    st.markdown("##### Suggestions")

                    q_cols = st.columns(4)

                    sample_queries = [

                              "How much did I spend this month?",

                              "Where is most of my money going?",

                              "Am I over my shopping budget?",

                              "Compare my expenses with last month.",

                    ]

                    chosen_prompt = None

                    for i, q in enumerate(sample_queries):

                              if q_cols[i].button(q, key=f"quick_p_{i}", use_container_width=True):

                                        chosen_prompt = q



                    st.write("")



                    # Chat session state

                    if "messages" not in st.session_state:

                              st.session_state.messages = [

                                        {

                                                  "role": "assistant",

                                                  "content": (

                                                            "Hello! I am BudgetWise AI. I can calculate your spending totals, "

                                                            "track your monthly category budgets, check for unusual increases, "

                                                            "and record your manual expenses. How can I help you today?"

                                                  ),

                                        }

                              ]



                    # Display message history

                    for msg in st.session_state.messages:

                              with st.chat_message(msg["role"]):

                                        st.markdown(msg["content"])



                    # Input handling

                    user_input = st.chat_input("Ask a question or record an expense...")

                    if chosen_prompt:

                              user_input = chosen_prompt



                    if user_input:

                              st.session_state.messages.append({"role": "user", "content": user_input})

                              with st.chat_message("user"):

                                        st.markdown(user_input)



                              with st.chat_message("assistant"):

                                        with st.spinner("Looking through your records..."):

                                                  resp = chatbot.process_message(ChatQuery(message=user_input))

                                                  st.markdown(resp.reply)

                                                  st.session_state.messages.append({"role": "assistant", "content": resp.reply})



          # =========================================================================

          # TAB 2: Financial Dashboard

          # =========================================================================

          with tab_dashboard:

                    st.header(f"Summary for {selected_month}")

                    st.caption("Your income, spending and savings at a glance.")



                    # Filter transactions for selected month

                    df_month = (

                              all_txns_df[pd.to_datetime(all_txns_df["date"]).dt.strftime("%Y-%m") == selected_month]

                              if not all_txns_df.empty

                              else pd.DataFrame()

                    )



                    summary = analytics.calculate_summary(df_month, period_label=selected_month, currency=currency)



                    # KPI Metric Cards

                    c1, c2, c3, c4 = st.columns(4)

                    c1.metric(

                              label=f"Income ({currency})",

                              value=f"₹{summary['total_income_rupees']:,.2f}",

                              help="Total recorded earnings and cash inflows.",

                    )

                    c2.metric(

                              label=f"Expenses ({currency})",

                              value=f"₹{summary['total_expense_rupees']:,.2f}",

                              help="Net direct consumption expenses (gross expense minus refunds).",

                    )

                    c3.metric(

                              label=f"Net savings ({currency})",

                              value=f"₹{summary['net_savings_rupees']:,.2f}",

                              delta=f"{summary['savings_rate_percentage']:.1f}% Savings Rate",

                              help="Income minus net expenses.",

                    )

                    c4.metric(

                              label="Transactions",

                              value=f"{summary['transaction_count']} txns",

                              help="Total entries recorded in this billing period.",

                    )



                    st.write("")



                    if df_month.empty:

                              st.info(f"Nothing recorded for **{selected_month}** yet. Add a transaction or import a CSV statement to see your charts.")

                    else:

                              col_chart1, col_chart2 = st.columns(2)



                              with col_chart1:

                                        st.subheader("Spending by category")

                                        breakdown = analytics.category_breakdown(df_month, currency=currency)

                                        if breakdown["categories"]:

                                                  cat_df = pd.DataFrame(breakdown["categories"])

                                                  st.bar_chart(

                                                            data=cat_df.set_index("category")["total_rupees"],

                                                            use_container_width=True,

                                                            color=ACCENT,

                                                  )

                                        else:

                                                  st.write("No category expenses to display.")



                              with col_chart2:

                                        st.subheader("Monthly spending trend")

                                        if not all_txns_df.empty:

                                                  exp_df = all_txns_df[all_txns_df["transaction_type"] == "expense"].copy()

                                                  if not exp_df.empty:

                                                            exp_df["month"] = pd.to_datetime(exp_df["date"]).dt.strftime("%Y-%m")

                                                            monthly_trend = exp_df.groupby("month")["amount_rupees"].sum().reset_index()

                                                            st.line_chart(

                                                                      data=monthly_trend.set_index("month")["amount_rupees"],

                                                                      use_container_width=True,

                                                                      color=ACCENT,

                                                            )



                              st.divider()

                              st.subheader("Insights")

                              i_col1, i_col2 = st.columns(2)



                              with i_col1:

                                        st.markdown("**Unusual increases**")

                                        anomalies = analytics.detect_spending_anomalies(all_txns_df, current_month=selected_month, currency=currency)

                                        detected = anomalies.get("anomalies_detected", [])

                                        if not detected:

                                                  st.success("All steady. No unusual category spikes this month.")

                                        else:

                                                  for a in detected:

                                                            st.warning(

                                                                      f"**{a['category']}**: Spent ₹{a['target_spend_rupees']:,.2f} "

                                                                      f"(baseline avg ₹{a['baseline_average_rupees']:,.2f}, +{a['increase_factor']}). "

                                                                      f"*{a['status']}*."

                                                            )



                              with i_col2:

                                        st.markdown("**Recurring subscriptions**")

                                        subs = analytics.find_recurring_candidates(all_txns_df, currency=currency)

                                        if not subs:

                                                  st.info("No recurring monthly subscriptions found.")

                                        else:

                                                  sub_df = pd.DataFrame(subs)[["description", "category", "amount_rupees", "frequency"]]

                                                  st.dataframe(sub_df, use_container_width=True, hide_index=True)



          # =========================================================================

          # TAB 3: Transactions & CSV Import

          # =========================================================================

          with tab_transactions:

                    st.header("Transactions")

                    st.caption("Add entries by hand or bring in a bank statement.")



                    t_sub1, t_sub2 = st.tabs(["Add manually", "Import CSV"])



                    with t_sub1:

                              st.subheader("New transaction")

                              with st.form("manual_tx_form", clear_on_submit=True):

                                        f_date = st.date_input("Date", value=date.today())

                                        f_desc = st.text_input("Merchant / Description", placeholder="e.g. Swiggy Food Delivery, Blinkit")

                                        f_amount = st.number_input("Amount (₹)", min_value=1.0, value=250.0, step=10.0)

                                        f_type = st.selectbox(

                                                  "Transaction Type",

                                                  options=[t.value for t in TransactionType],

                                                  index=1,     # Default to 'expense'

                                        )

                                        f_cat = st.selectbox(

                                                  "Category (Leave as 'Other' to auto-categorize)",

                                                  options=[c.value for c in StandardCategory],

                                                  index=len(StandardCategory) - 1,

                                        )

                                        f_notes = st.text_input("Notes (optional)", placeholder="e.g. Lunch with team")

                                        submit_btn = st.form_submit_button("Record transaction", type="primary", use_container_width=True)



                                        if submit_btn:

                                                  if not f_desc.strip():

                                                            st.error("Please add a description.")

                                                  else:

                                                            cat_to_use = f_cat

                                                            if cat_to_use == "Other":

                                                                      pred = categorizer.predict(f_desc, f_type)

                                                                      cat_to_use = pred.predicted_category



                                                            txn_service.add_transaction(

                                                                      TransactionCreate(

                                                                                date=f_date.isoformat(),

                                                                                description=f_desc.strip(),

                                                                                amount_paise=rupees_to_paise(f_amount),

                                                                                currency=currency,

                                                                                transaction_type=TransactionType(f_type),

                                                                                category=cat_to_use,

                                                                                notes=f_notes.strip() if f_notes else None,

                                                                      )

                                                            )

                                                            st.success(f"Recorded transaction: ₹{f_amount:,.2f} for '{f_desc}' under '{cat_to_use}'!")

                                                            st.rerun()



                    with t_sub2:

                              st.subheader("Import a bank or card statement")

                              st.markdown(

                                        "Upload any standard CSV statement containing Date, Description, and Amount columns. "

                                        "Headers, dates and categories are sorted out for you."

                              )

                              MAX_CSV_SIZE_BYTES = 10 * 1024 * 1024     # 10 MB Limit

                              uploaded_file = st.file_uploader("Upload CSV Statement", type=["csv"])

                              skip_dupes = st.checkbox("Automatically skip duplicate transactions (same date & amount)", value=True)



                              if uploaded_file is not None:

                                        if uploaded_file.size > MAX_CSV_SIZE_BYTES:

                                                  st.error(f"File size exceeds maximum allowed limit of 10 MB ({uploaded_file.size / (1024*1024):.1f} MB uploaded).")

                                        elif st.button("Import statement", type="primary"):

                                                  try:

                                                            content = uploaded_file.getvalue().decode("utf-8")

                                                            import_res = txn_service.import_from_csv(content, currency=currency, skip_duplicates=skip_dupes)

                                                            st.success(

                                                                      f"Successfully imported **{import_res['imported_count']}** transactions! "

                                                                      f"(Skipped **{import_res['skipped_duplicates']}** duplicates, "

                                                                      f"**{import_res['invalid_rows']}** invalid rows)."

                                                            )

                                                            if import_res["duplicate_candidates"]:

                                                                      with st.expander("View skipped duplicates"):

                                                                                st.write(import_res["duplicate_candidates"])

                                                            st.rerun()

                                                  except UnicodeDecodeError:

                                                            st.error("Invalid file encoding. Please upload a standard UTF-8 encoded CSV file.")

                                                  except Exception as err:

                                                            st.error(f"Error importing CSV: {str(err)}")



                    st.divider()

                    st.subheader("History")



                    # Category and Search Filter

                    f_col1, f_col2 = st.columns([1, 2])

                    filter_cat = f_col1.selectbox("Category", options=["All"] + [c.value for c in StandardCategory])

                    search_kw = f_col2.text_input("Search", placeholder="e.g. Swiggy, Netflix, Salary")



                    all_records = txn_service.list_transactions(currency=currency)

                    if not all_records:

                              st.info("No transactions yet. Add your first one above.")

                    else:

                              filtered_records = all_records

                              if filter_cat != "All":

                                        filtered_records = [r for r in filtered_records if r.category == filter_cat]

                              if search_kw.strip():

                                        filtered_records = [r for r in filtered_records if search_kw.lower() in r.description.lower()]



                              if not filtered_records:

                                        st.warning("No transactions match your filters.")

                              else:

                                        table_data = [

                                                  {

                                                            "ID": r.id,

                                                            "Date": r.date,

                                                            "Description": r.description,

                                                            "Amount (₹)": f"₹{r.amount_rupees:,.2f}",

                                                            "Type": r.transaction_type.value.capitalize(),

                                                            "Category": r.category,

                                                            "Notes": r.notes or "-",

                                                  }

                                                  for r in filtered_records

                                        ]

                                        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)



          # =========================================================================

          # TAB 4: Monthly Budgets

          # =========================================================================

          with tab_budgets:

                    st.header(f"Budgets for {selected_month}")

                    st.caption("Set a limit per category and keep an eye on how you're pacing.")



                    b_col1, b_col2 = st.columns([1, 2])



                    with b_col1:

                              st.subheader("Set a limit")

                              with st.form("set_budget_form", clear_on_submit=True):

                                        b_cat = st.selectbox("Category", options=StandardCategory.get_expense_categories())

                                        b_limit = st.number_input("Monthly Limit (₹)", min_value=100.0, value=5000.0, step=500.0)

                                        b_submit = st.form_submit_button("Save limit", type="primary", use_container_width=True)



                                        if b_submit:

                                                  bgt_service.set_budget(

                                                            BudgetCreate(

                                                                      category=b_cat,

                                                                      monthly_limit_paise=rupees_to_paise(b_limit),

                                                                      month=selected_month,

                                                                      currency=currency,

                                                            )

                                                  )

                                                  st.success(f"Configured budget for **{b_cat}**: ₹{b_limit:,.2f} for {selected_month}!")

                                                  st.rerun()



                    with b_col2:

                              st.subheader("Your budgets")

                              bgt_summary = bgt_service.get_monthly_budget_summary(selected_month, currency)

                              statuses = bgt_summary.get("category_statuses", [])



                              if not statuses:

                                        st.info(f"No budgets set for **{selected_month}** yet. Choose a category and a limit to get started.")

                              else:

                                        for s in statuses:

                                                  prog_val = min(1.0, s.utilization_percentage / 100.0)

                                                  pill_class, pill_text = "bw-pill-ok", "On track"

                                                  if s.alert_level == "CRITICAL":

                                                            pill_class, pill_text = "bw-pill-bad", "Over budget"

                                                  elif s.alert_level == "WARNING":

                                                            pill_class, pill_text = "bw-pill-warn", "Nearing limit"



                                                  st.markdown(

                                                            f"""

                                                            <div class="bw-budget-card">

                                                                      <div class="bw-budget-head">

                                                                                <span class="bw-budget-title">{s.category}</span>

                                                                                <span class="bw-pill {pill_class}">{pill_text}</span>

                                                                      </div>

                                                                      <div class="bw-budget-meta">

                                                                                Limit <b>₹{s.limit_rupees:,.2f}</b> &nbsp;·&nbsp;

                                                                                Spent <b>₹{s.actual_spent_rupees:,.2f}</b> &nbsp;·&nbsp;

                                                                                Remaining <b>₹{s.remaining_rupees:,.2f}</b>

                                                                      </div>

                                                            </div>

                                                            """,

                                                            unsafe_allow_html=True,

                                                  )

                                                  st.progress(prog_val, text=f"{s.utilization_percentage:.1f}% used")

                                                  st.write("")





if __name__ == "__main__":

          main()