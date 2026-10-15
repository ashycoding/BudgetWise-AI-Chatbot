"""
BudgetWise AI - Expense Categorization Module.

Classifies transaction descriptions into standard financial categories using:
1. User-persisted category overrides (checked first).
2. Deterministic exact and keyword rules with explicit confidence bands.
3. Machine-learning baseline (TF-IDF + Naive Bayes) for fuzzy matching.
4. Transparent confidence scoring and explicit confirmation flagging.
"""

import re
from typing import Dict, List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from budgetwise.config import get_settings
from budgetwise.database import DatabaseManager
from budgetwise.models import (
    CategoryPrediction,
    ClassificationMethod,
    ConfidenceBand,
    StandardCategory,
)

# Deterministic domain keyword mappings with common Indian abbreviations and variants
DOMAIN_KEYWORDS: Dict[StandardCategory, List[str]] = {
    StandardCategory.GROCERIES: [
        "blinkit", "blinkt", "zepto", "zpto", "instamart", "bigbasket", "bigbaskt",
        "dmart", "dmrt", "spencer", "supermarket", "grocery", "groceries",
        "vegetable", "veggies", "fruits", "milk", "dairy", "kirana", "ration",
        "nature basket", "hypermarket", "provisions",
    ],
    StandardCategory.FOOD_AND_DINING: [
        "swiggy", "swgy", "zomato", "zomto", "restaurant", "cafe", "mcdonald", "mcd",
        "starbucks", "starbux", "domino", "dominos", "pizza", "burger", "dining",
        "bar", "pub", "kfc", "subway", "biryani", "haldiram", "chai", "coffee",
        "baker", "sweet", "eatery", "dosa", "darshini", "food court", "lunch",
        "dinner", "breakfast",
    ],
    StandardCategory.TRANSPORTATION: [
        "uber", "ubr", "ola", "rapido", "metro", "namma metro", "smart card", "smart card recharge", "irctc", "railway",
        "petrol", "diesel", "fuel", "shell", "hpcl", "bpcl", "indian oil", "toll",
        "fastag", "bus", "auto rickshaw", "auto ride", "autorickshaw", "cab",
        "parking", "train", "flight ticket",
    ],
    StandardCategory.HOUSING_AND_RENT: [
        "rent", "landlord", "flat rent", "house rent", "maintenance",
        "society maintenance", "apartment", "nobroker", "nestaway", "lease",
        "pg accommodation", "security maintenance",
    ],
    StandardCategory.UTILITIES: [
        "bescom", "electricity", "water bill", "water supply", "bwssb", "piped gas",
        "cylinder", "indane", "bharat gas", "broadband", "wifi", "airtel broadband",
        "jio fiber", "act fibernet", "tata play", "dth", "recharge", "postpaid", "utility",
    ],
    StandardCategory.SHOPPING: [
        "amazon", "amzn", "flipkart", "flipkrt", "myntra", "ajio", "zara", "h&m",
        "meesho", "decathlon", "croma", "reliance digital", "nykaa", "apparel",
        "footwear", "electronics", "mall", "clothing", "marketplace",
    ],
    StandardCategory.HEALTHCARE: [
        "apollo", "pharmacy", "medplus", "pharmeasy", "1mg", "hospital",
        "clinic", "doctor", "diagnostic", "medicine", "tablets", "dental",
        "pathology", "medical", "health insurance",
    ],
    StandardCategory.EDUCATION: [
        "school", "college", "tuition", "university", "fees", "unacademy",
        "coursera", "udemy", "byju", "books", "stationery", "exam fee",
        "semester fee",
    ],
    StandardCategory.ENTERTAINMENT: [
        "pvr", "inox", "cinema", "movie", "bookmyshow", "concert", "theatre",
        "gaming", "playstation", "steam", "amusement", "entertainment", "bowling",
    ],
    StandardCategory.SUBSCRIPTIONS: [
        "netflix", "spotify", "prime video", "youtube premium", "apple music",
        "hotstar", "disney", "icloud", "google one", "chatgpt", "github copilot",
    ],
    StandardCategory.TRAVEL: [
        "makemytrip", "goibibo", "cleartrip", "air india", "indigo", "vistara",
        "flight", "hotel", "resort", "airbnb", "hostel", "zostel", "tour", "travels",
    ],
    StandardCategory.PERSONAL_CARE: [
        "salon", "spa", "haircut", "urban company", "parlor", "beauty", "cosmetics",
        "naturals", "enrich", "massage", "grooming", "facial",
    ],
    StandardCategory.EMI_AND_DEBT: [
        "emi", "loan payment", "credit card payment", "home loan", "car loan",
        "hdfc loan", "sbi loan", "bajaj finance", "cred", "personal loan",
        "auto loan", "loan debit", "installment",
    ],
    StandardCategory.INVESTMENTS_AND_SAVINGS: [
        "zerodha", "groww", "mutual fund", "sip", "stocks", "fixed deposit",
        "recurring deposit", "ppf", "nps", "coin", "kite", "upstox", "shares",
        "equity",
    ],
    StandardCategory.INCOME: [
        "salary", "payroll", "dividend", "interest credit", "stipend",
        "freelance", "bonus", "consulting income",
    ],
}


class ExpenseCategorizer:
    """
    Categorizes transaction descriptions using deterministic rules,
    user corrections, and a scikit-learn baseline model.
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        confidence_threshold: Optional[float] = None,
    ) -> None:
        self.settings = get_settings()
        self.db = db_manager or DatabaseManager()
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.settings.categorizer_confidence_threshold
        )
        self._ml_pipeline: Optional[Pipeline] = None
        self._train_baseline_model()

    def _normalize(self, text: str) -> str:
        """Clean and normalize description for matching."""
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        return " ".join(cleaned.split())

    def _train_baseline_model(self) -> None:
        """
        Train an interpretable baseline Naive Bayes model on domain keywords
        using char-wb n-grams to handle abbreviations, partial tokens, and misspellings.
        """
        training_corpus: List[Tuple[str, str]] = []
        for cat, keywords in DOMAIN_KEYWORDS.items():
            for kw in keywords:
                training_corpus.append((kw, cat.value))
                training_corpus.append((f"payment to {kw}", cat.value))
                training_corpus.append((f"{kw} order", cat.value))
                training_corpus.append((f"{kw} purchase", cat.value))

        # Explicit training samples for Other / General Ambiguous
        ambiguous_samples = [
            "payment", "debit", "credit", "transfer", "counter", "atm", "pos",
            "txn", "ref", "adjustment", "settlement", "charges", "fee", "unknown",
            "general", "miscellaneous", "cash", "withdrawal",
        ]
        for amb in ambiguous_samples:
            training_corpus.append((amb, StandardCategory.OTHER.value))

        texts = [item[0] for item in training_corpus]
        labels = [item[1] for item in training_corpus]

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)),
            ("clf", MultinomialNB(alpha=0.1)),
        ])
        pipeline.fit(texts, labels)
        self._ml_pipeline = pipeline

    def get_user_override(self, normalized_desc: str) -> Optional[str]:
        """Check if user has previously registered a custom category correction."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT pattern, category FROM category_overrides;")
                rows = cursor.fetchall()
                for row in rows:
                    pattern = row["pattern"].lower()
                    if pattern in normalized_desc:
                        return row["category"]
        except Exception:
            return None
        return None

    def save_user_override(self, pattern: str, category: str) -> None:
        """Persist a user correction so future matching prioritizes this mapping."""
        normalized_pattern = self._normalize(pattern)
        if not normalized_pattern:
            return
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO category_overrides (pattern, category)
                VALUES (?, ?)
                ON CONFLICT(pattern) DO UPDATE SET category = excluded.category;
                """,
                (normalized_pattern, category),
            )

    def predict(self, description: str, transaction_type: str = "expense") -> CategoryPrediction:
        """
        Predict financial category with explicit reasoning and confidence band.
        Never invents calibrated probabilities for rule-based heuristics.
        """
        if not description or not description.strip():
            return CategoryPrediction(
                predicted_category=StandardCategory.OTHER.value,
                confidence_score=0.0,
                confidence_band=ConfidenceBand.UNKNOWN,
                classification_method=ClassificationMethod.FALLBACK,
                explanation="Empty transaction description provided.",
                needs_confirmation=True,
            )

        norm_desc = self._normalize(description)

        # 1. Check User Overrides
        override_cat = self.get_user_override(norm_desc)
        if override_cat:
            return CategoryPrediction(
                predicted_category=override_cat,
                confidence_score=1.0,
                confidence_band=ConfidenceBand.HIGH,
                classification_method=ClassificationMethod.USER_OVERRIDE,
                explanation=f"Matched persistent user override for '{norm_desc}'.",
                needs_confirmation=False,
            )

        # 2. Check Explicit Transaction Types (e.g. Income)
        if transaction_type.lower() == "income":
            for kw in DOMAIN_KEYWORDS[StandardCategory.INCOME]:
                if kw in norm_desc:
                    return CategoryPrediction(
                        predicted_category=StandardCategory.INCOME.value,
                        confidence_score=0.95,
                        confidence_band=ConfidenceBand.HIGH,
                        classification_method=ClassificationMethod.EXACT_RULE,
                        explanation=f"Explicit income keyword '{kw}' matched.",
                        needs_confirmation=False,
                    )
            return CategoryPrediction(
                predicted_category=StandardCategory.INCOME.value,
                confidence_score=0.85,
                confidence_band=ConfidenceBand.HIGH,
                classification_method=ClassificationMethod.EXACT_RULE,
                explanation="Transaction marked as income cash-flow.",
                needs_confirmation=False,
            )

        ordered_categories = [
            StandardCategory.EMI_AND_DEBT,
            StandardCategory.INVESTMENTS_AND_SAVINGS,
            StandardCategory.HOUSING_AND_RENT,
            StandardCategory.TRANSPORTATION,
            StandardCategory.UTILITIES,
            StandardCategory.SUBSCRIPTIONS,
            StandardCategory.GROCERIES,
            StandardCategory.FOOD_AND_DINING,
            StandardCategory.HEALTHCARE,
            StandardCategory.EDUCATION,
            StandardCategory.ENTERTAINMENT,
            StandardCategory.TRAVEL,
            StandardCategory.PERSONAL_CARE,
            StandardCategory.SHOPPING,
        ]

        for cat in ordered_categories:
            keywords = DOMAIN_KEYWORDS[cat]
            for kw in keywords:
                # Word boundary check
                if re.search(rf"\b{re.escape(kw)}\b", norm_desc):
                    return CategoryPrediction(
                        predicted_category=cat.value,
                        confidence_score=0.90,
                        confidence_band=ConfidenceBand.HIGH,
                        classification_method=ClassificationMethod.KEYWORD_RULE,
                        explanation=f"Keyword '{kw}' mapped directly to category '{cat.value}'.",
                        needs_confirmation=False,
                    )

        # 4. ML Baseline Prediction for ambiguous/fuzzy descriptions
        if self._ml_pipeline is not None:
            try:
                tfidf_vec = self._ml_pipeline.named_steps["tfidf"]
                transformed = tfidf_vec.transform([norm_desc])

                # Check if ANY character n-grams matched the training vocabulary
                if transformed.nnz == 0:
                    return CategoryPrediction(
                        predicted_category=StandardCategory.OTHER.value,
                        confidence_score=0.10,
                        confidence_band=ConfidenceBand.UNKNOWN,
                        classification_method=ClassificationMethod.FALLBACK,
                        explanation="Out-of-vocabulary description. No recognizable merchant terms found.",
                        needs_confirmation=True,
                    )

                probs = self._ml_pipeline.predict_proba([norm_desc])[0]
                classes = self._ml_pipeline.classes_
                max_idx = probs.argmax()
                best_prob = float(probs[max_idx])
                best_cat = str(classes[max_idx])

                # Check against confidence threshold
                if best_prob >= self.confidence_threshold and best_cat != StandardCategory.OTHER.value:
                    band = ConfidenceBand.MEDIUM
                    needs_conf = best_prob < 0.85
                    return CategoryPrediction(
                        predicted_category=best_cat,
                        confidence_score=round(best_prob, 2),
                        confidence_band=band,
                        classification_method=ClassificationMethod.ML_MODEL,
                        explanation=f"Statistical text classifier matched category '{best_cat}' with score {best_prob:.2f}.",
                        needs_confirmation=needs_conf,
                    )
                else:
                    # Below confidence threshold or matched generic Other
                    return CategoryPrediction(
                        predicted_category=StandardCategory.OTHER.value,
                        confidence_score=round(best_prob, 2),
                        confidence_band=ConfidenceBand.LOW,
                        classification_method=ClassificationMethod.FALLBACK,
                        explanation="Ambiguous description below threshold. Mapped to Other for user review.",
                        needs_confirmation=True,
                    )
            except Exception:
                pass

        # 5. Fallback for completely unknown descriptions
        return CategoryPrediction(
            predicted_category=StandardCategory.OTHER.value,
            confidence_score=0.15,
            confidence_band=ConfidenceBand.UNKNOWN,
            classification_method=ClassificationMethod.FALLBACK,
            explanation="No deterministic rule or reliable keyword matched this description.",
            needs_confirmation=True,
        )
