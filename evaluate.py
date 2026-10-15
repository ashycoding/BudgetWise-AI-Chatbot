"""
BudgetWise AI - Evaluation Runner Shortcut.

Run directly via:
    python evaluate.py
"""

import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from budgetwise.evaluation import ModelEvaluator

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    print(evaluator.generate_report())
