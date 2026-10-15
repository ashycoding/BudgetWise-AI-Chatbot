"""Pytest configuration and environment bootstrap for BudgetWise AI."""

import sys
from pathlib import Path

# Add src to sys.path so test discovery always finds budgetwise
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
