"""
MedAgentix AI — Notebook 01: Common Cleaning & EDA
=====================================================
Steps covered:
  Step 1 — Data Loading (all 9 datasets)
  Step 2 — Data Cleaning (duplicates, missing, labels, outliers)
  Step 3 — EDA & Visualization (distributions, heatmaps, class balance)

Run: python datasets/notebooks/01_common_cleaning_eda.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline import config
from data_pipeline.load_data import load_all
from data_pipeline.preprocess import preprocess_all
from data_pipeline.eda import run_eda


def main():
    print("=" * 60)
    print("  Notebook 01: Common Cleaning & EDA")
    print("=" * 60)

    # Ensure output directories exist
    config.ensure_dirs()

    # ─── Step 1: Load All Datasets ──────────────────────────────
    print("\n\n📂 STEP 1: Loading all 9 datasets...")
    datasets = load_all()

    # Show before-cleaning summary
    print("\n📊 Pre-Cleaning Summary:")
    print(f"{'Dataset':<30} {'Rows':>8} {'Cols':>6} {'Nulls':>8} {'Dupes':>8}")
    print("─" * 65)
    for name, df in datasets.items():
        nulls = df.isnull().sum().sum()
        dupes = df.duplicated().sum()
        print(f"{name:<30} {df.shape[0]:>8} {df.shape[1]:>6} {nulls:>8} {dupes:>8}")

    # ─── Step 2: Preprocess All ─────────────────────────────────
    print("\n\n🧹 STEP 2: Cleaning all datasets...")
    cleaned = preprocess_all(datasets)

    # Show after-cleaning summary
    print("\n📊 Post-Cleaning Summary:")
    print(f"{'Dataset':<30} {'Rows':>8} {'Cols':>6} {'Nulls':>8}")
    print("─" * 55)
    for name, df in cleaned.items():
        nulls = df.isnull().sum().sum()
        print(f"{name:<30} {df.shape[0]:>8} {df.shape[1]:>6} {nulls:>8}")

    # ─── Step 3: EDA ────────────────────────────────────────────
    print("\n\n📈 STEP 3: Running EDA & Visualization...")
    run_eda(cleaned)

    # ─── Summary ────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("  ✅ Notebook 01 Complete")
    print("=" * 60)
    print(f"\n  Outputs:")
    print(f"  ├── Cleaned CSVs: {config.CLEANED_DIR}")
    print(f"  └── EDA Plots:    {config.EDA_PLOTS_DIR}")
    print(f"\n  Next: Run 02_feature_engineering.py")

    return cleaned


if __name__ == "__main__":
    main()
