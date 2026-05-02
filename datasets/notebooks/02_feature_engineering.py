"""
MedAgentix AI -- Notebook 02: Feature Engineering
===================================================
Steps covered:
  Step 4 -- Data Encoding (label, ordinal, binary, one-hot)
  Step 5 -- Feature Engineering (symptom count, risk score, severity index, etc.)
  Step 6 -- Class Balancing (SMOTE/oversampling for classification datasets)
  Step 7 -- Save processed datasets

Prerequisite: Run 01_common_cleaning_eda.py first.
Run: python datasets/notebooks/02_feature_engineering.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from data_pipeline import config
from data_pipeline.encoding import encode_all
from data_pipeline.feature_engineering import engineer_all
from data_pipeline.balancing import balance_classes, check_imbalance


def load_cleaned_datasets() -> dict:
    """Load all cleaned datasets from Step 2 output."""
    datasets = {}
    for name in config.DATASET_REGISTRY:
        filepath = config.CLEANED_DIR / f"clean_{name}.csv"
        if filepath.exists():
            datasets[name] = pd.read_csv(filepath)
            print(f"  [OK] Loaded clean_{name}.csv ({datasets[name].shape[0]} rows)")
        else:
            print(f"  [WARN] {filepath} not found -- skipping")
    return datasets


def main():
    print("=" * 60)
    print("  Notebook 02: Feature Engineering")
    print("=" * 60)

    config.ensure_dirs()

    # Load cleaned datasets from Step 2
    print("\n Loading cleaned datasets...")
    cleaned = load_cleaned_datasets()

    if not cleaned:
        print("\n[ERROR] No cleaned datasets found! Run 01_common_cleaning_eda.py first.")
        return

    # --- Step 4: Encoding ---------------------------------------
    print("\n\n STEP 4: Encoding all datasets...")
    encoded = encode_all(cleaned)

    # Show encoding summary
    print("\n Encoding Summary:")
    print(f"{'Dataset':<30} {'Before Cols':>12} {'After Cols':>12}")
    print("-" * 55)
    for name in encoded:
        before = cleaned[name].shape[1] if name in cleaned else 0
        after = encoded[name].shape[1]
        print(f"{name:<30} {before:>12} {after:>12}")

    # --- Step 5: Feature Engineering ----------------------------
    print("\n\nSTEP 5: Feature Engineering...")
    engineered = engineer_all(encoded)

    # Show new features summary
    print("\n Feature Engineering Summary:")
    print(f"{'Dataset':<30} {'Encoded Cols':>12} {'Engineered Cols':>16} {'New Features':>13}")
    print("-" * 75)
    for name in engineered:
        enc_cols = encoded[name].shape[1] if name in encoded else 0
        eng_cols = engineered[name].shape[1]
        new = eng_cols - enc_cols
        print(f"{name:<30} {enc_cols:>12} {eng_cols:>16} {new:>13}")

    # --- Step 6: Class Balancing --------------------------------
    print("\n\nSTEP 6: Class Balancing (selective)...")
    balanced = {}

    for name, df in engineered.items():
        target = config.BALANCING_TARGETS.get(name)
        if target and target in df.columns:
            print(f"\n  --- {name} (target: {target}) ---")
            print(f"  Before balancing:")
            check_imbalance(df, target)
            balanced[name] = balance_classes(df, target, method="oversample")
        else:
            balanced[name] = df

    # --- Step 7: Save ------------------------------------------
    print("\n\n STEP 7: All processed datasets saved:")
    print(f"  |-- Encoded:    {config.ENCODED_DIR}")
    print(f"  +-- Engineered: {config.ENGINEERED_DIR}")

    # --- Summary ------------------------------------------------
    print("\n\n" + "=" * 60)
    print("  [OK] Notebook 02 Complete")
    print("=" * 60)
    print(f"\n  Part A processing complete for all 9 datasets.")
    print(f"\n  Next: Run 03_merge_and_training.py")

    return engineered


if __name__ == "__main__":
    main()
