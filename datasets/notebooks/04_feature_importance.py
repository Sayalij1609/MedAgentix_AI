"""
MedAgentix AI — Notebook 04: Feature Importance
==================================================
Steps covered:
  Step 12 — Feature Importance (XGBoost, SHAP, correlation, mutual info)
  Step 13 — Create Final Feature Store (ranked features, top selections)

Prerequisite: Run 03_merge_and_training.py first.
Run: python datasets/notebooks/04_feature_importance.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from data_pipeline import config
from data_pipeline.feature_importance import (
    xgb_feature_importance,
    shap_analysis,
    correlation_ranking,
    feature_scoring,
    run_feature_importance,
)


def load_splits() -> dict:
    """Load train/test splits from the feature store."""
    splits = {}

    for key in ["X_train", "X_test", "y_train", "y_test", "X_val", "y_val"]:
        filepath = config.FEATURE_STORE_DIR / f"{key}.csv"
        if filepath.exists():
            df = pd.read_csv(filepath)
            if key.startswith("y_"):
                splits[key] = df.iloc[:, 0]  # Series for target
            else:
                splits[key] = df
            print(f"  ✅ Loaded {key}: {df.shape}")

    return splits


def main():
    print("=" * 60)
    print("  Notebook 04: Feature Importance")
    print("=" * 60)

    config.ensure_dirs()

    # Load splits
    print("\n📂 Loading train/test splits...")
    splits = load_splits()

    if "X_train" not in splits or "y_train" not in splits:
        print("\n❌ Train splits not found! Run 03_merge_and_training.py first.")
        return

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_test = splits.get("X_test")

    print(f"\n  Training data: {X_train.shape[0]} samples × {X_train.shape[1]} features")
    print(f"  Features: {list(X_train.columns)}")

    # ─── Step 12: Feature Importance ────────────────────────────
    print("\n\n📊 STEP 12: Computing Feature Importance...")

    # Run all importance methods
    results = run_feature_importance(splits)

    # ─── Step 13: Feature Store Summary ─────────────────────────
    print("\n\n🏪 STEP 13: Feature Store Summary")

    if "combined" in results:
        combined = results["combined"]
        print(f"\n  All features ranked ({len(combined)} total):")
        print(f"\n  {'Rank':<6} {'Feature':<35} {'Unified Score':<15}")
        print("  " + "─" * 56)
        for _, row in combined.iterrows():
            print(f"  {int(row['final_rank']):<6} {row['feature']:<35} {row['unified_score']:.4f}")

    if "top_features" in results:
        top = results["top_features"]
        print(f"\n  ⭐ Top {len(top)} selected features:")
        for i, feat in enumerate(top, 1):
            print(f"    {i:2d}. {feat}")

    # ─── Summary ────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("  ✅ Notebook 04 Complete — Feature Store Ready")
    print("=" * 60)
    print(f"\n  Final Outputs in {config.FEATURE_STORE_DIR}:")
    print(f"  ├── X_train.csv / X_test.csv")
    print(f"  ├── y_train.csv / y_test.csv")
    print(f"  ├── selected_features.csv (all features ranked)")
    print(f"  ├── engineered_features.csv (top features only)")
    print(f"  ├── xgb_feature_importance.png")
    print(f"  ├── shap_feature_importance.png")
    print(f"  └── mutual_info_scores.png")
    print(f"\n  🎯 Data pipeline complete! Ready for model training (next phase).")

    return results


if __name__ == "__main__":
    main()
