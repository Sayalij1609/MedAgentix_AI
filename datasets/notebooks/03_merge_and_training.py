"""
MedAgentix AI -- Notebook 03: Merge & Dataset Integration
===========================================================
Steps covered:
  Step 8  -- Merge Group A datasets -> master_diagnostic.csv
  Step 9  -- Prepare Group B agent datasets (separate)
  Step 10 -- Prepare Group C RAG knowledge text chunks
  Step 11 -- Train-test split

Model training is deferred to the next phase.

Prerequisite: Run 02_feature_engineering.py first.
Run: python datasets/notebooks/03_merge_and_training.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from data_pipeline import config
from data_pipeline.merge_datasets import (
    build_master_diagnostic,
    prepare_agent_datasets,
    prepare_rag_knowledge,
)
from data_pipeline.train_split import train_test_split_pipeline


def load_engineered_datasets() -> dict:
    """Load all engineered datasets from Step 5 output."""
    datasets = {}
    for name in config.DATASET_REGISTRY:
        filepath = config.ENGINEERED_DIR / f"engineered_{name}.csv"
        if filepath.exists():
            datasets[name] = pd.read_csv(filepath)
            print(f"  [OK] Loaded engineered_{name}.csv ({datasets[name].shape[0]} rows)")
        else:
            # Fallback to encoded
            fallback = config.ENCODED_DIR / f"encoded_{name}.csv"
            if fallback.exists():
                datasets[name] = pd.read_csv(fallback)
                print(f"  [WARN] Using encoded_{name}.csv (no engineered version)")
            else:
                print(f"  [WARN] {name} not found -- skipping")
    return datasets


def load_cleaned_datasets() -> dict:
    """Load cleaned datasets (needed for RAG text chunks)."""
    datasets = {}
    for name in config.GROUP_C_RAG:
        filepath = config.CLEANED_DIR / f"clean_{name}.csv"
        if filepath.exists():
            datasets[name] = pd.read_csv(filepath)
    return datasets


def main():
    print("=" * 60)
    print("  Notebook 03: Merge & Dataset Integration")
    print("=" * 60)

    config.ensure_dirs()

    # Load engineered datasets
    print("\n Loading engineered datasets...")
    engineered = load_engineered_datasets()

    if not engineered:
        print("\n[ERROR] No engineered datasets found! Run 02_feature_engineering.py first.")
        return

    # --- Step 8: Merge Group A -> Master Diagnostic -------------
    print("\n\n STEP 8: Building Master Diagnostic Dataset...")
    print("  Merging: Core Clinical + Risk Factor + Temporal + (opt) Differential")
    master = build_master_diagnostic(engineered)

    print(f"\n  Master Diagnostic Dataset:")
    print(f"    Rows:    {master.shape[0]}")
    print(f"    Columns: {master.shape[1]}")
    print(f"    Columns: {list(master.columns)}")

    # --- Step 9: Prepare Agent Datasets -------------------------
    print("\n\n STEP 9: Preparing Agent Datasets (separate)...")
    agent_data = prepare_agent_datasets(engineered)

    print(f"\n  Agent Datasets:")
    for agent, df in agent_data.items():
        print(f"    {agent}: {df.shape[0]} rows x {df.shape[1]} cols")

    # --- Step 10: RAG Knowledge Store ---------------------------
    print("\n\n STEP 10: Preparing RAG Knowledge Store...")
    cleaned = load_cleaned_datasets()
    # Merge cleaned data for RAG (needs text columns, not encoded)
    rag_data = {**engineered}
    rag_data.update(cleaned)
    rag_chunks = prepare_rag_knowledge(rag_data)

    # --- Step 11: Train-Test Split ------------------------------
    print("\n\nSTEP 11: Train-Test Split...")
    splits = train_test_split_pipeline(master)

    print(f"\n  Split Summary:")
    for key, data in splits.items():
        shape = data.shape if hasattr(data, 'shape') else (len(data),)
        print(f"    {key}: {shape}")

    # --- Summary ------------------------------------------------
    print("\n\n" + "=" * 60)
    print("  [OK] Notebook 03 Complete")
    print("=" * 60)
    print(f"\n  Outputs:")
    print(f"  |-- Master Diagnostic: {config.MERGED_DIR / 'master_diagnostic.csv'}")
    print(f"  |-- Agent Datasets:    {config.AGENT_DATASETS_DIR}")
    print(f"  |-- RAG Knowledge:     {config.RAG_KNOWLEDGE_DIR}")
    print(f"  +-- Train/Test Splits: {config.FEATURE_STORE_DIR}")
    print(f"\n  Model training deferred to next phase.")
    print(f"\n  Next: Run 04_feature_importance.py")

    return master, splits


if __name__ == "__main__":
    main()
