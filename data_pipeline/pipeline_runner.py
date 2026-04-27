"""
MedAgentix AI — Pipeline Runner
==================================
Master orchestrator for the full data processing pipeline.
Runs the 13-step workflow:

PART A — Common Processing (all 9 datasets):
  1. Load all datasets
  2. Clean all (preprocess)
  3. EDA all (optional)
  4. Encode all
  5. Feature engineer all
  6. Balance (selective)
  7. Save processed datasets

PART B — Integration Pipeline:
  8.  Merge Group A → master_diagnostic.csv
  9.  Prepare Group B agent datasets (separate)
  10. Prepare Group C RAG knowledge store
  11. Train-test split (master_diagnostic only)
  12. Feature importance
  13. Save final feature store

Usage:
    python -m data_pipeline.pipeline_runner
    python -m data_pipeline.pipeline_runner --skip-eda
    python -m data_pipeline.pipeline_runner --part-a-only
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_pipeline import config
from data_pipeline.load_data import load_all
from data_pipeline.preprocess import preprocess_all
from data_pipeline.eda import run_eda
from data_pipeline.encoding import encode_all
from data_pipeline.feature_engineering import engineer_all
from data_pipeline.balancing import balance_classes, check_imbalance
from data_pipeline.merge_datasets import (
    build_master_diagnostic,
    prepare_agent_datasets,
    prepare_rag_knowledge,
)
from data_pipeline.train_split import train_test_split_pipeline
from data_pipeline.feature_importance import run_feature_importance


def run_pipeline(skip_eda: bool = False, part_a_only: bool = False):
    """
    Run the full MedAgentix AI data processing pipeline.

    Parameters
    ----------
    skip_eda : bool
        If True, skip Step 3 (EDA & Visualization).
    part_a_only : bool
        If True, only run Part A (common processing), skip Part B.
    """
    start_time = time.time()

    print("\n" + "█" * 60)
    print("  MedAgentix AI — Data Processing Pipeline")
    print("█" * 60)
    print(f"\n  Mode: {'Part A only' if part_a_only else 'Full Pipeline'}")
    print(f"  EDA:  {'Skipped' if skip_eda else 'Enabled'}")
    print(f"  Output: {config.PROCESSED_DIR}")

    # Ensure all output directories exist
    config.ensure_dirs()

    # ═══════════════════════════════════════════════════════════════
    #  PART A — Common Processing for ALL 9 Datasets
    # ═══════════════════════════════════════════════════════════════
    print("\n\n" + "▓" * 60)
    print("  PART A — Common Processing (all 9 datasets)")
    print("▓" * 60)

    # Step 1: Load all datasets
    print("\n\n" + "=" * 60)
    print("  Step 1: Data Loading")
    print("=" * 60)
    datasets = load_all()

    # Step 2: Preprocess all
    cleaned = preprocess_all(datasets)

    # Step 3: EDA (optional)
    if not skip_eda:
        run_eda(cleaned)
    else:
        print("\n  ⏭️ Skipping EDA (--skip-eda flag)")

    # Step 4: Encode all
    encoded = encode_all(cleaned)

    # Step 5: Feature engineering
    engineered = engineer_all(encoded)

    # Step 6: Balance selective datasets
    print("\n" + "=" * 60)
    print("  PART A — Step 6: Class Balancing (selective)")
    print("=" * 60)

    balanced = {}
    for name, df in engineered.items():
        target = config.BALANCING_TARGETS.get(name)
        if target and target in df.columns:
            print(f"\n  Balancing: {name} (target: {target})")
            balanced[name] = balance_classes(df, target, method="oversample")
        else:
            balanced[name] = df

    # Step 7: Save processed datasets
    print("\n" + "=" * 60)
    print("  PART A — Step 7: Save Processed Datasets")
    print("=" * 60)
    print(f"  ✅ All processed datasets saved during previous steps:")
    print(f"     Cleaned:    {config.CLEANED_DIR}")
    print(f"     Encoded:    {config.ENCODED_DIR}")
    print(f"     Engineered: {config.ENGINEERED_DIR}")

    if part_a_only:
        elapsed = time.time() - start_time
        print(f"\n\n{'█'*60}")
        print(f"  Part A Complete — {elapsed:.1f}s")
        print(f"{'█'*60}")
        return

    # ═══════════════════════════════════════════════════════════════
    #  PART B — Integration & Modeling Pipeline
    # ═══════════════════════════════════════════════════════════════
    print("\n\n" + "▓" * 60)
    print("  PART B — Integration Pipeline")
    print("▓" * 60)

    # Step 8: Merge Group A → master_diagnostic.csv
    master = build_master_diagnostic(engineered)

    # Step 9: Prepare Group B agent datasets
    agent_data = prepare_agent_datasets(engineered)

    # Step 10: Prepare Group C RAG knowledge store
    rag_chunks = prepare_rag_knowledge(cleaned)

    # Step 11: Train-test split
    splits = train_test_split_pipeline(master)

    # Step 12-13: Feature importance & feature store
    importance_results = run_feature_importance(splits)

    # ═══════════════════════════════════════════════════════════════
    #  Pipeline Complete
    # ═══════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    print(f"\n\n{'█'*60}")
    print(f"  ✅ PIPELINE COMPLETE — {elapsed:.1f}s")
    print(f"{'█'*60}")
    print(f"\n  Output Summary:")
    print(f"  ├── Cleaned datasets:     {config.CLEANED_DIR}")
    print(f"  ├── EDA plots:            {config.EDA_PLOTS_DIR}")
    print(f"  ├── Encoded datasets:     {config.ENCODED_DIR}")
    print(f"  ├── Engineered datasets:  {config.ENGINEERED_DIR}")
    print(f"  ├── Master diagnostic:    {config.MERGED_DIR / 'master_diagnostic.csv'}")
    print(f"  ├── Agent datasets:       {config.AGENT_DATASETS_DIR}")
    print(f"  ├── RAG knowledge:        {config.RAG_KNOWLEDGE_DIR}")
    print(f"  └── Feature store:        {config.FEATURE_STORE_DIR}")
    print(f"\n  No model training — deferred to next phase.")
    print(f"\n  Next step: Run training on master_diagnostic.csv")


def main():
    parser = argparse.ArgumentParser(
        description="MedAgentix AI — Data Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--skip-eda', action='store_true',
        help='Skip EDA and visualization step (faster execution)',
    )
    parser.add_argument(
        '--part-a-only', action='store_true',
        help='Run only Part A (common processing), skip integration',
    )
    args = parser.parse_args()
    run_pipeline(skip_eda=args.skip_eda, part_a_only=args.part_a_only)


if __name__ == "__main__":
    main()
