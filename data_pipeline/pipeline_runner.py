"""
MedAgentix AI -- Pipeline Runner
==================================
Master orchestrator for the full data processing pipeline.
Runs the 13-step workflow:

PART A -- Common Processing (all 9 datasets):
  1. Load all datasets
  2. Clean all (preprocess)
  3. EDA all (optional)
  4. Encode all
  5. Feature engineer all
  6. Save processed datasets

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
from data_pipeline.integration import prepare_agent_datasets, prepare_rag_knowledge


def run_pipeline(skip_eda: bool = False, part_a_only: bool = False):
    """
    Run the full MedAgentix AI data processing pipeline.

    Parameters
    ----------
    skip_eda : bool
        If True, skip Step 3 (EDA & Visualization).
    """
    start_time = time.time()

    print("\n" + "=" * 60)
    print("  MedAgentix AI -- Data Processing Pipeline")
    print("=" * 60)
    print(f"\n  EDA:  {'Skipped' if skip_eda else 'Enabled'}")
    print(f"  Output: {config.PROCESSED_DIR}")

    # Ensure all output directories exist
    config.ensure_dirs()

    # ===============================================================
    #  Data Cleaning (all 9 datasets)
    # ===============================================================

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
        print("\n  Skipping EDA (--skip-eda flag)")

    # Step 4: Encode all
    encoded = encode_all(cleaned)

    # Step 5: Feature engineering
    engineered = engineer_all(encoded)

    # Step 6: Save processed datasets
    print("\n" + "=" * 60)
    print("  Step 6: Save Processed Datasets")
    print("=" * 60)
    print(f"  [OK] All processed datasets saved during previous steps:")
    print(f"     Cleaned:    {config.CLEANED_DIR}")
    print(f"     Encoded:    {config.ENCODED_DIR}")
    print(f"     Engineered: {config.ENGINEERED_DIR}")

    # Step 7: Prepare Group B agent datasets
    agent_data = prepare_agent_datasets(engineered)

    # Step 8: Prepare Group C RAG knowledge store
    rag_chunks = prepare_rag_knowledge(cleaned)

    elapsed = time.time() - start_time
    print(f"\n\n{'='*60}")
    print(f"  [OK] PIPELINE COMPLETE -- {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"\n  Output Summary:")
    print(f"  |-- Cleaned datasets:     {config.CLEANED_DIR}")
    print(f"  |-- EDA plots:            {config.EDA_PLOTS_DIR}")
    print(f"  |-- Encoded datasets:     {config.ENCODED_DIR}")
    print(f"  |-- Engineered datasets:  {config.ENGINEERED_DIR}")
    print(f"  |-- Agent datasets:       {config.AGENT_DATASETS_DIR}")
    print(f"  +-- RAG knowledge:        {config.RAG_KNOWLEDGE_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="MedAgentix AI -- Data Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--skip-eda', action='store_true',
        help='Skip EDA and visualization step (faster execution)',
    )
    args = parser.parse_args()
    run_pipeline(skip_eda=args.skip_eda)


if __name__ == "__main__":
    main()
