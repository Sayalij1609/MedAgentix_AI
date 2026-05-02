"""
MedAgentix AI -- Data Loading Module
=====================================
Functions to load all 9 raw CSV datasets.
Each loader prints inspection info: shape, dtypes, head().
"""

import pandas as pd
from . import config


def _inspect(df: pd.DataFrame, name: str) -> None:
    """Print basic inspection info for a loaded dataset."""
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print(f"  LOADED: {name}")
        print(f"{'='*60}")
        print(f"  Shape : {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Dtypes:\n{df.dtypes.to_string()}")
        print(f"  Nulls :\n{df.isnull().sum().to_string()}")
        print(f"{'='*60}\n")


def load_dataset(name: str) -> pd.DataFrame:
    """
    Load a single dataset by its registry name.

    Parameters
    ----------
    name : str
        Short name from DATASET_REGISTRY (e.g. 'core', 'drug', 'risk').

    Returns
    -------
    pd.DataFrame
    """
    if name not in config.DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {list(config.DATASET_REGISTRY.keys())}"
        )
    filepath = config.RAW_DIR / config.DATASET_REGISTRY[name]
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    df = pd.read_csv(filepath)
    _inspect(df, name)
    return df


def load_core() -> pd.DataFrame:
    """Load Core Clinical Dataset."""
    return load_dataset("core")


def load_drug() -> pd.DataFrame:
    """Load Drug/Medication Dataset."""
    return load_dataset("drug")


def load_emergency() -> pd.DataFrame:
    """Load Emergency Condition Dataset."""
    return load_dataset("emergency")


def load_medical_knowledge() -> pd.DataFrame:
    """Load Medical Knowledge Dataset."""
    return load_dataset("medical_knowledge")


def load_risk() -> pd.DataFrame:
    """Load Risk Factor Dataset."""
    return load_dataset("risk")


def load_symptom_intelligence() -> pd.DataFrame:
    """Load Symptom Intelligence Dataset."""
    return load_dataset("symptom_intelligence")


def load_temporal() -> pd.DataFrame:
    """Load Temporal Dataset."""
    return load_dataset("temporal")


def load_differential() -> pd.DataFrame:
    """Load Differential Diagnosis Dataset."""
    return load_dataset("differential")


def load_diagnostic() -> pd.DataFrame:
    """Load Test Diagnostic Recommendation Dataset."""
    return load_dataset("diagnostic")


def load_all() -> dict:
    """
    Load all 9 datasets and return as a dictionary.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are registry names, values are DataFrames.
    """
    datasets = {}
    for name in config.DATASET_REGISTRY:
        datasets[name] = load_dataset(name)

    print(f"\n[OK] All {len(datasets)} datasets loaded successfully.")
    print(f"   Total rows across all datasets: {sum(df.shape[0] for df in datasets.values()):,}")
    return datasets
