"""
MedAgentix AI — Encoding Module
=================================
All data encoding functions: label, ordinal, one-hot, binary symptom encoding.
Strategies are driven by config.COLUMN_CONFIG per dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from . import config


def label_encode(df: pd.DataFrame, columns: list) -> tuple:
    """
    Apply label encoding to specified columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str

    Returns
    -------
    tuple(pd.DataFrame, dict)
        Encoded DataFrame and mapping dict {column: {original: encoded}}.
    """
    df = df.copy()
    mappings = {}

    for col in columns:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        # Handle NaN by converting to string first
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    return df, mappings


def one_hot_encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Apply one-hot encoding to specified columns.
    Uses pd.get_dummies with prefix matching column name.
    """
    df = df.copy()
    existing_cols = [c for c in columns if c in df.columns]
    if existing_cols:
        df = pd.get_dummies(df, columns=existing_cols, prefix=existing_cols, drop_first=True)
    return df


def ordinal_encode(df: pd.DataFrame, column: str, order_map: dict) -> pd.DataFrame:
    """
    Apply ordinal encoding using a predefined order mapping.

    Example: {"Low": 1, "Moderate": 2, "High": 3}
    """
    df = df.copy()
    if column in df.columns:
        # Case-insensitive mapping
        case_map = {str(k).strip().title(): v for k, v in order_map.items()}
        df[column] = df[column].astype(str).str.strip().str.title().map(case_map)
        # Fill unmapped values with 0
        df[column] = df[column].fillna(0).astype(int)
    return df


def binary_encode_symptoms(df: pd.DataFrame, binary_map: dict) -> pd.DataFrame:
    """
    Convert binary symptom columns (Yes/No) to 0/1.

    Parameters
    ----------
    binary_map : dict
        {column_name: {"Yes": 1, "No": 0}} style mapping.
    """
    df = df.copy()
    for col, mapping in binary_map.items():
        if col in df.columns:
            case_map = {str(k).strip().title(): v for k, v in mapping.items()}
            df[col] = df[col].astype(str).str.strip().str.title().map(case_map)
            df[col] = df[col].fillna(0).astype(int)
    return df


def encode_dataset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Encode a single dataset using its config-driven strategy.

    Applies in order:
    1. Binary encoding (Yes/No → 0/1)
    2. Ordinal encoding (Low/Medium/High → 1/2/3)
    3. Label encoding for remaining categorical columns
       (excluding text/join key columns)
    """
    print(f"\n{'-'*50}")
    print(f"  ENCODING: {name}")
    print(f"{'-'*50}")

    df = df.copy()
    col_config = config.COLUMN_CONFIG.get(name, {})

    # Step 1: Binary encoding
    binary_map = col_config.get("binary", {})
    if binary_map:
        df = binary_encode_symptoms(df, binary_map)
        if config.VERBOSE:
            print(f"  [{name}] Binary encoded: {list(binary_map.keys())}")

    # Step 2: Ordinal encoding
    ordinal_map = col_config.get("ordinal", {})
    if ordinal_map:
        for col, order in ordinal_map.items():
            df = ordinal_encode(df, col, order)
        if config.VERBOSE:
            print(f"  [{name}] Ordinal encoded: {list(ordinal_map.keys())}")

    # Step 3: Label encode remaining categorical columns
    # Exclude: text columns, join keys, already encoded columns
    text_cols = col_config.get("text", [])
    join_key = col_config.get("join_key", "")
    already_encoded = list(binary_map.keys()) + list(ordinal_map.keys())
    exclude = set(text_cols + [join_key] + already_encoded)

    remaining_cat = [
        col for col in df.select_dtypes(include='object').columns
        if col not in exclude
    ]

    if remaining_cat:
        df, mappings = label_encode(df, remaining_cat)
        if config.VERBOSE:
            print(f"  [{name}] Label encoded: {remaining_cat}")

    print(f"  [OK] [{name}] Encoding complete — {df.shape[1]} columns")

    # Save encoded output
    config.ensure_dirs()
    output_path = config.ENCODED_DIR / f"encoded_{name}.csv"
    df.to_csv(output_path, index=False)
    print(f"   Saved: {output_path}")

    return df


def encode_all(datasets: dict) -> dict:
    """
    Encode all datasets.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    print("\n" + "=" * 60)
    print("  PART A — Step 4: Data Encoding (all 9 datasets)")
    print("=" * 60)

    encoded = {}
    for name, df in datasets.items():
        encoded[name] = encode_dataset(df, name)

    print(f"\n[OK] All {len(encoded)} datasets encoded and saved.")
    return encoded
