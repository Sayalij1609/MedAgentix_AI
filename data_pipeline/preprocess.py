"""
MedAgentix AI — Preprocessing Module
======================================
Common cleaning functions applied to every dataset individually.
Handles: column names, duplicates, missing values, labels, outliers, symptom names.
"""

import pandas as pd
import numpy as np
from . import config


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to snake_case.
    Strips whitespace, lowercases, replaces spaces/special chars with underscores.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r'[^a-z0-9_]', '_', regex=True)
        .str.replace(r'_+', '_', regex=True)
        .str.strip('_')
    )
    return df


def remove_duplicates(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """
    Remove duplicate rows. Logs the number of duplicates found.
    """
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    after = len(df)
    removed = before - after
    if config.VERBOSE and removed > 0:
        print(f"  [{name}] Removed {removed} duplicate rows ({before} → {after})")
    elif config.VERBOSE:
        print(f"  [{name}] No duplicates found ({before} rows)")
    return df


def handle_missing(df: pd.DataFrame, name: str = "", strategy: str = "auto") -> pd.DataFrame:
    """
    Handle missing values.

    Strategy:
    - 'auto': mode for categorical, median for numeric, drop rows if >50% null
    - 'drop': drop all rows with any null
    - 'mode': fill all with mode
    """
    df = df.copy()
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()

    if config.VERBOSE:
        print(f"  [{name}] Missing values: {total_nulls} total")
        if total_nulls > 0:
            for col in null_counts[null_counts > 0].index:
                print(f"    - {col}: {null_counts[col]} nulls ({null_counts[col]/len(df)*100:.1f}%)")

    if total_nulls == 0:
        return df

    if strategy == "drop":
        df = df.dropna().reset_index(drop=True)
    elif strategy == "mode":
        for col in df.columns[df.isnull().any()]:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
    elif strategy == "auto":
        # Drop columns with >50% missing
        high_null_cols = [col for col in df.columns if df[col].isnull().sum() / len(df) > 0.5]
        if high_null_cols and config.VERBOSE:
            print(f"  [{name}] Dropping columns with >50% nulls: {high_null_cols}")
        df = df.drop(columns=high_null_cols, errors='ignore')

        # Fill remaining nulls
        for col in df.columns[df.isnull().any()]:
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                df[col] = df[col].fillna(df[col].median())
            else:
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")

    if config.VERBOSE:
        remaining = df.isnull().sum().sum()
        print(f"  [{name}] After handling: {remaining} nulls remaining")

    return df


def normalize_labels(df: pd.DataFrame, columns: list = None, name: str = "") -> pd.DataFrame:
    """
    Normalize string labels: strip whitespace, title-case.
    Applied to specified columns or all object columns.
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include='object').columns.tolist()

    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip().str.title()

    if config.VERBOSE:
        print(f"  [{name}] Normalized labels in {len(columns)} columns")
    return df


def handle_outliers(df: pd.DataFrame, columns: list = None, name: str = "") -> pd.DataFrame:
    """
    Cap outliers using IQR method for numeric columns.
    Values beyond 1.5×IQR are capped to the whisker bounds.
    """
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    capped_count = 0
    for col in columns:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            capped_count += outliers

    if config.VERBOSE and capped_count > 0:
        print(f"  [{name}] Capped {capped_count} outlier values in {len(columns)} numeric columns")
    return df


def standardize_symptom_names(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """
    Standardize symptom names across datasets.
    Strips whitespace, underscores, and applies consistent formatting.
    """
    df = df.copy()
    symptom_cols = [col for col in df.columns if 'symptom' in col.lower()]

    for col in symptom_cols:
        if df[col].dtype == 'object':
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace('_', ' ', regex=False)
                .str.strip()
                .str.title()
            )

    # Also standardize 'symptom' if it's a join key column
    if 'symptom' in df.columns and df['symptom'].dtype == 'object':
        df['symptom'] = (
            df['symptom']
            .astype(str)
            .str.strip()
            .str.replace('_', ' ', regex=False)
            .str.strip()
            .str.title()
        )

    if config.VERBOSE and symptom_cols:
        print(f"  [{name}] Standardized symptom names in: {symptom_cols}")
    return df


def preprocess_dataset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline for a single dataset.
    Applies all cleaning steps in sequence.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    name : str
        Dataset registry name.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    print(f"\n{'─'*50}")
    print(f"  PREPROCESSING: {name}")
    print(f"{'─'*50}")

    df = clean_column_names(df)
    df = remove_duplicates(df, name)
    df = handle_missing(df, name, strategy="auto")
    df = normalize_labels(df, name=name)
    df = handle_outliers(df, name=name)
    df = standardize_symptom_names(df, name)

    print(f"  ✅ [{name}] Preprocessing complete — {df.shape[0]} rows × {df.shape[1]} cols")

    # Save cleaned output
    config.ensure_dirs()
    output_path = config.CLEANED_DIR / f"clean_{name}.csv"
    df.to_csv(output_path, index=False)
    print(f"  💾 Saved: {output_path}")

    return df


def preprocess_all(datasets: dict) -> dict:
    """
    Preprocess all datasets.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Dictionary from load_all().

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary of cleaned DataFrames.
    """
    print("\n" + "=" * 60)
    print("  PART A — Step 2: Data Cleaning (all 9 datasets)")
    print("=" * 60)

    cleaned = {}
    for name, df in datasets.items():
        cleaned[name] = preprocess_dataset(df, name)

    print(f"\n✅ All {len(cleaned)} datasets preprocessed and saved.")
    return cleaned
