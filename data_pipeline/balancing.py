"""
MedAgentix AI -- Class Balancing Module
========================================
Handles class imbalance using SMOTE, oversampling, or undersampling.
Applied selectively to datasets with classification targets.
"""

import pandas as pd
import numpy as np
from collections import Counter
from . import config


def check_imbalance(df: pd.DataFrame, target_col: str) -> dict:
    """
    Check class distribution and report imbalance.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str

    Returns
    -------
    dict with class counts and imbalance ratio
    """
    if target_col not in df.columns:
        print(f"  [WARN] Target column '{target_col}' not found")
        return {}

    counts = df[target_col].value_counts().to_dict()
    total = sum(counts.values())
    min_class = min(counts.values())
    max_class = max(counts.values())
    ratio = max_class / min_class if min_class > 0 else float('inf')

    result = {
        "class_counts": counts,
        "total": total,
        "min_class": min_class,
        "max_class": max_class,
        "imbalance_ratio": round(ratio, 2),
        "is_imbalanced": ratio > 1.5,
    }

    if config.VERBOSE:
        print(f"  Class distribution for '{target_col}':")
        for cls, count in counts.items():
            pct = count / total * 100
            print(f"    {cls}: {count} ({pct:.1f}%)")
        print(f"  Imbalance ratio: {ratio:.2f}x", end="")
        print(" [WARN] IMBALANCED" if result["is_imbalanced"] else " [OK] Balanced")

    return result


def balance_classes(
    df: pd.DataFrame,
    target_col: str,
    method: str = "smote",
) -> pd.DataFrame:
    """
    Balance classes in a dataset.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    method : str
        'smote' -- SMOTE oversampling (requires numeric features)
        'oversample' -- Random oversampling of minority class
        'undersample' -- Random undersampling of majority class

    Returns
    -------
    pd.DataFrame -- Balanced dataset
    """
    if target_col not in df.columns:
        print(f"  [WARN] Target '{target_col}' not found -- skipping balancing")
        return df

    print(f"\n  Balancing '{target_col}' using method: {method}")
    imbalance = check_imbalance(df, target_col)

    if not imbalance.get("is_imbalanced", False):
        print(f"  [OK] Already balanced -- no action needed")
        return df

    if method == "smote":
        return _smote_balance(df, target_col)
    elif method == "oversample":
        return _random_oversample(df, target_col)
    elif method == "undersample":
        return _random_undersample(df, target_col)
    else:
        print(f"  [WARN] Unknown method '{method}' -- defaulting to oversample")
        return _random_oversample(df, target_col)


def _smote_balance(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """SMOTE oversampling -- only works with numeric features."""
    try:
        from imblearn.over_sampling import SMOTE

        # Separate features and target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target_col]

        if len(feature_cols) < 2:
            print(f"  [WARN] Not enough numeric features for SMOTE -- falling back to oversample")
            return _random_oversample(df, target_col)

        X = df[feature_cols].fillna(0)
        y = df[target_col]

        # Determine minimum samples needed for SMOTE
        min_samples = min(Counter(y).values())
        k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1

        if k_neighbors < 1:
            print(f"  [WARN] Too few samples for SMOTE -- falling back to oversample")
            return _random_oversample(df, target_col)

        smote = SMOTE(random_state=config.RANDOM_STATE, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Reconstruct DataFrame
        result = pd.DataFrame(X_resampled, columns=feature_cols)
        result[target_col] = y_resampled

        # Add back non-numeric columns from original (will have NaN for synthetic rows)
        non_numeric = [c for c in df.columns if c not in feature_cols and c != target_col]
        for col in non_numeric:
            result[col] = np.nan
            result.loc[:len(df) - 1, col] = df[col].values

        if config.VERBOSE:
            print(f"  [OK] SMOTE applied: {len(df)} -> {len(result)} rows")
            check_imbalance(result, target_col)

        return result

    except ImportError:
        print("  [WARN] imblearn not installed -- falling back to random oversample")
        print("     Install with: pip install imbalanced-learn")
        return _random_oversample(df, target_col)


def _random_oversample(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Random oversampling of minority classes to match majority class."""
    max_count = df[target_col].value_counts().max()
    dfs = []
    for cls in df[target_col].unique():
        cls_df = df[df[target_col] == cls]
        if len(cls_df) < max_count:
            oversampled = cls_df.sample(max_count, replace=True, random_state=config.RANDOM_STATE)
            dfs.append(oversampled)
        else:
            dfs.append(cls_df)

    result = pd.concat(dfs, ignore_index=True)
    result = result.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)

    if config.VERBOSE:
        print(f"  [OK] Oversampled: {len(df)} -> {len(result)} rows")
        check_imbalance(result, target_col)

    return result


def _random_undersample(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Random undersampling of majority classes to match minority class."""
    min_count = df[target_col].value_counts().min()
    dfs = []
    for cls in df[target_col].unique():
        cls_df = df[df[target_col] == cls]
        if len(cls_df) > min_count:
            undersampled = cls_df.sample(min_count, random_state=config.RANDOM_STATE)
            dfs.append(undersampled)
        else:
            dfs.append(cls_df)

    result = pd.concat(dfs, ignore_index=True)
    result = result.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)

    if config.VERBOSE:
        print(f"  [OK] Undersampled: {len(df)} -> {len(result)} rows")
        check_imbalance(result, target_col)

    return result
