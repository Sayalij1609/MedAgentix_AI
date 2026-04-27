"""
MedAgentix AI — Train-Test Split Module
=========================================
Splits the master diagnostic dataset into train/test (and optional validation).
Saves splits to the feature store.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from . import config


def train_test_split_pipeline(
    df: pd.DataFrame,
    target_col: str = None,
    test_size: float = None,
    val_size: float = None,
    random_state: int = None,
) -> dict:
    """
    Split the merged diagnostic dataset into train/test (and optional validation).

    Parameters
    ----------
    df : pd.DataFrame
        Master diagnostic dataset.
    target_col : str, optional
        Target column. Defaults to config.TARGET_COLUMN.
    test_size : float, optional
        Test set proportion. Defaults to config.TEST_SIZE.
    val_size : float, optional
        Validation set proportion. If None, no validation split.
        Defaults to config.VALIDATION_SIZE.
    random_state : int, optional
        Random seed. Defaults to config.RANDOM_STATE.

    Returns
    -------
    dict with keys: X_train, X_test, y_train, y_test, and optionally X_val, y_val.
    """
    print("\n" + "=" * 60)
    print("  PART B — Step 11: Train-Test Split")
    print("=" * 60)

    target_col = target_col or config.TARGET_COLUMN
    test_size = test_size or config.TEST_SIZE
    val_size = val_size if val_size is not None else config.VALIDATION_SIZE
    random_state = random_state or config.RANDOM_STATE

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Drop rows where target is null
    df_clean = df.dropna(subset=[target_col]).copy()
    dropped = len(df) - len(df_clean)
    if dropped > 0:
        print(f"  ⚠️ Dropped {dropped} rows with null target values")

    # Separate features and target
    # Only keep numeric columns as features (encoded data)
    X = df_clean.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
    y = df_clean[target_col]

    # Fill remaining NaN in features with 0
    null_count = X.isnull().sum().sum()
    if null_count > 0:
        print(f"  ℹ️ Filling {null_count} NaN values in features with 0")
        X = X.fillna(0)

    print(f"\n  Features: {X.shape[1]} columns")
    print(f"  Target: '{target_col}' ({y.nunique()} unique classes)")
    print(f"  Total samples: {len(X)}")

    result = {}

    if val_size and val_size > 0:
        # Three-way split: train / val / test
        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        # Second split: train vs val
        val_relative = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_relative,
            random_state=random_state, stratify=y_train_val
        )
        result = {
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
        }
        print(f"\n  Split (train/val/test):")
        print(f"    Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"    Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
        print(f"    Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    else:
        # Two-way split: train vs test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        result = {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
        }
        print(f"\n  Split (train/test):")
        print(f"    Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"    Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    # Save splits to feature store
    config.ensure_dirs()
    for key, data in result.items():
        output_path = config.FEATURE_STORE_DIR / f"{key}.csv"
        if isinstance(data, pd.Series):
            data.to_csv(output_path, index=False, header=True)
        else:
            data.to_csv(output_path, index=False)
        print(f"  💾 {key}: {output_path}")

    print(f"\n  ✅ Train-test split complete. Saved to: {config.FEATURE_STORE_DIR}")
    return result
