"""
MedAgentix AI — Feature Engineering Module
============================================
Creates derived features: symptom counts, risk scores, severity indices,
temporal scores, and interaction features.
"""

import pandas as pd
import numpy as np
from . import config


def create_symptom_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of non-null/non-zero symptom columns per row.
    Works for:
    - Core dataset: binary symptom columns (fever, cough, etc.)
    - Differential dataset: symptom_1 through symptom_17
    - Diagnostic dataset: symptom_1 through symptom_3
    """
    df = df.copy()

    # Detect symptom columns
    symptom_cols = [c for c in df.columns if 'symptom' in c.lower() and c.lower() != 'symptom']
    binary_symptoms = [c for c in ['fever', 'cough', 'fatigue', 'difficulty_breathing'] if c in df.columns]

    if binary_symptoms:
        # Binary Yes/No or 1/0 columns
        df['symptom_count'] = df[binary_symptoms].apply(
            lambda row: sum(1 for v in row if v in [1, 'Yes', 'yes', '1']), axis=1
        )
    elif symptom_cols:
        # Named symptom columns (symptom_1, symptom_2, ...)
        df['symptom_count'] = df[symptom_cols].apply(
            lambda row: sum(1 for v in row if pd.notna(v) and str(v).strip() != ''), axis=1
        )

    return df


def create_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a composite risk score from available risk-related features.
    Combines severity, risk factors, and demographic features.
    """
    df = df.copy()
    risk_components = []

    # Severity contribution
    if 'severity' in df.columns:
        if df['severity'].dtype in ['int64', 'float64']:
            risk_components.append(df['severity'])
        else:
            severity_map = {'Low': 1, 'Moderate': 2, 'High': 3, 'Critical': 4}
            risk_components.append(
                df['severity'].astype(str).str.strip().str.title().map(severity_map).fillna(1)
            )

    # Risk score from risk factor dataset
    if 'risk_score' in df.columns:
        risk_components.append(df['risk_score'])

    # Age-based risk (higher age = higher risk)
    if 'age' in df.columns:
        age_risk = pd.cut(
            df['age'], bins=[0, 18, 40, 60, 100],
            labels=[1, 2, 3, 4], ordered=True
        ).astype(float).fillna(2)
        risk_components.append(age_risk)

    # Blood pressure contribution
    if 'blood_pressure' in df.columns and df['blood_pressure'].dtype in ['int64', 'float64']:
        risk_components.append(df['blood_pressure'])

    if risk_components:
        df['composite_risk_score'] = sum(risk_components) / len(risk_components)
        df['composite_risk_score'] = df['composite_risk_score'].round(2)

    return df


def create_severity_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a numeric severity index from severity-related columns.
    Maps textual severity to a 1-5 scale.
    """
    df = df.copy()
    severity_cols = [c for c in df.columns if 'severity' in c.lower()]

    for col in severity_cols:
        idx_col = f"{col}_index"
        if df[col].dtype in ['int64', 'float64']:
            df[idx_col] = df[col]
        else:
            severity_map = {
                'None': 0, 'Mild': 1, 'Low': 1,
                'Moderate': 2, 'Medium': 2,
                'High': 3, 'Severe': 3,
                'Critical': 4, 'Very High': 4,
            }
            df[idx_col] = (
                df[col].astype(str).str.strip().str.title()
                .map(severity_map).fillna(1)
            )

    return df


def create_temporal_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal scores from duration-related columns.
    Converts duration text to numeric scores.
    """
    df = df.copy()

    if 'duration' in df.columns:
        if df['duration'].dtype == 'object':
            # Extract numeric part from duration strings like "14 days", "<1 day"
            duration_map = {
                '<1 Day': 1, '1 Day': 1, '1-3 Days': 2,
                '3-7 Days': 3, '7 Days': 3, '7-14 Days': 4,
                '14 Days': 4, '>14 Days': 5, '>21 Days': 6,
            }
            df['duration_score'] = (
                df['duration'].astype(str).str.strip().str.title()
                .map(duration_map)
            )
            # Fallback: try to extract numbers
            if df['duration_score'].isnull().all():
                df['duration_score'] = (
                    df['duration'].astype(str)
                    .str.extract(r'(\d+)', expand=False)
                    .astype(float)
                )
            df['duration_score'] = df['duration_score'].fillna(3)  # median default
        else:
            df['duration_score'] = df['duration']

    # Severity flag score
    if 'severity_flag' in df.columns:
        if df['severity_flag'].dtype in ['int64', 'float64']:
            df['temporal_severity_score'] = df['severity_flag']
        else:
            flag_map = {'Low': 1, 'Moderate': 2, 'High': 3}
            df['temporal_severity_score'] = (
                df['severity_flag'].astype(str).str.strip().str.title()
                .map(flag_map).fillna(1)
            )

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between existing columns.
    Examples: age × severity, symptom_count × risk_score.
    """
    df = df.copy()

    # Age × Severity interaction
    if 'age' in df.columns and 'severity' in df.columns:
        sev = df['severity']
        if sev.dtype not in ['int64', 'float64']:
            sev = sev.astype(str).str.strip().str.title().map(
                {'Low': 1, 'Moderate': 2, 'High': 3}
            ).fillna(1)
        df['age_severity_interaction'] = df['age'] * sev

    # Symptom count × Risk score
    if 'symptom_count' in df.columns and 'composite_risk_score' in df.columns:
        df['symptom_risk_interaction'] = df['symptom_count'] * df['composite_risk_score']

    return df


def engineer_dataset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Apply all feature engineering steps to a single dataset.
    Only applies features that are relevant to the dataset's columns.
    """
    print(f"\n{'─'*50}")
    print(f"  FEATURE ENGINEERING: {name}")
    print(f"{'─'*50}")

    df = df.copy()
    initial_cols = set(df.columns)

    df = create_symptom_count(df)
    df = create_risk_score(df)
    df = create_severity_index(df)
    df = create_temporal_score(df)
    df = create_interaction_features(df)

    new_cols = set(df.columns) - initial_cols
    if config.VERBOSE and new_cols:
        print(f"  [{name}] New features created: {sorted(new_cols)}")
    elif config.VERBOSE:
        print(f"  [{name}] No new features applicable for this dataset")

    print(f"  ✅ [{name}] Feature engineering complete — {df.shape[1]} columns")

    # Save engineered output
    config.ensure_dirs()
    output_path = config.ENGINEERED_DIR / f"engineered_{name}.csv"
    df.to_csv(output_path, index=False)
    print(f"  💾 Saved: {output_path}")

    return df


def engineer_all(datasets: dict) -> dict:
    """
    Apply feature engineering to all datasets.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]

    Returns
    -------
    dict[str, pd.DataFrame]
    """
    print("\n" + "=" * 60)
    print("  PART A — Step 5: Feature Engineering (all 9 datasets)")
    print("=" * 60)

    engineered = {}
    for name, df in datasets.items():
        engineered[name] = engineer_dataset(df, name)

    print(f"\n✅ All {len(engineered)} datasets feature-engineered and saved.")
    return engineered
