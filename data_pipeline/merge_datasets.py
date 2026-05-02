"""
MedAgentix AI — Dataset Merging Module
========================================
Handles:
- Step 8:  Merge Group A datasets → master_diagnostic.csv
- Step 9:  Prepare Group B agent datasets (separate, no merge)
- Step 10: Prepare Group C RAG knowledge text chunks
"""

import pandas as pd
import numpy as np
from . import config


# --- GROUP A: Diagnostic Merge -----------------------------------------------

def merge_core_risk(core_df: pd.DataFrame, risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Core Clinical with Risk Factor dataset.

    Strategy:
    - Risk has multiple rows per Condition (different risk factors).
    - Aggregate risk features per condition: max risk_score, count of risk factors,
      concatenated risk factors, most common lifestyle profile.
    - Left join Core ← Aggregated Risk on disease = condition.
    """
    print("\n  Step 8a: Merging Core ← Risk Factor")

    # Standardize join keys
    core = core_df.copy()
    risk = risk_df.copy()

    # Ensure join key columns exist (handle both old and new column names)
    core_key = 'disease' if 'disease' in core.columns else None
    risk_key = None
    for candidate in ['associated_condition', 'condition']:
        if candidate in risk.columns:
            risk_key = candidate
            break

    if not core_key or not risk_key:
        print(f"  [WARN] Join keys not found (core has 'disease': {core_key is not None}, "
              f"risk has join key: {risk_key}) — skipping merge")
        return core

    # Aggregate risk by condition
    agg_dict = {}
    if 'risk_score' in risk.columns:
        agg_dict['risk_score'] = ['max', 'mean']
    if 'risk_factor' in risk.columns:
        agg_dict['risk_factor'] = ['count', lambda x: ', '.join(x.astype(str).unique()[:5])]
    if 'weight' in risk.columns:
        agg_dict['weight'] = 'max'
    if 'risk_type' in risk.columns:
        agg_dict['risk_type'] = lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
    if 'lifestyle_profile' in risk.columns:
        agg_dict['lifestyle_profile'] = lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'

    if not agg_dict:
        print("  [WARN] No aggregatable columns in risk dataset — skipping")
        return core

    risk_agg = risk.groupby(risk_key).agg(agg_dict)
    risk_agg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                        for col in risk_agg.columns]
    # Clean column names and fix lambda bug
    risk_agg.columns = [c.replace('<lambda_0>', 'list').replace('<lambda>', 'list') for c in risk_agg.columns]
    risk_agg = risk_agg.reset_index()

    # Rename condition to disease for join
    risk_agg = risk_agg.rename(columns={risk_key: core_key})

    # Expand risk_agg using config.RISK_CONDITION_TO_CORE_DISEASE
    expanded_rows = []
    for _, row in risk_agg.iterrows():
        condition = str(row[core_key]).strip()
        # Find matching core diseases
        matched_diseases = []
        for risk_cond, core_diseases in config.RISK_CONDITION_TO_CORE_DISEASE.items():
            if condition.lower() == risk_cond.lower():
                matched_diseases.extend(core_diseases)
        
        if not matched_diseases:
            # If no mapping, keep the original name
            matched_diseases = [condition]
            
        for disease in matched_diseases:
            new_row = row.copy()
            new_row[core_key] = disease
            expanded_rows.append(new_row)
            
    if expanded_rows:
        risk_agg = pd.DataFrame(expanded_rows)

    # Standardize join key values
    core[core_key] = core[core_key].astype(str).str.strip().str.title()
    risk_agg[core_key] = risk_agg[core_key].astype(str).str.strip().str.title()

    before = len(core)
    merged = core.merge(risk_agg, on=core_key, how='left')
    matched = merged[risk_agg.columns[1]].notna().sum()

    print(f"  [OK] Core ← Risk merged: {before} rows, {matched} matched ({matched/before*100:.1f}%)")
    return merged


def merge_with_temporal(merged_df: pd.DataFrame, temporal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge with Temporal dataset features.

    Strategy:
    - Temporal is symptom-level: each row = (Symptom, Duration, Severity_Flag, ...).
    - Aggregate temporal features per symptom.
    - For each row in Core, look up temporal info for present symptoms
      and create aggregate temporal features.
    """
    print("\n  Step 8b: Merging ← Temporal features")

    merged = merged_df.copy()
    temporal = temporal_df.copy()

    if 'symptom' not in temporal.columns:
        print("  [WARN] 'symptom' column not found in temporal — skipping")
        return merged

    # Create temporal summary per symptom
    temporal_agg_dict = {}

    # Handle risk level column (new CSVs use 'risk_level', old used 'severity_flag')
    risk_col = None
    for candidate in ['risk_level', 'severity_flag']:
        if candidate in temporal.columns:
            risk_col = candidate
            break

    if risk_col:
        if temporal[risk_col].dtype in ['int64', 'float64']:
            temporal_agg_dict[risk_col] = 'max'
        else:
            temporal[f'{risk_col}_num'] = temporal[risk_col].astype(str).str.strip().str.title().map(
                {'Low': 1, 'Medium': 2, 'Moderate': 2, 'High': 3, 'Critical': 4}
            ).fillna(1)
            temporal_agg_dict[f'{risk_col}_num'] = 'max'

    # Handle pattern column (new CSVs use 'clinical_category', old used 'temporal_pattern')
    for pattern_col in ['clinical_category', 'temporal_pattern']:
        if pattern_col in temporal.columns:
            temporal_agg_dict[pattern_col] = lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown'
            break

    if temporal_agg_dict:
        temporal_summary = temporal.groupby('symptom').agg(temporal_agg_dict).reset_index()
        temporal_summary.columns = [
            c.replace('<lambda>', 'mode') if isinstance(c, str) else '_'.join(c)
            for c in temporal_summary.columns
        ]

        # Map symptom names
        temporal_summary['symptom'] = temporal_summary['symptom'].astype(str).str.strip().str.title()

        # Create temporal feature for merged dataset
        # Determine the risk numeric column name
        risk_num_col = f'{risk_col}_num' if risk_col and f'{risk_col}_num' in temporal_summary.columns else (
            risk_col if risk_col and risk_col in temporal_summary.columns else None
        )

        # Check for binary symptom columns (fever, cough, etc.)
        symptom_cols = [c for c in merged.columns if c in ['fever', 'cough', 'fatigue', 'difficulty_breathing']]

        if symptom_cols and risk_num_col:
            # For each symptom column, look up temporal severity
            temporal_lookup = dict(zip(
                temporal_summary['symptom'],
                temporal_summary[risk_num_col]
            ))

            def calc_temporal_score(row):
                score = 0
                count = 0
                for col in symptom_cols:
                    if row.get(col) in [1, 'Yes', 'yes']:
                        symptom_name = col.title()
                        score += temporal_lookup.get(symptom_name, 1)
                        count += 1
                return score / count if count > 0 else 0

            merged['temporal_risk_score'] = merged.apply(calc_temporal_score, axis=1)
            print(f"  [OK] Temporal features added: temporal_risk_score")
        else:
            print(f"  [INFO] No binary symptom columns found — adding temporal summary as separate features")
            # Fall back to just adding overall temporal stats
            if risk_num_col and risk_num_col in temporal_summary.columns:
                merged['avg_temporal_severity'] = temporal_summary[risk_num_col].mean()
            else:
                merged['avg_temporal_severity'] = 1.0

    return merged


def merge_differential_features(merged_df: pd.DataFrame, diff_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optionally merge engineered features from Differential Diagnosis dataset.

    Strategy:
    - Extract: number of possible diseases per disease, symptom count.
    - Left join on Disease.
    """
    print("\n  Step 8c: Merging ← Differential features (optional)")

    merged = merged_df.copy()
    diff = diff_df.copy()

    if 'disease' not in diff.columns:
        print("  [WARN] 'disease' column not found in differential — skipping")
        return merged

    # Engineer features from differential
    symptom_cols = [c for c in diff.columns if c.startswith('symptom_')]
    if symptom_cols:
        diff['diff_symptom_count'] = diff[symptom_cols].apply(
            lambda row: sum(1 for v in row if pd.notna(v) and str(v).strip() != ''), axis=1
        )

    if 'possible_diseases' in diff.columns:
        diff['diff_possible_disease_count'] = diff['possible_diseases'].astype(str).str.count(',') + 1

    # Aggregate per disease
    agg_cols = {}
    if 'diff_symptom_count' in diff.columns:
        agg_cols['diff_symptom_count'] = 'max'
    if 'diff_possible_disease_count' in diff.columns:
        agg_cols['diff_possible_disease_count'] = 'max'

    if not agg_cols:
        print("  [WARN] No features to extract from differential — skipping")
        return merged

    # Map disease names before aggregation
    def map_diff_disease(disease_name):
        disease_name = str(disease_name).strip()
        for diff_name, core_name in config.DIFF_DISEASE_TO_CORE_DISEASE.items():
            if disease_name.lower() == diff_name.lower():
                return core_name
        return disease_name

    diff['disease'] = diff['disease'].apply(map_diff_disease)

    diff_agg = diff.groupby('disease').agg(agg_cols).reset_index()
    diff_agg['disease'] = diff_agg['disease'].astype(str).str.strip().str.title()

    if 'disease' in merged.columns:
        merged['disease'] = merged['disease'].astype(str).str.strip().str.title()
        before_cols = merged.shape[1]
        merged = merged.merge(diff_agg, on='disease', how='left')
        new_cols = merged.shape[1] - before_cols
        print(f"  [OK] Differential features added: {new_cols} new columns")

    return merged


def build_master_diagnostic(datasets: dict) -> pd.DataFrame:
    """
    Build the master diagnostic dataset by merging Group A datasets.

    Merge order:
    1. Core Clinical (anchor)
    2. ← Risk Factor (aggregated by condition)
    3. ← Temporal (aggregated symptom-level features)
    4. ← Differential (optional engineered features)

    Output: datasets/processed/merged/master_diagnostic.csv
    """
    print("\n" + "=" * 60)
    print("  PART B - Step 8: Build Master Diagnostic Dataset")
    print("=" * 60)

    # Get Group A datasets
    core = datasets.get("core")
    risk = datasets.get("risk")
    temporal = datasets.get("temporal")
    differential = datasets.get("differential")

    if core is None:
        raise ValueError("Core Clinical dataset is required for diagnostic merge!")

    result = core.copy()
    print(f"\n  Starting with Core Clinical: {result.shape[0]} rows x {result.shape[1]} cols")

    if risk is not None:
        result = merge_core_risk(result, risk)
    else:
        print("  [WARN] Risk dataset not available - skipping")

    if temporal is not None:
        result = merge_with_temporal(result, temporal)
    else:
        print("  [WARN] Temporal dataset not available - skipping")

    # Merge Differential (optional)
    if differential is not None:
        result = merge_differential_features(result, differential)
    else:
        print("  [INFO] Differential dataset not included (optional)")

    # Fill remaining missing values to ensure a non-empty dataset
    for col in result.columns:
        if result[col].dtype == 'object':
            result[col] = result[col].fillna("0")
        else:
            result[col] = result[col].fillna(0.0)

    # Save master diagnostic
    config.ensure_dirs()
    output_path = config.MERGED_DIR / "master_diagnostic.csv"
    result.to_csv(output_path, index=False)

    print(f"\n  {'='*50}")
    print(f"  [OK] Master Diagnostic Dataset: {result.shape[0]} rows x {result.shape[1]} cols")
    print(f"   Saved: {output_path}")
    print(f"  {'='*50}")

    return result


# --- GROUP B: Agent Dataset Preparation --------------------------------------

def prepare_agent_datasets(datasets: dict) -> dict:
    """
    Prepare Group B datasets as separate agent-ready datasets.
    Each agent gets its own subdirectory under agent_datasets/.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]

    Returns
    -------
    dict[str, pd.DataFrame]
        Agent name → DataFrame
    """
    print("\n" + "=" * 60)
    print("  PART B — Step 9: Prepare Separate Agent Datasets")
    print("=" * 60)

    config.ensure_dirs()
    agent_data = {}

    for agent_name, dataset_name in config.GROUP_B_AGENTS.items():
        df = datasets.get(dataset_name)
        if df is None:
            print(f"  [WARN] {dataset_name} not found — skipping {agent_name}")
            continue

        # Create agent directory
        agent_dir = config.AGENT_DATASETS_DIR / agent_name
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Save agent dataset
        output_path = agent_dir / f"{dataset_name}.csv"
        df.to_csv(output_path, index=False)

        agent_data[agent_name] = df
        print(f"  [OK] {agent_name}: {df.shape[0]} rows × {df.shape[1]} cols → {output_path}")

    print(f"\n[OK] {len(agent_data)} agent datasets prepared.")
    return agent_data


# --- GROUP C: RAG Knowledge Store --------------------------------------------

def prepare_rag_knowledge(datasets: dict) -> pd.DataFrame:
    """
    Prepare Medical Knowledge dataset as text chunks for RAG.
    Concatenates relevant text fields into single text chunks.

    Output: datasets/processed/rag_knowledge/knowledge_chunks.csv
    """
    print("\n" + "=" * 60)
    print("  PART B — Step 10: Prepare RAG Knowledge Store")
    print("=" * 60)

    config.ensure_dirs()
    rag_dir = config.RAG_KNOWLEDGE_DIR
    rag_dir.mkdir(parents=True, exist_ok=True)

    mk = datasets.get("medical_knowledge")
    if mk is None:
        print("  [WARN] Medical Knowledge dataset not found — skipping")
        return pd.DataFrame()

    # Build text chunks
    chunks = []
    for _, row in mk.iterrows():
        disease = str(row.get('disease', '')).strip()
        description = str(row.get('description', '')).strip()
        cause = str(row.get('cause', '')).strip()
        category = str(row.get('category', '')).strip()
        severity = str(row.get('severity', '')).strip()
        progression = str(row.get('disease_progression', '')).strip()
        complications = str(row.get('common_complications', '')).strip()
        management = str(row.get('primary_management', row.get('clinical_management', ''))).strip()
        prevalence = str(row.get('prevalence', '')).strip()

        # Create structured text chunk
        text = (
            f"Disease: {disease}\n"
            f"Description: {description}\n"
            f"Cause: {cause}\n"
            f"Category: {category}\n"
            f"Severity: {severity}\n"
            f"Prevalence: {prevalence}\n"
            f"Disease Progression: {progression}\n"
            f"Common Complications: {complications}\n"
            f"Primary Management: {management}"
        )

        chunks.append({
            "disease": disease,
            "category": category,
            "severity": severity,
            "text_chunk": text,
            "chunk_length": len(text),
        })

    chunks_df = pd.DataFrame(chunks)

    # Save
    output_path = rag_dir / "knowledge_chunks.csv"
    chunks_df.to_csv(output_path, index=False)

    print(f"  [OK] Knowledge chunks: {len(chunks_df)} entries")
    print(f"   Avg chunk length: {chunks_df['chunk_length'].mean():.0f} chars")
    print(f"   Saved: {output_path}")

    return chunks_df
