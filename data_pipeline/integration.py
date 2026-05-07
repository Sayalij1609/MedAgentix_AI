"""
MedAgentix AI -- Data Integration Module
==========================================
Handles preparation of Agent Datasets (Group B) and RAG Knowledge chunks (Group C).
"""

import pandas as pd
from . import config

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
    print("  Step 8: Prepare Separate Agent Datasets (Group B)")
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
    print("  Step 9: Prepare RAG Knowledge Store (Group C)")
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
