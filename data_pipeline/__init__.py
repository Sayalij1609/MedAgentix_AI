"""
MedAgentix AI -- Data Pipeline Package
=======================================
Public API for notebook and script imports.

Usage:
    from data_pipeline import load_all, preprocess_all, run_eda
    from data_pipeline import encode_all, engineer_all
    from data_pipeline import encode_all, engineer_all
"""

from .load_data import (
    load_all,
    load_dataset,
    load_core,
    load_drug,
    load_emergency,
    load_medical_knowledge,
    load_risk,
    load_symptom_intelligence,
    load_temporal,
    load_differential,
    load_diagnostic,
)

from .preprocess import (
    preprocess_all,
    preprocess_dataset,
    clean_column_names,
    remove_duplicates,
    handle_missing,
    normalize_labels,
    handle_outliers,
    standardize_symptom_names,
)

from .eda import (
    run_eda,
    summary_stats,
    plot_missing_values,
    plot_distributions,
    plot_boxplots,
    heatmap,
    plot_class_distribution,
)

from .encoding import (
    encode_all,
    encode_dataset,
    label_encode,
    one_hot_encode,
    ordinal_encode,
    binary_encode_symptoms,
)

from .feature_engineering import (
    engineer_all,
    engineer_dataset,
    create_symptom_count,
    create_risk_score,
    create_severity_index,
    create_temporal_score,
    create_interaction_features,
)

from .integration import (
    prepare_agent_datasets,
    prepare_rag_knowledge,
)



__all__ = [
    # Loading
    "load_all", "load_dataset", "load_core", "load_drug", "load_emergency",
    "load_medical_knowledge", "load_risk", "load_symptom_intelligence",
    "load_temporal", "load_differential", "load_diagnostic",
    # Preprocessing
    "preprocess_all", "preprocess_dataset", "clean_column_names",
    "remove_duplicates", "handle_missing", "normalize_labels",
    "handle_outliers", "standardize_symptom_names",
    # EDA
    "run_eda", "summary_stats", "plot_missing_values", "plot_distributions",
    "plot_boxplots", "heatmap", "plot_class_distribution",
    # Encoding
    "encode_all", "encode_dataset", "label_encode", "one_hot_encode",
    "ordinal_encode", "binary_encode_symptoms",
    # Feature Engineering
    "engineer_all", "engineer_dataset", "create_symptom_count",
    "create_risk_score", "create_severity_index", "create_temporal_score",
    "create_interaction_features",
    "create_risk_score", "create_severity_index", "create_temporal_score",
    "create_interaction_features",
    # Integration
    "prepare_agent_datasets", "prepare_rag_knowledge",
]
