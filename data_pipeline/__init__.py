"""
MedAgentix AI — Data Pipeline Package
=======================================
Public API for notebook and script imports.

Usage:
    from data_pipeline import load_all, preprocess_all, run_eda
    from data_pipeline import encode_all, engineer_all
    from data_pipeline import build_master_diagnostic
    from data_pipeline import train_test_split_pipeline
    from data_pipeline import run_feature_importance
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

from .balancing import (
    balance_classes,
    check_imbalance,
)

from .merge_datasets import (
    build_master_diagnostic,
    prepare_agent_datasets,
    prepare_rag_knowledge,
    merge_core_risk,
    merge_with_temporal,
    merge_differential_features,
)

from .train_split import train_test_split_pipeline

from .feature_importance import (
    run_feature_importance,
    xgb_feature_importance,
    shap_analysis,
    correlation_ranking,
    feature_scoring,
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
    # Balancing
    "balance_classes", "check_imbalance",
    # Merging
    "build_master_diagnostic", "prepare_agent_datasets", "prepare_rag_knowledge",
    "merge_core_risk", "merge_with_temporal", "merge_differential_features",
    # Train Split
    "train_test_split_pipeline",
    # Feature Importance
    "run_feature_importance", "xgb_feature_importance", "shap_analysis",
    "correlation_ranking", "feature_scoring",
]
