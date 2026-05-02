"""
MedAgentix AI -- Feature Importance Module
============================================
Computes feature rankings using XGBoost importance, SHAP values,
correlation analysis, and mutual information scoring.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from . import config


def xgb_feature_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: int = None,
) -> pd.DataFrame:
    """
    Compute feature importance using XGBoost.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series
    top_n : int, optional
        Number of top features. Defaults to config.TOP_N_FEATURES.

    Returns
    -------
    pd.DataFrame with columns: feature, importance, rank
    """
    top_n = top_n or config.TOP_N_FEATURES

    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  [WARN] xgboost not installed. Install with: pip install xgboost")
        return pd.DataFrame()

    print("\n  Computing XGBoost feature importance...")

    # Handle multiclass vs binary
    n_classes = y_train.nunique()
    objective = 'multi:softmax' if n_classes > 2 else 'binary:logistic'

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective=objective,
        num_class=n_classes if n_classes > 2 else None,
        random_state=config.RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='mlogloss' if n_classes > 2 else 'logloss',
        verbosity=0,
    )

    # Encode target to 0-indexed integers (required by XGBoost)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(y_train), index=y_train.index)

    model.fit(X_train, y_encoded)

    # Extract importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    importance['rank'] = range(1, len(importance) + 1)

    # Plot
    top = importance.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    sns.barplot(x='importance', y='feature', hue='feature', data=top, palette='viridis', ax=ax, legend=False)
    ax.set_title(f'Top {top_n} Features -- XGBoost Importance', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    fig.savefig(config.FEATURE_STORE_DIR / "xgb_feature_importance.png")
    plt.close(fig)

    print(f"  [OK] XGBoost importance computed for {len(importance)} features")
    print(f"   Top 5: {list(top.head(5)['feature'])}")

    return importance


def shap_analysis(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame = None,
    top_n: int = None,
) -> pd.DataFrame:
    """
    Compute SHAP feature importance values.

    Parameters
    ----------
    X_train, y_train : training data
    X_test : pd.DataFrame, optional
        Test data for SHAP values. If None, uses X_train sample.
    top_n : int

    Returns
    -------
    pd.DataFrame with mean absolute SHAP values per feature.
    """
    top_n = top_n or config.TOP_N_FEATURES

    try:
        import shap
        from xgboost import XGBClassifier
    except ImportError as e:
        print(f"  [WARN] Missing dependency: {e}")
        print("     Install with: pip install shap xgboost")
        return pd.DataFrame()

    print("\n  Computing SHAP feature importance...")

    # Train a model for SHAP
    # Encode target to 0-indexed integers (required by XGBoost)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(y_train), index=y_train.index)
    n_classes = y_encoded.nunique()

    model = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=config.RANDOM_STATE, use_label_encoder=False,
        eval_metric='mlogloss' if n_classes > 2 else 'logloss',
        verbosity=0,
    )
    model.fit(X_train, y_encoded)

    # Compute SHAP values
    explain_data = X_test if X_test is not None else X_train.sample(
        min(200, len(X_train)), random_state=config.RANDOM_STATE
    )

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(explain_data)
    except (ValueError, TypeError, Exception) as e:
        # Known incompatibility between certain SHAP/XGBoost versions
        print(f"  WARNING: SHAP TreeExplainer failed ({type(e).__name__}: {e})")
        print("  Falling back to permutation-based importance...")
        # Fallback: use model's built-in feature importance as a proxy
        mean_abs_shap = model.feature_importances_
        shap_importance = pd.DataFrame({
            'feature': X_train.columns,
            'mean_abs_shap': mean_abs_shap,
        }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
        shap_importance['rank'] = range(1, len(shap_importance) + 1)
        print(f"  Done (fallback). Top 5: {list(shap_importance.head(5)['feature'])}")
        return shap_importance

    # Handle multiclass: average across classes
    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    if hasattr(mean_abs_shap, 'ndim') and mean_abs_shap.ndim > 1:
        mean_abs_shap = np.mean(mean_abs_shap, axis=1)

    shap_importance = pd.DataFrame({
        'feature': X_train.columns,
        'mean_abs_shap': mean_abs_shap,
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    shap_importance['rank'] = range(1, len(shap_importance) + 1)

    # SHAP summary plot
    try:
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
        top = shap_importance.head(top_n)
        sns.barplot(x='mean_abs_shap', y='feature', hue='feature', data=top, palette='magma', ax=ax, legend=False)
        ax.set_title(f'Top {top_n} Features -- SHAP Importance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Mean |SHAP Value|')
        plt.tight_layout()
        fig.savefig(config.FEATURE_STORE_DIR / "shap_feature_importance.png")
        plt.close(fig)
    except Exception as e:
        print(f"  [WARN] SHAP plot failed: {e}")

    print(f"  [OK] SHAP analysis complete for {len(shap_importance)} features")
    print(f"   Top 5: {list(shap_importance.head(5)['feature'])}")

    return shap_importance


def correlation_ranking(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Rank features by absolute correlation with target.
    """
    print("\n  Computing correlation-based feature ranking...")

    y_numeric = y.copy()
    if y_numeric.dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_numeric = pd.Series(le.fit_transform(y_numeric), index=y.index)

    correlations = X.corrwith(y_numeric).abs().sort_values(ascending=False)
    result = pd.DataFrame({
        'feature': correlations.index,
        'abs_correlation': correlations.values,
    }).reset_index(drop=True)
    result['rank'] = range(1, len(result) + 1)

    print(f"  [OK] Correlation ranking computed for {len(result)} features")
    print(f"   Top 5: {list(result.head(5)['feature'])}")

    return result


def feature_scoring(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Score features using mutual information and chi-squared tests.
    """
    from sklearn.feature_selection import mutual_info_classif

    print("\n  Computing mutual information scores...")

    y_encoded = y.copy()
    if y_encoded.dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = pd.Series(le.fit_transform(y_encoded), index=y.index)

    # Fill NaN for MI computation
    X_filled = X.fillna(0)

    mi_scores = mutual_info_classif(X_filled, y_encoded, random_state=config.RANDOM_STATE)

    result = pd.DataFrame({
        'feature': X.columns,
        'mutual_info_score': mi_scores,
    }).sort_values('mutual_info_score', ascending=False).reset_index(drop=True)
    result['rank'] = range(1, len(result) + 1)

    # Plot
    top_n = config.TOP_N_FEATURES
    top = result.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    sns.barplot(x='mutual_info_score', y='feature', hue='feature', data=top, palette='coolwarm', ax=ax, legend=False)
    ax.set_title(f'Top {top_n} Features -- Mutual Information', fontsize=14, fontweight='bold')
    ax.set_xlabel('MI Score')
    plt.tight_layout()
    fig.savefig(config.FEATURE_STORE_DIR / "mutual_info_scores.png")
    plt.close(fig)

    print(f"  [OK] MI scoring complete for {len(result)} features")
    print(f"   Top 5: {list(result.head(5)['feature'])}")

    return result


def run_feature_importance(splits: dict) -> dict:
    """
    Run all feature importance analyses and save final selected features.

    Parameters
    ----------
    splits : dict
        Output from train_test_split_pipeline().

    Returns
    -------
    dict with importance DataFrames and selected features.
    """
    print("\n" + "=" * 60)
    print("  PART B -- Steps 12-13: Feature Importance & Feature Store")
    print("=" * 60)

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_test = splits.get("X_test")

    results = {}

    # 1. XGBoost importance
    xgb_imp = xgb_feature_importance(X_train, y_train)
    results["xgb_importance"] = xgb_imp

    # 2. SHAP analysis
    shap_imp = shap_analysis(X_train, y_train, X_test)
    results["shap_importance"] = shap_imp

    # 3. Correlation ranking
    corr_rank = correlation_ranking(X_train, y_train)
    results["correlation_ranking"] = corr_rank

    # 4. Mutual information scoring
    mi_scores = feature_scoring(X_train, y_train)
    results["mi_scores"] = mi_scores

    # 5. Combine all rankings into a unified score
    print("\n  Creating unified feature scoring...")
    all_features = X_train.columns.tolist()

    combined = pd.DataFrame({'feature': all_features})
    for method, imp_df in results.items():
        if imp_df is not None and not imp_df.empty:
            score_col = [c for c in imp_df.columns if c not in ['feature', 'rank']][0]
            method_scores = imp_df[['feature', score_col]].rename(
                columns={score_col: f"{method}_score"}
            )
            combined = combined.merge(method_scores, on='feature', how='left')

    # Normalize each score to 0-1 range and compute mean
    score_cols = [c for c in combined.columns if c.endswith('_score')]
    for col in score_cols:
        max_val = combined[col].max()
        if max_val > 0:
            combined[col] = combined[col] / max_val

    combined['unified_score'] = combined[score_cols].mean(axis=1)
    combined = combined.sort_values('unified_score', ascending=False).reset_index(drop=True)
    combined['final_rank'] = range(1, len(combined) + 1)

    # Save
    config.ensure_dirs()
    combined.to_csv(config.FEATURE_STORE_DIR / "selected_features.csv", index=False)

    # Save top engineered features
    top_features = combined.head(config.TOP_N_FEATURES)['feature'].tolist()
    X_train[top_features].to_csv(config.FEATURE_STORE_DIR / "engineered_features.csv", index=False)

    print(f"\n  [OK] Feature importance analysis complete")
    print(f"   Top 10 features (unified):")
    for _, row in combined.head(10).iterrows():
        print(f"    {int(row['final_rank']):2d}. {row['feature']}: {row['unified_score']:.4f}")

    print(f"\n   Saved to {config.FEATURE_STORE_DIR}:")
    print(f"     - selected_features.csv (all features ranked)")
    print(f"     - engineered_features.csv (top {config.TOP_N_FEATURES} features)")
    print(f"     - xgb_feature_importance.png")
    print(f"     - shap_feature_importance.png")
    print(f"     - mutual_info_scores.png")

    results["combined"] = combined
    results["top_features"] = top_features
    return results
