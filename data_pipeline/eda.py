"""
MedAgentix AI — EDA (Exploratory Data Analysis) Module
========================================================
Reusable visualization functions for all datasets.
Generates histograms, countplots, boxplots, heatmaps, and class distributions.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script usage
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from . import config


# Set global plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'figure.dpi': 100,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})


def _get_plot_dir(name: str) -> Path:
    """Get and create the plot output directory for a dataset."""
    plot_dir = config.EDA_PLOTS_DIR / name
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir


def summary_stats(df: pd.DataFrame, name: str) -> None:
    """
    Print comprehensive summary statistics for a dataset.
    """
    print(f"\n{'='*60}")
    print(f"  EDA SUMMARY: {name}")
    print(f"{'='*60}")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n  --- Data Types ---")
    print(f"  {df.dtypes.value_counts().to_string()}")
    print(f"\n  --- Null Counts ---")
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"  {null_counts[null_counts > 0].to_string()}")
    else:
        print("  No missing values")
    print(f"\n  --- Numeric Summary ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe().to_string())
    else:
        print("  No numeric columns")
    print(f"\n  --- Categorical Unique Counts ---")
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        print(f"  {col}: {df[col].nunique()} unique values")


def plot_missing_values(df: pd.DataFrame, name: str) -> None:
    """
    Bar chart of null values per column.
    """
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]

    if len(null_counts) == 0:
        if config.VERBOSE:
            print(f"  [{name}] No missing values — skipping plot")
        return

    plot_dir = _get_plot_dir(name)
    fig, ax = plt.subplots(figsize=(max(10, len(null_counts) * 0.8), 5))
    null_counts.sort_values(ascending=False).plot(kind='bar', color='coral', edgecolor='black', ax=ax)
    ax.set_title(f"Missing Values — {name}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Column")
    ax.set_ylabel("Null Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(plot_dir / "missing_values.png")
    plt.close(fig)
    if config.VERBOSE:
        print(f"  [{name}] 📊 Missing values plot saved")


def plot_distributions(df: pd.DataFrame, columns: list = None, name: str = "") -> None:
    """
    Plot distributions: histograms for numeric, countplots for categorical.
    """
    plot_dir = _get_plot_dir(name)

    # Numeric distributions
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if columns:
        numeric_cols = [c for c in columns if c in numeric_cols]

    if numeric_cols:
        n = len(numeric_cols)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        axes = np.array(axes).flatten() if n > 1 else [axes]

        for i, col in enumerate(numeric_cols):
            axes[i].hist(df[col].dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
            axes[i].set_title(col, fontsize=11)
            axes[i].set_xlabel('')

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Numeric Distributions — {name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(plot_dir / "numeric_distributions.png")
        plt.close(fig)

    # Categorical distributions (top 10 values for readability)
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if columns:
        cat_cols = [c for c in columns if c in cat_cols]

    # Filter to columns with reasonable cardinality
    cat_cols = [c for c in cat_cols if df[c].nunique() <= 30]

    if cat_cols:
        for col in cat_cols:
            fig, ax = plt.subplots(figsize=(10, 5))
            value_counts = df[col].value_counts().head(15)
            sns.barplot(x=value_counts.values, y=value_counts.index, palette="viridis", ax=ax)
            ax.set_title(f"{col} — {name}", fontsize=13, fontweight='bold')
            ax.set_xlabel("Count")
            plt.tight_layout()
            fig.savefig(plot_dir / f"dist_{col}.png")
            plt.close(fig)

    if config.VERBOSE:
        print(f"  [{name}] 📊 Distribution plots saved ({len(numeric_cols)} numeric, {len(cat_cols)} categorical)")


def plot_boxplots(df: pd.DataFrame, columns: list = None, name: str = "") -> None:
    """
    Boxplots for numeric columns (outlier detection).
    """
    plot_dir = _get_plot_dir(name)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if columns:
        numeric_cols = [c for c in columns if c in numeric_cols]

    if not numeric_cols:
        return

    n = len(numeric_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(numeric_cols):
        sns.boxplot(y=df[col].dropna(), ax=axes[i], color='lightblue')
        axes[i].set_title(col, fontsize=11)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Boxplots — {name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(plot_dir / "boxplots.png")
    plt.close(fig)

    if config.VERBOSE:
        print(f"  [{name}] 📊 Boxplots saved")


def heatmap(df: pd.DataFrame, name: str) -> None:
    """
    Correlation heatmap for numeric columns.
    """
    plot_dir = _get_plot_dir(name)
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        if config.VERBOSE:
            print(f"  [{name}] Skipping heatmap — fewer than 2 numeric columns")
        return

    fig, ax = plt.subplots(figsize=(max(8, numeric_df.shape[1]), max(6, numeric_df.shape[1] * 0.8)))
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
        center=0, linewidths=0.5, ax=ax, square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"Correlation Heatmap — {name}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(plot_dir / "correlation_heatmap.png")
    plt.close(fig)

    if config.VERBOSE:
        print(f"  [{name}] 📊 Correlation heatmap saved")


def plot_class_distribution(df: pd.DataFrame, target: str, name: str) -> None:
    """
    Bar chart of target variable class distribution.
    """
    if target not in df.columns:
        if config.VERBOSE:
            print(f"  [{name}] Target '{target}' not found — skipping class distribution")
        return

    plot_dir = _get_plot_dir(name)
    fig, ax = plt.subplots(figsize=(8, 5))
    value_counts = df[target].value_counts()
    colors = sns.color_palette("Set2", len(value_counts))
    value_counts.plot(kind='bar', color=colors, edgecolor='black', ax=ax)
    ax.set_title(f"Class Distribution: {target} — {name}", fontsize=14, fontweight='bold')
    ax.set_xlabel(target)
    ax.set_ylabel("Count")

    # Add percentage labels
    total = len(df)
    for i, (idx, val) in enumerate(value_counts.items()):
        ax.text(i, val + total * 0.01, f"{val/total*100:.1f}%", ha='center', fontsize=10)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(plot_dir / f"class_distribution_{target}.png")
    plt.close(fig)

    if config.VERBOSE:
        print(f"  [{name}] 📊 Class distribution for '{target}' saved")


def run_eda(datasets: dict) -> None:
    """
    Run full EDA on all datasets.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Dictionary of cleaned DataFrames.
    """
    print("\n" + "=" * 60)
    print("  PART A — Step 3: EDA & Visualization (all 9 datasets)")
    print("=" * 60)

    config.ensure_dirs()

    for name, df in datasets.items():
        print(f"\n{'─'*50}")
        print(f"  EDA: {name}")
        print(f"{'─'*50}")

        summary_stats(df, name)
        plot_missing_values(df, name)
        plot_distributions(df, name=name)
        plot_boxplots(df, name=name)
        heatmap(df, name)

        # Plot class distribution for datasets with targets
        col_config = config.COLUMN_CONFIG.get(name, {})
        target = col_config.get("target")
        if target and target in df.columns:
            plot_class_distribution(df, target, name)

    print(f"\n✅ EDA complete for all {len(datasets)} datasets.")
    print(f"   Plots saved to: {config.EDA_PLOTS_DIR}")
