"""
features.py — Feature Analysis & Selection
============================================
Member 2 — Feature Analysis

Responsibilities:
  • Correlation heatmap between all features and the outcome
  • Feature importance using Random Forest
  • Compare all features vs best features vs worst features
  • Return the best features to use for the models
  • Deliver a bar chart of feature importance
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ──────────────────────────────────────────────────────────
#  PATHS
# ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHART_DIR = os.path.join(BASE_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────
#  1. CORRELATION HEATMAP
# ──────────────────────────────────────────────────────────
def plot_correlation_heatmap(df_encoded: pd.DataFrame):
    """
    Generate a correlation heatmap between all features and the target.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df_encoded.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        vmin=-1,
        vmax=1,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "correlation_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[CHART] Saved: {path}")
    return corr


# ──────────────────────────────────────────────────────────
#  2. FEATURE IMPORTANCE (Random Forest)
# ──────────────────────────────────────────────────────────
def compute_feature_importance(X, y, feature_cols):
    """
    Train a Random Forest and extract feature importances.
    Returns a sorted DataFrame of feature importances.
    """
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importances = rf.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    print("\n[INFO] Feature Importances (Random Forest):")
    print(importance_df.to_string(index=False))
    return importance_df, rf


def plot_feature_importance(importance_df: pd.DataFrame):
    """Bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
    bars = ax.barh(
        importance_df["Feature"][::-1],
        importance_df["Importance"][::-1],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title("Feature Importance (Random Forest)", fontsize=14, fontweight="bold")

    # Add value labels
    for bar, val in zip(bars, importance_df["Importance"][::-1]):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    path = os.path.join(CHART_DIR, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[CHART] Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────
#  3. FEATURE SUBSET COMPARISON
# ──────────────────────────────────────────────────────────
def compare_feature_subsets(X, y, feature_cols, importance_df):
    """
    Compare model accuracy using:
      1. All features
      2. Best features (top 50%)
      3. Worst features (bottom 50%)

    Uses Logistic Regression with 5-fold cross-validation.
    """
    n = len(feature_cols)
    mid = max(1, n // 2)

    best_features = importance_df["Feature"].head(mid).tolist()
    worst_features = importance_df["Feature"].tail(mid).tolist()

    subsets = {
        "All Features": feature_cols,
        "Best Features (Top 50%)": best_features,
        "Worst Features (Bottom 50%)": worst_features,
    }

    results = {}
    scaler = StandardScaler()

    for label, cols in subsets.items():
        X_sub = X[cols] if isinstance(X, pd.DataFrame) else X[:, [feature_cols.index(c) for c in cols]]
        X_scaled = scaler.fit_transform(X_sub)
        scores = cross_val_score(
            LogisticRegression(max_iter=1000, random_state=42),
            X_scaled, y, cv=5, scoring="accuracy"
        )
        mean_acc = scores.mean()
        results[label] = {
            "features": cols,
            "accuracy": mean_acc,
            "std": scores.std(),
        }
        print(f"[INFO] {label}: Accuracy = {mean_acc:.4f} ± {scores.std():.4f}  (features: {cols})")

    # Plot comparison
    _plot_feature_subset_comparison(results)
    return results


def _plot_feature_subset_comparison(results: dict):
    """Bar chart comparing accuracy across feature subsets."""
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(results.keys())
    accs = [results[k]["accuracy"] for k in labels]
    stds = [results[k]["std"] for k in labels]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    bars = ax.bar(labels, accs, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy (5-Fold CV)", fontsize=12)
    ax.set_title("Feature Subset Comparison", fontsize=14, fontweight="bold")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.4f}", ha="center", fontweight="bold", fontsize=11)

    plt.tight_layout()
    path = os.path.join(CHART_DIR, "feature_subset_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[CHART] Saved: {path}")


# ──────────────────────────────────────────────────────────
#  4. SELECT BEST FEATURES
# ──────────────────────────────────────────────────────────
def select_best_features(importance_df: pd.DataFrame, threshold: float = 0.05):
    """
    Return features whose importance exceeds the threshold.
    If all are below threshold, return the top 3.
    """
    best = importance_df[importance_df["Importance"] >= threshold]["Feature"].tolist()
    if len(best) == 0:
        best = importance_df["Feature"].head(3).tolist()
    print(f"\n[INFO] Selected best features (threshold={threshold}): {best}")
    return best


# ──────────────────────────────────────────────────────────
#  5. FULL PIPELINE
# ──────────────────────────────────────────────────────────
def run_feature_analysis(preprocessing_result: dict):
    """
    Execute the full feature analysis pipeline.
    Returns a dict with analysis outputs.
    """
    print("=" * 60)
    print("  FEATURE ANALYSIS")
    print("=" * 60)

    df_encoded = preprocessing_result["df_encoded"]
    X = preprocessing_result["X"]
    y = preprocessing_result["y"]
    feature_cols = preprocessing_result["feature_cols"]

    # 1. Correlation heatmap
    corr_matrix = plot_correlation_heatmap(df_encoded)

    # 2. Feature importance
    importance_df, rf_model = compute_feature_importance(X, y, feature_cols)
    plot_feature_importance(importance_df)

    # 3. Feature subset comparison
    subset_results = compare_feature_subsets(X, y, feature_cols, importance_df)

    # 4. Best features
    best_features = select_best_features(importance_df)

    result = {
        "corr_matrix": corr_matrix,
        "importance_df": importance_df,
        "rf_model": rf_model,
        "subset_results": subset_results,
        "best_features": best_features,
    }

    print("\n[SUCCESS] Feature analysis complete.\n")
    return result


if __name__ == "__main__":
    from preprocessing.preprocessing import run_preprocessing_pipeline
    prep = run_preprocessing_pipeline()
    feat = run_feature_analysis(prep)
