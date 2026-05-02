"""
evaluation.py — Model Comparison & Evaluation
===============================================
Member 5 — Evaluation

Responsibilities:
  • Collect results from all 4 models
  • Build comparison table: Accuracy, Precision, Recall, F1
  • Build confusion matrix for each model
  • Write rationale explaining model performance
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHART_DIR = os.path.join(BASE_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────
#  1. COMPARISON TABLE
# ──────────────────────────────────────────────────────────
def build_comparison_table(all_results: list) -> pd.DataFrame:
    """
    Build a comparison DataFrame from all model results.

    Parameters:
        all_results: list of dicts from each model

    Returns:
        DataFrame with columns: Model, Accuracy, Precision, Recall, F1, CV_Accuracy
    """
    rows = []
    for r in all_results:
        rows.append({
            "Model": r["model_name"],
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1-Score": r["f1"],
            "CV Accuracy": r["cv_accuracy"],
            "CV Std": r["cv_accuracy_std"],
        })

    table = pd.DataFrame(rows)
    table = table.sort_values("F1-Score", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("  MODEL COMPARISON TABLE")
    print("=" * 70)
    print(table.to_string(index=False))
    return table


def plot_comparison_chart(table: pd.DataFrame):
    """
    Grouped bar chart comparing all metrics across models.
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    models = table["Model"].tolist()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.18
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]

    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, table[metric], width, label=metric,
                      color=colors[i], edgecolor="black", linewidth=0.5)
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.005,
                    f"{height:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(CHART_DIR, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[CHART] Saved: {path}")
    return path


def plot_cv_comparison(table: pd.DataFrame):
    """Bar chart of cross-validation accuracy with error bars."""
    fig, ax = plt.subplots(figsize=(10, 5))
    models = table["Model"].tolist()
    cv_accs = table["CV Accuracy"].tolist()
    cv_stds = table["CV Std"].tolist()
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

    bars = ax.bar(models, cv_accs, yerr=cv_stds, capsize=6,
                  color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("CV Accuracy", fontsize=12)
    ax.set_title("Cross-Validation Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    for bar, acc in zip(bars, cv_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.4f}", ha="center", fontweight="bold", fontsize=10)

    plt.tight_layout()
    path = os.path.join(CHART_DIR, "cv_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[CHART] Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────
#  2. CONFUSION MATRICES
# ──────────────────────────────────────────────────────────
def plot_confusion_matrices(all_results: list, y_test):
    """Plot confusion matrix for each model side by side."""
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for i, r in enumerate(all_results):
        cm = r["confusion_matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[i],
            cbar=False,
            xticklabels=["Not Dangerous", "Dangerous"],
            yticklabels=["Not Dangerous", "Dangerous"],
        )
        axes[i].set_title(r["model_name"], fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    plt.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "confusion_matrices.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[CHART] Saved: {path}")
    return path


def plot_individual_confusion_matrix(result: dict):
    """Plot a single confusion matrix for one model."""
    cm = result["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Not Dangerous", "Dangerous"],
        yticklabels=["Not Dangerous", "Dangerous"],
    )
    ax.set_title(f"Confusion Matrix — {result['model_name']}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()

    safe_name = result["model_name"].replace(" ", "_").replace("(", "").replace(")", "").lower()
    path = os.path.join(CHART_DIR, f"cm_{safe_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[CHART] Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────
#  3. RATIONALE / ANALYSIS
# ──────────────────────────────────────────────────────────
def generate_rationale(table: pd.DataFrame, all_results: list) -> str:
    """
    Generate a written rationale explaining why each model
    performed the way it did.
    """
    rationale = []
    rationale.append("=" * 70)
    rationale.append("  COMPARATIVE ANALYSIS — RATIONALE")
    rationale.append("=" * 70)
    rationale.append("")

    best_model = table.iloc[0]["Model"]
    best_f1 = table.iloc[0]["F1-Score"]
    worst_model = table.iloc[-1]["Model"]
    worst_f1 = table.iloc[-1]["F1-Score"]

    rationale.append(f"BEST PERFORMER: {best_model} (F1-Score: {best_f1:.4f})")
    rationale.append(f"WORST PERFORMER: {worst_model} (F1-Score: {worst_f1:.4f})")
    rationale.append("")

    for _, row in table.iterrows():
        model = row["Model"]
        rationale.append(f"--- {model} ---")

        if "Neural Network" in model:
            rationale.append(
                "  The Neural Network (MLPClassifier) is a non-linear model capable of "
                "learning complex decision boundaries through multiple hidden layers. "
                "It typically performs well when features have non-linear relationships "
                "with the target. However, it requires more data and is sensitive to "
                "hyperparameter choices (learning rate, architecture, regularization). "
                "Feature scaling is critical for MLP performance."
            )
        elif "kNN" in model or "Nearest" in model:
            rationale.append(
                "  k-Nearest Neighbors is an instance-based learner that classifies "
                "based on the majority vote of nearest neighbors in feature space. "
                "It works well with small datasets and when decision boundaries are "
                "locally defined. Performance depends heavily on the choice of k and "
                "the distance metric. kNN is sensitive to irrelevant features and "
                "the curse of dimensionality, which is why feature scaling is essential."
            )
        elif "Naive Bayes" in model or "Bayes" in model:
            rationale.append(
                "  Gaussian Naive Bayes assumes feature independence and Gaussian "
                "feature distributions. It is fast and works well even with small "
                "datasets. However, its 'naive' independence assumption can hurt "
                "performance when features are correlated. If symptoms tend to "
                "co-occur, this model may underperform compared to models that "
                "capture feature interactions (like SVM with RBF kernel or NN)."
            )
        elif "SVM" in model:
            rationale.append(
                "  Support Vector Machine finds the optimal hyperplane that maximally "
                "separates classes. With kernel tricks (RBF, polynomial), SVM can "
                "handle non-linear decision boundaries effectively. It is robust to "
                "overfitting in high-dimensional spaces and works well even with "
                "relatively small datasets. The C parameter controls regularization, "
                "and proper feature scaling is critical for SVM performance."
            )

        rationale.append(f"  Accuracy: {row['Accuracy']:.4f} | F1: {row['F1-Score']:.4f} | "
                         f"CV: {row['CV Accuracy']:.4f} ± {row['CV Std']:.4f}")
        rationale.append("")

    rationale.append("CONCLUSION:")
    rationale.append(
        f"  Based on the comparative analysis, {best_model} achieved the highest "
        f"F1-Score ({best_f1:.4f}), making it the most effective classifier for "
        f"this animal condition dataset. The F1-Score is the primary evaluation "
        f"metric as it balances precision and recall, which is crucial in a "
        f"medical/health-related classification task where both false positives "
        f"and false negatives carry significant consequences."
    )

    rationale_text = "\n".join(rationale)
    print(rationale_text)

    # Save to file
    path = os.path.join(CHART_DIR, "rationale.txt")
    with open(path, "w") as f:
        f.write(rationale_text)
    print(f"\n[FILE] Rationale saved: {path}")

    return rationale_text


# ──────────────────────────────────────────────────────────
#  4. FULL EVALUATION PIPELINE
# ──────────────────────────────────────────────────────────
def run_evaluation(all_results: list, y_test):
    """
    Execute the full evaluation pipeline.

    Parameters:
        all_results: list of result dicts from all 4 models
        y_test: test labels

    Returns:
        dict with table, charts, rationale
    """
    print("\n" + "=" * 60)
    print("  EVALUATION & COMPARISON")
    print("=" * 60)

    # 1. Comparison table
    table = build_comparison_table(all_results)

    # 2. Charts
    plot_comparison_chart(table)
    plot_cv_comparison(table)
    plot_confusion_matrices(all_results, y_test)

    # Individual confusion matrices
    for r in all_results:
        plot_individual_confusion_matrix(r)

    # 3. Rationale
    rationale = generate_rationale(table, all_results)

    result = {
        "comparison_table": table,
        "rationale": rationale,
    }

    print("\n[SUCCESS] Evaluation complete.\n")
    return result


if __name__ == "__main__":
    print("Run from main.py to see full evaluation.")
