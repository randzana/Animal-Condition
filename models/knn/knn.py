"""
knn.py — k-Nearest Neighbors Classifier
==========================================
Member 4 — kNN

Responsibilities:
  • Build kNN using KNeighborsClassifier
  • Try different values of k
  • Tune using GridSearchCV
  • Cross-validation evaluation
  • Return accuracy, precision, recall, F1
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHART_DIR = os.path.join(BASE_DIR, "charts")


def plot_k_vs_accuracy(X_train, y_train, X_test, y_test, k_range=range(1, 21)):
    """Plot accuracy for different values of k to find the optimal k."""
    train_accs = []
    test_accs = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        train_accs.append(knn.score(X_train, y_train))
        test_accs.append(knn.score(X_test, y_test))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(k_range, train_accs, "o-", label="Train Accuracy", color="#3498db", linewidth=2)
    ax.plot(k_range, test_accs, "s-", label="Test Accuracy", color="#e74c3c", linewidth=2)
    ax.set_xlabel("k (Number of Neighbors)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("kNN: Accuracy vs. Number of Neighbors (k)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xticks(list(k_range))
    ax.grid(True, alpha=0.3)

    best_k = list(k_range)[np.argmax(test_accs)]
    ax.axvline(x=best_k, linestyle="--", color="green", alpha=0.7, label=f"Best k={best_k}")
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = os.path.join(CHART_DIR, "knn_k_vs_accuracy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[CHART] Saved: {path}")
    return best_k


def build_knn(X_train, y_train, X_test, y_test, tune=True):
    """
    Build and evaluate a kNN classifier.

    Parameters:
        X_train, y_train : Training data (scaled)
        X_test, y_test   : Test data (scaled)
        tune             : Whether to perform GridSearchCV

    Returns:
        dict with model, predictions, and metrics
    """
    print("=" * 60)
    print("  K-NEAREST NEIGHBORS (kNN)")
    print("=" * 60)

    # Plot k vs accuracy
    best_k_visual = plot_k_vs_accuracy(X_train, y_train, X_test, y_test)
    print(f"[INFO] Visual best k: {best_k_visual}")

    if tune:
        print("[INFO] Performing GridSearchCV for hyperparameter tuning...")
        param_grid = {
            "n_neighbors": list(range(1, 21)),
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan", "minkowski"],
            "p": [1, 2],
        }

        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        print(f"[INFO] Best parameters: {grid_search.best_params_}")
        print(f"[INFO] Best CV accuracy: {grid_search.best_score_:.4f}")
    else:
        best_model = KNeighborsClassifier(n_neighbors=best_k_visual, weights="distance")
        best_model.fit(X_train, y_train)

    # Cross-validation
    cv_results = cross_validate(
        best_model, X_train, y_train, cv=5,
        scoring=["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"],
        return_train_score=False,
    )

    # Predictions on test set
    y_pred = best_model.predict(X_test)

    # Metrics
    metrics = {
        "model_name": "k-Nearest Neighbors (kNN)",
        "model": best_model,
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "cv_accuracy": cv_results["test_accuracy"].mean(),
        "cv_accuracy_std": cv_results["test_accuracy"].std(),
        "cv_precision": cv_results["test_precision_weighted"].mean(),
        "cv_recall": cv_results["test_recall_weighted"].mean(),
        "cv_f1": cv_results["test_f1_weighted"].mean(),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    if tune:
        metrics["best_params"] = grid_search.best_params_
        metrics["grid_search"] = grid_search

    # Print results
    print(f"\n[RESULTS] kNN:")
    print(f"  Test Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Test Precision: {metrics['precision']:.4f}")
    print(f"  Test Recall:    {metrics['recall']:.4f}")
    print(f"  Test F1-Score:  {metrics['f1']:.4f}")
    print(f"  CV Accuracy:    {metrics['cv_accuracy']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
    print(f"\n{metrics['classification_report']}")

    return metrics


if __name__ == "__main__":
    from preprocessing.preprocessing import run_preprocessing_pipeline
    prep = run_preprocessing_pipeline()
    result = build_knn(
        prep["X_train_scaled"], prep["y_train"],
        prep["X_test_scaled"], prep["y_test"]
    )
