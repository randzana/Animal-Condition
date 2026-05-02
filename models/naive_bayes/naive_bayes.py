"""
naive_bayes.py — Gaussian Naive Bayes Classifier
==================================================
Member 3 — Naive Bayes

Responsibilities:
  • Build Naive Bayes using GaussianNB
  • Tune hyperparameters using GridSearchCV
  • Cross-validation evaluation
  • Return accuracy, precision, recall, F1
"""

import os
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHART_DIR = os.path.join(BASE_DIR, "charts")


def build_naive_bayes(X_train, y_train, X_test, y_test, tune=True):
    """
    Build and evaluate a Gaussian Naive Bayes classifier.

    Parameters:
        X_train, y_train : Training data (scaled)
        X_test, y_test   : Test data (scaled)
        tune             : Whether to perform GridSearchCV

    Returns:
        dict with model, predictions, and metrics
    """
    print("=" * 60)
    print("  NAIVE BAYES (GaussianNB)")
    print("=" * 60)

    if tune:
        print("[INFO] Performing GridSearchCV for hyperparameter tuning...")
        param_grid = {
            "var_smoothing": np.logspace(-12, -1, 30),
        }

        grid_search = GridSearchCV(
            GaussianNB(),
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
        best_model = GaussianNB()
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
        "model_name": "Naive Bayes (GaussianNB)",
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
    print(f"\n[RESULTS] Naive Bayes:")
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
    result = build_naive_bayes(
        prep["X_train_scaled"], prep["y_train"],
        prep["X_test_scaled"], prep["y_test"]
    )
