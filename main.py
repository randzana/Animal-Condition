"""
main.py — Project Orchestrator
================================
Runs the full Animal Condition Classification pipeline:
  1. Preprocessing (load, clean, encode, split, scale)
  2. Feature Analysis (importance, correlation, selection)
  3. Model Training (Neural Network, kNN, Naive Bayes, SVM)
  4. Evaluation (comparison table, confusion matrices, rationale)
  5. UI (optional — interactive prediction)

Usage:
    python3 main.py          # Full pipeline
    python3 main.py --ui     # Full pipeline + interactive UI
    python3 main.py --skip   # Skip tuning (faster, uses defaults)
"""

import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────────────────
from preprocessing.preprocessing import run_preprocessing_pipeline
from features.features import run_feature_analysis
from models.neural_network.neural_network import build_neural_network
from models.knn.knn import build_knn
from models.naive_bayes.naive_bayes import build_naive_bayes
from models.svm.svm import build_svm
from evaluation.evaluation import run_evaluation
from ui.ui import run_ui


def main(use_ui=False, skip_tuning=False):
    """
    Execute the full pipeline.

    Parameters:
        use_ui       : Launch interactive UI after training
        skip_tuning  : Skip GridSearchCV (faster execution)
    """
    tune = not skip_tuning

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + "  ANIMAL CONDITION CLASSIFICATION PROJECT".center(58) + "║")
    print("║" + "  Neural Network | kNN | Naive Bayes | SVM".center(58) + "║")
    print("╚" + "═" * 58 + "╝\n")

    # ──────────────────────────────────────────────────────
    #  STEP 1: PREPROCESSING
    # ──────────────────────────────────────────────────────
    prep = run_preprocessing_pipeline()

    # ──────────────────────────────────────────────────────
    #  STEP 2: FEATURE ANALYSIS
    # ──────────────────────────────────────────────────────
    features = run_feature_analysis(prep)

    # ──────────────────────────────────────────────────────
    #  STEP 3: TRAIN ALL MODELS
    # ──────────────────────────────────────────────────────
    X_train = prep["X_train_scaled"]
    X_test = prep["X_test_scaled"]
    y_train = prep["y_train"]
    y_test = prep["y_test"]

    print("\n" + "=" * 60)
    print("  TRAINING ALL MODELS")
    print("=" * 60 + "\n")

    # 3a. Neural Network
    nn_result = build_neural_network(X_train, y_train, X_test, y_test, tune=tune)

    # 3b. k-Nearest Neighbors
    knn_result = build_knn(X_train, y_train, X_test, y_test, tune=tune)

    # 3c. Naive Bayes
    nb_result = build_naive_bayes(X_train, y_train, X_test, y_test, tune=tune)

    # 3d. Support Vector Machine
    svm_result = build_svm(X_train, y_train, X_test, y_test, tune=tune)

    all_results = [nn_result, knn_result, nb_result, svm_result]

    # ──────────────────────────────────────────────────────
    #  STEP 4: EVALUATION
    # ──────────────────────────────────────────────────────
    eval_result = run_evaluation(all_results, y_test)

    # ──────────────────────────────────────────────────────
    #  STEP 5: SUMMARY
    # ──────────────────────────────────────────────────────
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + "  PIPELINE COMPLETE".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n  📁 Charts saved in: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'charts')}")
    print(f"  📊 Models trained: {len(all_results)}")
    print(f"  🏆 Best model: {eval_result['comparison_table'].iloc[0]['Model']}")
    print(f"     F1-Score: {eval_result['comparison_table'].iloc[0]['F1-Score']:.4f}")
    print()

    # ──────────────────────────────────────────────────────
    #  STEP 6: UI (optional)
    # ──────────────────────────────────────────────────────
    if use_ui:
        run_ui(prep, all_results)

    return {
        "preprocessing": prep,
        "features": features,
        "models": all_results,
        "evaluation": eval_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animal Condition Classification")
    parser.add_argument("--ui", action="store_true", help="Launch interactive UI after training")
    parser.add_argument("--skip", action="store_true", help="Skip GridSearchCV (faster)")
    args = parser.parse_args()

    main(use_ui=args.ui, skip_tuning=args.skip)
