"""
ui.py — Simple Prediction Interface
=====================================
Member 6 — UI

Simple command-line interface:
  • Input 5 symptoms
  • Get prediction from all 4 models
  • Display results in a formatted table
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


def display_banner():
    """Display the application banner."""
    print("\n" + "═" * 60)
    print("  🐾  ANIMAL CONDITION CLASSIFIER  🐾")
    print("  Predict if an animal's condition is dangerous")
    print("═" * 60)


def get_symptom_options(encoders, df_clean):
    """Get available symptom values from the encoders."""
    symptom_cols = [col for col in df_clean.columns
                    if "symptom" in col.lower() or "symp" in col.lower()]
    animal_col = None
    for col in df_clean.columns:
        if "animal" in col.lower() or "aniname" in col.lower():
            animal_col = col
            break

    options = {}
    if animal_col and animal_col in encoders:
        options[animal_col] = sorted(encoders[animal_col].classes_.tolist())
    for col in symptom_cols:
        if col in encoders:
            options[col] = sorted(encoders[col].classes_.tolist())

    return options


def display_options(options):
    """Display available options for each field."""
    print("\n📋 Available values for each field:")
    print("-" * 50)
    for col, values in options.items():
        print(f"\n  {col}:")
        # Show in columns
        for i, v in enumerate(values):
            if i > 0 and i % 5 == 0:
                print()
            print(f"    [{i}] {v}", end="")
        print()


def get_user_input(options, encoders, feature_cols):
    """Get symptoms from the user and encode them."""
    print("\n" + "-" * 50)
    print("Enter values (type the value or its number):")
    print("-" * 50)

    input_values = {}
    for col in feature_cols:
        if col in options:
            available = options[col]
            while True:
                user_input = input(f"\n  {col}: ").strip().lower()

                # Check if it's a number index
                try:
                    idx = int(user_input)
                    if 0 <= idx < len(available):
                        value = available[idx]
                        encoded = encoders[col].transform([value])[0]
                        input_values[col] = encoded
                        print(f"    → Selected: {value} (encoded: {encoded})")
                        break
                except ValueError:
                    pass

                # Check if it matches a value
                if user_input in available:
                    encoded = encoders[col].transform([user_input])[0]
                    input_values[col] = encoded
                    print(f"    → Encoded: {encoded}")
                    break
                else:
                    # Try fuzzy match
                    matches = [v for v in available if user_input in v]
                    if len(matches) == 1:
                        encoded = encoders[col].transform([matches[0]])[0]
                        input_values[col] = encoded
                        print(f"    → Matched: {matches[0]} (encoded: {encoded})")
                        break
                    elif len(matches) > 1:
                        print(f"    ⚠ Multiple matches: {matches}. Be more specific.")
                    else:
                        print(f"    ⚠ Unknown value. Available: {available[:10]}...")
        else:
            # Numeric column
            while True:
                user_input = input(f"\n  {col} (numeric): ").strip()
                try:
                    input_values[col] = float(user_input)
                    break
                except ValueError:
                    print("    ⚠ Please enter a number.")

    return input_values


def predict_with_all_models(input_values, models, scaler, feature_cols):
    """Run prediction through all models and display results."""
    # Create feature vector
    X_input = np.array([[input_values.get(col, 0) for col in feature_cols]])
    X_scaled = scaler.transform(X_input)

    print("\n" + "═" * 60)
    print("  📊  PREDICTION RESULTS")
    print("═" * 60)

    results = []
    for model_info in models:
        name = model_info["model_name"]
        model = model_info["model"]
        pred = model.predict(X_scaled)[0]
        label = "🔴 DANGEROUS" if pred == 1 else "🟢 NOT DANGEROUS"
        results.append({"Model": name, "Prediction": label, "Raw": pred})
        print(f"\n  {name}:")
        print(f"    Prediction: {label}")

    # Summary
    dangerous_count = sum(1 for r in results if r["Raw"] == 1)
    total = len(results)
    print("\n" + "-" * 50)
    print(f"  CONSENSUS: {dangerous_count}/{total} models predict DANGEROUS")

    if dangerous_count > total / 2:
        print("  ⚠️  MAJORITY VERDICT: DANGEROUS CONDITION")
    else:
        print("  ✅  MAJORITY VERDICT: NOT DANGEROUS")
    print("═" * 60)

    return results


def run_ui(preprocessing_result, all_model_results):
    """
    Main UI loop.

    Parameters:
        preprocessing_result: dict from run_preprocessing_pipeline()
        all_model_results: list of dicts from all model builds
    """
    encoders = preprocessing_result["encoders"]
    scaler = preprocessing_result["scaler"]
    feature_cols = preprocessing_result["feature_cols"]
    df_clean = preprocessing_result["df_clean"]

    options = get_symptom_options(encoders, df_clean)

    while True:
        display_banner()
        display_options(options)

        input_values = get_user_input(options, encoders, feature_cols)
        predict_with_all_models(input_values, all_model_results, scaler, feature_cols)

        print("\n")
        again = input("  Run another prediction? (y/n): ").strip().lower()
        if again != "y":
            print("\n  Goodbye! 🐾\n")
            break


if __name__ == "__main__":
    print("Run from main.py to use the UI with trained models.")
