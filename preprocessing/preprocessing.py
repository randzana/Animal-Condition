"""
preprocessing.py — Data Loading, Cleaning & Preprocessing
==========================================================
Member 1 — Data & Preprocessing

Responsibilities:
  • Load the raw CSV from data/
  • Clean missing / malformed values
  • Encode text columns to numeric (LabelEncoder)
  • Split into features X and target y
  • Return train/test splits ready for modelling
  • Produce exploratory charts (distribution, class balance)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ──────────────────────────────────────────────────────────
#  PATHS
# ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
CHART_DIR = os.path.join(BASE_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────
#  1. LOAD DATASET
# ──────────────────────────────────────────────────────────
def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and return a DataFrame."""
    df = pd.read_csv(path)
    print(f"[INFO] Loaded dataset with shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    return df


# ──────────────────────────────────────────────────────────
#  2. CLEAN DATA
# ──────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
      - Strip whitespace from all string columns
      - Lowercase all string values for consistency
      - Fix common spelling mistakes in symptoms
      - Handle missing values
      - Remove duplicate rows
    """
    df = df.copy()

    # Strip and lowercase all object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Replace 'nan' strings with actual NaN
    df.replace("nan", np.nan, inplace=True)

    # Drop rows where target is missing
    target_col = _find_target_column(df)
    df.dropna(subset=[target_col], inplace=True)

    # Fill remaining missing values with 'unknown'
    df.fillna("unknown", inplace=True)

    # Remove exact duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    if before != after:
        print(f"[INFO] Removed {before - after} duplicate rows")

    df.reset_index(drop=True, inplace=True)
    print(f"[INFO] Cleaned dataset shape: {df.shape}")
    return df


def _find_target_column(df: pd.DataFrame) -> str:
    """Find the target column (Dangerous / dangerous)."""
    for col in df.columns:
        if col.lower() in ("dangerous", "danger"):
            return col
    raise ValueError("Target column 'Dangerous' not found in dataset")


def _find_symptom_columns(df: pd.DataFrame) -> list:
    """Find all symptom columns."""
    return [col for col in df.columns if "symptom" in col.lower() or "symp" in col.lower()]


def _find_animal_column(df: pd.DataFrame) -> str:
    """Find the animal name column."""
    for col in df.columns:
        if "animal" in col.lower() or "aniname" in col.lower() or col.lower() == "animalname":
            return col
    return None


# ──────────────────────────────────────────────────────────
#  3. ENCODE DATA
# ──────────────────────────────────────────────────────────
def encode_data(df: pd.DataFrame):
    """
    Encode all categorical columns to numeric using LabelEncoder.
    Returns:
      - df_encoded: DataFrame with numeric values
      - encoders: dict of {column_name: fitted LabelEncoder}
    """
    df_encoded = df.copy()
    encoders = {}

    for col in df_encoded.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
        print(f"[INFO] Encoded '{col}': {len(le.classes_)} unique values")

    return df_encoded, encoders


# ──────────────────────────────────────────────────────────
#  4. SPLIT FEATURES & TARGET
# ──────────────────────────────────────────────────────────
def split_features_target(df_encoded: pd.DataFrame):
    """
    Split into X (features) and y (target).
    Features = all symptom columns (and optionally animal name)
    Target   = Dangerous column
    """
    target_col = _find_target_column(df_encoded)
    feature_cols = [col for col in df_encoded.columns if col != target_col]

    X = df_encoded[feature_cols]
    y = df_encoded[target_col]

    print(f"[INFO] Features ({len(feature_cols)}): {feature_cols}")
    print(f"[INFO] Target: '{target_col}' — classes: {sorted(y.unique())}")
    return X, y, feature_cols, target_col


def get_train_test_split(X, y, test_size=0.2, random_state=42):
    """Stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Standardize features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ──────────────────────────────────────────────────────────
#  5. EXPLORATORY CHARTS
# ──────────────────────────────────────────────────────────
def plot_class_distribution(y, target_col: str):
    """Bar chart of the target class distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = pd.Series(y).value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    counts.plot(kind="bar", ax=ax, color=colors[:len(counts)], edgecolor="black")
    ax.set_title("Target Class Distribution (Dangerous)", fontsize=14, fontweight="bold")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    for i, v in enumerate(counts):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "class_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[CHART] Saved: {path}")
    return path


def plot_feature_distributions(df_encoded, feature_cols):
    """Histogram of each encoded feature."""
    n = len(feature_cols)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(feature_cols):
        axes[i].hist(df_encoded[col], bins=20, color="#3498db", edgecolor="black", alpha=0.7)
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Encoded Value")
        axes[i].set_ylabel("Frequency")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Distributions (Encoded)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "feature_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[CHART] Saved: {path}")
    return path


def plot_dataset_info(df):
    """Summary statistics table as a chart."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    stats = df.describe(include="all").T
    stats = stats[["count", "unique", "top", "freq"]].dropna(how="all")
    table = ax.table(
        cellText=stats.values,
        colLabels=stats.columns,
        rowLabels=stats.index,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.set_title("Dataset Summary Statistics", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "dataset_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[CHART] Saved: {path}")
    return path


# ──────────────────────────────────────────────────────────
#  6. FULL PIPELINE
# ──────────────────────────────────────────────────────────
def run_preprocessing_pipeline(data_path: str = DATA_PATH):
    """
    Execute the full preprocessing pipeline:
      1. Load → 2. Clean → 3. Chart (raw) → 4. Encode → 5. Split → 6. Scale
    Returns a dict with all outputs needed by downstream modules.
    """
    print("=" * 60)
    print("  PREPROCESSING PIPELINE")
    print("=" * 60)

    # 1. Load
    df_raw = load_dataset(data_path)

    # 2. Clean
    df_clean = clean_data(df_raw)

    # 3. Exploratory charts on raw data
    plot_dataset_info(df_clean)

    # 4. Encode
    df_encoded, encoders = encode_data(df_clean)

    # 5. Split features / target
    X, y, feature_cols, target_col = split_features_target(df_encoded)

    # Charts on encoded data
    plot_class_distribution(y, target_col)
    plot_feature_distributions(df_encoded, feature_cols)

    # 6. Train/test split
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    # 7. Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    result = {
        "df_raw": df_raw,
        "df_clean": df_clean,
        "df_encoded": df_encoded,
        "encoders": encoders,
        "X": X,
        "y": y,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
    }

    print("\n[SUCCESS] Preprocessing complete.\n")
    return result


# ──────────────────────────────────────────────────────────
#  CLI entry point
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_preprocessing_pipeline()
    print(f"X_train shape: {result['X_train'].shape}")
    print(f"X_test  shape: {result['X_test'].shape}")
