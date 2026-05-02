# 🐾 Animal Condition Classification

Predict whether an animal's condition is dangerous or not using symptom data and classical / neural ML models.

This repository contains a full end-to-end pipeline: data preprocessing, feature analysis, training and hyperparameter tuning for four classifiers (Neural Network, k-Nearest Neighbors, Naive Bayes, SVM), evaluation and visualization, plus an optional interactive UI.

---

## Table of contents

- [Highlights](#highlights)
- [Dataset](#dataset)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [What the pipeline produces](#what-the-pipeline-produces)
- [Notebooks](#notebooks)
- [Notes & next steps](#notes--next-steps)
- [Contributing](#contributing)
- [License](#license)

---

## Highlights

- Implements preprocessing (cleaning, encoding, scaling, train/test split)
- Feature analysis (correlations, RandomForest importances, subset comparisons)
- Trains and tunes:
  - Neural Network (MLP)
  - k-Nearest Neighbors (kNN)
  - Gaussian Naive Bayes
  - Support Vector Machine (SVM)
- Produces comparison tables, confusion matrices and charts for model comparison
- Optional UI for interactive predictions

---

## Dataset

This project uses the "Animal Condition Classification" dataset (Kaggle):  
https://www.kaggle.com/datasets/willianoliveiragibin/animal-condition

Place the dataset CSV at:
- `data/dataset.csv`

(If you downloaded a different filename, either rename it or update code that reads `data/dataset.csv`.)

---

## Repository structure

- `.gitignore`
- `main.py` — project orchestrator that runs the full pipeline
- `generate_notebook.py` — programmatically generates `notebook.ipynb`
- `notebook.ipynb` — interactive notebook for exploration and model training
- `charts/` — (output) saved figures and charts
- `data/` — dataset should be placed here (`data/dataset.csv`)
- `evaluation/` — (output) evaluation artifacts and results
- `features/` — feature analysis code
- `models/` — saved model objects / model code
- `preprocessing/` — preprocessing pipeline code
- `ui/` — optional interactive UI code

---

## Requirements

Tested with Python 3.8+. The repository uses (at least) the following packages:

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- notebook (optional)
- nbformat (for `generate_notebook.py`)

You can install them with pip:

pip install numpy pandas scikit-learn matplotlib seaborn notebook nbformat

(If you prefer a pinned list, I can generate a `requirements.txt`.)

---

## Installation

1. Clone the repository:
   - git clone https://github.com/randzana/Animal-Condition.git
2. Create and activate a virtual environment (recommended):
   - python3 -m venv .venv
   - source .venv/bin/activate  (macOS / Linux)
   - .\.venv\Scripts\activate   (Windows)
3. Install dependencies:
   - pip install -r requirements.txt
   - OR: pip install numpy pandas scikit-learn matplotlib seaborn notebook nbformat

Note: `requirements.txt` is not included in the repo. I can add one if you’d like.

---

## Usage

Run the full pipeline from the command line:

- Full pipeline (preprocessing, feature analysis, train, evaluate):
  - python3 main.py

- Launch pipeline and run the optional interactive UI after training:
  - python3 main.py --ui

- Skip expensive hyperparameter tuning (GridSearchCV) and use defaults (faster):
  - python3 main.py --skip

Generate the Jupyter notebook programmatically (creates `notebook.ipynb`):
- python3 generate_notebook.py

Open the provided notebook to reproduce exploratory analysis and model training:
- jupyter notebook notebook.ipynb

---

## What the pipeline produces

- Charts and figures: saved to `charts/`
- Trained model objects and artifacts: stored under `models/` (implementation-dependent)
- Evaluation results and comparison tables: `evaluation/`
- Terminal summary after run (best model, F1-score, charts location)

---

## Notebooks

- `notebook.ipynb` — step-by-step exploration, preprocessing, feature importance, model training and comparative analysis. You can either open this directly or regenerate it with `generate_notebook.py`.

---

## Notes & next steps

- The pipeline expects the dataset at `data/dataset.csv`. If you don't have it, download from the Kaggle link above.
- Consider adding:
  - `requirements.txt` with pinned versions
  - `LICENSE` file (none currently in repo)
  - CI workflow that runs basic linting / tests (optional)
  - A small sample dataset (or instructions) to allow quick smoke tests without the full Kaggle download
  - Model serialization (e.g., via joblib) into `models/` with versioning

---

## Contributing

Contributions, bug reports and feature requests are welcome. Open an issue or send a PR with a clear description of changes.

---

## License

No license file is included in the repository. If you’d like, I can add an appropriate license (e.g., MIT) — tell me which license you prefer.

---

## Contact

Repository owner: randzana

If you want, I can:
- Add a requirements.txt and commit the README to the repo
- Create a LICENSE file
- Create convenient scripts (Makefile or small CLI) to simplify common tasks
