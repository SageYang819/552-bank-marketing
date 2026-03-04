# DATA 552 Bank Marketing  
**Term Deposit Subscription Prediction**

This repository supports a DATA 552 project using the Portuguese bank direct marketing dataset to predict whether a client will subscribe to a term deposit (`y=yes/no`). It includes an **EDA notebook** and a reproducible **baseline modeling pipeline** (training/testing, metrics, top-k business metrics, and saved figures).

---

## Repository Structure

```
.
├── bank/                 # Raw dataset folder (Bank Marketing "bank" version)
├── bank-additional/      # Raw dataset folder (Bank Marketing "bank-additional" version)
├── code/
│   ├── baseline_pipeline.py   # Main pipeline script (train/evaluate/save figures)
│   └── eda.ipynb              # EDA notebook
├── data/                 # Intermediate/processed data (recommended: do NOT commit)
└── outputs/              # Run artifacts (metrics/figures, recommended: do NOT commit)
```

> Note: `bank/` and `bank-additional/` typically come from different variants of the UCI Bank Marketing dataset. The actual folder used depends on the paths configured in `code/baseline_pipeline.py`.

---

## Quick Start

### 1) Create an environment (recommended: conda + pip)

```bash
conda create -n bank552 python=3.11 -y
conda activate bank552

pip install -U pip
pip install numpy pandas scikit-learn matplotlib joblib jupyter
```

If your pipeline imports additional libraries (e.g., `xgboost`, `lightgbm`), install them as needed.

---

## Data Setup

### Raw data placement

Place the unzipped raw data folders at the repo root:

- `bank/` and/or `bank-additional/`

### Intermediate data & outputs

- `data/`: intermediate/processed data (optional)
- `outputs/`: generated figures, metrics, etc.

Recommended: do **not** commit `data/` or `outputs/` to keep the repo lightweight and reproducible.

---

## How to Run

From the repo root:

```bash
python code/baseline_pipeline.py
```

Typical outputs:
- Classification metrics (ROC-AUC, PR-AUC, F1, etc.)
- Business-oriented metrics (e.g., Top 10% Precision / Lift)
- Figures and artifacts saved under `outputs/` (often `outputs/figs/`)

---

## Modeling & Evaluation Conventions

- **Target**: `y` (`yes` / `no`)
- **Class imbalance**: `y=yes` is often the minority class, so **PR-AUC** and **Top-k** metrics are especially informative.
- **Deployable vs Upper-bound (if applicable)**
  - *Deployable models*: exclude leakage features (commonly `duration`)
  - *Upper-bound models*: may include `duration` as a performance ceiling (not deployable)

---

## EDA (Optional but Recommended)

Open the notebook:

```bash
jupyter notebook code/eda.ipynb
```

Before committing, it’s recommended to clear notebook outputs to reduce repo size and noisy diffs:

```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace code/eda.ipynb
```

---

## Git Recommendations (Optional)

Your `.gitignore` should typically include:

- `data/`
- `outputs/`
- `.ipynb_checkpoints/`
- `__pycache__/`

Example:

```gitignore
data/
outputs/
.ipynb_checkpoints/
__pycache__/
*.pyc
.DS_Store
```

---

## Reproducibility Notes

To make results fully reproducible for others, ensure the following are fixed/documented:

- Data file names/paths used by `baseline_pipeline.py`
- Random seeds (`random_state`)
- Train/test splitting strategy
- Output paths under `outputs/`

---

## License

Course project use. Add a license if you plan to open-source this repository.
