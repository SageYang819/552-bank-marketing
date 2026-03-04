# DATA 552 — Bank Marketing

Predict whether a client subscribes to a term deposit (`y=yes/no`) using the Portuguese bank direct marketing dataset. This repo includes an EDA notebook and a baseline training/evaluation pipeline.

## Structure

```
.
├── bank/                 # raw data (bank version)
├── bank-additional/      # raw data (bank-additional version)
├── code/
│   ├── baseline_pipeline.py
│   └── eda.ipynb
├── data/                 # intermediate data (not committed)
└── outputs/              # figures/metrics (not committed)
```

## Run

```bash
python code/baseline_pipeline.py
```

Outputs (metrics + figures) are saved under `outputs/` (e.g., `outputs/figs/`).

## EDA 

```bash
jupyter notebook code/eda.ipynb
```

Clear outputs before committing:

```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace code/eda.ipynb
```
