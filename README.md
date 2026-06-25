# Bank Marketing Project (Group 8)

## Overview

A reproducible machine learning pipeline for predicting customer subscription to bank term deposits based on the **Bank Marketing Dataset**. This project implements a complete data science workflow including feature engineering, model training, and comprehensive evaluation metrics.

## Dataset

**Source:** UCI Machine Learning Repository - Bank Marketing Dataset  
**File:** `data/bank-additional-full.csv` (semicolon-separated)  
**Size:** 41,188 records × 21 features  
**Target:** Subscription rate (y=yes): 11.27%

### Features
- Client demographics (age, job, marital status, education)
- Account information (balance, previous contact history)
- Campaign details (contact duration, number of contacts, pdays)
- Last contact information (month, day of week, duration)
- Economic indicators (employment variation rate, consumer price index, euribor rate)

## Project Structure

```
bank+marketing/
├── data/
│   ├── bank-additional-full.csv         # Complete dataset with all features
│   └── bank-additional-names.txt        # Feature descriptions
├── code/
│   └── baseline_pipeline.py             # Main training and evaluation pipeline
├── outputs/
│   └── figs/                            # Generated visualizations
│       ├── fig_top10_precision_shortlabels.png
│       ├── fig_pr_auc_shortlabels.png
│       └── fig_contact_rate_shortlabels.png
├── bank/                                # Backup data
├── README.md                            # This file
└── LICENSE
```

## Methodology

### 1. Feature Engineering

The pipeline creates two feature sets:

- **Deployable Features:** Removes `duration` to avoid data leakage
  - Handles missing pdays values (999 = not previously contacted)
  - Creates binary `ever_contacted` indicator
  - Encodes target variable as binary (yes=1, no=0)

- **Upper-Bound Features:** Includes `duration` for performance ceiling comparison
  - Shows maximum achievable performance with full information
  - Demonstrates impact of data leakage

### 2. Data Splitting

Stratified train/validation/test split maintaining class distribution:
- **Training:** 60% (24,713 samples)
- **Validation:** 20% (8,237 samples)
- **Test:** 20% (8,237 samples)
- `RANDOM_STATE=42` for reproducibility

### 3. Preprocessing

**Numeric Features:**
- Imputation: median strategy
- Scaling: StandardScaler

**Categorical Features:**
- Imputation: most frequent strategy
- Encoding: OneHotEncoder (handle_unknown='ignore')

### 4. Models Trained

1. **Logistic Regression (Deployable)**
   - `max_iter=2000`, `solver='lbfgs'`
   - `class_weight='balanced'` to handle class imbalance

2. **HistGradientBoostingClassifier (Deployable)**
   - `max_depth=6`, `learning_rate=0.05`
   - `max_iter=400`, `random_state=42`

3. **HistGradientBoostingClassifier + Duration (Upper-Bound)**
   - Same parameters with `duration` feature included

### 5. Evaluation Metrics

- **ROC-AUC:** Area under receiver operating characteristic curve
- **PR-AUC:** Area under precision-recall curve
- **F1 Score:** Threshold chosen to maximize F1 on validation set
- **Top-k Metrics:** For k ∈ {1%, 2%, 5%, 10%, 20%}
  - Precision in top-k percentile
  - Fraction of true positives captured
  - Lift vs. base rate

## Results (Test Set)

| Model | ROC-AUC | PR-AUC | F1 | Top10% Precision | Lift (Top10%) |
|-------|---------|--------|-----|------------------|---------------|
| **Logistic Regression** | 0.8012 | 0.4597 | 0.5102 | 0.5097 | 4.52 |
| **HGB (Deployable)** | 0.8108 | 0.4793 | 0.5309 | 0.5267 | 4.68 |
| **HGB + Duration** | 0.9528 | 0.6977 | 0.6718 | 0.6735 | 5.98 |

### Key Insights

1. **Data Leakage Impact:** Including `duration` increases ROC-AUC from 0.81 to 0.95, demonstrating significant performance boost from leaked information.

2. **Model Comparison:** HistGradientBoosting outperforms Logistic Regression in deployable setting (ROC-AUC: 0.8108 vs 0.8012).

3. **Targeting Efficiency:** In top 10% of ranked customers:
   - Deployable HGB achieves 52.67% subscription rate (vs 11.27% base rate)
   - 4.68x lift vs. random targeting
   - Upper-bound HGB reaches 67.35% with 5.98x lift

## Running the Pipeline

### Prerequisites

```bash
pip install pandas scikit-learn matplotlib numpy
```

### Execution

```bash
python code/baseline_pipeline.py
```

### Output

- Console: Summary metrics for all models on test set
- Figures: Three PNG visualizations saved to `outputs/figs/`
  - Top-10% precision comparison
  - PR-AUC comparison
  - Subscription rate by contact channel

## Files Generated

After running the pipeline, the following outputs are created:

- `outputs/figs/fig_top10_precision_shortlabels.png` - Top-10% targeting precision
- `outputs/figs/fig_pr_auc_shortlabels.png` - Precision-Recall AUC comparison
- `outputs/figs/fig_contact_rate_shortlabels.png` - Subscription rate by contact method

## Authors

Group 8

## License

MIT License - See LICENSE file for details

## References

- [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing) - UCI Machine Learning Repository
- Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing.

## Last Updated

June 24, 2026
