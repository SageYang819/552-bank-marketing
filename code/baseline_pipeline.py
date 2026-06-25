"""
baseline_pipeline.py

Reproducible baseline pipeline for the Bank Marketing project (Group 8).

What this script does:
- Loads data from: data/bank-additional-full.csv (semicolon-separated)
- Creates deployable features (drops `duration`, encodes pdays=999 via `ever_contacted`)
- Creates an upper-bound (non-deployable) feature set including `duration` (to illustrate leakage)
- Stratified train/valid/test split (60/20/20) with fixed RANDOM_STATE
- Trains:
  - Deployable Logistic Regression (balanced class weights)
  - Deployable HistGradientBoostingClassifier (HGB)
  - Upper-bound HGB (+duration)
- Evaluates:
  - ROC-AUC, PR-AUC, F1 (threshold chosen to maximize F1 on each split)
  - Top-k targeting metrics (precision/lift/captured positives) for k in {1%,2%,5%,10%,20%}
- Saves sponsor-friendly figures to outputs/figs:
  - fig_top10_precision_shortlabels.png
  - fig_pr_auc_shortlabels.png
  - fig_contact_rate_shortlabels.png

Run:
  python code/baseline_pipeline.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)

import matplotlib.pyplot as plt


RANDOM_STATE = 42


# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "bank-additional-full.csv"
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Feature engineering
# ----------------------------
def make_features(data: pd.DataFrame, include_duration: bool) -> Tuple[pd.DataFrame, pd.Series]:
    d = data.copy()

    # Target
    assert "y" in d.columns, "Target column 'y' not found."
    d["y"] = (d["y"].astype(str).str.lower() == "yes").astype(int)

    # pdays=999 means "not previously contacted"
    if "pdays" in d.columns:
        d["ever_contacted"] = (d["pdays"] != 999).astype(int)
        d.loc[d["pdays"] == 999, "pdays"] = np.nan

    # Remove duration for deployable model
    if not include_duration and "duration" in d.columns:
        d = d.drop(columns=["duration"])

    y = d.pop("y")
    return d, y


# ----------------------------
# Split
# ----------------------------
def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    valid_size: float = 0.2,
    seed: int = RANDOM_STATE,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    X_trval, X_te, y_trval, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    valid_frac_of_trval = valid_size / (1 - test_size)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_trval, y_trval, test_size=valid_frac_of_trval, stratify=y_trval, random_state=seed
    )
    return {"train": (X_tr, y_tr), "valid": (X_va, y_va), "test": (X_te, y_te)}


# ----------------------------
# Preprocess
# ----------------------------
def build_preprocess(X: pd.DataFrame) -> Tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return pre, numeric_cols, categorical_cols


# ----------------------------
# Metrics
# ----------------------------
@dataclass
class EvalResult:
    roc_auc: float
    pr_auc: float
    f1: float
    threshold: float
    topk_table: pd.DataFrame


def topk_metrics(y_true: np.ndarray, y_score: np.ndarray, ks=(0.01, 0.02, 0.05, 0.10, 0.20)) -> pd.DataFrame:
    n = len(y_true)
    base_rate = float(y_true.mean())
    order = np.argsort(-y_score)

    rows = []
    for k in ks:
        m = max(1, int(round(k * n)))
        idx = order[:m]
        y_k = y_true[idx]

        precision = float(y_k.mean())
        captured = float(y_k.sum() / max(1, y_true.sum()))
        lift = float(precision / base_rate) if base_rate > 0 else np.nan

        rows.append(
            {
                "k_pct": k,
                "n_called": m,
                "precision_in_topk": precision,
                "captured_positives_frac": captured,
                "lift_vs_base": lift,
            }
        )
    return pd.DataFrame(rows)


def choose_threshold_max_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    p, r, t = precision_recall_curve(y_true, y_score)
    f1s = (2 * p[:-1] * r[:-1]) / np.maximum(1e-12, (p[:-1] + r[:-1]))
    best_idx = int(np.nanargmax(f1s))
    return float(t[best_idx])


def evaluate_binary(y_true: np.ndarray, y_score: np.ndarray) -> EvalResult:
    roc = float(roc_auc_score(y_true, y_score))
    pr = float(average_precision_score(y_true, y_score))
    thr = choose_threshold_max_f1(y_true, y_score)
    y_pred = (y_score >= thr).astype(int)
    f1 = float(f1_score(y_true, y_pred))
    tkt = topk_metrics(y_true, y_score)
    return EvalResult(roc_auc=roc, pr_auc=pr, f1=f1, threshold=thr, topk_table=tkt)


def summarize_top10(res: EvalResult) -> tuple[float, float]:
    row = res.topk_table.loc[res.topk_table["k_pct"] == 0.10].iloc[0]
    return float(row["precision_in_topk"]), float(row["lift_vs_base"])


# ----------------------------
# Training helpers
# ----------------------------
def train_deployable_models(splits: Dict[str, Tuple[pd.DataFrame, pd.Series]]):
    X_tr, y_tr = splits["train"]
    pre, _, _ = build_preprocess(X_tr)

    logit = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    logit_pipe = Pipeline(steps=[("pre", pre), ("model", logit)])
    logit_pipe.fit(X_tr, y_tr)

    hgb = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.05, max_iter=400, random_state=RANDOM_STATE
    )
    hgb_pipe = Pipeline(steps=[("pre", pre), ("model", hgb)])
    hgb_pipe.fit(X_tr, y_tr)

    return logit_pipe, hgb_pipe


def train_upper_bound_hgb(splits: Dict[str, Tuple[pd.DataFrame, pd.Series]]):
    X_tr, y_tr = splits["train"]
    pre, _, _ = build_preprocess(X_tr)

    hgb = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.05, max_iter=400, random_state=RANDOM_STATE
    )
    hgb_pipe = Pipeline(steps=[("pre", pre), ("model", hgb)])
    hgb_pipe.fit(X_tr, y_tr)
    return hgb_pipe


# ----------------------------
# Figures
# ----------------------------
def save_fig_top10_precision(base_rate: float, rows: list[dict], out_path: Path) -> None:
    labels = [r["label"] for r in rows]
    vals = [r["top10_precision"] for r in rows]

    plt.figure()
    plt.bar(labels, vals)
    plt.axhline(base_rate, linestyle="--")
    plt.ylim(0, 0.8)
    plt.ylabel("Precision in Top-10%")
    plt.title("Top-10% Targeting Precision (TEST)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_fig_pr_auc(rows: list[dict], out_path: Path) -> None:
    labels = [r["label"] for r in rows]
    vals = [r["pr_auc"] for r in rows]

    plt.figure()
    plt.bar(labels, vals)
    plt.ylim(0, 0.8)
    plt.ylabel("PR-AUC")
    plt.title("PR-AUC Comparison (TEST)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_fig_contact_rate(df: pd.DataFrame, out_path: Path) -> None:
    g = df.copy()
    g["y"] = (g["y"].astype(str).str.lower() == "yes").astype(int)
    tbl = g.groupby("contact")["y"].mean().reset_index()
    plt.figure()
    plt.bar(tbl["contact"].tolist(), tbl["y"].tolist())
    plt.ylim(0, max(0.18, float(tbl["y"].max()) * 1.2))
    plt.ylabel("Subscription Rate (y=yes)")
    plt.title("Subscription Rate by Contact Channel")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cannot find data file at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, sep=";")
    X_deploy, y = make_features(df, include_duration=False)
    X_upper, y_upper = make_features(df, include_duration=True)
    assert y.equals(y_upper), "Target mismatch after feature engineering."

    base_rate = float(y.mean())
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols | Base rate (y=yes)={base_rate:.4f}")

    splits_deploy = stratified_split(X_deploy, y)
    splits_upper = stratified_split(X_upper, y)

    logit_pipe, hgb_pipe = train_deployable_models(splits_deploy)
    hgb_upper_pipe = train_upper_bound_hgb(splits_upper)

    def eval_pipe(pipe: Pipeline, splits: Dict[str, Tuple[pd.DataFrame, pd.Series]], split_name: str) -> EvalResult:
        X_part, y_part = splits[split_name]
        score = pipe.predict_proba(X_part)[:, 1]
        return evaluate_binary(y_part.to_numpy(), score)

    # Deployable evaluations
    res_logit_te = eval_pipe(logit_pipe, splits_deploy, "test")
    res_hgb_te = eval_pipe(hgb_pipe, splits_deploy, "test")

    # Upper-bound evaluation
    res_upper_te = eval_pipe(hgb_upper_pipe, splits_upper, "test")

    # Print compact summary
    def print_summary(name: str, res: EvalResult) -> None:
        top10_p, top10_l = summarize_top10(res)
        print(f"{name}: ROC-AUC={res.roc_auc:.4f} | PR-AUC={res.pr_auc:.4f} | F1={res.f1:.4f} | Top10_P={top10_p:.4f} | Top10_Lift={top10_l:.2f}")

    print("\n=== TEST summary ===")
    print_summary("Deployable Logit", res_logit_te)
    print_summary("Deployable HGB", res_hgb_te)
    print_summary("Upper-bound HGB (+duration)", res_upper_te)

    # Save figures (short x-axis labels)
    rows = [
        {"label": "Logit (deploy)", "pr_auc": res_logit_te.pr_auc, "top10_precision": summarize_top10(res_logit_te)[0]},
        {"label": "HGB (deploy)", "pr_auc": res_hgb_te.pr_auc, "top10_precision": summarize_top10(res_hgb_te)[0]},
        {"label": "HGB (+duration)", "pr_auc": res_upper_te.pr_auc, "top10_precision": summarize_top10(res_upper_te)[0]},
    ]

    save_fig_top10_precision(base_rate, rows, FIG_DIR / "fig_top10_precision_shortlabels.png")
    save_fig_pr_auc(rows, FIG_DIR / "fig_pr_auc_shortlabels.png")
    save_fig_contact_rate(df, FIG_DIR / "fig_contact_rate_shortlabels.png")

    print(f"\nSaved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
