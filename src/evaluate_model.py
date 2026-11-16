"""Evaluation work: fold-to-fold spread and calibration (training data only)."""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from src.data_preprocessing import load_salaried_data, split_data, build_preprocessor
from src.feature_engineering import build_features, get_target
from src.train_model import make_models


def fold_scores(X_train, y_train):
    """Return the score of every cross-validation fold for each real model."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}
    results = {}
    for name, model in make_models().items():
        if name.startswith("Baseline"):
            continue
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        results[name] = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
    return results


def make_logistic():
    """The chosen model, before calibration."""
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def out_of_fold_scores(model, X_train, y_train):
    """Score every training applicant using a model that never saw them."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
    # predict_proba returns two columns: P(class 0) and P(class 1). We keep column 1.
    return cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]


def calibration_report(X_train, y_train):
    """Compare raw scores with calibrated probabilities."""
    raw = out_of_fold_scores(make_logistic(), X_train, y_train)
    # CalibratedClassifierCV learns the score-to-frequency mapping with its own
    # inner 5-fold cross-validation, so it never calibrates on rows it trained on.
    calibrated = out_of_fold_scores(
        CalibratedClassifierCV(make_logistic(), method="sigmoid", cv=5),
        X_train, y_train,
    )

    actual_rate = float(np.mean(y_train))
    print(f"Actual share of TARGET = 1:  {actual_rate:.4f}")
    print(f"Brier score if we always predicted that share: "
          f"{actual_rate * (1 - actual_rate):.4f}")
    print()

    for label, p in [("Raw score", raw), ("Calibrated", calibrated)]:
        print(f"{label}: mean predicted {p.mean():.4f}   "
              f"Brier {brier_score_loss(y_train, p):.4f}   "
              f"ROC-AUC {roc_auc_score(y_train, p):.4f}")

    # Reliability table: 10 groups of equal size, from lowest to highest score
    frac_pos, mean_pred = calibration_curve(
        y_train, calibrated, n_bins=10, strategy="quantile"
    )
    table = pd.DataFrame({
        "mean predicted": mean_pred,
        "actual share of TARGET = 1": frac_pos,
    })
    print()
    print("Reliability table (calibrated), lowest to highest score:")
    print(table.round(4).to_string())


if __name__ == "__main__":
    df = load_salaried_data()
    train_df, _ = split_data(df)          # the test set is not used here
    X_train = build_features(train_df)
    y_train = get_target(train_df)
    calibration_report(X_train, y_train)