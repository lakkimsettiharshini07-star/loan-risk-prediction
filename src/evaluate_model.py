"""Evaluation work: fold-to-fold spread, calibration, cut-off, final test check."""

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data_preprocessing import load_salaried_data, split_data, build_preprocessor
from src.feature_engineering import build_features, get_target
from src.train_model import make_models


def fold_scores(X_train, y_train):
    """Return the score of every cross-validation fold for each real model."""
    # Same settings as before, so every model sees exactly the same folds
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}
    results = {}
    for name, model in make_models().items():
        if name.startswith("Baseline"):
            continue                      # the baseline is not a real contender
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        results[name] = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
    return results


if __name__ == "__main__":
    df = load_salaried_data()
    train_df, _ = split_data(df)          # the test set is not used here
    X_train = build_features(train_df)
    y_train = get_target(train_df)

    results = fold_scores(X_train, y_train)
    for metric in ["roc_auc", "pr_auc"]:
        print(metric)
        for name, scores in results.items():
            values = scores[f"test_{metric}"]
            print(f"  {name:20s} folds: {np.round(values, 4)}  "
                  f"mean {values.mean():.4f}  std {values.std():.4f}")
        # Per-fold difference: positive means Random Forest was better in that fold
        diff = (results["Random Forest"][f"test_{metric}"]
                - results["Logistic Regression"][f"test_{metric}"])
        print(f"  RF minus LR per fold: {np.round(diff, 4)}  "
              f"mean {diff.mean():.4f}  std {diff.std():.4f}")
        print()