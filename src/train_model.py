"""Train and compare models using cross-validation on the training set only."""

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data_preprocessing import load_salaried_data, split_data, build_preprocessor
from src.feature_engineering import build_features, get_target

# Metrics to report. zero_division=0 avoids warnings when a model never predicts 1.
SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",   # area under the precision-recall curve
    "recall": make_scorer(recall_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "accuracy": "accuracy",
}


def make_models():
    """Models to compare."""
    return {
        "Baseline (always predicts 0)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,            # allow enough steps to converge
            class_weight="balanced",  # give the rare class more weight
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,               # number of trees
            max_depth=10,                   # limit tree height (reduces overfitting)
            min_samples_leaf=100,           # each final group needs at least 100 rows
            class_weight="balanced_subsample",
            n_jobs=-1,                      # use all CPU cores
            random_state=42,                # reproducible results
        ),
    }


def compare_models(X_train, y_train):
    """Cross-validate each model; return the mean score for every metric."""
    # Stratified: every fold keeps the same share of risky applicants
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, model in make_models().items():
        # Pipeline = preprocessing + model, so each fold re-fits the
        # clipping bounds, medians and encoders on that fold's training part
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING)
        row = {"model": name}
        for metric in SCORING:
            row[metric] = scores[f"test_{metric}"].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


if __name__ == "__main__":
    # Show every column instead of hiding the middle ones with "..."
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    df = load_salaried_data()
    train_df, _ = split_data(df)       # the test set is deliberately ignored here
    X_train = build_features(train_df)
    y_train = get_target(train_df)
    print(compare_models(X_train, y_train).round(3).to_string())