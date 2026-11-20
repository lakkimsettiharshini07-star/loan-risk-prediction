"""Fit the final model on the training set, evaluate ONCE on the test set, and save it."""

from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.data_preprocessing import (
    PROJECT_ROOT, load_salaried_data, split_data, build_preprocessor,
)
from src.feature_engineering import (
    build_features, get_target, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)
from src.evaluate_model import make_logistic, out_of_fold_scores
from src.risk_bands import band_table

MODEL_PATH = PROJECT_ROOT / "models" / "risk_model.joblib"
MODEL_VERSION = "lr-calibrated-v1"


def make_final_pipeline():
    """Preprocessing + Logistic Regression + sigmoid calibration."""
    return Pipeline([
        ("prep", build_preprocessor()),
        ("model", CalibratedClassifierCV(make_logistic(), method="sigmoid", cv=5)),
    ])


if __name__ == "__main__":
    pd.set_option("display.width", 200)

    df = load_salaried_data()
    train_df, test_df = split_data(df)
    X_train, y_train = build_features(train_df), get_target(train_df)
    X_test, y_test = build_features(test_df), np.asarray(get_target(test_df))

    # Category edges come from out-of-fold TRAINING scores only
    oof = out_of_fold_scores(
        CalibratedClassifierCV(make_logistic(), method="sigmoid", cv=5),
        X_train, y_train,
    )
    lower_edge, upper_edge = np.quantile(oof, [0.50, 0.80])

    # Fit the final pipeline on all training rows
    pipe = make_final_pipeline()
    pipe.fit(X_train, y_train)

    # The one and only look at the test set
    probs = pipe.predict_proba(X_test)[:, 1]
    base_rate = y_test.mean()

    print("TEST SET RESULTS")
    print(f"Applicants: {len(y_test)}   actual share of TARGET = 1: {base_rate:.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_test, probs):.4f}")
    print(f"Brier:   {brier_score_loss(y_test, probs):.4f}   "
          f"(always predicting the base rate: {base_rate * (1 - base_rate):.4f})")
    print(f"Mean predicted: {probs.mean():.4f}")

    # Manual-review rule: score at or above the upper edge ("Higher" category)
    flagged = probs >= upper_edge
    tn, fp, fn, tp = confusion_matrix(y_test, flagged).ravel()
    print()
    print(f"Review rule: flag score >= {upper_edge:.4f}   flagged share: {flagged.mean():.1%}")
    print(f"Confusion matrix -> true negatives {tn}, false positives {fp}, "
          f"false negatives {fn}, true positives {tp}")
    print(f"Recall: {tp / (tp + fn):.3f}   Precision: {tp / (tp + fp):.3f}   "
          f"Lift: {tp / (tp + fp) / base_rate:.2f}")

    print()
    print(band_table(y_test, probs, [lower_edge, upper_edge],
                     ["Lower", "Moderate", "Higher"]).round(4).to_string())

    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
    print()
    print(pd.DataFrame({"mean predicted": mean_pred,
                        "actual share": frac_pos}).round(4).to_string())

        # Reference point for explanations: the average transformed training applicant
    prep = pipe.named_steps["prep"]
    reference_means = prep.transform(X_train).mean(axis=0)

    # Save the pipeline together with everything the app needs to use it correctly
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({
        "pipeline": pipe,
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "lower_edge": float(lower_edge),
        "upper_edge": float(upper_edge),
        "base_rate": float(y_train.mean()),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "n_train": len(y_train),
        "reference_means": [float(v) for v in reference_means],
        "feature_names": [str(n) for n in prep.get_feature_names_out()],
    }, MODEL_PATH)
    print()
    print("Saved:", MODEL_PATH)