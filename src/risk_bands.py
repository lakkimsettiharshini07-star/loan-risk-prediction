"""Choose risk categories and a manual-review cut-off (training data only)."""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.data_preprocessing import load_salaried_data, split_data
from src.feature_engineering import build_features, get_target
from src.evaluate_model import make_logistic, out_of_fold_scores


def cutoff_table(y, scores, review_shares):
    """For each 'flag the top X% of applicants' policy, show what it catches."""
    y = np.asarray(y)
    base_rate = y.mean()
    rows = []
    for share in review_shares:
        # Score above which the top `share` of applicants lie
        threshold = np.quantile(scores, 1 - share)
        flagged = scores >= threshold
        precision = y[flagged].mean()              # share of flagged who are risky
        recall = y[flagged].sum() / y.sum()        # share of risky who get flagged
        rows.append({
            "flag top": f"{share:.0%}",
            "score cut-off": threshold,
            "recall": recall,
            "precision": precision,
            "lift": precision / base_rate,
        })
    return pd.DataFrame(rows)


def band_table(y, scores, edges, labels):
    """Group applicants into risk categories and show each group's real default rate."""
    y = np.asarray(y)
    # pd.cut sorts each score into a category using the two edges
    bands = pd.cut(scores, bins=[-np.inf, *edges, np.inf], labels=labels)
    frame = pd.DataFrame({"band": bands, "score": scores, "target": y})
    table = frame.groupby("band", observed=True).agg(
        applicants=("target", "size"),
        mean_score=("score", "mean"),
        actual_rate=("target", "mean"),
    )
    table["share of applicants"] = table["applicants"] / len(frame)
    return table


if __name__ == "__main__":
    pd.set_option("display.width", 200)

    df = load_salaried_data()
    train_df, _ = split_data(df)          # the test set is not used here
    X_train = build_features(train_df)
    y_train = get_target(train_df)

    # Calibrated out-of-fold scores: every applicant is scored by a model
    # that never trained on them
    scores = out_of_fold_scores(
        CalibratedClassifierCV(make_logistic(), method="sigmoid", cv=5),
        X_train, y_train,
    )

    print("Cut-off options (flag the highest-scoring share of applicants):")
    print(cutoff_table(y_train, scores, [0.10, 0.20, 0.30, 0.40, 0.50]).round(3).to_string(index=False))

    # Category edges at the 50th and 80th percentile of the training scores
    lower_edge, upper_edge = np.quantile(scores, [0.50, 0.80])
    print()
    print(f"Category edges: Lower < {lower_edge:.4f} <= Moderate < {upper_edge:.4f} <= Higher")
    print(band_table(y_train, scores, [lower_edge, upper_edge],
                     ["Lower", "Moderate", "Higher"]).round(4).to_string())