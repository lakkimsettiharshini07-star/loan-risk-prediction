"""Feature engineering: turn raw dataset columns into model inputs."""

import numpy as np
import pandas as pd

# Numeric features the model uses
NUMERIC_FEATURES = ["employment_years", "loan_to_income", "annuity_to_income"]

# Categorical (text) features the model uses
CATEGORICAL_FEATURES = [
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "OCCUPATION_TYPE",
]

# Placeholder the dataset uses in DAYS_EMPLOYED for people without a job
DAYS_EMPLOYED_PLACEHOLDER = 365243


def build_features(df):
    """Return a table with only the model's input columns."""
    out = pd.DataFrame(index=df.index)

    # DAYS_EMPLOYED is negative days before the application.
    # Replace the placeholder with NaN, then convert days to years.
    days = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_PLACEHOLDER, np.nan)
    out["employment_years"] = -days / 365.25

    # Ratios are scale-free, so they do not depend on the currency
    out["loan_to_income"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    out["annuity_to_income"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]

    # Dividing by zero income gives infinity; treat it as missing instead
    out = out.replace([np.inf, -np.inf], np.nan)

    # Copy the categorical columns unchanged
    for col in CATEGORICAL_FEATURES:
        out[col] = df[col]

    return out


def get_target(df):
    """Return the TARGET column (1 = recorded payment difficulty)."""
    return df["TARGET"]