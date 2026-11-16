"""Data loading and splitting for the credit risk project."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import NUMERIC_FEATURES, CATEGORICAL_FEATURES

# Build the path from this file's location, so it works from any folder.
# __file__ = this file; .parent.parent goes up from src/ to the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "application_train.csv"

# Income types treated as salaried.
# Assumption: "Commercial associate" means private-sector employee
# (the dataset does not define it).
SALARIED_INCOME_TYPES = ["Working", "State servant", "Commercial associate"]


def load_salaried_data(path=RAW_DATA_PATH):
    """Read the raw CSV and keep only salaried applicants."""
    df = pd.read_csv(path)
    # isin() gives True/False per row; indexing with it keeps the True rows
    df = df[df["NAME_INCOME_TYPE"].isin(SALARIED_INCOME_TYPES)].copy()
    # reset_index renumbers rows 0, 1, 2... after the filter
    return df.reset_index(drop=True)


def split_data(df, test_size=0.2, random_state=42):
    """Split into training and test sets, keeping the TARGET share equal."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,       # 0.2 = 20% of rows go to the test set
        stratify=df["TARGET"],     # same share of 0s and 1s in both parts
        random_state=random_state, # fixed seed = same split every run
    )
    return train_df, test_df

class PercentileClipper(BaseEstimator, TransformerMixin):
    """Limit extreme values to percentile bounds learned from training data."""

    def __init__(self, lower=0.001, upper=0.999):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        # nanquantile ignores missing values when finding the bounds
        self.lower_bounds_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        # clip squeezes every value into [lower, upper]; NaN stays NaN
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor():
    """Create the (unfitted) preprocessing step shared by every model."""
    numeric_pipe = Pipeline([
        ("clip", PercentileClipper()),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])