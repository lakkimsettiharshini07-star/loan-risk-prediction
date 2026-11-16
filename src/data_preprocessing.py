"""Data loading and splitting for the credit risk project."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

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