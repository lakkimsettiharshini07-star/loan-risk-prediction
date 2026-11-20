"""Turn application form details into a risk assessment using the saved model."""

import math

import joblib
import pandas as pd

from src.data_preprocessing import PROJECT_ROOT
from src.feature_engineering import build_features

MODEL_PATH = PROJECT_ROOT / "models" / "risk_model.joblib"

# Allowed values come from the categories seen in the training data
CONTRACT_TYPES = ["Cash loans", "Revolving loans"]
INCOME_TYPES = ["Working", "State servant", "Commercial associate"]
EDUCATION_TYPES = ["Lower secondary", "Secondary / secondary special",
                   "Incomplete higher", "Higher education", "Academic degree"]
OCCUPATIONS = [
    "Accountants", "Cleaning staff", "Cooking staff", "Core staff", "Drivers",
    "HR staff", "High skill tech staff", "IT staff", "Laborers", "Low-skill Laborers",
    "Managers", "Medicine staff", "Private service staff", "Realty agents",
    "Sales staff", "Secretaries", "Security staff", "Waiters/barmen staff",
]

# How monthly income from the form is mapped onto the dataset's income column.
# The dataset does not state its time basis. Using monthly income as typed keeps
# applicants' ratios inside the range seen in training, and the model's output
# changed little under the alternative mapping (see README).
INCOME_BASIS_DEFAULT = "same_basis"


def load_model(path=MODEL_PATH):
    """Load the saved bundle: pipeline, version, category edges and base rate."""
    return joblib.load(path)


def validate_application(app):
    """Raise ValueError with a clear message if an input is unusable."""
    for key in ["monthly_gross_income", "loan_amount", "monthly_instalment"]:
        value = app.get(key)
        if value is None or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{key} must be a number greater than zero.")
    years = app.get("employment_years")
    if years is None or not math.isfinite(float(years)) or not 0 <= float(years) <= 60:
        raise ValueError("employment_years must be between 0 and 60.")
    if app.get("loan_type") not in CONTRACT_TYPES:
        raise ValueError(f"loan_type must be one of {CONTRACT_TYPES}.")
    if app.get("employment_type") not in INCOME_TYPES:
        raise ValueError(f"employment_type must be one of {INCOME_TYPES}.")
    if app.get("education") not in EDUCATION_TYPES:
        raise ValueError(f"education must be one of {EDUCATION_TYPES}.")
    occupation = app.get("occupation")
    if occupation not in (None, "", "Unknown") and occupation not in OCCUPATIONS:
        raise ValueError("occupation is not one of the known categories.")


def to_model_row(app, income_basis=INCOME_BASIS_DEFAULT):
    """Map form fields onto the raw dataset columns, as a one-row table."""
    monthly_income = float(app["monthly_gross_income"])
    income = monthly_income * 12 if income_basis == "annual" else monthly_income
    return pd.DataFrame([{
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": float(app["loan_amount"]),
        "AMT_ANNUITY": float(app["monthly_instalment"]),
        # the dataset stores employment as NEGATIVE days before the application
        "DAYS_EMPLOYED": -float(app["employment_years"]) * 365.25,
        "NAME_CONTRACT_TYPE": app["loan_type"],
        "NAME_INCOME_TYPE": app["employment_type"],
        "NAME_EDUCATION_TYPE": app["education"],
        # training filled missing occupations with "Unknown", so we do the same
        "OCCUPATION_TYPE": app.get("occupation") or "Unknown",
    }])


def assess_application(app, bundle=None, income_basis=INCOME_BASIS_DEFAULT):
    """Return the calibrated probability, risk category and review flag."""
    validate_application(app)
    bundle = bundle or load_model()
    # The SAME feature code that was used in training, so training and the
    # app cannot drift apart
    features = build_features(to_model_row(app, income_basis))
    probability = float(bundle["pipeline"].predict_proba(features)[0, 1])

    if probability >= bundle["upper_edge"]:
        category = "Higher"
    elif probability >= bundle["lower_edge"]:
        category = "Moderate"
    else:
        category = "Lower"

    return {
        "probability": probability,
        "category": category,
        "needs_manual_review": probability >= bundle["upper_edge"],
        "model_version": bundle["model_version"],
        "base_rate": bundle["base_rate"],
        "income_basis": income_basis,
        "features": features.iloc[0].to_dict(),
    }


if __name__ == "__main__":
    # Sensitivity check: the same applicants under the two income mappings
    pd.set_option("display.width", 200)
    bundle = load_model()
    base = dict(loan_type="Cash loans", employment_type="Working")
    profiles = [
        dict(base, monthly_gross_income=30000, loan_amount=150000, monthly_instalment=5000,
             occupation="Laborers", education="Secondary / secondary special",
             employment_years=3),
        dict(base, monthly_gross_income=30000, loan_amount=300000, monthly_instalment=10000,
             occupation="Laborers", education="Secondary / secondary special",
             employment_years=3),
        dict(base, monthly_gross_income=60000, loan_amount=200000, monthly_instalment=7000,
             employment_type="Commercial associate", occupation="Sales staff",
             education="Higher education", employment_years=6),
        dict(base, monthly_gross_income=45000, loan_amount=500000, monthly_instalment=15000,
             employment_type="State servant", occupation="Core staff",
             education="Higher education", employment_years=10),
        dict(base, monthly_gross_income=25000, loan_amount=400000, monthly_instalment=14000,
             occupation="Drivers", education="Lower secondary", employment_years=1),
    ]
    rows = []
    for number, profile in enumerate(profiles, start=1):
        for basis in ["annual", "same_basis"]:
            result = assess_application(profile, bundle, basis)
            rows.append({
                "profile": number, "basis": basis,
                "loan_to_income": result["features"]["loan_to_income"],
                "annuity_to_income": result["features"]["annuity_to_income"],
                "probability": result["probability"],
                "category": result["category"],
            })
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print()
    print("Category edges:", round(bundle["lower_edge"], 4), round(bundle["upper_edge"], 4))