"""Explain an assessment: which factors moved the estimate, and by how much.

Method: for each factor, ask the full saved model "what would the estimated
probability be if this one factor were average, with everything else as entered?"
The difference is that factor's effect, in probability points. Only the public
predict_proba of the saved model is used.
"""

import numpy as np

from src.feature_engineering import build_features
from src.predict import (
    INCOME_BASIS_DEFAULT, load_model, to_model_row, validate_application,
)

# (start of the transformed column names, label shown to the lender)
FACTORS = [
    ("num__employment_years", "Employment duration"),
    ("num__loan_to_income", "Loan amount relative to income"),
    ("num__annuity_to_income", "Instalment relative to income"),
    ("cat__NAME_CONTRACT_TYPE_", "Loan type"),
    ("cat__NAME_INCOME_TYPE_", "Employment type"),
    ("cat__NAME_EDUCATION_TYPE_", "Education"),
    ("cat__OCCUPATION_TYPE_", "Occupation"),
]

NEGLIGIBLE = 0.002   # effects under 0.2 percentage points are called negligible


def _value_text(label, row, income_basis):
    """The applicant's value for a factor, in words."""
    if label == "Employment duration":
        return f"{row['employment_years']:.1f} years"
    if label == "Loan amount relative to income":
        if income_basis == "same_basis":
            return f"{row['loan_to_income']:.1f}x monthly income"
        return f"ratio {row['loan_to_income']:.2f}"
    if label == "Instalment relative to income":
        if income_basis == "same_basis":
            return f"{row['annuity_to_income']:.0%} of monthly income"
        return f"ratio {row['annuity_to_income']:.3f}"
    column = {"Loan type": "NAME_CONTRACT_TYPE", "Employment type": "NAME_INCOME_TYPE",
              "Education": "NAME_EDUCATION_TYPE", "Occupation": "OCCUPATION_TYPE"}[label]
    return str(row[column])


def explain_assessment(app, bundle=None, income_basis=INCOME_BASIS_DEFAULT):
    """Return the effect of each factor on the estimated probability."""
    validate_application(app)
    bundle = bundle or load_model()
    if "reference_means" not in bundle:
        raise ValueError("The saved model has no reference means. Re-run finalize_model.")

    pipe = bundle["pipeline"]
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    names = list(bundle["feature_names"])
    means = np.asarray(bundle["reference_means"], dtype=float)

    # Every transformed column must belong to exactly one factor
    groups = [[i for i, n in enumerate(names) if n.startswith(prefix)]
              for prefix, _ in FACTORS]
    if sorted(i for g in groups for i in g) != list(range(len(names))):
        raise ValueError("The factor groups do not cover the model's columns exactly once.")

    features = build_features(to_model_row(app, income_basis))
    x = np.asarray(prep.transform(features), dtype=float)

    probability = float(model.predict_proba(x)[0, 1])
    reference = float(model.predict_proba(means.reshape(1, -1))[0, 1])

    row = features.iloc[0]
    factors = []
    for (prefix, label), idx in zip(FACTORS, groups):
        x_average = x.copy()
        x_average[0, idx] = means[idx]       # set this one factor to the average
        effect = probability - float(model.predict_proba(x_average)[0, 1])
        if effect > NEGLIGIBLE:
            direction = "raises"
        elif effect < -NEGLIGIBLE:
            direction = "lowers"
        else:
            direction = "negligible"
        factors.append({"factor": label, "value": _value_text(label, row, income_basis),
                        "effect": effect, "direction": direction})
    factors.sort(key=lambda f: abs(f["effect"]), reverse=True)

    return {
        "probability": probability,
        "reference_probability": reference,
        "factors": factors,
        "notes": [
            "Each effect is the change in the estimated probability if that one factor "
            "were average, with the other factors as entered. The effects do not add up "
            "exactly to the difference from the reference.",
            "The reference is an average of the training applicants, not a real applicant.",
            "These factors describe patterns in historical Home Credit data, not causes. "
            "Education and occupation can act as proxies for background, so a lender "
            "should review how they are used.",
        ],
    }


def format_explanation(result):
    """Plain-text version for testing and reports."""
    lines = [f"Estimated probability: {result['probability']:.1%}   "
             f"(average training profile: {result['reference_probability']:.1%})"]
    for f in result["factors"]:
        if f["direction"] == "negligible":
            change = "about the same as average"
        else:
            change = f"{f['direction']} the estimate by about {abs(f['effect']) * 100:.1f} points"
        lines.append(f"  {f['factor']}: {f['value']} -> {change}")
    lines += [f"  Note: {note}" for note in result["notes"]]
    return "\n".join(lines)


if __name__ == "__main__":
    from src.predict import assess_application

    bundle = load_model()
    base = dict(loan_type="Cash loans", employment_type="Working")
    profiles = [
        dict(base, monthly_gross_income=30000, loan_amount=150000, monthly_instalment=5000,
             occupation="Laborers", education="Secondary / secondary special",
             employment_years=3),
        dict(base, monthly_gross_income=60000, loan_amount=200000, monthly_instalment=7000,
             employment_type="Commercial associate", occupation="Sales staff",
             education="Higher education", employment_years=6),
        dict(base, monthly_gross_income=25000, loan_amount=400000, monthly_instalment=14000,
             occupation="Drivers", education="Lower secondary", employment_years=1),
    ]
    for number, profile in enumerate(profiles, start=1):
        print("=" * 70)
        print("Profile", number)
        result = explain_assessment(profile, bundle)
        # The two code paths must give the same probability
        check = assess_application(profile, bundle)["probability"]
        print(f"Check against assess_application: {check:.6f} vs {result['probability']:.6f}")
        print(format_explanation(result))