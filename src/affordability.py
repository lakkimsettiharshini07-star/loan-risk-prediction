"""Affordability indicators: plain arithmetic on the applicant's own figures.

These indicators do NOT use the machine-learning model. They are shown
separately from the model's risk estimate.
"""

import math

# Illustrative policy settings. They are NOT derived from data and are NOT
# regulatory limits. A lender should set its own values.
INSTALMENT_LOWER = 0.30     # below this share of monthly income: "Lower" burden
INSTALMENT_HIGHER = 0.50    # at or above this share: "Higher" burden
MIN_TERM_MONTHS = 3         # loan / instalment below this looks like a typing slip
MAX_TERM_MONTHS = 600       # above this (50 years of instalments) also looks wrong


def _positive_number(value, name):
    """Convert to float and reject missing, zero, negative or non-finite values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number.")
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return number


def burden_label(instalment_ratio):
    """Descriptive label for the instalment-to-income ratio."""
    if instalment_ratio < INSTALMENT_LOWER:
        return "Lower"
    if instalment_ratio < INSTALMENT_HIGHER:
        return "Moderate"
    return "Higher"


def affordability_indicators(monthly_income, loan_amount, monthly_instalment):
    """Return the indicators, input-sanity flags and caveats as a dictionary."""
    income = _positive_number(monthly_income, "monthly_income")
    loan = _positive_number(loan_amount, "loan_amount")
    instalment = _positive_number(monthly_instalment, "monthly_instalment")

    annual_income = income * 12
    instalment_ratio = instalment / income
    loan_ratio = loan / annual_income
    # Instalments needed just to repay the amount borrowed, ignoring interest.
    # With interest the real term is longer, so this is a lower bound.
    min_repayment_months = loan / instalment

    flags = []
    if instalment >= income:
        flags.append("The monthly instalment is at or above the monthly gross income. "
                     "Check the figures.")
    if min_repayment_months < MIN_TERM_MONTHS:
        flags.append(f"The instalment would repay the loan amount in under "
                     f"{MIN_TERM_MONTHS} months, which is unusual. Check that the "
                     f"instalment and loan amount are correct.")
    if min_repayment_months > MAX_TERM_MONTHS:
        flags.append(f"The instalment would need over {MAX_TERM_MONTHS} months to cover "
                     f"the loan amount even without interest. Check the figures.")

    return {
        "instalment_to_income": instalment_ratio,
        "loan_to_annual_income": loan_ratio,
        "loan_in_months_of_income": loan / income,
        "min_repayment_months": min_repayment_months,
        "instalment_burden": burden_label(instalment_ratio),
        "flags": flags,
        "notes": [
            "Based on gross income and this loan's instalment only. Other loans, "
            "tax and living costs are not known, so the real burden may be higher.",
            "These indicators are arithmetic on the declared figures. A good ratio "
            "does not guarantee repayment.",
            "The burden labels use illustrative thresholds, not regulatory limits.",
        ],
    }


def format_indicators(result):
    """Plain-text version for testing and reports."""
    lines = [
        f"Instalment-to-income: {result['instalment_to_income']:.1%}  "
        f"({result['instalment_burden']} burden)",
        f"Loan amount: {result['loan_to_annual_income']:.2f}x annual income "
        f"({result['loan_in_months_of_income']:.1f} months of gross income)",
        f"Instalments needed to repay the loan amount, ignoring interest: "
        f"at least {result['min_repayment_months']:.1f} months",
    ]
    lines += [f"  FLAG: {flag}" for flag in result["flags"]]
    lines += [f"  Note: {note}" for note in result["notes"]]
    return "\n".join(lines)


if __name__ == "__main__":
    # (monthly income, loan amount, monthly instalment)
    cases = [
        (30000, 150000, 5000),
        (30000, 300000, 10000),
        (25000, 400000, 14000),
        (40000, 90000, 50000),    # instalment above income
        (0, 100000, 5000),        # invalid income
    ]
    for income, loan, instalment in cases:
        print("=" * 70)
        print(f"Income {income:,}  Loan {loan:,}  Instalment {instalment:,}")
        try:
            print(format_indicators(affordability_indicators(income, loan, instalment)))
        except ValueError as error:
            print("  Rejected:", error)