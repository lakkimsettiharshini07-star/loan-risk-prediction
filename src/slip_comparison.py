"""Compare salary slip fields with the applicant's declared details."""

import re
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

from src.document_verification import read_document_text
from src.field_extraction import extract_fields

CONSISTENT = "Consistent"
NEEDS_REVIEW = "Needs Review"
MISMATCH = "Mismatch"
_SEVERITY = {CONSISTENT: 0, NEEDS_REVIEW: 1, MISMATCH: 2}

# Policy settings. These are assumptions a lender can change; they are not
# learned from data and have not been validated on real salary slips.
NAME_MATCH = 0.92            # similarity at or above this: names agree
NAME_NEAR = 0.70             # below this: names clearly differ
EMPLOYER_MATCH = 0.90
EMPLOYER_NEAR = 0.70
SALARY_TOLERANCE = 0.01      # up to 1% difference counts as a match
SALARY_NEAR = 0.05           # 1% to 5%: needs review; above 5%: mismatch
ARITHMETIC_TOLERANCE = 1.0   # rupees allowed in "gross - deductions = net"
MAX_SLIP_AGE_MONTHS = 3

# Words ignored when comparing employer names
LEGAL_SUFFIXES = {"pvt", "private", "ltd", "limited", "llp", "inc",
                  "corp", "corporation", "co", "company"}


def _tokens(text):
    """Lowercase words and numbers, with punctuation removed."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _token_similarity(a, b):
    """Similarity of two words from 0 to 1. A single letter counts as an initial."""
    if len(a) == 1 or len(b) == 1:
        return 1.0 if a[0] == b[0] else 0.0
    return SequenceMatcher(None, a, b).ratio()


def compare_names(declared, extracted):
    """Return (status, message) for the applicant name."""
    d_tokens, e_tokens = _tokens(declared or ""), _tokens(extracted or "")
    if not d_tokens:
        return NEEDS_REVIEW, "No applicant name was declared."
    if not e_tokens:
        return NEEDS_REVIEW, "No readable employee name was found on the slip."

    best_scores, used = [], set()
    for d in d_tokens:
        # Find the extracted word that resembles this declared word most
        score, index = max((_token_similarity(d, e), i) for i, e in enumerate(e_tokens))
        best_scores.append(score)
        used.add(index)
    similarity = sum(best_scores) / len(best_scores)
    extra = len(e_tokens) - len(used)       # extracted words with no declared partner

    if similarity >= NAME_MATCH:
        if extra <= 1:
            return CONSISTENT, f"Names agree (similarity {similarity:.2f})."
        return NEEDS_REVIEW, (f"The names agree (similarity {similarity:.2f}), but the slip "
                              f"text has {extra} extra words. This may be OCR noise.")
    if similarity >= NAME_NEAR:
        return NEEDS_REVIEW, (f"The names are similar but not the same "
                              f"(similarity {similarity:.2f}). This could be an OCR "
                              f"error, a spelling variation, or a different person.")
    return MISMATCH, f"The names clearly differ (similarity {similarity:.2f})."


def _employer_key(text):
    """Employer name without legal words such as Pvt or Ltd."""
    words = _tokens(text)
    kept = [w for w in words if w not in LEGAL_SUFFIXES]
    return " ".join(kept or words)


def compare_employers(declared, extracted, source):
    """Return (status, message) for the employer name."""
    d_key, e_key = _employer_key(declared or ""), _employer_key(extracted or "")
    if not d_key:
        return NEEDS_REVIEW, "No employer was declared."
    if not e_key:
        return NEEDS_REVIEW, "No employer name could be read from the slip."

    similarity = SequenceMatcher(None, d_key, e_key).ratio()
    guessed = source == "guessed from header"
    if similarity >= EMPLOYER_MATCH:
        where = " The name was taken from the slip header, not a labelled field." if guessed else ""
        return CONSISTENT, f"Employer names agree (similarity {similarity:.2f}).{where}"
    if similarity >= EMPLOYER_NEAR:
        return NEEDS_REVIEW, (f"Employer names are similar but not the same "
                              f"(similarity {similarity:.2f}).")
    if guessed:
        return NEEDS_REVIEW, (f"The slip header does not match the declared employer "
                              f"(similarity {similarity:.2f}), but the header was only "
                              f"a guess, so a person should check.")
    return MISMATCH, f"Employer names clearly differ (similarity {similarity:.2f})."


def check_arithmetic(gross, deductions, net):
    """Does gross - deductions = net? Returns 'passed', 'failed' or 'not possible'."""
    if gross is None or deductions is None or net is None:
        return "not possible"
    return "passed" if abs(gross - deductions - net) <= ARITHMETIC_TOLERANCE else "failed"


def _money(value):
    return f"{value:,.2f}"


def compare_salary(declared_income, fields, text_method):
    """Return (status, message) comparing declared gross income with the slip gross."""
    gross = fields["gross_salary"]["value"]
    net = fields["net_salary"]["value"]
    deductions = fields["total_deductions"]["value"]

    if declared_income is None or declared_income <= 0:
        return NEEDS_REVIEW, "No valid declared monthly gross income was provided."
    if gross is None:
        return NEEDS_REVIEW, "The gross salary could not be read from the slip."

    arithmetic = check_arithmetic(gross, deductions, net)
    exact_text = text_method == "pdf text layer"
    # OCR numbers are trusted only if the slip's own arithmetic confirms them.
    # Text taken directly from a PDF is exact, so it is trusted unless arithmetic fails.
    trusted = arithmetic == "passed" or (exact_text and arithmetic == "not possible")
    difference = abs(declared_income - gross) / declared_income

    arithmetic_text = {
        "passed": "The slip's own figures add up (gross - deductions = net).",
        "failed": "The slip's own figures do not add up (gross - deductions is not net).",
        "not possible": "The slip's arithmetic could not be checked "
                        "(total deductions or net not found).",
    }[arithmetic]

    net_hint = ""
    if net is not None and abs(declared_income - net) / declared_income <= SALARY_TOLERANCE:
        net_hint = " The declared figure matches the slip's NET salary, but the form asks for GROSS."

    if arithmetic == "failed":
        return NEEDS_REVIEW, (f"{arithmetic_text} Either a number was misread or the "
                              f"slip is inconsistent.{net_hint}")
    if difference <= SALARY_TOLERANCE:
        if trusted:
            return CONSISTENT, (f"Declared gross income matches the slip gross "
                                f"({_money(gross)}). {arithmetic_text}")
        return NEEDS_REVIEW, (f"The figures match ({_money(gross)}), but OCR can misread "
                              f"digits and the slip's arithmetic could not be checked. "
                              f"A person should confirm.")
    if difference <= SALARY_NEAR:
        return NEEDS_REVIEW, (f"Declared gross income is close to, but not equal to, the "
                              f"slip gross ({_money(gross)}); difference {difference:.1%}.{net_hint}")
    if trusted:
        message = (f"Declared gross income differs clearly from the slip gross "
                   f"({_money(gross)}); difference {difference:.1%}. "
                   f"{arithmetic_text}{net_hint}")
        # Declaring net pay instead of gross is a common honest mix-up
        return (NEEDS_REVIEW if net_hint else MISMATCH), message


def compare_period(fields, application_date):
    """Return (status, message): is the pay period readable and recent?"""
    raw = fields["pay_period"]["value"]
    parsed = fields["pay_period_parsed"]
    if raw is None:
        return NEEDS_REVIEW, "The pay period was not found on the slip."
    if parsed is None:
        return NEEDS_REVIEW, "The pay period was found but its month could not be read."
    year, month = parsed
    age = (application_date.year - year) * 12 + (application_date.month - month)
    if age < 0:
        return NEEDS_REVIEW, "The pay period is in the future."
    if age > MAX_SLIP_AGE_MONTHS:
        return NEEDS_REVIEW, f"The slip is {age} months old; the limit is {MAX_SLIP_AGE_MONTHS}."
    return CONSISTENT, f"The slip is {age} month(s) old, within the {MAX_SLIP_AGE_MONTHS}-month limit."


def _finding(field, declared, extracted, result):
    status, message = result
    return {"field": field, "declared": declared, "extracted": extracted,
            "status": status, "message": message}


def compare_declared_with_slip(declared, fields, text_method, application_date=None):
    """Compare the form details with extracted slip fields and build a report."""
    application_date = application_date or date.today()
    income = declared.get("monthly_gross_income")
    gross = fields["gross_salary"]["value"]

    findings = [
        _finding("Employee name", declared.get("full_name"),
                 fields["employee_name"]["value"],
                 compare_names(declared.get("full_name"), fields["employee_name"]["value"])),
        _finding("Employer name", declared.get("employer_name"),
                 fields["employer_name"]["value"],
                 compare_employers(declared.get("employer_name"),
                                   fields["employer_name"]["value"],
                                   fields["employer_name"]["source"])),
        _finding("Gross monthly salary",
                 _money(income) if income else None,
                 _money(gross) if gross is not None else None,
                 compare_salary(income, fields, text_method)),
        _finding("Pay period", "(recent slip expected)",
                 fields["pay_period"]["value"],
                 compare_period(fields, application_date)),
    ]
    overall = max((f["status"] for f in findings), key=_SEVERITY.get)

    notes = ["This check compares text only. It does not prove the document is "
             "authentic, and a mismatch is not proof of fraud."]
    if text_method == "ocr":
        notes.append("The text was read by OCR and may contain errors.")
    return {"overall_status": overall, "findings": findings,
            "text_method": text_method, "needs_manual_review": overall != CONSISTENT,
            "notes": notes}


def verify_salary_slip(path, declared, application_date=None):
    """Read a salary slip file, extract fields and compare with the form."""
    try:
        text, method = read_document_text(path)
    except Exception as error:      # unreadable, corrupt or unsupported file
        text, method = "", f"failed ({type(error).__name__})"
    report = compare_declared_with_slip(declared, extract_fields(text), method, application_date)
    if not text.strip():
        report["notes"].insert(0, "No text could be read from the document.")
    return report


def format_report(report):
    """Plain-text discrepancy report."""
    lines = [f"Overall: {report['overall_status']}   (text from: {report['text_method']})"]
    for f in report["findings"]:
        lines.append(f"  [{f['status']}] {f['field']}")
        lines.append(f"      declared: {f['declared']}   |   slip: {f['extracted']}")
        lines.append(f"      {f['message']}")
    for note in report["notes"]:
        lines.append(f"  Note: {note}")
    return "\n".join(lines)


if __name__ == "__main__":
    slips = Path(__file__).resolve().parent.parent / "data" / "sample_slips"
    today = date(2026, 11, 18)      # fixed date, so the results are repeatable

    asha = dict(full_name="Asha Verma", employer_name="Sample Tech Pvt Ltd",
                monthly_gross_income=60000)
    rohan = dict(full_name="Rohan Mehta", employer_name="Lakeside Services Ltd",
                 monthly_gross_income=38000)
    meera = dict(full_name="Meera Iyer", employer_name="Brightpath Logistics Pvt Ltd",
                 monthly_gross_income=30000)

    scenarios = [
        ("1. Matching details, clean image", "slip_01_clean.png", asha, today),
        ("2. Matching details, digital PDF", "slip_01_digital.pdf", asha, today),
        ("3. Declared income too high", "slip_01_clean.png",
         {**asha, "monthly_gross_income": 75000}, today),
        ("4. Declared income equals NET salary", "slip_01_clean.png",
         {**asha, "monthly_gross_income": 54000}, today),
        ("5. Different label wording", "slip_02_other_labels.png", rohan, today),
        ("6. Different applicant name", "slip_02_other_labels.png",
         {**rohan, "full_name": "Priya Nair"}, today),
        ("7. Noisy scan, PNG", "slip_03_scan_like.png", meera, today),
        ("8. Noisy scan, PDF", "slip_03_scan_like.pdf", meera, today),
        ("9. Slip is old", "slip_01_clean.png", asha, date(2027, 3, 1)),
        ("10. File cannot be read", "does_not_exist.png", asha, today),
    ]
    for title, filename, declared, when in scenarios:
        print("=" * 70)
        print(title)
        print(format_report(verify_salary_slip(slips / filename, declared, when)))