"""Extract employee name, employer, salary and pay period from salary slip text."""

import re
from pathlib import Path

from src.document_verification import ALLOWED_EXTENSIONS, read_document_text

# Label patterns, most specific first. Each is matched at the START of a line.
NAME_LABELS = [
    r"employee\s+name",
    r"name\s+of\s+(?:the\s+)?employee",
    r"emp\.?\s+name",
    r"name",
]
EMPLOYER_LABELS = [
    r"employer(?:\s+name)?",
    r"company(?:\s+name)?",
    r"organi[sz]ation(?:\s+name)?",
    r"establishment(?:\s+name)?",
]
PERIOD_LABELS = [
    r"pay\s+period",
    r"salary\s+period",
    r"salary\s+month",
    r"pay\s+month",
    r"for\s+the\s+month\s+of",
    r"period",
    r"month",
]
GROSS_LABELS = [
    r"gross\s+(?:earnings?|salary|pay|wages|total)",
    r"total\s+earnings?",
    r"gross",
]
NET_LABELS = [
    r"net\s+(?:salary|pay|amount\s+payable|amount|payable|take\s+home)",
    r"take[\s-]*home(?:\s+(?:pay|salary))?",
]
DEDUCTION_LABELS = [
    r"tot[a-z]{1,2}\s+deductions?",
]

# A word that usually ends a company name (used only for the header guess)
COMPANY_SUFFIX = re.compile(
    r"\b(?:pvt|private|ltd|limited|llp|inc|corporation|corp)\b", re.IGNORECASE
)

# An amount such as 30,000.00 or 4500
AMOUNT = re.compile(r"\d[\d,]*(?:\.\d{1,2})?")
# A whole line that is only an amount, e.g. "Rs. 54,000.00" or ". 54,000.00"
AMOUNT_ONLY_LINE = re.compile(
    r"^\W{0,3}(?:rs\.?|inr)?\s*(\d[\d,]*(?:\.\d{1,2})?)\W{0,3}$", re.IGNORECASE
)

MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
          "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
MONTH_YEAR = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s*(\d{4})\b",
    re.IGNORECASE,
)
NUMERIC_MONTH_YEAR = re.compile(r"\b(0?[1-9]|1[0-2])[/-](\d{4})\b")


def find_labelled_text(lines, label_patterns):
    """Return the text after the first 'Label: value' line, or None."""
    for pattern in label_patterns:
        regex = re.compile(rf"^\W*(?:{pattern})\s*[:;\-]\s*(.+)$", re.IGNORECASE)
        for line in lines:
            match = regex.match(line)
            if match and match.group(1).strip():
                return match.group(1).strip()
    return None


def guess_employer_from_header(lines):
    """Fallback: the first of the top lines that looks like a company name."""
    for line in lines[:6]:
        if COMPANY_SUFFIX.search(line):
            return re.sub(r"^\W+|\W+$", "", line)    # strip stray symbols at both ends
    return None


def to_number(text):
    """'30,000.00' -> 30000.0"""
    return float(text.replace(",", ""))


def find_labelled_amount(lines, label_patterns):
    """Amount on the label's own line, or alone on the next line. None if absent."""
    for pattern in label_patterns:
        regex = re.compile(rf"^\W*(?:{pattern})\b(.*)$", re.IGNORECASE)
        for i, line in enumerate(lines):
            match = regex.match(line)
            if not match:
                continue
            found = AMOUNT.search(match.group(1))          # same line
            if found:
                return to_number(found.group())
            if i + 1 < len(lines):                         # next line
                following = AMOUNT_ONLY_LINE.match(lines[i + 1])
                if following:
                    return to_number(following.group(1))
    return None


def parse_period(text):
    """Return (year, month) of the first month found in the text, or None."""
    if not text:
        return None
    match = MONTH_YEAR.search(text)
    if match:
        return int(match.group(2)), MONTHS[match.group(1).lower()]
    match = NUMERIC_MONTH_YEAR.search(text)
    if match:
        return int(match.group(2)), int(match.group(1))
    return None


def _field(value, source):
    """Package a value with how it was found, so the report can show it."""
    return {"value": value, "source": source if value is not None else "not found"}


def extract_fields(text):
    """Extract the salary slip fields from OCR or PDF text."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    employer = find_labelled_text(lines, EMPLOYER_LABELS)
    employer_source = "labelled"
    if employer is None:
        employer = guess_employer_from_header(lines)
        employer_source = "guessed from header"

    period = find_labelled_text(lines, PERIOD_LABELS)

    return {
        "employee_name": _field(find_labelled_text(lines, NAME_LABELS), "labelled"),
        "employer_name": _field(employer, employer_source),
        "gross_salary": _field(find_labelled_amount(lines, GROSS_LABELS), "labelled"),
        "net_salary": _field(find_labelled_amount(lines, NET_LABELS), "labelled"),
        "total_deductions": _field(find_labelled_amount(lines, DEDUCTION_LABELS), "labelled"),
        "pay_period": _field(period, "labelled"),
        "pay_period_parsed": parse_period(period),
    }


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent.parent / "data" / "sample_slips"
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in ALLOWED_EXTENSIONS:
            text, method = read_document_text(f)
            fields = extract_fields(text)
            print("=" * 60)
            print(f.name, "| text method:", method)
            for key in ["employee_name", "employer_name", "gross_salary",
                        "net_salary", "pay_period"]:
                item = fields[key]
                print(f"  {key:14s}: {item['value']!r}   ({item['source']})")
            print("  pay period parsed (year, month):", fields["pay_period_parsed"])