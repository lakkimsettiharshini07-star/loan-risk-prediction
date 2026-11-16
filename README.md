# Alternative Credit Risk Assessment and Applicant Verification System

A machine-learning **decision-support prototype** for microfinance and small private lenders
assessing **salaried** loan applicants, including those with limited credit history.

> **This system never approves or rejects an applicant.** It gives the lender evidence.
> The lender makes and records the final decision.

## Status
Work in progress.

## What the system does
1. **ML risk estimate:** a model trained on historical loan data estimates the likelihood of
   recorded repayment difficulty and shows a descriptive risk category.
2. **Affordability indicators:** loan-to-income and instalment-to-income ratios, shown
   separately from the model output.
3. **Salary-slip check (OCR):** extracts employee name, employer name, salary amount, and pay
   period, compares them with the declared details, and reports
   **Consistent / Needs Review / Mismatch** with a discrepancy report.
4. **Assessment report:** combines all of the above for the lender.
5. **Lender decision:** recorded separately from the model output, with reason and timestamp.

## Inputs
- **Applicant:** full name, employment type, occupation, employment duration, employer name.
- **Financial:** monthly gross income, requested loan amount, expected monthly instalment,
  loan type.
- **Optional:** credit bureau score, if the lender already has one. It is **displayed in the
  report only**. The model does not use it, because the training data has no such column.
- **Document:** salary slip (PDF, JPG, JPEG, PNG).

## Dataset
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data),
file `application_train.csv` only. It is **not included** in this repository. Download it from
Kaggle (accept the competition rules first) and place it in `data/raw/`.

## Planned technology
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn (Logistic Regression and Random
Forest, compared and calibrated), Tesseract with Pytesseract, PyMuPDF, Pillow, Streamlit,
SQLite, Joblib.

## Limitations
- **The ML model** is trained on historical Home Credit data from a different country and
  currency. It does **not** represent Indian microfinance borrowers and is **not validated
  for Indian lenders**. Before real use it would need to be retrained and validated on the
  lender's own data.
- The dataset contains applicants who were given loans, not only rejected applicants.
- Model output is a risk indicator for human review. It is not a guarantee of repayment,
  and it is not a credit score.
- A good affordability ratio does not guarantee repayment.
- **OCR extracts text; it does not prove a document is authentic.** A mismatch is not proof
  of fraud. Unreadable or missing information goes to manual review.
- The document check is demonstrated only on **synthetic sample salary slips**.
- **Bank history is not verified.** Real verification of employment, bank accounts, or PAN
  requires the lender's own checks or consent-based, regulated channels.
- The first version supports **salaried applicants only**.
- The model uses education level, which can act as a proxy for social background. A lender
  should review this choice before real use. Its contribution to model performance was not
  separately measured.

## Privacy
Uploaded documents, the SQLite database, raw data, and secrets are excluded from Git through
`.gitignore`. Use only synthetic documents for demonstrations.

## Future work
- Support for self-employed applicants (for example, bank statements or tax records).
- Consent-based bank data through India's Account Aggregator framework.
- PAN and employer verification through authorised channels.
- Retraining and validation on real Indian lender data.

## Project structure
```
LoanRisk/
  data/raw, data/processed
  notebooks/        exploratory analysis
  models/           saved model pipeline
  src/              data, features, training, prediction, OCR, database modules
  uploads/          temporary document storage (not committed)
  app.py            Streamlit application
```

## Setup (Windows)
1. Create a virtual environment: `python -m venv .venv`
2. Activate it: `.venv\Scripts\activate`
3. Install packages: `pip install -r requirements.txt`
4. Place `application_train.csv` in `data/raw/`