"""Generate FICTIONAL salary slips for testing OCR. All names and companies are invented."""

import random
from functools import lru_cache
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_slips"
FOOTER = "SYNTHETIC SAMPLE - FICTIONAL DATA FOR SOFTWARE TESTING ONLY"

SLIPS = {
    "slip_01_clean": dict(
        header="Sample Tech Pvt Ltd", company_label=None, company="Sample Tech Pvt Ltd",
        name_label="Employee Name", name="Asha Verma", emp_id="ST-1042",
        designation="Software Engineer", period_label="Pay Period", period="October 2026",
        earnings=[("Basic Pay", 30000), ("House Rent Allowance", 12000),
                  ("Special Allowance", 18000)],
        deductions=[("Provident Fund", 3600), ("Professional Tax", 200),
                    ("Income Tax (TDS)", 2200)],
        gross_label="Gross Earnings", net_label="Net Salary",
    ),
    "slip_02_other_labels": dict(
        header="PAYSLIP", company_label="Organisation", company="Lakeside Services Ltd",
        name_label="Name of Employee", name="Rohan Mehta", emp_id="LS-2207",
        designation="Accounts Executive", period_label="Month", period="September 2026",
        earnings=[("Basic", 22000), ("HRA", 8800), ("Conveyance", 1600),
                  ("Other Allowances", 5600)],
        deductions=[("PF", 2640), ("Prof. Tax", 200)],
        gross_label="Gross Salary", net_label="Net Pay",
    ),
    "slip_03_scan_like": dict(
        header="Brightpath Logistics Pvt Ltd", company_label=None,
        company="Brightpath Logistics Pvt Ltd",
        name_label="Employee Name", name="Meera Iyer", emp_id="BL-3318",
        designation="Operations Associate", period_label="Pay Period",
        period="01 Oct 2026 - 31 Oct 2026",
        earnings=[("Basic Pay", 18000), ("HRA", 7200), ("Special Allowance", 4800)],
        deductions=[("Provident Fund", 2160), ("Professional Tax", 200)],
        gross_label="Total Earnings", net_label="Net Amount Payable",
    ),
}


def money(amount):
    return f"Rs. {amount:,.2f}"


@lru_cache(maxsize=None)
def load_font(size, bold=False):
    """Use a Windows Arial font; fall back to Pillow's built-in font."""
    try:
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def slip_lines(s):
    """List of (left text, right text or None, bold) rows. The totals are calculated."""
    gross = sum(a for _, a in s["earnings"])
    deductions = sum(a for _, a in s["deductions"])
    net = gross - deductions
    items = []
    if s["company_label"]:
        items.append((f'{s["company_label"]}: {s["company"]}', None, False))
    items += [
        (f'{s["name_label"]}: {s["name"]}', None, False),
        (f'Employee ID: {s["emp_id"]}', None, False),
        (f'Designation: {s["designation"]}', None, False),
        (f'{s["period_label"]}: {s["period"]}', None, False),
        ("", None, False),
        ("Earnings", None, True),
    ]
    items += [(label, money(a), False) for label, a in s["earnings"]]
    items += [(s["gross_label"], money(gross), True), ("Deductions", None, True)]
    items += [(label, money(a), False) for label, a in s["deductions"]]
    items += [("Total Deductions", money(deductions), True), (s["net_label"], money(net), True)]
    return items


def draw_slip_image(s):
    """Draw the slip as a clean image."""
    img = Image.new("RGB", (1240, 1150), "white")
    d = ImageDraw.Draw(img)
    d.text((60, 40), s["header"], fill="black", font=load_font(46, True))
    d.line((60, 110, 1180, 110), fill="black", width=3)
    y = 140
    for left, right, bold in slip_lines(s):
        font = load_font(30, bold)
        d.text((60, y), left, fill="black", font=font)
        if right:
            d.text((760, y), right, fill="black", font=font)
        y += 48
    d.text((60, y + 30), FOOTER, fill="gray", font=load_font(22))
    return img


def make_scan_like(img):
    """Make a clean image look like a poor scan: tilt, blur, speckle noise."""
    img = img.convert("L")
    img = img.rotate(2, expand=True, fillcolor=255)      # tilt by 2 degrees
    img = img.filter(ImageFilter.GaussianBlur(1.2))      # slight blur
    rng = random.Random(0)                               # fixed seed = same noise each run
    pixels = img.load()
    for _ in range(40000):
        x, y = rng.randrange(img.width), rng.randrange(img.height)
        pixels[x, y] = rng.randrange(120, 255)
    return img


def save_text_pdf(s, path):
    """A digital PDF: the text is stored as real, selectable text."""
    doc = pymupdf.open()
    page = doc.new_page()                                # A4 page
    page.insert_text((50, 60), s["header"], fontsize=18, fontname="hebo")
    y = 95
    for left, right, bold in slip_lines(s):
        font = "hebo" if bold else "helv"                # bold / regular Helvetica
        page.insert_text((50, y), left, fontsize=11, fontname=font)
        if right:
            page.insert_text((350, y), right, fontsize=11, fontname=font)
        y += 20
    page.insert_text((50, y + 20), FOOTER, fontsize=8, fontname="helv")
    doc.save(str(path))
    doc.close()


def save_image_pdf(img, path):
    """A scanned-style PDF: the page is only a picture, so OCR is needed."""
    img.convert("RGB").save(str(path), "PDF", resolution=150)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    s1, s2, s3 = SLIPS["slip_01_clean"], SLIPS["slip_02_other_labels"], SLIPS["slip_03_scan_like"]

    draw_slip_image(s1).save(OUT_DIR / "slip_01_clean.png")
    save_text_pdf(s1, OUT_DIR / "slip_01_digital.pdf")
    draw_slip_image(s2).save(OUT_DIR / "slip_02_other_labels.png")

    scan_like = make_scan_like(draw_slip_image(s3))
    scan_like.save(OUT_DIR / "slip_03_scan_like.png")
    save_image_pdf(scan_like, OUT_DIR / "slip_03_scan_like.pdf")

    print("Created files in:", OUT_DIR)
    for f in sorted(OUT_DIR.iterdir()):
        print(" ", f.name)