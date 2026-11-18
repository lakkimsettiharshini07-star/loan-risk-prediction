"""Salary slip checking: read text, extract fields, compare with the form."""

import shutil
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image, ImageOps

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
MIN_TEXT_CHARS = 50     # a PDF page with less text than this is treated as a scan
MAX_PAGES = 3           # read at most this many pages of a PDF


def _configure_tesseract():
    """If 'tesseract' is not on PATH, use the default Windows install location."""
    if shutil.which("tesseract") is None:
        default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if default.exists():
            pytesseract.pytesseract.tesseract_cmd = str(default)


_configure_tesseract()


def prepare_image(img):
    """Clean an image before OCR: fix phone rotation, grayscale, enlarge, boost contrast."""
    img = ImageOps.exif_transpose(img)        # honour the orientation stored by phone cameras
    img = img.convert("L")                    # grayscale
    if img.width < 1600:                      # Tesseract prefers large, clear text
        scale = 1600 / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)),
                         Image.Resampling.LANCZOS)
    return ImageOps.autocontrast(img)


def ocr_image(img):
    """Run Tesseract on one image and return the text."""
    return pytesseract.image_to_string(prepare_image(img), config="--psm 6")


def read_document_text(path):
    """Return (text, method). `method` records how the text was obtained."""
    path = Path(path)
    extension = path.suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")

    if extension == ".pdf":
        pages, used_ocr = [], False
        with pymupdf.open(path) as doc:
            for number, page in enumerate(doc):
                if number >= MAX_PAGES:
                    break
                text = page.get_text()                      # real text, if the PDF has any
                if len(text.strip()) < MIN_TEXT_CHARS:      # looks like a scan: use OCR
                    pix = page.get_pixmap(dpi=200)          # render the page as an image
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    text = ocr_image(img)
                    used_ocr = True
                pages.append(text)
        return "\n".join(pages), ("ocr" if used_ocr else "pdf text layer")

    with Image.open(path) as img:
        return ocr_image(img), "ocr"


if __name__ == "__main__":
    # Quick check: read every sample slip and print what was extracted
    folder = Path(__file__).resolve().parent.parent / "data" / "sample_slips"
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in ALLOWED_EXTENSIONS:
            text, method = read_document_text(f)
            print("=" * 60)
            print(f.name, "| method:", method)
            print("=" * 60)
            print(text)