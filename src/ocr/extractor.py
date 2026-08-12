"""Extract text from PDF and image medical reports."""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pdfplumber
from PIL import Image, ImageOps

from src.utils.file_handlers import is_image, is_pdf, load_image_from_bytes

logger = logging.getLogger(__name__)

# Common Windows install locations for Tesseract (optional fallback)
TESSERACT_CANDIDATES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\{user}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
]


@dataclass
class ExtractionResult:
    text: str
    method: str
    pages: int = 1
    warnings: list[str] = field(default_factory=list)


def extract_text_from_pdf_bytes(data: bytes) -> ExtractionResult:
    """Extract embedded text from a PDF. Falls back to OCR for scanned pages."""
    warnings: list[str] = []
    texts: list[str] = []

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        page_count = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                texts.append(page_text)
            else:
                warnings.append(f"Page {i} had little/no embedded text; attempting OCR.")
                try:
                    ocr_text = _ocr_pdf_page(page)
                    if ocr_text.strip():
                        texts.append(ocr_text)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"OCR failed on page {i}: {exc}")

    combined = "\n\n".join(texts).strip()
    method = "pdfplumber"
    if any("OCR" in w for w in warnings):
        method = "pdfplumber+ocr"

    if not combined:
        warnings.append("No text could be extracted from this PDF.")

    return ExtractionResult(text=combined, method=method, pages=page_count, warnings=warnings)


def _ocr_pdf_page(page) -> str:
    """Render a pdfplumber page to an image and OCR it."""
    try:
        pil_image = page.to_image(resolution=200).original
        return ocr_image(pil_image)[0]
    except Exception:
        from pdf2image import convert_from_bytes

        images = convert_from_bytes(
            page.pdf.stream.getvalue(),
            first_page=page.page_number,
            last_page=page.page_number,
        )
        if not images:
            return ""
        return ocr_image(images[0])[0]


def _preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """Light preprocessing to improve OCR on lab printouts / phone photos."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Upscale small images (phone crops / screenshots)
    min_side = min(image.size)
    if min_side < 1000:
        scale = 1000 / min_side
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.LANCZOS,
        )

    gray = ImageOps.grayscale(image)
    # Mild contrast boost
    gray = ImageOps.autocontrast(gray)
    return gray.convert("RGB")


@lru_cache(maxsize=1)
def _get_rapidocr():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def _ocr_with_rapid(image: Image.Image) -> str:
    import numpy as np

    engine = _get_rapidocr()
    arr = np.array(image)
    result, _ = engine(arr)
    if not result:
        return ""
    # Each item: [box, text, confidence]
    lines = [item[1] for item in result if item and len(item) > 1 and item[1]]
    return "\n".join(lines).strip()


def _configure_tesseract() -> bool:
    """Return True if pytesseract can find a Tesseract binary."""
    try:
        import pytesseract
        from pytesseract import TesseractNotFoundError
    except ImportError:
        return False

    # Already discoverable on PATH?
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        pass

    user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    for candidate in TESSERACT_CANDIDATES:
        path = candidate.format(user=user)
        if Path(path).exists():
            pytesseract.pytesseract.tesseract_cmd = path
            try:
                pytesseract.get_tesseract_version()
                return True
            except Exception:
                continue
    return False


def _ocr_with_tesseract(image: Image.Image) -> str:
    import pytesseract

    if not _configure_tesseract():
        raise RuntimeError("Tesseract OCR binary not found on this system.")

    config = "--oem 3 --psm 6"
    return pytesseract.image_to_string(image, config=config).strip()


def ocr_image(image: Image.Image) -> tuple[str, str, list[str]]:
    """
    OCR a PIL image.

    Returns (text, method, warnings).
    Prefers RapidOCR (pip-only). Falls back to Tesseract if installed.
    """
    warnings: list[str] = []
    prepared = _preprocess_for_ocr(image)

    # 1) RapidOCR — no system install required
    try:
        text = _ocr_with_rapid(prepared)
        if text.strip():
            return text, "rapidocr", warnings
        warnings.append("RapidOCR returned empty text; trying Tesseract fallback.")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"RapidOCR unavailable ({exc}); trying Tesseract fallback.")

    # 2) Tesseract — optional system binary
    try:
        text = _ocr_with_tesseract(prepared)
        if text.strip():
            return text, "tesseract", warnings
        warnings.append("Tesseract returned empty text.")
    except Exception as exc:  # noqa: BLE001
        warnings.append(str(exc))

    return (
        "",
        "failed",
        warnings
        + [
            "Could not read text from this image.",
            "Tips: use a clearer photo, paste the report text manually, or try the built-in sample PNG.",
        ],
    )


def extract_text_from_image_bytes(data: bytes) -> ExtractionResult:
    image = load_image_from_bytes(data)
    text, method, warnings = ocr_image(image)

    if not text.strip() and method != "failed":
        warnings.append("OCR returned empty text. Try a clearer image or paste text manually.")

    return ExtractionResult(text=text, method=method, pages=1, warnings=warnings)


def extract_text(filename: str, data: bytes) -> ExtractionResult:
    """Route extraction based on file type."""
    if is_pdf(filename):
        return extract_text_from_pdf_bytes(data)
    if is_image(filename):
        return extract_text_from_image_bytes(data)
    return ExtractionResult(
        text="",
        method="unsupported",
        warnings=[f"Unsupported file type for '{filename}'."],
    )


def ocr_status() -> dict[str, bool | str]:
    """Report which OCR backends are available (for UI). Cached after first probe."""
    return _ocr_status_cached()


@lru_cache(maxsize=1)
def _ocr_status_cached() -> dict[str, bool | str]:
    """Report which OCR backends are available (for UI)."""
    rapid_ok = False
    tess_ok = False
    try:
        _get_rapidocr()
        rapid_ok = True
    except Exception as exc:  # noqa: BLE001
        rapid_msg = str(exc)
    else:
        rapid_msg = "ready"

    tess_ok = _configure_tesseract()
    return {
        "rapidocr": rapid_ok,
        "tesseract": tess_ok,
        "rapidocr_detail": rapid_msg,
        "primary": "rapidocr" if rapid_ok else ("tesseract" if tess_ok else "none"),
    }
