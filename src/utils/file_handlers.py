"""File handling helpers for uploaded medical reports."""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

from PIL import Image

SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def normalize_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def is_supported(filename: str) -> bool:
    return normalize_extension(filename) in SUPPORTED_EXTENSIONS


def is_image(filename: str) -> bool:
    return normalize_extension(filename) in IMAGE_EXTENSIONS


def is_pdf(filename: str) -> bool:
    return normalize_extension(filename) == ".pdf"


def load_image_from_bytes(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    return image


def read_upload_bytes(upload: BinaryIO) -> bytes:
    upload.seek(0)
    return upload.read()
