"""Generate a sample lab-report PNG for OCR testing."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_TXT = ROOT / "samples" / "sample_blood_report.txt"
OUT_PNG = ROOT / "samples" / "sample_blood_report.png"


def main() -> None:
    text = SAMPLE_TXT.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Prefer a monospaced font so columns stay readable for OCR
    font = None
    for candidate in [
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\cour.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]:
        if Path(candidate).exists():
            font = ImageFont.truetype(candidate, 18)
            break
    if font is None:
        font = ImageFont.load_default()

    padding = 40
    line_h = 26
    width = 900
    height = padding * 2 + line_h * (len(lines) + 2)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill="black", font=font)
        y += line_h

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT_PNG, format="PNG")
    print(f"Wrote {OUT_PNG} ({OUT_PNG.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
