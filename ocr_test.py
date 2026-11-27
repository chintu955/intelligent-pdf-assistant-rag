import pymupdf
import pytesseract

from pathlib import Path
from PIL import Image

pdf_files = list(
    Path("data/uploads").glob(
        "*Python_Complete_Notes_ask_sreekanth.pdf"
    )
)

pdf_path = pdf_files[0]
pdf = pymupdf.open(pdf_path)

print("Total pages:", len(pdf))

# Test pages 2 to 5
for page_number in range(1, min(5, len(pdf))):

    page = pdf[page_number]

    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(2, 2),
        alpha=False
    )

    image = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples
    )

    extracted_text = pytesseract.image_to_string(
        image,
        lang="eng"
    )

    print("\n==============================")
    print("PAGE:", page_number + 1)
    print("Characters:", len(extracted_text))
    print("==============================")

    print(extracted_text[:700])

pdf.close()