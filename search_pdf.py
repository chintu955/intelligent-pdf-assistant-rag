from pypdf import PdfReader

pdf = PdfReader("data/ML Notes.pdf")

page_number = 12

text = pdf.pages[page_number - 1].extract_text()

print(f"\n========== PDF PAGE {page_number} ==========\n")
print(text)