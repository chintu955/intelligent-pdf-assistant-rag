from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

pdf_files = list(
    Path("data/uploads").glob("*Python*")
)

pdf_path = pdf_files[0]

loader = PyPDFLoader(str(pdf_path))

documents = loader.load()

print("\n========== PDF TEXT ANALYSIS ==========")

for index, doc in enumerate(documents, start=1):

    text = doc.page_content

    print(f"\nPage: {index}")
    print("Characters extracted:", len(text))

    if text.strip():

        print("Text preview:")
        print(text[:300])

    else:

        print("WARNING: No text extracted!")

    print("-" * 50)