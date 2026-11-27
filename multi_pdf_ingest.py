import hashlib
from pathlib import Path

import pymupdf
import pytesseract

from PIL import Image

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# =====================================
# 1. CONFIGURATION
# =====================================

UPLOAD_FOLDER = Path("data/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Tesseract installation path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# =====================================
# 2. EMBEDDING MODEL
# =====================================

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)


# =====================================
# 3. CHROMADB CONNECTION
# =====================================

vector_store = Chroma(
    collection_name="beginner_rag",
    embedding_function=embedding_model,
    persist_directory="./vector_db_experiment"
)


# =====================================
# 4. TEXT SPLITTER
# =====================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)


# =====================================
# 5. PDF TEXT EXTRACTION + OCR
# =====================================

def extract_pdf_text(pdf_path):

    pdf = pymupdf.open(pdf_path)

    documents = []

    print("\nProcessing PDF:", pdf_path.name)
    print("Total pages:", len(pdf))

    for page_number, page in enumerate(pdf):

        # First try normal text extraction
        extracted_text = page.get_text("text")

        extraction_method = "Direct extraction"

        # If text is missing, use OCR
        if len(extracted_text.strip()) < 30:

            print(f"Page {page_number + 1}: Using OCR")

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

            extraction_method = "OCR"

        else:

            print(f"Page {page_number + 1}: Direct extraction")

        # Store only pages containing text
        if extracted_text.strip():

            document = Document(
                page_content=extracted_text,
                metadata={
                    "source": str(pdf_path),
                    "page": page_number,
                    "extraction_method": extraction_method
                }
            )

            documents.append(document)

    pdf.close()

    return documents


# =====================================
# 6. MULTIPLE PDF UPLOAD FUNCTION
# =====================================

def add_uploaded_pdfs(uploaded_files):

    results = []

    for uploaded_file in uploaded_files:

        try:

            pdf_bytes = uploaded_file.getvalue()

            # Generate unique file hash
            file_hash = hashlib.sha256(
                pdf_bytes
            ).hexdigest()

            original_filename = Path(
                uploaded_file.name
            ).name

            # Check whether PDF already exists in ChromaDB
            existing = vector_store._collection.get(
                where={"file_hash": file_hash},
                limit=1
            )

            if existing["ids"]:

                results.append({
                    "filename": original_filename,
                    "status": "Already uploaded"
                })

                continue

            # Save uploaded PDF
            saved_filename = (
                f"{file_hash[:12]}_{original_filename}"
            )

            pdf_path = UPLOAD_FOLDER / saved_filename

            pdf_path.write_bytes(pdf_bytes)

            # Extract text using direct extraction or OCR
            documents = extract_pdf_text(pdf_path)

            if not documents:

                results.append({
                    "filename": original_filename,
                    "status": "No readable text found"
                })

                continue

            # Add original filename and hash to metadata
            for document in documents:

                document.metadata["filename"] = (
                    original_filename
                )

                document.metadata["file_hash"] = file_hash

            # Divide text into chunks
            chunks = text_splitter.split_documents(
                documents
            )

            if not chunks:

                results.append({
                    "filename": original_filename,
                    "status": "No chunks generated"
                })

                continue

            # Store chunks in ChromaDB
            vector_store.add_documents(chunks)

            results.append({
                "filename": original_filename,
                "status": "Successfully indexed",
                "pages": len(documents),
                "chunks": len(chunks)
            })

            print("\nSuccessfully indexed:", original_filename)
            print("Pages:", len(documents))
            print("Chunks:", len(chunks))

        except Exception as error:

            results.append({
                "filename": uploaded_file.name,
                "status": f"Error: {str(error)}"
            })

    return results