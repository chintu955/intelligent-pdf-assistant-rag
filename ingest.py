
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# STEP 1: Load PDF
loader = PyPDFLoader("data/ML Notes.pdf")
documents = loader.load()

print("PDF loaded successfully!")
print("Total pages:", len(documents))


# STEP 2: Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))

# STEP 3: Create embedding model
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)


# STEP 4: Store chunks in ChromaDB
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./vector_db_experiment",
    collection_name="beginner_rag"
)

print("Data stored successfully in ChromaDB!")
print("Total stored chunks:", vector_store._collection.count())
