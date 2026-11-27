from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# STEP 1: Initialize embedding model
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)


# STEP 2: Connect to ChromaDB
vector_store = Chroma(
    collection_name="beginner_rag",
    embedding_function=embedding_model,
    persist_directory="./vector_db_clean"
)


# STEP 3: Retrieve relevant documents
def retrieve_documents(query):

    results = vector_store.similarity_search(
        query,
        k=20
    )

    return results