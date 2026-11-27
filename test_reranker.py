from langchain_core.documents import Document
from reranker import rerank_documents


# Sample user question
query = "What is Machine Learning?"


# Sample documents
documents = [
    Document(
        page_content="Python is a programming language used for software development."
    ),
    Document(
        page_content="Machine Learning is a branch of AI that learns patterns from data."
    ),
    Document(
        page_content="Deep Learning uses neural networks to learn complex patterns."
    ),
    Document(
        page_content="HTML is used to structure web pages."
    ),
]


# Apply reranking
results = rerank_documents(
    query=query,
    documents=documents,
    top_n=3
)


# Display results
print("\n========== RERANKING RESULTS ==========\n")

for index, document in enumerate(results, start=1):
    print(f"Rank {index}:")
    print(document.page_content)
    print("-" * 50)