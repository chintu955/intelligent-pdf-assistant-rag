from retriever import retrieve_documents
from reranker import rerank_documents

query = "What is the difference between classification and regression?"

print("\n========== STEP 1: INITIAL RETRIEVAL ==========\n")

documents = retrieve_documents(query)

print(f"Total retrieved documents: {len(documents)}")

for i, doc in enumerate(documents, start=1):
    print(f"\n--- Document {i} ---")
    print("Page:", doc.metadata.get("page"))
    print("Content:", doc.page_content[:300])


print("\n========== STEP 2: RERANKING ==========\n")

reranked_documents = rerank_documents(
    query,
    documents,
    top_n=5,
    debug=True
)

print(f"\nTotal documents after reranking: {len(reranked_documents)}")

for i, doc in enumerate(reranked_documents, start=1):
    print(f"\n--- Reranked Document {i} ---")
    print("Page:", doc.metadata.get("page"))
    print("Content:", doc.page_content[:300])