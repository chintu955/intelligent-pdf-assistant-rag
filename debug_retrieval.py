from retriever import retrieve_documents
from reranker import rerank_documents


question = "What is the difference between classification and regression?"

# STEP 1: Retrieve 20 candidate chunks
documents = retrieve_documents(question)

print("\nTOTAL RETRIEVED DOCUMENTS:", len(documents))


# STEP 2: Rerank all retrieved chunks
reranked_documents = rerank_documents(
    question,
    documents,
    top_n=20,
    debug=True
)


# STEP 3: Display the top 10 ranked chunks
print("\n========== TOP 10 RERANKED CHUNKS ==========")

for index, doc in enumerate(reranked_documents[:10], start=1):

    print(f"\n--- RANK {index} ---")

    print("PAGE:", doc.metadata.get("page"))

    print("CONTENT:")
    print(doc.page_content[:700])