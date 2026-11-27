from sentence_transformers import CrossEncoder

# STEP 1: Initialize Cross-Encoder
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# STEP 2: Rerank documents
def rerank_documents(query, documents, top_n=3, debug=False):

    if not documents:
        return []

    # STEP 3: Create question-document pairs
    pairs = [
        [query, document.page_content]
        for document in documents
    ]

    # STEP 4: Calculate relevance scores
    scores = reranker.predict(pairs)

    # STEP 5: Attach scores to documents
    scored_documents = list(zip(scores, documents))

    # STEP 6: Sort by relevance
    scored_documents.sort(
        key=lambda item: item[0],
        reverse=True
    )

    # STEP 7: Debug information
    if debug:
        print("\n========== RERANKING SCORES ==========")

        for rank, (score, document) in enumerate(
            scored_documents,
            start=1
        ):
            print(f"\nRank: {rank}")
            print(f"Score: {float(score):.4f}")
            print(f"Content: {document.page_content[:300]}")

    # STEP 8: Select top documents
    reranked_documents = [
        document
        for score, document in scored_documents[:top_n]
    ]

    return reranked_documents