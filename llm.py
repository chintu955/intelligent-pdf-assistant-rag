import os
from reranker import rerank_documents
from langchain_ollama import ChatOllama
from retriever import retrieve_documents


# ==========================================
# STEP 1: Initialize Ollama LLM
# ==========================================

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# ==========================================
# STEP 2: Format Chat History
# ==========================================

def format_chat_history(chat_history):

    if not chat_history:
        return "No previous conversation."

    history_text = ""

    for message in chat_history[-6:]:

        if isinstance(message, dict):

            role = message.get("role", "user")
            content = message.get("content", "")

        else:

            role = "user"
            content = str(message)

        history_text += f"{role}: {content}\n"

    return history_text


# ==========================================
# STEP 3: Query Rewriting
# ==========================================

def rewrite_query(query, chat_history=None):

    history_text = format_chat_history(chat_history)

    if not chat_history:
        return query

    prompt = f"""
You are a question rewriting assistant.

Convert the user's question into a clear
standalone question.

RULES:

1. Understand conversation history.
2. Identify what the user is referring to.
3. Rewrite follow-up questions independently.
4. Do not answer the question.
5. Do not add unnecessary information.
6. If the question is already clear, return it unchanged.
7. Return only the rewritten question.

Conversation History:

{history_text}

Current Question:

{query}

Standalone Question:
"""

    response = llm.invoke(prompt)

    rewritten_query = response.content.strip()

    if not rewritten_query:
        return query

    print("\nOriginal Question:", query)
    print("Rewritten Question:", rewritten_query)

    return rewritten_query



# ==========================================
# STEP 4: Generate Answer
# ==========================================

def generate_answer(query, chat_history=None, use_reranker=True):

    # STEP 1: Rewrite the question
    rewritten_query = rewrite_query(
        query,
        chat_history
    )

    # STEP 2: Retrieve top 20 documents
    retrieved_docs = retrieve_documents(rewritten_query)

    # STEP 3: Select top 5 documents

    if use_reranker:

        # Advanced RAG: Rerank and select top 8
        retrieved_docs = rerank_documents(
            query=rewritten_query,
            documents=retrieved_docs,
            top_n=8
        )

    else:

        # Baseline RAG: Select first 8 without reranking
        retrieved_docs = retrieved_docs[:8]

    # STEP 4: Handle empty retrieval
    if not retrieved_docs:

       return {
    "answer": answer,
    "sources": sources,
    "rewritten_query": rewritten_query,
    "context": context
}

    # STEP 5: Prepare Context and Sources

    context_parts = []
    sources = []
    source_mapping = {}

    for doc in retrieved_docs:

        content = doc.page_content
        metadata = doc.metadata

        filename = metadata.get("filename")

        if not filename:
            filename = os.path.basename(
                metadata.get("source", "Unknown")
            )

        page_number = metadata.get("page", "Unknown")

        if isinstance(page_number, int):
            page_number += 1

        source_key = (filename, page_number)

        # Create source ID for each unique PDF page
        if source_key not in source_mapping:

            source_id = f"S{len(source_mapping) + 1}"

            source_mapping[source_key] = source_id

            sources.append({
                "id": source_id,
                "filename": filename,
                "page": page_number,
                "content": content
            })

        else:

            source_id = source_mapping[source_key]

            # Combine chunks belonging to the same page
            for source in sources:

                if source["id"] == source_id:

                    source["content"] += "\n\n" + content
                    break

        # Add each retrieved chunk to LLM context
        context_parts.append(
            f"""
[Source ID: {source_id}]
Filename: {filename}
Page: {page_number}
Content:
{content}
"""
        )

    context = "\n\n".join(context_parts)

    # STEP 6: Format Chat History
    history_text = format_chat_history(chat_history)

    # STEP 7: Citation-Based RAG Prompt
    prompt = f"""
You are an accurate and helpful PDF assistant.

Your task is to answer the user's question
using ONLY the provided PDF context.

STRICT RULES:

1. Use only the PDF context for factual answers.
2. Never invent facts or information.
3. Answer the exact question asked.
4. Keep answers clear and easy to understand.
5. Avoid unnecessary repetition.
6. Do not use outside knowledge.
7. Conversation history is only for understanding
   previous messages. It is NOT factual evidence.
8. Every factual answer must include source citations.
9. Use the exact source ID provided in the context.
10. Citation format:
    [Source: S1, Page: 3]
11. If multiple sources support the answer,
    cite all relevant sources.
12. Never invent source IDs or page numbers.
13. If the answer is not available in the PDF
    context, respond exactly:
    Information not found in the PDF.
14. Do not provide citations for unsupported information.

Conversation History:

{history_text}

PDF Context:

{context}

User Question:

{rewritten_query}

Answer:
"""

    # STEP 8: Generate Answer
    response = llm.invoke(prompt)

    answer = response.content.strip()

    # STEP 9: Return Final Result
    return {
        "answer": answer,
        "sources": sources,
        "rewritten_query": rewritten_query
    }
