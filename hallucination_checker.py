from langchain_ollama import ChatOllama

# Initialize the checker LLM
checker_llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


def check_hallucination(answer, context):

    prompt = f"""
You are a strict AI answer evaluator.

Your task is to check whether the answer
is supported by the provided PDF context.

PDF CONTEXT:
{context}

GENERATED ANSWER:
{answer}

RULES:
1. Check every factual claim in the answer.
2. Use only the provided PDF context.
3. Do not use your external knowledge.
4. If all important claims are supported,
   return SUPPORTED.
5. If any important claim is unsupported
   or contradicts the context, return
   NOT_SUPPORTED.

Return your result in this format:

VERDICT: SUPPORTED or NOT_SUPPORTED

REASON: Explain why in one or two sentences.
"""

    response = checker_llm.invoke(prompt)

    return response.content