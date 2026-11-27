from langchain_ollama import ChatOllama


# Initialize evaluation model
evaluator_llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


def evaluate_answer_quality(query, answer, context):

    prompt = f"""
You are an expert AI answer quality evaluator.

Evaluate the quality of the generated answer
using ONLY the provided PDF context.

USER QUESTION:
{query}

PDF CONTEXT:
{context}

GENERATED ANSWER:
{answer}


Evaluate these three metrics:

1. CORRECTNESS
Does the answer contain information supported
by the PDF context?

2. RELEVANCE
Does the answer directly answer the user's question?

3. COMPLETENESS
Does the answer cover all important information
needed to answer the question?


SCORING RULES:

Give each metric a score from 1 to 5.

1 = Very Poor
2 = Poor
3 = Average
4 = Good
5 = Excellent

Do not reward unsupported information.

If the answer says:
"Information not found in the PDF."

Then evaluate whether refusing to answer was
appropriate based on the context.


Return your evaluation in exactly this format:

CORRECTNESS: [score]/5
RELEVANCE: [score]/5
COMPLETENESS: [score]/5

OVERALL SCORE: [score]/5

FEEDBACK:
[Write a short explanation of the scores.]
"""

    response = evaluator_llm.invoke(prompt)

    return response.content