from llm import generate_answer
from quality_decision import make_quality_decision
from hallucination_checker import check_hallucination
from citation_validator import validate_citations
from answer_quality_evaluator import evaluate_answer_quality

# ====================================
# 1. USER QUESTION
# ====================================

query = """
What is the difference between classification and regression?
"""

print("\n====================================")
print("       BEGINNER RAG ANSWER TEST")
print("====================================")

print("\nQUESTION:")
print(query)

# ====================================
# 2. GENERATE ANSWER
# ====================================

print("\nGenerating answer... Please wait.")

result = generate_answer(
    query,
    chat_history=None,
    use_reranker=True
)

answer = result.get("answer", "")
sources = result.get("sources", [])

print("\nGENERATED ANSWER:")


# ====================================
# 3. RECONSTRUCT PDF CONTEXT
# ====================================

context = "\n\n".join(
    f"[Page: {source.get('page')}]\n{source.get('content', '')}"
    for source in sources
    if isinstance(source, dict)
)

# ====================================
# 4. ANSWER VALIDATION
# ====================================

print("\n====================================")
print("       ANSWER VALIDATION")
print("====================================")

if not answer.strip():

    print("ANSWER REJECTED: No answer generated.")

elif answer.strip() == "Information not found in the PDF.":

    print("\nFINAL ANSWER:")
    print(answer)

elif not context.strip():

    print("ANSWER REJECTED: PDF context is missing.")

else:

    print("\nChecking answer against PDF context...")

    # Hallucination verification
    verification = check_hallucination(answer, context)

    print("\nHALLUCINATION CHECK:")
    print(verification)

    hallucination_supported = (
        "VERDICT: SUPPORTED" in verification.upper()
        and "VERDICT: NOT_SUPPORTED" not in verification.upper()
    )

    # Citation verification
    citation_result = validate_citations(answer, sources)

    print("\nCITATION VERIFICATION:")
    print(citation_result["message"])

    # Final decision
    if hallucination_supported and citation_result["valid"]:

        print("\n====================================")
        print("          VERIFIED ANSWER")
        print("====================================")

        print(answer)

        print("\n====================================")
        print("       ANSWER QUALITY EVALUATION")
        print("====================================")

        quality_result = evaluate_answer_quality(
            query=query,
            answer=answer,
            context=context
        )

        print(quality_result)
                # Make automatic quality decision
        decision_result = make_quality_decision(
            quality_result
        )

        print("\n====================================")
        print("       FINAL QUALITY DECISION")
        print("====================================")

        print("Decision:", decision_result["decision"])
        print("Reason:", decision_result["reason"])
                # Enforce final quality decision

        if decision_result["decision"] == "ACCEPT":

            print("\n====================================")
            print("          ANSWER ACCEPTED")
            print("====================================")

            print(answer)

        elif decision_result["decision"] == "WARNING":

            print("\n====================================")
            print("       ANSWER ACCEPTED WITH WARNING")
            print("====================================")

            print("WARNING: Please review this answer.")

            print(answer)

        else:

            print("\n====================================")
            print("          ANSWER REJECTED")
            print("====================================")

            print(
                "The answer did not meet the required "
                "quality standards."
            )

    else:

        print("\n====================================")
        print("          ANSWER REJECTED")
        print("====================================")

        print(
            "The answer failed hallucination or citation verification."
        )

# ====================================
# 5. TEST COMPLETE
# ====================================

print("\n====================================")
print("             TEST COMPLETE")
print("====================================")