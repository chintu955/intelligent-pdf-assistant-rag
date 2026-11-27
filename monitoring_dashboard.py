import re
import streamlit as st

from llm import generate_answer
from hallucination_checker import check_hallucination
from citation_validator import validate_citations
from answer_quality_evaluator import evaluate_answer_quality
from quality_decision import make_quality_decision


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="RAG Monitoring Dashboard",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------
# DASHBOARD TITLE
# --------------------------------

st.title("🤖 RAG Monitoring Dashboard")

st.write(
    "Ask questions about your PDF and monitor "
    "the performance of your RAG system."
)

st.divider()


# --------------------------------
# USER QUESTION
# --------------------------------

st.subheader("Ask Your PDF")

query = st.text_area(
    "Enter your question:",
    placeholder="Example: What is Machine Learning?"
)


# --------------------------------
# PROCESS QUESTION
# --------------------------------

if st.button("Generate Answer", type="primary"):

    if not query.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Generating and validating answer..."):

            try:

                # Generate RAG answer
                result = generate_answer(
                    query,
                    chat_history=None,
                    use_reranker=True
                )

                answer = result.get("answer", "")
                sources = result.get("sources", [])

                if not isinstance(answer, str):
                    answer = str(answer)

                if not isinstance(sources, list):
                    sources = []

                # --------------------------------
                # BUILD PDF CONTEXT
                # --------------------------------

                context_parts = []

                for source in sources:

                    if isinstance(source, dict):

                        page = source.get("page", "Unknown")

                        content = source.get(
                            "content",
                            ""
                        )

                        context_parts.append(
                            f"[Page: {page}]\n{content}"
                        )

                context = "\n\n".join(context_parts)

                # --------------------------------
                # DEFAULT STATUS VALUES
                # --------------------------------

                hallucination_status = "NOT CHECKED"
                citation_status = "NOT CHECKED"

                quality_result = "NOT EVALUATED"

                decision = "REJECT"

                reason = "Answer could not be verified."

                hallucination_supported = False
                citation_valid = False

                # --------------------------------
                # CHECK EMPTY ANSWER
                # --------------------------------

                if not answer.strip():

                    reason = "No answer was generated."

                # --------------------------------
                # CHECK PDF FALLBACK
                # --------------------------------

                elif answer.strip() == "Information not found in the PDF.":

                    reason = "Information was not found in the PDF."

                # --------------------------------
                # CHECK MISSING CONTEXT
                # --------------------------------

                elif not context.strip():

                    reason = "PDF context is missing."

                else:

                    # --------------------------------
                    # HALLUCINATION CHECK
                    # --------------------------------

                    verification = check_hallucination(
                        answer,
                        context
                    )

                    verification_text = str(
                        verification
                    ).upper()

                    hallucination_supported = (
                        "VERDICT: SUPPORTED" in verification_text
                        and "VERDICT: NOT_SUPPORTED"
                        not in verification_text
                    )

                    if hallucination_supported:

                        hallucination_status = "SUPPORTED"

                    else:

                        hallucination_status = "NOT_SUPPORTED"

                    # --------------------------------
                    # CITATION VALIDATION
                    # --------------------------------

                    citation_result = validate_citations(
                        answer,
                        sources
                    )

                    citation_valid = (
                        isinstance(citation_result, dict)
                        and citation_result.get("valid", False)
                    )

                    if citation_valid:

                        citation_status = "VALID"

                    else:

                        citation_status = "INVALID"

                    # --------------------------------
                    # QUALITY EVALUATION
                    # --------------------------------

                    if hallucination_supported and citation_valid:

                        quality_result = evaluate_answer_quality(
                            query=query,
                            answer=answer,
                            context=context
                        )

                        decision_result = make_quality_decision(
                            quality_result
                        )

                        decision = decision_result.get(
                            "decision",
                            "REJECT"
                        )

                        reason = decision_result.get(
                            "reason",
                            "Quality evaluation completed."
                        )

                    else:

                        reason = (
                            "Answer failed hallucination "
                            "or citation verification."
                        )

            except Exception as error:

                st.error(
                    f"An error occurred while processing the question: {error}"
                )

                st.stop()

        # --------------------------------
        # DISPLAY MONITORING RESULTS
        # --------------------------------

        st.divider()

        st.subheader("RAG Performance Monitoring")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Hallucination Check",
            hallucination_status
        )

        col2.metric(
            "Citation Check",
            citation_status
        )

        # --------------------------------
        # EXTRACT QUALITY SCORE
        # --------------------------------

        overall_score = "N/A"

        if quality_result != "NOT EVALUATED":

            match = re.search(
                r"OVERALL SCORE:\s*(\d+(?:\.\d+)?)\s*/\s*5",
                str(quality_result),
                re.IGNORECASE
            )

            if match:

                overall_score = f"{match.group(1)}/5"

        col3.metric(
            "Quality Score",
            overall_score
        )

        col4.metric(
            "Final Decision",
            decision
        )

        # --------------------------------
        # DISPLAY FINAL ANSWER
        # --------------------------------

        st.divider()

        st.subheader("Final Answer")

        if decision == "ACCEPT":

            st.success("Answer Accepted")

            st.markdown(answer)

        elif decision == "WARNING":

            st.warning(
                "Answer accepted with warning. "
                "Please review it."
            )

            st.markdown(answer)

        else:

            st.error("Answer Rejected")

            st.write(reason)

        # --------------------------------
        # DISPLAY QUALITY DETAILS
        # --------------------------------

        if quality_result != "NOT EVALUATED":

            with st.expander("View Quality Evaluation"):

                st.text(str(quality_result))

        # --------------------------------
        # DISPLAY RETRIEVED PDF SOURCES
        # --------------------------------

        with st.expander("View Retrieved PDF Sources"):

            if sources:

                for index, source in enumerate(
                    sources,
                    start=1
                ):

                    if isinstance(source, dict):

                        st.markdown(
                            f"### Retrieved Source {index}"
                        )

                        source_id = source.get(
                            "id",
                            f"S{index}"
                        )

                        filename = source.get(
                            "filename",
                            "PDF Document"
                        )

                        page = source.get(
                            "page",
                            "Unknown"
                        )

                        content = source.get(
                            "content",
                            "No content available."
                        )

                        st.write(
                            f"**Source ID:** {source_id}"
                        )

                        st.write(
                            f"**File:** {filename}"
                        )

                        st.write(
                            f"**Page:** {page}"
                        )

                        st.text_area(
                            label=f"Source Content {index}",
                            value=str(content),
                            height=150,
                            disabled=True,
                            key=f"source_content_{index}"
                        )

                        st.divider()

                    else:

                        st.write(source)

            else:

                st.info("No sources retrieved.")

        # --------------------------------
        # FOOTER
        # --------------------------------

        st.caption(
            "Beginner RAG Project | AI/ML Engineering"
        )