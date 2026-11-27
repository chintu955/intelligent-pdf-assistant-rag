import streamlit as st
from llm import generate_answer
from multi_pdf_ingest import add_uploaded_pdfs


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Intelligent PDF Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------
# CUSTOM CSS
# --------------------------------

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.title {
    font-size: 36px;
    font-weight: bold;
    color: #2563eb;
}

.subtitle {
    font-size: 17px;
    color: #64748b;
}

.stChatMessage {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# SESSION STATE
# --------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------
# SIDEBAR
# --------------------------------

with st.sidebar:

    st.title("📚 PDF Assistant")

    st.write("Advanced RAG-powered document assistant.")

    st.divider()

    st.subheader("Project Information")

    st.write("🤖 LLM: Llama 3.2")

    st.write("🧠 Embeddings: Nomic")

    st.write("🗄️ Vector DB: ChromaDB")

    st.write("🔍 Retrieval: Semantic Search")

    st.write("📊 Evaluation: RAGAS")

    st.divider()

    # ==================================
    # MULTIPLE PDF UPLOAD
    # ==================================

    st.subheader("📁 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("📤 Index Documents"):

        if not uploaded_files:

            st.warning("Please upload at least one PDF.")

        else:

            with st.spinner("Processing PDF documents..."):

                try:

                    results = add_uploaded_pdfs(
                        uploaded_files
                    )

                    st.success("Document processing completed!")

                    for result in results:

                        st.write(f"📄 {result['filename']}")

                        st.write(f"Status: {result['status']}")

                        if "pages" in result:

                            st.write(f"Pages: {result['pages']}")

                            st.write(f"Chunks: {result['chunks']}")

                        st.divider()

                except Exception as e:

                    st.error(f"PDF processing failed: {e}")

    st.divider()

    # ==================================
    # CLEAR CONVERSATION
    # ==================================

    if st.button("🗑️ Clear Conversation"):

        st.session_state.messages = []

        st.rerun()


# --------------------------------
# MAIN HEADER
# --------------------------------

st.markdown(
    '<div class="title">Intelligent PDF Research Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions and get answers directly from your PDF documents.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if "sources" in message:

            with st.expander("📄 View Retrieved Sources"):

                for index, source in enumerate(
                    message["sources"], start=1
                ):

                    st.markdown(f"**Source {index}**")

                    st.write("File:", source.get("filename", "Unknown"))

                    st.write("Page:", source.get("page", "Unknown"))

                    st.write("Retrieved Content:")

                    st.write(source.get("content", ""))

                    st.divider()


# --------------------------------
# USER INPUT
# --------------------------------

question = st.chat_input("Ask something about your PDFs...")


if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------
    # GENERATE AI RESPONSE
    # --------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Searching PDFs and generating answer..."):

            try:

                result = generate_answer(question)

                answer = result["answer"]

                sources = result.get("sources", [])

                st.markdown(answer)

                if sources:

                    with st.expander("📄 View Retrieved Sources"):

                        for index, source in enumerate(
                            sources, start=1
                        ):

                            st.markdown(f"**Source {index}**")

                            st.write(
                                "File:",
                                source.get("filename", "Unknown")
                            )

                            st.write(
                                "Page:",
                                source.get("page", "Unknown")
                            )

                            st.write(
                                "Retrieved Content:"
                            )

                            st.write(
                                source.get("content", "")
                            )

                            st.divider()


                st.session_state.messages.append({

                    "role": "assistant",

                    "content": answer,

                    "sources": sources

                })


            except Exception as e:

                st.error(f"Something went wrong: {e}")

                st.info(
                    "Please check Ollama and your RAG backend."
                )