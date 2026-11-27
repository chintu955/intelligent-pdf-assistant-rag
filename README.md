# 🤖 Intelligent PDF Assistant – RAG Monitoring Dashboard

An AI-powered PDF Question Answering system built using **Retrieval-Augmented Generation (RAG)**.

This project allows users to ask questions about PDF documents and receive context-based answers with source citations.

It also includes hallucination detection, citation validation, answer quality evaluation, and a final answer acceptance or rejection mechanism.

## 🚀 Project Overview

Traditional Large Language Models (LLMs) can sometimes generate incorrect or unsupported information.

This project addresses that problem by retrieving relevant information from PDF documents before generating answers.

The generated answers are then checked using multiple validation components to improve reliability.

## ✨ Key Features

- PDF document question answering
- Retrieval-Augmented Generation (RAG)
- Semantic search using ChromaDB
- Document reranking
- Local LLM integration using Ollama
- Hallucination detection
- Citation validation
- Answer quality evaluation
- Automated ACCEPT / WARNING / REJECT decisions
- Interactive Streamlit monitoring dashboard
- PDF source and page references

## 🏗️ System Architecture

```text
              PDF Documents
                    |
                    v
             Document Loading
                    |
                    v
             Text Extraction
                    |
                    v
              Text Chunking
                    |
                    v
             Embedding Model
                    |
                    v
                ChromaDB
                    |
                    v
              User Question
                    |
                    v
                Retrieval
                    |
                    v
                Reranking
                    |
                    v
             Context + Query
                    |
                    v
                Local LLM
                    |
                    v
             Generated Answer
                    |
          ---------------------
          |         |         |
          v         v         v
     Hallucination Citation  Quality
       Check      Check    Evaluation
          |         |         |
          ---------------------
                    |
                    v
              Final Decision
                    |
          ---------------------
          |         |         |
          v         v         v
        ACCEPT   WARNING    REJECT
                    |
                    v
           Streamlit Dashboard
```

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web application and dashboard |
| LangChain | Document and RAG components |
| ChromaDB | Vector database |
| Ollama | Local LLM execution |
| Llama 3.2 | Answer generation |
| Nomic Embed Text | Text embeddings |
| Sentence Transformers | Reranking |
| PyMuPDF | PDF text extraction |

## 📊 Answer Validation Pipeline

Every generated answer passes through multiple validation stages.

1. **Hallucination Check:** Checks whether the answer is supported by retrieved PDF context.
2. **Citation Validation:** Checks whether source citations are valid.
3. **Quality Evaluation:** Evaluates correctness, relevance, and completeness.
4. **Final Decision:** Determines whether the answer should be accepted, shown with a warning, or rejected.

### Final Decision Categories

| Decision | Meaning |
|---|---|
| ACCEPT | Answer passes the validation checks |
| WARNING | Answer needs additional attention |
| REJECT | Answer should not be accepted |

## 🧪 Project Testing

The following test cases were performed:

| Test Question | Expected Result | Actual Result |
|---|---|---|
| What is Machine Learning? | ACCEPT | ACCEPT |
| What is the difference between classification and regression? | ACCEPT | ACCEPT |
| What is the capital city of France? | REJECT | REJECT |

The France question tests whether the system avoids answering questions that are outside the available PDF context.


## 📁 Project Structure

```text
intelligent-pdf-assistant-rag/
│
├── app.py
├── monitoring_dashboard.py
├── llm.py
├── retriever.py
├── reranker.py
├── multi_pdf_ingest.py
├── hallucination_checker.py
├── citation_validator.py
├── answer_quality_evaluator.py
├── quality_decision.py
├── evaluate.py
├── test_answer.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

**Note:** PDF documents and generated vector database files are excluded from this public repository.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.12
- Git
- Ollama

### 1. Clone the repository

```bash
git clone https://github.com/chintu955/intelligent-pdf-assistant-rag.git
```

### 2. Navigate to the project folder

```bash
cd intelligent-pdf-assistant-rag
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Install Ollama models

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Make sure Ollama is running before starting the application.

### 7. Run the Streamlit dashboard

```bash
python -m streamlit run monitoring_dashboard.py
```

## 🎯 Project Objective

<<<<<<< HEAD
The objective of this project is to develop a document-based AI assistant that combines Retrieval-Augmented Generation with answer validation, citation verification, and quality-based decision-making to improve the reliability of LLM-generated responses.

## 🔮 Future Improvements

- Support for additional document formats
- Improved retrieval and reranking strategies
- Advanced evaluation using RAGAS
- Better monitoring and logging
- User feedback integration
- Cloud deployment

=======
The objective of this project is to develop a document-based AI assistant that combines Retrieval-Augmented Generation with answer validation, citation verification, and quality-based decision-making to improve the reliability of LLM-generated responses.

## 🔮 Future Improvements

- Support for additional document formats
- Improved retrieval and reranking strategies
- Advanced evaluation using RAGAS
- Better monitoring and logging
- User feedback integration
- Cloud deployment

## 👨‍💻 Author

**AI/ML Engineering Portfolio Project**

**Project:** Intelligent PDF Assistant – RAG Monitoring Dashboard

**GitHub:** https://github.com/chintu955