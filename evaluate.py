import json
import pandas as pd

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall
)

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from langchain_ollama import ChatOllama, OllamaEmbeddings

from llm import generate_answer


# ==========================================
# STEP 1: Configure Evaluation LLM
# ==========================================

local_llm = ChatOllama(
    model="llama3:latest",
    temperature=0
)

evaluator_llm = LangchainLLMWrapper(local_llm)


# ==========================================
# STEP 2: Configure Embeddings
# ==========================================

local_embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)

evaluator_embeddings = LangchainEmbeddingsWrapper(
    local_embeddings
)


# ==========================================
# STEP 3: Prepare Test Questions
# ==========================================

test_cases = [

    {
        "question": "What is Machine Learning?",
        "reference": (
            "Machine Learning is the practice of finding patterns "
            "in historical data and using those patterns to make "
            "predictions on new, unseen data."
        )
    },

    {
        "question": "Explain the advantages of Machine Learning.",
        "reference": (
            "Machine Learning automatically extracts complex patterns "
            "from historical data. It can handle very large datasets, "
            "discover rules automatically and generalise them to new data."
        )
    },

    {
        "question": "What are the three types of Machine Learning?",
        "reference": (
            "The three main types are supervised learning, "
            "unsupervised learning, and reinforcement learning."
        )
    },

    {
        "question": "What is supervised learning?",
        "reference": (
            "Supervised learning uses labelled data. "
            "The model learns from historical labelled examples "
            "to predict outcomes."
        )
    },

    {
        "question": "What is unsupervised learning?",
        "reference": (
            "Unsupervised learning uses data without target labels. "
            "The algorithm discovers patterns and structures automatically."
        )
    },

    {
        "question": "What is reinforcement learning?",
        "reference": (
            "Reinforcement learning trains an agent by rewarding "
            "good decisions and penalising bad decisions."
        )
    },

    {
        "question": "What is the difference between classification and regression?",
        "reference": (
            "Regression predicts continuous numerical values. "
            "Classification predicts categories from a finite set."
        )
    },

    {
        "question": "What is the difference between binary and multi-class classification?",
        "reference": (
            "Binary classification predicts between two categories. "
            "Multi-class classification predicts one category "
            "from three or more categories."
        )
    },

    {
        "question": "Explain missing-value imputation strategies.",
        "reference": (
            "Missing-value imputation replaces missing entries "
            "with substitute values. Mean is useful without significant "
            "outliers, median is useful when outliers exist, "
            "and mode is suitable for categorical columns."
        )
    },

    {
        "question": "What is the difference between traditional programming and Machine Learning?",
        "reference": (
            "Traditional programming uses explicit human-written rules. "
            "Machine Learning discovers patterns and rules from data."
        )
    }

]


# ==========================================
# STEP 4: Generate Evaluation Dataset
# ==========================================

def generate_evaluation_data(use_reranker):

    evaluation_data = []

    for test in test_cases:

        question = test["question"]

        print("\nEvaluating:", question)

        result = generate_answer(
            question,
            use_reranker=use_reranker
        )

        contexts = [
            source["content"]
            for source in result["sources"]
        ]

        evaluation_data.append({

            "user_input": question,

            "response": result["answer"],

            "retrieved_contexts": contexts,

            "reference": test["reference"]

        })

    return evaluation_data


# ==========================================
# STEP 5: Configure RAGAS Metrics
# ==========================================

def get_metrics():

    faithfulness.llm = evaluator_llm

    answer_relevancy.llm = evaluator_llm

    answer_relevancy.embeddings = evaluator_embeddings

    context_precision = LLMContextPrecisionWithReference(
        llm=evaluator_llm
    )

    context_recall = LLMContextRecall(
        llm=evaluator_llm
    )

    return [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]


# ==========================================
# STEP 6: Run RAGAS Evaluation
# ==========================================

def run_evaluation(data, pipeline_name):

    print("\n===================================")
    print("Starting:", pipeline_name)
    print("===================================")

    dataset = Dataset.from_list(data)

    result = evaluate(
        dataset=dataset,
        metrics=get_metrics(),
        run_config=RunConfig(
            max_workers=1,
            timeout=600
        )
    )

    df = result.to_pandas()

    print("\nResults:", pipeline_name)
    print(result)

    print("\nDetailed Results:")
    print(df)

    filename = pipeline_name.lower().replace(" ", "_")

    df.to_csv(
        f"{filename}_results.csv",
        index=False
    )

    with open(
        f"{filename}_dataset.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Calculate average metric scores

    metric_columns = [
        "faithfulness",
        "answer_relevancy",
        "llm_context_precision_with_reference",
        "context_recall"
    ]

    scores = {}

    for metric in metric_columns:

        if metric in df.columns:

            scores[metric] = {
                "mean": df[metric].mean(),
                "valid_count": df[metric].count(),
                "total_count": len(df)
            }

    return scores


# ==========================================
# STEP 7: BEFORE RERANKING
# ==========================================

# ==========================================
# STEP 7: BEFORE RERANKING
# ==========================================

print("\n\n####################################")
print("EVALUATION WITHOUT RERANKING")
print("####################################")

before_data = generate_evaluation_data(
    use_reranker=False
)

before_scores = run_evaluation(
    before_data,
    "Without Reranking"
)


# ==========================================
# STEP 8: AFTER RERANKING
# ==========================================

print("\n\n####################################")
print("EVALUATION WITH RERANKING")
print("####################################")

after_data = generate_evaluation_data(
    use_reranker=True
)

after_scores = run_evaluation(
    after_data,
    "With Reranking"
)


# ==========================================
# STEP 9: Compare Results
# ==========================================

metrics = [
    ("Faithfulness", "faithfulness"),
    ("Answer Relevancy", "answer_relevancy"),
    ("Context Precision", "llm_context_precision_with_reference"),
    ("Context Recall", "context_recall")
]

comparison_rows = []

for display_name, metric_name in metrics:

    before = before_scores.get(metric_name, {})
    after = after_scores.get(metric_name, {})

    before_mean = before.get("mean")
    after_mean = after.get("mean")

    if pd.notna(before_mean) and pd.notna(after_mean):
        improvement = after_mean - before_mean
    else:
        improvement = None

    comparison_rows.append({

        "Metric": display_name,

        "Without Reranking": before_mean,

        "With Reranking": after_mean,

        "Improvement": improvement,

        "Before Valid Questions": before.get("valid_count"),

        "After Valid Questions": after.get("valid_count")

    })

comparison = pd.DataFrame(comparison_rows)

# ==========================================
# STEP 10: Display Final Comparison
# ==========================================

print("\n\n========================================")
print("FINAL RAGAS PERFORMANCE COMPARISON")
print("========================================")

display_comparison = comparison.copy()

score_columns = [
    "Without Reranking",
    "With Reranking",
    "Improvement"
]

for column in score_columns:
    display_comparison[column] = (
        display_comparison[column] * 100
    ).round(2)

print(display_comparison.to_string(index=False))

comparison.to_csv(
    "ragas_before_after_comparison.csv",
    index=False
)

print("\nEvaluation completed successfully!")

print("\nAll evaluation files saved successfully!")