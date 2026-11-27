import pandas as pd


# ==========================================
# STEP 1: Load Evaluation Results
# ==========================================

df = pd.read_csv("with_reranking_results.csv")

print("\n========== RAG ERROR ANALYSIS ==========")

print("\nTotal Questions:", len(df))


# ==========================================
# STEP 2: Define Thresholds
# ==========================================

thresholds = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.80,
    "llm_context_precision_with_reference": 0.80,
    "context_recall": 0.80
}


# ==========================================
# STEP 3: Identify Errors
# ==========================================

error_reports = []

for index, row in df.iterrows():

    errors = []

    for metric, threshold in thresholds.items():

        score = row[metric]

        if pd.isna(score):

            errors.append(f"{metric}: Missing score")

        elif score < threshold:

            errors.append(
                f"{metric}: Low score ({score:.2f})"
            )

    if errors:

        error_reports.append({

            "Question": row["user_input"],

            "Faithfulness": row["faithfulness"],

            "Answer Relevancy": row["answer_relevancy"],

            "Context Precision": row[
                "llm_context_precision_with_reference"
            ],

            "Context Recall": row["context_recall"],

            "Errors": " | ".join(errors)

        })


# ==========================================
# STEP 4: Display Error Report
# ==========================================

if error_reports:

    error_df = pd.DataFrame(error_reports)

    print("\n========== IDENTIFIED ISSUES ==========")

    print(error_df.to_string(index=False))

else:

    print("\nNo low scores or missing scores detected!")


# ==========================================
# STEP 5: Save Error Report
# ==========================================

error_df = pd.DataFrame(error_reports)

error_df.to_csv(
    "rag_error_analysis.csv",
    index=False
)

print("\nError analysis report saved successfully!")

print("\n========== ANALYSIS COMPLETED ==========")