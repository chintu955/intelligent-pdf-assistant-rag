import pandas as pd


# Load evaluation results
df = pd.read_csv("with_reranking_results.csv")


# Questions we want to investigate
target_questions = [
    "What is the difference between classification and regression?",
    "Explain the advantages of Machine Learning.",
    "What is the difference between binary and multi-class classification?"
]


for question in target_questions:

    matching_rows = df[
        df["user_input"] == question
    ]

    if matching_rows.empty:
        print("\nQuestion not found:", question)
        continue

    row = matching_rows.iloc[0]

    print("\n" + "=" * 70)
    print("QUESTION:", row["user_input"])

    print("\nGENERATED ANSWER:")
    print(row["response"])

    print("\nEXPECTED REFERENCE:")
    print(row["reference"])

    print("\nRETRIEVED CONTEXTS:")

    contexts = row["retrieved_contexts"]

    if isinstance(contexts, str):

        try:
            import ast
            contexts = ast.literal_eval(contexts)
        except (ValueError, SyntaxError):
            pass

    if isinstance(contexts, list):

        for index, context in enumerate(contexts, start=1):

            print(f"\n--- Context {index} ---")
            print(context)

    else:
        print(contexts)

    print("\nRAGAS SCORES:")

    print("Faithfulness:", row["faithfulness"])
    print("Answer Relevancy:", row["answer_relevancy"])
    print("Context Precision:",
          row["llm_context_precision_with_reference"])
    print("Context Recall:", row["context_recall"])