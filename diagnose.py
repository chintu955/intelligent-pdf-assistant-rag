from llm import generate_answer

test_questions = [
    "What is Machine Learning?",
    "Explain the advantages of Machine Learning."
]

for question in test_questions:

    print("\n" + "=" * 60)
    print("QUESTION:", question)
    print("=" * 60)

    result = generate_answer(question)

    print("\nREWRITTEN QUESTION:")
    print(result["rewritten_query"])

    print("\nGENERATED ANSWER:")
    print(result["answer"])

    print("\nRETRIEVED SOURCES:")

    for index, source in enumerate(result["sources"], start=1):

        print(f"\n--- Source {index} ---")
        print("File:", source.get("filename"))
        print("Page:", source.get("page"))
        print("Content:")
        print(source["content"])

    print("\n" + "=" * 60)
