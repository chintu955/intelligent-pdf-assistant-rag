from retriever import vector_store

data = vector_store._collection.get(
    include=["metadatas"]
)

print("\n========== DATABASE REPORT ==========")

print("Total chunks:", len(data["metadatas"]))

file_counts = {}

for metadata in data["metadatas"]:

    filename = metadata.get("filename", "Unknown")

    file_counts[filename] = file_counts.get(filename, 0) + 1

print("\n========== FILE-WISE CHUNKS ==========")

for filename, count in file_counts.items():

    print(filename, ":", count, "chunks")

print("\n========== RECENT METADATA ==========")

for metadata in data["metadatas"][-5:]:

    print(metadata)