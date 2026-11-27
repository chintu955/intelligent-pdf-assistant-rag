import re


def validate_citations(answer, sources):

    # Find citations in the answer
    pattern = r"\[Source:\s*(S\d+)\s*,\s*Page:\s*(\d+)\s*\]"

    citations = re.findall(pattern, answer)

    # Check whether citations exist
    if not citations:
        return {
            "valid": False,
            "message": "No source citations found."
        }

    # Create a dictionary of available sources
    source_map = {}

    for source in sources:

        source_id = source.get("id")
        page = source.get("page")

        if source_id is not None:
            source_map[source_id] = str(page)

    # Validate every citation
    for source_id, page in citations:

        if source_id not in source_map:

            return {
                "valid": False,
                "message": f"Invalid source ID: {source_id}"
            }

        if source_map[source_id] != page:

            return {
                "valid": False,
                "message": (
                    f"Page mismatch for {source_id}. "
                    f"Expected page {source_map[source_id]}, "
                    f"but answer cited page {page}."
                )
            }

    return {
        "valid": True,
        "message": "All source citations are valid."
    }