import re


def make_quality_decision(evaluation):

    # Extract overall score
    match = re.search(
        r"OVERALL SCORE:\s*(\d+(?:\.\d+)?)\s*/\s*5",
        evaluation,
        re.IGNORECASE
    )

    # Reject if score is missing
    if not match:
        return {
            "decision": "REJECT",
            "reason": "Could not extract a valid quality score."
        }

    score = float(match.group(1))

    # Validate score range
    if score < 1 or score > 5:
        return {
            "decision": "REJECT",
            "reason": "Quality score is outside the valid range."
        }

    # Make final decision
    if score >= 4:
        return {
            "decision": "ACCEPT",
            "reason": "Answer quality is good."
        }

    elif score >= 3:
        return {
            "decision": "WARNING",
            "reason": "Answer quality is average. Review recommended."
        }

    else:
        return {
            "decision": "REJECT",
            "reason": "Answer quality is too low."
        }