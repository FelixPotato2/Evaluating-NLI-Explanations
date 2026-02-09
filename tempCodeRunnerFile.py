def calculate_counts(result, key):
    """Return TP/FP/FN totals for debugging/analysis."""
    TP = sum(v[key] for v in result.values() if isinstance(v, dict))
    FP = sum(v["total_LLM_answers"] - v[key] for v in result.values() if isinstance(v, dict))
    FN = sum(v["total_answers"] - v[key] for v in result.values() if isinstance(v, dict))
    return {"TP": TP, "FP": FP, "FN": FN}
