"""Pure QSLI parameter normalization shared by single and batch tests."""

PARAM_NAMES = (
    "batch_size", "query_length", "sequence_length", "cmp_ratio",
    "storage_mode", "mask_mode", "topk", "candidate_block_length",
    "output_idx_offset", "seed",
)


def normalize_qsli_params(params):
    if isinstance(params, dict):
        values = dict(params)
    else:
        if len(params) != len(PARAM_NAMES):
            raise ValueError(
                f"QSLI parameter count mismatch: got {len(params)}, "
                f"expected {len(PARAM_NAMES)}"
            )
        values = dict(zip(PARAM_NAMES, params))
    normalized = {
        "batch_size": int(values["batch_size"]),
        "query_length": int(values["query_length"]),
        "sequence_length": int(values["sequence_length"]),
        "cmp_ratio": int(values.get("cmp_ratio", 1)),
        "storage_mode": str(values.get("storage_mode", "contiguous")),
        "mask_mode": int(values.get("mask_mode", 3)),
        "topk": int(values.get("topk", 512)),
        "candidate_block_length": values.get("candidate_block_length"),
        "output_idx_offset": values.get("output_idx_offset", True),
        "seed": int(values.get("seed", 1)),
    }
    return normalized
