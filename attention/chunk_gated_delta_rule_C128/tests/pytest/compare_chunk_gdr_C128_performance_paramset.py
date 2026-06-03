PREFILL_SHAPES = [
    # B, S, D, Hq, Hkv
    (8, 1024, 128, 16, 4),
    (8, 2048, 128, 16, 4),
    (8, 4096, 128, 16, 4),
    (8, 8192, 128, 16, 4),
    (8, 10240, 128, 16, 4),
    (8, 12288, 128, 16, 4),
    (8, 16384, 128, 16, 4),
    (8, 20480, 128, 16, 4),
]


def make_prefill_case(batch_size, seqlen, dim, query_heads, kv_heads, index):
    return {
        "name": f"prefill_b{batch_size}_s{seqlen}_d{dim}_h{query_heads}_hkv{kv_heads}",
        "seed": 20260514 + index,
        "batch_size": batch_size,
        "seqlen": seqlen,
        "nk": kv_heads,
        "nv": query_heads,
        "dk": dim,
        "dv": dim,
        "g_mode": "random",
    }


PERFORMANCE_CASES = [
    make_prefill_case(batch_size, seqlen, dim, query_heads, kv_heads, index)
    for index, (batch_size, seqlen, dim, query_heads, kv_heads) in enumerate(PREFILL_SHAPES)
]


def case_id(case):
    return case["name"]
