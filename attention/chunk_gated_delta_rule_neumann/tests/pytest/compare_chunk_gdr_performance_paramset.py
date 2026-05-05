PREFILL_SHAPES = [
    # B, S, D, Hq, Hkv
    (8, 16, 64, 16, 4),
    (8, 64, 64, 16, 4),
    (8, 125, 64, 16, 4),
    (8, 128, 64, 16, 4),
    (8, 256, 64, 16, 4),
    (8, 512, 64, 32, 8),
    (8, 512, 128, 32, 8),
    (8, 1024, 128, 32, 8),
    (8, 2048, 128, 16, 4),
    (16, 16, 64, 16, 4),
    (16, 64, 64, 16, 4),
    (16, 128, 64, 16, 4),
    (16, 256, 64, 16, 4),
    (16, 512, 64, 32, 8),
    (16, 512, 128, 32, 8),
    (16, 1024, 128, 32, 8),
    (16, 2048, 128, 16, 4),
    (32, 16, 64, 16, 4),
    (32, 64, 64, 16, 4),
    (32, 128, 64, 16, 4),
    (32, 256, 64, 16, 4),
    (32, 512, 64, 32, 8),
    (32, 512, 128, 32, 8),
    (32, 1024, 128, 32, 8),
    (32, 2048, 128, 16, 4),
    (64, 16, 64, 16, 4),
    (64, 64, 64, 16, 4),
    (64, 128, 64, 16, 4),
    (64, 256, 64, 16, 4),
    (64, 512, 64, 32, 8),
    (64, 512, 128, 32, 8),
    (64, 1024, 128, 32, 8),
    (64, 2048, 128, 16, 4),
]


def make_prefill_case(batch_size, seqlen, dim, query_heads, kv_heads, index):
    return {
        "name": f"prefill_b{batch_size}_s{seqlen}_d{dim}_h{query_heads}_hkv{kv_heads}",
        "seed": 20260429 + index,
        "batch_size": batch_size,
        "seqlen": seqlen,
        "nk": kv_heads,
        "nv": query_heads,
        "dk": dim,
        "dv": dim,
    }


PERFORMANCE_CASES = [
    make_prefill_case(batch_size, seqlen, dim, query_heads, kv_heads, index)
    for index, (batch_size, seqlen, dim, query_heads, kv_heads) in enumerate(PREFILL_SHAPES)
]


def case_id(case):
    return case["name"]
