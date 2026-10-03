"""Performance cases: required P-01 to P-07 and supplemental shapes.

Tuple layout: (name, seq, kv_heads, batch, dtype, is_causal)
Default heads: 32 (unless specified by case, e.g. X-02 has 16)
Default dim: 128 (unless specified by case, e.g. X-04 has 64)
"""

from typing import Dict, NamedTuple


class BenchmarkCase(NamedTuple):
    name: str
    seq: int
    kv_heads: int
    batch: int
    dtype: str
    is_causal: bool
    heads: int = 32
    dim: int = 128
    tier: str = "S1"


# P-01 to P-07 are the required benchmark cases defined in task/06_benchmark.md
CASES = [
    ("P-01", 4096, 32, 1, "fp16", False),
    ("P-02", 8192, 32, 1, "fp16", False),
    ("P-03", 8192, 32, 1, "fp16", True),
    ("P-04", 8192, 32, 2, "fp16", False),
    ("P-05", 4096, 8, 1, "fp16", False),
    ("P-06", 4096, 32, 1, "bf16", False),
    ("P-07", 16384, 32, 1, "fp16", False),
]

CASE_TIERS = {
    "P-01": "S1",
    "P-02": "S1",
    "P-03": "S2",
    "P-04": "S1",
    "P-05": "S3",
    "P-06": "S4",
    "P-07": "S5",
}

ALL_CASES = CASES + [
    ("P-08", 32768, 32, 1, "fp16", False),
    ("P-09", 65536, 32, 1, "fp16", False),
    ("P-10", 131072, 32, 1, "fp16", False),
    ("X-01", 2048, 32, 1, "fp16", False),
    ("X-02", 4096, 16, 1, "fp16", False),
    ("X-03", 4096, 4, 1, "fp16", False),
    ("X-04", 4096, 32, 1, "fp16", False),
    ("X-05", 2048, 32, 4, "fp16", False),
    ("X-06", 8192, 32, 1, "bf16", False),
    ("X-07", 8192, 8, 1, "fp16", True),
]


def get_case(name: str) -> BenchmarkCase:
    """Find a benchmark case by name and return BenchmarkCase with full metadata."""
    for entry in ALL_CASES:
        if entry[0] == name:
            c_name, seq, kv_heads, batch, dtype, is_causal = entry
            heads = 16 if name == "X-02" else (4 if name == "X-03" else 32)
            dim = 64 if name == "X-04" else 128
            tier = CASE_TIERS.get(name, "Extended")
            return BenchmarkCase(
                name=c_name,
                seq=seq,
                kv_heads=kv_heads,
                batch=batch,
                dtype=dtype,
                is_causal=is_causal,
                heads=heads,
                dim=dim,
                tier=tier,
            )
    raise KeyError(f"Case '{name}' not found in benchmark cases")


CASES_DICT: Dict[str, BenchmarkCase] = {
    c[0]: get_case(c[0]) for c in ALL_CASES
}


if __name__ == "__main__":
    print(f"Total benchmark cases: {len(ALL_CASES)} (Required: {len(CASES)})")
    for case_tuple in ALL_CASES:
        c = get_case(case_tuple[0])
        print(
            f"Case {c.name:<5} | Tier: {c.tier:<3} | Shape: [{c.batch}, {c.heads}, {c.seq}, {c.dim}] | "
            f"KV_Heads: {c.kv_heads:<2} | Dtype: {c.dtype:<4} | Causal: {c.is_causal}"
        )
