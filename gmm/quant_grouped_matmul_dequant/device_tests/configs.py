"""
qgmmd (quant_grouped_matmul_dequant) device test matrix.

Reuses the shape dictionary from talk/qgmmd/test/generate_data.py and runs
each shape across the interesting operator-call variants:

    dynamic       — default path: dynamic per-token quant, fp32 weight scale
    dynamic_u64   — uint64-encoded weight scale (VDEQ16 direct path)
    dynamic_smooth— dynamic + smooth_scale input (fp16 per-K)

Shapes that fail to fit (either N unaligned or a known-blocked combo) are
auto-skipped. LIGHT + HEAVY tiers from generate_data.py are both exercised
on device — simulator-only restrictions don't apply here.
"""
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Walk up from gmm/quant_grouped_matmul_dequant/device_tests/ to repo root,
# then into talk/qgmmd/test/ where the shared shape dictionary +
# data generator live. device_tests here keeps only the device wrapper.
_REPO = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
TEST_DIR = os.path.join(_REPO, "talk", "qgmmd", "test")
sys.path.insert(0, TEST_DIR)
from generate_data import (  # noqa: E402
    LIGHT as _LIGHT_CFGS,
    HEAVY as _HEAVY_CFGS,
    QWEN3 as _QWEN3_CFGS,
)

# Flatten into ordered list used for output-dir prefix + --shape filter.
# LIGHT first, then HEAVY, then QWEN3. Name → (E, M, K, N).
SHAPES_LIGHT = [(n, c["E"], c["M"], c["K"], c["N"]) for n, c in _LIGHT_CFGS.items()]
SHAPES_HEAVY = [(n, c["E"], c["M"], c["K"], c["N"]) for n, c in _HEAVY_CFGS.items()]
SHAPES_QWEN3 = [(n, c["E"], c["M"], c["K"], c["N"]) for n, c in _QWEN3_CFGS.items()]
SHAPES = SHAPES_LIGHT + SHAPES_HEAVY + SHAPES_QWEN3

VARIANTS = ["dynamic", "dynamic_u64", "dynamic_smooth"]

# --------------------------------------------------------------------
# Shape-vs-variant gating. Some variants may not make sense for a given
# shape (e.g. smooth_scale on very small K). Default: all 3 variants per
# shape unless explicitly skipped.
# --------------------------------------------------------------------
SKIP_RULES = [
    # (shape name substring, variant, reason)
]


def is_valid(shape_name, E, M, K, N, variant):
    for sub, var, reason in SKIP_RULES:
        if sub in shape_name and var == variant:
            return False, reason
    return True, None


def build_all_configs(tiers=("LIGHT", "HEAVY", "QWEN3")):
    """Return list of all configs across the requested tiers."""
    pool = []
    if "LIGHT" in tiers:
        pool += SHAPES_LIGHT
    if "HEAVY" in tiers:
        pool += SHAPES_HEAVY
    if "QWEN3" in tiers:
        pool += SHAPES_QWEN3
    configs = []
    for name, E, M, K, N in pool:
        for variant in VARIANTS:
            ok, _ = is_valid(name, E, M, K, N, variant)
            if not ok:
                continue
            configs.append({
                "name": name, "E": E, "M": M, "K": K, "N": N,
                "variant": variant,
            })
    return configs


def config_tag(cfg):
    """Short unique tag per (shape, variant). Used for OPPROF parent dir."""
    return f"{cfg['name']}_{cfg['variant']}"


def data_tag(cfg):
    """Data is keyed by shape only — variants share the same input data.

    generate_data.py writes both the fp32 and uint64 weight scale and
    both pertoken-on and pertoken-off goldens; the variant only picks
    which file run_one.py loads at runtime."""
    return cfg["name"]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true", help="print all configs")
    p.add_argument("--shape", help="filter by shape name")
    p.add_argument("--tier", choices=("LIGHT", "HEAVY", "QWEN3", "ALL"), default="ALL")
    args = p.parse_args()

    tiers = ("LIGHT", "HEAVY", "QWEN3") if args.tier == "ALL" else (args.tier,)
    configs = build_all_configs(tiers=tiers)
    if args.shape:
        configs = [c for c in configs if c["name"] == args.shape]

    if args.list:
        for c in configs:
            print(f"  {config_tag(c):40s} E={c['E']:3d} M={c['M']:5d} "
                  f"K={c['K']:6d} N={c['N']:6d}")
    from collections import Counter
    by_shape = Counter(c["name"] for c in configs)
    for name, *_ in SHAPES:
        if name in by_shape:
            print(f"  {name:15s} {by_shape[name]:2d} variants")
    print(f"  {'TOTAL':15s} {len(configs):3d} configs")
