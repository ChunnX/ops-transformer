#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`block-search` mode: optimal baseM/baseN/baseK search per matmul."""

from ...model.hardware import AscendHWSpec
from ...model.operator_params import OperatorParams
from ...tiling.block_search import search_all_matmul_blocks
from ...tiling.config import TilingConfig
from ...tiling.mm2_split_k import search_mm2_split_k
from ..formatting import print_header


def run(params: OperatorParams, tiling: TilingConfig, hw: AscendHWSpec):
    """Search optimal baseM/baseN/baseK for each matmul."""
    print_header("BLOCK SIZE SEARCH — Optimal baseM/baseN/baseK")

    if not hw.has_cycle_spec:
        print("  Block search requires per-cycle HW spec. Use --chip 950.")
        return

    print(f"Parameters: T={tiling.step_batch_size}, He={params.He}, N={params.N}")
    print(f"Cache: L0A={hw.cache.l0a_size//1024}KB, L0B={hw.cache.l0b_size//1024}KB, "
          f"L0C={hw.cache.l0c_size//1024}KB, L1={hw.cache.l1_size//1024}KB")
    print()

    all_results = search_all_matmul_blocks(params, tiling, hw)

    for mm_name in ["MM1_Cq", "MM2_CkvKr", "MM3_QcQr", "MM4_Qn"]:
        results = all_results.get(mm_name, [])
        if not results:
            print(f"  {mm_name}: no valid block configurations found")
            print()
            continue

        print(f"  {mm_name} (top 5 of {len(results)} valid configs):")
        header = f"    {'Rank':>4} {'baseM':>5} {'baseN':>5} {'baseK':>5} {'stepK':>5} | {'Total(us)':>10} {'Bound':<8} {'InnerBound':<10}"
        print(header)
        print("    " + "-" * (len(header) - 4))

        for i, (block, timing) in enumerate(results[:5], 1):
            marker = " <-- best" if i == 1 else ""
            print(f"    {i:>4} {block.baseM:>5} {block.baseN:>5} {block.baseK:>5} {block.stepK:>5} | "
                  f"{timing.total_us:>9.2f} {timing.bound:<8} {timing.inner_bound:<10}{marker}")

        # Best config summary
        best_block, best_timing = results[0]
        print(f"    Optimal: baseM={best_block.baseM}, baseN={best_block.baseN}, "
              f"baseK={best_block.baseK}, stepK={best_block.stepK} "
              f"-> {best_timing.bound}-bound, {best_timing.total_us:.2f} us")
        print()

    # MM2 split-K cross-core comparison
    mm2_sk_results = search_mm2_split_k(params, tiling, hw)
    if mm2_sk_results:
        print("  MM2_CkvKr split-N vs split-K comparison:")
        header = f"    {'Rank':>4} {'Mode':<8} {'K-cores':>7} | {'Cube(us)':>9} {'Accum(us)':>9} {'Total(us)':>10} {'Bound':<8}"
        print(header)
        print("    " + "-" * (len(header) - 4))
        for i, r in enumerate(mm2_sk_results, 1):
            marker = " <-- best" if i == 1 else ""
            bound = r.cube_timing.bound if r.cube_timing else ""
            print(f"    {i:>4} {r.mode:<8} {r.k_cores:>7} | "
                  f"{r.cube_us:>8.2f} {r.accum_us:>8.2f} {r.total_us:>9.2f} {bound:<8}{marker}")
        best = mm2_sk_results[0]
        baseline = next((r for r in mm2_sk_results if r.mode == "split_n"), None)
        if baseline and best.mode == "split_k":
            delta = baseline.total_us - best.total_us
            print(f"    -> split-K ({best.k_cores} cores) saves {delta:.2f} us vs split-N ({baseline.k_cores} cores)")
        elif baseline:
            print(f"    -> split-N ({baseline.k_cores} cores) is already optimal")
        print()

    # Fitting formulas summary
    print("Fitting formulas (for tiling code):")
    qm_label = params.quant_mode.name
    for mm_name in ["MM1_Cq", "MM2_CkvKr", "MM3_QcQr", "MM4_Qn"]:
        results = all_results.get(mm_name, [])
        if results:
            b, t = results[0]
            print(f"  {mm_name} ({qm_label}@{hw.name}): "
                  f"baseM={b.baseM}, baseN={b.baseN}, baseK={b.baseK}, stepK={b.stepK}")
