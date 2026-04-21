#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Joint MM1 × MM2 Split-KN search.

The default search_split_kn in perf/tiling/split_kn_search.py evaluates MM1
and MM2 independently (fixing the other at baseline) then combines the best
of each. This misses pareto-optimal joint configs where (e.g.) a slightly
sub-optimal MM1 leaves AIV headroom that lets MM2 use a more aggressive
K-split.

This script enumerates the full Cartesian product of (MM1_spec, MM2_spec)
including the "None" (baseline) option on each side and reports the top
configurations vs the independent best+best combination.

Run:
    python joint_split_kn_search.py [testcase_name]
"""
import sys
from perf.cli.formatting import params_from_testcase
from perf.model.hardware import ASCEND_950
from perf.model.matmul_split_kn import SplitKNSpec
from perf.model.orchestration import estimate_all_stages
from perf.pipeline.dag import build_pipeline_dag
from perf.tiling.config import compute_tiling
from perf.tiling.split_kn_search import _gen_kn_candidates


def joint_search(tc_name):
    params = params_from_testcase(tc_name)
    hw = ASCEND_950
    tiling = compute_tiling(params, hw)
    max_cores = hw.aic_num

    # MM1 K-axis = He, N-axis = Hcq
    mm1_specs = [None]
    for sn, sk, ng, kg in _gen_kn_candidates(params.He, params.Hcq, max_cores):
        mm1_specs.append(SplitKNSpec("MM1_Cq", ng, kg, sn, sk))

    # MM2 K-axis = He, N-axis = Hckv+Dr
    mm2_specs = [None]
    for sn, sk, ng, kg in _gen_kn_candidates(params.He, params.Hckv + params.Dr, max_cores):
        mm2_specs.append(SplitKNSpec("MM2_CkvKr", ng, kg, sn, sk))

    print(f"Testcase: {tc_name}")
    print(f"  MM1 candidates: {len(mm1_specs)-1} + baseline")
    print(f"  MM2 candidates: {len(mm2_specs)-1} + baseline")
    print(f"  Joint space: {len(mm1_specs) * len(mm2_specs)} configs")
    print()

    def lbl(spec, name):
        if spec is None:
            return f"{name}:base"
        return f"{name}:N{spec.n_groups}xK{spec.k_groups}"

    results = []
    for m1 in mm1_specs:
        for m2 in mm2_specs:
            try:
                stages = estimate_all_stages(params, tiling, hw,
                                             mm1_split_kn=m1, mm2_split_kn=m2)
                r = build_pipeline_dag(params, tiling, hw, stages)
                label = f"{lbl(m1,'MM1')} + {lbl(m2,'MM2')}"
                results.append((r.total_us, label, m1, m2,
                                r.aic_busy_us, r.aiv_busy_us))
            except Exception as e:
                pass

    results.sort(key=lambda x: x[0])

    # Independent best-of-each (mimic default search)
    mm1_indep = min(
        [(r[0], r[1]) for r in results if r[2] is not None and r[3] is None],
        default=(None, None))
    mm2_indep = min(
        [(r[0], r[1]) for r in results if r[2] is None and r[3] is not None],
        default=(None, None))
    baseline = next(r for r in results if r[2] is None and r[3] is None)

    print(f"Baseline (no split):               {baseline[0]:.3f} us")
    if mm1_indep[0]:
        print(f"Independent best MM1-only:         {mm1_indep[0]:.3f} us  [{mm1_indep[1]}]")
    if mm2_indep[0]:
        print(f"Independent best MM2-only:         {mm2_indep[0]:.3f} us  [{mm2_indep[1]}]")
    print()

    print(f"  {'Rank':>4} {'Joint config':<42} | {'Pipe':>7} {'AIC':>7} {'AIV':>7}")
    print("  " + "-" * 76)
    for i, (us, label, m1, m2, aic, aiv) in enumerate(results[:15], 1):
        marker = " <--" if i == 1 else ""
        print(f"  {i:>4} {label:<42} | {us:>6.2f}  {aic:>6.2f}  {aiv:>6.2f}{marker}")

    best = results[0]
    savings = baseline[0] - best[0]
    print()
    print(f"Joint best:   {best[1]}  ->  {best[0]:.3f} us  "
          f"(saves {savings:.2f} us / {savings/baseline[0]*100:.1f}% vs baseline)")

    # Check if joint best differs from what the default search would pick
    # Default search picks: best-MM1-only + best-MM2-only
    # We need to see if ANY joint config with BOTH mm1 != None and mm2 != None
    # beats that "combined" config.
    joint_both = [r for r in results if r[2] is not None and r[3] is not None]
    if joint_both:
        joint_both.sort(key=lambda x: x[0])
        print(f"Best joint (both split):  {joint_both[0][1]}  ->  {joint_both[0][0]:.3f} us")


if __name__ == "__main__":
    tcs = sys.argv[1:] if len(sys.argv) > 1 else [
        "perf_decode_bs1", "perf_decode_int8", "perf_decode_bs8"]
    for tc in tcs:
        joint_search(tc)
        print("=" * 80)
