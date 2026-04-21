#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI entry point: argument parsing, hardware spec construction, mode dispatch."""

import argparse
import dataclasses

from ..model.hardware import ASCEND_910B, CHIP_REGISTRY, AscendHWSpec
from ..model.operator_params import OperatorParams
from ..tiling.config import compute_tiling
from .formatting import params_from_testcase
from .modes import (
    advice, block_search, bound, dequant_opt, estimate, pipeline, prefill,
    prefill_l0c_ub, prefill_msplit, report, roofline, search, split_kn,
    split_kn_fused, split_m,
)

_USAGE_EPILOG = """\
Usage:
    python perf_analyzer.py --batch-size 2 --seq-len 4 --head-num 32 --He 7168 --mode full
    python perf_analyzer.py --from-testcase base_default --mode bound
    python perf_analyzer.py --from-testcase base_default --mode pipeline
    python perf_analyzer.py --from-testcase base_default --mode estimate
    python perf_analyzer.py --from-testcase base_default --mode advice
    python perf_analyzer.py --from-testcase base_default --mode search
    python perf_analyzer.py --mode report   # run all perf test cases
    python perf_analyzer.py --chip 950 --from-testcase perf_decode_bs1 --mode roofline
    python perf_analyzer.py --chip 950 --from-testcase perf_decode_bs1 --mode block-search
"""


def build_hw(args) -> AscendHWSpec:
    """Build hardware spec from CLI arguments."""
    hw = CHIP_REGISTRY.get(args.chip, ASCEND_910B)
    # Apply aic/aiv overrides if explicitly different from chip defaults
    overrides = {}
    if args.aic_num != hw.aic_num:
        overrides["aic_num"] = args.aic_num
    if args.aiv_num != hw.aiv_num:
        overrides["aiv_num"] = args.aiv_num
    if overrides:
        hw = dataclasses.replace(hw, **overrides)
    return hw


def main():
    parser = argparse.ArgumentParser(
        description="MLA Prolog V3 Performance Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_USAGE_EPILOG,
    )

    # Parameter sources (mutually exclusive, not required for 'report' mode)
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--from-testcase", type=str, metavar="NAME",
                     help="Load parameters from a named test case in testcases.py")
    src.add_argument("--batch-size", type=int,
                     help="Batch size (requires other params too)")

    # Manual parameters
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--head-num", type=int, default=8)
    parser.add_argument("--He", type=int, default=7168)
    parser.add_argument("--Hcq", type=int, default=1536)
    parser.add_argument("--Hckv", type=int, default=512)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--Dr", type=int, default=64)
    parser.add_argument("--weight-quant-mode", type=int, default=0)
    parser.add_argument("--kv-cache-quant-mode", type=int, default=0)
    parser.add_argument("--query-quant-mode", type=int, default=0)

    # Analysis mode
    parser.add_argument("--mode", type=str, default="full",
                        choices=["bound", "pipeline", "estimate", "advice",
                                 "search", "report", "roofline", "block-search",
                                 "split-kn", "split-kn-fused", "split-m",
                                 "prefill", "prefill-msplit", "prefill-l0c-ub",
                                 "dequant-opt", "full"],
                        help="Analysis mode (default: full = all modes)")

    # Hardware
    parser.add_argument("--chip", type=str, default="910b",
                        choices=list(CHIP_REGISTRY.keys()),
                        help="Target chip (default: 910b)")
    parser.add_argument("--aic-num", type=int, default=None)
    parser.add_argument("--aiv-num", type=int, default=None)

    args = parser.parse_args()

    # Set defaults for aic/aiv based on chip
    chip_default = CHIP_REGISTRY.get(args.chip, ASCEND_910B)
    if args.aic_num is None:
        args.aic_num = chip_default.aic_num
    if args.aiv_num is None:
        args.aiv_num = chip_default.aiv_num

    hw = build_hw(args)

    # 'report' mode uses PERF_CASE_NAMES, no manual params needed
    if args.mode == "report":
        print(f"MLA Prolog V3 Performance Analysis")
        print(f"  HW: {hw.name}, AIC={hw.aic_num}, AIV={hw.aiv_num}")
        report.run(hw)
        return

    # All other modes require a param source
    if not args.from_testcase and args.batch_size is None:
        parser.error("--from-testcase or --batch-size is required for this mode")

    # Build params
    if args.from_testcase:
        params = params_from_testcase(args.from_testcase)
    else:
        params = OperatorParams(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            head_num=args.head_num,
            He=args.He,
            Hcq=args.Hcq,
            Hckv=args.Hckv,
            D=args.D,
            Dr=args.Dr,
            weight_quant_mode=args.weight_quant_mode,
            kv_cache_quant_mode=args.kv_cache_quant_mode,
            query_quant_mode=args.query_quant_mode,
        )

    # Print params summary
    print(f"MLA Prolog V3 Performance Analysis")
    print(f"  B={params.batch_size}, S={params.seq_len}, T={params.T}, N={params.N}, He={params.He}")
    print(f"  Hcq={params.Hcq}, Hckv={params.Hckv}, D={params.D}, Dr={params.Dr}")
    print(f"  weight_quant={params.weight_quant_mode}, kv_quant={params.kv_cache_quant_mode}, "
          f"query_quant={params.query_quant_mode}")
    print(f"  Resolved quant_mode: {params.quant_mode.name}")
    print(f"  HW: {hw.name}, AIC={hw.aic_num}, AIV={hw.aiv_num}")
    if hw.has_cycle_spec:
        print(f"  Freq: {hw.freq_ghz} GHz, Cache: L1={hw.cache.l1_size//1024}KB, "
              f"L0A={hw.cache.l0a_size//1024}KB, L0B={hw.cache.l0b_size//1024}KB, "
              f"L0C={hw.cache.l0c_size//1024}KB")

    if args.mode == "search":
        search.run(params, hw)
        return

    tiling = compute_tiling(params, hw)
    print(f"\nTiling:")
    print(f"  {tiling.summary()}")

    modes = {
        "bound": bound.run,
        "pipeline": pipeline.run,
        "estimate": estimate.run,
        "advice": advice.run,
        "roofline": roofline.run,
        "block-search": block_search.run,
        "split-kn": split_kn.run,
        "split-kn-fused": split_kn_fused.run,
        "split-m": split_m.run,
        "prefill": prefill.run,
        "prefill-msplit": prefill_msplit.run,
        "prefill-l0c-ub": prefill_l0c_ub.run,
        "dequant-opt": dequant_opt.run,
    }

    if args.mode == "full":
        for mode_fn in [bound.run, pipeline.run, estimate.run, advice.run]:
            mode_fn(params, tiling, hw)
        search.run(params, hw)
        if hw.has_cycle_spec:
            roofline.run(params, tiling, hw)
            block_search.run(params, tiling, hw)
            split_kn.run(params, tiling, hw)
            split_m.run(params, tiling, hw)
            prefill.run(params, tiling, hw)
            # Note: dequant_opt is intentionally NOT in `full` mode — adding it
            # would change every existing `--mode full` golden snapshot, breaking
            # the bit-identical regression invariant. Run it explicitly via
            # `--mode dequant-opt`.
    else:
        modes[args.mode](params, tiling, hw)


if __name__ == "__main__":
    main()
