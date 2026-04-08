#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`search` mode: stepBatchSize search ranking by estimated kernel time."""

from ...model.hardware import AscendHWSpec
from ...model.operator_params import OperatorParams
from ...tiling.config import search_best_tiling
from ..formatting import print_header


def run(params: OperatorParams, hw: AscendHWSpec):
    """Mode 5: Best tiling search."""
    print_header("TILING SEARCH — Best Configuration")

    candidates = [8, 16, 32, 64, 128]
    results = search_best_tiling(params, hw, step_candidates=candidates)

    print(f"Parameters: T={params.T}, He={params.He}, N={params.N}")
    print(f"Searching stepBatchSize in {candidates}...")
    print()

    header = f"{'Rank':>4} {'Step':>6} {'MM1c':>5} {'MM2c':>5} {'MM3c':>5} {'MM4c':>5} {'Est(us)':>10} {'GroupOpt':>8} {'DeqOpt':>7}"
    print(header)
    print("-" * len(header))

    for i, (cfg, est_us) in enumerate(results, 1):
        marker = " <-- best" if i == 1 else ""
        print(f"{i:>4} {cfg.step_batch_size:>6} {cfg.mm1_block_num:>5} {cfg.mm2_block_num:>5} "
              f"{cfg.mm3_block_num:>5} {cfg.mm4_block_num:>5} {est_us:>10.2f} "
              f"{'Y' if cfg.enable_group_compute_opt else 'N':>8} "
              f"{'Y' if cfg.enable_dequant_opt else 'N':>7}{marker}")

    if results:
        print()
        best_cfg, best_us = results[0]
        print(f"Best: stepBatchSize={best_cfg.step_batch_size}, estimated {best_us:.2f} us")
        print()
        print("Tiling details:")
        print(best_cfg.summary())
