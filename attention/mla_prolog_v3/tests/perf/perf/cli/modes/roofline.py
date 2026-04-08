#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`roofline` mode: per-matmul roofline analysis (requires --chip 950)."""

from ...model.hardware import AscendHWSpec
from ...model.operator_params import OperatorParams
from ...pipeline.roofline import roofline_analysis, format_roofline_report
from ...tiling.config import TilingConfig
from ..formatting import print_header


def run(params: OperatorParams, tiling: TilingConfig, hw: AscendHWSpec):
    """Roofline model showing compute/memory balance for each matmul."""
    print_header("ROOFLINE ANALYSIS")

    if not hw.has_cycle_spec:
        print("  Roofline analysis requires per-cycle HW spec. Use --chip 950.")
        return

    results = roofline_analysis(params, tiling, hw)
    print(format_roofline_report(results, hw))
