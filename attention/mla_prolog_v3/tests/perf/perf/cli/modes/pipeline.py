#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`pipeline` mode: AIC/AIV timeline visualization."""

from ...model.hardware import AscendHWSpec
from ...model.operator_params import OperatorParams
from ...pipeline.dag import build_pipeline_dag
from ...pipeline.timeline import format_timeline
from ...tiling.config import TilingConfig
from ..formatting import print_header


def run(params: OperatorParams, tiling: TilingConfig, hw: AscendHWSpec):
    """Mode 2: Timeline visualization."""
    print_header("PIPELINE TIMELINE — AIC/AIV Execution Overlap")

    result = build_pipeline_dag(params, tiling, hw)
    print(format_timeline(result))
