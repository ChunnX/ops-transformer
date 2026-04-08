#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compat shim for the original pipeline_model.py.

The implementation has been split into the perf.pipeline package. This module
re-exports the public API so existing imports continue to work unchanged.
New code should prefer the package paths:

    from perf.pipeline.dag import build_pipeline_dag
    from perf.pipeline.timeline import format_timeline
"""

from perf.pipeline.dag import (
    PipelineStage, PipelineResult, build_pipeline_dag,
)
from perf.pipeline.kernel_time import estimate_kernel_time
from perf.pipeline.timeline import format_timeline
from perf.pipeline.roofline import (
    RooflineResult, roofline_analysis, format_roofline_report,
)
