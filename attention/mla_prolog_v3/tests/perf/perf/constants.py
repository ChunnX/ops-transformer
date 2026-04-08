#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Named constants used across the perf model.

These were previously magic numbers scattered across multiple files. The
values are kept exactly as they were before this refactoring; they are
empirically derived and any change requires re-validation against hardware
profiling data.
"""

# Pipeline overlap factor for multi-step kernel time estimation.
# Steady-state steps after the first overlap ~5% with previous step's
# wind-down (estimate_kernel_time uses this in `single_step + (n-1) * single_step * 0.95`).
# Empirical fit; do not change without re-validation.
PIPELINE_OVERLAP_FACTOR = 0.95

# Performance advice thresholds (used by perf.cli.modes.advice).
# These flag the operator as imbalanced and surface a recommendation.
AIV_AIC_BALANCE_LOW = 0.3       # AIV/AIC busy ratio below this -> AIV underutilized
AIV_AIC_BALANCE_HIGH = 3.0      # AIV/AIC ratio above this -> AIC underutilized
SYNC_OVERHEAD_THRESHOLD = 0.15  # Sync time as fraction of total -> flag in advice

# Tiling defaults (used by perf.tiling.config).
DEFAULT_STEP_BATCH_SIZE = 128
MIN_MM1_SINGLE_CORE_N = 64
