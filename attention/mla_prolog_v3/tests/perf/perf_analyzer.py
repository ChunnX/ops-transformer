#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI entry point for MLA Prolog V3 performance analysis.

The implementation has been split into the perf.cli package. This module is
now a thin wrapper that delegates to perf.cli.main.main(). Run with --help
for usage details.

    python perf_analyzer.py --batch-size 2 --seq-len 4 --head-num 32 --He 7168 --mode full
    python perf_analyzer.py --from-testcase base_default --mode bound
    python perf_analyzer.py --mode report
    python perf_analyzer.py --chip 950 --batch-size 1 --head-num 128 --mode split-kn
"""

from perf.cli.main import main

if __name__ == "__main__":
    main()
