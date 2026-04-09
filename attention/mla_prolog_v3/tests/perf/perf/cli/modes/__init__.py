"""perf.cli.modes - one module per analysis mode.

Each module exposes a single `run(...)` function as its entry point.
The dispatcher in perf.cli.main maps `--mode` strings to these `run` callables.
"""

from . import (
    bound, pipeline, estimate, advice, search, report,
    roofline, block_search, split_kn, split_kn_fused, split_m, prefill,
    dequant_opt, prefill_msplit,
)

__all__ = [
    "bound", "pipeline", "estimate", "advice", "search", "report",
    "roofline", "block_search", "split_kn", "split_kn_fused", "split_m",
    "prefill", "dequant_opt", "prefill_msplit",
]
