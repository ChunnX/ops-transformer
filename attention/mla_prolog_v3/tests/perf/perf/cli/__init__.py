"""perf.cli - command-line interface for performance analysis modes.

The entry point is `perf.cli.main.main()`. Each analysis mode lives in
`perf.cli.modes.<mode_name>` and exposes a single `run(...)` function.
"""

from .main import main, build_hw

__all__ = ["main", "build_hw"]
