#!/usr/bin/env python3
"""Single-operator test runner.

Usage:
    python scripts/run_test.py -task_name Sqrt -stages compile precision
    python scripts/run_test.py -task_name Sqrt -stages precision          # re-use compiled artifacts
    python scripts/run_test.py -task_name Add Gelu -stages code_gen compile precision -chat
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from npukernelbench.cli import parse_args
from npukernelbench.config.loader import load_config
from npukernelbench.discovery.scanner import scan_tasks, filter_by_name
from npukernelbench.discovery.factory import create_build_handles, expand_test_handles
from npukernelbench.pipeline.runner import PipelineRunner
from npukernelbench.stages.code_gen.stage import CodeGenStage
from npukernelbench.stages.compile.stage import CompileStage
from npukernelbench.stages.precision.stage import PrecisionStage
from npukernelbench.stages.perf.stage import PerfStage
from npukernelbench.statistics.reporter import generate_statistics


def main():
    args = parse_args()
    config = load_config()

    # Discover tasks
    tasks_dir = ROOT / "tasks"
    all_paths = scan_tasks(tasks_dir)
    print(f"Discovered {len(all_paths)} operators")

    if args.task_name:
        selected = filter_by_name(all_paths, args.task_name)
    else:
        selected = all_paths

    if not selected:
        print("No tasks found. Exiting.")
        sys.exit(1)

    print(f"Selected {len(selected)} operators")
    print(f"Config: stages={config.active_stages}, n_case={config.n_case}")
    print(f"Static shape: {config.static_shape_mode}, Kernel only: {config.kernel_only_mode}")

    # Create handles
    builds = create_build_handles(selected, config)
    tests = expand_test_handles(builds, config)

    # Build pipeline stages (always include all — inactive ones auto-skip
    # and try_restore from disk)
    build_stages = [
        CodeGenStage(config),
        CompileStage(config),
    ]

    pipeline = PipelineRunner(stages=build_stages, config=config)
    pipeline.run(builds)

    # Test stages (precision + perf)
    test_stages = []
    if "precision" in config.active_stages:
        test_stages.append(PrecisionStage(config))
    if "perf" in config.active_stages:
        test_stages.append(PerfStage(config))

    if test_stages:
        eligible = [
            t for t in tests
            if t.build.compile_result and t.build.compile_result.success
        ]
        if eligible:
            test_pipeline = PipelineRunner(stages=test_stages, config=config)
            test_pipeline.run(eligible)

    # Statistics (pass both builds and tests for complete picture)
    generate_statistics(builds, config, tests=tests)


if __name__ == "__main__":
    main()
