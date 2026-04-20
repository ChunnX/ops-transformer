"""Entry point for python -m npukernelbench."""

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
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent

    args = parse_args()
    config = load_config()

    all_paths = scan_tasks(root / "tasks")
    selected = filter_by_name(all_paths, args.task_name) if args.task_name else all_paths

    if not selected:
        print("No tasks found. Exiting.")
        return

    print(f"NPUKernelBench v0.1.0 | {len(selected)} operators | stages={config.active_stages}")

    builds = create_build_handles(selected, config)
    tests = expand_test_handles(builds, config)

    pipeline = PipelineRunner(
        stages=[CodeGenStage(config), CompileStage(config)],
        config=config,
    )
    pipeline.run(builds)

    test_stages = []
    if "precision" in config.active_stages:
        test_stages.append(PrecisionStage(config))
    if "perf" in config.active_stages:
        test_stages.append(PerfStage(config))

    if test_stages:
        eligible = [t for t in tests if t.build.compile_result and t.build.compile_result.success]
        if eligible:
            PipelineRunner(stages=test_stages, config=config).run(eligible)

    generate_statistics(builds, config, tests=tests)


if __name__ == "__main__":
    main()
