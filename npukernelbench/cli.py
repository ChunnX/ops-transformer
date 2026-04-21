"""CLI argument parsing — no side effects on import."""

from __future__ import annotations

import argparse
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NPUKernelBench: Ascend NPU kernel benchmarking framework"
    )

    # --- Task selection ---
    parser.add_argument(
        "-task_name", "--task-name", nargs="+",
        help="Operator names to test (e.g. Matmul Add Conv2D)",
    )

    # --- Stage selection ---
    parser.add_argument(
        "-stages", "--stages", nargs="+",
        choices=["code_gen", "compile", "precision", "perf"],
        default=["code_gen", "compile", "precision"],
        help="Pipeline stages to execute",
    )

    # --- LLM code generation ---
    parser.add_argument(
        "-chat", "--chat", action="store_true",
        help="Enable LLM-based code generation",
    )

    # --- Shape / template mode ---
    parser.add_argument(
        "-test_mode", "--test-mode", type=str, default=None,
        choices=["static", "dynamic"],
        help="Shape mode: static (per-case) or dynamic",
    )
    parser.add_argument(
        "-template_mode", "--template-mode", type=str, default=None,
        choices=["kernel_only", "full"],
        help="Template mode: kernel_only or full scaffolding",
    )

    # --- Sampling ---
    parser.add_argument(
        "-n_sample", "--n-sample", type=int, default=None,
        help="Number of LLM generation samples per operator",
    )
    parser.add_argument(
        "-n_case", "--n-case", type=int, default=None,
        help="Number of test cases to run (omit or 0 = all from CSV)",
    )

    # --- Paths ---
    parser.add_argument(
        "-run_dir", "--run-dir", type=str, default=None,
        help="Root directory for run outputs",
    )
    parser.add_argument(
        "-config", "--config-path", type=str, default="base_config.yaml",
        help="Path to YAML config file (relative to project root)",
    )

    # --- Tiling dump ---
    parser.add_argument(
        "-tiling_dump", "--tiling-dump", action="store_true",
        help="Auto-inject tiling parameter prints into compiled code",
    )

    # --- Skip operator build ---
    parser.add_argument(
        "-skip_op_build", "--skip-op-build", action="store_true",
        help="Skip operator compilation when artifacts exist, only rebuild pybind",
    )

    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)
