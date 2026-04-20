"""Immutable configuration models — no global singletons."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class ChatConfig:
    active: bool = False
    model_path: str = ""
    api_model: str = "vllm"
    base_url: str = "http://0.0.0.0:5600/v1/"
    api_max_retries: int = 3
    api_retry_delay: int = 5
    num_processes: int = 16
    timeout: int = 1200
    temperature: float = 0.7
    max_tokens: int = 16384


@dataclass(frozen=True)
class CompileConfig:
    mode: str = "msopgen"
    common_dir: str = "libs/common_include"
    timeout: int = 400
    num_processes: int = 128


@dataclass(frozen=True)
class EvalConfig:
    verbose: bool = True
    num_perf_trials: int = 100
    num_correct_trials: int = 1
    gpu_devices: List[int] = field(default_factory=lambda: [0])
    max_rel_error: float = 0.001
    max_abs_error: float = 0.001
    timeout_perf: int = 60
    timeout_precision: int = 300


@dataclass(frozen=True)
class FullConfig:
    run_dir: str = "runs"
    src_dir: str = "src"
    log_dir: str = "log"
    build_dir: str = "build"
    encoding: str = "utf-8"
    n_sample: int = 1
    n_case: int = 1
    static_shape_mode: bool = True
    kernel_only_mode: bool = True
    show_result_log_file: bool = False
    tiling_dump: bool = False
    skip_op_build: bool = False

    chat: ChatConfig = field(default_factory=ChatConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Computed at load time, not from YAML
    active_stages: List[str] = field(default_factory=lambda: ["compile", "precision"])
