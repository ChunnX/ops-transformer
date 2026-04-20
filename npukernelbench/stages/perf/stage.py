"""Performance evaluation stage — torch/torch_npu imported lazily."""

from __future__ import annotations

import os
import sys

import pandas as pd

from npukernelbench.config.models import FullConfig
from npukernelbench.models.task import TestHandle
from npukernelbench.models.results import PerfResult
from npukernelbench.pipeline.context import PipelineContext
from npukernelbench.stages.perf.timer import PerformanceTimer
from npukernelbench.stages.precision.stage import (
    _set_custom_opp_path, _transfer_to_device, _preprocess_dataframe,
)
from npukernelbench.utils.modules import load_module_from_path
from npukernelbench.utils.logger import init_logger, add_file_handler


class PerfStage:
    """Measure kernel performance (latency)."""

    name = "perf"

    def __init__(self, config: FullConfig):
        self.config = config

    def execute(self, task: TestHandle, ctx: PipelineContext) -> None:
        import torch
        import torch_npu
        from torch_npu.contrib import transfer_to_npu

        logger = init_logger(f"perf.{task.short_id}")
        os.makedirs(os.path.dirname(task.test_paths.perf_log), exist_ok=True)
        add_file_handler(task.test_paths.perf_log, logger=logger)

        build = task.build
        n_success = 0
        n_failed = 0
        results_dict = {}

        try:
            so_dir = os.path.abspath(build.paths.build_dir)
            if so_dir not in sys.path:
                sys.path.append(so_dir)
            _set_custom_opp_path(build)

            device_id = self.config.eval.gpu_devices[0]
            torch.npu.set_device(device_id)
            device = torch.device(f"cuda:{device_id}")

            logger.info(f"Perf test: {task.short_id} on device {device_id}")

            module = load_module_from_path("model_module", build.paths.module_path)
            if hasattr(module, "get_init_inputs"):
                prepare_inputs = module
            else:
                prepare_inputs = load_module_from_path("prepare_inputs_module", build.paths.prepare_inputs_path)

            df = pd.read_csv(build.paths.test_cases_path, sep=";")
            df = _preprocess_dataframe(df)

            timer = PerformanceTimer(device)

            for i, row in df.iterrows():
                if self.config.static_shape_mode and row.get("case_id") != task.case_id:
                    continue

                try:
                    init_params = prepare_inputs.get_init_inputs(row, device=device)
                    inputs = prepare_inputs.get_inputs(row, device=device)

                    model = module.Model(*init_params).to(device)
                    model_new = module.ModelNew(*init_params).to(device)

                    avg_model = timer.warmup_and_measure(model, inputs, self.config.eval.num_perf_trials)
                    avg_new = timer.warmup_and_measure(model_new, inputs, self.config.eval.num_perf_trials)

                    n_success += 1
                    results_dict[i] = dict(**dict(row), avg_model_time=avg_model, avg_model_new_time=avg_new)

                    logger.info(f"Case {i}: model={avg_model:.3f}ms, new={avg_new:.3f}ms")

                except Exception as e:
                    n_failed += 1
                    logger.error(f"Case {i} failed: {e}", exc_info=True)

            success = n_failed == 0 and n_success > 0
            task.perf_result = PerfResult(
                success=success, n_success=n_success, n_failed=n_failed,
                log_file=task.test_paths.perf_log, results_dict=results_dict,
            )

        except Exception as e:
            logger.error(f"Perf test failed: {e}", exc_info=True)
            task.perf_result = PerfResult(
                success=False, n_success=n_success, n_failed=n_failed,
                log_file=task.test_paths.perf_log,
            )

    def execute_batch(self, tasks: list[TestHandle], ctx: PipelineContext) -> None:
        for task in tasks:
            self.execute(task, ctx)
