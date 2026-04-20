"""Precision evaluation stage — torch/torch_npu imported lazily."""

from __future__ import annotations

import os
import sys
from typing import Optional

import pandas as pd

from npukernelbench.config.models import FullConfig
from npukernelbench.models.task import TestHandle, BuildHandle
from npukernelbench.models.results import PrecisionResult
from npukernelbench.pipeline.context import PipelineContext
from npukernelbench.stages.precision.comparator import check_precision
from npukernelbench.utils.modules import load_module_from_path
from npukernelbench.utils.logger import init_logger, add_file_handler


class PrecisionStage:
    """Run precision tests comparing reference vs custom operator."""

    name = "precision"

    def __init__(self, config: FullConfig):
        self.config = config

    def execute(self, task: TestHandle, ctx: PipelineContext) -> None:
        import torch
        import torch_npu
        from torch_npu.contrib import transfer_to_npu

        logger = init_logger(f"precision.{task.short_id}")
        os.makedirs(os.path.dirname(task.test_paths.precision_log), exist_ok=True)
        add_file_handler(task.test_paths.precision_log, logger=logger)

        build = task.build
        n_success = 0
        n_failed = 0
        results_dict = []

        try:
            # Setup
            so_dir = os.path.abspath(build.paths.build_dir)
            if so_dir not in sys.path:
                sys.path.append(so_dir)
            _set_custom_opp_path(build)

            device_id = self.config.eval.gpu_devices[0]
            torch.npu.set_device(device_id)
            device = torch.device(f"cuda:{device_id}")

            logger.info(f"Precision test: {task.short_id} on device {device_id}")

            # Load modules
            module = load_module_from_path("model_module", build.paths.module_path)
            if hasattr(module, "get_init_inputs"):
                prepare_inputs = module
            else:
                prepare_inputs = load_module_from_path("prepare_inputs_module", build.paths.prepare_inputs_path)

            # Read test cases
            df = pd.read_csv(build.paths.test_cases_path, sep=";")
            df = _preprocess_dataframe(df)

            # Redirect C++ stdout to log file so [TILING] output is captured
            stdout_redirector = _StdoutRedirector(task.test_paths.precision_log) \
                if self.config.tiling_dump else None
            if stdout_redirector:
                stdout_redirector.start()

            try:
                for i, row in df.iterrows():
                    if self.config.static_shape_mode and row.get("case_id") != task.case_id:
                        continue

                    try:
                        result = self._run_single_case(
                            module, prepare_inputs, row, device, logger, i
                        )
                        if result["success"]:
                            n_success += 1
                        else:
                            n_failed += 1
                        results_dict.append(result)
                    except Exception as e:
                        logger.error(f"Case {i} failed: {e}", exc_info=True)
                        n_failed += 1
            finally:
                if stdout_redirector:
                    stdout_redirector.stop()

            success = n_success > 0 and n_failed == 0
            task.precision_result = PrecisionResult(
                success=success,
                n_success=n_success,
                n_failed=n_failed,
                log_file=task.test_paths.precision_log,
                results_dict=results_dict,
            )
            logger.info(
                f"Precision result: success={success}, n_success={n_success}, n_failed={n_failed}"
            )

        except Exception as e:
            logger.error(f"Precision test failed: {e}", exc_info=True)
            task.precision_result = PrecisionResult(
                success=False, n_success=n_success, n_failed=n_failed,
                log_file=task.test_paths.precision_log,
            )
            logger.info(
                f"Precision result: success=False, n_success={n_success}, n_failed={n_failed}"
            )

    def execute_batch(self, tasks: list[TestHandle], ctx: PipelineContext) -> None:
        for task in tasks:
            self.execute(task, ctx)

    def try_restore(self, task: TestHandle) -> bool:
        """Check disk for existing precision log and restore precision_result.

        Returns True if a previous successful result was found.
        """
        import re
        log_path = task.test_paths.precision_log
        if not os.path.exists(log_path):
            return False

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return False

        # Parse key metrics from log
        success_match = re.search(r"success\s*=\s*(True|False)", content)
        n_success_match = re.search(r"n_success\s*=\s*(\d+)", content)
        n_failed_match = re.search(r"n_failed\s*=\s*(\d+)", content)

        if not success_match:
            return False

        task.precision_result = PrecisionResult(
            success=success_match.group(1) == "True",
            n_success=int(n_success_match.group(1)) if n_success_match else 0,
            n_failed=int(n_failed_match.group(1)) if n_failed_match else 0,
            log_file=log_path,
        )
        return True

    def _run_single_case(self, module, prepare_inputs, row, device, logger, case_idx):
        import torch
        import torch_npu

        init_params = prepare_inputs.get_init_inputs(row, device=device)
        init_params_cpu = _transfer_to_device(init_params, "cpu")
        inputs = prepare_inputs.get_inputs(row, device=device)
        inputs_cpu = _transfer_to_device(inputs, "cpu")

        model = module.Model(*init_params_cpu).to("cpu")
        model_new = module.ModelNew(*init_params).to(device)

        n_acc = 0
        all_abs, all_rel, all_acc = [], [], []

        for trial_idx in range(self.config.eval.num_correct_trials):
            inputs_copy = _deep_copy_tensors(inputs_cpu)
            ref_output = model(*inputs_copy)
            torch_npu.npu.synchronize()

            inputs_copy = _deep_copy_tensors(inputs)
            test_output = model_new(*inputs_copy)
            torch_npu.npu.synchronize()

            test_output = _transfer_to_device(test_output, "cpu")

            if hasattr(prepare_inputs, "custom_check_precision"):
                acc, abs_diff, rel_diff = prepare_inputs.custom_check_precision(row, ref_output, test_output)
            else:
                acc, abs_diff, rel_diff = check_precision(
                    ref_output, test_output,
                    self.config.eval.max_abs_error,
                    self.config.eval.max_rel_error,
                )

            # Log detailed tensor info on first trial for debugging
            if trial_idx == 0:
                _log_tensor_details(logger, inputs_cpu, ref_output, test_output, abs_diff, rel_diff)

            n_acc += acc
            all_acc.append(acc)
            all_abs.append(abs_diff)
            all_rel.append(rel_diff)

        import torch as th
        all_abs_cat = th.cat(all_abs, dim=0)
        all_rel_cat = th.cat(all_rel, dim=0)

        stats = {
            "case_idx": case_idx,
            "success": n_acc == self.config.eval.num_correct_trials,
            "avg_abs_error": all_abs_cat.float().mean().item(),
            "avg_rel_error": all_rel_cat.float().mean().item(),
            "max_abs_error": all_abs_cat.max().item(),
            "max_rel_error": all_rel_cat.max().item(),
            "avg_acc": th.mean(th.Tensor(all_acc).float()).item(),
        }

        logger.info(
            f"Case {case_idx}: avg_abs={stats['avg_abs_error']:.4f}, "
            f"avg_rel={stats['avg_rel_error']:.4f}, "
            f"max_abs={stats['max_abs_error']:.4f}, "
            f"max_rel={stats['max_rel_error']:.4f}, acc={stats['avg_acc']:.2f}"
        )

        return {**dict(row), **stats}


def _set_custom_opp_path(build: BuildHandle) -> None:
    from npukernelbench.stages.compile.verifier import _find_custom_lib_path
    lib_path = _find_custom_lib_path(build.paths.src_dir)
    if lib_path:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = f"{lib_path}/"


def _transfer_to_device(params, device="cpu"):
    import torch
    if params is None:
        return None
    if isinstance(params, torch.Tensor):
        return params.to(device)
    if isinstance(params, torch.nn.Module):
        return params.to(device)
    if isinstance(params, (list, tuple)):
        return type(params)(_transfer_to_device(x, device) for x in params)
    if isinstance(params, dict):
        return {k: _transfer_to_device(v, device) for k, v in params.items()}
    return params


def _deep_copy_tensors(obj):
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, (tuple, list)):
        return type(obj)(_deep_copy_tensors(item) for item in obj)
    if isinstance(obj, dict):
        return {k: _deep_copy_tensors(v) for k, v in obj.items()}
    return obj


def _preprocess_dataframe(df):
    df_copy = df.copy()
    for col in df_copy.columns:
        if pd.api.types.is_numeric_dtype(df_copy[col]) and not df_copy[col].isna().any():
            if df_copy[col].apply(lambda x: float(x) == int(float(x))).all():
                df_copy[col] = df_copy[col].astype(int)
    return df_copy


def _log_tensor_details(logger, inputs, ref_output, test_output, abs_diff, rel_diff):
    """Log inputs, ref_output, test_output tensors and diff stats for debugging."""
    import torch

    SEP = "=" * 80

    def _log_section(title):
        logger.debug(f"\n{SEP}\n  {title}\n{SEP}")

    def _log_one(obj, name):
        if isinstance(obj, (list, tuple)):
            for idx, item in enumerate(obj):
                _log_single(item, f"{name}[{idx}]")
        else:
            _log_single(obj, name)

    def _log_single(obj, name):
        if isinstance(obj, torch.Tensor):
            tensor_to_check = obj.coalesce().values() if obj.is_sparse else obj
            has_non_finite = torch.any(~torch.isfinite(tensor_to_check))
            logger.debug(
                f"  [{name}]  shape={list(obj.shape)}  dtype={obj.dtype}  "
                f"min={obj.float().min().item():.6f}  max={obj.float().max().item():.6f}  "
                f"non_finite={has_non_finite}\n"
                f"{obj}"
            )
        elif hasattr(obj, "shape"):
            logger.debug(f"  [{name}]  shape={obj.shape}\n{obj}")
        else:
            logger.debug(f"  [{name}]  type={type(obj).__name__}  value={obj}")

    _log_section("INPUTS")
    _log_one(inputs, "inputs")
    _log_section("REF OUTPUT (PyTorch)")
    _log_one(ref_output, "ref_output")
    _log_section("TEST OUTPUT (Custom Op)")
    _log_one(test_output, "test_output")
    _log_section("DIFF")
    logger.debug(f"  abs_diff  max={abs_diff.float().max().item():.8f}")
    logger.debug(f"  rel_diff  max={rel_diff.float().max().item():.8f}")
    logger.debug(SEP)


class _StdoutRedirector:
    """Redirect C++ stdout (fd 1) to a log file so printf output is captured."""

    def __init__(self, log_path: str):
        self._log_path = log_path
        self._saved_fd = None

    def start(self):
        self._saved_fd = os.dup(1)
        self._log_fd = os.open(self._log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        os.dup2(self._log_fd, 1)
        os.close(self._log_fd)

    def stop(self):
        if self._saved_fd is not None:
            import ctypes
            sys.stdout.flush()
            ctypes.CDLL(None).fflush(None)
            os.dup2(self._saved_fd, 1)
            os.close(self._saved_fd)
            self._saved_fd = None
