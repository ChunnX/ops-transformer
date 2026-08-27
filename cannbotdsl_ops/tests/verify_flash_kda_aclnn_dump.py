#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Compare one FlashKDA ACLNN dump with the checked-in CPU chunk golden."""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import torch


BATCH = 1
HEADS = 3
SEQUENCE = 8192
HEAD_DIM = 128
SCALE = 0.0883883476
LOWER_BOUND = -1.0


def _load_module(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("flash_kda_dump_goldens", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load golden module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _checked_array(path: pathlib.Path, dtype, shape: tuple[int, ...]) -> np.ndarray:
    result = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape))
    if result.size != expected:
        raise ValueError(
            f"{path}: expected {expected} elements for shape {shape}, "
            f"got {result.size}"
        )
    return result.reshape(shape)


def _load_bf16(path: pathlib.Path, shape: tuple[int, ...]) -> torch.Tensor:
    storage = _checked_array(path, np.uint16, shape)
    return torch.from_numpy(storage.copy()).view(torch.bfloat16)


def _load_float(path: pathlib.Path, shape: tuple[int, ...]) -> torch.Tensor:
    storage = _checked_array(path, np.float32, shape)
    return torch.from_numpy(storage.copy())


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    difference = (actual.float() - expected.float()).abs()
    max_absolute = difference.max().item()
    denominator = expected.float().abs().max().clamp_min(1.0e-30)
    return max_absolute, (difference.max() / denominator).item()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--cannbotdsl-source",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[3] / "cannbot-dsl",
    )
    parser.add_argument("--tolerance", type=float, default=5.0e-3)
    parser.add_argument("--verify-native", action="store_true")
    arguments = parser.parse_args()

    dump = arguments.dump_dir
    activation = (BATCH, HEADS, SEQUENCE, HEAD_DIM)
    state = (BATCH, HEADS, HEAD_DIM, HEAD_DIM)
    golden_path = (
        arguments.cannbotdsl_source
        / "examples/kimi_delta_attention/flash_kda/test/kda_goldens.py"
    )
    goldens = _load_module(golden_path)

    kernel_initial_state = _load_float(dump / "initial_state.f32", state)
    k = _load_bf16(dump / "k.bf16", activation)
    v = _load_bf16(dump / "v.bf16", activation)
    q = _load_bf16(dump / "q.bf16", activation)
    beta = _load_bf16(dump / "beta.bf16", (BATCH, HEADS, SEQUENCE, 1))
    g = _load_bf16(dump / "g.bf16", activation)
    a_log = _load_float(dump / "a_log.f32", (HEADS,))
    dt_bias = _load_float(dump / "dt_bias.f32", (HEADS, HEAD_DIM))
    inputs = goldens.KdaInputs(
        K=k, V=v, Q=q, beta_brc=beta, g=g,
        initial_state=kernel_initial_state.transpose(-1, -2).contiguous(),
        scale_value=SCALE,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=LOWER_BOUND,
    )
    expected_out, expected_state = goldens.compute_golden("cpu_chunk", inputs)
    actual_out = _load_bf16(dump / "out.bf16", activation)
    actual_state = _load_float(dump / "final_state.f32", state)

    if not torch.isfinite(actual_out.float()).all() or not torch.isfinite(actual_state).all():
        raise AssertionError("ACLNN output contains Inf or NaN")
    output_metrics = _metrics(actual_out, expected_out)
    state_metrics = _metrics(actual_state, expected_state)
    print(
        f"output max_abs={output_metrics[0]:.6e} rel={output_metrics[1]:.6e}; "
        f"state max_abs={state_metrics[0]:.6e} rel={state_metrics[1]:.6e}"
    )
    torch.testing.assert_close(
        actual_out.float(), expected_out.float(),
        atol=arguments.tolerance, rtol=arguments.tolerance,
    )
    torch.testing.assert_close(
        actual_state, expected_state.float(),
        atol=arguments.tolerance, rtol=arguments.tolerance,
    )
    if arguments.verify_native:
        import torch_npu  # noqa: F401

        implementation = _load_module(
            arguments.cannbotdsl_source
            / "examples/kimi_delta_attention/flash_kda/flash_kda.py"
        )
        native_out, native_state = implementation.flash_kda(
            q.npu(), k.npu(), v.npu(), g.npu(), beta.squeeze(-1).npu(),
            SCALE, kernel_initial_state.npu(), a_log.npu(), dt_bias.npu(),
            LOWER_BOUND, "BNSD",
        )
        torch.npu.synchronize()
        native_out = native_out.cpu().float()
        native_state = native_state.cpu().float()
        native_output_metrics = _metrics(native_out, expected_out)
        native_state_metrics = _metrics(native_state, expected_state)
        aclnn_native_output = _metrics(actual_out, native_out)
        aclnn_native_state = _metrics(actual_state, native_state)
        print(
            f"native/golden output max_abs={native_output_metrics[0]:.6e}; "
            f"state max_abs={native_state_metrics[0]:.6e}; "
            f"aclnn/native output max_abs={aclnn_native_output[0]:.6e}; "
            f"state max_abs={aclnn_native_state[0]:.6e}"
        )
        torch.testing.assert_close(
            native_out, expected_out.float(),
            atol=arguments.tolerance, rtol=arguments.tolerance,
        )
        torch.testing.assert_close(
            native_state, expected_state.float(),
            atol=arguments.tolerance, rtol=arguments.tolerance,
        )
        torch.testing.assert_close(actual_out.float(), native_out, atol=0.0, rtol=0.0)
        torch.testing.assert_close(actual_state, native_state, atol=0.0, rtol=0.0)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
