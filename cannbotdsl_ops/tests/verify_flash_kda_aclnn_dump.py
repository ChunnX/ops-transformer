#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Compare one FlashKDA ACLNN dump with the checked-in CPU chunk golden."""

from __future__ import annotations

import argparse
import importlib.util
import math
import pathlib
import sys

import numpy as np
import torch


HEAD_DIM = 128
_METADATA_KEYS = {
    "schema",
    "batch",
    "heads",
    "sequence",
    "head_dim",
    "scale",
    "lower_bound",
}


def _load_module(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("flash_kda_dump_goldens", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load golden module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_metadata(path: pathlib.Path) -> dict[str, int | float]:
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key in entries:
            raise ValueError(f"invalid FlashKDA dump metadata line: {line!r}")
        entries[key] = value
    if set(entries) != _METADATA_KEYS:
        raise ValueError(
            f"metadata keys differ: expected {sorted(_METADATA_KEYS)}, "
            f"got {sorted(entries)}"
        )
    integer_keys = ("schema", "batch", "heads", "sequence", "head_dim")
    result: dict[str, int | float] = {key: int(entries[key]) for key in integer_keys}
    result.update(
        scale=float(entries["scale"]),
        lower_bound=float(entries["lower_bound"]),
    )
    if (
        result["schema"] != 1
        or result["batch"] <= 0
        or result["heads"] <= 0
        or result["sequence"] <= 0
        or result["sequence"] % 64 != 0
        or result["head_dim"] != HEAD_DIM
        or not math.isfinite(result["scale"])
        or result["scale"] <= 0
        or not math.isfinite(result["lower_bound"])
        or not -5.0 <= result["lower_bound"] <= 0.0
    ):
        raise ValueError(f"unsupported FlashKDA dump metadata: {result}")
    return result


def _checked_memmap(path: pathlib.Path, dtype, shape: tuple[int, ...]) -> np.memmap:
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(
            f"{path}: expected {expected} bytes for shape {shape}, got {actual}"
        )
    return np.memmap(path, mode="r", dtype=dtype, shape=shape)


def _slice_bf16(storage: np.ndarray, batch: int, head: int) -> torch.Tensor:
    copied = np.array(storage[batch : batch + 1, head : head + 1], copy=True)
    return torch.from_numpy(copied).view(torch.bfloat16)


def _slice_float(storage: np.ndarray, batch: int, head: int) -> torch.Tensor:
    copied = np.array(storage[batch : batch + 1, head : head + 1], copy=True)
    return torch.from_numpy(copied)


def _all_finite_bf16(storage: np.memmap) -> bool:
    flat = storage.reshape(-1)
    chunk = 16 * 1024 * 1024
    return all(
        np.all((flat[offset : offset + chunk] & 0x7F80) != 0x7F80)
        for offset in range(0, flat.size, chunk)
    )


def _all_finite_float(storage: np.memmap) -> bool:
    flat = storage.reshape(-1)
    chunk = 8 * 1024 * 1024
    return all(
        np.isfinite(flat[offset : offset + chunk]).all()
        for offset in range(0, flat.size, chunk)
    )


def _sample(text: str) -> tuple[int, int]:
    batch, separator, head = text.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError("sample must have B:H form")
    try:
        result = int(batch), int(head)
    except ValueError as error:
        raise argparse.ArgumentTypeError("sample must have integer B:H form") from error
    if result[0] < 0 or result[1] < 0:
        raise argparse.ArgumentTypeError("sample indices must be non-negative")
    return result


def _resolve_samples(
    batch: int,
    heads: int,
    selected: list[tuple[int, int]],
    all_samples: bool,
) -> list[tuple[int, int]]:
    if selected and all_samples:
        raise ValueError("--golden-sample and --all-golden-samples are exclusive")
    samples = (
        [(batch_index, head) for batch_index in range(batch) for head in range(heads)]
        if all_samples
        else selected or [(0, 0)]
    )
    if len(samples) != len(set(samples)):
        raise ValueError("golden samples must be unique")
    for sample_batch, sample_head in samples:
        if sample_batch >= batch or sample_head >= heads:
            raise ValueError(
                f"sample {sample_batch}:{sample_head} outside B{batch}/N{heads}"
            )
    return samples


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
    parser.add_argument(
        "--golden-sample",
        action="append",
        type=_sample,
        default=[],
        metavar="B:H",
        help="batch/head slice checked against CPU golden; repeatable (default 0:0)",
    )
    parser.add_argument(
        "--all-golden-samples",
        action="store_true",
        help="sequentially check every batch/head slice with O(S*D) peak memory",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="print aggregate CPU golden metrics instead of every sample",
    )
    parser.add_argument("--verify-native", action="store_true")
    arguments = parser.parse_args()

    dump = arguments.dump_dir
    metadata = _load_metadata(dump / "metadata.txt")
    batch = int(metadata["batch"])
    heads = int(metadata["heads"])
    sequence = int(metadata["sequence"])
    scale = float(metadata["scale"])
    lower_bound = float(metadata["lower_bound"])
    activation = (batch, heads, sequence, HEAD_DIM)
    state = (batch, heads, HEAD_DIM, HEAD_DIM)
    golden_path = (
        arguments.cannbotdsl_source
        / "examples/kimi_delta_attention/flash_kda/test/kda_goldens.py"
    )
    goldens = _load_module(golden_path)

    kernel_initial_state = _checked_memmap(
        dump / "initial_state.f32", np.float32, state
    )
    k = _checked_memmap(dump / "k.bf16", np.uint16, activation)
    v = _checked_memmap(dump / "v.bf16", np.uint16, activation)
    q = _checked_memmap(dump / "q.bf16", np.uint16, activation)
    beta = _checked_memmap(dump / "beta.bf16", np.uint16, (batch, heads, sequence, 1))
    g = _checked_memmap(dump / "g.bf16", np.uint16, activation)
    a_log = _checked_memmap(dump / "a_log.f32", np.float32, (heads,))
    dt_bias = _checked_memmap(dump / "dt_bias.f32", np.float32, (heads, HEAD_DIM))
    actual_out = _checked_memmap(dump / "out.bf16", np.uint16, activation)
    actual_state = _checked_memmap(dump / "final_state.f32", np.float32, state)

    if not _all_finite_bf16(actual_out) or not _all_finite_float(actual_state):
        raise AssertionError("ACLNN output contains Inf or NaN")
    print(f"shape=B{batch}/N{heads}/S{sequence}; full ACLNN output/state finite")
    samples = _resolve_samples(
        batch, heads, arguments.golden_sample, arguments.all_golden_samples
    )
    max_output_absolute = 0.0
    max_output_relative = 0.0
    max_state_absolute = 0.0
    max_state_relative = 0.0
    for sample_batch, sample_head in samples:
        sample_initial_state = _slice_float(
            kernel_initial_state, sample_batch, sample_head
        )
        sample_k = _slice_bf16(k, sample_batch, sample_head)
        sample_v = _slice_bf16(v, sample_batch, sample_head)
        sample_q = _slice_bf16(q, sample_batch, sample_head)
        sample_beta = _slice_bf16(beta, sample_batch, sample_head)
        sample_g = _slice_bf16(g, sample_batch, sample_head)
        sample_a_log = torch.from_numpy(
            np.array(a_log[sample_head : sample_head + 1], copy=True)
        )
        sample_dt_bias = torch.from_numpy(
            np.array(dt_bias[sample_head : sample_head + 1], copy=True)
        )
        inputs = goldens.KdaInputs(
            K=sample_k,
            V=sample_v,
            Q=sample_q,
            beta_brc=sample_beta,
            g=sample_g,
            initial_state=sample_initial_state.transpose(-1, -2).contiguous(),
            scale_value=scale,
            A_log=sample_a_log,
            dt_bias=sample_dt_bias,
            lower_bound=lower_bound,
        )
        expected_out, expected_state = goldens.compute_golden("cpu_chunk", inputs)
        sample_actual_out = _slice_bf16(actual_out, sample_batch, sample_head).float()
        sample_actual_state = _slice_float(actual_state, sample_batch, sample_head)
        output_metrics = _metrics(sample_actual_out, expected_out)
        state_metrics = _metrics(sample_actual_state, expected_state)
        max_output_absolute = max(max_output_absolute, output_metrics[0])
        max_output_relative = max(max_output_relative, output_metrics[1])
        max_state_absolute = max(max_state_absolute, state_metrics[0])
        max_state_relative = max(max_state_relative, state_metrics[1])
        if not arguments.summary_only:
            print(
                f"sample={sample_batch}:{sample_head} "
                f"output max_abs={output_metrics[0]:.6e} "
                f"rel={output_metrics[1]:.6e}; "
                f"state max_abs={state_metrics[0]:.6e} "
                f"rel={state_metrics[1]:.6e}"
            )
        torch.testing.assert_close(
            sample_actual_out,
            expected_out.float(),
            atol=arguments.tolerance,
            rtol=arguments.tolerance,
        )
        torch.testing.assert_close(
            sample_actual_state,
            expected_state.float(),
            atol=arguments.tolerance,
            rtol=arguments.tolerance,
        )
        if arguments.verify_native:
            _verify_native(
                arguments,
                inputs,
                sample_actual_out,
                sample_actual_state,
                expected_out,
                expected_state,
                sample_batch,
                sample_head,
            )
    print(
        f"validated_samples={len(samples)}; "
        f"output max_abs={max_output_absolute:.6e} "
        f"max_rel={max_output_relative:.6e}; "
        f"state max_abs={max_state_absolute:.6e} "
        f"max_rel={max_state_relative:.6e}"
    )
    return 0


def _verify_native(
    arguments,
    inputs,
    actual_out,
    actual_state,
    expected_out,
    expected_state,
    sample_batch: int,
    sample_head: int,
) -> None:
    import torch_npu  # noqa: F401

    implementation = _load_module(
        arguments.cannbotdsl_source
        / "examples/kimi_delta_attention/flash_kda/flash_kda.py"
    )
    native_out, native_state = implementation.flash_kda(
        inputs.Q.npu(),
        inputs.K.npu(),
        inputs.V.npu(),
        inputs.g.npu(),
        inputs.beta_brc.squeeze(-1).npu(),
        inputs.scale_value,
        inputs.initial_state.transpose(-1, -2).contiguous().npu(),
        inputs.A_log.npu(),
        inputs.dt_bias.npu(),
        inputs.lower_bound,
        "BNSD",
    )
    torch.npu.synchronize()
    native_out = native_out.cpu().float()
    native_state = native_state.cpu().float()
    native_output_metrics = _metrics(native_out, expected_out)
    native_state_metrics = _metrics(native_state, expected_state)
    aclnn_native_output = _metrics(actual_out, native_out)
    aclnn_native_state = _metrics(actual_state, native_state)
    print(
        f"sample={sample_batch}:{sample_head} "
        f"native/golden output max_abs={native_output_metrics[0]:.6e}; "
        f"state max_abs={native_state_metrics[0]:.6e}; "
        f"aclnn/native output max_abs={aclnn_native_output[0]:.6e}; "
        f"state max_abs={aclnn_native_state[0]:.6e}"
    )
    torch.testing.assert_close(
        native_out,
        expected_out.float(),
        atol=arguments.tolerance,
        rtol=arguments.tolerance,
    )
    torch.testing.assert_close(
        native_state,
        expected_state.float(),
        atol=arguments.tolerance,
        rtol=arguments.tolerance,
    )
    torch.testing.assert_close(actual_out.float(), native_out, atol=0.0, rtol=0.0)
    torch.testing.assert_close(actual_state, native_state, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    raise SystemExit(main())
