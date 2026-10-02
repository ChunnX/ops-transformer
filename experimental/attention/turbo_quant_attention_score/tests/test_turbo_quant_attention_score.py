# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Deterministic hardware accuracy and latency tests for the real aclnn API."""

import argparse
import gc
import json
import math
from pathlib import Path
import statistics

import torch
import torch_npu  # noqa: F401

from aclnn_runner import AclnnRunner
from task_golden import CENTROIDS, calc_expect_func


def unpack(packed, bits, dim=128):
    # Independent bitstream decoder; unlike the kernel it uses integer shifts.
    packed = packed.to(torch.int32)
    d = torch.arange(dim)
    byte, shift = d * bits // 8, d * bits % 8
    padded = torch.nn.functional.pad(packed, (0, 1))
    return ((padded[..., byte] | (padded[..., byte + 1] << 8)) >> shift) & (
        (1 << bits) - 1
    )


def make_inputs(
    tokens, length, bits=3, batch=1, heads=32, kv_heads=8, norm_dtype=torch.bfloat16
):
    blocks_per_seq = (length + 127) // 128
    blocks = blocks_per_seq * batch
    query = torch.rand(tokens, heads, 128).mul_(2).sub_(1).bfloat16()
    residual = torch.rand_like(query.float()).mul_(2).sub_(1).bfloat16()
    indices = torch.randint(
        0, 256, (blocks, 128, kv_heads, bits * 16), dtype=torch.uint8
    )
    signs = torch.randint(0, 256, (blocks, 128, kv_heads, 16), dtype=torch.uint8)
    norm = torch.rand(blocks, 128, kv_heads).mul_(1.9).add_(0.1).to(norm_dtype)
    gamma = torch.rand_like(norm.float()).mul_(1.9).add_(0.1).to(norm_dtype)
    table = torch.randperm(blocks).reshape(batch, blocks_per_seq).int()
    lengths = torch.tensor(
        [max(0, length - 17 * b) for b in range(batch)], dtype=torch.int32
    )
    return [query, residual, indices, signs, norm, gamma, table, lengths]


def check(output, inputs, bits):
    q, qr, idx, sign, norm, gamma, table, lengths = inputs
    tokens, heads, _ = q.shape
    kv_heads = idx.shape[2]
    groups = heads // kv_heads
    q = q.float().reshape(tokens, kv_heads, groups, 128)
    qr = qr.float().reshape_as(q)
    centroid = torch.tensor(CENTROIDS[bits])
    max_abs = max_scaled = squared = ref_squared = 0.0
    count = bad = strict_bad = 0
    for start in range(0, output.shape[-1], 512):
        end = min(start + 512, output.shape[-1])
        pos = torch.arange(start, end)
        physical = table[:, pos // 128].long()
        offset = (pos % 128).expand_as(physical)
        primary = centroid[unpack(idx[physical, offset], bits).long()]
        signs = unpack(sign[physical, offset], 1).float().mul_(2).sub_(1)
        n = norm[physical, offset].float()
        g = gamma[physical, offset].float()
        if table.shape[0] == 1:
            main = torch.einsum("thgd,lhd->thgl", q, primary[0])
            corr = torch.einsum("thgd,lhd->thgl", qr, signs[0])
            expected = main * n[0].T[None, :, None, :] + math.sqrt(
                math.pi / 2
            ) / 128 * (corr * g[0].T[None, :, None, :])
            valid = (pos < lengths[0])[None, None, :].expand(tokens, heads, -1)
        else:
            main = torch.einsum("bhgd,blhd->bhgl", q, primary)
            corr = torch.einsum("bhgd,blhd->bhgl", qr, signs)
            expected = main * n.permute(0, 2, 1)[:, :, None, :] + math.sqrt(
                math.pi / 2
            ) / 128 * (corr * g.permute(0, 2, 1)[:, :, None, :])
            valid = (pos[None, :] < lengths[:, None])[:, None, :].expand(-1, heads, -1)
        expected = expected.reshape(tokens, heads, end - start)
        actual = output[..., start:end].cpu()
        if not torch.isneginf(actual[~valid]).all():
            raise AssertionError("masked positions must be negative infinity")
        actual, expected = actual[valid], expected[valid]
        if not torch.isfinite(actual).all():
            raise AssertionError("non-finite valid output")
        if not actual.numel():
            continue
        error = (actual - expected).abs()
        # Mixed tolerance handles scores close to zero after cancellation.
        scaled = error / (1e-5 + 1e-3 * expected.abs())
        max_abs = max(max_abs, error.max().item())
        max_scaled = max(max_scaled, scaled.max().item())
        bad += (scaled > 1).sum().item()
        strict_bad += (error > 1e-3 * expected.abs()).sum().item()
        squared += error.double().square().sum().item()
        ref_squared += expected.double().square().sum().item()
        count += actual.numel()
    result = {
        "elements": count,
        "max_abs_error": max_abs,
        "max_tolerance_ratio": max_scaled,
        "relative_l2_error": math.sqrt(squared / max(ref_squared, 1e-30)),
        "bad_elements": bad,
        "pure_relative_1e3_exceedances": strict_bad,
    }
    if bad:
        raise AssertionError(result)
    return result


def check_task_golden(output, inputs, bits):
    arrays = [
        t.float().numpy() if t.dtype in (torch.float16, torch.bfloat16) else t.numpy()
        for t in inputs
    ]
    expected = torch.from_numpy(calc_expect_func(*arrays, bits)[0])
    max_abs, bad, strict_bad = 0.0, 0, 0
    for start in range(0, output.shape[-1], 512):
        actual = output[..., start : start + 512].cpu()
        ref = expected[..., start : start + 512]
        valid = torch.isfinite(ref)
        if not torch.equal(torch.isneginf(actual), torch.isneginf(ref)):
            raise AssertionError("task golden infinity mask differs")
        actual, ref = actual[valid], ref[valid]
        if not torch.isfinite(actual).all():
            raise AssertionError(
                "task golden output is nonfinite at a finite reference position"
            )
        error = (actual - ref).abs()
        if error.numel():
            max_abs = max(max_abs, error.max().item())
            bad += (error > 1e-5 + 1e-3 * ref.abs()).sum().item()
            strict_bad += (error > 1e-3 * ref.abs()).sum().item()
    result = {
        "task_golden_max_abs_error": max_abs,
        "task_golden_mixed_exceedances": bad,
        "task_golden_pure_relative_exceedances": strict_bad,
    }
    if bad:
        raise AssertionError(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", choices=["smoke", "full", "edge", "invalid"], default="smoke"
    )
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--task-golden", action="store_true")
    parser.add_argument(
        "--perf-only",
        action="store_true",
        help="Skip reference calculation; not an accuracy result",
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(20260923)
    torch.npu.set_device(0)
    runner = AclnnRunner()
    if args.suite == "invalid":
        results = []
        for name in (
            "mse_bits",
            "query_shape",
            "head_dim",
            "gqa",
            "packed_width",
            "norm_shape",
            "negative_length",
            "excess_length",
            "output_shape",
            "query_dtype",
            "mixed_norm_dtype",
        ):
            cpu = make_inputs(1, 128, heads=4, kv_heads=2)
            bits, max_len = 3, 128
            if name == "mse_bits":
                bits = 5
            elif name == "query_shape":
                cpu[1] = cpu[1][:, :2].contiguous()
            elif name == "head_dim":
                cpu[0], cpu[1] = (
                    cpu[0][..., :64].contiguous(),
                    cpu[1][..., :64].contiguous(),
                )
            elif name == "gqa":
                cpu[0], cpu[1] = cpu[0][:, :3].contiguous(), cpu[1][:, :3].contiguous()
            elif name == "packed_width":
                cpu[2] = cpu[2][..., :32].contiguous()
            elif name == "norm_shape":
                cpu[4] = cpu[4][..., :1].contiguous()
            elif name == "negative_length":
                cpu[7].fill_(-1)
            elif name == "excess_length":
                cpu[7].fill_(129)
            elif name == "output_shape":
                max_len = 127
            elif name == "query_dtype":
                cpu[0] = cpu[0].half()
            elif name == "mixed_norm_dtype":
                cpu[4] = cpu[4].half()
            inputs = [t.npu() for t in cpu]
            torch.npu.synchronize()
            try:
                runner(inputs, bits, max_len, host_lengths=True)
            except RuntimeError as error:
                if "GetWorkspaceSize failed" not in str(error):
                    raise
                result = {"case": name, "rejected": True, "error": str(error)}
                print(json.dumps(result), flush=True)
                results.append(result)
            else:
                raise AssertionError(f"invalid case accepted: {name}")
        Path(args.output).write_text(json.dumps(results, indent=2))
        return
    if args.suite == "full":
        cases = [
            (
                c["case_name"],
                c["input_desc"][0]["shape"][0],
                c["output_desc"][0]["shape"][-1],
                3,
                1,
                32,
                8,
                torch.bfloat16,
            )
            for c in json.loads(Path(__file__).with_name("case.json").read_text())
        ]
    elif args.suite == "edge":
        cases = [
            (
                f"bits{b}_batch{batch}_len{length}_{dtype}",
                batch if batch > 1 else 3,
                length,
                b,
                batch,
                4,
                2,
                dtype,
            )
            for b in (2, 3, 4)
            for batch in (1, 3)
            for length in (1, 127, 129)
            for dtype in (torch.bfloat16, torch.float16)
        ]
        cases += [
            ("mha", 1, 33, 3, 1, 4, 4, torch.bfloat16),
            ("gqa8_fallback", 9, 129, 3, 1, 16, 2, torch.bfloat16),
            ("empty_kv", 1, 0, 3, 1, 4, 2, torch.bfloat16),
        ]
    else:
        cases = [("smoke", 1, 128, 3, 1, 4, 2, torch.bfloat16)]
    results = []
    for name, tokens, length, bits, batch, heads, kv_heads, dtype in cases:
        cpu = make_inputs(tokens, max(1, length), bits, batch, heads, kv_heads, dtype)
        if not length:
            cpu[-1].zero_()
        inputs = [t.npu() for t in cpu]
        torch.npu.synchronize()
        allocated_before = torch.npu.memory_allocated()
        torch.npu.reset_peak_memory_stats()
        output, timing = runner(inputs, bits, length)
        call_peak_bytes = torch.npu.max_memory_allocated() - allocated_before
        accuracy = (
            {"accuracy_skipped": True} if args.perf_only else check(output, cpu, bits)
        )
        if args.suite == "smoke":
            host_output, _ = runner(inputs, bits, length, host_lengths=True)
            torch.testing.assert_close(host_output.cpu(), output.cpu(), rtol=0, atol=0)
            inputs64 = [*inputs[:7], inputs[7].long()]
            int64_output, _ = runner(inputs64, bits, length)
            torch.testing.assert_close(int64_output.cpu(), output.cpu(), rtol=0, atol=0)
            del host_output, int64_output, inputs64
            accuracy["host_array_and_int64_tensor_match"] = True
        if args.task_golden and length and not args.perf_only:
            accuracy.update(check_task_golden(output, cpu, bits))
        del output
        times = []
        walls = []
        for _ in range(args.repeats):
            output, timing = runner(inputs, bits, length)
            times.append(timing["device_us"])
            walls.append(timing["wall_us"])
            del output
        result = {
            "case": name,
            "tokens": tokens,
            "kv_len": length,
            "bits": bits,
            "batch": batch,
            "heads": heads,
            "kv_heads": kv_heads,
            "norm_dtype": str(dtype),
            **accuracy,
            "device_us_median": statistics.median(times),
            "device_us_samples": times,
            "wall_us_median": statistics.median(walls),
            "workspace_bytes": timing["workspace_bytes"],
            "npu_peak_call_bytes": call_peak_bytes,
            "packed_key_bytes": cpu[2].numel() + cpu[3].numel() + cpu[4].numel() * 4,
            "fp16_key_bytes": cpu[2].shape[0] * 128 * kv_heads * 128 * 2,
            "output_bytes": tokens * heads * length * 4,
        }
        print(json.dumps(result), flush=True)
        results.append(result)
        Path(args.output).write_text(json.dumps(results, indent=2))
        del inputs, cpu
        gc.collect()
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()
