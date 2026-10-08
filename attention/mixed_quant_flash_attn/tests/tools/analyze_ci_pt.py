#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compare CI PT blobs directly from log, without checkout or unsafe unpickling."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import torch


def git_blob(repo, ref, path):
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{ref}:{path}"])


def load_pt(repo, ref, path):
    return torch.load(
        io.BytesIO(git_blob(repo, ref, path)), map_location="cpu", weights_only=True
    )


def input_fingerprint(inputs):
    """Hash semantic inputs, excluding serialization IDs and test-only options."""
    digest = hashlib.sha256()
    keys = (
        "q",
        "k",
        "v",
        "k_descale",
        "v_descale",
        "block_table",
        "cu_seqlens_q",
        "seqused_q",
        "seqused_kv",
        "sinks",
        "attn_mask",
        "softmax_scale",
        "quant_compute_mode",
        "mask_mode",
        "win_left",
        "win_right",
        "layout_q",
        "layout_kv",
        "layout_attn_out",
        "return_softmax_lse",
    )
    for key in keys:
        value = inputs.get(key)
        digest.update(key.encode())
        if torch.is_tensor(value):
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(
                value.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            )
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()


def tensor_metrics(cpu, npu, atol, rtol):
    if cpu.shape != npu.shape:
        raise ValueError(
            f"Shape mismatch: CPU {tuple(cpu.shape)}, NPU {tuple(npu.shape)}"
        )
    a, b = cpu.float(), npu.float()
    finite = torch.isfinite(a) & torch.isfinite(b)
    same = a == b
    delta = torch.where(finite, (a - b).abs(), torch.zeros_like(a))
    # Matching +/-inf is valid for LSE. NaNs (even paired NaNs) are not proof
    # of correctness. Do not let nan reductions hide bad elements.
    failed = (~finite & ~same) | (finite & (delta > atol + rtol * a.abs()))
    return (
        {
            "shape": list(cpu.shape),
            "cpu_dtype": str(cpu.dtype),
            "npu_dtype": str(npu.dtype),
            "total": a.numel(),
            "changed_elements": int((~same).sum()),
            "failed_elements": int(failed.sum()),
            "bad_nonfinite": int((~finite & ~same).sum()),
            "max_abs": float(delta.max()) if delta.numel() else 0.0,
            "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
            "cpu_nonzero": int((a != 0).sum()),
            "npu_nonzero": int((b != 0).sum()),
            "atol": atol,
            "rtol": rtol,
        },
        delta,
        failed,
    )


def worst_rows(cpu, delta, failed, count):
    if cpu.ndim < 2 or cpu.numel() == 0:
        return []
    errors = delta.reshape(-1, cpu.shape[-1])
    failures = failed.reshape_as(errors).sum(-1)
    maximum = errors.amax(-1)
    rows = torch.argsort(maximum, descending=True)[:count]
    return [
        {
            "row": list(torch.unravel_index(row, cpu.shape[:-1])),
            "max_abs": float(maximum[row]),
            "failed_elements": int(failures[row]),
        }
        for row in rows
        if maximum[row] > 0 or failures[row] > 0
    ]


def analyze_run(repo, ref, run, case, top, expected_commit=None):
    root = f"build_verify_logs/{run}"
    fields = dict(
        line.split("=", 1)
        for line in git_blob(repo, ref, f"{root}/SUMMARY.txt").decode().splitlines()
        if "=" in line
    )
    if expected_commit and not fields.get("commit", "").startswith(expected_commit):
        raise ValueError(
            f"Summary tested {fields.get('commit')}, expected {expected_commit}"
        )
    stem = f"{root}/functional_stc_{case}_"
    inputs, cpu, npu = [
        load_pt(repo, ref, f"{stem}{kind}.pt") for kind in ("input", "cpu", "npu")
    ]
    report = {
        "run": run,
        "tested_commit": fields.get("commit"),
        "summary_result": fields.get("result"),
        "case": case,
        "input_fingerprint": input_fingerprint(inputs),
        "layout_q": inputs.get("layout_q"),
        "seqused_kv": inputs["seqused_kv"].tolist()
        if inputs.get("seqused_kv") is not None
        else None,
        "sinks": inputs["sinks"].float().tolist()
        if inputs.get("sinks") is not None
        else None,
        "outputs": {},
    }
    metadata = npu.get("metadata")
    if metadata is not None:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pytests"))
        from core.ci_diagnostics import metadata_details

        report["runtime_metadata"] = metadata_details(metadata)
    else:
        report["runtime_metadata"] = "not recorded; FD coverage unproven by this PT"
    for key in ("attn_out", "softmax_lse"):
        a, b = cpu.get(key), npu.get(key)
        if a is None and b is None:
            continue
        if not torch.is_tensor(a) or not torch.is_tensor(b):
            raise ValueError(f"CPU/NPU output presence mismatch: {key}")
        # Use the existing comparison tolerances, without loosening them.
        atol, rtol = (
            (0.0001, 0.0078125) if a.dtype == torch.bfloat16 else (0.000025, 0.005)
        )
        metrics, delta, failed = tensor_metrics(a, b, atol, rtol)
        metrics["worst_rows"] = worst_rows(a, delta, failed, top)
        report["outputs"][key] = metrics
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--ref", default="origin/log")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="CI timestamp; repeat to compare history",
    )
    parser.add_argument("--case", default="b270_sink_n1_6")
    parser.add_argument(
        "--commit", help="Require the SUMMARY commit to match this prefix"
    )
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Require zero failed elements in every output",
    )
    args = parser.parse_args()
    if args.top < 0:
        parser.error("--top must be nonnegative")
    reports = [
        analyze_run(args.repo, args.ref, run, args.case, args.top, args.commit)
        for run in args.run
    ]
    print(json.dumps(reports, indent=2, default=lambda value: int(value)))
    if args.check and any(
        output["failed_elements"]
        for report in reports
        for output in report["outputs"].values()
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
