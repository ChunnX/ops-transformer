# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate all output elements with bounded CPU reference memory.

Example: python run_generalization_accuracy.py --phase decode
"""

from __future__ import annotations
import argparse
import dataclasses
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import time

import torch

HERE = Path(__file__).resolve().parent
import fp8_criterion
from attn_prologue_golden import attribute_e4m3_mismatch
from generalization_inputs import (
    SharedWeights,
    all_cases,
    device_input,
    golden_chunks,
    make_case_inputs,
)


def parse_kernel_kwargs(value):
    """Reject obsolete options before importing or initializing the NPU runtime."""
    try:
        kwargs = json.loads(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid kernel kwargs JSON: {exc}") from exc
    if not isinstance(kwargs, dict):
        raise argparse.ArgumentTypeError("kernel kwargs must be a JSON object")
    unknown = kwargs.keys() - {"norm_eps"}
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unsupported kernel kwargs: {', '.join(sorted(unknown))}; only norm_eps is supported. "
            "Remove stage_a_only/max_cores/template; execution is full, with platform cores and automatic tiling"
        )
    eps = kwargs.get("norm_eps", 1e-6)
    try:
        valid = (
            not isinstance(eps, bool)
            and isinstance(eps, (int, float))
            and math.isfinite(eps)
            and eps > 0
        )
    except OverflowError:
        valid = False
    if not valid:
        raise argparse.ArgumentTypeError(
            "norm_eps must be a finite positive real number"
        )
    return kwargs


@dataclasses.dataclass
class Comparison:
    q_elements: int = 0
    q_bad: int = 0
    q_nonfinite: int = 0
    q_max_abs: float = 0.0
    q_max_rel: float = 0.0
    scale_elements: int = 0
    scale_bad: int = 0
    qr_elements: int = 0
    qr_mismatched: int = 0
    qr_tie_break: int = 0
    qr_hard: int = 0
    qr_max_code_distance: int = 0
    kv_elements: int = 0
    kv_mismatched: int = 0
    kv_tie_break: int = 0
    kv_hard: int = 0
    kv_max_code_distance: int = 0
    untouched_elements: int = 0
    untouched_bad: int = 0
    hard_examples: list = dataclasses.field(default_factory=list)
    shadow_reviews: list = dataclasses.field(default_factory=list)
    qr_shadow_reviewed: int = 0
    qr_shadow_remaining_hard: int = 0
    kv_shadow_reviewed: int = 0
    kv_shadow_remaining_hard: int = 0
    qr_adjudicated: int = 0
    qr_value_identical: int = 0
    qr_adjacent_bracket: int = 0
    qr_own_scale_consistent: int = 0
    kv_adjudicated: int = 0
    kv_value_identical: int = 0
    kv_adjacent_bracket: int = 0
    kv_own_scale_consistent: int = 0
    scale_groups_reviewed: int = 0

    @property
    def passed(self):
        return (
            self.q_elements > 0
            and self.q_bad / self.q_elements <= 1e-3
            and self.q_nonfinite == 0
            and self.qr_hard == 0
            and self.kv_hard == 0
            and self.untouched_bad == 0
        )

    def output(self):
        return dict(
            **dataclasses.asdict(self),
            q_bad_rate=self.q_bad / max(1, self.q_elements),
            criterion="fp64_value_domain",
            passed=self.passed,
        )


def compare_chunk(stats, got_q, got_qr, got_scales, got_kv, expected, *, start=0):
    candidates = []
    q, ref = got_q.cpu().float(), expected["q"].float()
    diff = (q - ref).abs()
    finite = torch.isfinite(q)
    stats.q_elements += q.numel()
    stats.q_nonfinite += int((~finite).sum())
    stats.q_bad += int(((diff > 0.02 + 0.02 * ref.abs()) | ~finite).sum())
    stats.q_max_abs = max(
        stats.q_max_abs, float(diff.nan_to_num(posinf=float("inf")).max())
    )
    stats.q_max_rel = max(
        stats.q_max_rel, float((diff / ref.abs().clamp_min(1e-6)).nan_to_num().max())
    )
    scales = got_scales.cpu().contiguous().view(torch.uint8)
    stats.scale_elements += scales.numel()
    stats.scale_bad += int((scales != expected["descale_qr"].view(torch.uint8)).sum())
    scale_diff = fp8_criterion.groups_needing_review(
        scales,
        expected["descale_qr"],
        expected["qr"].shape[1] // fp8_criterion.MX_GROUP,
    )
    stats.scale_groups_reviewed += int(scale_diff.shape[0])
    for name, got, exp, truth in (
        ("qr", got_qr, expected["qr"], expected["qr_normalized"]),
        ("kv", got_kv, expected["kv_bytes"], expected["kv_roped"]),
    ):
        got = got.cpu().contiguous().view(torch.uint8)
        exp = exp.contiguous().view(torch.uint8)
        attr = attribute_e4m3_mismatch(got, exp, truth, boundary_rel=1e-5)
        stats.__dict__[f"{name}_elements"] += attr.total
        stats.__dict__[f"{name}_mismatched"] += attr.mismatched
        setattr(
            stats,
            f"{name}_max_code_distance",
            max(getattr(stats, f"{name}_max_code_distance"), attr.max_code_distance),
        )
        rows_hit = set(torch.nonzero(got != exp, as_tuple=False)[:, 0].tolist())
        if name == "qr":
            rows_hit |= set(scale_diff[:, 0].tolist())
        for row in sorted(rows_hit):
            candidates.append(dict(output=name, row=start + row, local_row=row))
    return candidates


def _adjudicate_rows(
    stats,
    shadow,
    inputs,
    candidates,
    got_qr,
    got_scales,
    got_kv,
    start,
    *,
    norm_eps=1e-6,
):
    """Adjudicate nominated rows against complete FP64 recomputations."""
    for name in ("qr", "kv"):
        local = sorted({c["local_row"] for c in candidates if c["output"] == name})
        if not local:
            continue
        absolute = [start + row for row in local]
        ref_payload, ref_scale, _, truth = shadow.recompute_full(
            inputs, absolute, name, norm_eps=norm_eps
        )
        if name == "qr":
            device_payload = got_qr.cpu().contiguous().view(torch.uint8)[local]
            groups = device_payload.shape[1] // fp8_criterion.MX_GROUP
            got_factor = fp8_criterion.expand_scale(
                got_scales.cpu().contiguous()[local], groups, device_payload.shape[1]
            )
            ref_factor = fp8_criterion.expand_scale(
                ref_scale, groups, device_payload.shape[1]
            )
        else:
            device_payload = got_kv.contiguous().view(torch.uint8)[local]
            got_factor = ref_factor = None
        verdict = fp8_criterion.adjudicate(
            device_payload,
            ref_payload,
            truth,
            got_scale=got_factor,
            ref_scale=ref_factor,
        )
        stats.__dict__[f"{name}_adjudicated"] += verdict.checked
        stats.__dict__[f"{name}_value_identical"] += verdict.value_identical
        stats.__dict__[f"{name}_adjacent_bracket"] += verdict.adjacent_bracket
        stats.__dict__[f"{name}_own_scale_consistent"] += verdict.own_scale_consistent
        stats.__dict__[f"{name}_hard"] += verdict.hard
        stats.__dict__[f"{name}_tie_break"] += verdict.legitimate
        for row, col in verdict.hard_index.tolist():
            if len(stats.hard_examples) >= 16:
                break
            stats.hard_examples.append(
                dict(
                    output=name,
                    row=absolute[row],
                    column=col,
                    got_byte=int(device_payload[row, col]),
                    reference_byte=int(ref_payload[row, col]),
                    fp64_truth=float(truth[row, col]),
                    criterion="fp64_value_domain",
                )
            )


def validate_outputs(
    inputs,
    weights,
    outputs,
    cache_device,
    *,
    row_chunk=256,
    progress=None,
    shadow=None,
    norm_eps=1e-6,
):
    if shadow is None:
        raise ValueError("FP64 reference is required by the value-domain criterion")
    stats = Comparison()
    q, qr, scales = outputs
    d = weights.profile[3]
    # Cache is at most ~128 MiB at requested max T; q remains on device, copied
    # by chunks (~16 MiB for 256 target rows), never as an entire CPU tensor.
    cache = cache_device.cpu().view(-1, d)
    for rows, expected in golden_chunks(
        inputs, weights, row_chunk=row_chunk, norm_eps=norm_eps
    ):
        got_kv = cache[inputs.cache_index[rows]]
        candidates = compare_chunk(
            stats, q[rows], qr[rows], scales[rows], got_kv, expected, start=rows.start
        )
        if candidates:
            _adjudicate_rows(
                stats,
                shadow,
                inputs,
                candidates,
                qr[rows],
                scales[rows],
                got_kv,
                rows.start,
                norm_eps=norm_eps,
            )
        if progress is not None:
            progress(rows.stop, stats)
    untouched = torch.ones(cache.shape[0], dtype=torch.bool)
    untouched[inputs.cache_index] = False
    initial = inputs.kv_cache.view(-1, d)
    stats.untouched_elements = int(untouched.sum()) * d
    stats.untouched_bad = int((cache[untouched] != initial[untouched]).sum())
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("all", "decode", "prefill"), default="all")
    ap.add_argument(
        "--case-id", action="append", help="repeat to select specific logical cases"
    )
    ap.add_argument(
        "--t", type=int, action="append", help="filter requested matrix by T"
    )
    ap.add_argument(
        "--unique-t",
        action="store_true",
        help="only one logical case per T; report is explicitly partial",
    )
    ap.add_argument("--row-chunk", type=int, default=256)
    ap.add_argument("--cpu-threads", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--device", default="npu:0")
    ap.add_argument(
        "--package",
        default="cann_ops_transformer",
        help="installed wheel package, e.g. cann_ops_transformer_custom",
    )
    ap.add_argument(
        "--backend",
        choices=("torch", "prepared"),
        default="torch",
        help="validate the public dispatcher by default, or the reusable DSL launcher",
    )
    ap.add_argument(
        "--kernel-kwargs",
        type=parse_kernel_kwargs,
        default="{}",
        help='JSON object containing only norm_eps, e.g. {"norm_eps": 1e-6}',
    )
    ap.add_argument("--output", type=Path, default=HERE / "results" / "accuracy.jsonl")
    ap.add_argument("--list", action="store_true")
    ap.add_argument(
        "--fp64-shadow",
        action="store_true",
        help="include detailed FP64 review of surviving hard examples",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip passing cases with identical kernel SHA256/criterion/platform core count/kernel kwargs",
    )
    args = ap.parse_args()
    cases = [
        c
        for c in all_cases()
        if (args.phase == "all" or c.phase == args.phase)
        and (not args.case_id or c.case_id in args.case_id)
        and (not args.t or c.T in args.t)
    ]
    if args.unique_t:
        cases = list({c.T: c for c in reversed(cases)}.values())[::-1]
    if not cases:
        ap.error("no matching cases")
    if args.list:
        print(json.dumps([c.record() for c in cases], indent=2))
        return 0
    if args.row_chunk <= 0 or args.cpu_threads <= 0:
        ap.error("row-chunk and cpu-threads must be positive")
    import torch_npu  # noqa: F401
    from cannbotdsl import get_platform_info

    extension = importlib.import_module(
        args.package + ".ops.attention.attn_prologue_dsl"
    )
    kernel = importlib.import_module(extension.__name__ + "._kernel")
    prepare_attn_prologue, to_nz = extension.prepare_attn_prologue, extension.to_nz
    # Keep the runtime context consistent with tensors created on --device.
    # Without this, --device npu:1 launches against the default npu:0 context
    # and may return 107003 with zero-filled outputs.
    torch.npu.set_device(args.device)
    cube_core_num = get_platform_info().cube_core_num
    torch.set_num_threads(args.cpu_threads)
    kernel_path = Path(kernel.__file__).resolve()
    kernel_sha256 = hashlib.sha256(kernel_path.read_bytes()).hexdigest()
    interface_sha256 = hashlib.sha256(
        (kernel_path.parent / "attn_prologue.py").read_bytes()
        + (kernel_path.parent / "__init__.py").read_bytes()
    ).hexdigest()
    kernel_kwargs = args.kernel_kwargs
    norm_eps = kernel_kwargs.get("norm_eps", 1e-6)
    if args.resume and args.output.exists():
        prior = [
            json.loads(line)
            for line in args.output.read_text().splitlines()
            if line.strip()
        ]
        done = {
            r["case_id"]
            for r in prior
            if r.get("passed")
            and r.get("kernel_sha256") == kernel_sha256
            and r.get("interface_sha256") == interface_sha256
            and r.get("backend") == args.backend
            and r.get("criterion") == "fp64_value_domain"
            and r.get("cube_core_num") == cube_core_num
            and r.get("reference_norm_eps", 1e-6) == norm_eps
            and r.get("kernel_kwargs", {}) == kernel_kwargs
        }
        cases = [c for c in cases if c.case_id not in done]
        print(
            f"RESUME remaining_cases={len(cases)} kernel_sha256={kernel_sha256}",
            flush=True,
        )
    weights = SharedWeights()
    dev_weights = weights.device_fields(args.device, to_nz)
    from fp64_shadow import FP64Shadow

    shadow = FP64Shadow(weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_passed = True
    for case in cases:
        if hashlib.sha256(kernel_path.read_bytes()).hexdigest() != kernel_sha256:
            raise RuntimeError(
                "kernel source changed during run; restart to preserve JIT source and validation provenance"
            )
        start_time = time.monotonic()
        print(f"START {case.case_id}", flush=True)
        inputs = make_case_inputs(case, weights)
        cache = inputs.kv_cache.to(args.device)
        values = dict(dev_weights)
        for name in ("x", "descale_x", "rope_sin", "rope_cos", "cache_index"):
            value = getattr(inputs, name)
            values[name] = device_input(name, value, args.device, to_nz)
        values["kv_cache"] = cache
        prepared = None
        if args.backend == "prepared":
            prepared = prepare_attn_prologue(**values, norm_eps=norm_eps)
            print(f"RUN {case.case_id} tiling={prepared.tiling}", flush=True)
            outputs = prepared.run()
        else:
            print(f"RUN {case.case_id} backend=torch", flush=True)
            outputs = extension.attn_prologue(**values, norm_eps=norm_eps)
        torch.npu.synchronize()
        last_progress = [time.monotonic()]

        def progress(rows, stats):
            if time.monotonic() - last_progress[0] >= 30 or rows == case.T:
                print(
                    f"CHECK {case.case_id} rows={rows}/{case.T} q_bad={stats.q_bad} "
                    f"qr_hard={stats.qr_hard} kv_hard={stats.kv_hard}",
                    flush=True,
                )
                last_progress[0] = time.monotonic()

        stats = validate_outputs(
            inputs,
            weights,
            outputs,
            cache,
            row_chunk=args.row_chunk,
            progress=progress,
            shadow=shadow,
            norm_eps=norm_eps,
        )
        if args.fp64_shadow and stats.hard_examples:
            stats.shadow_reviews = shadow.review(
                inputs, stats.hard_examples, norm_eps=norm_eps
            )
            for review in stats.shadow_reviews:
                name = review["output"]
                setattr(
                    stats,
                    f"{name}_shadow_reviewed",
                    getattr(stats, f"{name}_shadow_reviewed") + 1,
                )
                if review["shadow_classification"] == "hard":
                    setattr(
                        stats,
                        f"{name}_shadow_remaining_hard",
                        getattr(stats, f"{name}_shadow_remaining_hard") + 1,
                    )
        record = dict(
            **case.record(),
            **stats.output(),
            kernel_sha256=kernel_sha256,
            elapsed_s=time.monotonic() - start_time,
            row_chunk=args.row_chunk,
            cube_core_num=cube_core_num,
            fp64_shadow_enabled=True,
            fp64_review_dump_enabled=args.fp64_shadow,
            kernel_kwargs=kernel_kwargs,
            reference_norm_eps=norm_eps,
            interface_sha256=interface_sha256,
            backend=args.backend,
            tiling=dataclasses.asdict(prepared.tiling) if prepared else None,
            workspace_bytes=prepared.workspace_bytes if prepared else None,
            full_output_validation=True,
            q_atol=0.02,
            q_rtol=0.02,
            q_max_bad_rate=0.001,
        )
        with args.output.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(
            f"{'PASS' if stats.passed else 'FAIL'} {case.case_id}: "
            f"q_bad={stats.q_bad}/{stats.q_elements}, qr_hard={stats.qr_hard}, "
            f"kv_hard={stats.kv_hard}, scale_bad={stats.scale_bad}, "
            f"untouched_bad={stats.untouched_bad}",
            flush=True,
        )
        all_passed &= stats.passed
        del prepared, outputs, values, cache, inputs
        gc.collect()
        torch.npu.empty_cache()
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
