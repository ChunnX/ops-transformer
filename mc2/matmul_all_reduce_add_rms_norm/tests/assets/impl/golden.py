#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Golden for aclnnMatmulAllReduceAddRmsNorm.

MatmulAllReduceAddRmsNorm: matmul(x1,x2)+bias -> all_reduce(SUM) -> add_rms_norm(residual, gamma).
"""

import logging
import os
import tempfile
from pathlib import Path
import numpy as np


_next_port_base = [30140]


def _next_port():
    import threading

    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "50000-50100")
    with threading.Lock():
        port = _next_port_base[0]
        _next_port_base[0] += 13
        if _next_port_base[0] > 60000:
            _next_port_base[0] = 30140
        return port


def _save_inputs_per_rank(thread_contexts, device_ids, path):
    import torch

    arrays = {}
    dtypes = {}
    for did in device_ids:
        tc = thread_contexts[did]
        for i, tensor in enumerate(tc.flatten_tensors):
            if tensor is None:
                continue
            dtype_str = str(tensor.dtype).replace("torch.", "") if hasattr(tensor, "dtype") else "numpy"
            if dtype_str in ("torch.bfloat16", "bfloat16"):
                arrays[f"did{did}_t{i}"] = tensor.view(torch.uint16).cpu().numpy()
            elif hasattr(tensor, "cpu") and hasattr(tensor, "numpy"):
                arrays[f"did{did}_t{i}"] = tensor.cpu().numpy()
            else:
                arrays[f"did{did}_t{i}"] = np.asarray(tensor)
            dtypes[f"did{did}_t{i}_dtype"] = dtype_str
    arrays.update(dtypes)
    np.savez(path, **arrays)


def _get_transpose_flag(first_ctx):
    remark = first_ctx.remark or ""
    for part in remark.split(","):
        kv = part.split("=", 1)
        if len(kv) == 2 and kv[0].strip() == "is_trans_b":
            return kv[1].strip() == "1"
    return False


def _run_arn_cascade(thread_contexts, device_ids, is_trans_b, epsilon):
    import subprocess
    import sys
    import torch

    n = len(device_ids)
    if n < 2:
        return {}
    port = _next_port()
    worker_script = str(Path(__file__).with_name("cascade_worker.py"))
    with tempfile.TemporaryDirectory(prefix="ttk_cascade_arn_") as tmpdir:
        input_path = os.path.join(tmpdir, "inputs.npz")
        result_path = os.path.join(tmpdir, "results.npz")
        error_path = os.path.join(tmpdir, "errors.log")
        _save_inputs_per_rank(thread_contexts, device_ids, input_path)
        procs = []
        for rank in range(n):
            cmd = [
                sys.executable,
                worker_script,
                str(rank),
                str(n),
                str(port),
                input_path,
                result_path,
                error_path,
                str(is_trans_b),
                str(epsilon),
            ]
            procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
        for proc in procs:
            proc.wait()
        error_msg = ""
        if os.path.exists(error_path):
            with open(error_path) as f:
                error_msg = f.read()
        for proc in procs:
            if proc.returncode != 0:
                stderr = proc.stderr.read().decode(errors="replace")[-2000:] if proc.stderr else ""
                raise RuntimeError(f"ARN cascade worker exited {proc.returncode}\n{error_msg}\n{stderr}")

        results = {}
        for did in device_ids:
            rank_file = f"{result_path}.did{did}.npz"
            if not os.path.exists(rank_file):
                raise RuntimeError(f"ARN cascade result missing for did={did}")
            data = np.load(rank_file, allow_pickle=False)
            import torch
            y = torch.from_numpy(data["y"].copy())
            norm_out = torch.from_numpy(data["norm_out"].copy())
            results[did] = {"y": y, "norm_out": norm_out}
        return results


def _load_framework():
    """Load golden utils from mc2/common/tests/assets."""
    import importlib.util
    from pathlib import Path

    utils_path = (
        Path(__file__).resolve().parents[4]
        / "common"
        / "tests"
        / "assets"
        / "golden_utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        f"mc2_common_golden_utils_{abs(hash(utils_path))}", utils_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.to_torch_f32, mod.fmt_compare_result, mod.apply_goldens_and_compare


def matmul_all_reduce_add_rms_norm_multi_device_golden(
    thread_contexts, device_ids, all_precision
):
    import torch

    to_torch_f32, fmt_compare, apply_compare = _load_framework()
    first_ctx = next(iter(thread_contexts.values()))
    attrs = first_ctx.attributes
    eps = float(attrs.get("epsilon", 1e-6))

    local_results = {}
    for did in device_ids:
        tc = thread_contexts[did]
        x1 = to_torch_f32(tc.flatten_tensors[0])
        x2 = to_torch_f32(tc.flatten_tensors[1])
        mm_out = torch.matmul(x1, x2)
        bias = tc.flatten_tensors[2] if len(tc.flatten_tensors) > 2 else None
        if bias is not None and isinstance(bias, torch.Tensor) and bias.numel() > 0:
            mm_out = mm_out + to_torch_f32(bias)
        local_results[did] = mm_out

    total = torch.zeros_like(local_results[device_ids[0]])
    for did in device_ids:
        total = total + local_results[did]
    total = total.float()
    del local_results

    # Add RMS Norm: out = all_reduce + residual, then rms_norm
    rank_goldens = {}
    for did in device_ids:
        tc = thread_contexts[did]
        residual = tc.flatten_tensors[3] if len(tc.flatten_tensors) > 3 else None
        gamma = tc.flatten_tensors[4] if len(tc.flatten_tensors) > 4 else None
        out = total.clone()
        if (
            residual is not None
            and isinstance(residual, torch.Tensor)
            and residual.numel() > 0
        ):
            out = out + to_torch_f32(residual)
        if gamma is not None and isinstance(gamma, torch.Tensor) and gamma.numel() > 0:
            gamma_f = to_torch_f32(gamma)
            rms = torch.sqrt(out.pow(2).mean(dim=-1, keepdim=True) + eps)
            out = out / rms * gamma_f
        norm_out = total.clone()
        if (
            residual is not None
            and isinstance(residual, torch.Tensor)
            and residual.numel() > 0
        ):
            norm_out = norm_out + to_torch_f32(residual)
        rank_goldens[did] = {"main": norm_out, "gather": out}
    is_trans_b = _get_transpose_flag(first_ctx)
    rank_third_parties = None
    try:
        cascade_outs = _run_arn_cascade(thread_contexts, device_ids, is_trans_b, eps)
        rank_third_parties = {
            did: [cascade_outs[did]["y"], cascade_outs[did]["norm_out"]]
            for did in device_ids
        }
        logging.info("MatmulAllReduceAddRmsNorm: real HCCL cascade succeeded")
    except Exception:
        logging.exception("MatmulAllReduceAddRmsNorm: real HCCL cascade failed")
        rank_third_parties = None

    apply_compare(
        thread_contexts,
        device_ids,
        rank_goldens,
        all_precision,
        rank_third_parties=rank_third_parties,
    )


def matmul_all_reduce_add_rms_norm_e2e_golden(
    cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True
):
    import torch

    world_size = len(cpu_inputs_per_rank)
    eps = float(attrs.get("epsilon", 1e-6))
    local_results = []
    for r in range(world_size):
        inp = cpu_inputs_per_rank[r]
        x1 = inp[0].float() if len(inp) > 0 and inp[0] is not None else None
        x2 = inp[1].float() if len(inp) > 1 and inp[1] is not None else None
        if x1 is None or x2 is None:
            local_results.append(None)
            continue
        mm_out = torch.matmul(x1, x2)
        local_results.append(mm_out)
    total = None
    for r in range(world_size):
        if local_results[r] is not None:
            total = (
                local_results[r].clone() if total is None else total + local_results[r]
            )
    if total is None:
        return [None] * world_size
    goldens = []
    for r in range(world_size):
        inp = cpu_inputs_per_rank[r]
        out = total.clone()
        residual = inp[3] if len(inp) > 3 and inp[3] is not None else None
        gamma = inp[4] if len(inp) > 4 and inp[4] is not None else None
        if residual is not None and hasattr(residual, "numel") and residual.numel() > 0:
            out = out + residual.float()
        if gamma is not None and hasattr(gamma, "numel") and gamma.numel() > 0:
            gamma_f = gamma.float()
            rms = torch.sqrt(out.pow(2).mean(dim=-1, keepdim=True) + eps)
            out = out / rms * gamma_f
        goldens.append(out.contiguous())
    return goldens
