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

"""Isolate FD accumulation corruption with saved inputs and metadata-only repartitioning."""

import argparse
import contextlib
import importlib
import io
import json
import random
import sys
import types
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--tag", default="baseline")
p.add_argument("--artifact-dir", type=Path, required=True)
p.add_argument(
    "--package-root",
    type=Path,
    help="Optional path to a cann_ops_transformer Python package directory.",
)
p.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
p.add_argument(
    "--no-lse",
    action="store_true",
    help="Disable returned LSE; FD still needs max/sum staging.",
)
p.add_argument(
    "--check",
    action="store_true",
    help="Exit unsuccessfully if any requested precision check fails.",
)
p.add_argument(
    "--edge-cases",
    action="store_true",
    help="Also check uneven head splits and non-FD short sequences.",
)
a = p.parse_args()
import torch_npu

if a.package_root is not None:
    for module_name, path in [
        ("cann_ops_transformer", a.package_root),
        ("cann_ops_transformer.ops", a.package_root / "ops"),
    ]:
        module = types.ModuleType(module_name)
        module.__path__ = [str(path)]
        sys.modules[module_name] = module
name = "sparse_flash_mla"
registration = importlib.import_module("cann_ops_transformer.ops.attention." + name)
for key in (name, name + "_metadata"):
    setattr(sys.modules["cann_ops_transformer.ops"], key, getattr(registration, key))
sys.path.insert(0, str(HERE.parent))
process = importlib.import_module("batch.sparse_flash_mla_process")
golden = importlib.import_module("sparse_flash_mla_golden")
compare = importlib.import_module("result_compare_method")
torch.set_num_threads(4)
torch_npu.npu.set_device(0)
current = {}
origmeta = torch.ops.cann_ops_transformer.sparse_flash_mla_metadata
origop = torch.ops.cann_ops_transformer.sparse_flash_mla


def meta(*args, **kwargs):
    assert torch_npu.npu._get_deterministic_level() == 0
    result = origmeta(*args, **kwargs)
    torch.npu.synchronize()
    current["metadata"] = result.detach().cpu().reshape(-1).clone()
    return result


def kernel(*args, **kwargs):
    assert torch_npu.npu._get_deterministic_level() == 0
    return origop(*args, **kwargs)


torch.ops.cann_ops_transformer.sparse_flash_mla = kernel


def execute(data):
    return process.call_npu(data)


def metric(expected, actual):
    if expected is None:
        return dict(status="NotRequested")
    expected = expected.detach().cpu()
    actual = actual.detach().cpu()
    assert expected.shape == actual.shape, (expected.shape, actual.shape)
    with contextlib.redirect_stdout(io.StringIO()):
        result = compare.check_result(expected, actual)
    error = (expected.float() - actual.float()).abs()
    return dict(
        status=result[0],
        fulfill=float(result[1]),
        max_abs=float(error.max()),
        mean_abs=float(error.mean()),
        nan=int(actual.isnan().sum()),
        inf=int(actual.isinf().sum()),
    )


out = a.artifact_dir / a.tag
out.mkdir(parents=True, exist_ok=True)
inputs = a.artifact_dir / "inputs"
if a.dtype != "bfloat16":
    inputs = inputs / a.dtype
inputs.mkdir(parents=True, exist_ok=True)
summary = []
forced_parts = None


def repartition_meta(*args, **kwargs):
    md = meta(*args, **kwargs)
    if forced_parts is None:
        return md
    assert kwargs["batch_size"] == kwargs["max_seqlen_q"] == 1
    assert sum(forced_parts) * 128 == kwargs["max_seqlen_ori_kv"]
    # Single BSND row, G=64, no split-G: use two FA cores and 64 one-head FD tasks.
    host = md.cpu().clone()
    flat = host.reshape(-1)
    flat[:900] = 0
    cut = forced_parts[0]
    flat[:9] = torch.tensor([1, 0, 0, 0, 0, 0, cut, 0, 0], dtype=flat.dtype)
    flat[9:18] = torch.tensor([1, 0, 0, cut, 1, 0, 0, 1, 0], dtype=flat.dtype)
    for head in range(64):
        flat[324 + 8 * head : 332 + 8 * head] = torch.tensor(
            [1, 0, 0, 0, 2, head, 1, 0], dtype=flat.dtype
        )
    current["metadata"] = flat.clone()
    return host.to(md.device)


torch.ops.cann_ops_transformer.sparse_flash_mla_metadata = repartition_meta
cases = [
    (1, 1, 512, (2, 2)),
    (1, 1, 512, (3, 1)),
    (1, 1, 8192, None),
    (1, 1, 12288, None),
    (1, 2, 4096, None),
    (2, 1, 4096, None),
    (2, 2, 4096, None),
]
cases = [(*case, 64, True) for case in cases]
if a.edge_cases:
    cases += [
        (1, 1, 12288, None, heads, True) for heads in (1, 7, 31, 33, 63, 65, 127, 128)
    ]
    cases += [(1, 1, length, None, 64, False) for length in (1, 127, 128)]
for b, q, length, forced_parts, heads, expect_fd in cases:
    name = f"swa_b{b}_q{q}_kv{length}"
    if heads != 64:
        name += f"_heads{heads}"
    path = inputs / (name + ".pt")
    if forced_parts is not None:
        name += "_parts" + "_".join(map(str, forced_parts))
    if path.exists():
        data = torch.load(path, weights_only=False)
    else:
        params = json.loads((HERE / "base_params.json").read_text())
        for key in ("q_type", "ori_kv_type", "cmp_kv_type"):
            params[key] = getattr(torch, a.dtype)
        params.update(
            Testcase_Name=name,
            testcase_name=name,
            B=b,
            S1=q,
            S2=length,
            T1=b * q,
            T2=b * length,
            N1=heads,
            seqused_q=[q] * b,
            cu_seqlens_q=[i * q for i in range(b + 1)],
            seqused_ori_kv=[length] * b,
            cu_seqlens_ori_kv=[i * length for i in range(b + 1)],
            block_num1=b * ((length + 127) // 128),
            ori_mask_mode=0,
        )
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        with contextlib.redirect_stdout(io.StringIO()):
            data = golden.gen_data(params, prepare_device_storage=False)
        torch.save(data, path)
    data["params"]["return_softmax_lse"] = not a.no_lse
    torch_npu.npu.set_deterministic_level(0)
    with contextlib.redirect_stdout(io.StringIO()):
        actual, lse = execute(data)
    torch.npu.synchronize()
    md = current["metadata"]
    fa = md[:324].reshape(36, 9)
    fd = md[324:900].reshape(72, 8)
    rec = dict(
        case=name,
        dtype=a.dtype,
        heads=heads,
        expected_fd=expect_fd,
        return_softmax_lse=not a.no_lse,
        forced_parts=forced_parts,
        output=metric(data["cpu_output"], actual),
        lse=metric(None if a.no_lse else data["softmax_lse"], lse),
        fa=[v.tolist() for v in fa if v[0]],
        fd=[v.tolist() for v in fd if v[0]],
    )
    assert bool(rec["fd"]) == expect_fd, "Unexpected FD activation"
    error = (
        (data["cpu_output"].float() - actual.cpu().float())
        .abs()
        .reshape(-1, heads, 512)
    )
    rec["head_max_abs"] = error.amax(dim=(0, 2)).tolist()
    torch.save(
        dict(
            input=data,
            metadata=md,
            output=actual.cpu(),
            lse=lse.cpu() if lse is not None else None,
        ),
        out / (name + ".pt"),
    )
    summary.append(rec)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("RESULT", name, rec["output"], rec["lse"], "FA", len(rec["fa"]), flush=True)

print(
    "TOTAL",
    json.dumps(
        dict(
            count=len(summary),
            output_pass=sum(r["output"]["status"] == "Pass" for r in summary),
            lse_pass=sum(r["lse"]["status"] == "Pass" for r in summary),
        )
    ),
)
if a.check and not all(
    r["output"]["status"] == "Pass" and r["lse"]["status"] in ("Pass", "NotRequested")
    for r in summary
):
    raise SystemExit(1)
