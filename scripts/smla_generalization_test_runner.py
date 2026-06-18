#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Generalization runner for SparseFlashMla PR validation.
# The runner intentionally imports only the two SMLA torch extension modules to
# avoid triggering unrelated JIT extension builds from cann_ops_transformer.ops.

import argparse
import contextlib
import importlib.util
import io
import json
import math
import os
import pathlib
import random
import sys
import time
import traceback
import types
from dataclasses import dataclass, asdict, replace
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch_npu


def debug_stage(name: str) -> None:
    if os.environ.get("SMLA_DEBUG_STAGES"):
        print(json.dumps({"debug_stage": name, "time": time.time()}, ensure_ascii=False), flush=True)


HEAD_RATIOS = [1, 2, 4, 8, 16, 32, 64, 128]
LAYOUT_QS = ["BSND", "TND"]
LAYOUT_KVS = ["BSND", "TND", "PA_BBND"]
SCFA_TOPKS = [512, 1024]
DEFAULT_D = 512
PASS_PERCENT = 99.5
RTOL = 0.0078125
ATOL = 0.0001


@dataclass
class CaseSpec:
    case_id: int
    template_mode: str
    layout_q: str
    layout_kv: str
    head_ratio: int
    cmp_ratio: Optional[int]
    topk: Optional[int]
    q_lens: List[int]
    ori_lens: List[int]
    sparse_mode: str
    topk_mode: str
    q_stride_pad: int
    ori_stride_pad: int
    cmp_stride_pad: int
    cmp_store_pad: int
    cmp_pad_nan: bool
    return_softmax_lse: bool
    scenario: str


def load_module(module_name: str, module_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def import_smla_ops(repo_root: pathlib.Path) -> None:
    package_root = repo_root / "torch_extension" / "cann_ops_transformer"

    pkg = types.ModuleType("cann_ops_transformer")
    pkg.__file__ = str(package_root / "__init__.py")
    pkg.__path__ = [str(package_root)]
    sys.modules["cann_ops_transformer"] = pkg

    op_builder_pkg = types.ModuleType("cann_ops_transformer.op_builder")
    op_builder_pkg.__path__ = [str(package_root / "op_builder")]
    sys.modules["cann_ops_transformer.op_builder"] = op_builder_pkg

    ops_pkg = types.ModuleType("cann_ops_transformer.ops")
    ops_pkg.__path__ = [str(package_root / "ops")]
    sys.modules["cann_ops_transformer.ops"] = ops_pkg

    load_module("cann_ops_transformer.op_builder.builder", package_root / "op_builder" / "builder.py")
    load_module("cann_ops_transformer.ops.sparse_flash_mla", package_root / "ops" / "sparse_flash_mla.py")


def import_pytest_modules(repo_root: pathlib.Path):
    pytest_dir = repo_root / "attention" / "sparse_flash_mla" / "tests" / "pytest"
    if str(pytest_dir) not in sys.path:
        sys.path.insert(0, str(pytest_dir))
    return load_module("sparse_flash_mla_golden", pytest_dir / "sparse_flash_mla_golden.py")


def prefix_sum(lengths: Iterable[int]) -> List[int]:
    out = [0]
    for item in lengths:
        out.append(out[-1] + int(item))
    return out


def ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def to_npu_tensor(tensor: Optional[torch.Tensor], device: torch.device, pad_last_dim: int = 0) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if pad_last_dim <= 0 or tensor.dim() != 4:
        return tensor.to(device)

    shape = list(tensor.shape)
    row_logical_size = shape[1] * shape[2] * shape[3]
    stride0 = row_logical_size + pad_last_dim
    strides = (stride0, shape[2] * shape[3], shape[3], 1)
    physical_numel = stride0 * shape[0]
    storage_tensor = torch.full((physical_numel,), float("nan"), dtype=tensor.dtype, device=device)
    view = torch.empty((0,), dtype=tensor.dtype, device=device)
    view.set_(storage_tensor.untyped_storage(), 0, tuple(shape), strides)
    src = tensor.to(device)
    for row_idx in range(shape[0]):
        view[row_idx].copy_(src[row_idx])
    torch.npu.synchronize()
    return view


def extend_cmp_kv_store(input_data: Dict, pad_rows: int, fill_nan: bool) -> None:
    if pad_rows <= 0:
        return
    cmp_kv = input_data["input"].get("cmp_kv")
    if cmp_kv is None:
        return

    layout_kv = input_data["layout_kv"]
    fill_value = float("nan") if fill_nan else 0.0

    if layout_kv == "BSND":
        new_shape = (cmp_kv.shape[0], cmp_kv.shape[1] + pad_rows, cmp_kv.shape[2], cmp_kv.shape[3])
        new_cmp = torch.empty(new_shape, dtype=cmp_kv.dtype)
        new_cmp[:, : cmp_kv.shape[1], :, :].copy_(cmp_kv)
        new_cmp[:, cmp_kv.shape[1] :, :, :].fill_(fill_value)
        input_data["input"]["cmp_kv"] = new_cmp
    elif layout_kv == "TND" and input_data["B"] == 1:
        new_shape = (cmp_kv.shape[0] + pad_rows, cmp_kv.shape[1], cmp_kv.shape[2])
        new_cmp = torch.empty(new_shape, dtype=cmp_kv.dtype)
        new_cmp[: cmp_kv.shape[0], :, :].copy_(cmp_kv)
        new_cmp[cmp_kv.shape[0] :, :, :].fill_(fill_value)
        input_data["input"]["cmp_kv"] = new_cmp
    elif layout_kv == "PA_BBND":
        new_shape = (cmp_kv.shape[0] + pad_rows, cmp_kv.shape[1], cmp_kv.shape[2], cmp_kv.shape[3])
        new_cmp = torch.empty(new_shape, dtype=cmp_kv.dtype)
        new_cmp[: cmp_kv.shape[0], :, :, :].copy_(cmp_kv)
        new_cmp[cmp_kv.shape[0] :, :, :].fill_(fill_value)
        input_data["input"]["cmp_kv"] = new_cmp


def inspect_cp_sparse_first_rows(input_data: Dict, row_count: int = 3) -> Optional[Dict]:
    cmp_sparse_indices = input_data["input"].get("cmp_sparse_indices")
    if cmp_sparse_indices is None:
        return None

    layout_q = input_data["layout_q"]
    rows = []
    if layout_q == "BSND":
        valid_rows = min(row_count, cmp_sparse_indices.shape[1])
        for row_idx in range(valid_rows):
            row = cmp_sparse_indices[0, row_idx, 0, :]
            rows.append({"input_row": row_idx, "valid_count": int((row != -1).sum().item())})
    elif layout_q == "TND":
        cu_q = input_data["input"]["cu_seqlens_q"]
        valid_rows = min(row_count, int(cu_q[1].item() - cu_q[0].item()))
        base = int(cu_q[0].item())
        for row_idx in range(valid_rows):
            row = cmp_sparse_indices[base + row_idx, 0, :]
            rows.append({"input_row": base + row_idx, "valid_count": int((row != -1).sum().item())})
    else:
        return None

    return {
        "layout_q": layout_q,
        "rows": rows,
        "all_first_rows_have_valid_cmp": bool(rows and all(item["valid_count"] > 0 for item in rows)),
    }


def metadata_and_run(input_data: Dict, device: torch.device, spec: CaseSpec, cmp_ratio_override: Optional[int] = None):
    metadata_input = input_data["metadata_input"]
    tensor_input = input_data["input"]
    layout_q = tensor_input["layout_q"] if isinstance(tensor_input["layout_q"], str) else tensor_input["layout_q"][0]
    layout_kv = tensor_input["layout_kv"]
    cmp_ratio = metadata_input["cmp_ratio"] if cmp_ratio_override is None else cmp_ratio_override
    if cmp_ratio is None:
        cmp_ratio = 1

    q = to_npu_tensor(tensor_input["q"], device, spec.q_stride_pad)
    ori_kv = to_npu_tensor(tensor_input["ori_kv"], device, spec.ori_stride_pad)
    cmp_kv = to_npu_tensor(tensor_input["cmp_kv"], device, spec.cmp_stride_pad)

    def move(key):
        value = tensor_input.get(key)
        return value.to(device) if isinstance(value, torch.Tensor) else None

    cu_seqlens_q = move("cu_seqlens_q")
    cu_seqlens_ori_kv = move("cu_seqlens_ori_kv")
    cu_seqlens_cmp_kv = move("cu_seqlens_cmp_kv")
    seqused_q = move("seqused_q")
    seqused_ori_kv = None if layout_kv == "TND" else move("seqused_ori_kv")
    seqused_cmp_kv = move("seqused_cmp_kv")
    cmp_residual_kv = move("cmp_residual_kv")
    ori_sparse_indices = move("ori_sparse_indices")
    cmp_sparse_indices = move("cmp_sparse_indices")
    ori_block_table = move("ori_block_table")
    cmp_block_table = move("cmp_block_table")
    sinks = tensor_input["sinks"].to(device)

    ori_topk_length = metadata_input["ori_topk_length"]
    if ori_topk_length is not None:
        ori_topk_length = ori_topk_length.unsqueeze(-1).to(device)
    cmp_topk_length = metadata_input["cmp_topk_length"]
    if cmp_topk_length is not None:
        cmp_topk_length = cmp_topk_length.unsqueeze(-1).to(device)

    topk = metadata_input["K"] if metadata_input["K"] is not None else 0
    ori_topk = topk if ori_sparse_indices is not None else 0
    debug_stage("metadata_begin")
    metadata = torch.ops.cann_ops_transformer.sparse_flash_mla_metadata(
        num_heads_q=metadata_input["N1"],
        num_heads_kv=metadata_input["N2"],
        head_dim=metadata_input["D"],
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        ori_topk_length=ori_topk_length,
        cmp_topk_length=cmp_topk_length,
        batch_size=metadata_input["B"],
        max_seqlen_q=metadata_input["max_seqlen_q"],
        max_seqlen_ori_kv=metadata_input["max_seqlen_ori_kv"],
        max_seqlen_cmp_kv=metadata_input["max_seqlen_cmp_kv"],
        ori_topk=ori_topk,
        cmp_topk=topk if cmp_sparse_indices is not None else 0,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=metadata_input["ori_mask_mode"],
        cmp_mask_mode=metadata_input["cmp_mask_mode"] if metadata_input["cmp_mask_mode"] is not None else 3,
        ori_win_left=metadata_input["ori_win_left"],
        ori_win_right=metadata_input["ori_win_right"],
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=ori_kv is not None,
        has_cmp_kv=cmp_kv is not None,
    )
    debug_stage("metadata_done")

    debug_stage("sparse_flash_mla_begin")
    out, softmax_lse = torch.ops.cann_ops_transformer.sparse_flash_mla(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q if layout_q == "TND" else None,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        ori_topk_length=ori_topk_length,
        cmp_topk_length=cmp_topk_length,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=tensor_input["softmax_scale"],
        cmp_ratio=cmp_ratio,
        ori_mask_mode=tensor_input["ori_mask_mode"],
        cmp_mask_mode=tensor_input["cmp_mask_mode"] if tensor_input["cmp_mask_mode"] is not None else 3,
        ori_win_left=tensor_input["ori_win_left"],
        ori_win_right=tensor_input["ori_win_right"],
        layout_q=layout_q,
        layout_kv=layout_kv,
        topk_value_mode=1,
        return_softmax_lse=bool(input_data["params"].get("return_softmax_lse", False)),
    )
    debug_stage("sparse_flash_mla_done")
    torch.npu.synchronize()
    return out.detach().cpu(), None if softmax_lse is None else softmax_lse.detach().cpu()


def compare_tensors(actual: torch.Tensor, expected: torch.Tensor) -> Dict:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = actual_f - expected_f
    finite = torch.isfinite(actual_f).all()
    close = torch.isclose(actual_f, expected_f, rtol=RTOL, atol=ATOL, equal_nan=False)
    close_count = int(close.sum().item())
    total = actual.numel()
    close_percent = close_count * 100.0 / total if total else 100.0
    denom = torch.maximum(expected_f.abs(), torch.tensor(1e-12, dtype=torch.float32))
    rel = diff.abs() / denom
    return {
        "finite": bool(finite.item()),
        "nan_count": int(torch.isnan(actual_f).sum().item()),
        "inf_count": int(torch.isinf(actual_f).sum().item()),
        "elements": int(total),
        "close_elements": close_count,
        "close_percent": close_percent,
        "pass_percent": PASS_PERCENT,
        "rtol": RTOL,
        "atol": ATOL,
        "max_abs_diff": float(torch.nan_to_num(diff.abs(), nan=0.0, posinf=0.0, neginf=0.0).max().item()) if total else 0.0,
        "max_rel_diff": float(torch.nan_to_num(rel, nan=0.0, posinf=0.0, neginf=0.0).max().item()) if total else 0.0,
    }


def get_valid_q_mask(input_data: Dict) -> Optional[torch.Tensor]:
    if input_data["layout_q"] != "BSND":
        return None
    seqused_q = input_data["input"].get("seqused_q")
    if seqused_q is None:
        return None

    q = input_data["input"]["q"]
    mask = torch.zeros((q.shape[0], q.shape[1]), dtype=torch.bool)
    for batch_idx, q_len in enumerate(seqused_q.tolist()):
        mask[batch_idx, : int(q_len)] = True
    return mask


def compare_by_batch(actual: torch.Tensor, expected: torch.Tensor, input_data: Dict) -> List[Dict]:
    summaries = []
    if input_data["layout_q"] == "BSND":
        q = input_data["input"]["q"]
        seqused_q = input_data["input"].get("seqused_q")
        for batch_idx in range(q.shape[0]):
            q_len = int(seqused_q[batch_idx].item()) if seqused_q is not None else q.shape[1]
            cmp = compare_tensors(actual[batch_idx, :q_len], expected[batch_idx, :q_len])
            summaries.append({"batch": batch_idx, "q_len": q_len, **cmp})
    else:
        cu_q = input_data["input"]["cu_seqlens_q"]
        for batch_idx in range(len(cu_q) - 1):
            start = int(cu_q[batch_idx].item())
            end = int(cu_q[batch_idx + 1].item())
            cmp = compare_tensors(actual[start:end], expected[start:end])
            summaries.append({"batch": batch_idx, "q_start": start, "q_end": end, "q_len": end - start, **cmp})
    return summaries


def summarize_cmp_sparse_indices(input_data: Dict) -> Optional[List[Dict]]:
    cmp_sparse_indices = input_data["input"].get("cmp_sparse_indices")
    if cmp_sparse_indices is None:
        return None

    cmp_lens = []
    seqused_cmp = input_data["input"].get("seqused_cmp_kv")
    if seqused_cmp is not None:
        cmp_lens = [int(item) for item in seqused_cmp.tolist()]
    elif input_data["input"].get("cu_seqlens_cmp_kv") is not None:
        cu_cmp = input_data["input"]["cu_seqlens_cmp_kv"]
        cmp_lens = [int(cu_cmp[idx + 1].item() - cu_cmp[idx].item()) for idx in range(len(cu_cmp) - 1)]
    else:
        return None

    summaries = []
    if input_data["layout_q"] == "BSND":
        seqused_q = input_data["input"].get("seqused_q")
        for batch_idx, cmp_len in enumerate(cmp_lens):
            q_len = int(seqused_q[batch_idx].item()) if seqused_q is not None else cmp_sparse_indices.shape[1]
            rows = cmp_sparse_indices[batch_idx, :q_len]
            valid = rows != -1
            summaries.append({
                "batch": batch_idx,
                "q_len": q_len,
                "cmp_len": cmp_len,
                "valid_count": int(valid.sum().item()),
                "out_of_cmp_len": int(((rows >= cmp_len) & valid).sum().item()),
            })
    else:
        cu_q = input_data["input"]["cu_seqlens_q"]
        for batch_idx, cmp_len in enumerate(cmp_lens):
            start = int(cu_q[batch_idx].item())
            end = int(cu_q[batch_idx + 1].item())
            rows = cmp_sparse_indices[start:end]
            valid = rows != -1
            summaries.append({
                "batch": batch_idx,
                "q_start": start,
                "q_end": end,
                "q_len": end - start,
                "cmp_len": cmp_len,
                "valid_count": int(valid.sum().item()),
                "out_of_cmp_len": int(((rows >= cmp_len) & valid).sum().item()),
            })
    return summaries


def make_params(spec: CaseSpec) -> Dict:
    q_lens = [int(x) for x in spec.q_lens]
    ori_lens = [int(x) for x in spec.ori_lens]
    bsz = len(q_lens)
    if bsz != len(ori_lens):
        raise ValueError("q_lens and ori_lens must have the same batch size")
    if any(q <= 0 or ori <= 0 for q, ori in zip(q_lens, ori_lens)):
        raise ValueError("zero length is not used in this generalization matrix")
    if any(q > ori for q, ori in zip(q_lens, ori_lens)):
        raise ValueError("q length must not exceed ori length for causal suffix semantics")

    template = spec.template_mode
    cmp_ratio = spec.cmp_ratio if template != "SWA" else None
    block_size1 = 16 if spec.layout_kv == "PA_BBND" else None
    block_size2 = 16 if spec.layout_kv == "PA_BBND" and template != "SWA" else None
    cmp_lens = [ori // int(cmp_ratio) for ori in ori_lens] if cmp_ratio is not None else []
    cmp_residual = [ori % int(cmp_ratio) for ori in ori_lens] if cmp_ratio is not None else None

    s1 = max(q_lens)
    s2 = max(ori_lens)
    t1 = sum(q_lens) if spec.layout_q == "TND" else None
    t2 = sum(ori_lens) if spec.layout_kv == "TND" else None
    t3 = sum(cmp_lens) if spec.layout_kv == "TND" and cmp_ratio is not None else None

    cu_q = prefix_sum(q_lens) if spec.layout_q == "TND" else None
    cu_ori = prefix_sum(ori_lens) if spec.layout_kv == "TND" else None
    cu_cmp = prefix_sum(cmp_lens) if spec.layout_kv == "TND" and cmp_ratio is not None else None

    seqused_q = q_lens if spec.layout_q == "BSND" and any(item != s1 for item in q_lens) else None
    # TND KV still needs per-batch lengths when cmp_sparse_indices is BSND.
    # Otherwise the index generator falls back to max S2 for every sequence and
    # may mark out-of-sequence cmp ids as valid for shorter batches.
    seqused_ori = ori_lens
    seqused_cmp = cmp_lens if cmp_ratio is not None else None

    block_num1 = None
    block_num2 = None
    if spec.layout_kv == "PA_BBND":
        block_num1 = sum(ceil_div(item, block_size1) for item in ori_lens)
        if template != "SWA":
            block_num2 = sum(ceil_div(item, block_size2) for item in cmp_lens)

    return {
        "testcase_name": f"gen_{spec.case_id}_{spec.scenario}",
        "layout_q": spec.layout_q,
        "layout_kv": spec.layout_kv,
        "q_type": torch.bfloat16,
        "ori_kv_type": torch.bfloat16,
        "cmp_kv_type": torch.bfloat16,
        "B": bsz,
        "S1": s1,
        "S2": s2,
        "T1": t1,
        "T2": t2,
        "T3": t3,
        "N1": spec.head_ratio,
        "N2": 1,
        "D": DEFAULT_D,
        "K": spec.topk,
        "block_num1": block_num1,
        "block_num2": block_num2,
        "block_size1": block_size1,
        "block_size2": block_size2,
        "seqused_q": seqused_q,
        "cu_seqlens_q": cu_q,
        "cu_seqlens_ori_kv": cu_ori,
        "cu_seqlens_cmp_kv": cu_cmp,
        "seqused_ori_kv": seqused_ori,
        "seqused_cmp_kv": seqused_cmp,
        "cmp_residual_kv": cmp_residual,
        "softmax_scale": 1.0 / math.sqrt(DEFAULT_D),
        "cmp_ratio": cmp_ratio,
        "ori_mask_mode": 4,
        "cmp_mask_mode": 3,
        "ori_win_left": 127,
        "ori_win_right": 0,
        "ori_kv_topk_mode": "full",
        "cmp_kv_topk_mode": spec.topk_mode,
        "ori_sparse_indices_mode": "full",
        "cmp_sparse_indices_mode": spec.sparse_mode,
        "actlen_mode": "full",
        "template_mode": template,
        "q_datarange": [-0.2, 0.2],
        "ori_kv_datarange": [-0.2, 0.2],
        "cmp_kv_datarange": [-0.2, 0.2],
        "qk_equal_len": False,
        "return_softmax_lse": spec.return_softmax_lse,
    }


def iter_shape_scenarios(cmp_ratio: Optional[int], template_mode: str):
    if template_mode == "SWA":
        yield "b1_short", [8], [18]
        yield "b2_var", [3, 9], [18, 33]
        yield "b4_tnd_mix", [1, 4, 7, 12], [16, 31, 48, 65]
        yield "b1_long", [16], [1024]
        return

    if cmp_ratio == 1:
        yield "b1_r1", [8], [8]
        yield "b2_mix_r1", [3, 9], [8, 16]
        yield "b4_mix_r1", [1, 4, 7, 12], [4, 8, 16, 32]
        yield "b1_long_r1", [16], [256]
    elif cmp_ratio == 128:
        yield "b1_res0_r128", [4], [128]
        yield "b1_res1_r128", [5], [129]
        yield "b2_mix_r128", [1, 6], [129, 257]
        yield "b4_mix_r128", [1, 2, 4, 8], [128, 129, 255, 257]
        yield "b1_long_r128", [16], [1024]
    else:
        yield "b1_res2_r4", [18], [18]
        yield "b1_res0_r4", [16], [16]
        if template_mode == "SCFA":
            yield "cp_rank1_offset_r4", [16], [80]
            yield "cp_mid_offset_r4", [24], [140]
        yield "b2_mix_r4", [5, 13], [18, 33]
        yield "b4_mix_r4", [1, 4, 9, 16], [4, 18, 33, 65]
        yield "b1_long_r4", [32], [512]


def generate_specs(total: int) -> List[CaseSpec]:
    pool: List[CaseSpec] = []
    random.seed(20260612)
    case_id = 0

    for template_mode in ["SWA", "CFA", "SCFA"]:
        cmp_ratios = [None] if template_mode == "SWA" else [1, 4, 128]
        topks = [None] if template_mode != "SCFA" else SCFA_TOPKS
        for layout_q in LAYOUT_QS:
            for layout_kv in LAYOUT_KVS:
                for head_ratio in HEAD_RATIOS:
                    for cmp_ratio in cmp_ratios:
                        for topk in topks:
                            for scenario, q_lens, ori_lens in iter_shape_scenarios(cmp_ratio, template_mode):
                                if template_mode == "SCFA" and layout_q == "BSND" and len(set(q_lens)) > 1:
                                    q_len = min(max(q_lens), min(ori_lens))
                                    q_lens = [q_len] * len(q_lens)
                                q_pad = 0
                                ori_pad = 32 if layout_kv != "TND" and case_id % 11 == 0 else 0
                                cmp_pad = 16 if template_mode != "SWA" and layout_kv != "TND" and case_id % 13 == 0 else 0
                                store_pad = 1 if template_mode != "SWA" and case_id % 9 == 0 else 0
                                if layout_kv == "TND" and len(q_lens) != 1:
                                    store_pad = 0
                                return_lse = case_id % 97 == 0
                                pool.append(
                                    CaseSpec(
                                        case_id=case_id,
                                        template_mode=template_mode,
                                        layout_q=layout_q,
                                        layout_kv=layout_kv,
                                        head_ratio=head_ratio,
                                        cmp_ratio=cmp_ratio,
                                        topk=topk,
                                        q_lens=q_lens,
                                        ori_lens=ori_lens,
                                        sparse_mode="full",
                                        topk_mode="full",
                                        q_stride_pad=q_pad,
                                        ori_stride_pad=ori_pad,
                                        cmp_stride_pad=cmp_pad,
                                        cmp_store_pad=store_pad,
                                        cmp_pad_nan=bool(store_pad and case_id % 2 == 0),
                                        return_softmax_lse=return_lse,
                                        scenario=scenario,
                                    )
                                )
                                case_id += 1

    if total >= len(pool):
        selected = [pool[idx % len(pool)] for idx in range(total)]
    else:
        selected = [pool[idx * len(pool) // total] for idx in range(total)]
    return [replace(spec, case_id=idx) for idx, spec in enumerate(selected)]


def force_continuous_kv(spec: CaseSpec) -> CaseSpec:
    return replace(
        spec,
        q_stride_pad=0,
        ori_stride_pad=0,
        cmp_stride_pad=0,
        cmp_store_pad=0,
        cmp_pad_nan=False,
    )


def run_precision_case(golden_module, spec: CaseSpec, device: torch.device) -> Dict:
    torch.manual_seed(1000 + spec.case_id)
    random.seed(1000 + spec.case_id)
    params = make_params(spec)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        input_data = golden_module.gen_data(params, spec.template_mode)
    cp_sparse_check = inspect_cp_sparse_first_rows(input_data) if spec.scenario.startswith("cp_") else None
    extend_cmp_kv_store(input_data, spec.cmp_store_pad, spec.cmp_pad_nan)

    start = time.time()
    actual, _ = metadata_and_run(input_data, device, spec)
    elapsed = time.time() - start

    valid_q_mask = get_valid_q_mask(input_data)
    if valid_q_mask is None:
        actual_for_compare = actual
        expected_for_compare = input_data["cpu_output"]
        compared_valid_q_only = False
    else:
        actual_for_compare = actual[valid_q_mask]
        expected_for_compare = input_data["cpu_output"][valid_q_mask]
        compared_valid_q_only = True

    compare = compare_tensors(actual_for_compare, expected_for_compare)
    status = "PASS" if compare["finite"] and compare["close_percent"] >= PASS_PERCENT else "FAIL"
    if cp_sparse_check is not None and not cp_sparse_check["all_first_rows_have_valid_cmp"]:
        status = "FAIL"
    result = {
        "case_id": spec.case_id,
        "status": status,
        "elapsed_sec": elapsed,
        "spec": asdict(spec),
        "metadata": {
            "B": input_data["B"],
            "S1": input_data["S1"],
            "S2": input_data["S2"],
            "T1": input_data["T1"],
            "T2": input_data["T2"],
            "T3": input_data["T3"],
            "N1": input_data["N1"],
            "N2": input_data["N2"],
            "K": input_data["K"],
            "layout_q": input_data["layout_q"],
            "layout_kv": input_data["layout_kv"],
            "template_run_mode": input_data["template_run_mode"],
        },
        "compare": compare,
        "batch_compare": compare_by_batch(actual, input_data["cpu_output"], input_data),
        "compared_valid_q_only": compared_valid_q_only,
        "cmp_sparse_summary": summarize_cmp_sparse_indices(input_data),
        "cp_sparse_first_rows": cp_sparse_check,
        "q_stride": list(actual.stride()),
    }
    return result


def summarize(results: List[Dict]) -> Dict:
    total = len(results)
    passed = sum(1 for item in results if item.get("status") == "PASS")
    failed = sum(1 for item in results if item.get("status") == "FAIL")
    errors = sum(1 for item in results if item.get("status") == "ERROR")
    by_template = {}
    for item in results:
        spec = item.get("spec", {})
        template = spec.get("template_mode", "other")
        by_template.setdefault(template, {"total": 0, "pass": 0, "fail": 0, "error": 0})
        by_template[template]["total"] += 1
        if item.get("status") == "PASS":
            by_template[template]["pass"] += 1
        elif item.get("status") == "FAIL":
            by_template[template]["fail"] += 1
        else:
            by_template[template]["error"] += 1
    return {"total": total, "pass": passed, "fail": failed, "error": errors, "by_template": by_template}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(pathlib.Path(__file__).resolve().parents[1]))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--total", type=int, default=700)
    parser.add_argument("--case-start", type=int, default=0)
    parser.add_argument("--case-end", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--override-ori-stride-pad", type=int, default=None)
    parser.add_argument("--override-cmp-stride-pad", type=int, default=None)
    parser.add_argument("--override-q-stride-pad", type=int, default=None)
    parser.add_argument("--override-return-softmax-lse", type=int, choices=[0, 1], default=None)
    parser.add_argument("--min-head-ratio", type=int, default=None)
    parser.add_argument("--continuous-kv-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = pathlib.Path(args.repo_root).resolve()
    out_path = pathlib.Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = pathlib.Path(args.summary_out).resolve() if args.summary_out else out_path.with_suffix(".summary.json")

    specs = generate_specs(args.total)
    end = args.case_end if args.case_end is not None else len(specs)
    selected = specs[args.case_start : end]
    if args.min_head_ratio is not None:
        selected = [spec for spec in selected if spec.head_ratio >= args.min_head_ratio]
    if args.continuous_kv_only:
        selected = [force_continuous_kv(spec) for spec in selected]
    if args.list_cases:
        for spec in selected:
            print(json.dumps(asdict(spec), ensure_ascii=False), flush=True)
        return

    import_smla_ops(repo_root)
    golden_module = import_pytest_modules(repo_root)
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    results: List[Dict] = []

    with out_path.open("w", encoding="utf-8") as out_file:
        for spec in selected:
            if args.override_ori_stride_pad is not None:
                spec = replace(spec, ori_stride_pad=args.override_ori_stride_pad)
            if args.override_cmp_stride_pad is not None:
                spec = replace(spec, cmp_stride_pad=args.override_cmp_stride_pad)
            if args.override_q_stride_pad is not None:
                spec = replace(spec, q_stride_pad=args.override_q_stride_pad)
            if args.override_return_softmax_lse is not None:
                spec = replace(spec, return_softmax_lse=bool(args.override_return_softmax_lse))
            try:
                item = run_precision_case(golden_module, spec, device)
            except Exception as exc:
                item = {
                    "case_id": spec.case_id,
                    "status": "ERROR",
                    "spec": asdict(spec),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(limit=30),
                }
            out_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            out_file.flush()
            results.append(item)
            print(json.dumps({"case_id": item["case_id"], "status": item["status"]}, ensure_ascii=False), flush=True)
            if args.stop_on_fail and item["status"] != "PASS":
                summary_path.write_text(json.dumps(summarize(results), indent=2, ensure_ascii=False), encoding="utf-8")
                raise SystemExit(1)

    summary = summarize(results)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if summary["fail"] or summary["error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
