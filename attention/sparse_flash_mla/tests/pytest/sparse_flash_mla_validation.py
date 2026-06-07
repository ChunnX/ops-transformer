#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.

import argparse
import importlib
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import torch
import torch_npu


def progress(msg):
    print(f"[sparse_flash_mla_validation] {msg}", flush=True)


def repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[4]


def install_sparse_only_torch_extension(repo_root: Path):
    """Load only the SparseFlashMla torch adapters without importing every op."""
    pkg_root = repo_root / "torch_extension" / "npu_ops_transformer"
    fake_pkg = types.ModuleType("npu_ops_transformer")
    fake_pkg.__file__ = str(pkg_root / "__init__.py")
    fake_pkg.__path__ = [str(pkg_root)]
    sys.modules["npu_ops_transformer"] = fake_pkg

    fake_ops_pkg = types.ModuleType("npu_ops_transformer.ops")
    fake_ops_pkg.__file__ = str(pkg_root / "ops" / "__init__.py")
    fake_ops_pkg.__path__ = [str(pkg_root / "ops")]
    sys.modules["npu_ops_transformer.ops"] = fake_ops_pkg

    metadata_mod = importlib.import_module("npu_ops_transformer.ops.sparse_flash_mla_metadata")
    mla_mod = importlib.import_module("npu_ops_transformer.ops.sparse_flash_mla")
    return metadata_mod, mla_mod


def import_legacy_golden(repo_root: Path):
    golden_dir = repo_root / "experimental" / "attention" / "sparse_attn_sharedkv" / "tests" / "pytest"
    sys.path.insert(0, str(golden_dir))
    return importlib.import_module("sparse_attn_sharedkv_golden")


def make_stride_padded_pa(src: torch.Tensor, pad_len: int = 64) -> torch.Tensor:
    if src is None or src.dim() != 4:
        return src
    shape = list(src.shape)
    row_logical_size = shape[1] * shape[2] * shape[3]
    stride0 = row_logical_size + pad_len
    storage = torch.full((stride0 * shape[0],), float("nan"), dtype=src.dtype, device=src.device)
    out = torch.empty((0,), dtype=src.dtype, device=src.device)
    out.set_(storage.untyped_storage(), 0, shape, (stride0, shape[2] * shape[3], shape[3], 1))
    for i in range(shape[0]):
        out[i].copy_(src[i])
    torch.npu.synchronize()
    return out


def to_npu(x):
    return x.npu() if isinstance(x, torch.Tensor) else None


def cpu_swa_bsnd(q, ori_kv, sinks, softmax_scale, ori_win_left, ori_win_right):
    bsz, s1, n1, dim = q.shape
    _, s2, n2, _ = ori_kv.shape
    group = n1 // n2
    out = torch.zeros_like(q)
    for b_idx in range(bsz):
        for n2_idx in range(n2):
            cur_sinks = sinks[n2_idx * group:(n2_idx + 1) * group].unsqueeze(1)
            for s1_idx in range(s1):
                threshold = s2 - s1 + s1_idx + 1
                win_start = max(threshold - ori_win_left - 1, 0)
                win_end = threshold + ori_win_right
                k = ori_kv[b_idx, win_start:win_end, n2_idx, :].to(torch.float32)
                q_cur = q[b_idx, s1_idx, n2_idx * group:(n2_idx + 1) * group, :].to(torch.float32)
                score = torch.matmul(q_cur, k.T) * softmax_scale
                score_with_sink = torch.cat([score, cur_sinks], dim=1)
                row_max = score_with_sink.max(dim=-1, keepdim=True).values
                exp_score = torch.exp(score - row_max)
                denom = exp_score.sum(dim=-1, keepdim=True) + torch.exp(cur_sinks - row_max)
                prob = exp_score / denom
                out[b_idx, s1_idx, n2_idx * group:(n2_idx + 1) * group, :] = torch.matmul(prob, k).to(q.dtype)
    return out


def compare_tensors(name, expected, actual, rtol=5e-2, atol=5e-2):
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    if not torch.isfinite(actual_cpu.float()).all():
        raise AssertionError(f"{name}: output contains inf or nan")
    diff = (expected_cpu.float() - actual_cpu.float()).abs()
    max_diff = float(diff.max().item()) if diff.numel() else 0.0
    ok = torch.allclose(expected_cpu.float(), actual_cpu.float(), rtol=rtol, atol=atol)
    if not ok:
        raise AssertionError(f"{name}: precision mismatch, max_diff={max_diff}")
    return max_diff


def metadata_for_case(metadata_mod, data, has_cmp_kv):
    meta_in = data["metadata_input"]
    inp = data["input"]
    return metadata_mod.npu_sparse_flash_mla_metadata(
        meta_in["N1"],
        meta_in["N2"],
        meta_in["D"],
        cu_seqlens_q=to_npu(inp.get("cu_seqlens_q")),
        cu_seqlens_ori_kv=to_npu(inp.get("cu_seqlens_ori_kv")),
        cu_seqlens_cmp_kv=to_npu(inp.get("cu_seqlens_cmp_kv")),
        seqused_q=to_npu(inp.get("seqused_q")),
        seqused_ori_kv=to_npu(inp.get("seqused_kv")),
        batch_size=meta_in["B"],
        max_seqlen_q=meta_in["max_seqlen_q"],
        max_seqlen_kv=meta_in["max_seqlen_kv"],
        ori_topk=0,
        cmp_topk=0 if meta_in["K"] is None else meta_in["K"],
        cmp_ratio=-1 if meta_in["cmp_ratio"] is None else meta_in["cmp_ratio"],
        ori_mask_mode=meta_in["ori_mask_mode"],
        cmp_mask_mode=3 if meta_in["cmp_mask_mode"] is None else meta_in["cmp_mask_mode"],
        ori_win_left=meta_in["ori_win_left"],
        ori_win_right=meta_in["ori_win_right"],
        layout_q=meta_in["layout_q"],
        layout_kv=meta_in["layout_kv"],
        has_ori_kv=True,
        has_cmp_kv=has_cmp_kv,
    )


def run_legacy_case(name, golden, metadata_mod, mla_mod, params, stride_pad=False, repeat=False):
    data = golden.gen_data(params)
    inp = data["input"]
    meta_in = data["metadata_input"]
    has_cmp = inp.get("cmp_kv") is not None
    metadata = metadata_for_case(metadata_mod, data, has_cmp)

    ori_kv = to_npu(inp["ori_kv"])
    cmp_kv = to_npu(inp.get("cmp_kv"))
    if stride_pad and inp["layout_kv"] == "PA_ND":
        ori_kv = make_stride_padded_pa(ori_kv)
        cmp_kv = make_stride_padded_pa(cmp_kv)

    def launch():
        out, lse = mla_mod.npu_sparse_flash_mla(
            to_npu(inp["q"]),
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            ori_sparse_indices=None,
            cmp_sparse_indices=to_npu(inp.get("cmp_sparse_indices")),
            ori_block_table=to_npu(inp.get("ori_block_table")),
            cmp_block_table=to_npu(inp.get("cmp_block_table")),
            cu_seqlens_q=to_npu(inp.get("cu_seqlens_q")),
            cu_seqlens_ori_kv=to_npu(inp.get("cu_seqlens_ori_kv")),
            cu_seqlens_cmp_kv=to_npu(inp.get("cu_seqlens_cmp_kv")),
            seqused_q=to_npu(inp.get("seqused_q")),
            seqused_ori_kv=to_npu(inp.get("seqused_kv")),
            seqused_cmp_kv=None,
            cmp_residual_kv=None,
            ori_topk_length=None,
            cmp_topk_length=None,
            sinks=to_npu(inp["sinks"]),
            metadata=metadata,
            softmax_scale=inp["softmax_scale"],
            cmp_ratio=inp.get("cmp_ratio", meta_in.get("cmp_ratio")),
            ori_mask_mode=inp["ori_mask_mode"],
            cmp_mask_mode=inp["cmp_mask_mode"] if inp["cmp_mask_mode"] is not None else 3,
            ori_win_left=inp["ori_win_left"],
            ori_win_right=inp["ori_win_right"],
            layout_q=inp["layout_q"],
            layout_kv=inp["layout_kv"],
            topk_value_mode=1,
            return_softmax_lse=False,
        )
        torch.npu.synchronize()
        return out, lse

    out, lse = launch()
    max_diff = compare_tensors(name, data["cpu_output"], out)
    if tuple(lse.shape) != (0,):
        raise AssertionError(f"{name}: expected empty softmax_lse when return_softmax_lse=False, got {tuple(lse.shape)}")
    if repeat:
        out_2, _ = launch()
        det_diff = compare_tensors(name + "_determinism", out.detach().cpu(), out_2.detach().cpu(), rtol=0, atol=0)
    else:
        det_diff = None
    return {"name": name, "max_diff": max_diff, "determinism_max_diff": det_diff}


def run_bsnd_case(metadata_mod, mla_mod):
    torch.manual_seed(123)
    bsz, s1, s2, n1, n2, dim = 1, 16, 128, 4, 1, 512
    q = (torch.rand((bsz, s1, n1, dim)) * 2 - 1).to(torch.bfloat16)
    ori_kv = (torch.rand((bsz, s2, n2, dim)) * 2 - 1).to(torch.bfloat16)
    sinks = (torch.rand((n1,)) * 0.2 - 0.1).to(torch.float32)
    softmax_scale = 1.0 / math.sqrt(dim)
    expected = cpu_swa_bsnd(q, ori_kv, sinks, softmax_scale, 127, 0)
    metadata = metadata_mod.npu_sparse_flash_mla_metadata(
        n1,
        n2,
        dim,
        batch_size=bsz,
        max_seqlen_q=s1,
        max_seqlen_kv=s2,
        cmp_ratio=-1,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="BSND",
        layout_kv="BSND",
        has_ori_kv=True,
        has_cmp_kv=False,
    )
    out, _ = mla_mod.npu_sparse_flash_mla(
        q.npu(),
        ori_kv=ori_kv.npu(),
        sinks=sinks.npu(),
        metadata=metadata,
        softmax_scale=softmax_scale,
        ori_mask_mode=4,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="BSND",
        layout_kv="BSND",
    )
    torch.npu.synchronize()
    return {"name": "swa_bsnd_bsnd", "max_diff": compare_tensors("swa_bsnd_bsnd", expected, out)}


def run_invalid_case(metadata_mod):
    try:
        metadata_mod.npu_sparse_flash_mla_metadata(
            4,
            2,
            512,
            batch_size=1,
            max_seqlen_q=16,
            max_seqlen_kv=128,
            layout_q="BSND",
            layout_kv="BSND",
            has_ori_kv=True,
            has_cmp_kv=False,
        )
        torch.npu.synchronize()
    except Exception as exc:
        return {"name": "invalid_num_heads_kv", "caught": type(exc).__name__}
    raise AssertionError("invalid_num_heads_kv: expected metadata validation failure")


def make_params(name, layout_kv, mode):
    bsz, s1, s2, n1, n2, dim = 1, 16, 128, 4, 1, 512
    block_size = 128
    ratio = None
    k_top = None
    cmp_block_num = None
    cmp_block_size = None
    cmp_mask_mode = None
    if mode == "cfa":
        ratio, cmp_block_num, cmp_block_size, cmp_mask_mode = 128, 1, block_size, 3
    elif mode == "scfa":
        ratio, k_top, cmp_block_num, cmp_block_size, cmp_mask_mode = 4, 512, 1, block_size, 3
    elif mode != "swa":
        raise ValueError(mode)

    if layout_kv == "PA_ND":
        seqused_kv = [s2]
        cu_ori = None
        cu_cmp = None
        block_num = 1
        t2 = None
    else:
        seqused_kv = None
        cu_ori = [0, s2]
        cu_cmp = None if ratio is None else [0, s2 // ratio]
        block_num = 1
        t2 = s2

    return (
        "TND", layout_kv, torch.bfloat16, torch.bfloat16, torch.bfloat16, bsz, s1, s1,
        n1, n2, dim, k_top, block_num, cmp_block_num, block_size, cmp_block_size,
        [0, s1], seqused_kv, 1.0 / math.sqrt(dim), ratio, 4, cmp_mask_mode, 127, 0,
        name, s2, [-1, 1], [-1, 1], [-1, 1], None, cu_ori, cu_cmp, t2
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo_root = repo_root_from_this_file()
    progress(f"repo_root={repo_root}")
    progress(f"set device {args.device}")
    torch.npu.set_device(args.device)
    progress("load torch adapter modules")
    metadata_mod, mla_mod = install_sparse_only_torch_extension(repo_root)
    progress("import legacy golden")
    golden = import_legacy_golden(repo_root)

    start = time.time()
    results = []
    progress("run swa_tnd_pa_stride")
    results.append(run_legacy_case("swa_tnd_pa_stride", golden, metadata_mod, mla_mod,
                                   make_params("swa_tnd_pa_stride", "PA_ND", "swa"), stride_pad=True, repeat=True))
    progress("run cfa_tnd_tnd")
    results.append(run_legacy_case("cfa_tnd_tnd", golden, metadata_mod, mla_mod,
                                   make_params("cfa_tnd_tnd", "TND", "cfa")))
    progress("run scfa_tnd_pa_stride")
    results.append(run_legacy_case("scfa_tnd_pa_stride", golden, metadata_mod, mla_mod,
                                   make_params("scfa_tnd_pa_stride", "PA_ND", "scfa"), stride_pad=True))
    progress("run swa_bsnd_bsnd")
    results.append(run_bsnd_case(metadata_mod, mla_mod))
    progress("run invalid_num_heads_kv")
    results.append(run_invalid_case(metadata_mod))
    summary = {"elapsed_sec": round(time.time() - start, 3), "results": results}
    print(json.dumps(summary, indent=2, ensure_ascii=False) if args.json else summary)


if __name__ == "__main__":
    main()
