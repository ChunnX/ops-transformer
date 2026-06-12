#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Minimal BSND + CFA + cmp_residual_kv construction and torch_npu call.
#
# Expected behavior:
#   seqused_cmp_kv marks only the first cmp_act_len rows as valid.
#   If cmp_kv padding rows are filled with NaN, the output must still be finite.

import argparse
import json
import math
import pathlib
import sys
from typing import Dict, List

import torch
import torch_npu


def add_torch_extension_to_path(repo_root: pathlib.Path) -> None:
    torch_extension = repo_root / "torch_extension"
    if str(torch_extension) not in sys.path:
        sys.path.insert(0, str(torch_extension))


def import_smla_ops() -> None:
    import npu_ops_transformer.ops.sparse_flash_mla_metadata  # noqa: F401
    import npu_ops_transformer.ops.sparse_flash_mla  # noqa: F401


def make_tensor_with_stride_padding(shape, dtype, device, scale=0.02, pad_stride=0):
    if pad_stride <= 0:
        return torch.randn(shape, dtype=dtype, device=device) * scale

    padded_shape = list(shape)
    padded_shape[-1] += pad_stride
    storage = torch.randn(padded_shape, dtype=dtype, device=device) * scale
    strides = list(storage.stride())
    return torch.as_strided(storage, size=shape, stride=strides)


def nan_rows_by_batch(tensor: torch.Tensor) -> List[List[int]]:
    cpu = tensor.detach().cpu().float()
    nan_mask = torch.isnan(cpu).any(dim=(2, 3))
    rows = []
    for batch_idx in range(cpu.shape[0]):
        rows.append(torch.nonzero(nan_mask[batch_idx], as_tuple=False).flatten().tolist())
    return rows


def run_case(args, pad_nan: bool) -> Dict:
    device = torch.device(f"npu:{args.device}")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    torch.npu.manual_seed(args.seed)

    q = make_tensor_with_stride_padding(
        (args.batch, args.q_len, args.num_heads_q, args.head_dim),
        dtype,
        device,
        pad_stride=args.q_stride_pad,
    )
    ori_kv = make_tensor_with_stride_padding(
        (args.batch, args.ori_len, args.num_heads_kv, args.head_dim),
        dtype,
        device,
        pad_stride=args.ori_stride_pad,
    )
    cmp_kv = make_tensor_with_stride_padding(
        (args.batch, args.cmp_store_len, args.num_heads_kv, args.head_dim),
        dtype,
        device,
        pad_stride=args.cmp_stride_pad,
    )
    if pad_nan:
        cmp_kv[:, args.cmp_act_len :, :, :] = float("nan")
    else:
        cmp_kv[:, args.cmp_act_len :, :, :] = 0

    seqused_q = torch.full((args.batch,), args.q_len, dtype=torch.int32, device=device)
    seqused_ori_kv = torch.full((args.batch,), args.ori_len, dtype=torch.int32, device=device)
    seqused_cmp_kv = torch.full((args.batch,), args.cmp_act_len, dtype=torch.int32, device=device)
    cmp_residual_kv = torch.full((args.batch,), args.cmp_residual, dtype=torch.int32, device=device)
    sinks = torch.zeros((args.num_heads_q,), dtype=torch.float32, device=device)

    metadata = torch.ops.npu_ops_transformer.npu_sparse_flash_mla_metadata(
        args.num_heads_q,
        args.num_heads_kv,
        args.head_dim,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        batch_size=args.batch,
        max_seqlen_q=args.q_len,
        max_seqlen_kv=args.ori_len,
        ori_topk=0,
        cmp_topk=0,
        cmp_ratio=args.cmp_ratio,
        ori_mask_mode=args.ori_mask_mode,
        cmp_mask_mode=args.cmp_mask_mode,
        ori_win_left=args.ori_win_left,
        ori_win_right=args.ori_win_right,
        layout_q="BSND",
        layout_kv="BSND",
        has_ori_kv=True,
        has_cmp_kv=True,
    )

    out, softmax_lse = torch.ops.npu_ops_transformer.npu_sparse_flash_mla(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=args.softmax_scale,
        cmp_ratio=args.cmp_ratio,
        ori_mask_mode=args.ori_mask_mode,
        cmp_mask_mode=args.cmp_mask_mode,
        ori_win_left=args.ori_win_left,
        ori_win_right=args.ori_win_right,
        layout_q="BSND",
        layout_kv="BSND",
        return_softmax_lse=False,
    )
    torch.npu.synchronize()

    out_cpu = out.detach().cpu().float()
    return {
        "pad_nan": pad_nan,
        "finite": bool(torch.isfinite(out_cpu).all()),
        "nan_count": int(torch.isnan(out_cpu).sum().item()),
        "inf_count": int(torch.isinf(out_cpu).sum().item()),
        "max_abs": float(torch.nan_to_num(out_cpu, nan=0.0, posinf=0.0, neginf=0.0).abs().max().item()),
        "nan_rows_by_batch": nan_rows_by_batch(out),
        "q_shape": list(q.shape),
        "ori_kv_shape": list(ori_kv.shape),
        "cmp_kv_shape": list(cmp_kv.shape),
        "q_stride": list(q.stride()),
        "ori_kv_stride": list(ori_kv.stride()),
        "cmp_kv_stride": list(cmp_kv.stride()),
        "seqused_q": seqused_q.detach().cpu().tolist(),
        "seqused_ori_kv": seqused_ori_kv.detach().cpu().tolist(),
        "seqused_cmp_kv": seqused_cmp_kv.detach().cpu().tolist(),
        "cmp_residual_kv": cmp_residual_kv.detach().cpu().tolist(),
        "reconstructed_ori_len": args.cmp_act_len * args.cmp_ratio + args.cmp_residual,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(pathlib.Path(__file__).resolve().parents[1]))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")

    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-len", type=int, default=18)
    parser.add_argument("--ori-len", type=int, default=18)
    parser.add_argument("--cmp-act-len", type=int, default=4)
    parser.add_argument("--cmp-store-len", type=int, default=5)
    parser.add_argument("--cmp-ratio", type=int, default=4)
    parser.add_argument("--cmp-residual", type=int, default=2)
    parser.add_argument("--num-heads-q", type=int, default=64)
    parser.add_argument("--num-heads-kv", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=512)

    parser.add_argument("--ori-mask-mode", type=int, default=4)
    parser.add_argument("--cmp-mask-mode", type=int, default=3)
    parser.add_argument("--ori-win-left", type=int, default=127)
    parser.add_argument("--ori-win-right", type=int, default=0)
    parser.add_argument("--softmax-scale", type=float, default=None)

    parser.add_argument("--q-stride-pad", type=int, default=0)
    parser.add_argument("--ori-stride-pad", type=int, default=0)
    parser.add_argument("--cmp-stride-pad", type=int, default=0)
    parser.add_argument("--pad-mode", choices=["zero", "nan", "both"], default="both")
    return parser.parse_args()


def main():
    args = parse_args()
    args.repo_root = pathlib.Path(args.repo_root).resolve()
    if args.softmax_scale is None:
        args.softmax_scale = 1.0 / math.sqrt(args.head_dim)

    add_torch_extension_to_path(args.repo_root)
    import_smla_ops()
    torch.npu.set_device(args.device)

    expected_ori_len = args.cmp_act_len * args.cmp_ratio + args.cmp_residual
    if expected_ori_len != args.ori_len:
        raise ValueError(
            f"Invalid residual semantics: cmp_act_len * cmp_ratio + cmp_residual "
            f"= {expected_ori_len}, but ori_len = {args.ori_len}"
        )
    if args.cmp_store_len < args.cmp_act_len:
        raise ValueError("cmp_store_len must be >= cmp_act_len")
    if args.num_heads_kv != 1:
        raise ValueError("This SMLA validation case keeps num_heads_kv == 1")

    modes = [False, True] if args.pad_mode == "both" else [args.pad_mode == "nan"]
    results = [run_case(args, pad_nan) for pad_nan in modes]
    print(json.dumps({"results": results}, indent=2))

    failed = [item for item in results if not item["finite"]]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
