#!/usr/bin/env python3
"""BSA FP16 单次前向，供 msprof op 包一层。不做 golden。

走 origin/bsa_fp8_512_zn 的 torch 绑定（cann_ops_transformer_custom）+ zn opp。
B=1 N=1 D=128 BNSD block=(128,128)；Q/K/V/out=float16；quant_mode=0。
"""

import math
import os
import sys

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import torch_npu  # noqa: F401
from cann_ops_transformer_custom.ops.attention.block_sparse_attention.block_sparse_attention import (
    block_sparse_attention,
)

QUANT_MODE_NONE = 0


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _make_mask(b, n1, s1, s2, keep, device, block):
    bx, by = block
    xb = (s1 + bx - 1) // bx
    yb = (s2 + by - 1) // by
    rnd = torch.rand((b, n1, xb, yb), device=device)
    mask = (rnd < keep).to(torch.int8)
    empty = mask.sum(dim=-1, keepdim=True) == 0
    if bool(empty.any()):
        j0 = rnd.argmin(dim=-1, keepdim=True)
        mask.scatter_(-1, j0, torch.ones_like(j0, dtype=torch.int8))
    return mask


def main():
    s = _env_int("BSA_S", 8422)
    keep = _env_float("BSA_KEEP", 0.5)
    b = _env_int("BSA_B", 1)
    n1 = _env_int("BSA_N1", 1)
    n2 = _env_int("BSA_N2", 1)
    d = _env_int("BSA_D", 128)
    device_id = _env_int("BSA_DEVICE", 0)
    out_name = os.environ.get("BSA_OUT", "float16")
    out_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[out_name]
    if out_dtype != torch.float16:
        raise ValueError("quant_mode=0 FP16 path requires BSA_OUT=float16")
    block = (_env_int("BSA_BX", 128), _env_int("BSA_BY", 128))

    torch_npu.npu.set_device(device_id)
    device = f"npu:{device_id}"
    torch.manual_seed(42)

    # 造数放 CPU，避免 msprof 把 Random 当成目标 kernel
    def _fp16(shape):
        return torch.randn(shape, dtype=torch.float16).to(device)

    q = _fp16((b, n1, s, d))
    k = _fp16((b, n2, s, d))
    v = _fp16((b, n2, s, d))
    mask = _make_mask(b, n1, s, s, keep, "cpu", block).to(device)
    act_q = [s] * b
    act_kv = [s] * b
    scale = 1.0 / math.sqrt(d)
    torch.npu.synchronize()

    print(
        f"[fp16_perf] start S={s} keep={keep} B={b} N={n1} D={d} "
        f"block={block} mask={tuple(mask.shape)} q={tuple(q.shape)}",
        flush=True,
    )
    npu_out, _ = block_sparse_attention(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        mask.contiguous(),
        list(block),
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        num_key_value_heads=n2,
        scale_value=scale,
        inner_precise=4,
        actual_seq_lengths=act_q,
        actual_seq_lengths_kv=act_kv,
        return_softmax_lse=False,
        mask_type=0,
        quant_mode=QUANT_MODE_NONE,
        block_size=0,
        pre_tokens=2147483647,
        next_tokens=2147483647,
        dst_type_max=0.0,
        attention_out_dtype=out_dtype,
    )
    torch.npu.synchronize()
    print(
        f"[fp16_perf] done out={tuple(npu_out.shape)} dtype={npu_out.dtype}", flush=True
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
