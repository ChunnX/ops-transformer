#!/usr/bin/env python3
"""BSA MXFP8 OCP 单次前向，供 msprof op 包一层。不做 golden。

环境变量:
  BSA_S              qs=kvs
  BSA_KEEP           每行保留的 KV block 比例 (0.5=稀疏度50%, 0.2=稀疏度80%)
  BSA_DEVICE         逻辑 device id (配合 ASCEND_RT_VISIBLE_DEVICES)
  BSA_B / BSA_N1 / BSA_N2 / BSA_D / BSA_OUT
"""
import math
import os
import sys

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import torch_npu  # noqa: F401
from cann_ops_transformer.ops import block_sparse_attention_v3

QUANT_MODE_MXFP8_OCP = 4
E8M0_BIAS = 127
BLOCK = (128, 128)


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _pack_qk_e8m0(b, n, s, d, device):
    ng = d // 32
    assert d % 32 == 0 and ng % 2 == 0
    raw = torch.full((b, n, s, ng), E8M0_BIAS, dtype=torch.uint8, device=device)
    return raw.view(b, n, s, ng // 2, 2).contiguous().view(torch.float8_e8m0fnu)


def _pack_v_e8m0(b, n2, s, d, device):
    nb = (s + 31) // 32
    raw = torch.full((b, n2, nb, d), E8M0_BIAS, dtype=torch.uint8, device=device)
    if nb % 2 != 0:
        pad = torch.full((b, n2, 1, d), E8M0_BIAS, dtype=torch.uint8, device=device)
        raw = torch.cat([raw, pad], dim=2)
        nb += 1
    return raw.reshape(b, n2, nb // 2, 2, d).permute(0, 1, 2, 4, 3).contiguous().view(
        torch.float8_e8m0fnu)


def _make_mask(b, n1, s1, s2, keep, device):
    bx, by = BLOCK
    xb = (s1 + bx - 1) // bx
    yb = (s2 + by - 1) // by
    rnd = torch.rand((b, n1, xb, yb), device=device)
    mask = (rnd < keep).to(torch.int8)
    # 每行至少保留 1 个 KV block，避免空行
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
    out_name = os.environ.get("BSA_OUT", "bfloat16")
    out_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[out_name]

    torch_npu.npu.set_device(device_id)
    device = f"npu:{device_id}"
    torch.manual_seed(42)

    # 造数放 CPU，避免 msprof 把 RandomNormal 当成目标 kernel
    def _fp8(shape):
        return torch.randint(0, 120, shape, dtype=torch.uint8).view(torch.float8_e4m3fn).to(device)

    q = _fp8((b, n1, s, d))
    k = _fp8((b, n2, s, d))
    v = _fp8((b, n2, s, d))
    q_scale = _pack_qk_e8m0(b, n1, s, d, "cpu").to(device)
    k_scale = _pack_qk_e8m0(b, n2, s, d, "cpu").to(device)
    v_scale = _pack_v_e8m0(b, n2, s, d, "cpu").to(device)
    mask = _make_mask(b, n1, s, s, keep, "cpu").to(device)
    act_q = [s] * b
    act_kv = [s] * b
    scale = 1.0 / math.sqrt(d)
    torch.npu.synchronize()

    print(
        f"[mxfp8_perf] start S={s} keep={keep} B={b} N={n1} D={d} "
        f"mask={tuple(mask.shape)} q={tuple(q.shape)}",
        flush=True,
    )
    npu_out, _ = block_sparse_attention_v3(
        q.contiguous(), k.contiguous(), v.contiguous(),
        mask.contiguous(), list(BLOCK), act_q, act_kv,
        num_heads=n1, num_key_value_heads=n2,
        q_input_layout="BNSD", kv_input_layout="BNSD",
        mask_type=0, scale_value=scale, inner_precise=4, block_size=0,
        pre_tokens=2147483647, next_tokens=2147483647,
        softmax_lse_flag=False, quant_mode=QUANT_MODE_MXFP8_OCP, dst_type_max=0.0,
        q_dequant_scale=q_scale, k_dequant_scale=k_scale, v_dequant_scale=v_scale,
        attention_out_dtype=out_dtype,
    )
    torch.npu.synchronize()
    print(f"[mxfp8_perf] done out={tuple(npu_out.shape)} dtype={npu_out.dtype}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
