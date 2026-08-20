#!/usr/bin/env python3
"""FIA MXFP8 OCP 单次前向，供 msprof op 包一层。非 PA、TND、无 RoPE/LSE。

环境变量:
  FIA_S              qs=kvs
  FIA_SPARSE         0=dense 无 mask, 3=rightDownCausal (2048x2048)
  FIA_DEVICE         逻辑 device id (配合 ASCEND_RT_VISIBLE_DEVICES)
  FIA_B / FIA_N1 / FIA_N2 / FIA_D / FIA_OUT
"""
import math
import os
import sys

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
import torch_npu  # noqa: F401

E8M0_BIAS = 127
GROUP = 32


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _pack_qk_e8m0_tnd(t, n, d, device):
    ng = d // GROUP
    assert d % GROUP == 0 and ng % 2 == 0
    raw = torch.full((t, n, ng), E8M0_BIAS, dtype=torch.uint8, device=device)
    # FIA 走 uint8 存储 + dequant_scale_*_dtype=e8m0，不要 view 成 e8m0 tensor
    return raw.view(t, n, ng // 2, 2).contiguous()


def _pack_v_e8m0_tnd(t, n, d, device):
    sg = (t + GROUP - 1) // GROUP
    raw = torch.full((sg, n, d), E8M0_BIAS, dtype=torch.uint8, device=device)
    if sg % 2 != 0:
        pad = torch.full((1, n, d), E8M0_BIAS, dtype=torch.uint8, device=device)
        raw = torch.cat([raw, pad], dim=0)
        sg += 1
    packed = torch.zeros((sg // 2, n, d, 2), dtype=torch.uint8, device=device)
    packed[..., 0] = raw[0::2, :, :]
    packed[..., 1] = raw[1::2, :, :]
    return packed.contiguous()


def main():
    s = _env_int("FIA_S", 8422)
    sparse = _env_int("FIA_SPARSE", 0)
    b = _env_int("FIA_B", 1)
    n1 = _env_int("FIA_N1", 1)
    n2 = _env_int("FIA_N2", 1)
    d = _env_int("FIA_D", 128)
    device_id = _env_int("FIA_DEVICE", 0)
    out_name = os.environ.get("FIA_OUT", "bfloat16")
    out_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[out_name]
    if b != 1:
        raise ValueError("this perf one-shot only supports B=1 TND")
    if sparse not in (0, 3):
        raise ValueError("MxFP8 FIA sparse_mode only 0 or 3")

    torch_npu.npu.set_device(device_id)
    device = f"npu:{device_id}"
    torch.manual_seed(42)

    def _fp8(shape):
        return torch.randint(0, 120, shape, dtype=torch.uint8).view(torch.float8_e4m3fn).to(device)

    q = _fp8((s, n1, d))
    k = _fp8((s, n2, d))
    v = _fp8((s, n2, d))
    q_scale = _pack_qk_e8m0_tnd(s, n1, d, "cpu").to(device)
    k_scale = _pack_qk_e8m0_tnd(s, n2, d, "cpu").to(device)
    v_scale = _pack_v_e8m0_tnd(s, n2, d, "cpu").to(device)
    p_scale = torch.tensor([1.0], dtype=torch.float32).to(device)
    mask = None
    if sparse == 3:
        mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1).to(device)
    accum_q = [s]
    accum_kv = [s]
    scale = 1.0 / math.sqrt(d)
    torch.npu.synchronize()

    print(
        f"[fia_mxfp8_perf] start S={s} sparse={sparse} B={b} Nq={n1} Nkv={n2} D={d} "
        f"q={tuple(q.shape)} k_scale={tuple(k_scale.shape)} v_scale={tuple(v_scale.shape)}",
        flush=True,
    )
    atten_out, _ = torch_npu.npu_fused_infer_attention_score_v2(
        q.contiguous(), k.contiguous(), v.contiguous(),
        query_rope=None,
        key_rope=None,
        atten_mask=mask,
        actual_seq_qlen=accum_q,
        actual_seq_kvlen=accum_kv,
        dequant_scale_query=q_scale,
        dequant_scale_key=k_scale,
        dequant_scale_value=v_scale,
        block_table=None,
        block_size=0,
        num_query_heads=n1,
        num_key_value_heads=n2,
        softmax_scale=scale,
        input_layout="TND",
        sparse_mode=sparse,
        quant_scale_p=p_scale,
        out_dtype=out_dtype,
        query_quant_mode=6,
        key_quant_mode=6,
        value_quant_mode=8,
        query_dtype=torch.float8_e4m3fn,
        key_dtype=torch.float8_e4m3fn,
        value_dtype=torch.float8_e4m3fn,
        dequant_scale_query_dtype=torch_npu.float8_e8m0fnu,
        dequant_scale_key_dtype=torch_npu.float8_e8m0fnu,
        dequant_scale_value_dtype=torch_npu.float8_e8m0fnu,
        return_softmax_lse=False,
    )
    torch.npu.synchronize()
    print(
        f"[fia_mxfp8_perf] done out={tuple(atten_out.shape)} dtype={atten_out.dtype}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
