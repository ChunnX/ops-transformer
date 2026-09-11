#!/usr/bin/env python3
"""Minimal BSA HIF8 kernel-entry smoke case.

Shape: B=1, Nq=Nkv=1, QS=128, KVS=256, D=128, layout=BNSD.
This case only checks that the Torch wrapper, aclnn/tiling dispatch and AICore
kernel can be entered and return. It is intentionally not a numerical golden.
"""

import argparse

import torch
import torch_npu
from cann_ops_transformer.ops import block_sparse_attention
from compare_utils import check_result_printf_res

import generate_hifloat8_data as hif8_codec  # noqa: E402


HIF8_MAX = 32768.0


def quantize_hif8_per_tensor(src: torch.Tensor):
    """Return uint8 HIF8 storage and its FP32 per-tensor dequant scale [1]."""
    max_abs = float(src.abs().max())
    scale_value = max_abs / HIF8_MAX if max_abs != 0.0 else 1.0
    scale = torch.tensor([scale_value], dtype=torch.float32)
    encoded = hif8_codec.trans_float_tensor_to_hifuint8(
        (src / scale_value).contiguous(), round_mode="round", over_mode=True
    )
    return encoded.contiguous(), scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--p-scale", type=float, default=32.0)
    parser.add_argument("--unit-case", action="store_true")
    parser.add_argument("--unit-value", type=float, default=1.0)
    parser.add_argument("--q-value", type=float, default=None)
    parser.add_argument("--k-value", type=float, default=None)
    parser.add_argument("--v-value", type=float, default=None)
    args = parser.parse_args()

    torch.manual_seed(1)
    torch_npu.npu.set_device(args.device)

    batch = 1
    q_heads = 1
    kv_heads = 1
    q_seq = 128
    kv_seq = 256
    head_dim = 128
    block_shape = [128, 256]

    if args.unit_case:
        q_value = args.unit_value if args.q_value is None else args.q_value
        k_value = args.unit_value if args.k_value is None else args.k_value
        v_value = args.unit_value if args.v_value is None else args.v_value
        query_fp32 = torch.full((batch, q_heads, q_seq, head_dim), q_value, dtype=torch.float32)
        key_fp32 = torch.full((batch, kv_heads, kv_seq, head_dim), k_value, dtype=torch.float32)
        value_fp32 = torch.full((batch, kv_heads, kv_seq, head_dim), v_value, dtype=torch.float32)
        query = hif8_codec.trans_float_tensor_to_hifuint8(query_fp32, round_mode="round", over_mode=True)
        key = hif8_codec.trans_float_tensor_to_hifuint8(key_fp32, round_mode="round", over_mode=True)
        value = hif8_codec.trans_float_tensor_to_hifuint8(value_fp32, round_mode="round", over_mode=True)
        q_dequant_scale = torch.ones(1, dtype=torch.float32)
        k_dequant_scale = torch.ones(1, dtype=torch.float32)
        v_dequant_scale = torch.ones(1, dtype=torch.float32)
    else:
        query_fp32 = torch.randn(batch, q_heads, q_seq, head_dim, dtype=torch.float32)
        key_fp32 = torch.randn(batch, kv_heads, kv_seq, head_dim, dtype=torch.float32)
        value_fp32 = torch.randn(batch, kv_heads, kv_seq, head_dim, dtype=torch.float32)
        query, q_dequant_scale = quantize_hif8_per_tensor(query_fp32)
        key, k_dequant_scale = quantize_hif8_per_tensor(key_fp32)
        value, v_dequant_scale = quantize_hif8_per_tensor(value_fp32)

    # ceil(128 / 128) x ceil(256 / 256) == 1 x 1; value 1 means attend.
    block_sparse_mask = torch.ones(
        batch, q_heads, 1, 1, dtype=torch.int8
    )
    p_quant_scale = torch.tensor([args.p_scale], dtype=torch.float32)

    print(
        "[BSA HIF8] calling: "
        f"layout=BNSD, Q={tuple(query.shape)}, K/V={tuple(key.shape)}, "
        f"block_shape={block_shape}, quant_mode=5, inner_precise=0"
    )
    attention_out, softmax_lse = block_sparse_attention(
        query.npu(),
        key.npu(),
        value.npu(),
        block_sparse_mask.npu(),
        block_shape,
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        num_key_value_heads=kv_heads,
        scale_value=1.0 if args.unit_case else head_dim ** -0.5,
        inner_precise=0,
        actual_seq_lengths=[q_seq],
        actual_seq_lengths_kv=[kv_seq],
        return_softmax_lse=True,
        mask_type=0,
        quant_mode=5,
        block_size=0,
        pre_tokens=2147483647,
        next_tokens=2147483647,
        dst_type_max=0.0,
        q_dequant_scale=q_dequant_scale.npu(),
        k_dequant_scale=k_dequant_scale.npu(),
        v_dequant_scale=v_dequant_scale.npu(),
        # Exercise the non-default HIF8 P scale and the matching LSE correction.
        p_quant_scale=p_quant_scale.npu(),
        attention_out_dtype=torch.bfloat16,
    )
    torch_npu.npu.synchronize()

    output_fp32 = attention_out.float().cpu()
    lse_fp32 = softmax_lse.float().cpu()
    query_dequant = hif8_codec.trans_hifuint8_tensor_to_float(query) * q_dequant_scale
    key_dequant = hif8_codec.trans_hifuint8_tensor_to_float(key) * k_dequant_scale
    value_dequant = hif8_codec.trans_hifuint8_tensor_to_float(value) * v_dequant_scale
    scale_value = 1.0 if args.unit_case else head_dim ** -0.5
    score_ref = torch.matmul(query_dequant, key_dequant.transpose(-2, -1)) * scale_value
    output_ref = torch.matmul(torch.softmax(score_ref, dim=-1), value_dequant)
    lse_ref = torch.logsumexp(score_ref, dim=-1)
    output_pct, output_result = check_result_printf_res(output_ref, output_fp32, "bfloat16")
    lse_pct, lse_result = check_result_printf_res(lse_ref, lse_fp32.reshape_as(lse_ref), "float32")
    print(
        "[BSA HIF8] returned: "
        f"out_shape={tuple(attention_out.shape)}, out_dtype={attention_out.dtype}, "
        f"finite={torch.isfinite(output_fp32).all().item()}, "
        f"out_sum={output_fp32.sum().item():.8f}, "
        f"lse_numel={softmax_lse.numel()}, "
        f"lse_finite={torch.isfinite(lse_fp32).all().item()}, "
        f"lse_sum={lse_fp32.sum().item():.8f}"
    )
    print(
        "[BSA HIF8] compare: "
        f"out={output_result} ({output_pct:.4f}%), "
        f"lse={lse_result} ({lse_pct:.4f}%)"
    )


if __name__ == "__main__":
    main()
