#!/usr/bin/env python3
"""Minimal PTA precision test for the D400 MXFP4 QSFA path."""

import argparse
import math
from pathlib import Path
import sys

import torch
import torch_npu


EXAMPLE_DIR = Path(__file__).resolve().parent
GOLDEN_DIR = EXAMPLE_DIR.parent / "tests" / "pytest"
sys.path.insert(0, str(GOLDEN_DIR))
import kv_quant_sparse_flash_attention_golden as qsfa_golden


B = 4
S1 = 8
G = 128
S2 = 135
N2 = 1
NOPE_D = 512
ROPE_D = 64
QUERY_D = NOPE_D + ROPE_D
FP4_BYTES = NOPE_D // 2
SCALE_COUNT = NOPE_D // 32
KEY_D = FP4_BYTES + ROPE_D * 2 + SCALE_COUNT


def pack_fp4_e2m1(fp4_codes: torch.Tensor) -> torch.Tensor:
    """Pack D[2i] into the low nibble and D[2i+1] into the high nibble."""
    if fp4_codes.dtype != torch.uint8 or fp4_codes.shape[-1] != NOPE_D:
        raise ValueError(f"fp4_codes must be uint8 [..., {NOPE_D}]")
    if torch.any(fp4_codes > 0xF):
        raise ValueError("each FP4 E2M1 code must be in [0, 15]")

    low = fp4_codes[..., 0::2].to(torch.int16)
    high = fp4_codes[..., 1::2].to(torch.int16)
    packed = low | (high << 4)
    return packed.to(torch.uint8).view(torch.int8)


def unpack_fp4_e2m1(packed: torch.Tensor) -> torch.Tensor:
    raw = packed.view(torch.uint8).to(torch.int16)
    low = raw & 0xF
    high = raw >> 4
    return torch.stack((low, high), dim=-1).flatten(-2)


def build_inputs():
    batch_token = torch.arange(B, dtype=torch.int32).reshape(B, 1, 1, 1) * (S2 + 17)
    token = batch_token + torch.arange(S2, dtype=torch.int32).reshape(1, S2, N2, 1)
    d = torch.arange(NOPE_D, dtype=torch.int32).reshape(1, 1, 1, NOPE_D)
    fp4_codes = torch.remainder(token + d, 16).to(torch.uint8)
    packed_nope = pack_fp4_e2m1(fp4_codes)

    rope_d = torch.arange(ROPE_D, dtype=torch.float32).reshape(1, 1, 1, ROPE_D)
    key_rope = (0.03125 * (token.float() + 1.0) * (torch.remainder(rope_d, 9) - 4.0)).to(torch.bfloat16)

    group = torch.arange(SCALE_COUNT, dtype=torch.int32).reshape(1, 1, 1, SCALE_COUNT)
    scale_e8m0 = (126 + torch.remainder(token + group, 4)).to(torch.uint8)

    key = torch.cat(
        (packed_nope, key_rope.view(torch.int8), scale_e8m0.view(torch.int8)),
        dim=-1,
    ).contiguous()
    if key.shape != (B, S2, N2, KEY_D):
        raise AssertionError(f"unexpected packed key shape: {tuple(key.shape)}")

    batch = torch.arange(B, dtype=torch.float32).reshape(B, 1, 1, 1)
    s1 = torch.arange(S1, dtype=torch.float32).reshape(1, S1, 1, 1)
    head = torch.arange(G, dtype=torch.float32).reshape(1, 1, G, 1)
    qd = torch.arange(QUERY_D, dtype=torch.float32).reshape(1, 1, 1, QUERY_D)
    query = 0.00390625 * torch.remainder(qd * 3 + head * 5 + batch * 7 + s1 * 11, 29)
    query = (query - 0.00390625 * 14).to(torch.bfloat16)

    return {
        "query": query,
        "key": key,
        "value": torch.zeros((B, S2, N2, NOPE_D), dtype=torch.int8),
        "sparse_indices": torch.arange(S2, dtype=torch.int32)
        .reshape(1, 1, N2, S2)
        .expand(B, S1, N2, S2)
        .contiguous(),
        "key_dequant_scale": torch.ones((1,), dtype=torch.float32),
        "value_dequant_scale": torch.ones((1,), dtype=torch.float32),
        "actual_seq_lengths_query": torch.full((B,), S1, dtype=torch.int32),
        "actual_seq_lengths_kv": torch.full((B,), S2, dtype=torch.int32),
        "packed_nope": packed_nope,
        "key_rope": key_rope,
        "scale_e8m0": scale_e8m0,
    }


def build_golden(inputs):
    fp4_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
    )
    fp4_codes = unpack_fp4_e2m1(inputs["packed_nope"])
    fp4_values = fp4_lut[fp4_codes.long()]
    scale_values = torch.ldexp(
        torch.ones(inputs["scale_e8m0"].shape, dtype=torch.float32),
        inputs["scale_e8m0"].to(torch.int32) - 127,
    )
    nope = (fp4_values * torch.repeat_interleave(scale_values, 32, dim=-1)).to(torch.bfloat16)
    decoded_key = torch.cat((nope, inputs["key_rope"]), dim=-1).permute(0, 2, 1, 3).contiguous()

    fa_param = {
        "b": B,
        "numHeads": G,
        "numKeyValueHeads": N2,
        "actualSeqLengths_q": [S1] * B,
        "actualSeqLengths_kv": [S2] * B,
        "scaleValue": 1.0 / math.sqrt(float(QUERY_D)),
        "sparse_blocksize": 1,
        "sparse_blockcount": S2,
        "sparse_indices_bnsd_tensor": inputs["sparse_indices"].permute(0, 2, 1, 3).contiguous(),
        "q_bnsd_shape": [B, G, S1, QUERY_D],
        "v_bnsd_shape": [B, N2, S2, NOPE_D],
        "q_bnsd_tensor": inputs["query"].permute(0, 2, 1, 3).contiguous(),
        "k_bnsd_tensor": decoded_key,
        "k_dequant_scale_bnsd_tensor": torch.ones((B, N2, S2, 4), dtype=torch.float32),
        "v_bnsd_tensor": nope.permute(0, 2, 1, 3).contiguous(),
        "v_dequant_scale_bnsd_tensor": torch.ones((B, N2, S2, 4), dtype=torch.float32),
        "q_dtype": "bfloat16",
        "sparse_mode": 3,
        "sinks_tensor": None,
    }
    return qsfa_golden._t_increattention_bnsd(fa_param).permute(0, 2, 1, 3).contiguous()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.npu.set_device(args.device)

    inputs = build_inputs()
    expected = build_golden(inputs)

    actual = torch_npu.npu_kv_quant_sparse_flash_attention(
        inputs["query"].npu(),
        inputs["key"].npu(),
        inputs["value"].npu(),
        inputs["sparse_indices"].npu(),
        1.0 / math.sqrt(float(QUERY_D)),
        2,
        2,
        key_dequant_scale=inputs["key_dequant_scale"].npu(),
        value_dequant_scale=inputs["value_dequant_scale"].npu(),
        actual_seq_lengths_query=inputs["actual_seq_lengths_query"].npu(),
        actual_seq_lengths_kv=inputs["actual_seq_lengths_kv"].npu(),
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=3,
        pre_tokens=(1 << 63) - 1,
        next_tokens=(1 << 63) - 1,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
    )
    torch.npu.synchronize()

    actual_f32 = actual.cpu().float()
    expected_f32 = expected.float()
    close = torch.isclose(actual_f32, expected_f32, rtol=2e-2, atol=5e-2)
    close_ratio = float(close.float().mean())
    max_abs_error = float((actual_f32 - expected_f32).abs().max())
    if close_ratio < 0.99:
        raise AssertionError(
            f"close_ratio={close_ratio:.6f}, max_abs_error={max_abs_error:.6f}, "
            f"actual[:8]={actual_f32.flatten()[:8].tolist()}, "
            f"expected[:8]={expected_f32.flatten()[:8].tolist()}"
        )
    print(
        f"[PASS] B={B}, S1={S1}, G={G}, S2={S2}, key_D={KEY_D}, "
        f"close_ratio={close_ratio:.6f}, max_abs_error={max_abs_error:.6f}"
    )


if __name__ == "__main__":
    main()
