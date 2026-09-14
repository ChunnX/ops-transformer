"""Shared Torch demo for grouped_matmul_swiglu_quant_v3."""

import argparse
import math

import torch
import torch_npu


def run_demo(mode, transpose=False, activate_left=True):
    torch.manual_seed(0)
    # K=128 is intentional: V3 mode-2 selects the Tensor API path only when
    # K is aligned to 128.
    e, m, k, n = 1, 64, 128, 128
    x = torch.randint(-4, 5, (m, k), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
    weight_nd = torch.randint(-4, 5, (e, k, n), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
    weight = torch_npu.npu_format_cast(weight_nd, 29)
    weight_scale = torch.full((e, math.ceil(k / 64), n, 2), 127, dtype=torch.int8).npu()
    if transpose:
        weight = weight.transpose(1, 2)
        weight_scale = weight_scale.transpose(1, 2)
    x_scale = torch.full((m, math.ceil(k / 64), 2), 127, dtype=torch.int8).npu()
    group_list = torch.tensor([m], dtype=torch.int64).npu()
    output, output_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
        x,
        [weight],
        [weight_scale],
        x_scale,
        group_list,
        dequant_mode=2,
        dequant_dtype=torch.float32,
        quant_mode=2,
        quant_dtype=torch.float8_e4m3fn,
        weight_scale_dtype=torch_npu.float8_e8m0fnu,
        x_scale_dtype=torch_npu.float8_e8m0fnu,
        activate_dim=-1,
        activate_left=activate_left,
        swiglu_mode=mode,
        clamp_limit=7.0,
        glu_alpha=1.702,
        glu_bias=1.0,
        axis=-1,
        round_mode="rint",
        scale_alg=0,
        dst_type_max=0.0,
    )
    torch.npu.synchronize()
    print(
        f"torch demo mode={mode} transpose={int(transpose)} activate_left={int(activate_left)} "
        f"output.shape={tuple(output.shape)} output_scale.shape={tuple(output_scale.shape)}"
    )
    return output, output_scale


def main(mode):
    parser = argparse.ArgumentParser()
    parser.add_argument("--transpose", action="store_true")
    parser.add_argument("--activate-left", type=int, default=1, choices=(0, 1))
    args = parser.parse_args()
    run_demo(mode, args.transpose, bool(args.activate_left))
