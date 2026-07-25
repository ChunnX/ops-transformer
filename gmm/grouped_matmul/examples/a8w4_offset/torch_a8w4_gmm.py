#!/usr/bin/env python3
"""A8W4 grouped-matmul examples for null and non-null offset.

The reference path runs on CPU with unpacked int8 values in [-8, 7].  PyTorch
has an int4 dtype, but CPU conversion/matmul kernels are not implemented.  With
--run-npu the script also tries torch_npu.npu_grouped_matmul on an NPU.
"""

import argparse
import sys
from typing import List, Tuple

import torch


E, M, K, N = 2, 8, 512, 16
GROUP_SIZE = 256


def make_case(with_offset: bool) -> Tuple[torch.Tensor, ...]:
    gen = torch.Generator().manual_seed(20260711)
    x = torch.randint(-8, 8, (M, K), dtype=torch.int8, generator=gen)
    weight = torch.randint(-8, 8, (E, K, N), dtype=torch.int8, generator=gen)
    group_list = torch.tensor([M // E, M // E], dtype=torch.int64)  # groupListType=1: counts

    group_num = 1 if with_offset else K // GROUP_SIZE
    scale = torch.rand((E, group_num, N), generator=gen, dtype=torch.float32) * 0.02 + 0.005
    zero_point = torch.randint(-3, 4, (E, 1, N), generator=gen, dtype=torch.int32).float()
    # The operator expects the already fused value antiquantOffset * scale.
    offset = zero_point * scale if with_offset else None

    scale_per_k = scale.repeat_interleave(K // group_num, dim=1)
    fused_offset_per_k = offset.expand(E, K, N) if offset is not None else 0.0
    dequant_weight = weight.float() * scale_per_k + fused_offset_per_k
    # Required A8W4 MSD correction: it cancels the internal (x - 8)
    # transformation. It is not an application-level "+ bias" term.
    auxiliary_bias = (8.0 * weight.float() * scale_per_k).sum(dim=1)
    return x, weight, scale, offset, auxiliary_bias, group_list, dequant_weight


def torch_reference(with_offset: bool) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
    case = make_case(with_offset)
    x, _, _, _, _, group_list, dequant_weight = case
    outputs: List[torch.Tensor] = []
    start = 0
    for expert, token_count in enumerate(group_list.tolist()):
        end = start + token_count
        outputs.append(x[start:end].float() @ dequant_weight[expert])
        start = end
    return torch.cat(outputs, dim=0).to(torch.float16), case


def report_torch_support() -> None:
    print(f"torch={torch.__version__}; torch.int4 present={hasattr(torch, 'int4')}")
    try:
        torch.tensor([-1, 0, 1], dtype=torch.int8).to(torch.int4)
        print("CPU int8 -> int4 conversion: supported")
    except (NotImplementedError, RuntimeError) as exc:
        print(f"CPU int8 -> int4 conversion: unsupported ({type(exc).__name__})")
    print("CPU INT4 grouped matmul: no native PyTorch operator; reference uses unpacked int8")


def run_npu(with_offset: bool, case: Tuple[torch.Tensor, ...]) -> None:
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("torch_npu is not installed") from exc
    if not torch.npu.is_available():
        raise RuntimeError("torch_npu is installed, but no NPU is available")

    x, weight, scale, offset, bias, group_list, _ = case
    # The official packer also applies the device-required interleaved layout.
    packed_weight = torch.stack([
        torch_npu.npu_convert_weight_to_int4pack(w.to(torch.int32).npu()) for w in weight
    ])
    # A8W4 V5 takes the fused dequant parameter in UINT64 representation.
    fused_scale = torch.stack([
        torch.stack([torch_npu.npu_trans_quant_param(s.npu()) for s in expert_scale])
        for expert_scale in scale
    ])
    per_token_scale = torch.ones((M, 1), dtype=torch.float32, device="npu")
    kwargs = dict(
        bias=[bias.npu()],
        scale=[fused_scale],
        offset=None if offset is None else [offset.npu()],
        per_token_scale=[per_token_scale],
        group_list=group_list.npu(),
        split_item=3,
        group_type=0,
        group_list_type=1,
        act_type=0,
        tuning_config=[M // E, 0, -1],
        output_dtype=torch.float16,
    )
    out = torch.ops.npu.npu_grouped_matmul([x.npu()], [packed_weight], **kwargs)
    torch.npu.synchronize()
    actual = out[0].cpu()
    expected, _ = torch_reference(with_offset)
    print("torch_npu output:", actual.shape, actual.dtype, actual.flatten()[:8])
    # npu_trans_quant_param truncates FP32 scale into the device FP19 field.
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=5e-2)
    print("torch_npu vs reference: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", choices=("null", "non-null"), required=True)
    parser.add_argument("--run-npu", action="store_true", help="also invoke torch_npu.npu_grouped_matmul")
    args = parser.parse_args()
    with_offset = args.offset == "non-null"

    report_torch_support()
    out, case = torch_reference(with_offset)
    _, _, scale, offset, bias, group_list, _ = case
    print(f"case: offset={args.offset}, x=[{M},{K}], weight=[{E},{K},{N}]")
    print(f"scale.shape={tuple(scale.shape)}, offset.shape={None if offset is None else tuple(offset.shape)}")
    print(f"bias.shape={tuple(bias.shape)}, groupList={group_list.tolist()}")
    print("reference output:", out.shape, out.dtype, out.flatten()[:8])
    if args.run_npu:
        run_npu(with_offset, case)
    return 0


if __name__ == "__main__":
    sys.exit(main())
