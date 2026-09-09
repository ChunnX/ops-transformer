# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Run with separately selected upstream/local operator packages on Ascend950.

Baseline: python check_local_matmul.py --save /tmp/fa-baseline.pt
Candidate: python check_local_matmul.py --reference /tmp/fa-baseline.pt
The selected package must contain FP16/BF16 ND configs 0..7 and DN configs 0/2/6.
"""
import argparse

import torch
import torch_npu  # noqa: F401
from cann_ops_transformer.ops import flash_attn, flash_attn_metadata


def run_cases(whitebox=False):
    torch.npu.set_device(0)
    results = {}
    shapes = [(1, 257, 3, 1, 1), (17, 129, 3, 1, 1), (128, 513, 8, 1, 1)]
    if whitebox:
        # M=1/16/17/31/32/33/63/64/65; S2=1/127/128/129/255/256/257.
        # Multiple rows and batches exercise persistent slots and task draining.
        shapes += [(1, 1, 1, 1, 1), (16, 127, 1, 1, 1), (17, 128, 1, 1, 1),
                   (31, 129, 1, 1, 1), (32, 255, 1, 1, 1), (33, 256, 1, 1, 1),
                   (63, 257, 1, 1, 1), (64, 128, 1, 1, 1), (65, 129, 1, 1, 1),
                   (129, 1025, 6, 2, 2)]
    for dtype in (torch.float16, torch.bfloat16):
        for d, dv in ((64, 64), (72, 72), (128, 128), (192, 128), (256, 256)):
            for sq, sk, nq, batch, nkv in shapes:
                torch.manual_seed(417)
                q = torch.randn(batch, sq, nq, d, dtype=dtype)
                k = torch.randn(batch, sk, nkv, d, dtype=dtype)
                v = torch.randn(batch, sk, nkv, dv, dtype=dtype)
                options = dict(
                    max_seqlen_q=sq, max_seqlen_kv=sk, mask_mode=0,
                    win_left=-1, win_right=-1,
                    layout_q="BSND", layout_kv="BSND", layout_out="BSND",
                )
                metadata = flash_attn_metadata(
                    batch_size=batch, num_heads_q=nq, num_heads_kv=nkv,
                    head_dim=d, head_dim_v=dv, **options,
                )
                result = flash_attn(
                    q.npu(), k.npu(), v.npu(), metadata=metadata,
                    softmax_scale=d ** -0.5, return_softmax_lse=True, **options,
                )
                torch.npu.synchronize()
                name = f"{dtype}_{d}_{dv}_{sq}_{sk}_{nq}"
                if whitebox:
                    name += f"_b{batch}_kv{nkv}"
                results[name] = [tensor.cpu() for tensor in result]
                assert len(results[name]) == 2, "Expected attention output and LSE"
                assert all(torch.isfinite(t).all() for t in results[name]), name
                reference = torch.nn.functional.scaled_dot_product_attention(
                    q.float().transpose(1, 2),
                    k.float().transpose(1, 2).repeat_interleave(nq // nkv, dim=1),
                    v.float().transpose(1, 2).repeat_interleave(nq // nkv, dim=1),
                ).transpose(1, 2)
                # Smoke precision check; exact upstream comparison below is the
                # regression criterion for this instruction-preserving refactor.
                actual = results[name][0].float()
                atol = 0.015 if dtype == torch.bfloat16 else 0.003
                assert torch.allclose(actual, reference, atol=atol, rtol=0.015), name
                print(f"{name}: PASS, max_abs={(actual - reference).abs().max().item()}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", help="Save outputs from the selected package")
    parser.add_argument("--reference", help="Compare outputs and LSE exactly with a saved baseline")
    parser.add_argument("--whitebox", action="store_true", help="Exercise tile and task boundaries (130 cases)")
    args = parser.parse_args()
    results = run_cases(args.whitebox)
    if args.reference:
        reference = torch.load(args.reference, map_location="cpu", weights_only=True)
        assert results.keys() == reference.keys(), "Baseline case set differs"
        for name, outputs in results.items():
            assert len(outputs) == len(reference[name]), name
            for actual, expected in zip(outputs, reference[name]):
                assert torch.equal(actual, expected), f"{name}: output/LSE differs from upstream"
        print(f"Exact upstream comparison: {len(results)} cases PASS")
    if args.save:
        torch.save(results, args.save)


if __name__ == "__main__":
    main()
