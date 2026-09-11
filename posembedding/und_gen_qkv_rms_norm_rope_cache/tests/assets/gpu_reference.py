#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""PyTorch GPU reference for TTK XPU cross-check and a standalone smoke test.

Run on a CUDA machine:
    python gpu_reference.py --device cuda:0
CPU-only development check (does not validate CUDA):
    python gpu_reference.py --device cpu

The reference uses eager PyTorch operations on the input device. It is not a
fused CUDA performance baseline. No CPU golden helpers are used in computation.
"""

import argparse

import torch


def gpu_reference(
    und_qkv,
    und_weights_q,
    und_weights_k,
    cos_sin_cache,
    k_cache,
    v_cache,
    slot_mapping,
    positions,
    gen_qkv=None,
    gen_weights_q=None,
    gen_weights_k=None,
    cat_indices=None,
    num_heads_q=8,
    num_heads_k=1,
    num_heads_v=1,
    norm_eps=1e-6,
    mrope_section=None,
    **kwargs,
):
    """Return (q, k_cache, v_cache) on the input device, updating caches in place.

    TTK supplies validated inputs and unique, in-range slots. Q/K normalization
    and rotation use FP32; V is copied unchanged. Unlike the CPU golden, RoPE
    uses ordinary PyTorch multiply/add operations rather than emulated FMA.
    """
    heads = num_heads_q + num_heads_k + num_heads_v
    head_dim = und_qkv.shape[-1] if und_qkv.ndim == 3 else und_qkv.shape[-1] // heads
    und = und_qkv.reshape(und_qkv.shape[0], heads, head_dim)
    inputs = (
        und
        if gen_qkv is None
        else torch.cat(
            (und, gen_qkv.reshape(gen_qkv.shape[0], heads, und.shape[-1])), dim=0
        )
    )
    src = (
        torch.arange(inputs.shape[0], device=inputs.device)
        if cat_indices is None
        else cat_indices.long()
    )
    rows = inputs.index_select(0, src).float()
    q, k, v = torch.split(rows, (num_heads_q, num_heads_k, num_heads_v), dim=1)
    is_und = (src < und.shape[0]).reshape(-1, 1, 1)
    half = und.shape[-1] // 2
    section = (
        [half, 0, 0]
        if mrope_section is None or len(mrope_section) == 0
        else mrope_section
    )
    if len(section) != 3 or min(section) < 0 or sum(section) > half:
        raise ValueError(
            "mrope_section must have three nonnegative entries summing to <= D/2"
        )
    dims = torch.arange(half, device=inputs.device)
    axes = torch.zeros_like(dims)
    axes = torch.where((dims % 3 == 1) & (dims < 3 * section[1]), 1, axes)
    axes = torch.where((dims % 3 == 2) & (dims < 3 * section[2]), 2, axes)
    pos = positions.long()
    if pos.ndim == 1:
        pos = pos.unsqueeze(0).expand(3, -1)
    selected_pos = pos.index_select(0, axes).transpose(0, 1)
    cache = cos_sin_cache.float()
    cos = cache[selected_pos, dims].unsqueeze(1)
    sin = cache[selected_pos, dims + half].unsqueeze(1)

    def normalize_rotate(x, und_weight, gen_weight):
        gen_weight = und_weight if gen_weight is None else gen_weight
        weight = torch.where(is_und, und_weight.float(), gen_weight.float())
        x = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + norm_eps) * weight
        lo, hi = x.chunk(2, dim=-1)
        return torch.cat((lo * cos - hi * sin, hi * cos + lo * sin), dim=-1)

    q = normalize_rotate(q, und_weights_q, gen_weights_q)
    k = normalize_rotate(k, und_weights_k, gen_weights_k)
    slots = slot_mapping.long()
    k_cache[slots // k_cache.shape[1], slots % k_cache.shape[1]] = k.to(k_cache.dtype)
    v_cache[slots // v_cache.shape[1], slots % v_cache.shape[1]] = v.to(v_cache.dtype)
    return q.to(torch.bfloat16), k_cache, v_cache


def smoke_test(device):
    """Check both head configurations, indexing, optional inputs and cache holes."""
    from golden import golden_und_gen_qkv_rms_norm_rope_cache, make_cos_sin_cache

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; use a CUDA-enabled PyTorch environment"
        )
    rng = torch.Generator().manual_seed(2026)

    def rand(*shape):
        return torch.randn(*shape, generator=rng).to(torch.bfloat16)

    count = 0
    for hq, hk, hv in ((8, 1, 1), (16, 2, 2)):
        for mode in ("und_only", "mixed", "repeated", "gen_only"):
            und_len, gen_len = (0, 3) if mode == "gen_only" else (5, 3)
            if mode == "und_only":
                gen_len = 0
            total = und_len + gen_len
            cat = None
            if mode == "mixed":
                cat = torch.arange(total - 1, -1, -1)
            elif mode == "repeated":
                cat = torch.tensor([7, 0, 7, 3, 1, 5, 0, 6])
            slots = torch.randperm(21, generator=rng)[:total]
            pos_shape = (total,) if mode == "und_only" else (3, total)
            args = dict(
                und_qkv=rand(und_len, hq + hk + hv, 128),
                und_weights_q=rand(128),
                und_weights_k=rand(128),
                cos_sin_cache=make_cos_sin_cache(32, 128),
                k_cache=rand(3, 7, hk, 128),
                v_cache=rand(3, 7, hv, 128),
                slot_mapping=slots,
                positions=torch.randint(32, pos_shape, generator=rng),
                gen_qkv=rand(gen_len, hq + hk + hv, 128) if gen_len else None,
                gen_weights_q=rand(128) if gen_len else None,
                gen_weights_k=rand(128) if gen_len else None,
                cat_indices=cat,
                num_heads_q=hq,
                num_heads_k=hk,
                num_heads_v=hv,
                mrope_section=[] if mode == "und_only" else [16, 16, 16],
            )
            expected = golden_und_gen_qkv_rms_norm_rope_cache(**args)
            placed = {
                key: value.to(device).clone()
                if isinstance(value, torch.Tensor)
                else value
                for key, value in args.items()
            }
            with torch.no_grad():
                actual = gpu_reference(**placed)
            # Compare written rows separately so unused cache capacity cannot dilute errors.
            for idx, (out, ref) in enumerate(zip(actual, expected)):
                assert out.device == placed["und_qkv"].device
                assert out.shape == ref.shape and out.dtype == ref.dtype
                if idx:
                    assert out is placed[("k_cache", "v_cache")[idx - 1]]
                    out = out.cpu().flatten(0, 1)
                    ref = ref.flatten(0, 1)
                    untouched = torch.ones(21, dtype=torch.bool)
                    untouched[slots] = False
                    torch.testing.assert_close(
                        out[untouched], ref[untouched], rtol=0, atol=0
                    )
                    out, ref = out[slots], ref[slots]
                close = torch.isclose(
                    out.cpu().float(), ref.float(), rtol=2**-7, atol=1e-4
                )
                ratio = (~close).float().mean().item()
                assert ratio <= 0.001, f"{mode} output {idx}: mismatch ratio {ratio}"
                if idx == 2:
                    torch.testing.assert_close(out, ref, rtol=0, atol=0)
            count += 1
            print(f"PASS {device} heads={hq}/{hk}/{hv} {mode}")
    print(f"{count} cases passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="cuda:0", help="CUDA device; cpu for local logic checks"
    )
    smoke_test(parser.parse_args().device)
