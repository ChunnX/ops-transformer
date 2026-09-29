# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generalization cases, bounded fixtures, and reusable CPU reference weights."""

from __future__ import annotations
import dataclasses
import torch
from attn_prologue_golden import (
    AttnPrologueInputs,
    apply_tail_rope,
    dequantize_mxfp8,
    make_golden_inputs,
    make_paired_scale,
    make_rope_tables_for_positions,
    quantize_mxfp8,
    rms_norm,
)

TARGET_PROFILE = (5120, 1280, 64, 512, 64)


def device_input(name, value, device, to_nz):
    """Move an ND fixture to the public API, preserving encoded FP8 bytes.

    Legacy int8/uint8 payloads already contain FP8 encodings: reinterpret them,
    never numerically cast them. Only the three weights require NZ storage.
    """
    weights = ("wqa", "wqb", "wkv")
    if name in ("x",) + weights:
        dtype = torch.float8_e4m3fn
    elif name in ("descale_x", "descale_wqa", "descale_wqb", "descale_wkv"):
        dtype = torch.float8_e8m0fnu
    else:
        return value.to(device)
    if value.dtype not in (torch.int8, torch.uint8, dtype):
        raise TypeError(f"{name} must contain encoded {dtype} bytes, got {value.dtype}")
    payload = value.contiguous().view(torch.int8).to(device)
    if name in weights:
        payload = to_nz(payload)
    return payload.view(dtype)


@dataclasses.dataclass(frozen=True)
class Case:
    phase: str
    batch: int
    s: int
    H: int = 5120
    R: int = 1280
    N: int = 64
    D: int = 512
    Dr: int = 64

    @property
    def T(self):
        return self.batch * self.s

    @property
    def case_id(self):
        return f"{self.phase}-b{self.batch}-s{self.s}-t{self.T}"

    def record(self):
        return dict(
            case_id=self.case_id,
            **dataclasses.asdict(self),
            T=self.T,
            mtp_extra_tokens=5 if self.phase == "decode" and self.s == 6 else 0,
            expected_template="split_k" if self.phase == "decode" else "split_t",
        )


def all_cases():
    """Return the 12 decode and 20 prefill production-shape cases."""
    return [Case("decode", b, s) for b in (1, 4, 8, 12, 16, 32) for s in (1, 6)] + [
        Case("prefill", b, s)
        for b in (1, 4, 8, 16, 32)
        for s in (1024, 2048, 4096, 8192)
    ]


class SharedWeights:
    """Generate weights once per profile; CPU dequantization is lazy and cached."""

    def __init__(self, profile=TARGET_PROFILE, seed=2026):
        self.profile = profile
        h, r, n, d, dr = profile
        fixture = make_golden_inputs(
            1,
            hidden_size=h,
            rank_size=r,
            head_count=n,
            head_size=d,
            rope_width=dr,
            seed=seed,
        )
        self.fields = {
            f.name: getattr(fixture, f.name)
            for f in dataclasses.fields(fixture)
            if f.name.startswith("w")
            or f.name.startswith("descale_w")
            or f.name.startswith("norm_weight")
        }
        self._dequantized = None
        self._device = {}

    def dequantized(self):
        if self._dequantized is None:
            self._dequantized = {
                name: dequantize_mxfp8(
                    self.fields[name], self.fields[f"descale_{name}"]
                )
                for name in ("wqa", "wqb", "wkv")
            }
        return self._dequantized

    def device_fields(self, device, to_nz):
        if device not in self._device:
            self._device[device] = {
                name: device_input(name, value, device, to_nz)
                for name, value in self.fields.items()
            }
        return self._device[device]


def make_case_inputs(case: Case, weights, *, seed=2026, generation_chunk=1024):
    """Host inputs only, O(T*H) bytes; never allocate a T*H floating tensor.

    Prefix/suffix cache slots are randomized and indices permuted across pages.
    All cases keep at least one full page unmodified, including the largest T.
    """
    h, r, n, d, dr = weights.profile
    t = case.T
    generator = torch.Generator().manual_seed(seed + case.batch * 100003 + case.s)
    slots = ((t + 127) // 128 + 1) * 128
    index = torch.randperm(slots, generator=generator)[:t].contiguous()
    x = torch.empty((t, h), dtype=torch.float8_e4m3fn)
    scales = torch.empty((t, h // 64, 2), dtype=torch.int8)
    for start in range(0, t, generation_chunk):
        stop = min(t, start + generation_chunk)
        x[start:stop] = (
            torch.randn((stop - start, h), generator=generator) * 0.125
        ).to(torch.float8_e4m3fn)
        scales[start:stop] = make_paired_scale(stop - start, h, generator)
    positions = torch.arange(case.s).repeat(case.batch)
    if case.phase == "decode":
        positions += 8192 - case.s
    sin, cos = make_rope_tables_for_positions(positions, dr)
    cache = torch.randint(
        0, 256, (slots // 128, 128, 1, d), dtype=torch.uint8, generator=generator
    )
    return AttnPrologueInputs(
        x=x,
        descale_x=scales,
        rope_sin=sin,
        rope_cos=cos,
        cache_index=index,
        kv_cache=cache,
        **weights.fields,
    )


def golden_chunks(inputs, weights, *, row_chunk=256, norm_eps=1e-6):
    """Yield independent reference results, retaining only one chunk of trace.

    Total q is never allocated. Every row is evaluated, including all prefill
    rows. Uses only the independent reference math; never imports device kernel.
    """
    if row_chunk <= 0:
        raise ValueError("row_chunk must be positive")
    w = weights.dequantized()
    t = inputs.x.shape[0]
    _, _, n, d, _ = weights.profile
    for start in range(0, t, row_chunk):
        stop = min(t, start + row_chunk)
        rows = slice(start, stop)
        x = dequantize_mxfp8(inputs.x[rows], inputs.descale_x[rows])
        qa = x @ w["wqa"].T
        qr_norm = rms_norm(qa, inputs.norm_weight_qr, norm_eps)
        qr, scales, normalized = quantize_mxfp8(qr_norm)
        qb = dequantize_mxfp8(qr, scales) @ w["wqb"].T
        q = apply_tail_rope(
            qb.reshape(stop - start, n, d), inputs.rope_sin[rows], inputs.rope_cos[rows]
        ).to(torch.bfloat16)
        kv = rms_norm(x @ w["wkv"].T, inputs.norm_weight_kv, norm_eps)
        kv_roped = apply_tail_rope(kv, inputs.rope_sin[rows], inputs.rope_cos[rows])
        kv_cast = kv_roped.clamp(-448.0, 448.0)
        yield (
            rows,
            dict(
                q=q,
                qr=qr,
                descale_qr=scales,
                qr_normalized=normalized,
                kv_bytes=kv_cast.to(torch.float8_e4m3fn).view(torch.uint8),
                kv_roped=kv_cast,
            ),
        )
