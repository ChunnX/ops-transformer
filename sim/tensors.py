# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU-only deterministic fixtures; all floating point formats are encoded as raw bytes."""

import numpy as np


def bf16(values):
    bits = np.asarray(values, dtype="<f4").view("<u4")
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")


def routing(cfg, rank):
    m = cfg["model"]
    b, k = m["tokens_per_rank"], m["top_k"]
    experts = m["ep"] * m["moe_experts_per_rank"]
    # Flattened route index cycles through all experts. No duplicate expert in a row.
    # For the default case: 18 tokens/source/expert, 144 tokens/expert globally.
    global_token = rank * b + np.arange(b, dtype=np.int64)
    return ((global_token[:, None] * k + np.arange(k)) % experts).astype("<i4")


def fp4_nz_encode(raw):
    """[E,N,K] E2M1 nibbles -> [E,K/32,N/16,16,16 bytes], low nibble first."""
    e, n, k = raw.shape
    if n % 16 or k % 32 or np.any(raw > 15):
        raise ValueError("FP4 NZ needs N%16=K%32=0 and raw nibbles in 0..15")
    nz = raw.reshape(e, n // 16, 16, k // 32, 32).transpose(0, 3, 1, 2, 4)
    return np.ascontiguousarray(nz[..., 0::2] | (nz[..., 1::2] << 4))


def fp4_nz_decode(packed, e, n, k):
    packed = np.asarray(packed, dtype=np.uint8).reshape(e, k // 32, n // 16, 16, 16)
    raw = np.empty((*packed.shape[:-1], 32), dtype=np.uint8)
    raw[..., 0::2], raw[..., 1::2] = packed & 15, packed >> 4
    return raw.transpose(0, 2, 3, 1, 4).reshape(e, n, k)


def inputs(cfg, rank):
    m, fixture = cfg["model"], cfg["fixture"]
    b, h, k = m["tokens_per_rank"], m["hidden"], m["top_k"]
    rng = np.random.default_rng(np.random.SeedSequence([fixture["seed"], rank, 0]))
    # Exact finite E4M3 values: +/-0.5, +/-1, +/-2. Never arbitrary uint8/NaN data.
    choices = np.array([0x30, 0x38, 0x40, 0xB0, 0xB8, 0xC0], np.uint8)
    if fixture["kind"] == "random":
        x = rng.choice(choices, (b, h))
    else:
        x = np.zeros((b, h), np.uint8)
        x[:, 0] = choices[(np.arange(b) + rank) % len(choices)]
        x[:, 1] = choices[(np.arange(b) * 3 + rank + 1) % len(choices)]
    scales = np.full((b, h // 64, 2), 127 + fixture["scale_exponent"], np.uint8)
    gates = np.full((b, k), 1 / k, dtype="<f4")
    if cfg.get("topk_weights_dtype", "bfloat16") == "bfloat16":
        gates = bf16(gates)
    return {
        1: ("x", x),
        2: ("topk_ids", routing(cfg, rank)),
        3: ("topk_weights", gates),
        11: ("x_scales", scales),
    }


def expert_tensors(cfg, rank, shared=False):
    m, fixture = cfg["model"], cfg["fixture"]
    h, n = m["hidden"], m["intermediate"]
    e = m["shared_experts_per_rank" if shared else "moe_experts_per_rank"]
    rng = np.random.default_rng(
        np.random.SeedSequence([fixture["seed"], rank, 2 if shared else 1])
    )
    # Raw exact +/-0.5, +/-1, +/-2 in E4M3 (shared) or E2M1 (routed).
    values = np.array(
        [0x30, 0x38, 0x40, 0xB0, 0xB8, 0xC0] if shared else [1, 2, 4, 9, 10, 12],
        np.uint8,
    )
    if fixture["kind"] == "random":
        w1, w2 = rng.choice(values, (e, 2 * n, h)), rng.choice(values, (e, h, n))
    else:
        w1, w2 = np.zeros((e, 2 * n, h), np.uint8), np.zeros((e, h, n), np.uint8)
        for expert in range(e):
            w1[expert, :n, 0] = values[(np.arange(n) + expert + rank) % 3]
            w1[expert, n:, 1] = values[(np.arange(n) + 2 * expert + rank) % 6]
            w2[expert, np.arange(h), np.arange(h) % n] = values[
                (np.arange(h) + expert + rank) % 6
            ]
    if not shared:
        w1, w2 = fp4_nz_encode(w1), fp4_nz_encode(w2)
    s1 = np.full((e, 2 * n, h // 64, 2), 127 + fixture["scale_exponent"], np.uint8)
    s2 = np.full((e, h, n // 64, 2), 127 + fixture["scale_exponent"], np.uint8)
    start = 12 if shared else 4
    prefix = "shared_" if shared else "moe_"
    for slot, name, tensor in zip(
        range(start, start + 4), ("w1", "w2", "s1", "s2"), (w1, w2, s1, s2)
    ):
        yield slot, prefix + name, tensor
