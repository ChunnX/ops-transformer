"""Self-check driver for the QkvRmsNormRopeCacheWithKScale numpy golden.

Builds inputs matching the design (head_nums, cu_seq_len prefix sum, fp8 caches,
external dense rotation) and exercises the recommended cases from
``03_golden_design.md`` section 5: smoke / multi-batch / non-contiguous slots /
v_scale==0 / max_abs==0. Run: ``python3 inputs.py``.
"""

import numpy as np

from golden import (
    to_bf16,
    cast_to_e4m3fn_sat_rint,
    e4m3fn_to_uint8,
    qkv_rms_norm_rope_cache_with_kscale_golden,
)


def make_inputs(B, seq_lens, Nq, Nk, Nv, D=128, block_size=16, seed=0, max_slot=None):
    rng = np.random.default_rng(seed)
    assert len(seq_lens) == B
    T = int(sum(seq_lens))
    cu_seq_len = np.zeros(B + 1, dtype=np.int32)
    cu_seq_len[1:] = np.cumsum(seq_lens)
    max_seq_len = int(max(seq_lens))

    H = Nq + Nk + Nv
    qkv = to_bf16(rng.standard_normal((T, H, D)).astype(np.float32))
    q_gamma = rng.standard_normal(D).astype(np.float32)
    k_gamma = rng.standard_normal(D).astype(np.float32)
    cos_sin = rng.standard_normal((max_seq_len, D)).astype(np.float32)
    rotation = to_bf16(rng.standard_normal((D, D)).astype(np.float32) * 0.1)
    v_scale = rng.uniform(0.1, 1.0, size=Nv).astype(np.float32)

    # cache big enough for all referenced slots (default: T contiguous slots)
    highest_slot = (T - 1) if max_slot is None else int(max_slot)
    block_num = highest_slot // block_size + 2
    slot_mapping = np.arange(T, dtype=np.int32)

    k_cache = cast_to_e4m3fn_sat_rint(
        rng.standard_normal((block_num, Nk, block_size, D)).astype(np.float32))
    v_cache = cast_to_e4m3fn_sat_rint(
        rng.standard_normal((block_num, Nv, block_size, D)).astype(np.float32))
    k_scale_cache = rng.standard_normal(
        (block_num, Nk, block_size, 1)).astype(np.float32)

    return dict(
        qkv=qkv, q_gamma=q_gamma, k_gamma=k_gamma, cos_sin=cos_sin,
        slot_mapping=slot_mapping, k_cache=k_cache, v_cache=v_cache,
        k_scale_cache=k_scale_cache, cu_seq_len=cu_seq_len, rotation=rotation,
        v_scale=v_scale, head_nums=[Nq, Nk, Nv], epsilon=1e-6,
    )


def run(name, inp):
    k_in = inp["k_cache"].copy()
    v_in = inp["v_cache"].copy()
    q_out, q_scale, k_cache, v_cache, k_scale_cache = \
        qkv_rms_norm_rope_cache_with_kscale_golden(**inp)

    T = inp["qkv"].shape[0]
    Nq, Nk, Nv = inp["head_nums"]
    D = inp["qkv"].shape[2]
    bs = inp["k_cache"].shape[2]
    slot = inp["slot_mapping"]

    # untouched cache slots must be preserved (alias semantics)
    touched = {(int(s) // bs, int(s) % bs) for s in slot}
    bn = k_in.shape[0]
    preserved = True
    for b in range(bn):
        for o in range(bs):
            if (b, o) not in touched:
                if not np.array_equal(k_cache[b, :, o, :], k_in[b, :, o, :]):
                    preserved = False
                if not np.array_equal(v_cache[b, :, o, :], v_in[b, :, o, :]):
                    preserved = False

    # All outputs must be FP8-E4M3FN-representable. NaN is itself a valid E4M3FN
    # pattern (the max_abs==0 row), so equal_nan=True accepts it without any
    # per-case special handling — abnormal-value policy belongs to golden.py.
    def is_fp8(a):
        return np.array_equal(a, cast_to_e4m3fn_sat_rint(a), equal_nan=True)
    fp8_ok = is_fp8(q_out) and is_fp8(k_cache) and is_fp8(v_cache)

    print(f"[{name}] T={T} heads=({Nq},{Nk},{Nv}) D={D} "
          f"q_out{tuple(q_out.shape)} q_scale{tuple(q_scale.shape)} "
          f"| alias_preserved={preserved} fp8_ok={fp8_ok} "
          f"q_scale[min,max]=[{q_scale.min():.4g},{q_scale.max():.4g}]")
    assert preserved and fp8_ok
    return q_out, q_scale, k_cache, v_cache, k_scale_cache


def main():
    # smoke: B=1, S=1
    run("smoke", make_inputs(1, [1], 16, 2, 2))

    # normal: B=1, S=128
    run("normal", make_inputs(1, [128], 32, 4, 4, seed=1))

    # multi-batch: B=2, unequal lengths -> exercises cu_seq_len prefix sum RoPE
    run("multi-batch", make_inputs(2, [2, 7], 16, 2, 2, seed=2))

    # head combos
    run("head-combo", make_inputs(1, [64], 64, 8, 8, seed=3))

    # non-contiguous / cross-block slot_mapping
    inp = make_inputs(1, [8], 16, 2, 2, seed=4, max_slot=60)
    inp["slot_mapping"] = np.array([3, 31, 5, 48, 0, 17, 9, 60], dtype=np.int32)
    run("noncontig-slots", inp)

    # v_scale == 0 (legal multiplier -> V quantizes to 0)
    inp = make_inputs(1, [4], 16, 2, 2, seed=5)
    inp["v_scale"][:] = 0.0
    _, _, _, v_cache, _ = run("v_scale==0", inp)
    bs = inp["k_cache"].shape[2]
    slot = inp["slot_mapping"]
    Nv = inp["head_nums"][2]
    zeros_ok = all(
        np.all(v_cache[int(s) // bs, nv, int(s) % bs, :] == 0.0)
        for s in slot for nv in range(Nv))
    print(f"    -> all written V slots == 0: {zeros_ok}")
    assert zeros_ok

    # max_abs == 0 row: force one qkv token row to zero so the whole Q/K row is
    # all-zero after the chain. The scale==0 / NaN policy is owned by
    # golden.dynamic_quant_e4m3fn (no zero-row guard); here we just confirm it.
    inp = make_inputs(1, [4], 16, 2, 2, seed=6)
    inp["qkv"][0, :, :] = 0.0          # token 0: all Q/K/V heads zero
    q_out, q_scale, _, _, _ = run("max_abs==0", inp)
    nan_frac = np.mean(np.isnan(q_out[0]))
    print(f"    -> token0 q_scale==0 rows: {int(np.sum(q_scale[0] == 0))}/{q_scale.shape[1]}, "
          f"token0 q_out NaN fraction: {nan_frac:.2f} (no zero-row guard, as designed)")

    # bit-level encode sanity: a few canonical FP8 values
    vals = np.array([0.0, 1.0, -2.0, 448.0, 0.5, 2.0 ** -9, 1000.0], dtype=np.float32)
    enc = e4m3fn_to_uint8(cast_to_e4m3fn_sat_rint(vals))
    print(f"\nFP8 encode check {vals.tolist()} -> {[hex(b) for b in enc]}")
    # 1.0 -> exp=7 (0b0111), man=0 -> 0x38 ; 448 -> 0x7E ; 1000 saturates -> 0x7E
    assert enc[1] == 0x38 and enc[3] == 0x7E and enc[6] == 0x7E

    print("\nAll golden self-checks passed.")


if __name__ == "__main__":
    main()
