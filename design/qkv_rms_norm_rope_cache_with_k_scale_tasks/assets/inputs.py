"""Self-check driver for the QkvRmsNormRopeCacheWithKScale numpy golden.

Builds inputs matching the design (head_nums, query_start_loc local prefix sum,
seq_lens history offsets, fp8 caches, and the shared external rotation
matrix) and exercises cases aligned with the current torch_extension example.
Run: ``python3 inputs.py``.
"""

import numpy as np

from golden import (
    to_bf16,
    cast_to_e4m3fn_sat_rint,
    e4m3fn_to_uint8,
    apply_half_split_rope,
    qkv_rms_norm_rope_cache_with_k_scale_golden,
)

ROTATION_SEED_OFFSET = 2000003


def make_inputs(B, batch_seq_lens, seq_lens, Nq, Nk, Nv,
                D=128, block_size=16, seed=0, max_slot=None,
                random_slot_mapping=False, layout_qkv="TND",
                layout_q_out="NTD", dense_rotation=True,
                random_cache_init=False):
    layout_qkv = "TND" if layout_qkv is None or str(layout_qkv) == "" else str(layout_qkv).upper()
    layout_q_out = "NTD" if layout_q_out is None or str(layout_q_out) == "" else str(layout_q_out).upper()
    assert layout_qkv in ("NTD", "TND")
    assert layout_q_out in ("NTD", "TND")
    assert not (layout_qkv == "NTD" and layout_q_out == "TND")
    assert D == 128
    assert Nq > 0 and Nk > 0 and Nv > 0, "Nq, Nk, and Nv must be positive"
    assert Nq <= 64, "current kernel requires Nq <= 64"
    assert Nq == 8 * Nk, "current kernel requires Nq == 8 * Nk"
    assert Nk == Nv, "current kernel requires Nk == Nv"
    rng = np.random.default_rng(seed)
    assert len(batch_seq_lens) == B
    assert len(seq_lens) == B
    T = int(sum(batch_seq_lens))
    query_start_loc = np.zeros(B + 1, dtype=np.int32)
    query_start_loc[1:] = np.cumsum(batch_seq_lens)
    seq_lens = np.asarray(seq_lens, dtype=np.int32)
    assert np.all(seq_lens >= np.asarray(batch_seq_lens, dtype=np.int32))
    max_seq_len = int(np.max(seq_lens))

    H = Nq + Nk + Nv
    qkv_ntd = to_bf16(rng.standard_normal((H, T, D)).astype(np.float32) * 0.5)
    qkv = qkv_ntd if layout_qkv == "NTD" else np.transpose(qkv_ntd, (1, 0, 2)).copy()
    q_gamma = (rng.random(D, dtype=np.float32) + 0.5).astype(np.float32)
    k_gamma = (rng.random(D, dtype=np.float32) + 0.5).astype(np.float32)
    cos_sin = rng.standard_normal((max_seq_len, D)).astype(np.float32)
    if dense_rotation:
        rotation_rng = np.random.default_rng(seed + ROTATION_SEED_OFFSET)
        rotation = to_bf16(
            rotation_rng.standard_normal((D, D)).astype(np.float32) * (D ** -0.5)
        )
    else:
        rotation = to_bf16(np.eye(D, dtype=np.float32))
    v_scale = (rng.random(Nv, dtype=np.float32) + 0.5).astype(np.float32)

    # cache big enough for all referenced slots (default: T contiguous slots)
    highest_slot = (T - 1) if max_slot is None else int(max_slot)
    block_num = highest_slot // block_size + 2
    if random_slot_mapping:
        assert highest_slot + 1 >= T
        slot_mapping = rng.choice(highest_slot + 1, size=T, replace=False).astype(np.int32)
    else:
        slot_mapping = np.arange(T, dtype=np.int32)

    if random_cache_init:
        k_cache_init = rng.standard_normal((block_num, Nk, block_size, D)).astype(np.float32)
        v_cache_init = rng.standard_normal((block_num, Nv, block_size, D)).astype(np.float32)
        k_scale_cache = rng.standard_normal((block_num, Nk, block_size, 1)).astype(np.float32)
    else:
        k_cache_init = np.zeros((block_num, Nk, block_size, D), dtype=np.float32)
        v_cache_init = np.zeros((block_num, Nv, block_size, D), dtype=np.float32)
        k_scale_cache = np.zeros((block_num, Nk, block_size, 1), dtype=np.float32)
    k_cache = cast_to_e4m3fn_sat_rint(k_cache_init)
    v_cache = cast_to_e4m3fn_sat_rint(v_cache_init)

    return dict(
        qkv=qkv, q_gamma=q_gamma, k_gamma=k_gamma, cos_sin=cos_sin,
        slot_mapping=slot_mapping, k_cache=k_cache, v_cache=v_cache,
        k_scale_cache=k_scale_cache, query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        rotation=rotation, v_scale=v_scale,
        head_nums=[Nq, Nk, Nv], layout_qkv=layout_qkv,
        layout_q_out=layout_q_out, epsilon=1e-6,
    )


def run(name, inp):
    k_in = inp["k_cache"].copy()
    v_in = inp["v_cache"].copy()
    q_out, q_scale, k_cache, v_cache, k_scale_cache = \
        qkv_rms_norm_rope_cache_with_k_scale_golden(**inp)

    layout = inp.get("layout_qkv", "TND")
    out_layout = inp.get("layout_q_out", "NTD")
    T = inp["qkv"].shape[1] if layout == "NTD" else inp["qkv"].shape[0]
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

    print(f"[{name}] layout_qkv={layout} layout_q_out={out_layout} "
          f"T={T} heads=({Nq},{Nk},{Nv}) D={D} "
          f"q_out{tuple(q_out.shape)} q_scale{tuple(q_scale.shape)} "
          f"| alias_preserved={preserved} fp8_ok={fp8_ok} "
          f"q_scale[min,max]=[{q_scale.min():.4g},{q_scale.max():.4g}]")
    assert preserved and fp8_ok
    return q_out, q_scale, k_cache, v_cache, k_scale_cache


def main():
    # smoke: B=1, S=1
    run("smoke", make_inputs(1, [1], [1], 16, 2, 2))

    # normal: B=1, S=128
    run("normal", make_inputs(1, [128], [128], 32, 4, 4, seed=1))

    # TND layout: qkv/q_out/q_scale keep token-major N/T order at the interface
    run("tnd-normal",
        make_inputs(1, [16], [16], 16, 2, 2, seed=9, layout_qkv="TND", layout_q_out="TND"))

    # TND input with NTD Q output: qkv is token-major, q_out/q_scale are head-major.
    run("tnd-to-ntd",
        make_inputs(1, [13], [13], 16, 2, 2, seed=12,
                    layout_qkv="TND", layout_q_out="NTD"))

    # multi-batch: B=2, unequal lengths -> exercises query_start_loc local prefix RoPE
    run("multi-batch", make_inputs(2, [2, 7], [2, 7], 16, 2, 2, seed=2))

    # multi-batch offset: position jumps from 8208 back to 0 at the batch boundary
    run("multi-batch-offset",
        make_inputs(2, [17, 1], [8209, 1], 16, 2, 2, seed=7))

    # head combos
    run("head-combo", make_inputs(1, [64], [64], 64, 8, 8, seed=3))

    # Dense random rotation is the default; keep an explicit dense case in the
    # self-check to make the non-diagonal Cube path visible.
    run("dense-rotation",
        make_inputs(1, [8], [8], 16, 2, 2, seed=10, dense_rotation=True))

    # non-contiguous / cross-block slot_mapping
    inp = make_inputs(1, [8], [8], 16, 2, 2, seed=4, max_slot=60)
    inp["slot_mapping"] = np.array([3, 31, 5, 48, 0, 17, 9, 60], dtype=np.int32)
    run("noncontig-slots", inp)

    # random non-contiguous / cross-block slot_mapping, deterministic by seed
    run("random-slots",
        make_inputs(1, [8], [8], 16, 2, 2, seed=8,
                    max_slot=60, random_slot_mapping=True))

    # Random initial cache contents verify input/output alias semantics.
    run("random-cache-init",
        make_inputs(1, [8], [8], 16, 2, 2, seed=11,
                    max_slot=60, random_slot_mapping=True,
                    random_cache_init=True))

    # v_scale == 0 (legal multiplier -> V quantizes to 0)
    inp = make_inputs(1, [4], [4], 16, 2, 2, seed=5)
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
    inp = make_inputs(1, [4], [4], 16, 2, 2, seed=6)
    inp["qkv"][:, 0, :] = 0.0          # token 0: all Q/K/V heads zero
    q_out, q_scale, _, _, _ = run("max_abs==0", inp)
    nan_frac = np.mean(np.isnan(q_out[:, 0, :]))
    print(f"    -> token0 q_scale==0 rows: {int(np.sum(q_scale[:, 0] == 0))}/{q_scale.shape[0]}, "
          f"token0 q_out NaN fraction: {nan_frac:.2f} (no zero-row guard, as designed)")

    # q_rope/k_rope helper sanity: Q and K RoPE outputs are materialized
    # separately but must use the same half-split RoPE equation and actual length offsets.
    y = np.array([[[1., 2., 3., 4.]], [[-1., -2., 5., 6.]]], dtype=np.float32)
    cs = np.array([[0.5, 0.25, 2.0, 4.0], [1.5, -0.5, -2.0, 0.75]], dtype=np.float32)
    rot = apply_half_split_rope(
        y, cs, np.array([0, 2], dtype=np.int32), np.array([2], dtype=np.int32))
    expect = np.array([[[-5.5, -15.5, 3.5, 9.0]], [[8.5, -3.5, 9.5, -4.5]]], dtype=np.float32)
    print(f"    -> q/k half-split RoPE helper max diff: {np.max(np.abs(rot - expect)):.4g}")
    assert np.allclose(rot, expect)

    # bit-level encode sanity: a few canonical FP8 values
    vals = np.array([0.0, 1.0, -2.0, 448.0, 0.5, 2.0 ** -9, 1000.0], dtype=np.float32)
    enc = e4m3fn_to_uint8(cast_to_e4m3fn_sat_rint(vals))
    print(f"\nFP8 encode check {vals.tolist()} -> {[hex(b) for b in enc]}")
    # 1.0 -> exp=7 (0b0111), man=0 -> 0x38 ; 448 -> 0x7E ; 1000 saturates -> 0x7E
    assert enc[1] == 0x38 and enc[3] == 0x7E and enc[6] == 0x7E

    print("\nAll golden self-checks passed.")


if __name__ == "__main__":
    main()
