# Add MXFP4 NoPE cache records to KvQuantSparseFlashAttention

## Why

The A5 MLA path currently consumes a 656-byte, per-KV-token raw cache record
whose NoPE values are INT8/FP8/HIF8, whose 64 RoPE values are already `Q_T`,
and whose four FP32 scales are in the record tail. The B path introduces a
smaller raw record that preserves the same V0 output and all post-V0 behavior.

## What Changes

- Add an A5-only `MXFP4_NOPE_ROPE_COMBINED` cache format.
- Define `key` as an opaque `DT_INT8` storage record with last-axis width 400
  bytes: `[256B packed FP4 E2M1 NoPE][128B raw Q_T RoPE][16B E8M0 scale]`.
- Keep the existing `key_dequant_scale` and `value_dequant_scale` inputs for
  public-interface compatibility; the combined A5 B path MUST NOT consume
  `key_dequant_scale` for V0 numerical dequantization.
- Preserve sparse routing, `sparse_block_size=1`, V0's 16-row UB batch,
  split-G synchronization, V0 output layout, BMM1, softmax, BMM2, V1, and V2.

## Non-goals

- Full-D=576 MXFP4 (the former A path).
- A separate FP4 data/E8M0 scale gather path.
- Changing sparse routing, task scheduling, or core partitioning.
- Performance tuning, benchmark targets, or changing the V0 UB batch size.

## Owner decision

The owner selected the minimal B-I contract on 2026-08-24: NoPE512 is MXFP4,
RoPE64 remains raw `Q_T`, and the 16 E8M0 scales are embedded in the same
opaque key record.
