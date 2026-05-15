# Work Log - 2026-05-15

## A131 Tail Padding Duplicate

- User decision: skip A131a contract-deletion diagnostic and directly replace scalar tail padding with `Duplicate`.
- Active file before cleanup: `op_kernel/packed64local/chunk_gated_delta_rule_neumann_stage1.h`.
- Cleanup result: A131 is merged into `op_kernel/chunk_gated_delta_rule_neumann_stage1.h`; `baseline/`, `packed64local/`, and `selected/` are removed.
- Official Stage1 SHA256 after cleanup: `8d4932568da50f3118e3e496dd2508c79cd2d10951d7af53fed6165e12440984`.
- Host tiling SHA256 after cleanup: `37c7ca69868f2bf80ea9ce8d64efd1964c32cd2b9b7ec292df2676d41ee73406`.
- Change: `ApplyTailPaddingToAttn(validLen)` no longer uses per-element tail `SetValue`.
  - `validLen <= subOffset_`: zero whole `halfChunkSize_ x chunkSize_` block.
  - partial valid rows: row-start masked `Duplicate` zeros invalid columns while preserving aligned vector destination.
  - invalid rows: contiguous row block `Duplicate`.
- A first unaligned per-row-tail Duplicate attempt had failed at `S=5` with AIV UB unaligned vector access; the accepted implementation uses row-start masked Duplicate to avoid that address boundary.
- Build cache note: `../../build` initially contained stale C128-only cache (`ASCEND_OP_NAME=chunk_gated_delta_rule_C128`). Moved `../../build` and `../../build_out` aside with suffix `before_a131b_cache_polluted_20260515_110201` before rebuilding the dual-op package.

## Build And Install Evidence

- Build command: `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`.
- `../../build/CMakeCache.txt` after rebuild:
  - `ASCEND_OP_NAME=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann`
  - `COMPILED_OPS=chunk_gated_delta_rule;chunk_gated_delta_rule_neumann`
- Cleanup rebuild package SHA256: `f36da5e37df82acf4e1da9c0c05172265e0933bd9c7ca70c462fd1f69afded0a`.
- Package contains `ChunkGatedDeltaRuleNeumann_9a797d238ac079458b1d4199497996b7.o/json`.
- Installed top-level Neumann object SHA256: `45cb88e35547cd0d5181c21b26e85625160abe45ef027e80b5a9728ba17c70f6`.
- Installed top-level Neumann JSON SHA256: `4436e86e9abe9ef5f5c328323e04e676bcf8bbfbf80e2b4da757c162104ad4a0`.

## Correctness Gate

- Tail smoke PASS: `S=5/16/31/32/33/34/48/49/63/65/129`.
- Aligned smoke PASS: `S=64/256/1024`.
- Full compare PASS: `30 passed in 6.79s`.

## Profiling

- Output dir: `tests/pytest/prof_out/packed64_phaseA131_tail_padding_duplicate_20260515_8shapes/`.
- Compare CSV SHA256: `7d18ff50b66c660ffa66cb2764ddee12956b9a9c9a37141f9646d82a0957d33f`.
- Raw CSV SHA256: `13a73b00e08578679b7d04ccc20b38474bce177eb94d66a35ca8a116342d758f`.
- Neumann/normal ratios for `S=64/65/512/513/2048/2049/10240/10241`: `0.811 / 0.812 / 0.793 / 0.793 / 0.836 / 0.832 / 0.840 / 0.845`.
- Absolute Neumann task vs A130: `1.022 / 0.793 / 0.974 / 0.713 / 1.003 / 0.747 / 1.005 / 1.000`.
- Absolute Neumann task vs Tail-T2: `0.891 / 0.768 / 0.963 / 0.691 / 0.986 / 0.748 / 0.997 / 0.996`.
- Absolute Neumann task vs tailpad-disabled diagnostic: `0.982 / 1.005 / 1.004 / 0.998 / 0.986 / 0.992 / 0.999 / 0.993`.

## Cleanup

- Pre-cleanup A131 source SHA256: `966465538bf97e8a6f3966b1f59d341195d948f177197bc9a57b89fbad8f4c0f`.
- Duplicate baseline headers are no longer used after merging the packed64 implementation into the official Stage1 file.
