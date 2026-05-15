# Packed64 Tail Padding Next Step Plan

Date: 2026-05-15

## Current State

- Active Stage1 path after cleanup: `op_kernel/chunk_gated_delta_rule_neumann_stage1.h`.
- Pre-cleanup correctness-safe input: A130 source recorded in `todo.md` and earlier dated logs. Duplicate baseline headers are retired after merging packed64 into the official Stage1 file.
- A130 8-shape ratios: `0.798 / 1.037 / 0.811 / 1.091 / 0.838 / 1.120 / 0.835 / 0.849`.
- Diagnostic-only run disabled `ApplyTailPaddingToAttn(validLen)` and ignored precision:
  - result: `tests/pytest/prof_out/packed64_phaseA130_tailpad_disabled_20260515_8shapes/performance_compare.csv`
  - ratios: `0.842 / 0.821 / 0.794 / 0.794 / 0.846 / 0.850 / 0.843 / 0.856`
- The diagnostic shows the scalar tail padding loop is a major cost on `S=65/513/2049`, but the disabled path is not a valid baseline because tail correctness was not checked.

## Target

Keep Tail-T2 semantics, but remove the extra AIV scalar work from tail padding.

The intended data contract is:

1. Q/K/beta/g/value tail rows are padded before or while entering Stage1 local/workspace buffers.
2. `attnUbFloat_` should already have invalid rows/cols zero before writing `L_powHalfWsGm_` and building `INVInitHalfWsGm_`.
3. `BuildInitialInverse()` keeps only the diagonal `SetValue(1)` for `I - L`; it must not become a general padding loop.

## Scope

Allowed:

- Stage1 inverse preparation only.
- `ApplyTailPaddingToAttn(validLen)` and its call site.
- Local verification around `attnUbFloat_ -> L_powHalfWsGm_ / INVInitHalfWsGm_`.

Forbidden:

- Stage2, Stage3, QK.
- `gBK/vBeta` generation.
- final RHS math.
- AIC inverse/compose formula changes.
- New GM fallback or old exact tail RHS fallback.

## Execution Plan

### A131a: Contract cleanup

First inspect and keep the original Stage1-style padding contract explicit in the code path:

- Q/K invalid rows must be zero after `QKPreProcess()`.
- beta invalid rows must be zero after `BetaCopyInWithStride()`.
- g invalid positions must be zero after `GCopyInWithStride()`.
- value invalid rows must stay zero before `v_beta` generation.

If this contract is already sufficient, delete the `ApplyTailPaddingToAttn(validLen)` call entirely and leave `BuildInitialInverse()` unchanged.

### A131b: Minimal vector padding fallback

If explicit `attnUbFloat_` padding is still required, replace the scalar nested loop with `Duplicate`:

- if `validLen <= subOffset_`, zero the whole `halfChunkSize_ x chunkSize_` block;
- if only some rows are invalid, zero invalid rows as whole rows;
- if valid rows have invalid columns, zero the row tail with one `Duplicate` per affected row;
- no per-element `SetValue` except diagonal identity writes.

Do not run A131b if A131a passes correctness without explicit padding.

Status: executed directly by user decision. The scalar tail padding loop was removed without first running the contract-deletion diagnostic because the requested path was explicit `Duplicate` padding.

Implementation:

- `validLen <= subOffset_`: `Duplicate` the whole `halfChunkSize_ x chunkSize_` `attnUbFloat_` block to zero.
- partial valid rows: use row-start masked `Duplicate` for invalid columns, keeping the destination address 32B aligned.
- invalid rows: use one contiguous `Duplicate` from `validRows * chunkSize_`.
- no per-element tail `SetValue`; `BuildInitialInverse()` keeps only the diagonal identity writes.

## Gate

Run only one candidate at a time.

1. Build package.
2. Install package once for the candidate.
3. Tail smoke: `S=5/16/31/32/33/34/48/49/63/65/129`.
4. Aligned smoke: `S=64/256/1024`.
5. Full compare.
6. If correctness passes, run 8-shape profiling:
   - `S=64/65/512/513/2048/2049/10240/10241`
7. Compare against:
   - A130 baseline profiling.
   - Tail-T2 profiling.
   - diagnostic tailpad-disabled profiling.

## Pass Criteria

- Precision must pass the full tail/aligned gate.
- `aiv_scalar_time_us` should approach the diagnostic run on non-64 shapes.
- Task ratios should improve over A130/Tail-T2 on `S=65/513/2049`, or at minimum not regress aligned shapes enough to erase the win.
- Record the source hash and gate evidence only after full correctness plus profiling pass.

## Failure Handling

- If A131a fails precision, restore the call and continue with A131b vectorized padding.
- If A131b fails precision, isolate whether the failure is invalid rows, invalid columns, or diagonal identity placement.
- Do not proceed to another inverse/resident optimization while tail padding correctness is failing.

## Result

Accepted as A131 active continuation.

 - Source SHA256: `966465538bf97e8a6f3966b1f59d341195d948f177197bc9a57b89fbad8f4c0f`.
 - Official Stage1 SHA256 after cleanup: `8d4932568da50f3118e3e496dd2508c79cd2d10951d7af53fed6165e12440984`.
- After cleanup, A131 lives in `op_kernel/chunk_gated_delta_rule_neumann_stage1.h`; `op_kernel/baseline/` is removed.
- Pre-cleanup A131 source SHA256: `966465538bf97e8a6f3966b1f59d341195d948f177197bc9a57b89fbad8f4c0f`.
 - Cleanup rebuild package SHA256: `f36da5e37df82acf4e1da9c0c05172265e0933bd9c7ca70c462fd1f69afded0a`.
- Cleanup rebuild package SHA256: `f36da5e37df82acf4e1da9c0c05172265e0933bd9c7ca70c462fd1f69afded0a`.
- Installed Neumann object SHA256: `45cb88e35547cd0d5181c21b26e85625160abe45ef027e80b5a9728ba17c70f6`.
- Installed Neumann JSON SHA256: `4436e86e9abe9ef5f5c328323e04e676bcf8bbfbf80e2b4da757c162104ad4a0`.

Correctness:

- Tail smoke PASS: `S=5/16/31/32/33/34/48/49/63/65/129`.
- Aligned smoke PASS: `S=64/256/1024`.
- Full compare PASS: `30 passed in 6.79s`.

Profiling:

- Result: `tests/pytest/prof_out/packed64_phaseA131_tail_padding_duplicate_20260515_8shapes/performance_compare.csv`.
- Compare CSV SHA256: `7d18ff50b66c660ffa66cb2764ddee12956b9a9c9a37141f9646d82a0957d33f`.
- Raw CSV SHA256: `13a73b00e08578679b7d04ccc20b38474bce177eb94d66a35ca8a116342d758f`.
- Neumann/normal ratios for `S=64/65/512/513/2048/2049/10240/10241`: `0.811 / 0.812 / 0.793 / 0.793 / 0.836 / 0.832 / 0.840 / 0.845`.
- Absolute Neumann task vs A130: `1.022 / 0.793 / 0.974 / 0.713 / 1.003 / 0.747 / 1.005 / 1.000`.
- Absolute Neumann task vs Tail-T2: `0.891 / 0.768 / 0.963 / 0.691 / 0.986 / 0.748 / 0.997 / 0.996`.
- Absolute Neumann task vs tailpad-disabled diagnostic: `0.982 / 1.005 / 1.004 / 0.998 / 0.986 / 0.992 / 0.999 / 0.993`.

Conclusion: A131 preserves Tail-T2 correctness and brings non-64 tail performance close to the diagnostic-disabled run, so it is the valid correctness-safe active continuation now merged into the official Stage1 file.
