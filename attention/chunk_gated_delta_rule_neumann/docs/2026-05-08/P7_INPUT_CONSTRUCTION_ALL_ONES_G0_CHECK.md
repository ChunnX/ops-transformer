# P7 Input Construction Check: Normalized All-Ones, g=0

## Context

- Date: 2026-05-08
- Active selector: `fulllocalnew/C64_1616_FP16`, `chunkSize=64`
- Active Stage1 status before this check:
  - restored to clean P8A baseline shape
  - no GM dump tensor path
  - no `debugDump/P7_DEBUG/CGDR_DEBUG` residue in active wrapper/Stage1/tiling files
- Purpose: use controlled inputs to localize behavior without saving intermediate tensors to GM, because GM dump/roundtrip can be overwritten or observe an inaccurate layout.

This check does not prove P7 packed64 correctness. It only establishes whether a simple deterministic input can pass the current clean P8A Stage1 path without relying on GM dump diagnostics.

## Case: q/k Normalized All-Ones

Input:

- `query = 1 / sqrt(dk) = 0.125`
- `key = 1 / sqrt(dk) = 0.125`
- `value = 1`
- `beta = 1`
- `initial_state = 1`
- `g = 0`
- `B=1, S=64, nk=4, nv=4, dk=64, dv=64`
- `scale = 1 / sqrt(64) = 0.125`

This preserves the normalization assumption used by the normal random compare script while keeping the values deterministic.

Result:

```text
case=all_ones_g0_qk_normalized
CPU vs ChunkGDR out: PASS max_abs=0.000000e+00
CPU vs ChunkGDR state: PASS max_abs=0.000000e+00
CPU vs ChunkGDRNeumann out: PASS max_abs=0.000000e+00
CPU vs ChunkGDRNeumann state: PASS max_abs=0.000000e+00
ChunkGDR vs ChunkGDRNeumann out: PASS max_abs=0.000000e+00
ChunkGDR vs ChunkGDRNeumann state: PASS max_abs=0.000000e+00
PASSED
```

Interpretation:

- The clean P8A Stage1 path passes this deterministic normalized all-ones, `g=0` case exactly.
- This confirms the current restored P8A path can handle a simple non-random 64-aligned input without GM dump diagnostics.
- Because this input is highly symmetric, it is only a sanity check. It does not distinguish P7 packed64 layout/math issues.

## Random Seed Confirmation

After the normalized all-ones case passed, run one standard random smoke case on the same 64-aligned shape.

Command:

```bash
timeout 90s env CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer \
  python3 tests/pytest/compare_chunk_gdr_three_way.py \
  --seed 0 --batch-size 1 --seqlen 64 --nk 4 --nv 4 --dk 64 --dv 64
```

Result:

```text
case=cli: B=1, seqlens=[64], nk=4, nv=4, dk=64, dv=64, golden_float64=False, seed=0, scale=0.125000
CPU vs ChunkGDR out: PASS max_abs=3.173694e-03
CPU vs ChunkGDR state: PASS max_abs=7.139370e-04
CPU vs ChunkGDRNeumann out: PASS max_abs=3.173694e-03
CPU vs ChunkGDRNeumann state: PASS max_abs=7.139370e-04
ChunkGDR vs ChunkGDRNeumann out: PASS max_abs=4.882812e-04
ChunkGDR vs ChunkGDRNeumann state: PASS max_abs=4.882812e-04
PASSED
```

Interpretation:

- The restored clean P8A path passes both:
  - normalized all-ones, `g=0`
  - standard random seed `0`, `S=64`
- This confirms the current rollback point is precision-clean for the active S=64 smoke. It is still not a P7 packed64 target pass.

## Next Debug Direction

Do not use GM tensor dumping as the primary precision diagnostic for P7.

Prefer controlled input construction:

1. Keep q/k normalized. Do not use strict unnormalized all-ones for this定位 line.
2. Vary one factor at a time:
   - `beta = 1`, then random `beta`
   - `g = 0`, then random negative `g`
   - q/k identical, then q/k with simple row pattern, then random normalized q/k
3. Compare normal op vs Neumann op at output/state level for each constructed case.
4. Only after a constructed case fails, use code inspection or resident-side reasoning to decide whether the difference points to P7 diagonal inverse, P8 compose, or later Stage1 consumers.
