# P8 First-Layer cInv/cTmp Accumulate Plan

## Context

- Active Stage1:
  - `op_kernel/p8afullocal/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a.h`
- Starting active source SHA256:
  - `1d23b34e174e24287b4046054bd6bce35fd68da23f18c031e45a55028b5bc328`
- Latest validated baseline before this cleanup surface:
  - A67 `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA67_identity16_cross_para_keep_baseline.h`
  - SHA256 `146d13040989a98f23436a083769b1a925fc8f91e9e0c25691f993c479acfbfe`

The new candidate is not attributed to A67 until it independently passes build, install, `S=64/256/1024`, full compare, and performance if requested.

## Formula

AIV already generates:

```text
INVInit = I - L
lowerNeg = -L
```

Do not add another negative sign. P8 computes:

```text
tmp = lowerNeg @ topInv
out = bottomInv @ tmp
```

This equals the lower-triangular inverse update:

```text
out = -bottomInv @ L @ topInv
```

## P7 Output State

After `P7Packed64DiagonalNeumannOneShotToResult()`, keep:

```text
cInv = aicL0C_
cTmp = aicL0C_[C64_L0C_TILE_ELEMS]
```

`cInv` holds the diagonal inverse:

```text
D0  0   0   0
0   D1  0   0
0   0   D2  0
0   0   0   D3
```

P8 first-layer must not use `aicL0C_` with `RAW_MMAD_INIT_C`, because that overwrites `cInv`.

## Phase 1: First-Layer Only

Only replace `P8ComposeFirstLayerPacked32Direct()`. Keep existing second-layer unchanged.

### Step 1: tmp in cTmp

Load full64 sparse operands, with L0A/L0B cleared before loading valid lanes:

```text
L0A lowerNeg:
0    0    0    0
L10  0    0    0
0    0    0    0
0    0    L32  0

L0B topInv:
D0   0    0    0
0    0    0    0
0    0    D2   0
0    0    0    0
```

Then:

```text
cTmp = lowerNeg @ topInv
```

Expected nonzero lanes:

```text
T10 = L10neg @ D0
T32 = L32neg @ D2
```

Full64 Fixpipe `cTmp` to `aicHalfL1TileInv_`. This avoids partial `CFG_NZ` L0C copy-out, which has failed before in A42/A45.

### Step 2: accumulate into cInv

Load full64 sparse operands:

```text
L0A bottomInv:
0   0   0   0
0   D1  0   0
0   0   0   0
0   0   0   D3

L0B tmp:
0    0    0    0
T10  0    0    0
0    0    0    0
0    0    T32  0
```

Then:

```text
cInv += bottomInv @ tmp
```

Expected updated lanes:

```text
cInv(16,0)  = D1 @ L10neg @ D0
cInv(48,32) = D3 @ L32neg @ D2
```

Finally full64 Fixpipe `cInv` to `aicHalfL1P3Result_`, so the existing second-layer can still load from L1.

## Phase 2: Second-Layer

After Phase 1 passes, update `P8Compose64From32Direct32()` separately:

1. use `cTmp` for `tmp64 = lowerNeg32 @ B0`;
2. accumulate into the same `cInv` with `cInv += B1 @ tmp64`;
3. full64 Fixpipe once to `aicHalfL1P3Result_`.

Do not combine this with Phase 1.

### A69 Implementation Result

Phase 2 has been implemented as A69:

- Active Stage1 SHA256:
  - `c135346911ae7915a4cb6ba272d997289bd19e54f774f4f21fac987d4441a4ea`
- Added full64 second-layer sparse loaders and MMAD helpers:
  - `LoadP8SecondLayerLowerToFull64L0A()`
  - `LoadP8SecondLayerTopInvToFull64L0B()`
  - `LoadP8SecondLayerBottomInvToFull64L0A()`
  - `LoadP8SecondLayerTmpToFull64L0B()`
  - `RawMmadP8SecondLayerLowerTopFull64()`
  - `RawMmadP8SecondLayerBottomTmpFull64()`
  - `P8Compose64From32AccumFull64()`
- Strict full64 branch now calls `P8Compose64From32AccumFull64(paraIdx)`.
- Tail/non-full64 fallback still keeps `P8Compose64From32Direct32(paraIdx)`.
- Formula remains unchanged:
  - `tmp64 = lowerNeg32 @ B0`
  - `cInv += B1 @ tmp64`
  - no extra negative sign.

Correctness gate:

- build PASS.
- isolated install PASS:
  - package `/tmp/cgdr_a69_20260512_220105.run`
  - package SHA256 `f85b7864e3c757df74d01f2eae656dd88068302c04dc59180989fafdeb28acee`
  - isolated OPP `/tmp/cgdr_a69_opp_20260512_220105/vendors/custom_transformer`
  - installed Neumann object SHA256 `2db6ce2a99cc7a55cbc8ed39fd94a6be09dccbbc3648554c61bf84a7ef85b280`
- installed active Stage1 header SHA256:
  - `c135346911ae7915a4cb6ba272d997289bd19e54f774f4f21fac987d4441a4ea`
- `S=64` PASS.
- `S=256` PASS.
- `S=1024` PASS.
- full compare PASS:
  - `16 passed in 6.22s`

Baseline saved:

- `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA69_p8_second_layer_cinv_ctmp_accum_baseline.h`
- SHA256:
  - `c135346911ae7915a4cb6ba272d997289bd19e54f774f4f21fac987d4441a4ea`

Performance:

- Output directory:
  - `tests/pytest/prof_out/packed64_phaseA69_p8_second_layer_cinv_ctmp_accum_20260512/`
- Compare CSV SHA256:
  - `484718014cde93891a68d633eb1f275afcd1d1a5323f346367437b4e7738ec56`
- Raw CSV SHA256:
  - `787211d7a3245f4ff9d27c5c6cd5f1fbc308d43cbaaeca524e219a6a54de836b`
- neumann/normal ratios:
  - `S=64`: `1.250`
  - `S=512`: `1.657`
  - `S=2048`: `2.438`
  - `S=10240`: `2.469`

Conclusion:

- A69 is the latest correctness-safe baseline for the full first-layer and second-layer `cInv/cTmp` accumulate design.
- A69 is not the preferred performance baseline:
  - compared with A68 `1.175 / 1.662 / 2.399 / 2.437`, A69 improves only `S=512` slightly and regresses `S=64/2048/10240`;
  - compared with A67 `1.181 / 1.635 / 2.425 / 2.456`, A69 regresses all four profiled sizes.

## Guards

- Do not change final RHS, Stage2, Stage3, QK, `gBK/vBeta`, BF16 GM materialization, or resident RHS exposure.
- Do not add a negative sign.
- Do not use partial L0C copy-out for this phase.
- Do not replace this with four independent `16x16` fallback solves.
- Keep tail/non-full64 behavior unchanged.

## Gate

Required before saving a baseline:

1. Build package.
2. Install package.
3. Run `S=64 -> S=256 -> S=1024`.
4. Run full compare.
5. Run performance only after correctness passes.
