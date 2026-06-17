# Bug Analysis: inplace_partial_rotary_mul_grad

> Updated after "fix bug 4" commit (cosRowStride changed to dLen, x input blocked)

## Bug #1 — FIXED: `cosRowStride` now uses `dLen`

### Fix Summary

Changed `cosRowStride` from `dAlignDy * sizeof(DTYPE_DY) / sizeof(DTYPE_COS)` to `dLen` (= `sliceLength`) in all VF functions. This correctly matches the packed DataCopyPad layout (dstStride=0), where the actual stride between S-rows of cos/sin data is `sliceLength` DTYPE_COS elements.

### Verification

For mixed precision (fp16 dy, fp32 cos, sliceLength=128):
- Old: `cosRowStride = 128 * 2 / 4 = 64` fp32 elements → wrong
- New: `cosRowStride = 128` fp32 elements → matches packed stride ✓

For same precision (sliceLength=128):
- Old: `cosRowStride = 128` → correct (happened to work)
- New: `cosRowStride = 128` → correct ✓

### Remaining Risk

When `sliceLength` is not aligned to `DEFAULT_BLOCK_SIZE / maxTypeSize`, `dAlignDy > sliceLength`, causing dy row stride mismatch with packed layout (pre-existing, inherited from reference operator).

---

## Bug #2 — PARTIALLY FIXED: x input blocked, dcos/dsin documented as unimplemented

### Fix Summary

- `CheckOptionalInput()` now always returns `GRAPH_FAILED` with error message, blocking x input.
- Proto doc updated: "dcos/dsin gradient computation is not yet implemented; x input must be set to nullptr."

### Remaining Issue

Dead code: lines after `return ge::GRAPH_FAILED` (L271+) in `CheckOptionalInput()` are unreachable. Should be removed.

---

## Remaining Potential Issues

### Issue A: dy row stride mismatch with packed layout (pre-existing)

VF uses `dAlignDy` (aligned, may > `sliceLength`) for dy row stride, but DataCopyPad writes packed (dstStride=0, actual stride = `sliceLength`). When `dAlignDy > sliceLength`, S-row>0 dy reads from wrong offsets.

`CheckRotaryModeShapeRelation` only validates `sliceLength % dSplitCoef == 0`, not 32-byte alignment.

Examples:
- fp16 dy, half mode, sliceLength=130: dAlignDy=132, packed stride=130 → off by 2
- fp32 dy, half mode, sliceLength=130: dAlignDy=144, packed stride=130 → off by 14

The reference operator `rotary_position_embedding_grad` has the same pattern (dstStride=0 + dAlign stride), so this is inherited behavior, not newly introduced.

**Recommendation**: Add tiling validation `sliceLength % (32/maxTypeSize) == 0`, or change both DataCopyPad dstStride and VF row strides to use aligned values.

### Issue B: cosRowStride and dAlignDy now independently computed

After the fix, `cosRowStride = dLen` and `dAlignDy = dAlign_`. When `dAlign_ != sliceLength`, same S-row's dy and cos are accessed at different offsets. Cos is correct (matches packed stride), dy may be off. Not a new bug, but conceptually inconsistent.

### Issue C: AB template sliceAlignX_ vs dAlign_ mismatch

AB buffer allocation uses `sliceAlignX_ = Ceil(sliceLength*maxTypeSize, 32)/sizeof(DY)`, while tiling passes `dAlign_` to VF. These differ for mixed-precision small dimensions. Buffer is oversized (no overflow), but wastes UB space.

---

## Original Bug #1 Report (before fix)

### Summary

### Summary

Every VF function in `inplace_partial_rotary_mul_grad_common.h` computes `cosRowStride` using the **dy** byte stride to derive the **cos** row stride in elements. This is only correct when `sizeof(DTYPE_DY) == sizeof(DTYPE_COS)` (same precision). For mixed precision, the formula produces wrong offsets for multi-S-row cos/sin access in UB.

### Current Implementation

```cpp
uint32_t rowStrideBytes = dAlignDy * sizeof(DTYPE_DY);
uint32_t cosRowStride = rowStrideBytes / sizeof(DTYPE_COS);
```

Used as:
```cpp
currSinUb = sinUb + sIdx * cosRowStride;
currCosUb = cosUb + sIdx * cosRowStride;
```

### Why It Is Wrong

The `cosRowStride` formula converts a dy-aligned byte stride to cos element count:

```
cosRowStride = dAlignDy * sizeof(DTYPE_DY) / sizeof(DTYPE_COS)
```

This gives the number of cos elements that span the **same byte range** as `dAlignDy` dy elements. But the actual UB layout for cos/sin data has rows with `dAlign_` cos elements each (not `dAlignDy * sizeof(DTYPE_DY) / sizeof(DTYPE_COS)`).

**Proof by comparison with reference operator `rotary_position_embedding_grad`:**

The reference operator (same type T for dy/cos) uses:
```cpp
currSinUb = sinUb + sIdx * dAlign;   // dAlign = CeilAlign(dLen, BLOCK_SIZE/sizeof(T))
currCosUb = cosUb + sIdx * dAlign;
```

Here `dAlign` is the aligned dimension in **T elements** (the same type used for both dy and cos). This is the row stride in UB for both dy and cos rows.

The inplace operator generalizes this to mixed precision by computing:
```
cosRowStride = dAlignDy * sizeof(DTYPE_DY) / sizeof(DTYPE_COS)
```

But the correct generalization should use the **cos-aligned** dimension as the cos row stride, not a byte-conversion from the dy-aligned dimension.

### Concrete Example

**fp16 dy, fp32 cos, half mode, sliceLength=128, dSplitCoef=2:**

- `dAlign_ = Ceil(128/2, 32/4) * 2 = Ceil(64, 8) * 2 = 128`
- `cosRowStride = 128 * sizeof(fp16) / sizeof(fp32) = 128 * 2 / 4 = 64` (fp32 elements)
- Actual UB cos row stride = `dAlign_ = 128` (fp32 elements)

VF would access:
- S-row 0: `cosUb[0..127]` (first 128 fp32 elements) — correct data range
- S-row 1: `cosUb[64..191]` (offset 64 fp32 elements) — **WRONG**, should be `cosUb[128..255]`

The VF reads S-row 1 cos data from the middle of S-row 0, producing incorrect dx results.

**fp32 dy, fp16 cos, half mode, sliceLength=128, dSplitCoef=2:**

- `dAlign_ = Ceil(128/2, 32/4) * 2 = 128`
- `cosRowStride = 128 * sizeof(fp32) / sizeof(fp16) = 128 * 4 / 2 = 256` (fp16 elements)
- Actual UB cos row stride = `dAlign_ = 128` (fp16 elements)

VF would access:
- S-row 0: `cosUb[0..127]` — correct
- S-row 1: `cosUb[256..383]` — **WRONG**, should be `cosUb[128..255]`. Skips over 128 elements of valid S-row 1 data and reads into uninitialized/padding area.

### Same-precision Impact

When `sizeof(DTYPE_DY) == sizeof(DTYPE_COS)`, `cosRowStride = dAlignDy`. If `dAlignDy == dAlign_ == sliceLength` (sliceLength already aligned), this equals the actual UB row stride. Bug does not manifest in this case.

When `sliceLength` is not aligned and `dAlign_ > sliceLength`, the actual UB data stride may differ from `dAlign_` (depending on whether `DataCopyPad` writes with `dstStride=0` packing or with row-aligned padding). Currently all DataCopyPad calls use `dstStride=0` (packed layout), making the actual stride = `sliceLength` rather than `dAlign_`. This means even same-precision could be affected when `dAlign_ > sliceLength`, though typical usage has aligned `sliceLength` values.

### Affected Functions

All VF functions that access multi-S-row cos/sin data via `cosRowStride`:

| Function | File | Line |
|----------|------|------|
| `HalfGradAlignVF` | `inplace_partial_rotary_mul_grad_common.h` | L53 |
| `InterleaveModeGradVF` | `inplace_partial_rotary_mul_grad_common.h` | L113-114 |
| `QuarterGradAlignVF` | `inplace_partial_rotary_mul_grad_common.h` | L192-193 |
| `InterleaveHalfModeGradVF` | `inplace_partial_rotary_mul_grad_common.h` | L284-285 |
| `BatchHalfGradAlignVF` | `inplace_partial_rotary_mul_grad_common.h` | L351 |
| `BatchQuarterGradAlignVF` | `inplace_partial_rotary_mul_grad_common.h` | L425 |
| `BatchInterleaveModeGradVF` | `inplace_partial_rotary_mul_grad_common.h` | L515 |
| `BatchInterleaveHalfModeGradVF` | `inplace_partial_rotary_mul_grad_common.h` | L593 |

### Root Cause

The `cosRowStride` formula assumes the UB cos row stride in bytes equals the dy row stride (`dAlignDy * sizeof(DTYPE_DY)`). This is a false assumption: the buffer is allocated with `dAlign_ * sizeof(DTYPE_COS)` bytes per cos row, so the cos row stride in DTYPE_COS elements is `dAlign_`, not `dAlignDy * sizeof(DTYPE_DY) / sizeof(DTYPE_COS)`.

The formula happens to produce `dAlign_` only when `sizeof(DTYPE_DY) == sizeof(DTYPE_COS)` (same precision), because then `dAlignDy * sizeof(DTYPE_DY) / sizeof(DTYPE_COS) = dAlignDy * 1 = dAlign_`.

### Fix

Replace `cosRowStride` calculation in all affected VF functions:

**HalfGradAlignVF:**
```cpp
// Before:
uint32_t rowStrideBytes = dAlignDy * sizeof(DTYPE_DY);
uint32_t cosRowStride = rowStrideBytes / sizeof(DTYPE_COS);

// After:
uint32_t cosRowStride = AscendC::Ceil(dLen, static_cast<uint32_t>(DEFAULT_BLOCK_SIZE / sizeof(DTYPE_COS)));
```

Note: `dAlign_` is already computed in the kernel templates (AAndB, ABAAndBA) using `maxTypeSize` alignment:
```cpp
dAlign_ = Ceil(sliceLength / dSplitCoef, DEFAULT_BLOCK_SIZE / maxTypeSize) * dSplitCoef;
```

This `dAlign_` is a universal row width that works for both dy and cos. For cos, the row stride in DTYPE_COS elements is simply `dAlign_`. The fix should pass `dAlign_` (or a cos-specific `dAlignCos`) to the VF as `cosRowStride`, instead of computing it from dy parameters.

Alternative: Compute `dAlignCos` in the VF:
```cpp
uint32_t dAlignCos = AscendC::Ceil(dLen, static_cast<uint32_t>(DEFAULT_BLOCK_SIZE / sizeof(DTYPE_COS)));
uint32_t cosRowStride = dAlignCos;
```

This is equivalent to the reference operator's `dAlign = CeilAlign(dLen, BLOCK_SIZE/sizeof(T))`, generalized for `DTYPE_COS`.

For the **Batch VF functions**, the `nStepCosUb` calculation also depends on `cosRowStride`:
```cpp
// Before:
uint32_t cosRowStride = dAlignDy * sizeof(DTYPE_DY) / sizeof(DTYPE_COS);
int32_t nStepCosUb = ubFactorS * cosRowStride;

// After:
uint32_t dAlignCos = AscendC::Ceil(static_cast<uint32_t>(dLen), static_cast<uint32_t>(DEFAULT_BLOCK_SIZE / sizeof(DTYPE_COS)));
uint32_t cosRowStride = dAlignCos;
int32_t nStepCosUb = ubFactorS * cosRowStride;
```

For `InterleaveModeGradVF` and `BatchInterleaveModeGradVF`, `dLen` is the full dimension (no half/quarter split), so:
```cpp
uint32_t dAlignCos = AscendC::Ceil(dLen, static_cast<uint32_t>(DEFAULT_BLOCK_SIZE / sizeof(DTYPE_COS)));
uint32_t cosRowStride = dAlignCos;
```

---

## Bug #2 — SIGNIFICANT: dcos/dsin outputs never computed

### Summary

The kernel entry (`inplace_partial_rotary_mul_grad_apt.cpp`) discards dcos/dsin with `(void)dcos; (void)dsin;`. These are declared as required outputs in the proto and their shapes are correctly set by infershape. When `x` input is provided (dCosFlag=1), the math requires computing gradients w.r.t. cos and sin, but these outputs remain uninitialized.

### Mathematical Requirement

Forward: `y = x * cos + rotate_half(x) * sin`

Backward w.r.t. cos and sin (when x is provided):
- `dcos = -dy * x_sin` (accumulate over B/N/S dimensions where cos broadcasts)
- `dsin = dy * x_cos` (accumulate over B/N/S dimensions where sin broadcasts)

### Location

- Proto: `op_graph/inplace_partial_rotary_mul_grad_proto.h` — declares dcos/dsin as required outputs
- Infershape: `op_host/inplace_partial_rotary_mul_grad_infershape.cpp` — correctly sets dcos/dsin shapes
- Kernel entry: `op_kernel/inplace_partial_rotary_mul_grad_apt.cpp` — `(void)dcos; (void)dsin;`

### Reference Comparison

`rotary_position_embedding_grad` (`rotate_half_grad.h`) correctly computes cosGrad/sinGrad when `IF_NEED_BACKWARD=true`, accumulating results across the reduction dimensions.

---

## Bug #3 — MINOR: GetWorkspaceSize returns success without setting workspace size

### Summary

`GetWorkspaceSize` in `inplace_partial_rotary_mul_grad_tiling_base_arch35.cpp` returns `GRAPH_SUCCESS` without actually setting the workspace size via the context API. Likely harmless for AIV-only kernels (no workspace needed), but should be verified against framework expectations.
