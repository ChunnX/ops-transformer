# Bug Analysis: inplace_partial_rotary_mul_grad

> Last updated after "fix bug 7" commit series (cosRowStride fix, dAlignDy removal, alignment validation)

---

## Bug #1 — FIXED: Mixed-precision `cosRowStride` wrong in all VF functions

### Root Cause

`cosRowStride` was derived from dy byte stride: `dAlignDy * sizeof(DTYPE_DY) / sizeof(DTYPE_COS)`. When `sizeof(DTYPE_DY) ≠ sizeof(DTYPE_COS)`, this produces a wrong cos row stride (e.g., fp16 dy + fp32 cos gives `sliceLength/2` instead of `sliceLength`).

### Fix: Use `dLen` directly for all row strides and within-row offsets

All VF functions now use raw logical dimension values instead of aligned ones:

| Old | New |
|-----|-----|
| `dAlignDy` | `dLen` (row stride) |
| `halfDAlignDy` / `halfDAlignCos` | `halfD` (= `dLen/2`) |
| `quarterDAlignDy` / `quarterDAlignCos` | `quarterD` (= `dLen/4`) |
| `cosRowStride = dAlignDy * sizeof(DY) / sizeof(COS)` | `cosRowStride = dLen` |
| `dHalfOffsetDy` / `dHalfOffsetCos` | `dHalfSize` (= `dLen/2`) |

This matches the DataCopyPad layout which uses `dstStride=0` (tightly packed, no padding between rows).

### Alignment Validation

Since the VF functions now use raw logical offsets, all addresses must satisfy the AscendC hardware requirement:
`(offset * sizeof(TYPE)) % 32 == 0` for both `DTYPE_DY` and `DTYPE_COS`.

A tiling-time validation was added in `CheckRotaryModeShapeRelation()` (`tiling_base_arch35.cpp`):

```cpp
int64_t typeSize = ge::GetSizeByDataType(dtype_);
int64_t cosTypeSize = ge::GetSizeByDataType(cosDtype_);
int64_t minTypeSize = typeSize < cosTypeSize ? typeSize : cosTypeSize;
int64_t alignUnit = 32 / minTypeSize;   // fp16→16, fp32→8, bf16→16
if (sliceLen % alignUnit != 0) { ... return GRAPH_FAILED; }
```

This uses the **stricter** alignment (based on the smaller type = larger element count per 32 bytes) to guarantee alignment for both dy and cos loads.

| dy type | cos type | alignUnit | Example non-aligned sliceLength |
|---------|----------|-----------|-------------------------------|
| fp16 (2B) | fp16 (2B) | 16 | 130 (130 % 16 = 2) |
| fp32 (4B) | fp32 (4B) | 8 | 130 (130 % 8 = 2) |
| fp16 (2B) | fp32 (4B) | 16 | 130 (130 % 16 = 2) |
| fp32 (4B) | fp16 (2B) | 16 | 130 (130 % 16 = 2) |
| bf16 (2B) | fp32 (4B) | 16 | 130 (130 % 16 = 2) |

All common model dimensions (64, 128, 256, 512, ...) are multiples of both 8 and 16, so they pass naturally.

### Why Not Use Aligned Offsets + Aligned DataCopy?

The reference operator uses aligned offsets but also allocates UB with alignment padding. Our operator uses `dstStride=0` (packed) in DataCopyPad. If we restored aligned VF offsets without also padding the DataCopy, the VF would read padding/garbage instead of data. Changing DataCopyPad to use aligned stride would require modifying all 4 kernel templates' data搬运 logic — high risk for marginal benefit.

The current approach (packed layout + logical offsets + alignment validation) is the simplest correct solution.

---

## Bug #2 — FIXED: x input blocked, dcos/dsin documented as unimplemented

### Summary

The kernel entry (`inplace_partial_rotary_mul_grad_apt.cpp`) discards dcos/dsin with `(void)dcos; (void)dsin;`. The dcos/dsin gradient computation is not yet implemented.

### Fix

1. **Tiling validation** (`CheckOptionalInput()`): When x is not nullptr, returns `GRAPH_FAILED` with a clear error message:
   ```
   "Input x is not supported yet: dcos/dsin gradient computation is not implemented.
    Please set x to nullptr."
   ```
2. **Proto documentation** (`inplace_partial_rotary_mul_grad_proto.h`): Updated x input description to state "Currently only nullptr is supported", and added NOTE on dcos/dsin outputs that gradient computation is not yet implemented.

### Remaining Work

When dcos/dsin is implemented:
1. Remove the early return in `CheckOptionalInput()`
2. Restore the shape validation code below it
3. Implement the gradient computation in the kernel (reference: `rotary_position_embedding_grad` with `IF_NEED_BACKWARD=true`)

---

## Bug #3 — OPEN (Low Priority): GetWorkspaceSize

`GetWorkspaceSize` returns `GRAPH_SUCCESS` without setting workspace size. Harmless for AIV-only kernels (no workspace needed), but should be verified against framework expectations.

---

## Fix History

| Commit | Description |
|--------|-------------|
| 434bb9ec4 | "fix bug 2": Fixed InterleaveMode mask mismatch + Quarter alignment |
| 81d01f6fc | "fix bug": Initial bug fixes to common.h, tiling, docs |
| "fix bug 4" | Changed `cosRowStride = dLen` in all VF functions |
| "fix bug 5-7" | Removed `dAlignDy` parameter; replaced all aligned offsets with logical values; cleaned up callers |
| (current) | Added 32-byte alignment validation in tiling; blocked x input; updated proto docs |
