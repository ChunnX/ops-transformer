# FIAS Boolean Template Type-Erasure Plan

## Goal
Reduce template instantiations from **21,417 to ~4,965** (4.3x reduction)
by converting boolean template parameters to runtime values read from TilingData.

## Why This Is Safe
Analysis of common/op_kernel/arch35/ confirms that the boolean template
parameters (`hasAtten`, `isPa`, `isFd`, `HasAttenMask`, `HasRope`,
`EnableKVPrefix`, `EnableS1OutSplit`) are:

1. **Only used in `if constexpr` blocks** for conditional initialization
2. **Never used in type aliases** (`using type = ...`)
3. **Never used for buffer allocation sizing** at compile time
4. **Already available** in CVSharedParams/TilingData at runtime

Converting `if constexpr (hasAtten) { ... }` to `if (sharedParams.hasAtten) { ... }`
is semantically equivalent. The branch predictor handles this with negligible
overhead (~5 cycles) compared to the matmul compute (~10K+ cycles per tile).

## Impact Per Parameter

| Parameter | Blocks with both T/F | Multiplier | Priority |
|-----------|---------------------|------------|----------|
| HasAttenMask | 158/191 | 2x | P0 |
| IsPa | 110/191 | 2x | P0 |
| IsFd | 86/191 | 2x | P1 |
| HasRope | 2/191 | ~1x | P2 (low impact) |
| EmptyTensor | 0/191 | 1x | P0 (dead, just remove) |
| EnableKVPrefix | 0/191 | 1x | skip |
| EnableS1OutSplit | 0/191 | 1x | skip |

## Files to Modify (in order)

### Phase 1: Kernel trait infrastructure (common/)

**File: `attention/common/op_kernel/arch35/infer_flash_attention_comm.h`**
- Lines 27-31: Remove `hasAtten`, `isPa`, `isFd` from `CHILD_SPEC_TEMPLATE` macro
- Lines 87-102: Remove from `CUBE_BLOCK_TRAITS_CONST_FIELDS` macro
- Add runtime accessors to read from sharedParams instead

Before:
```cpp
#define CHILD_SPEC_TEMPLATE \
    template <typename INPUT_T, typename T, ImplModeEnum implMode, LayOutTypeEnum layout, \
    S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType, \
    DTemplateType dVTemplateType, PseTypeEnum pseMode, bool hasAtten, bool hasDrop, bool hasRope, \
    typename OUTPUT_T, bool isInfer, bool isPa, bool isFd>
```

After:
```cpp
#define CHILD_SPEC_TEMPLATE \
    template <typename INPUT_T, typename T, ImplModeEnum implMode, LayOutTypeEnum layout, \
    S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType, \
    DTemplateType dVTemplateType, PseTypeEnum pseMode, bool hasDrop, \
    typename OUTPUT_T, bool isInfer>
```

### Phase 2: Kernel classes (common/op_kernel/arch35/)

**Files to modify** (change `if constexpr (hasAtten)` to `if (this->runtimeFlags.hasAtten)`):
- `flash_attention_noquant_kernel_base.h` (lines ~238)
- `flash_attention_noquant_kernel_infer.h` (lines ~53, ~73)
- `flash_attention_score_kernel_infer.h`
- `flash_attention_kernel_noquant_mla.h`
- `infer_flash_attention_sparse.h` (line ~28, heavy usage)
- `flash_attention_score_antiquant_kernel.h`
- `flash_attention_score_kernel_infer_mla_fullquant.h`
- `flash_attention_score_kernel_infer_gqa_fullquant.h`

Add a `RuntimeFlags` struct to the kernel base class:
```cpp
struct RuntimeFlags {
    bool hasAtten;
    bool isPa;
    bool isFd;
    bool enableKVPrefix;
    bool enableS1OutSplit;
};
```

Initialize from TilingData in kernel Init():
```cpp
runtimeFlags.hasAtten = (tilingData->maskParams.attenMaskFlag != 0);
runtimeFlags.isPa = (tilingData->pageAttenParams.blockSize > 0);
runtimeFlags.isFd = (tilingData->fdParams.splitKVNum > 1);
```

### Phase 3: INVOKE macros (IFA/PFA entry points)

**File: `attention/incre_flash_attention/op_kernel/arch35/incre_flash_attention_entry_regbase.h`**
- Lines 165-166: Remove `hasAttenMask`, `isPa`, `isFd` from template signature
- Lines 206-208: Remove from INVOKE_FA_OP_IMPL_ASCEND950_ANTIQUANT_BASEAPI calls

**File: `attention/prompt_flash_attention/op_kernel/arch35/prompt_flash_attention_entry_regbase.h`**
- Lines 304-305: Remove from template signature
- Update all INVOKE macro calls

### Phase 4: FIAS entry point

**File: `fused_infer_attention_score_apt.cpp`**
- Lines 40-41: Remove `hasAttenMask`, `isPa`, `isFd` from template params

### Phase 5: Template tiling key declaration

**File: `fused_infer_attention_score_template_tiling_key.h`**
- Lines 90-105: Remove HasAttenMask, IsPa, IsFd from ASCENDC_TPL_ARGS_DECL
- All ASCENDC_TPL_ARGS_SEL blocks: Remove HasAttenMask, IsPa, IsFd lines
- Recalculate bit positions for remaining parameters

### Phase 6: Host-side tiling key generation

**File: `fused_infer_attention_score_tiling_impl.cpp` (arch35)**
- Update `GenTilingKey()` to NOT encode hasAttenMask/isPa/isFd in the key
- These flags should still be written to TilingData for runtime use

## Verification Plan

1. Run `count_template_instantiations.py` before and after to verify count
2. Run existing UTs for all FIAS dtype combinations
3. Benchmark: verify no performance regression on key configs:
   - LLaMA-70B D=128 decode (IFA path)
   - LLaMA-70B D=128 prefill (PFA path)
   - DeepSeek-V3 D=576 MLA decode
4. Check binary size reduction

## Rollback Strategy
Each phase can be implemented independently. If performance regression
is found for a specific boolean (e.g., `isPa` affects PagedAttention
memory access patterns), that specific boolean can be kept as a template
parameter while still type-erasing the others.
