# Tiling Review Report: mla_prolog_v3

**Date**: 2026-04-03
**Reviewer**: Claude Code (automated)
**PR**: cann/ops-transformer#3758

## Summary

- Total issues identified: 13
- Confirmed after validation: 7
- False positives filtered: 6
- Fixes applied: 7

| Dimension | Errors | Warnings | Suggestions |
|-----------|--------|----------|-------------|
| Doc-Code Consistency | 3 | 0 | 0 |
| Diagnostic Accuracy | 2 | 1 | 0 |
| Interception Completeness | 0 | 0 | 0 |
| Readability | 0 | 1 | 0 |

## 1. Documentation vs Code Consistency

### Issue 1 (error): actual_seq_len documentation contradiction
- **File**: `README.md` line 102
- **Problem**: Documentation stated "预留参数，当前版本暂未使用，必须传入空指针" (reserved, currently unused, must pass null), but line 16 describes it as used for BS merge with CacheMode="PA_BLK_BSND"/"PA_BLK_NZ". The def.cpp defines it as OPTIONAL.
- **Fix**: Updated description to reflect actual usage: optional parameter for BS merge scenarios, pass null otherwise.

### Issue 2 (error): Attributes labeled as inputs in parameter table
- **File**: `README.md` lines 104-115
- **Problem**: 12 parameters (rmsnorm_epsilon_cq/ckv, cache_mode, query_norm_flag, weight_quant_mode, kv_cache_quant_mode, query_quant_mode, ckvkr_repo_mode, quant_scale_repo_mode, tile_size, qc_qr_scale, kc_scale) are defined as `.Attr()` in `mla_prolog_v3_def.cpp` but documented as "输入" (input) in the README.
- **Fix**: Changed all 12 parameters from "输入" to "属性" to match the def.cpp registration. Also added default value note to cache_mode description.

### Issue 3 (error): Type mismatch in infershape
- **File**: `mla_prolog_v3_infershape.cpp` line 128, 130
- **Problem**: `weightQuantMode` and `kvQuantMode` declared as `const int` but assigned from dereferenced `int64_t*` pointers, causing implicit narrowing conversion.
- **Fix**: Changed both to `const int64_t`.

## 2. Tiling Diagnostic Information Accuracy

### Issue 4 (error): Wrong format specifiers for int64_t values
- **File**: `mla_prolog_tiling_check.cpp` lines 98, 105, 112, 117, 122, 127, 132
- **Problem**: Log messages use `%d` (32-bit int) for values dereferenced from `int64_t*` pointers (`weightQuantMode`, `kvQuantMode`, `queryQuantMode`, `ckvkrRepoMode`, `quantScaleRepoMode`, `tileSize`). On LP64 systems, `int64_t` is `long` (64-bit), so `%d` causes undefined behavior due to argument size mismatch.
- **Fix**: Changed all 7 occurrences from `%d` to `%ld`.

### Issue 5 (warning): Typo "pertile" in error message
- **File**: `mla_prolog_tiling_check.cpp` line 1189
- **Problem**: Error message says "pertile effective" — inconsistent with documentation which uses "per-tile".
- **Fix**: Changed "pertile" to "per-tile".

## 3. Tiling Interception Completeness

No issues found. Validation noted:
- Per-tile repo_mode validation exists at lines 895-922 (confirmed by validator).
- Null pointer check for cacheMode exists at line 146 in CheckAttrsNotNull() (false positive filtered).

## 4. Documentation & Log Readability

### Issue 6 (warning): Chinese typo in quantization scenario descriptions
- **File**: `README.md` lines 193, 200
- **Problem**: "其余出参范围非量化数据" — "范围" (range/scope) is semantically incorrect; should be "返回" (return).
- **Fix**: Changed "范围" to "返回" on both lines.

## Recommendations

- **Error message guidance**: The per-tile cache mode error at line 1189 could be improved to list which cache modes ARE supported with per-tile quantization, rather than only listing unsupported ones.
- **Quantization validation context**: Error messages for mode validation (lines 95-133) could include which `weightQuantMode` triggered the constraint, helping users understand cross-parameter dependencies.
- **English documentation**: No English version of the design document exists. Consider adding one for non-Chinese-speaking developers.

## Changes Made

| File | Changes |
|------|---------|
| `attention/mla_prolog/op_host/mla_prolog_tiling_check.cpp` | Fixed 7 format specifiers (`%d` → `%ld` for `int64_t` values); fixed "pertile" → "per-tile" typo |
| `attention/mla_prolog_v3/README.md` | Fixed actual_seq_len description; corrected 12 attribute parameters from "输入" to "属性"; fixed 2 Chinese typos ("范围" → "返回") |
| `attention/mla_prolog_v3/op_host/mla_prolog_v3_infershape.cpp` | Fixed type narrowing: `const int` → `const int64_t` for `weightQuantMode` and `kvQuantMode` |
