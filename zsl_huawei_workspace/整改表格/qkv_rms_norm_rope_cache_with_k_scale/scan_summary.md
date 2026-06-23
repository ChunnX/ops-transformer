# qkv_rms_norm_rope_cache_with_k_scale error log rectification scan summary

- Branch: `qkv-kscale-clean-pr-squashed`
- Scope: op api and tiling error logs for `qkv_rms_norm_rope_cache_with_k_scale`
- Scanned files:
  - `posembedding/qkv_rms_norm_rope_cache_with_k_scale/op_host/op_api/qkv_rms_norm_rope_cache_with_k_scale_checker.h`
  - `posembedding/qkv_rms_norm_rope_cache_with_k_scale/op_host/op_api/aclnn_qkv_rms_norm_rope_cache_with_k_scale.cpp`
  - `posembedding/qkv_rms_norm_rope_cache_with_k_scale/op_host/op_api/qkv_rms_norm_rope_cache_with_k_scale.cpp`
  - `posembedding/qkv_rms_norm_rope_cache_with_k_scale/op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_k_scale_base_tiling.cpp`
  - `posembedding/qkv_rms_norm_rope_cache_with_k_scale/op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_k_scale_tiling.cpp`
- Rectified/reportable rows: 62
- Excluded rows: 54

## Rectification Coverage

- op api checker now logs parameter validation failures through `OP_LOGE_WITHOUT_REPORT` plus explicit `REPORT_PREDEFINED_ERR_MSG`, covering EZ0026, EZ0027, EZ0025, EZ0020, EZ0009, EZ0010, EZ0012 and EZ0017.
- `headNums` list length uses EZ0025, and `qkv/headNums` logical head-count mismatch uses EZ0027.
- arch35 tiling parameter checks use predefined `OP_LOGE_FOR_INVALID_*` macros for value, dtype, shape, shape dim, list size and stride errors.
- Remaining raw `OP_LOGE` / null-check logs are framework, platform, workspace, raw tiling buffer, compileInfo, launcher/executor, or chain-summary failures and are listed in the exclusion table.

## Verification

- `git diff --check`: passed
- `bash build.sh --opapi_test --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`: passed, 64 tests
- `bash build.sh --ophost_test --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`: passed previously for the tiling-side rectification
