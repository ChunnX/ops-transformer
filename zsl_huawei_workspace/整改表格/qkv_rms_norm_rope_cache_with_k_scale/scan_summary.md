# qkv_rms_norm_rope_cache_with_k_scale error log rectification scan summary

- Branch: `qkv-kscale-clean-pr-squashed`
- Scope: op api and tiling error logs for `qkv_rms_norm_rope_cache_with_k_scale`
- CANN package: `20260617000327045` toolkit + 950-ops installed under `/usr/local/Ascend`
- Scanned files:
  - `posembedding/qkv_rms_norm_rope_cache_with_k_scale/op_host/op_api/qkv_rms_norm_rope_cache_with_k_scale_checker.h`
  - `posembedding/qkv_rms_norm_rope_cache_with_k_scale/op_host/op_api/aclnn_qkv_rms_norm_rope_cache_with_k_scale.cpp`
  - `posembedding/qkv_rms_norm_rope_cache_with_k_scale/op_host/op_api/qkv_rms_norm_rope_cache_with_k_scale.cpp`
  - `posembedding/qkv_rms_norm_rope_cache_with_k_scale/op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_k_scale_base_tiling.cpp`
  - `posembedding/qkv_rms_norm_rope_cache_with_k_scale/op_host/op_tiling/arch35/qkv_rms_norm_rope_cache_with_k_scale_tiling.cpp`
- Rectified/reportable rows: 68
- Excluded rows: 48

## Rectification Coverage

- op api checker now logs parameter validation failures through `OP_LOGE_FOR_INVALID_*` helpers, covering EZ0026, EZ0027, EZ0025, EZ0020, EZ0009, EZ0010, EZ0012 and EZ0017. The qkv op_api source does not directly call the predefined error reporting interface.
- `headNums` list length uses EZ0025, and `qkv/headNums` logical head-count mismatch uses EZ0027.
- arch35 tiling parameter checks use predefined `OP_LOGE_FOR_INVALID_*` macros for value, dtype, shape, shape dim, list size and stride errors. Tiling prepare platform resources, compile info ratio checks and raw tiling size/capacity checks are also reported through predefined templates.
- Remaining excluded logs are framework pointer/resource failures, workspace/raw tiling buffer failures, launcher/executor failures or chain-summary failures; qkv source uses `OP_LOGE` for these internal/chain logs.

## Verification

- `git diff --check`: passed
- `bash build.sh --opapi_test --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`: passed, 64 tests
- `bash build.sh --ophost_test --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`: passed, 40 tests
- `bash build.sh --opapi_test --ops=grouped_matmul_finalize_routing --soc=ascend950 -j8`: passed, 169 tests
