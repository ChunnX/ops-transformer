# QkvRmsNormRopeCacheWithKScale Task Plan

## Rules

1. Work on `master` in this repository.
2. Complete tasks in dependency order.
3. Every task must leave the repository compiling.
4. Host-side tasks require UT coverage.
5. Kernel synchronization tasks currently require compile verification and
   static protocol review; sync-specific tests can be added later.
6. After a task is coded and verified, update this file and commit the task.

## Status

| ID | Task | Status | Verification | Commit |
|----|------|--------|--------------|--------|
| S00 | Create implementation scaffold | Done | Top-level CMake configure reached this no-op directory; configure then failed in pre-existing mc2/ffn targets unrelated to this scaffold | Included in scaffold commit |
| T00 | Host checker/tiling contract + UT | Done | Sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh` and passed `bash build.sh --ophost_test --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`; 9 host UT passed. `--noexec` host UT compile also passed | Included in T00 commit and host UT build.sh fix commit |
| P01 | Empty kernel package precheck | Done | Sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh` and passed `bash build.sh --pkg --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`; package generated at `build/cann-ops-transformer-custom_linux-x86_64.run` | e2b91bd90 |
| T01 | Controller and basic block skeleton | Done | Sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh`; passed `bash build.sh --ophost_test --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8` with 9 host UT, and passed `bash build.sh --pkg --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`; package generated at `build/cann-ops-transformer-custom_linux-x86_64.run`. Re-ran `--pkg` after splitting the D=128 fast-path kernel template into its own header. | 76d2f307f plus this commit |
| T02 | AIV local slice and tail row range | Done | Sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh`; passed `bash build.sh --ophost_test --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8` with 11 host UT, and passed `bash build.sh --pkg --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`; package generated at `build/cann-ops-transformer-custom_linux-x86_64.run` | This commit |
| T03 | Static buffer and tensor binding | Done | Direct `LocalTensor<T>(TPosition, offset, size)` binding without `TPipe`/`TBuf`; sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh`; passed `bash build.sh --pkg --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`, package generated at `build/cann-ops-transformer-custom_linux-x86_64.run`; passed `bash build.sh --ophost_test --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8` with 11 host UT | This commit |
| T04 | CrossCore wrappers and flag ids | Done | Added fixed semantic flags `SYNC_A_READY/C_READY/C_CONSUMED` and AIC/AIV wrappers; static review confirmed only `flag + 16` AIV1 handling and no `bufferId`/generation-derived flags; sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh`; passed `bash build.sh --pkg --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`, package generated at `build/cann-ops-transformer-custom_linux-x86_64.run`; passed `bash build.sh --ophost_test --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8` with 11 host UT | This commit |
| R01 | Split D=128 fast-path kernel modules | Done | Split `common`, `controller`, `basic_block`, `cube`, and `vec` into separate headers; `git diff --check` passed; source scan confirmed no `TPipe`/`TBuf`/`InitBuffer` or `BEGIN_TILING_DATA_DEF` usage in the kernel path; sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh`; passed `bash build.sh --pkg --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`, package generated at `build/cann-ops-transformer-custom_linux-x86_64.run`; passed `bash build.sh --ophost_test --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8` with 11 host UT | This commit |
| R02 | Remove empty fast-path aggregate header | Done | Deleted the empty aggregate header and made the kernel entry include `controller` directly; scan found no remaining reference to the removed header; `git diff --check` passed; source scan confirmed no `TPipe`/`TBuf`/`InitBuffer` or `BEGIN_TILING_DATA_DEF` usage in the kernel path; sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh`; passed `bash build.sh --pkg --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`, package generated at `build/cann-ops-transformer-custom_linux-x86_64.run` and no longer listed the removed header; passed `bash build.sh --ophost_test --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8` with 11 host UT | This commit |
| R03 | Move core token split to tiling | Done | Host tiling now computes `coreTokenTile = ceil(totalTokens / aicNum)` and `coreGroupNum`; controller consumes those tiling fields to build the core token range, expands it into token-size tiles, and calls the basic-block single-tile AIC/AIV entries; Cube/Vec bind their own typed GM tensors and controller/basic block do not retain a shared bound-GM bundle; core-type dispatch is centralized in controller AIC/AIV entries with no lower-layer `g_coreType` checks; `git diff --check` passed; source scan confirmed no old `baseTokenSize`/`totalTokens / core` math, no retained shared bound-GM bundle, and no `TPipe`/`TBuf`/`InitBuffer` or `BEGIN_TILING_DATA_DEF`; sourced `/usr/local/Ascend/cann-9.1.0/set_env.sh`; passed `bash build.sh --ophost_test --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8` with 11 host UT including `CoreTokenRangeUsesHostCeilTokenTile`; passed `bash build.sh --pkg --ops=qkv_rms_norm_rope_cache_with_k_scale --soc=ascend950 -j8`, package generated at `build/cann-ops-transformer-custom_linux-x86_64.run` | This commit |
| T05 | Intra-core event constants and Init/End | Not started | Kernel compile + event id static check | |
| T06 | AIV Q/K/V preprocess | Not started | Kernel compile + preprocess UT | |
| T07 | AIC rotation resident and Cube matmul | Not started | Kernel compile + Cube UT | |
| T08 | AIC Fixpipe distribution and C_READY | Not started | Kernel compile + static review | |
| T09 | V branch early postprocess | Not started | Kernel compile + V path UT | |
| T10 | AIV Q/K postprocess | Not started | Kernel compile + Q/K postprocess UT | |
| T11 | Rolling pipeline and drain | Not started | Kernel compile + static review | |
| T12 | Tail and empty-slice synchronization | Not started | Kernel compile + Host UT extension | |
| T13 | Synchronization protocol audit | Not started | Kernel compile + static audit | |
| T14 | Kernel UT/ST closure | Not started | Full UT/ST | |

## Next Task

T05 is the next implementation task. It should add intra-core event constants
and Init/End pairing for the live UB/L1 reuse paths while keeping every event id
within the allowed range. Do not allocate AIV intra-core event ids for
`a_rot_l1_nz`; it remains a CrossCore-protected Cube AL1 data slot.

## Kernel Layout

The D=128 fast path keeps the kernel entry as a thin dispatch shell that
includes `controller` directly. Shared constants, raw GM argument bundle, scalar
helpers, AIV slice helper, and CrossCore wrappers live in `common`; controller
owns core-range derivation, token-size tile expansion, and explicit AIC/AIV
entry dispatch; basic block owns lifecycle hooks and static local tensor
binding; Cube and Vector code bind their own typed GM tensor subsets in their
module headers.
Host tiling owns the ceil-div core token split and writes the resulting
`coreTokenTile` and `coreGroupNum` into the kernel POD tiling data.
