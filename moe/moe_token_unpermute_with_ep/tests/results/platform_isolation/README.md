# Ascend950 platform-isolation audit

Corrective commit: `289067d40d5d626785282cab8a9450a28cf1af61`.
Baseline used for all non-950 comparisons: `origin/master`.

## Public-path impact inventory

| File | Ascend950 / architecture 3510 | Non-3510 behavior |
|---|---|---|
| `op_kernel/moe_token_unpermute_with_ep.h` | Enables sorted-index assertions, stable high-TopK reduction, four reduction buffers, their initialization, and the new call sites. | The exact `origin/master` device chain is selected; all 430 non-empty baseline lines compare equal. |
| `op_host/moe_token_unpermute_with_ep_tiling.cpp` | Enables empty-range fallback/validation and budgets four stable-reduction buffers through `MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950`. | The exact `origin/master` validation, initialization, TopK handling, and UB formula are selected; all 338 non-empty baseline lines compare equal. |
| `op_host/moe_token_unpermute_with_ep_tiling.h` | Declares `STABLE_REDUCTION_BUFFER_NUM=4` under the host-only 950 definition. | The exact 88 non-empty `origin/master` lines remain. |
| `op_host/CMakeLists.txt` | `ASCEND_COMPUTE_UNIT=ascend950` selects `op_host_aclnnExc`, `ACLNNTYPE aclnn_exclude`, the 950-only manual API directory, and the host compile definition. | Selects the original `op_host_aclnn`, original compile options, and original automatic `ACLNNTYPE aclnn` path. |
| `op_host/op_api/ascend950/*` | Contains the manual ACLNN implementation required by the 950 route. | No manual API `.cpp` remains in the shared `op_api` directory, so non-950 builds cannot consume it. |

Host code does not use `__NPU_ARCH__`. Device code uses the repository-required
guard exactly:

```cpp
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    // Ascend950-only path
#else
    // origin/master path
#endif
```

## Static gate

`check_platform_isolation.py --baseline-ref origin/master` writes
`static_audit.json` and exits zero only when:

- every non-3510 Kernel/Host branch is line-equivalent to `origin/master`;
- the exact architecture guard is present and no host source uses
  `__NPU_ARCH__`;
- the old host target, old compile options, and automatic ACLNN selection are
  preserved exactly once;
- manual API sources exist only below `op_api/ascend950`.

Current result: **PASS**.

The same gate was rerun against the post-commit Ascend910B build snapshot. It
also checks the generated compile database and accepts the build only when the
950 host definition and `op_api/ascend950` sources are absent, the original
`op_host_aclnn` target drives `OPS_ACLNN_GEN=1`, and all three legacy Kernel
objects exist.

## Ascend950 build evidence

- Package build: PASS.
- Package SHA-256:
  `596cb40f6e33ccb51a6c7da4226984715afe3f22c83030dc4d6fffd57eb435e3`.
- Isolated install root:
  `/home/z50058944/Ascend/moe_token_unpermute_with_ep_validation_isolation_289067d40`.
- Repository and installed public Kernel header SHA-256 are both
  `ddbe0ef79169b692e3c62c9244659f52138ac29eea993d2194379f0b369eccdd`.
- Installed FP32/BF16/FP16 object SHA-256 values are respectively
  `141789515f510a9780786d338a9820d67a2cf801dab4f36c298d9eb95b4dd50f`,
  `df1dd4a471f8d6c49bdb5527b711e59bd76d1035b647d4fab430641ee556a3a6`,
  and `a8d87d0c0d607c7565f39eddf2e653840ee93f642b7a2258e8ab149d98a0cd57`.
- Generated host compile command contains
  `-DMOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950=1`.
- Generated OpDef command contains `OPS_ACLNN_GEN=0`, uses
  `op_host_aclnnExc`, and targets `ascend950`.
- Host UT: 30/30 PASS (tiling 21/21, InferShape 9/9).

Raw build/run packages and build logs remain outside Git under
`/home/z50058944/moe_platform_isolation_evidence_289067d40/`.

## Ascend910B compile evidence

- Full package compile after corrective commit `289067d40`: **PASS**.
- `ASCEND_COMPUTE_UNIT=ascend910b`; the generated OpDef command selects
  `libop_host_aclnn.so`, `OPS_ACLNN_GEN=1`, and
  `OPS_PRODUCT_NAME="ascend910b"`.
- The compile database contains the generated
  `build/autogen/aclnn_moe_token_unpermute_with_ep.cpp` and contains neither
  `op_api/ascend950` nor
  `MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950`.
- Package SHA-256:
  `01ed02ca818629112d2e25af9c04e49604611d3cd463e9a509a13b34aa29ec92`.
- FP32/BF16/FP16 object SHA-256 values are respectively
  `98130a1112ba087dcc5e7856bd3a1f5fcb9ec5a4bda09d5a892bbdfa7eb404d0`,
  `e765b21ab587ca38493180ce34bf1dffca6c501bf41d397a2bc8c3fbf8542d86`,
  and `3a66d23ce2c4c8727a548d5ec9496affeae56a65993ac861e375c662b1a6a4b9`.
- The exact build snapshot and hash manifest are under
  `/home/z50058944/moe_platform_isolation_evidence_289067d40/ascend910b/`.

This machine has no Ascend910B device assigned, so the 910B evidence is
compile-only. No 910B runtime regression is claimed.
