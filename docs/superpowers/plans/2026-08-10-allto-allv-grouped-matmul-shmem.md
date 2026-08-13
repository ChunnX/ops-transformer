# AllToAllV GroupedMatMul Ascend950 SHMEM Migration Implementation Plan

> **For Codex:** Execute this plan task-by-task with test-first red/green cycles. Use the Ascend operator development, compile/debug, precision, and performance skills at their corresponding gates. Do not modify the user's main worktree; execute only in `C:/Users/sangzhenguo/AppData/Local/Temp/ops-transformer-a2av-gmm-shmem-a5` on branch `codex/a2av-gmm-shmem-a5`.

**Goal:** Replace the public V2 Ascend950 HCCL path with the source CATCCOS UDMA/SHMEM AllToAllV + BF16 GroupedMatMul implementation while retaining V1/A3 HCCL behavior.

**Architecture:** Public `aclnnAlltoAllvGroupedMatMulV2` wraps the borrowed raw SHMEM context pointer as a one-element non-owning tensor and calls a new internal `AlltoAllvGroupedMatMulShmem` op registered only on Ascend950. This internal split is required because the existing `AlltoAllvGroupedMatMul` schema declares `MC2().HcclGroup({"group"})`; reusing it would let an executor with no group accidentally enter HCCL initialization. The new kernel ports the source's root/AIV/AIC two-stage schedule and UDMA PUT/MTE-local copy behavior with native AscendC and existing ops-transformer GMM primitives, without CATCCOS/CATLASS dependencies.

**Technology:** C++17 host/op_api, AscendC kernel, CANN SHMEM UDMA device API, ops-transformer tiling framework, GoogleTest UT, ACLNN multi-rank C++ example, Python golden/profiler harness.

**Source of truth:** `D:/1Code/catccos` branch `mingzhea2a_udma`, commit `1e04165`; design spec `docs/superpowers/specs/2026-08-10-allto-allv-grouped-matmul-shmem-design.md`.

---

## Task 1: Freeze source behavior and SHMEM SDK boundary

**Files:**

- Create: `mc2/allto_allv_grouped_mat_mul/op_kernel/arch35/shmem/allto_allv_grouped_mat_mul_shmem_constants.h`
- Create: `mc2/allto_allv_grouped_mat_mul/tests/ut/op_kernel/test_allto_allv_grouped_mat_mul_shmem_constants.cpp`
- Modify: `mc2/allto_allv_grouped_mat_mul/tests/ut/op_kernel/CMakeLists.txt`
- Modify: `mc2/allto_allv_grouped_mat_mul/op_host/CMakeLists.txt`

### Step 1: Add a failing constants/contract UT

Add a host-compilable header test that statically checks the exact source constants:

```cpp
TEST(AlltoAllvGroupedMatMulShmemConstants, MatchesSourceContract)
{
    EXPECT_EQ(ShmemConstants::kMaxEpWorldSize, 8U);
    EXPECT_EQ(ShmemConstants::kCommTileM, 64U);
    EXPECT_EQ(ShmemConstants::kCommBlockM, 64U);
    EXPECT_EQ(ShmemConstants::kCommInterval, 10U);
    EXPECT_EQ(ShmemConstants::kWorkspaceStages, 2U);
    EXPECT_EQ(ShmemConstants::kUbStages, 2U);
    EXPECT_EQ(ShmemConstants::kMaxPeerBlocks, 2U);
    EXPECT_EQ(ShmemConstants::kWorkspaceBytes, 400ULL * 1024ULL * 1024ULL + 4096ULL);
    EXPECT_EQ(ShmemConstants::kShmemAllocationLimitBytes, 1004ULL * 1024ULL * 1024ULL);
}
```

### Step 2: Run the UT and confirm RED

Run on the Linux build environment:

```bash
bash build.sh --opkernel_test --ops=allto_allv_grouped_mat_mul --soc=ascend950 --noexec
```

Expected: compilation fails because the constants header does not exist.

### Step 3: Add the constants header and conditional SHMEM dependency discovery

Define the constants in namespace `AlltoAllvGmmShmem`. In `op_host/CMakeLists.txt`, for `ascend950` only:

- resolve either the source SDK layout from `SHMEM_HOME_PATH` or the HCCL-integrated layout from `SHMEM_HOME`, accepting a CMake variable or environment variable;
- require the header that defines context setup, PE queries, UDMA PUT, fence, and signaling;
- add device include paths and link requirements only to the SHMEM op target;
- emit a fatal configuration error that names the missing variable/header instead of silently compiling HCCL.

Do not add `D:/1Code/catccos`, CATCCOS, or CATLASS include paths.

### Step 4: Run GREEN and dependency probe

```bash
bash build.sh --opkernel_test --ops=allto_allv_grouped_mat_mul --soc=ascend950 --noexec
bash build.sh --opkernel --ops=allto_allv_grouped_mat_mul --soc=ascend950 -j8
```

Expected: constants UT compiles; if SHMEM SDK is absent, the second command stops with the new explicit dependency diagnostic. Locate/install the matching CANN SHMEM SDK before continuing; do not stub device APIs.

### Step 5: Commit

```bash
git add mc2/allto_allv_grouped_mat_mul/op_kernel/arch35/shmem/allto_allv_grouped_mat_mul_shmem_constants.h mc2/allto_allv_grouped_mat_mul/tests/ut/op_kernel/test_allto_allv_grouped_mat_mul_shmem_constants.cpp mc2/allto_allv_grouped_mat_mul/tests/ut/op_kernel/CMakeLists.txt mc2/allto_allv_grouped_mat_mul/op_host/CMakeLists.txt
git commit -m "build: add alltoallv GMM SHMEM dependency boundary"
```

## Task 2: Replace the public V2 ABI and isolate the internal SHMEM op

**Files:**

- Modify: `mc2/allto_allv_grouped_mat_mul/op_api/aclnn_allto_allv_grouped_mat_mul_v2.h`
- Modify: `mc2/allto_allv_grouped_mat_mul/op_api/aclnn_allto_allv_grouped_mat_mul_v2.cpp`
- Modify: `mc2/allto_allv_grouped_mat_mul/tests/ut/op_api/test_aclnn_allto_allv_grouped_mat_mul_v2.cpp`
- Create: `mc2/allto_allv_grouped_mat_mul/op_host/allto_allv_grouped_mat_mul_shmem_def.cpp`
- Create: `mc2/allto_allv_grouped_mat_mul/op_host/allto_allv_grouped_mat_mul_shmem_infershape.cpp`
- Modify: `mc2/allto_allv_grouped_mat_mul/op_host/allto_allv_grouped_mat_mul_def.cpp`
- Modify: `mc2/allto_allv_grouped_mat_mul/op_host/CMakeLists.txt`

### Step 1: Rewrite API UTs first

Replace old group/commMode/shared-MM/permute tests with calls to:

```cpp
aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize(
    gmmX, gmmWeight, reinterpret_cast<void *>(0x1000), epWorldSize,
    sendCounts, recvCounts, gmmY, &workspaceSize, &executor);
```

Add cases for valid BF16, null context, null tensors/arrays/output/executor metadata, non-Ascend950 platform, FP16, EP 0/9, bad tensor ranks, mismatched K/N, count length not `EP*E`, negative count, `sum(sendCounts) != M`, and incorrect output M. Assert the valid workspace equals `400 MiB + 4096`.

### Step 2: Run RED

```bash
bash build.sh --opapi_test --ops=allto_allv_grouped_mat_mul --soc=ascend950
```

Expected: compilation fails because the public header still has the old signature.

### Step 3: Implement the ABI and context tensor wrapper

Declare exactly:

```cpp
ACLNN_API aclnnStatus aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize(
    const aclTensor *gmmX, const aclTensor *gmmWeight, void *shmemContext,
    int64_t epWorldSize, const aclIntArray *sendCounts,
    const aclIntArray *recvCounts, aclTensor *gmmY,
    uint64_t *workspaceSize, aclOpExecutor **executor);
```

In the `.cpp`:

- delete `CheckAndHandleCommMode`, HCCL server selection, and executor user-handle logic from V2;
- validate the source-only contract before creating an executor;
- wrap `shmemContext` with `aclCreateTensor` as borrowed storage, following `mc2/common/op_api/mc2_context.cpp::CreatMc2ContextTensor`;
- call generated `aclnnInnerAlltoAllvGroupedMatMulShmemGetWorkspaceSize`;
- execute with generated `aclnnInnerAlltoAllvGroupedMatMulShmem`;
- return the fixed user workspace requirement, while retaining any framework system workspace offset required by the generated executor.

### Step 4: Register the internal SHMEM schema

`AlltoAllvGroupedMatMulShmem` contains required inputs `gmm_x`, `gmm_weight`, `shmem_context`; required output `gmm_y`; required attrs `ep_world_size`, `send_counts`, `recv_counts`. It accepts only BF16 for data tensors, registers only `ascend950`, and must not declare `MC2().HcclGroup`.

Keep the existing internal op for V1/A3, but remove its `ascend950` AICore config. Do not alter public V1 signatures.

### Step 5: Run GREEN and A3 regression

```bash
bash build.sh --opapi_test --ops=allto_allv_grouped_mat_mul --soc=ascend950
bash build.sh --opapi_test --ops=allto_allv_grouped_mat_mul --soc=ascend910_93
```

Expected: new V2 API tests pass; existing V1 API tests pass on A3.

### Step 6: Commit

```bash
git add mc2/allto_allv_grouped_mat_mul/op_api mc2/allto_allv_grouped_mat_mul/op_host/allto_allv_grouped_mat_mul_def.cpp mc2/allto_allv_grouped_mat_mul/op_host/allto_allv_grouped_mat_mul_shmem_def.cpp mc2/allto_allv_grouped_mat_mul/op_host/allto_allv_grouped_mat_mul_shmem_infershape.cpp mc2/allto_allv_grouped_mat_mul/op_host/CMakeLists.txt mc2/allto_allv_grouped_mat_mul/tests/ut/op_api
git commit -m "feat: expose SHMEM context in alltoallv GMM V2"
```

## Task 3: Add source-compatible SHMEM tiling and validation

**Files:**

- Create: `mc2/allto_allv_grouped_mat_mul/op_host/op_tiling/arch35/allto_allv_grouped_mat_mul_shmem_tiling.h`
- Create: `mc2/allto_allv_grouped_mat_mul/op_host/op_tiling/arch35/allto_allv_grouped_mat_mul_shmem_tiling.cpp`
- Create: `mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_shmem_tiling.h`
- Create: `mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_shmem_tiling_key.h`
- Modify: `mc2/allto_allv_grouped_mat_mul/tests/ut/op_host/arch35/CMakeLists.txt`
- Replace: `mc2/allto_allv_grouped_mat_mul/tests/ut/op_host/arch35/test_allto_allv_grouped_mat_mul_tiling.cpp`

### Step 1: Write failing A5 tiling tests

Build fixtures for internal op `AlltoAllvGroupedMatMulShmem`. Cover:

- source case `M=4084,K=2048,N=4096,EP=2,E=4`;
- EP 1 through 8 and rejection of 0/9;
- both `M0=128,N0=256` and `M0=256,N0=128` selections;
- zero-token experts, uneven counts, K/N/M residue;
- `E=1` and `E=64`, rejecting 0/65;
- counts length, nonnegative values, send sum, output row sum;
- fixed workspace and `gmmX + workspace <= 1004 MiB` bounds.

Assert serialized counts values and offsets, not only a nonzero tiling key.

### Step 2: Run RED

```bash
bash build.sh --ophost_test --ops=allto_allv_grouped_mat_mul --soc=ascend950
```

Expected: registration/tiling lookup for the new internal op fails.

### Step 3: Define compact serialized tiling data

The tiling struct must include fixed-width fields for M/K/N/E/EP, selected tile sizes, communication constants, per-stage byte offsets, counts length, and embedded `sendCounts[512]`/`recvCounts[512]` (8 peers * 64 local experts). Use `BEGIN_TILING_DATA_DEF`/`REGISTER_TILING_DATA_CLASS` and static assertions that the serialized structure stays within the framework tiling buffer.

The SHMEM tiling key has no HCCL communication mode or transpose variants. Register one source-compatible BF16 key and one kernel symbol.

### Step 4: Implement validation and tiling selection

Port source checks from `ascend950_bf16_udma_moe_host_base.h` and dynamic launch logic. Compute prefix sums with checked 64-bit arithmetic. Reject overflow before narrowing kernel offsets. Select the tile family from the same shape rule as source commit `1e04165`; record the rule in a named helper and test both sides of its boundary.

### Step 5: Run GREEN and A3 regression

```bash
bash build.sh --ophost_test --ops=allto_allv_grouped_mat_mul --soc=ascend950
bash build.sh --ophost_test --ops=allto_allv_grouped_mat_mul --soc=ascend910_93
```

Expected: SHMEM tiling tests and untouched arch22 HCCL tiling tests pass.

### Step 6: Commit

```bash
git add mc2/allto_allv_grouped_mat_mul/op_host/op_tiling/arch35 mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_shmem_tiling.h mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_shmem_tiling_key.h mc2/allto_allv_grouped_mat_mul/tests/ut/op_host/arch35
git commit -m "feat: add source-compatible SHMEM tiling"
```

## Task 4: Port UDMA/MTE communication primitives

**Files:**

- Create: `mc2/allto_allv_grouped_mat_mul/op_kernel/arch35/shmem/shmem_context.h`
- Create: `mc2/allto_allv_grouped_mat_mul/op_kernel/arch35/shmem/route_table.h`
- Create: `mc2/allto_allv_grouped_mat_mul/op_kernel/arch35/shmem/udma_copy.h`
- Create: `mc2/allto_allv_grouped_mat_mul/op_kernel/arch35/shmem/communication_scheduler.h`
- Create: `mc2/allto_allv_grouped_mat_mul/tests/ut/op_kernel/test_allto_allv_grouped_mat_mul_shmem_route.cpp`
- Modify: `mc2/allto_allv_grouped_mat_mul/tests/ut/op_kernel/CMakeLists.txt`

### Step 1: Write route ownership and prefix-sum tests

For EP 1 through 8, assert:

- each peer has exactly one owner;
- local peer maps to MTE/subcore 0;
- remote peers map to UDMA/subcore 1 without sharing queue/sync ID;
- route-table prefix sums match the source flattened ordering;
- zero-payload routes do not wait on or signal an unused destination window;
- contiguous and row-fallback transfer descriptors cover identical bytes.

### Step 2: Run RED

```bash
bash build.sh --opkernel_test --ops=allto_allv_grouped_mat_mul --soc=ascend950 --noexec
```

Expected: missing route/scheduler types.

### Step 3: Implement context and route helpers

`shmem_context.h` accepts the context GM address, calls the SDK's context/sync-base setup exactly once before communication, obtains rank/size, and returns early on rank-size mismatch. `route_table.h` provides checked index and prefix helpers shared by AIV and AIC.

### Step 4: Implement UDMA and local-copy helpers

Port behavior—not CATCCOS headers—from:

- `include/catccos/comm/tile/tile_remote_copy_udma_2d.hpp`;
- `include/catccos/comm/block/comm_block_mte_udma.hpp`;
- `include/catccos/comm/block/comm_block_scheduler_alltoallv_allgather.hpp`.

Use SDK UDMA PUT for remote peers, MTE DataCopy for self, explicit fence before publishing completion, unique sync IDs, bounded 64-row blocks, one 2D/contiguous request where legal, and row-wise fallback otherwise.

### Step 5: Compile GREEN

```bash
bash build.sh --opkernel_test --ops=allto_allv_grouped_mat_mul --soc=ascend950 --noexec
```

Expected: helper and compile-only kernel UT targets pass with real SDK headers.

### Step 6: Commit

```bash
git add mc2/allto_allv_grouped_mat_mul/op_kernel/arch35/shmem mc2/allto_allv_grouped_mat_mul/tests/ut/op_kernel
git commit -m "feat: port alltoallv UDMA communication primitives"
```

## Task 5: Integrate the two-stage AIV/AIC kernel

**Files:**

- Create: `mc2/allto_allv_grouped_mat_mul/op_kernel/arch35/shmem/grouped_matmul_scheduler.h`
- Create: `mc2/allto_allv_grouped_mat_mul/op_kernel/arch35/shmem/allto_allv_grouped_mat_mul_shmem_kernel.h`
- Create: `mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_shmem.cpp`
- Create: `mc2/allto_allv_grouped_mat_mul/tests/ut/op_kernel/test_allto_allv_grouped_mat_mul_shmem.cpp`
- Modify: `mc2/allto_allv_grouped_mat_mul/tests/ut/op_kernel/allto_allv_grouped_mat_mul_tiling_def.h`
- Modify: `mc2/allto_allv_grouped_mat_mul/tests/ut/op_kernel/CMakeLists.txt`

### Step 1: Add failing kernel contract tests

Test kernel argument order (`gmmX`, `gmmWeight`, `shmemContext`, `gmmY`, framework workspace, tiling), workspace stage offsets, communication-to-compute flags, stage reuse, last partial tile, and zero-token expert behavior. Add compile instantiations for both M0/N0 families.

### Step 2: Run RED

```bash
bash build.sh --opkernel_test --ops=allto_allv_grouped_mat_mul --soc=ascend950 --noexec
```

Expected: missing kernel symbol and scheduler.

### Step 3: Implement the source AIV schedule

Port `grouped_matmul_alltoallv_tla.hpp` scheduling semantics:

1. root initializes compute/communication signals and synchronizes AIVs;
2. local owner performs MTE copy, remote owners perform UDMA PUT;
3. owners wait for prior compute only for non-empty payloads;
4. owners fence and publish completed block counts;
5. root waits for the local receive set and releases AIC;
6. alternate between the two symmetric stages.

Keep signal storage in the final 4096-byte workspace page and assert alignment/size at compile time.

### Step 4: Implement the AIC GroupedMatMul schedule

Use the target repository's `QuantGroupedMatmul`/matmul building blocks only where BF16 ND semantics match the source. For each stage, local expert, and source rank, derive received row ranges from `recvCounts`, run the selected M0/N0/K0 configuration, write valid rows contiguously, and publish stage reuse after all relevant AIC work completes.

The implementation must not instantiate shared-MM, transpose, quantization, permute output, or HCCL scheduler branches.

### Step 5: Compile GREEN and inspect generated kernel

```bash
bash build.sh --opkernel_test --ops=allto_allv_grouped_mat_mul --soc=ascend950 --noexec
bash build.sh --opkernel --ops=allto_allv_grouped_mat_mul --soc=ascend950 --dump_cce -j8
```

Expected: the SHMEM symbol is generated once, has the intended mixed AIC/AIV task type, contains UDMA/MTE operations, and has no HCCL/AICPU communication symbols.

### Step 6: Commit

```bash
git add mc2/allto_allv_grouped_mat_mul/op_kernel mc2/allto_allv_grouped_mat_mul/tests/ut/op_kernel
git commit -m "feat: add SHMEM alltoallv grouped matmul kernel"
```

## Task 6: Wire binary registration and package builds

**Files:**

- Create: `mc2/allto_allv_grouped_mat_mul/op_host/config/ascend950/allto_allv_grouped_mat_mul_shmem_binary.json`
- Create: `mc2/allto_allv_grouped_mat_mul/op_host/config/ascend950/allto_allv_grouped_mat_mul_shmem_simplified_key.ini`
- Modify: `mc2/allto_allv_grouped_mat_mul/op_host/config/ascend950/allto_allv_grouped_mat_mul_binary.json`
- Modify: `mc2/allto_allv_grouped_mat_mul/op_host/config/ascend950/allto_allv_grouped_mat_mul_simplified_key.ini`
- Modify: `mc2/allto_allv_grouped_mat_mul/op_host/CMakeLists.txt`
- Modify: `mc2/allto_allv_grouped_mat_mul/CMakeLists.txt`

### Step 1: Add a failing package smoke check

```bash
bash build.sh --pkg --ops=allto_allv_grouped_mat_mul --soc=ascend950 --vendor_name=customize -j8
```

Expected before registration: generated inner API or SHMEM binary metadata is missing.

### Step 2: Register only the SHMEM op on Ascend950

Map `AlltoAllvGroupedMatMulShmem` to `allto_allv_grouped_mat_mul_shmem` and its single tiling key. Remove/disable Ascend950 binary selection for the old HCCL internal op while leaving `ascend910_93` files untouched. Apply the A5 compiler flags to the new op name.

### Step 3: Run A5 and A3 package builds

```bash
bash build.sh --pkg --ops=allto_allv_grouped_mat_mul --soc=ascend950 --vendor_name=customize -j8
bash build.sh --pkg --ops=allto_allv_grouped_mat_mul --soc=ascend910_93 --vendor_name=customize -j8
```

Expected: A5 package contains the SHMEM binary/inner op; A3 package contains the existing HCCL binary and V1 API. No CATCCOS/CATLASS path appears in compile commands or package contents.

### Step 4: Commit

```bash
git add mc2/allto_allv_grouped_mat_mul/CMakeLists.txt mc2/allto_allv_grouped_mat_mul/op_host
git commit -m "build: register alltoallv GMM SHMEM kernel on A5"
```

## Task 7: Replace the A5 example and build a 30-case precision suite

**Files:**

- Replace: `mc2/allto_allv_grouped_mat_mul/examples/arch35/test_aclnn_allto_allv_grouped_mat_mul.cpp`
- Modify: `mc2/allto_allv_grouped_mat_mul/tests/assets/inputs.py`
- Modify: `mc2/allto_allv_grouped_mat_mul/tests/assets/golden.py`
- Create: `mc2/allto_allv_grouped_mat_mul/tests/pytest/test_allto_allv_grouped_mat_mul_shmem.py`
- Create: `mc2/allto_allv_grouped_mat_mul/tests/pytest/shmem_cases.jsonl`
- Create: `mc2/allto_allv_grouped_mat_mul/tests/pytest/run_shmem_precision.sh`

### Step 1: Generate the case manifest first

Create at least 30 explicit JSONL cases spanning every EP from 1 through 8, both tile families, uneven/zero routes, residue M/K/N, E boundaries, small smoke shapes, and source case `4084x2048x4096`. Validate the manifest itself: unique names, correct counts lengths, nonnegative counts, and send totals.

### Step 2: Write the caller-owned SHMEM example

Follow source `main.cpp`:

- initialize ACL/runtime and `aclshmemx_init_attr` with UDMA;
- obtain `shmemx_get_ffts_config()` and pass it directly as `shmemContext`;
- allocate `gmmX` and fixed workspace via `aclshmem_malloc/calloc` from the same heap;
- keep weights/output in normal device memory unless the SDK requires otherwise;
- call new V2 workspace/execution APIs;
- synchronize, report recent error, free all allocations, finalize SHMEM/ACL in reverse order.

### Step 3: Implement the exact golden mapping

Port `examples/utils/gen_data_gmm_alltoallv.py`: flatten send counts as destination-rank/expert, flatten receive counts as source-rank/local-expert, route rows, run per-local-expert FP32 matmul, concatenate valid rows, then cast/compare with BF16 tolerances. Do not replace route tables with uniform-count assumptions.

### Step 4: Run multi-rank precision

```bash
bash build.sh --run_example allto_allv_grouped_mat_mul eager --soc=ascend950 --example_name=test_aclnn_allto_allv_grouped_mat_mul
bash mc2/allto_allv_grouped_mat_mul/tests/pytest/run_shmem_precision.sh
```

Expected: source case and every hardware-executable JSONL case passes. Record EP cases skipped solely due to device count as skipped, never passed.

### Step 5: Commit

```bash
git add mc2/allto_allv_grouped_mat_mul/examples/arch35 mc2/allto_allv_grouped_mat_mul/tests/assets mc2/allto_allv_grouped_mat_mul/tests/pytest
git commit -m "test: add multi-rank SHMEM precision coverage"
```

## Task 8: Update the public documentation

**Files:**

- Modify: `mc2/allto_allv_grouped_mat_mul/README.md`
- Modify: `mc2/allto_allv_grouped_mat_mul/docs/aclnnAlltoAllvGroupedMatMulV2.md`
- Modify: `mc2/allto_allv_grouped_mat_mul/docs/aclnnAlltoAllvGroupedMatMul.md`

### Step 1: Write documentation assertions/checklist

Check the docs contain: new exact signature, Ascend950-only statement, BF16/TP1/EP2-8 constraints, counts ordering, output shape, borrowed context ownership, `aclshmem_calloc` workspace requirement, exact `400 MiB + 4096` size, `gmmX+workspace <=1004 MiB`, and unsupported shared-MM/transpose/permute features.

### Step 2: Rewrite V2 and cross-reference V1

Use the example as the canonical lifecycle. State V1 is A3/HCCL and V2 is A5/SHMEM; do not imply ABI compatibility or automatic HCCL-to-SHMEM conversion.

### Step 3: Verify stale names are absent

```bash
rg -n "commMode|comm_mode|const char \*group|shared expert|permuteOut" mc2/allto_allv_grouped_mat_mul/docs/aclnnAlltoAllvGroupedMatMulV2.md mc2/allto_allv_grouped_mat_mul/examples/arch35
```

Expected: no old V2 arguments/usages except explicit unsupported-feature prose.

### Step 4: Commit

```bash
git add mc2/allto_allv_grouped_mat_mul/README.md mc2/allto_allv_grouped_mat_mul/docs
git commit -m "docs: document alltoallv GMM SHMEM V2 contract"
```

## Task 9: Profile source and migrated implementations

**Files:**

- Create: `mc2/allto_allv_grouped_mat_mul/tests/performance/profile_shmem.py`
- Create: `mc2/allto_allv_grouped_mat_mul/tests/performance/cases.jsonl`
- Create: `mc2/allto_allv_grouped_mat_mul/tests/performance/README.md`
- Create: `mc2/allto_allv_grouped_mat_mul/tests/performance/results/.gitkeep`

### Step 1: Define representative cases and profiler contract

Select source case plus small/residue/uneven cases for each available EP. Use fixed warmup and active iterations, capture rank-local traces, and report total kernel duration plus communication/GMM stages only where trace names make them attributable.

### Step 2: Run source and migrated profiles

Run source commit `1e04165` and the migrated package with the same device set, shapes, routes, frequency mode, and iteration counts. Persist raw profiler output outside git; commit only scripts, metadata, and summarized results.

### Step 3: Validate claims

Reject summaries with missing ranks, failed precision, unequal shapes/routes, or mixed warmup settings. Report unavailable EP8 hardware as not executed. Do not claim improvement unless the comparable median result supports it.

### Step 4: Commit

```bash
git add mc2/allto_allv_grouped_mat_mul/tests/performance
git commit -m "perf: add SHMEM migration profiler suite"
```

## Task 10: Full verification and handoff

**Files:**

- Modify if needed: only files already listed above
- Review: `docs/superpowers/specs/2026-08-10-allto-allv-grouped-matmul-shmem-design.md`

### Step 1: Run focused UT matrix

```bash
bash build.sh --opapi_test --ophost_test --ops=allto_allv_grouped_mat_mul --soc=ascend950
bash build.sh --opkernel_test --ops=allto_allv_grouped_mat_mul --soc=ascend950 --noexec
bash build.sh --opapi_test --ophost_test --ops=allto_allv_grouped_mat_mul --soc=ascend910_93
bash build.sh --opkernel_test --ops=allto_allv_grouped_mat_mul --soc=ascend910_93 --noexec
```

### Step 2: Run clean package builds

```bash
bash build.sh --make_clean
bash build.sh --pkg --ops=allto_allv_grouped_mat_mul --soc=ascend950 --vendor_name=customize -j8
bash build.sh --make_clean
bash build.sh --pkg --ops=allto_allv_grouped_mat_mul --soc=ascend910_93 --vendor_name=customize -j8
```

### Step 3: Run executable validation

```bash
bash mc2/allto_allv_grouped_mat_mul/tests/pytest/run_shmem_precision.sh
python mc2/allto_allv_grouped_mat_mul/tests/performance/profile_shmem.py --validate-results
```

### Step 4: Inspect the diff and repository state

```bash
git diff --check
git status --short
git diff 95d31c5a4...HEAD -- mc2/allto_allv_grouped_mat_mul docs/superpowers
```

Confirm no sibling CATCCOS paths, unrelated files, generated binaries, raw profiler traces, or credentials are included. Confirm old V2 declarations/call sites are gone and V1 signatures are unchanged.

### Step 5: Record evidence

In the final handoff, list exact commands, pass/fail counts, tested device topology, skipped hardware cases, SHMEM SDK/CANN version, and profiler result location. If any required hardware case is not executable, state it as a limitation and do not mark the corresponding acceptance criterion as verified.
