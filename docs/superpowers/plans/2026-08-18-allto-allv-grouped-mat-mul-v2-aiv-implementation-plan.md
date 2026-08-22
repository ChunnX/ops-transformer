# AlltoAllvGroupedMatMulV2 AIV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `AlltoAllvGroupedMatMulV2` on Ascend 910C as an AIV-only, expert-at-a-time fused operator with manual HCCL-window communication and CATLASS computation, while keeping V1 behavior intact.

**Architecture:** Keep the public V2 ACLNN signature, add a dedicated arch22 AIV tiling/kernel path, publish local input and counts to the HCCL window, gather one local expert at a time from peer windows, and notify AIC cores to run one CATLASS GEMM per expert. Preserve the optional shared-expert MM and `permuteOut`, but remove V2 dependencies on the legacy HCCL AlltoAllV call, high-level GroupedMatMul, and `expertNum` batching.

**Tech Stack:** C++17, AscendC mixed AIC/AIV kernels, HCCL window/context ABI, CATLASS `Arch::AtlasA2`, ACLNN two-stage API, GoogleTest, ops-transformer build system, mc2_test, BF16.

## Global Constraints

- Repository: `/home/sangzhenguo/code/ops-transformer`.
- Test project: `/home/sangzhenguo/code/mc2_test`.
- Conda environment: `szg_py311`.
- Target platform: Ascend 910C / `ascend910_93` only.
- Target CANN: 9.2 or later. The current server has 9.1.0; use 9.1.0 only for early compile feedback and do not claim final compatibility until 9.2+ verification is run.
- V2 accepts only `commMode="aiv"`; V1 retains its legacy `ai_cpu` behavior.
- V2 supports only BF16 and ND.
- Communication loop granularity is exactly one local expert; do not implement chunk or `expertNum` batching.
- Do not call `hccl_.AlltoAllV()` or any AlltoAllV high-level communication API in the V2 path.
- Do not call MatMul/GroupedMatMul high-level compute APIs in the V2 path; use CATLASS for both GMM and optional shared MM.
- Keep `sendCountsTensorOptional` and `recvCountsTensorOptional` null-only; require `aclIntArray` send/recv counts.
- Preserve unrelated untracked files under `mc2/all_gather_matmul_v2/tests/st/` and `mc2/matmul_reduce_scatter_v2/tests/st/`.
- Follow TDD: add or change a test, observe failure, implement the minimum behavior, rerun, then commit.

---

## File Map

### ops-transformer files

- Modify `mc2/allto_allv_grouped_mat_mul/op_api/aclnn_allto_allv_grouped_mat_mul_v2.cpp`: V2 AIV-only validation and MTE server selection.
- Modify `mc2/allto_allv_grouped_mat_mul/op_host/op_tiling/allto_allv_grouped_mat_mul_tiling.cpp`: route V1 to legacy tiling and V2/AIV to the new arch22 tiling implementation.
- Modify `mc2/allto_allv_grouped_mat_mul/op_host/op_tiling/allto_allv_grouped_mat_mul_tiling.h`: declare the AIV tiling class and common validation helpers.
- Create `mc2/allto_allv_grouped_mat_mul/op_host/op_tiling/arch22/allto_allv_grouped_mat_mul_aiv_tiling.h`: AIV tiling class declaration.
- Create `mc2/allto_allv_grouped_mat_mul/op_host/op_tiling/arch22/allto_allv_grouped_mat_mul_aiv_tiling.cpp`: AIV block dim, workspace, CATLASS tile, and HCCL resource tiling.
- Modify `mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_tiling_key.h`: compile only AIV communication keys for V2 while retaining the legacy entry needed by V1.
- Modify `mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_tiling.h`: add the AIV-specific POD tiling data.
- Create `mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_aiv_comm.h`: peer-window layout, prefix offsets, sync protocol, and ping-pong GM movement.
- Create `mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_catlass.h`: main-expert and shared-expert CATLASS wrappers.
- Create `mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_aiv_mode.h`: mixed-kernel orchestrator and expert pipeline.
- Modify `mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul.cpp`: dispatch the AIV V2 tiling key to the new path and leave legacy V1 dispatch available.
- Modify `mc2/allto_allv_grouped_mat_mul/CMakeLists.txt`: register new host/kernel sources if the repository does not glob them.
- Modify `mc2/allto_allv_grouped_mat_mul/tests/ut/op_api/test_aclnn_allto_allv_grouped_mat_mul_v2.cpp`: V2 communication mode and dtype tests.
- Modify `mc2/allto_allv_grouped_mat_mul/tests/ut/op_host/arch22/test_allto_allv_grouped_mat_mul_tiling.cpp`: AIV tiling, workspace, counts, and V1 isolation tests.
- Modify `mc2/allto_allv_grouped_mat_mul/tests/ut/op_kernel/allto_allv_grouped_mat_mul_tiling_def.h`: AIV tiling fixture data.
- Modify `mc2/allto_allv_grouped_mat_mul/tests/ut/op_kernel/test_allto_allv_grouped_mat_mul.cpp`: expert-offset, zero-token, transpose, and optional-output kernel cases.
- Create `mc2/allto_allv_grouped_mat_mul/tests/allto_allv_grouped_mat_mul_v2-test-cases.md`: canonical functional, precision, boundary, and performance case set.
- Update `mc2/allto_allv_grouped_mat_mul/docs/aclnnAlltoAllvGroupedMatMulV2.md`: BF16/AIV-only public behavior.

### mc2_test files

- Create `/home/sangzhenguo/code/mc2_test/cpp/obp/aclnnAlltoAllvGroupedMatMulV2.cc`: V2 ACLNN executable.
- Modify `/home/sangzhenguo/code/mc2_test/cpp/Makefile`: dedicated V2 build target.
- Create `/home/sangzhenguo/code/mc2_test/op_class/aclnnAlltoAllvGroupedMatMulV2.py`: V2 registration and `comm_mode="aiv"` case handling while reusing the proven golden data model.
- Modify `/home/sangzhenguo/code/mc2_test/op_class/__init__.py`: import the V2 class.
- Create `/home/sangzhenguo/code/mc2_test/excel/aclnnAlltoAllvGroupedMatMulV2.xlsx`: BF16 EP=2/4/8 case matrix generated from the testcase document.

---

### Task 1: Freeze the testcase contract

**Required skill:** `ascendc-operator-testcase-gen`

**Files:**
- Create: `mc2/allto_allv_grouped_mat_mul/tests/allto_allv_grouped_mat_mul_v2-test-cases.md`
- Read: `docs/superpowers/specs/2026-08-17-allto-allv-grouped-mat-mul-v2-aiv-design.md`

**Interfaces:**
- Consumes: the approved design constraints and the current mc2_test case schema.
- Produces: named API, tiling, kernel, multi-rank precision, and performance cases used by Tasks 2-10.

- [ ] **Step 1: Write the case table**

Include this minimum matrix with concrete values:

```markdown
| ID | EP | E/rank | local M | K | N | send distribution | transGmmW | permute | shared MM |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| basic_ep2 | 2 | 2 | 16 | 256 | 128 | uniform | false | false | no |
| permute_ep2 | 2 | 2 | 15 | 256 | 96 | skewed | false | true | no |
| zero_peer_ep4 | 4 | 2 | 19 | 320 | 128 | source rank 2 sends zero | false | true | no |
| zero_expert_ep4 | 4 | 4 | 23 | 256 | 192 | local expert 1 receives zero | true | true | no |
| single_expert_ep8 | 8 | 2 | 31 | 512 | 256 | all tokens to one global expert | false | false | no |
| tail_tile_ep8 | 8 | 2 | 37 | 272 | 130 | uneven and non-tile-aligned | true | true | no |
| shared_ff | 2 | 2 | 16 | 256 | 128 | uniform | false | false | yes, transMmW=false |
| shared_ft | 2 | 2 | 16 | 256 | 128 | uniform | false | true | yes, transMmW=true |
```

- [ ] **Step 2: Add explicit invariants and golden ordering**

Document:

```text
sendCounts length = EP * E
recvCounts length = EP * E
sum(sendCounts) = local M
gmmY rows = sum(recvCounts)
output order = local expert major, source rank minor
zero-token experts participate in synchronization but emit no rows
```

- [ ] **Step 3: Add invalid cases**

Name the cases `invalid_comm_ai_cpu`, `invalid_comm_ccu`, `invalid_fp16`, `invalid_count_length`, `invalid_count_sum`, `invalid_window_size`, `invalid_partial_shared_mm`, and `invalid_permute_output` with the expected ACLNN or tiling failure stage.

- [ ] **Step 4: Verify the document is complete**

Run:

```bash
grep -nE 'basic_ep2|zero_expert_ep4|tail_tile_ep8|invalid_comm_ai_cpu|invalid_window_size' \
  mc2/allto_allv_grouped_mat_mul/tests/allto_allv_grouped_mat_mul_v2-test-cases.md
```

Expected: every named case appears exactly once in its primary case table.

- [ ] **Step 5: Commit**

```bash
git add mc2/allto_allv_grouped_mat_mul/tests/allto_allv_grouped_mat_mul_v2-test-cases.md
git commit -m "test: define AIV alltoallv grouped matmul v2 cases"
```

### Task 2: Make V2 AIV-only without breaking V1

**Required skills:** `superpowers:test-driven-development`, `ascendc-operator-code-gen`

**Files:**
- Modify: `mc2/allto_allv_grouped_mat_mul/tests/ut/op_api/test_aclnn_allto_allv_grouped_mat_mul_v2.cpp`
- Modify: `mc2/allto_allv_grouped_mat_mul/op_api/aclnn_allto_allv_grouped_mat_mul_v2.cpp`

**Interfaces:**
- Consumes: existing V2 two-stage ACLNN signature.
- Produces: `CheckAndHandleCommMode()` accepting only `"aiv"`; execute path selecting `NNOPBASE_HCCL_SERVER_TYPE_MTE`.

- [ ] **Step 1: Add failing API tests**

Add test cases asserting:

```cpp
EXPECT_EQ(CallGetWorkspaceSizeWithCommMode("aiv"), ACLNN_SUCCESS);
EXPECT_EQ(CallGetWorkspaceSizeWithCommMode("ai_cpu"), ACLNN_ERR_PARAM_INVALID);
EXPECT_EQ(CallGetWorkspaceSizeWithCommMode("ccu"), ACLNN_ERR_PARAM_INVALID);
EXPECT_EQ(CallGetWorkspaceSizeWithCommMode(""), ACLNN_ERR_PARAM_INVALID);
EXPECT_EQ(CallGetWorkspaceSizeWithDtype(ACL_BF16), ACLNN_SUCCESS);
EXPECT_EQ(CallGetWorkspaceSizeWithDtype(ACL_FLOAT16), ACLNN_ERR_PARAM_INVALID);
```

Keep the existing null-counts-tensor and optional-output tests.

- [ ] **Step 2: Run the API UT and observe failure**

Run:

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
conda activate szg_py311
bash build.sh --opapi_test --ops=allto_allv_grouped_mat_mul --soc=ascend910_93 -j8
```

Expected: the new `aiv` acceptance test fails because current V2 accepts `ai_cpu` instead.

- [ ] **Step 3: Implement AIV-only mode selection**

Use this behavior:

```cpp
static aclnnStatus CheckAndHandleCommMode(const char *commModeStr)
{
    if (commModeStr == nullptr || std::strcmp(commModeStr, "aiv") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "AlltoAllvGroupedMatMulV2 only supports commMode 'aiv'.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_2201) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "AlltoAllvGroupedMatMulV2 aiv mode only supports Ascend910_93.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
```

In the second-stage API, always select:

```cpp
NnopbaseSetHcclServerType(
    executor,
    NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_MTE);
```

Do not edit the V1 wrapper in `aclnn_allto_allv_grouped_mat_mul.cpp`.
Keep the shared operator-definition default `comm_mode="ai_cpu"` so direct V1 construction retains its old behavior; V2 always passes the validated `"aiv"` attribute explicitly.

- [ ] **Step 4: Run API UT**

Run the command from Step 2.

Expected: V2 AIV mode and BF16 tests pass; existing V1 API tests remain unchanged.

- [ ] **Step 5: Commit**

```bash
git add mc2/allto_allv_grouped_mat_mul/op_api/aclnn_allto_allv_grouped_mat_mul_v2.cpp \
        mc2/allto_allv_grouped_mat_mul/tests/ut/op_api/test_aclnn_allto_allv_grouped_mat_mul_v2.cpp
git commit -m "feat: make alltoallv grouped matmul v2 AIV-only"
```

### Task 3: Define the AIV tiling contract

**Required skills:** `superpowers:test-driven-development`, `ascendc-tiling-design`, `ascendc-operator-code-gen`

**Files:**
- Modify: `mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_tiling_key.h`
- Modify: `mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_tiling.h`
- Modify: `mc2/allto_allv_grouped_mat_mul/tests/ut/op_host/arch22/test_allto_allv_grouped_mat_mul_tiling.cpp`
- Modify: `mc2/allto_allv_grouped_mat_mul/tests/ut/op_kernel/allto_allv_grouped_mat_mul_tiling_def.h`

**Interfaces:**
- Consumes: host counts and shape attributes.
- Produces: `AlltoAllvGmmAivTilingData`, `AlltoAllvGmmInfo`, `CoCTiling`, workspace offsets, and four transpose tiling keys.

- [ ] **Step 1: Add failing tiling tests**

Add assertions for BF16/AIV:

```cpp
EXPECT_EQ(tiling.gmmInfo.rankSize, 4U);
EXPECT_EQ(tiling.gmmInfo.expertPerRank, 2U);
EXPECT_EQ(tiling.gmmInfo.K, 256U);
EXPECT_EQ(tiling.gmmInfo.N, 128U);
EXPECT_EQ(tiling.gmmCocTiling.ubMoveNum % 16, 0);
EXPECT_EQ(context.GetScheduleMode(), 1);
EXPECT_GT(context.GetBlockDim(), 0U);
EXPECT_EQ(context.GetTilingKey(), ExpectedAivKey(false, false));
```

Add failure assertions for FP16, `sendCounts.size()!=R*E`, negative counts, and `sum(sendCounts)!=M`.

- [ ] **Step 2: Run arch22 host UT and observe failure**

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
conda activate szg_py311
bash build.sh --ophost_test --ops=allto_allv_grouped_mat_mul --soc=ascend910_93 -j8
```

Expected: AIV tiling fields or key are absent.

- [ ] **Step 3: Add packed POD structures**

Define:

```cpp
#pragma pack(push, 8)
struct AlltoAllvGmmInfo {
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t rankSize;
    uint32_t expertPerRank;
    uint32_t maxOutputSize;
    uint32_t isTransposeB;
    uint32_t hasSharedExpert;
    uint32_t hasPermuteOut;
    uint32_t aivCoreNum;
    uint32_t aicCoreNum;
};

struct CoCTiling {
    int32_t m0;
    int32_t k0;
    int32_t n0;
    int32_t ubMoveNum;
    int32_t swizzlCount;
    int32_t swizzlDirect;
};

struct AlltoAllvGmmAivTilingData {
    Mc2InitTiling hcclInitTiling;
    Mc2CcTiling hcclCcTiling;
    AlltoAllvGmmInfo gmmInfo;
    AlltoAllvGmmInfo mmInfo;
    CoCTiling gmmCocTiling;
    CoCTiling mmCocTiling;
    int64_t sendCounts[MAX_EXPERT_NUM];
    int64_t recvCounts[MAX_EXPERT_NUM];
    uint64_t recvTokenOffset;
    uint64_t countTableOffset;
    uint64_t expertMetaOffset;
    uint64_t expertReadyOffset;
};
#pragma pack(pop)
```

Use the exact HCCL tiling types `Mc2InitTiling` and `Mc2CcTiling`, matching `AllGatherMatmulAIVModeTilingData` in the arch22 all-gather reference.

- [ ] **Step 4: Add fixed AIV key selection**

Retain the two transpose booleans and compile `TILINGKEY_TPL_AIV` for FF, FT, TF, and TT. Keep any legacy key entry required solely by V1 in a separately named legacy dispatch block.

- [ ] **Step 5: Run host UT**

Run the command from Step 2.

Expected: AIV key/data tests pass and existing arch22 V1 tiling tests pass.

- [ ] **Step 6: Commit**

```bash
git add mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_tiling_key.h \
        mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_tiling.h \
        mc2/allto_allv_grouped_mat_mul/tests/ut/op_host/arch22/test_allto_allv_grouped_mat_mul_tiling.cpp \
        mc2/allto_allv_grouped_mat_mul/tests/ut/op_kernel/allto_allv_grouped_mat_mul_tiling_def.h
git commit -m "feat: define AIV tiling for alltoallv grouped matmul v2"
```

### Task 4: Implement AIV host tiling and workspace layout

**Required skills:** `superpowers:test-driven-development`, `ascendc-operator-code-gen`

**Files:**
- Create: `mc2/allto_allv_grouped_mat_mul/op_host/op_tiling/arch22/allto_allv_grouped_mat_mul_aiv_tiling.h`
- Create: `mc2/allto_allv_grouped_mat_mul/op_host/op_tiling/arch22/allto_allv_grouped_mat_mul_aiv_tiling.cpp`
- Modify: `mc2/allto_allv_grouped_mat_mul/op_host/op_tiling/allto_allv_grouped_mat_mul_tiling.h`
- Modify: `mc2/allto_allv_grouped_mat_mul/op_host/op_tiling/allto_allv_grouped_mat_mul_tiling.cpp`
- Modify: `mc2/allto_allv_grouped_mat_mul/op_host/op_tiling/arch22/CMakeLists.txt`
- Test: `mc2/allto_allv_grouped_mat_mul/tests/ut/op_host/arch22/test_allto_allv_grouped_mat_mul_tiling.cpp`

**Interfaces:**
- Consumes: `AlltoAllvGmmAivTilingData` from Task 3.
- Produces: `AlltoAllvGmmAivTiling::DoTiling()`, mixed block dim, batch scheduling, HCCL resource data, and deterministic aligned workspace offsets.

- [ ] **Step 1: Extend failing workspace tests**

Assert these formulas with `AlignUp512`:

```cpp
recvBytes = AlignUp512(maxOutputSize * K * sizeof(bfloat16_t));
countBytes = AlignUp512(rankSize * expertPerRank * sizeof(uint64_t));
metaBytes = AlignUp512(expertPerRank * sizeof(ExpertMeta));
readyBytes = AlignUp512(expertPerRank * sizeof(uint32_t));
expected = systemWorkspace + recvBytes + countBytes + metaBytes + readyBytes + catlassAuxBytes;
EXPECT_EQ(context.GetWorkspaceSize(0), expected);
```

- [ ] **Step 2: Run host UT and observe workspace mismatch**

Run the Task 3 host-UT command.

Expected: legacy workspace layout does not match.

- [ ] **Step 3: Implement validation and dimension extraction**

Implement one path that rejects non-BF16/non-ND, validates `R*E`, counts length/sum, optional shared tensors, transpose-dependent K/N, and `permuteOutFlag` consistency before filling tiling data.

- [ ] **Step 4: Implement CATLASS tile selection**

Use the BF16 candidate set:

```cpp
constexpr CoCTiling kCandidates[] = {
    {128, 64, 128, 0, 1, 0},
    {128, 32, 128, 0, 1, 0},
    {64,  64, 128, 0, 1, 0},
    {64,  32, 64,  0, 1, 0},
};
```

Choose the first candidate satisfying L1/L0 constraints reported by the platform; compute `ubMoveNum` separately as a 16-element multiple within available UB.

- [ ] **Step 5: Initialize communication resources**

Set schedule mode 1, mixed block dim, MTE/full-mesh communication resource tiling, and both initialization/context tiling payloads by following the arch22 AIV path in `all_gather_matmul_v2`.

- [ ] **Step 6: Run host UT**

Expected: all AIV and legacy V1 host tests pass.

- [ ] **Step 7: Commit**

```bash
git add mc2/allto_allv_grouped_mat_mul/op_host/op_tiling \
        mc2/allto_allv_grouped_mat_mul/tests/ut/op_host/arch22/test_allto_allv_grouped_mat_mul_tiling.cpp
git commit -m "feat: add arch22 AIV tiling for alltoallv grouped matmul"
```

### Task 5: Implement peer-window communication helpers

**Required skills:** `superpowers:test-driven-development`, `ascendc-api-best-practices`, `ascendc-operator-code-gen`

**Files:**
- Create: `mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_aiv_comm.h`
- Modify: `mc2/allto_allv_grouped_mat_mul/tests/ut/op_kernel/test_allto_allv_grouped_mat_mul.cpp`

**Interfaces:**
- Consumes: HCCL context and AIV tiling data.
- Produces: `A2avWindowLayout`, `ExpertMeta`, `BuildExpertMeta()`, `PublishLocalInput()`, `GatherExpert()`, `CrossRankBarrier()`, and expert-ready states.

- [ ] **Step 1: Add kernel-unit cases for offsets**

For rank size 2, E=2, and peer send counts:

```text
rank0: [2, 1, 3, 0]
rank1: [1, 2, 0, 4]
```

Assert on local rank 0:

```text
expert0 M = 3, source counts = [2,1], source-major destination offsets = [0,2]
expert1 M = 3, source counts = [1,2], source-major destination offsets = [3,4]
```

Add a second case where one expert receives zero from every source.

- [ ] **Step 2: Run kernel UT and observe missing helper failure**

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
conda activate szg_py311
bash build.sh --opkernel_test --ops=allto_allv_grouped_mat_mul --soc=ascend910_93 -j8
```

Expected: communication helper symbols or AIV tiling fields are missing.

- [ ] **Step 3: Implement window and metadata types**

Define:

```cpp
struct ExpertMeta {
    uint64_t recvTokenBase;
    uint32_t tokenCount;
    uint32_t reserved;
};

struct A2avWindowLayout {
    uint64_t inputBytes;
    uint64_t countsOffset;
    uint64_t controlOffset;
};
```

Use 64-bit byte offsets and explicit overflow/range checks in host-tested helpers.

- [ ] **Step 4: Implement ping-pong movement**

Expose:

```cpp
template <typename T>
__aicore__ inline void CopyPeerGmToLocalGm(
    GlobalTensor<T> dst,
    GlobalTensor<T> src,
    uint64_t elementCount,
    uint32_t ubMoveNum,
    TQue<QuePosition::VECIN, 2> &queue);
```

Loop over aligned full tiles, then issue one bounded tail copy that writes only `elementCount % ubMoveNum` valid elements.

- [ ] **Step 5: Implement synchronization**

Use the 910C peer-window flag protocol from the reference operators with separate publish and final-release epochs. Store AIV→AIC ready states in user workspace indexed by `expertIdx`.

- [ ] **Step 6: Run kernel UT**

Expected: offset, zero-count, tail-move, and state-index tests pass.

- [ ] **Step 7: Commit**

```bash
git add mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_aiv_comm.h \
        mc2/allto_allv_grouped_mat_mul/tests/ut/op_kernel/test_allto_allv_grouped_mat_mul.cpp
git commit -m "feat: add manual AIV alltoallv communication helpers"
```

### Task 6: Add CATLASS expert and shared-MM wrappers

**Required skills:** `superpowers:test-driven-development`, `ascendc-operator-code-gen`

**Files:**
- Create: `mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_catlass.h`
- Modify: `mc2/allto_allv_grouped_mat_mul/tests/ut/op_kernel/test_allto_allv_grouped_mat_mul.cpp`

**Interfaces:**
- Consumes: `CoCTiling`, `ExpertMeta`, BF16 A/B/C pointers, and compile-time transpose flags.
- Produces: `RunExpertGemm<TRANS_B>()` and `RunSharedExpertGemm<TRANS_B>()`.

- [ ] **Step 1: Add failing CATLASS shape/dispatch cases**

Cover `M=0`, `M=1`, non-multiple M=17, `K=272`, `N=130`, and both B transpose layouts. Assert zero-M returns without writes and nonzero cases write the expected output region.

- [ ] **Step 2: Run kernel UT and observe missing CATLASS wrapper failure**

Run the Task 5 kernel-UT command.

- [ ] **Step 3: Implement the main wrapper**

Instantiate `Arch::AtlasA2`, BF16 A/B/C, FP32 accumulator, explicit `GemmCoord{M,N,K}`, and a CATLASS block scheduler. Compute offsets as:

```cpp
const uint64_t offsetA = expert.recvTokenBase * K;
const uint64_t offsetB = TRANS_B
    ? static_cast<uint64_t>(expertIdx) * N * K
    : static_cast<uint64_t>(expertIdx) * K * N;
const uint64_t offsetC = expert.recvTokenBase * N;
```

Pass actual tail shapes to CATLASS; do not pad-write C.

- [ ] **Step 4: Implement the shared-MM wrapper**

Use the same BF16/FP32-accumulate policy with its own `M/K/N` and `transMmWeight` template argument. Skip immediately when `hasSharedExpert==0`.

- [ ] **Step 5: Run kernel UT**

Expected: CATLASS wrapper cases pass for the four transpose-key combinations.

- [ ] **Step 6: Commit**

```bash
git add mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_catlass.h \
        mc2/allto_allv_grouped_mat_mul/tests/ut/op_kernel/test_allto_allv_grouped_mat_mul.cpp
git commit -m "feat: add CATLASS compute for alltoallv grouped matmul v2"
```

### Task 7: Integrate the expert-at-a-time mixed kernel

**Required skills:** `superpowers:test-driven-development`, `ascendc-operator-code-gen`

**Files:**
- Create: `mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_aiv_mode.h`
- Modify: `mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul.cpp`
- Modify: `mc2/allto_allv_grouped_mat_mul/CMakeLists.txt`
- Modify: `mc2/allto_allv_grouped_mat_mul/tests/ut/op_kernel/test_allto_allv_grouped_mat_mul.cpp`

**Interfaces:**
- Consumes: communication helpers from Task 5 and CATLASS wrappers from Task 6.
- Produces: `AlltoAllvGroupedMatMulAivMode<GMM_TRANS,MM_TRANS>::Process()` and AIV-key dispatch.

- [ ] **Step 1: Add failing pipeline cases**

Add kernel cases covering expert sequence `[3,0,5,1]`, `permuteOut=true`, and optional shared MM. Verify the main loop advances one `expertIdx` at a time and never reads `expertNum`.

- [ ] **Step 2: Run kernel UT and observe missing AIV mode failure**

Run the Task 5 kernel-UT command.

- [ ] **Step 3: Implement AIV flow**

Use this sequence:

```cpp
InitHcclContext();
InitWorkspaceAndQueues();
ClearLocalStates();
PublishLocalInputAndCounts();
CrossRankPublishBarrier();
BuildExpertMetadata();

for (uint32_t e = 0; e < expertPerRank; ++e) {
    GatherExpert(e);
    SyncAivCores();
    PublishExpertReady(e);
}

CrossRankFinalRelease();
```

- [ ] **Step 4: Implement AIC flow**

Use this sequence:

```cpp
if (hasSharedExpert) {
    RunSharedExpertGemm<MM_TRANS>();
}
for (uint32_t e = 0; e < expertPerRank; ++e) {
    WaitExpertReady(e);
    if (expertMeta[e].tokenCount != 0) {
        RunExpertGemm<GMM_TRANS>(e, expertMeta[e]);
    }
}
```

- [ ] **Step 5: Dispatch only the new V2 AIV key**

In the entry kernel, instantiate the AIV mode for FF, FT, TF, and TT. Keep the old class instantiation only under the legacy V1 key path and do not share the V2 scheduler object with it.

- [ ] **Step 6: Run all three UT groups**

```bash
bash build.sh --opapi_test --ophost_test --opkernel_test \
  --ops=allto_allv_grouped_mat_mul --soc=ascend910_93 -j8
```

Expected: API, host, and kernel tests pass.

- [ ] **Step 7: Commit**

```bash
git add mc2/allto_allv_grouped_mat_mul/op_kernel \
        mc2/allto_allv_grouped_mat_mul/CMakeLists.txt \
        mc2/allto_allv_grouped_mat_mul/tests/ut/op_kernel
git commit -m "feat: integrate AIV CATLASS alltoallv grouped matmul v2"
```

### Task 8: Build and install the operator package

**Required skills:** `ascendc-operator-compile-debug`, `superpowers:systematic-debugging`

**Files:**
- Modify only files required by concrete compiler diagnostics from Tasks 2-7.

**Interfaces:**
- Consumes: complete V2 AIV host/kernel implementation.
- Produces: installable Ascend 910C operator package.

- [ ] **Step 1: Build the kernel package**

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
conda activate szg_py311
bash build.sh --pkg --soc=ascend910_93 --ops=allto_allv_grouped_mat_mul -j8
```

Expected: exit code 0 and a run package under `build_out/` or `output/`.

- [ ] **Step 2: Inspect the compiled path**

```bash
grep -R "AlltoAllV<" mc2/allto_allv_grouped_mat_mul/op_kernel
grep -R "QuantGroupedMatmul\|GroupedMatMul<\|Matmul<" \
  mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_aiv_mode.h \
  mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_catlass.h
```

Expected: the new AIV files contain no high-level AlltoAllV/GroupedMatMul/MatMul calls; legacy files may still contain them for V1.

- [ ] **Step 3: Install the generated package**

Run the generated `.run` package with its documented install flag, then source its `set_env.sh`. Record the installed package path and version in the verification log.

- [ ] **Step 4: Re-run UT against the installed artifacts**

Run the Task 7 combined UT command.

- [ ] **Step 5: Commit compiler-driven fixes**

```bash
git add mc2/allto_allv_grouped_mat_mul
git commit -m "fix: compile AIV alltoallv grouped matmul v2 on 910C"
```

Create this commit only when tracked source changes exist.

### Task 9: Add the mc2_test V2 harness

**Required skills:** `superpowers:test-driven-development`, `ascendc-operator-code-gen`, `spreadsheets:Spreadsheets`

**Files:**
- Create: `/home/sangzhenguo/code/mc2_test/cpp/obp/aclnnAlltoAllvGroupedMatMulV2.cc`
- Modify: `/home/sangzhenguo/code/mc2_test/cpp/Makefile`
- Create: `/home/sangzhenguo/code/mc2_test/op_class/aclnnAlltoAllvGroupedMatMulV2.py`
- Modify: `/home/sangzhenguo/code/mc2_test/op_class/__init__.py`
- Create: `/home/sangzhenguo/code/mc2_test/excel/aclnnAlltoAllvGroupedMatMulV2.xlsx`

**Interfaces:**
- Consumes: V2 ACLNN API and Task 1 case contract.
- Produces: `aclnnAlltoAllvGroupedMatMulV2_aarch`, registered class name `aclnnAlltoAllvGroupedMatMulV2`, BF16 golden output, and EP=2/4/8 cases.

- [ ] **Step 1: Add the C++ executable source**

Copy the existing V1 harness structure, include `aclnnop/aclnn_allto_allv_grouped_mat_mul_v2.h`, and call:

```cpp
aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize(
    gmmX, gmmWeight, nullptr, nullptr, mmX, mmWeight,
    group.c_str(), "aiv", epWorldSize,
    sendCounts, recvCounts,
    transGmmWeight, transMmWeight, permuteOutFlag,
    gmmY, mmY, permuteOut, &workspaceSize, &executor);

aclnnAlltoAllvGroupedMatMulV2(
    workspace, workspaceSize, executor, stream);
```

- [ ] **Step 2: Add a dedicated Make target**

```make
aclnn_allto_allv_grouped_mat_mul_v2:
	@mkdir -p bin
	g++ $(CXXFLAGS) ./obp/aclnnAlltoAllvGroupedMatMulV2.cc $(SRC_COMMON) \
		$(INCLUDEDIRS) -o ./bin/aclnnAlltoAllvGroupedMatMulV2_$(ARCH) $(CUSTOM_LIBS)
```

- [ ] **Step 3: Add Python registration**

Register `@class_registry("aclnnAlltoAllvGroupedMatMulV2")`, set:

```python
self.exe_bin_name = "aclnnAlltoAllvGroupedMatMulV2"
self.comm_mode = "aiv"
```

Use the existing golden order: AlltoAllV, local-expert/source-rank permute, per-expert `torch.matmul`, optional shared `torch.matmul`.

- [ ] **Step 4: Generate the Excel cases from Task 1**

Include the eight valid cases and invalid cases that can be executed safely by a single launch. All tensor dtypes must be `bfloat16`; counts remain integer arrays.

- [ ] **Step 5: Build the harness**

```bash
cd /home/sangzhenguo/code/mc2_test/cpp
make aclnn_allto_allv_grouped_mat_mul_v2
```

Expected: `bin/aclnnAlltoAllvGroupedMatMulV2_aarch` exists.

- [ ] **Step 6: Run CPU golden generation**

```bash
cd /home/sangzhenguo/code/mc2_test
conda activate szg_py311
python mc2_test.py --xlsx excel/aclnnAlltoAllvGroupedMatMulV2.xlsx \
  --sheet level0 --num 1 --scene cpu
```

Expected: CPU golden binaries are created for every selected rank.

- [ ] **Step 7: Commit mc2_test changes in its repository**

```bash
cd /home/sangzhenguo/code/mc2_test
git add cpp/obp/aclnnAlltoAllvGroupedMatMulV2.cc cpp/Makefile \
        op_class/aclnnAlltoAllvGroupedMatMulV2.py op_class/__init__.py \
        excel/aclnnAlltoAllvGroupedMatMulV2.xlsx
git commit -m "test: add alltoallv grouped matmul v2 AIV harness"
```

### Task 10: Run multi-rank correctness and deadlock tests

**Required skills:** `ascendc-operator-compile-debug`, `ascendc-operator-precision-debug`, `superpowers:systematic-debugging`

**Files:**
- Modify only source or test files tied to a reproduced failure.
- Create verification logs under the existing test output convention; do not commit raw device logs.

**Interfaces:**
- Consumes: installed operator and V2 mc2_test harness.
- Produces: passing EP=2/4/8 BF16 results for `gmmY`, `permuteOut`, and `mmY`.

- [ ] **Step 1: Run EP=2 basic and zero-expert cases**

```bash
cd /home/sangzhenguo/code/mc2_test
timeout 300 python mc2_test.py --xlsx excel/aclnnAlltoAllvGroupedMatMulV2.xlsx \
  --sheet level0 --num 2 --scene npu
```

Expected: process exits 0, no rank timeout, all BF16 comparisons pass.

- [ ] **Step 2: Run EP=4 skew and zero-peer cases**

Select the EP=4 rows from the workbook with the test framework's row filter and run with a 300-second timeout.

Expected: every rank reaches final-release and outputs match golden.

- [ ] **Step 3: Run EP=8 tail-tile and single-expert cases**

Select the EP=8 rows and run with a 300-second timeout.

Expected: non-aligned K/N tails and heavily skewed counts pass.

- [ ] **Step 4: Run transpose and optional outputs**

Run all FF/FT/TF/TT combinations represented in the workbook, checking `gmmY`, `permuteOut`, and `mmY` independently.

- [ ] **Step 5: Re-run the complete UT suite**

```bash
cd /home/sangzhenguo/code/ops-transformer
bash build.sh --opapi_test --ophost_test --opkernel_test \
  --ops=allto_allv_grouped_mat_mul --soc=ascend910_93 -j8
```

Expected: all tests pass after device-driven fixes.

- [ ] **Step 6: Commit correctness fixes**

Commit each independently reproduced and verified fix with a message naming its cause, for example:

```bash
git commit -m "fix: handle zero-token expert in AIV pipeline"
```

### Task 11: Update public documentation and run final evaluation

**Required skills:** `ascendc-operator-doc-gen`, `ascendc-operator-precision-eval`, `ascendc-operator-performance-eval`, `superpowers:verification-before-completion`

**Files:**
- Modify: `mc2/allto_allv_grouped_mat_mul/docs/aclnnAlltoAllvGroupedMatMulV2.md`
- Create precision/performance reports in the locations required by the evaluation skills.

**Interfaces:**
- Consumes: verified implementation and Task 1 testcase contract.
- Produces: accurate AIV-only API documentation, at least 30 precision cases, profiler JSONL cases, and a Markdown performance comparison.

- [ ] **Step 1: Update the API document**

State exactly:

```text
Platform: Ascend 910C
CANN: 9.2+
commMode: only "aiv"
dtype: only bfloat16
sendCountsTensorOptional/recvCountsTensorOptional: must be null
communication: manual HCCL-window movement
compute: CATLASS
```

- [ ] **Step 2: Run at least 30 precision cases**

Use the precision-evaluation skill to expand Task 1 shapes across transpose, permute, shared-MM, skew, zero-peer, and zero-expert dimensions. Report MaxAbsErr, MeanAbsErr, MaxRelErr, MeanRelErr, and CosineSim.

- [ ] **Step 3: Run profiler comparison**

Use profiler cases with fixed `warmup=5` and `active=5`, compare the custom operator with the small-op composition and old V2, and aggregate `ASCEND_PROFILER_OUTPUT/op_statistic.csv` Total Time(us).

- [ ] **Step 4: Verify forbidden calls and expert granularity**

```bash
grep -R "AlltoAllV<" mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_aiv_*.h
grep -R "QuantGroupedMatmul\|GroupedMatMul<" mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_aiv_*.h
grep -R "expertNum\|chunk" mc2/allto_allv_grouped_mat_mul/op_kernel/allto_allv_grouped_mat_mul_aiv_*.h
```

Expected: no output, apart from comments explicitly describing unsupported strategies; remove those words from implementation comments if they obscure this check.

- [ ] **Step 5: Verify on CANN 9.2+**

Source the supplied 9.2+ `set_env.sh`, rebuild package and all UT, reinstall, and rerun the EP=2/4/8 representative cases. Do not mark the project complete if this environment is unavailable.

- [ ] **Step 6: Commit documentation and reports**

```bash
git add mc2/allto_allv_grouped_mat_mul/docs \
        mc2/allto_allv_grouped_mat_mul/tests
git commit -m "docs: document and evaluate AIV alltoallv grouped matmul v2"
```

## Final Verification Checklist

- [ ] V1 API/tiling/kernel tests pass unchanged.
- [ ] V2 rejects `ai_cpu` and `ccu`, accepts only `aiv`.
- [ ] V2 rejects non-BF16 inputs.
- [ ] V2 kernel contains no high-level AlltoAllV or GroupedMatMul/MatMul calls.
- [ ] V2 loops over `expertIdx++` and never batches experts.
- [ ] EP=2/4/8 valid cases pass without timeout.
- [ ] Zero-peer and zero-expert cases pass without deadlock.
- [ ] Transpose, `permuteOut`, and shared-MM combinations pass.
- [ ] CANN 9.2+ build and runtime verification passes.
- [ ] Precision report contains at least 30 cases and is shown in the task handoff.
- [ ] Performance report uses profiler data and is shown in the task handoff.
