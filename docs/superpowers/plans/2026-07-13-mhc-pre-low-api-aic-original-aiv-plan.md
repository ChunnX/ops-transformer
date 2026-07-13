# MhcPre Low-API AIC With Original AIV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the Arch35 N=4 low-level Cube and AIC-only HF32 while restoring the original Split-BS AIV preprocessing and postprocessing through an hMix GM handoff.

**Architecture:** `MhcPreKernelSplitBSLowApi` becomes a thin specialization of `MhcPreKernelSplitBS`. It reuses the original Init, V0, invRms, Gather, V1, and output functions, but owns a low-level Cube service and replaces only AIC matmul execution. The low-level service reads the existing FP32 workspace, accumulates across 256-wide K chunks, and Fixpipes the final row-major hMix to GM.

**Tech Stack:** Ascend C Arch35, C310 low-level Mmad/Fixpipe APIs, existing MhcPre host tiling and kernel tests, Ascend 950 A5 runtime.

## Global Constraints

- HF32 setup and reset execute only under `ASCEND_IS_AIC`.
- N=4 Split-BS uses the low-API key; N=6/8 and Split-ND remain unchanged.
- This phase intentionally restores the hMix GM handoff; direct L0C-to-AIV-UB is a later phase.
- Public ACLNN API, shapes, dtypes, optional outputs, and workspace semantics remain unchanged.

---

### Task 1: Extend the Cube Service for the Original Workspace Contract

**Files:**
- Modify: `mhc/mhc_pre/op_kernel/arch35/mhc_pre_cube_compute_arch35.h`

**Interfaces:**
- Consumes: FP32 A chunk in GM with shape `[m, k]`, phi chunk in GM with shape `[n, k]` viewed transposed, and existing L1 buffers.
- Produces: `CopyInA1Nd2Nz(...)` and `CopyOut(GlobalTensor<float>)` for row-major hMix GM output.

- [ ] **Step 1: Record the existing failing runtime test**

Run on A5:

```bash
source /home/matmul/l00947498/source_a5_env.zsh
source /home/matmul/l00947498/CANN/vendors/custom_transformer/bin/set_env.bash
cd /home/matmul/l00947498/testmhc
python test_hasGamma_lm_ori.py
```

Expected: all 40 N=4 HF32 cases report `FAIL`, proving the current full low-API AIV path does not meet the restored behavior.

- [ ] **Step 2: Add GM-to-A1 ND2NZ copy**

Add a method with this contract:

```cpp
__aicore__ inline void CopyInA1Nd2Nz(uint64_t m, uint64_t currentK,
    const AscendC::GlobalTensor<float>& aGlobal,
    const AscendC::LocalTensor<float>& al1Local);
```

Configure `Nd2NzParams` with `nValue=m`, `dValue=currentK`, `srcDValue=currentK`, and `dstNzC0Stride=LowApiAlign(m, BLOCK_CUBE)`. Pair MTE2/MTE1 events with the active A1 ping-pong buffer.

- [ ] **Step 3: Add final L0C-to-GM Fixpipe output**

Add:

```cpp
__aicore__ inline void CopyOut(const AscendC::GlobalTensor<float>& dstGlobal,
    uint32_t dstStride);
```

Use the Arch35 `FixpipeParamsC310`/L0C DataCopy form with `mSize=m_`, `nSize=n_`, `dstStride=dstStride`, `srcStride=LowApiAlign(baseM_, BLOCK_CUBE)`, `NoQuant`, row-major NZ-to-ND enabled, and final unit flag.

- [ ] **Step 4: Build to verify the service compiles**

Run:

```bash
bash build.sh --ops=mhc_pre --soc=ascend950 -j8 --pkg
```

Expected: exit code 0 with no new errors from `mhc_pre_cube_compute_arch35.h`.

### Task 2: Reuse the Original Split-BS AIV Pipeline

**Files:**
- Modify: `mhc/mhc_pre/op_kernel/arch35/mhc_pre_split_bs_low_api.h`
- Modify: `mhc/mhc_pre/op_kernel/mhc_pre_apt.cpp`

**Interfaces:**
- Consumes: `InitParams`, an uninitialized `MT&` required by the inherited class layout, and `MhcPreTilingData` using the low-API key.
- Produces: `MhcPreKernelSplitBSLowApi<T, P, HAS_RESI>` with original AIV behavior and low-level `AICProcessLowApi(offsetNd, outOffset)`.

- [ ] **Step 1: Replace the standalone full low-API kernel with a thin derived kernel**

Define:

```cpp
template <class T, class P, int8_t HAS_RESI>
class MhcPreKernelSplitBSLowApi : public MhcPreKernelSplitBS<T, P, HAS_RESI> {
public:
    using Parent = MhcPreKernelSplitBS<T, P, HAS_RESI>;
    __aicore__ inline explicit MhcPreKernelSplitBSLowApi(MT& matmul) : Parent(matmul) {}
    __aicore__ inline void Init(InitParams initParams);
    __aicore__ inline void Process();
private:
    __aicore__ inline void AICProcessLowApi(uint32_t offsetNd, uint32_t outOffset);
    MhcPreCubeCompute mmService_;
};
```

`Init` calls `Parent::Init(initParams)`, allocates A1 storage for A/B only on the low-API object, and calls `mmService_.Init()` only on AIC.

- [ ] **Step 2: Reproduce the original Process scheduling exactly**

Copy the parent Process control flow, including `SYNC_V2C`, `SYNC_C2V`, `SYNC_C2V1`, V0Prologue, V0PostProcess, AIV1Process, counters, and queue cleanup. Replace only:

```cpp
Parent::AICProcess(offsetNd, outOffset);
```

with:

```cpp
AICProcessLowApi(offsetNd, outOffset);
```

- [ ] **Step 3: Implement the low-level AIC chunk**

Use the same workspace offset as the original AIC path:

```cpp
uint64_t xOffset = chunkTSize_ * ND_LENGTH *
    (coreIdx_ + (cubeCount_ % PARALLEL_NUM) * coreNum_);
```

For each 256-wide chunk, copy A and B to L1, call Mmad with `isFirstKL1=(offsetNd==0)` and `isLastKL1=(offsetNd+curK>=matrixInfo_.nD)`, and only on the last chunk Fixpipe to `hMixGm_[outOffset]`. Increment `cubeCount_` once per chunk and call `End()` after the final output.

- [ ] **Step 4: Update the low-API entry construction**

In `mhc_pre_apt.cpp`, construct `MT mm;` without calling `mm.Init(...)`, then pass it to both low-API `HAS_RESI` specializations:

```cpp
MT mm;
MhcPreKernelSplitBSLowApi<DTYPE_X, float32_t, HAS_RESI> op(mm);
```

The old Split-BS and Split-ND branches continue to initialize and use `MatmulImpl` unchanged.

- [ ] **Step 5: Build the complete package**

Run:

```bash
bash build.sh --ops=mhc_pre --soc=ascend950 -j8 --pkg
```

Expected: exit code 0 and low-API template keys compile for BF16/FP16 and both residual variants.

### Task 3: Verify Precision and Establish the GM-Backed Performance Baseline

**Files:**
- Test: `mhc/mhc_pre/tests/ut/op_host/test_mhc_pre_tiling.cpp`
- Runtime test: `/home/matmul/l00947498/testmhc/test_hasGamma_lm_ori.py`

**Interfaces:**
- Consumes: packaged custom operator installed under `/home/matmul/l00947498/CANN`.
- Produces: precision log and msprof directory for the GM-backed low-level AIC baseline.

- [ ] **Step 1: Run host tiling tests**

Run the repository's mhc_pre host UT target and confirm N=4 selects `MHC_PRE_LOW_API_BS`, while N=6/8 remain on old keys.

- [ ] **Step 2: Install the package**

```bash
./build_out/cann-ops-transformer-custom_linux-aarch64.run \
  --quiet --install-path=/home/matmul/l00947498/CANN
```

Expected: `SUCCESS`.

- [ ] **Step 3: Run precision 10 times per shape**

```bash
source /home/matmul/l00947498/CANN/vendors/custom_transformer/bin/set_env.bash
cd /home/matmul/l00947498/testmhc
python test_hasGamma_lm_ori.py 2>&1 | tee mhc_pre_low_api_aic_original_aiv_10x.log
```

Expected: 40 `Case result: PASS` entries and `Overall result: PASS`.

- [ ] **Step 4: Profile the first case after precision passes**

Temporarily retain only `(1,1024,4,2560)` in the runtime script and run:

```bash
msprof python test_hasGamma_lm_ori.py
```

Compare task duration and AIC/AIV pipeline times with `PROF_000001_20260713125443789_00582874MKLBEDNC`. Expect AIV scalar time near the original `18.944 us`, not the current `6011 us`.

- [ ] **Step 5: Commit the implementation**

```bash
git add mhc/mhc_pre/op_kernel/arch35/mhc_pre_cube_compute_arch35.h \
        mhc/mhc_pre/op_kernel/arch35/mhc_pre_split_bs_low_api.h \
        mhc/mhc_pre/op_kernel/mhc_pre_apt.cpp \
        mhc/mhc_pre/tests/ut/op_host/test_mhc_pre_tiling.cpp
git commit -m "perf(mhc_pre): restore vector pipeline around low-api cube"
```
