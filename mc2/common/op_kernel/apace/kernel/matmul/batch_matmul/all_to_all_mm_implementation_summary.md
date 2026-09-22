# 非量化 All-to-All MatMul 实现总结

> 本文档总结 2026-09 期间非量化 all-to-all MatMul 算子的实现过程与**最新代码设计**。
> 覆盖：blaze 库改动、kernel、tiling 数据、通算融合接口、ST 测试、Kernel UT 测试。
> 旧的 GM 累加兜底方案（zero-bias workspace、atomic 逐块切换、`PipeBarrier`）已被 splitKIdx 方案取代，不再展开。

---

## 1. 背景与目标

基于量化版 all-to-all QuantMatMul（mxfp8）移植**非量化版**，由外部通算融合接口调度三种模式：

| 模式 | localMatmul | matmulMode | 说明 |
|---|---|---|---|
| 1 | 0 | REMOTE | 单次 launch，遍历所有 rank 一次算完 |
| 2 | 1 | DEFERRED_SYNC | 每通信 tile：先 local 驻留 L0C → wait → remote 累加 → 单次 fixpipe |
| 3 | 2 | 先 LOCAL 后 REMOTE | LOCAL 写 GM → REMOTE 跳过 self 原子累加 |

**约束与关键前提**：
- ops-tensor（blaze）代码**最初不可改**，只能 kernel 侧绕行；**后期放开可改** → 采用 splitKIdx 方案。
- 必须使用**非量化调度器** `BlockSchedulerMatmulBasic`。
- `TransB=true`（B 按 `[N,K]` DN 列主序存储）。

---

## 2. 最终设计（splitKIdx + 非量化调度器）

**核心机制**：给非量化 mmad `block_mmad_matmul_basic.h` 增加量化版同款 `splitKIdx` 接口，
同一输出块的 `rankSize` 次 mmad 调用在 **L0C 上跨调用累加**，最后一个 split 触发**单次 fixpipe 搬出**。

### 2.1 三种模式的执行语义

```
模式1 (REMOTE, localMatmul=0):  [wait] → 遍历 rank0..rankSize-1, splitKIdx=0..rankSize-1
    rank0: 清零 L0C + bias；中间: enable_keep 累加；最后 rank: enable_update + 单次 fixpipe 写 GM（无 atomic）

模式2 (DEFERRED_SYNC):  self(splitKIdx=0, 清零+bias, 驻留L0C) → [wait] → 其余 rank(splitKIdx=1..)
    → 最后 rank 单次 fixpipe（无 atomic）

模式3 (localMatmul=2):
    LOCAL launch:  splitKIdx=0, splitKNum=1 → 独立 matmul，覆盖写 GM（带 bias）
    REMOTE launch: 跳过 self, splitKIdx=0..rankSize-2 → L0C 累加 + 单次 atomic fixpipe 累加到 GM
```

### 2.2 各模式参数

| 模式 | splitKNum | 每块 GM 写次数 | atomic | bias 加几次 |
|---|---|---|---|---|
| 模式1 | rankSize | 1 | 无 | 1（splitKIdx==0）|
| 模式2 | rankSize | 1 | 无 | 1（self）|
| 模式3-LOCAL | 1 | 1 | 无 | 1 |
| 模式3-REMOTE | rankSize-1 | 1（atomic）| Run 级 `SetAtomicAdd` | 0（mmad isBias_=false）|

---

## 3. 文件清单与内容（最新版）

### 3.1 blaze 库改动（ops-tensor）

**文件**：`ops-tensor/include/blaze/gemm/block/block_mmad_matmul_basic.h`

给非量化基本 mmad 加 splitK 语义（5 处，向后兼容）：

1. `Params` 新增 `uint64_t splitKNum{1}`（split-K 分段数，1 = 独立 matmul）
2. `Init` 读入 `splitKNum_ = params.splitKNum`
3. `operator()` 新增末参 `uint64_t splitKIdx = 0`，门控：
   - `isFirstSplitK = (splitKIdx == 0)`：只在此清零 L0C（`initCmatrix`）、加 bias
   - `isLastSplitK = (splitKIdx == splitKNum_-1)`：只在此 `unitFlag=enable_update`、执行 `CopyL0CToGM`
4. `NeedProcessBias` 加 `isFirstSplitK`：bias 只在首个 split 加一次
5. `CopyL1FromGM` 加 `isFirstSplitK`：bias 拷贝也只发生在首个 split

> 向后兼容：`splitKNum=1, splitKIdx=0` 时行为与原来完全一致，其它调用方不受影响。
> L0C ping-pong 只在 `isLastSplitK` 后翻转 → 同块多 split 用同一 L0C 半区累加。

### 3.2 非量化 kernel（本目录 `all_to_all_mm_kernel.h`）

```cpp
template <class ProblemShape, class BlockMmad, class BlockScheduler, class CommPolicy>
class AllToAllMmKernel;
```

**要点**：
- 模板化 `BlockMmad`/`BlockScheduler`/`CommPolicy`，调度器用**非量化 `BlockSchedulerMatmulBasic`**。
- `Params`：`{problemShape, mmadParams(含 splitKNum), schParams, mmTiling, localParams}`。
- `LocalParams`：`{rankId, rankSize, originalM, localAGmAddr, localMatmul, matmulMode, headTileSize}`（**无 zero-bias**）。
- `Run()`：按 `isAtomicAdd_`（仅模式3-REMOTE）在 launch 级开/关 `SetAtomicAdd`；模式3-REMOTE 时把 `mmadParams.biasGmAddr` 置空跳过 bias。
- `ProcessSingleBatch` 三分支（LOCAL / DEFERRED_SYNC / REMOTE），把 `splitKIdx`（即 `remoteRankCnt`）传给 `mmadOp_`。
- bias 对**所有**调用传真 bias，mmad 按 `isFirstSplitK` 自动只加一次。
- 块迭代（非量化调度器）：
  ```cpp
  for (int64_t blockIdx = GetBlockIdx(); blockIdx < bs.GetBlockNums(); blockIdx += GetBlockNum()) {
      auto singleShape = bs.template GetBlockShape<transB, BType>(blockIdx); // {m,n,k_,batch}
      if (M<=0 || N<=0) continue;               // 空子片跳过
      auto blockCoord = bs.GetBlockCoord(blockIdx);  // 必须在 GetBlockShape 之后
      mPos = Get<MNK_M>(blockCoord); nPos = Get<MNK_N>(blockCoord);
      ...
  }
  ```
- 开头 `if (curBlockIdx >= bs.GetCoreNums()) return;` 保护。
- 无 `needUpdateTail_`/`UpdateTailTile`（尾块由调度器 Params 静态切分）。

### 3.3 tiling 数据（本目录 `all_to_all_mm_tiling_data.h`）

```cpp
struct allToAllMmMatmulTilingData {
    CommTilingData commTilingData;            // A 通信切分
    QuantMatmulTilingData tileMmTilingData;   // matmul 分块（scaleKL1 字段不使用）
    uint32_t localMatmul{0};
};
struct CommContext {
    Apace::AivComm::CommUdmaContext udmaCtx;
    Apace::AivComm::CommUbmemContext ubmemCtx;
};
```
- 对照量化版 `allToAllMatmulTilingData`：删 `scaleCommTilingData`、删 `hcommAllToAllMatmulTilingData`。
- 不 include `kernel_tiling/kernel_tiling.h`（避免 tikcfw/ops-tensor 头冲突）。
- `CommContext` host/kernel 共享定义。

### 3.4 通算融合接口（本目录 `all_to_all_mm_urma_impl.h`）

```cpp
template <typename AType, typename BType, typename CType, bool TransA, bool TransB>
class AllToAllMmUrmaImpl;
```

- `Init(hcommCtx, aGM, bGM, cGM, biasGM, tilingData)`（**无 scale、无 zero-bias**）。
- `Run()`：AIV → `RunAllToAll()`（仅 A 通信）；AIC → localMatmul==2 先 `RunLocalMatmul()` 再 `RunMatmul()`。
- `SetupParams` 按模式设置 `mmadParams.splitKNum`：LOCAL=1 / 模式3-REMOTE=rankSize-1 / 模式1·2=rankSize。
- `BlockScheduler = BlockSchedulerMatmulBasic<ProblemShape>`。
- `schedulerParams` 12 字段 + 5 默认：
  `{baseM, baseN, stepK*baseK, baseM, baseN, baseK, mTailTile, nTailTile, mBaseTailSplitCnt, nBaseTailSplitCnt, mTailMain, nTailMain}`。
- `RunMatmul`：`aGmAddr = selfWinAddr`（通信 buffer）、`localAGmAddr = aGm`。

### 3.5 ST 测试（`tests/st/all_to_all_matmul/`）

| 文件 | 说明 |
|---|---|
| `cases.csv` | 8 用例，6 列（m,k,n,rank_num,head_m_size,**localmatmul**），覆盖三模式 |
| `CMakeLists.txt` | target `apace_a2a_mm_st`，dav-3510，含 `asc/impl/...` include（9.3.0 需要）|
| `run.sh` | 生成数据→编译→多 rank 运行→精度比对；CSV/CLI 模式 |
| `src/apace_a2a_mm_kernel_launcher.h` | kernel entry：`KERNEL_TYPE_MIX_AIC_1_1`，`AllToAllMmUrmaImpl<half,half,half,false,true>` |
| `src/main.cpp` | host 主流程：无 scale、无 zero-bias、`localMatmul` CLI 参数、headMSize 用 CLI 值 |
| `scripts/gen_data.py` | fp16 无 scale，all-to-all 重排 + bias，golden 用 fp32 累加（已实测与 fp64 参考一致）|
| `scripts/verify_result.py` | fp16，rtol/atol=1e-2 |

### 3.6 Kernel UT（`batch_matmul/test/test_all_to_all_mm_kernel.cpp`）

- ICPU 模拟器跑 kernel（`KERNEL_RUN_KF`），预填通信 buffer + no-op CommPolicy。
- entry 直接用 `BlockSchedulerMatmulBasic` + 按模式算 `splitKNum`。
- 6 个用例：模式1/2/3 × 双 rank、模式1 多核（4 核）、单 rank 退化。
- CPU 参考：`C[m][n] = bias[n] + Σ_r Σ_k A[r*oriM+m][k]·B[r*Ka+k][n]`。
- 附 `CMakeLists.txt`（接入 ops-tensor Kernel UT 构建）、`README.md`。

---

## 4. 设计演进（关键决策）

1. **早期约束**：ops-tensor 不可改 → kernel 侧用 GM 累加兜底（首次覆盖写 + 其余原子累加），
   引入 zero-bias workspace、逐块 `SetAtomicAdd/None`、`PipeBarrier<PIPE_ALL>`。
2. **发现 `PipeBarrier<PIPE_FIX>` 在 dav-3510 非法**（arch22 才用），改为 `PIPE_ALL`。
3. **放开 blaze 可改** → 直接给基本 mmad 加 `splitKIdx`，恢复 L0C 跨 rank 累加 + 单次 fixpipe：
   - 删 zero-bias workspace、删逐块 atomic 切换、删 `PipeBarrier`、删 GM 累加语义。
4. **必须用非量化调度器** `BlockSchedulerMatmulBasic`：
   - 按下标 for 循环遍历（`GetBlockNums`/`GetCoreNums`/`GetBlockCoord`），空子片 `continue`。
   - `mTailCnt/nTailCnt` 直接映射 tiling 的 `mTailTile/nTailTile`（语义一致，basic 有 CeilDiv 归一）。
   - K 维度调度器直接返回 `k_`，无需 kernel 覆盖。

---

## 5. 遗留事项 / 待确认

1. **非量化 tiling**：
   - 方案 A：沿用 `QuantMatmulTilingSwat<DT_FLOAT16,DT_FLOAT16>`（正确，但 L1 预留 scale 空间，非最优）。
   - 方案 B：可改 blaze 加 `MatmulTilingBasic`（不含 scale 的 L1 预算）。**未定**。
2. **工具链版本**：ST 构建需 ASCEND_HOME_PATH 与 bisheng 编译器同版本（曾有 9.3.0/9.1.0 错配），
   改版本需 `rm -rf tests/st/build` 清缓存。
3. **编译验证**：本环境 `include/tensor_api` 未初始化，无法编译；需在完整 CANN 环境验证
   （重点：`KERNEL_TYPE_MIX_AIC_1_1` + blaze mmad 的 AIV 路径、`DT_FLOAT16` tiling、真实多 rank 运行）。
4. **昨天遗留**（CHANGELOG_debug.md 记录的旧 GM 累加尝试）"mmad 可能没执行"问题，
   在 urma 路径上是否复现需实测。
5. **`CommContext` 在 3 个 tiling 头重复定义**（脆弱点，可选重构到公共 comm 头）。

---

## 6. 四模式 splitKIdx / splitKNum 对照速查

| 模式 | matmulMode | localMatmul | splitKNum | kernel 里 splitKIdx 序列 |
|---|---|---|---|---|
| 1 REMOTE | REMOTE(0) | 0 | rankSize | 0,1,...,rankSize-1 |
| 2 DEFERRED | DEFERRED_SYNC(2) | 1 | rankSize | self=0, remote=1..rankSize-1 |
| 3 LOCAL | LOCAL(1) | 2 | 1 | 0 |
| 3 REMOTE | REMOTE(0) | 2 | rankSize-1 | 0..rankSize-2（跳过 self）|
