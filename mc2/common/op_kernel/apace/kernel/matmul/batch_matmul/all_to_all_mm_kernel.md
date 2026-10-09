# all_to_all_mm_kernel 非量化实现方案

> 2026-09 更新：随 splitKIdx 方案（L0C 跨 rank 累加 + 单次 fixpipe）与非量化 tiling 迁移（方案 B）落地，
> 本文档已同步为**当前实现**。早期 GM 累加兜底方案（zero-bias workspace、atomic 逐块切换、`PipeBarrier`）
> 已被取代，历史决策见 `all_to_all_mm_implementation_summary.md` §4。

## 1. 背景与目标

基于量化版本 `all_to_all_qbmm_mx_kernel.h`（mxfp8 场景 quantMatMul）移植**非量化版** all-to-all MatMul kernel，用于外部通算融合接口的三种调度模式：

| 模式 | localMatmul | matmulMode | launch 次数 | 说明 |
|---|---|---|---|---|
| 1 | 0 | REMOTE | 1 | 单 launch，一次循环算完所有 rank |
| 2 | 1 | DEFERRED_SYNC | 1 | 每个通信 tile：先 local 驻留 L0C → wait → remote 累加 → 单次 fixpipe |
| 3 | 2 | 先 LOCAL 后 REMOTE | 2 | LOCAL 覆盖写 GM → REMOTE 跳过 self 原子累加 |

**关键前提**：
- ops-tensor（blaze）可改 → 给非量化 mmad `block_mmad_matmul_basic.h` 增加 `splitKIdx` 接口（向后兼容），
  恢复量化版同款 **L0C 跨 rank 累加 + 单次 fixpipe** 语义。
- 使用**非量化调度器** `BlockSchedulerMatmulBasic`（按下标遍历，尾块由 Params 静态切分）。
- `TransB=true`（B 按 `[N,K]` DN 列主序存储）。

## 2. 总体设计

### 2.1 组件选型

| 组件 | 量化版 | 非量化版 | 说明 |
|---|---|---|---|
| mmad | `BlockMmad<MatmulWithScaleMx<...>>`（原生 splitKIdx） | `BlockMmad<MatmulMultiBlockBasic<NONE_FULL_LOAD_MODE, 0, KernelMmadMultiBlockBasic, 0>>` + blaze 侧 splitKIdx 扩展 | 无 scale |
| scheduler | `BlockSchedulerQuantBatchMatmulV3<...>` | `BlockSchedulerMatmulBasic<ProblemShape>` | 非量化专用 |
| CommPolicy | 外部注入（`WaitTile`） | 同款 | 不变 |
| 通信 | A + ScaleA 双通道 | 仅 A | ScaleA 通道删除 |

### 2.2 数据流

```
C[M][N] = bias[N] + Σ_{r∈[0,rankSize)} A_r[M][K] · B_r[K][N]
```

- `A_r`：rank r 的 M 段（self 用 localAGmAddr，其余用通信 buffer `selfWinAddr`），M 轴整体 `rankSize * originalM`。
- `B_r`：B 的 K 段，K 轴整体 `rankSize * K`。
- 每个输出块由调度器分配唯一核计算，块内按 rank 遍历做 rankSize 次 mmad，**L0C 上跨调用累加**。

### 2.3 三种模式的执行流程（splitKIdx 语义）

```
模式1 (REMOTE, localMatmul=0):  [wait] → 遍历 rank0..rankSize-1, splitKIdx=0..rankSize-1
    splitKIdx==0: 清零 L0C + bias；中间: enable_keep 累加；末 rank: enable_update + 单次 fixpipe 写 GM（无 atomic）

模式2 (DEFERRED_SYNC, localMatmul=1):  self(splitKIdx=0, 清零+bias, 驻留L0C) → [wait] → 其余 rank(splitKIdx=1..)
    → 末 rank 单次 fixpipe（无 atomic）

模式3 (localMatmul=2):
    LOCAL launch:  splitKIdx=0, splitKNum=1 → 独立 matmul，覆盖写 GM（带 bias）
    REMOTE launch: 跳过 self, splitKIdx=0..rankSize-2 → L0C 累加 + 单次 atomic fixpipe 累加到 GM
```

### 2.4 各模式参数

| 模式 | splitKNum | 每块 GM 写次数 | atomic | bias 加几次 |
|---|---|---|---|---|
| 模式1 | rankSize | 1 | 无 | 1（splitKIdx==0）|
| 模式2 | rankSize | 1 | 无 | 1（self）|
| 模式3-LOCAL | 1 | 1 | 无 | 1 |
| 模式3-REMOTE | rankSize-1 | 1（atomic）| Run 级 `SetAtomicAdd` | 0（`biasGmAddr=nullptr`，mmad 指针判断）|

## 3. 文件与命名

- 文件：`all_to_all_mm_kernel.h`，位于 `mc2/common/op_kernel/apace/kernel/matmul/batch_matmul/`。
- 命名风格对齐量化版：
  - 类名：`AllToAllMmKernel`（对照 `AllToAllQbmmMxKernel`）
  - 模板参数宏：`MM_KERNEL_CLASS_TEM_PARAMS` / `MM_KERNEL_FUNC_TEM_PARAMS`
  - 枚举：`MatmulMode { REMOTE=0, LOCAL=1, DEFERRED_SYNC=2 }`（保持不变）
  - 结构体：`MMTiling`（对照 `QBMMTiling`）、`LocalParams`、`Params`
  - 方法：`Init`/`Run`/`operator()`/`GetCommPolicy`/`ResetGmAddr`/`ProcessSingleBatch`/`CalcDependTileIdx`/`SetL2Cache`
  - 成员：`mmadOp_`/`commPolicy_`/`aGmAddr_`/`localAGmAddr_`/`bGmAddr_`/`cGmAddr_`/`biasGmAddr_`/`isAtomicAdd_`。
    **无** `zeroBiasGmAddr_`、`needUpdateTail_`、`isBias_`（bias 是否处理改由 mmad 按 `biasGmAddr` 指针判断）。

## 4. 类结构设计

### 4.1 模板参数

```cpp
template <class ProblemShape, class BlockMmad, class BlockScheduler, class CommPolicy>
class AllToAllMmKernel;
```

### 4.2 关键类型别名（对照量化版删改）

| 量化版 | 非量化版 | 说明 |
|---|---|---|
| `BlockMmad::Params` 含 a/b/c/bias/scaleA/scaleB | 删 scaleA/scaleB | 基本 mmad Params 自带 mL1/nL1/kL1/mL0/nL0/kL0/l1Stages/l0cStages/splitKNum |
| `L1Params {kL1, scaleKL1, nBufferNum}` | 删 | L1/L0 参数并入 `BlockMmadParams` |
| `MakeLayoutScaleA/B`、`SCALE_C0`、`C0_SIZE`（IsFp4 分支） | 删 | 无 scale；C0 由 `Te::C0_ELEMENT<T>` 推导 |
| `kCacheLineAlignMask`（IsFp4 ? 0xff : 0x7f） | 固定 `0x7f` | 非 fp16 无 packed dtype |
| `BlockMmad::WEIGHT_NZ` | `BlockMmad::WEIGHT_NZ_FORMAT` | 基本 mmad 的命名 |

### 4.3 Params 结构

```cpp
struct MMTiling {
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t dbL0C;
};

struct LocalParams {
    uint32_t rankId;
    uint32_t rankSize;
    uint64_t originalM;          // 单卡负责的总 M 行数
    GM_ADDR localAGmAddr;        // 本地输入基址
    uint32_t localMatmul;        // 0: remote, 1: DEFERRED_SYNC, 2: GM 累加
    MatmulMode matmulMode;       // REMOTE / LOCAL / DEFERRED_SYNC
    uint32_t headTileSize;
};

struct Params {
    ProblemShape problemShape;
    BlockMmadParams mmadParams;  // 含 splitKNum，由融合接口按模式设置
    BlockSchedulerParams schParams;
    MMTiling mmTiling;
    LocalParams localParams;
};
```

**差异**：删 `localScaleAGmAddr`、`zeroBiasGmAddr`；`splitKNum` 从量化版的 `LocalParams` 移到 `mmadParams`；无 `L1Params`。

### 4.4 mmad 初始化

基本 mmad 的 `Init(const BlockMmadParams&)` 一次到位（模式3-REMOTE 时 `biasGmAddr` 先置空）。
字段由**非量化 tiling 直取**（不再从量化 tiling 派生）：

```
mL1=tile.mL1, nL1=tile.nL1, kL1=tile.kL1,
mL0=tile.baseM, nL0=tile.baseN, kL0=tile.baseK,
l1Stages=tile.l1BufferNum, l0cStages=(tile.l0cDB>1)?2:1, splitKNum 按模式
```

## 5. 关键方法设计

### 5.1 Run / Init / ResetGmAddr

```cpp
__aicore__ inline void Run(const Params &params)
{
    Init(params);
    if (isAtomicAdd_) {
        AscendC::SetAtomicAdd<CType>();    // 仅模式3-REMOTE：Run 级原子累加
    }
    // GM累加模式（模式3-REMOTE）：bias 已在 LOCAL 阶段写入 GM，mmad 必须跳过 bias
    BlockMmadParams mmadParams = params.mmadParams;
    if ((matmulMode == MatmulMode::REMOTE) && (localMatmul == 2)) {
        mmadParams.biasGmAddr = nullptr;
    }
    BlockScheduler bs(params.problemShape, params.schParams);
    mmadOp_.Init(mmadParams);
    ProcessSingleBatch(params, bs, 0, true);
    if (isAtomicAdd_) {
        AscendC::SetAtomicNone();
    }
}
```

`Init` 判定：

```cpp
if ((matmulMode == MatmulMode::REMOTE) && (localMatmul == 2)) {
    isAtomicAdd_ = true;    // GM累加模式：local 已写 GM，REMOTE 全原子
    // bias 跳过：由 Run() 里把 mmadParams.biasGmAddr 置空，
    // mmad Init 内部据此推导 isBias_=false（指针判断，不传 isBias 参数）
}
ResetGmAddr(params);
```

### 5.2 ProcessSingleBatch 三分支

分支判定沿用量化版：

```cpp
bool localFirst = (matmulMode == MatmulMode::LOCAL) && (localMatmul == 2);
bool deferredSync = (matmulMode == MatmulMode::DEFERRED_SYNC);
```

**块迭代（非量化调度器，`BlockSchedulerMatmulBasic`）**：

```cpp
if (curBlockIdx >= bs.GetCoreNums()) {
    return;                       // 核数超出实际所需（如 blockNums < coreNums）
}
for (int64_t blockIdx = curBlockIdx; blockIdx < bs.GetBlockNums(); blockIdx += coreNums) {
    auto singleShape = bs.template GetBlockShape<transB, BType>(blockIdx); // {blkM, blkN, k_, batch_}
    if (M <= 0 || N <= 0) {
        continue;                 // 空子片：不参与该尾块切分的核
    }
    auto blockCoord = bs.GetBlockCoord(blockIdx);   // 必须在 GetBlockShape 之后
    mPos = Get<MNK_M>(blockCoord); nPos = Get<MNK_N>(blockCoord);
    ...
}
```

三分支的 mmad 调用统一为 7 参（`splitKIdx` 即 `remoteRankCnt`，bias 恒传真值）：

```cpp
mmadOp_(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, singleShape, nullptr, splitKIdx);
```

- **branch A：LOCAL**：每块一次 mmad，`splitKIdx=0`（`splitKNum=1`，独立 matmul 覆盖写 + bias）。
- **branch B：DEFERRED_SYNC**：Phase1 self（`splitKIdx=0`，清零 L0C + bias，驻留 L0C，读本地 GM 与 AIV
  UDMA put 并行）→ Phase2 wait（`WaitTile` 去重递增）→ Phase3 遍历其余 rank（`splitKIdx=1..`，末次单次 fixpipe）。
- **branch C：REMOTE**：wait 后遍历所有 rank（`localMatmul==2` 时跳过 self），`splitKIdx=0..`，
  L0C 累加 + 末 rank 单次 fixpipe。

末尾兜底：`!localFirst` 时把 `readyTileIdx` 等到 `totalTiles-1`（尾核不满核时防 AIV 侧 barrier 悬挂）。

### 5.3 atomic 开关规则（总结）

| 场景 | 开关 |
|---|---|
| 模式1/2 | 无 atomic（单次 fixpipe 覆盖写） |
| 模式3-LOCAL | 全程关（每块仅一次覆盖写） |
| 模式3-REMOTE | 全程开（`isAtomicAdd_`，Run 级单次开关） |

### 5.4 bias 处理

- 模式1/2/3-LOCAL：所有 mmad 调用传真 bias，mmad 内部按 `isFirstSplitK` 只在 `splitKIdx==0` 加一次
  （**无 zero-bias workspace**，由 blaze 侧 `NeedProcessBias`/`CopyL1FromGM` 的 isFirstSplitK 门控保证）。
- 模式3-REMOTE：`Run` 前把 `mmadParams.biasGmAddr` 置空（bias 已在 LOCAL 阶段写入 GM，避免重复累加）。

### 5.5 SetL2Cache / CalcDependTileIdx / 尾块

- `SetL2Cache`：删 scale 参数与 `SetScaleL2Cache`；保留 `isAtomicAdd_` 时 `gmC` 的 `CACHE_MODE_DISABLE`
  （防 RMW 读 L2 旧值）与 B 的 cache hint 逻辑。
- `CalcDependTileIdx(mPos, headTileSize, totalTiles)`：不变。
- 尾块：由调度器 Params 静态切分（`mBaseTailSplitCnt`/`nBaseTailSplitCnt`/`mTailMain`/`nTailMain`），
  **无** `needUpdateTail_`/`UpdateTailTile`。

## 6. 与量化版本逐项差异汇总

| 项 | 量化版 | 非量化版 | 原因 |
|---|---|---|---|
| scaleA/scaleB/scaleK | 有 | 删 | 非量化无 scale |
| mmad 调用 | 8 参（A,B,ScaleA,ScaleB,Bias,C,shape,splitKIdx） | 7 参（A,B,Bias,C,shape,nullptr,splitKIdx） | 无 scale，nullptr 为 gmScalePtr 占位 |
| mmad Init | `Init(problemShape, l0TileShape, L1Params, isBias, dbL0C, splitKNum)` | `Init(BlockMmadParams)` | 基本 mmad 参数自包含 |
| splitKNum 位置 | `LocalParams` | `mmadParams` | 融合接口按模式设置 |
| scheduler | `BlockSchedulerQuantBatchMatmulV3`（GetTileIdx 循环 + UpdateTailTile 动态尾块） | `BlockSchedulerMatmulBasic`（for 下标步进 + 静态尾块 + 空子片 continue） | 非量化调度器 |
| 调度器 Params | 8 字段 | 13 字段 + 4 默认（mL1/nL1/kL1 直取，含 isHf32） | 调度器按 L1 块切分 |
| 空块处理 | `return` | `continue` | 调度器循环结构差异 |
| 通信 | A + ScaleA | 仅 A | 无 scale |
| FP4 分支（C0_SIZE_B4 / mask 0xff） | 有 | 无 | 非 fp16 场景 |

## 7. 待确认事项与风险

1. **ST 完整编译 + 多 rank 精度/性能验证**：需真实 CANN 环境（本环境 `include/tensor_api` 未初始化），
   重点验证 `KERNEL_TYPE_MIX_AIC_1_1` 混核路径与 `MatmulTilingSwat` 真实平台查询路径。
2. **L0C ping-pong 与多 split 复用**：blaze 侧已保证 L0C 半区只在 `isLastSplitK` 后翻转（同块多 split
   用同一半区累加），`l0cStages=2` 下需实测算子级验证。
3. **headMSize 波次对齐**：ST 当前 headMSize 取 CLI/CSV 值（未按 tiling 推导 `CeilDiv(usedCoreNum, nTile)·baseM`），
   通信粒度与计算波次未对齐时的流水效率影响待性能数据评估（op_host 侧可用 `MatmulCommTilingBase` 推导）。
