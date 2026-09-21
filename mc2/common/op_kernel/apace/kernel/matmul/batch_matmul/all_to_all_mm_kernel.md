# all_to_all_mm_kernel 非量化实现方案

## 1. 背景与目标

基于量化版本 `all_to_all_qbmm_mx_kernel.h`（mxfp8 场景 quantMatmul）移植出**非量化版** all-to-all MatMul kernel，用于外部通算融合接口的三种调度模式：

| 模式 | localMatmul | matmulMode | launch 次数 | 说明 |
|---|---|---|---|---|
| 1 | 0 | REMOTE | 1 | 单 launch，一次循环算完所有 rank |
| 2 | 1 | DEFERRED_SYNC | 1 | 每个通信 tile：先 local 后 remote（保留本地先算 + wait 放中间的重叠） |
| 3 | 2 | 先 LOCAL 后 REMOTE | 2 | LOCAL 算完全部 local 写 GM，REMOTE 跳过 self 原子累加 |

**核心约束**：
- ops-tensor 库不可修改，仅基于现有接口在 kernel 侧改动。
- 非量化 mmad（`block_mmad_matmul_basic.h`）是自包含 matmul，无 splitKIdx 接口，无法跨调用保持 L0C 状态。
- 因此**放弃量化版的"L0C 跨 rank 累加 + 单次 fixpipe"，统一改为 GM 累加**：第一次写入覆盖写（atomic 关），其余写入原子累加（atomic 开）。
- 调度器**继续复用量化版 `block_scheduler_qbmm.h`**（调度逻辑本身与量化无关，`GetBlockShape` 的 QuantMode 对齐分支用 `QuantMode::DEFAULT` 编译期消掉），保留 `GetTileIdx`/`GetTileCoord`/`UpdateTailTile` 全套机制，kernel 循环结构与量化版保持一致。

## 2. 总体设计

### 2.1 组件选型

| 组件 | 量化版 | 非量化版 | 说明 |
|---|---|---|---|
| mmad | `BlockMmad<MatmulWithScaleMx<...>>` | `BlockMmad<MatmulMultiBlockBasic<NONE_FULL_LOAD_MODE, 0, KernelMmadMultiBlockBasic, 0>>` | 无 scale，无 splitKIdx |
| scheduler | `BlockSchedulerQuantBatchMatmulV3<...>` | 复用同款 | 模板参数 A/B 的 QuantMode 传 `QuantMode::DEFAULT` |
| CommPolicy | 外部注入 | 外部注入 | 不变 |
| 通信 | A + ScaleA 双通道 | 仅 A | ScaleA 通道删除 |

### 2.2 数据流

```
C[M][N] = Σ_{r∈[0,rankSize)} A_r[M][K] · B_r[K][N]
```

- `A_r`：rank r 的 M 段（self 用 localAGmAddr，其余用通信 buffer `selfWinAddr`），M 轴整体 `rankSize * originalM`。
- `B_r`：B 的 K 段，K 轴整体 `rankSize * K`。
- 每个输出块由调度器分配唯一核计算，块内按 rank 遍历做 rankSize 次 mmad。

### 2.3 三种模式的执行流程

```
模式1 (REMOTE, localMatmul=0):  [wait(commTile)] → 遍历 rank0..rankSize-1
    首次调用(rank0): 覆盖写 + 真bias (atomic关)
    后续调用:        原子累加 + zero-bias (atomic开)

模式2 (DEFERRED_SYNC, localMatmul=1):  self先算 → [wait(commTile)] → 遍历其余rank
    self调用:   覆盖写 + 真bias (atomic关)
    其余调用:   原子累加 + zero-bias (atomic开)

模式3 (localMatmul=2):
    LOCAL launch: 每块仅一次 mmad(覆盖写 + 真bias), atomic全程关
    REMOTE launch: 跳过self, 每块 rankSize-1 次 mmad(全原子累加, 无bias), atomic全程开
```

## 3. 文件与命名

- 文件：`all_to_all_mm_kernel.h`，位于 `mc2/common/op_kernel/apace/kernel/matmul/batch_matmul/`。
- 命名风格对齐量化版：
  - 类名：`AllToAllMmKernel`（对照 `AllToAllQbmmMxKernel`）
  - 模板参数宏：`MM_KERNEL_CLASS_TEM_PARAMS` / `MM_KERNEL_FUNC_TEM_PARAMS`
  - 枚举：`MatmulMode { REMOTE=0, LOCAL=1, DEFERRED_SYNC=2 }`（保持不变）
  - 结构体：`MMTiling`（对照 `QBMMTiling`）、`LocalParams`、`Params`
  - 方法：`Init`/`Run`/`operator()`/`GetCommPolicy`/`ResetGmAddr`/`ProcessSingleBatch`/`CalcDependTileIdx`/`SetL2Cache`
  - 成员：`mmadOp_`/`commPolicy_`/`aGmAddr_`/`localAGmAddr_`/`bGmAddr_`/`cGmAddr_`/`biasGmAddr_`/`zeroBiasGmAddr_`/`isAtomicAdd_`/`isBias_`/`needUpdateTail_`
  - 注释风格：文件头 `\file` + `\brief`；模式/机制关键点用中文块注释；与量化版一致。

## 4. 类结构设计

### 4.1 模板参数

```cpp
template <class ProblemShape, class BlockMmad, class BlockScheduler, class CommPolicy>
class AllToAllMmKernel;
```

### 4.2 关键类型别名（对照量化版删改）

| 量化版 | 非量化版 | 说明 |
|---|---|---|
| `BlockMmad::Params` 含 a/b/c/bias/scaleA/scaleB | 同，**删 scaleA/scaleB** | 基本 mmad 自带 mL1/nL1/kL1/mL0/nL0/kL0/l1Stages/l0cStages |
| `L1Params {kL1, scaleKL1, nBufferNum}` | 删 | L1 参数并入 `BlockMmadParams` |
| `C0_SIZE/SCALE_C0/kCacheLineAlignMask` | 删（scale 相关），保留 `C0_SIZE` 若用到 | — |
| `MakeLayoutScaleA/B` | 删 | — |
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
    GM_ADDR zeroBiasGmAddr;      // zero-bias workspace（baseN 大小，tiling 分配并清零）
    uint32_t localMatmul;        // 0: remote, 1: L0c 累加语义(DEFERRED), 2: GM 累加
    MatmulMode matmulMode;       // REMOTE / LOCAL / DEFERRED_SYNC
    uint32_t headTileSize;
};

struct Params {
    ProblemShape problemShape;
    BlockMmadParams mmadParams;
    BlockSchedulerParams schParams;
    MMTiling mmTiling;
    LocalParams localParams;
};
```

**差异**：删 `localScaleAGmAddr`、`splitKNum`（无 splitKIdx 接口）；`L1Params`/`QBMMTiling` 合并精简。

### 4.4 mmad 初始化

基本 mmad 的 `Init(const BlockMmadParams&)` 一次到位（量化版是 `Init(problemShape, l0TileShape, l1Params, isBias, dbL0C, splitKNum)`）。参数映射：

```
mL1=baseM, nL1=baseN, kL1=stepK*baseK,
mL0=baseM, nL0=baseN, kL0=baseK,
l1Stages=nBufferNum, l0cStages=(dbL0C>1)?2:1
```

## 5. 关键方法设计

### 5.1 Run / Init / ResetGmAddr

```cpp
__aicore__ inline void Run(const Params &params)
{
    Init(params);
    if (isAtomicAdd_) {
        AscendC::SetAtomicAdd<CType>();   // 模式3-REMOTE：全程原子累加
    }
    BlockScheduler bs(params.problemShape, params.schParams);
    BlockShape l0TileShape{baseM, baseN, baseK, 0};
    mmadOp_.Init(EffectiveMmadParams(params));   // 模式3-REMOTE 时 biasGmAddr 置空
    ProcessSingleBatch(params, bs, 0, true);
    if (isAtomicAdd_) {
        AscendC::SetAtomicNone();
    }
}
```

`Init` 判定（对照量化版 :274-278）：

```cpp
isBias_ = (params.mmadParams.biasGmAddr != nullptr);
if ((matmulMode == MatmulMode::REMOTE) && (localMatmul == 2)) {
    isAtomicAdd_ = true;    // GM累加模式：local 已写 GM，REMOTE 全原子
    isBias_ = false;        // bias 已在 LOCAL 阶段写入 GM，REMOTE 必须跳过
}
needUpdateTail_ = false;
ResetGmAddr(params);
```

### 5.2 ProcessSingleBatch 三分支

沿用量化版 `localFirst` / `deferredSync` 分支结构：

```cpp
bool localFirst = (matmulMode == MatmulMode::LOCAL) && (localMatmul == 2);
bool deferredSync = (matmulMode == MatmulMode::DEFERRED_SYNC);
```

**branch A：LOCAL**（localFirst，对照量化版 :380-393）
每块一次 mmad，覆盖写 + 真 bias，atomic 保持关：

```cpp
auto gmBlockBias = gmBias.Slice(Te::MakeCoord(0L, nPos), Te::MakeShape(1L, shapeN));
mmadOp_(gmBlockA, gmBlockB, gmBlockBias, gmBlockC, singleShape);   // 无 splitKIdx
```

**branch B：DEFERRED_SYNC**（对照量化版 :394-438，GM 累加版）
Phase1 self 覆盖写（真 bias）→ wait → Phase3 其余 rank 原子累加（zero-bias）：

```cpp
// Phase 1: 本 rank 的 local A × 本 rank 的 B 段 → GM 覆盖写
mmadOp_(gmBlockA_self, gmBlockB_self, gmBlockBias_real, gmBlockC, singleShape);

// Phase 2: wait（去重逻辑不变）
while (readyTileIdx < dependTileIdx) {
    readyTileIdx++;
    commPolicy_.WaitTile(readyTileIdx);
}

// Phase 3: 遍历其它 rank，原子累加（最后一个 rank 为最后一次 RMW）
AscendC::SetAtomicAdd<CType>();
for (rank != self) {
    mmadOp_(gmBlockA_remote, gmBlockB_r, gmBlockBias_zero, gmBlockC, singleShape);
}
AscendC::SetAtomicNone();
```

**branch C：REMOTE**（对照量化版 :440-481）
wait 后遍历所有 rank：首个（rank0）覆盖写 + 真 bias，其余原子 + zero-bias：

```cpp
bool isFirstWrite = true;
AscendC::SetAtomicNone();
for (rank = 0; rank < rankSize; rank++) {
    // self 数据源选择与量化版一致（localMatmul==2 时 continue 跳过）
    if (isFirstWrite) {
        mmadOp_(gmBlockA, gmBlockB, gmBlockBias_real, gmBlockC, singleShape);
        isFirstWrite = false;
    } else {
        if (isFirstWrite) { AscendC::SetAtomicAdd<CType>(); }
        mmadOp_(gmBlockA, gmBlockB, gmBlockBias_zero, gmBlockC, singleShape);
    }
}
AscendC::SetAtomicNone();
```

### 5.3 atomic 开关规则（总结）

| 场景 | 开关 |
|---|---|
| 模式1/2 首次调用（覆盖写） | atomic 关 |
| 模式1/2 后续调用 | atomic 开，块内临时开关 |
| 模式3-LOCAL | 全程关（每块仅一次覆盖写） |
| 模式3-REMOTE | 全程开（`isAtomicAdd_`，Run 级） |

> 风险点：块内 `SetAtomicAdd`/`SetAtomicNone` 连续切换需确保 fixpipe 流水同步，实现用 `PipeBarrier<PIPE_ALL>`（dav-3510 无 `PIPE_FIX`，见 CHANGELOG_debug.md）。

### 5.4 bias 处理（方案A：zero-bias workspace）

- 模式3-LOCAL、模式1/2 首次调用：`gmBias` 指向真 bias（`biasGmAddr_`）。
- 模式1/2 非首次调用：`gmBias` 指向 `zeroBiasGmAddr_`（全 0，加了等于没加）。
- 模式3-REMOTE：`Init` 时 mmad 的 `biasGmAddr` 置空（`isBias_=false`），gmBias 传任意值。
- `zeroBiasGmAddr_` 由 tiling 分配（baseN 大小），融合接口透传。

### 5.5 SetL2Cache / CalcDependTileIdx / 尾块

- `SetL2Cache`：删 scale 参数与 `SetScaleL2Cache`；保留 `isAtomicAdd_` 时 `gmC` 的 `CACHE_MODE_DISABLE`（防 RMW 读 L2 旧值）与 B 的 cache hint 逻辑。
- `CalcDependTileIdx(mPos, headTileSize, totalTiles)`：不变。
- `needUpdateTail_`/`UpdateTailTile`/`GetEndBlockIdx`/`GetTotalCnt`：**保留**（沿用 quantized scheduler）。

## 6. 与量化版本逐项差异汇总

| 项 | 量化版 | 非量化版 | 原因 |
|---|---|---|---|
| scaleA/scaleB/scaleK | 有 | 删 | 非量化无 scale |
| mmad 接口 | 带 `splitKIdx` | 无 | 基本 mmad 自包含 |
| L0C 跨 rank 累加 | 有 | 无（改 GM 累加） | 无 splitKIdx |
| 每块 GM 写次数 | 1 次 fixpipe | rankSize 次（1 覆盖 + N-1 原子） | 同上 |
| bias | mmad 内部按 `isFirstSplitK` | kernel 侧按首次调用传真/零 bias | 基本 mmad 每次调用都加 bias |
| scheduler | `BlockSchedulerQuantBatchMatmulV3` | 复用同款，QuantMode 传 DEFAULT | 调度逻辑与量化无关 |
| 循环/尾块机制 | `GetTileIdx` + `UpdateTailTile` | 不变 | 同上 |
| 通信 | A + ScaleA | 仅 A | 无 scale |
| `localMatmul==1` 收益 | L0C 驻留省带宽 | 仅保留 self-before-wait 重叠 | 接口限制 |

## 7. 待确认事项与风险

1. **块内 atomic 切换的流水同步**：`SetAtomicAdd` 连续开关是否影响 fixpipe 顺序，需实测/看硬件文档。
2. **zero-bias workspace 来源**：tiling 分配并清零的语义要写进融合接口约定。
3. **blockShape.K 构造**：量化 scheduler 返回 dims2/3 为 split offset，传给基本 mmad 前需覆盖为 `K_rank`。
4. **L0C ping-pong**：基本 mmad 的 `l0cStages` 映射 `dbL0C>1`，需验证多 rank 调用下 L0C 复用无冲突。
5. **模式1 首次写由 rank0 承担**（非 self），其 A 数据来自通信 buffer，wait 必须发生在首次调用前。
