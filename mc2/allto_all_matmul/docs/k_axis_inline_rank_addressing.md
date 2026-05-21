# K 轴内嵌 Rank 寻址优化方案

## 一、优化背景

### 1.1 前置优化

本文档是 `remove_transpose_optimization.md` 的后续优化。前置优化已去除了数据转置节点 `transStage_` 和 Scale 转置节点 `scaleTransStage_`，直接使用通信输出进行 matmul 计算。

### 1.2 当前问题

去除转置节点后，第一版实现采用 **rank 外循环 + fixpipe** 方案：

```
for rankIdx in [0..rankDim-1]:
    SetAtomicAdd (rankIdx >= 1)
    完整 matmul([tileM, rankK, N])   // BlockMmad
    fixpipe()                         // ← 性能瓶颈
```

每次 rank 迭代结束时调用 `CrossCoreSetFlag<0, PIPE_FIX>` + `CrossCoreWaitFlag`，排空 **MTE2 + MTE3 + FIX** 全部流水线，破坏了 BlockMmad 内部的 SWAT 流水线重叠，多次 fixpipe 导致性能较差。

### 1.3 优化目标

将 rank 循环「下沉」到 BlockMmad 内部 K 轴遍历，一次 BlockMmad 调用覆盖全部 rank，完全消除 rank 间同步点。

---

## 二、数据排布分析

### 2.1 通信后 GM 布局

以 `rankDim=4, tileM=12, rankK=1024, fullM=384, tileCnt=8` 为例：

```
全局 stride:
  globalStride = (fullM/rankDim) * rankK * sizeof(DTYPE_X1) = 96 * 1024 = 98304 bytes
  tileStride   = tileM * rankK * sizeof(DTYPE_X1)           = 12 * 1024 = 12288 bytes

commOutGM_ 物理布局:

offset=0                                offset=98304                          offset=196608
  ┤ rank0 数据区                          ┤ rank1 数据区                         ┤
  ├──────┬──────┬───┬──────┤            ├──────┬──────┬───┬──────┤            ├──────┬
  │Tile0 │Tile1 │...│Tile7 │            │Tile0 │Tile1 │...│Tile7 │            │Tile0 │...
  │12×1K │12×1K │   │12×1K │            │12×1K │12×1K │   │12×1K │            │12×1K │
  ├──────┼──────┼───┼──────┤            ├──────┼──────┼───┼──────┤            ├──────┼
   0    12288      86016                98304 110592      184320              196608

验证: globalStride = tileCnt × tileStride = 8 × 12288 = 98304 ✓
      rank 区间无额外空洞，每个 rank 区恰好容纳 tileCnt 个 tile
```

### 2.2 同一 Task 的数据非连续性

对于某 task（如 taskIndex=3），需要处理的逻辑 A 矩阵为 `[tileM=12, rankK*rankDim=4096]`：

```
K=0..1023   → rank0 数据 at offset: 0*globalStride + 3*tileStride = 36864
K=1024..2047 → rank1 数据 at offset: 1*globalStride + 3*tileStride = 135168
K=2048..3071 → rank2 数据 at offset: 2*globalStride + 3*tileStride = 233472
K=3072..4095 → rank3 数据 at offset: 3*globalStride + 3*tileStride = 331776
```

**关键特性**：每个 rank 区内的 `[tileM, rankK]` 数据是标准行主序、完全连续。不连续只发生在 rank 区之间。

### 2.3 K 块边界约束

- `baseK = 128` 固定
- `rankK = 1024`，`128 | 1024` ✓
- 约束 `kL1` 整除 `rankK`（tiling 侧保证）

→ **每个 K 加载块一定落在某个 rank 区内部，永不跨 rank 边界**

---

## 三、优化方案

### 3.1 核心思想

将 rank 循环下沉到 BlockMmad 内部的 K 轴遍历：

```
【旧】rank 外循环:
  rank0: BlockMmad([tileM, rankK, N]) → fixpipe
  rank1: BlockMmad([tileM, rankK, N]) → fixpipe
  ...

【新】统一 K 轴:
  BlockMmad([tileM, rankK*rankDim, N])
    K 块 0..7:    rank=0, kInRank=0..1023     → 加载 rank0 A 数据
    K 块 8..15:   rank=1, kInRank=1024..2047   → 加载 rank1 A 数据
    K 块 16..23:  rank=2, ...
    K 块 24..31:  rank=3, ...
    全部 K 块累加到同一 L0C → 一次 CopyL0C2GM
```

### 3.2 流水线对比

```
旧方案 (rank 外循环):
  rank0: [L0A][L0B][Cube][写C] → fixpipe(排空全部)
  rank1:                        [L0A][L0B][Cube][累C] → fixpipe
  ...

新方案 (统一K轴):
  K0(rank0):  [L0A₀][L0B₀][Cube][累C₀]
  K8(rank1):          [L0A₁][L0B₁][Cube][累C₁]
  K16(rank2):                 [L0A₂][L0B₂][Cube][累C₂]
  ... 全程无 barrier，天然流水重叠
```

### 3.3 技术要点

| 项目 | 说明 |
|------|------|
| **C 累加** | BlockMmad 内部 `NON_FINAL_ACCUMULATION` 在 L0C 累加所有 K 块，最终 `CopyL0C2GM` 一次写回，**无需 SetAtomicAdd** |
| **B 矩阵** | `x2` 是 `[rankK*rankDim, N]` 标准连续，K 偏移直接索引，**无需特殊处理** |
| **Scale A** | 与 A 数据同理，按 `kInRank / MXFP_GROUP_SIZE` 寻址 |
| **Scale B** | `x2_scale` 标准连续，**无需特殊处理** |
| **M 分块** | scheduler 循环中每 (M,N) block 重算 mPos 偏移后的 per-rank 地址 |

### 3.4 BlockMmad 修改原则

- 新增 `SetMultiRankInfo(rankDim, rankK, aGmAddrs, scaleAGmAddrs)` 接口
- K 循环内 `if (rankDim_ > 1)` 条件分支，走 multi-rank 路径
- 默认 `rankDim_ = 1`，走原始路径，**对其他算子零影响**
- 仅修改 A 和 ScaleA 的 GM→L1 拷贝（~25 行），其余逻辑完全不变

---

## 四、修改文件清单

| # | 文件 | 修改内容 | 改动量 |
|---|------|---------|:---:|
| 1 | `allto_all_mx_quant_matmul_tiling_base.cpp` | `fullK = orgKValue * rankDim`；新增 kL1 整除 rankK 约束 | +14 行 |
| 2 | `mx_block_mmad_swat.h` | 新增 `SetMultiRankInfo` + 成员字段；A/ScaleA GM→L1 拷贝加 `rankDim_>1` 分支 | +61 行 |
| 3 | `tensor_mx_matmul_stage.h` | 去掉 rank 外循环/SetAtomicAdd/fixpipe；改为预计算 per-rank 地址 + 统一 scheduler 循环 | ~90 行变化 |
| 4 | `allto_all_mx_tensor_matmul_arch35.h` | `fullK *= rankDim`（tile + tail 两处） | +6 行 |

---

## 五、Tiling 约束校验

| 约束 | 保证方式 |
|------|---------|
| `kL1` 整除 `rankK` | `FillMxTensorTiling` 中强制调整：`while (rankK % r.kL1 != 0) r.kL1 -= r.baseK` |
| `baseK` 整除 `rankK` | `baseK=128` 固定，`rankK=1024` 自动满足 |
| kL1 chunk 不跨 rank 边界 | 以上两约束保证 |
| `scaleKL1` ≤ `kL1` | `r.scaleKL1 = min(r.scaleKL1, r.kL1)` |

---

## 六、内存布局变化

优化后的 workspace 内存布局：

```
workspaceGM:
┌─────────────────────────────────────────────────┐
│ commX1Scale | commOut（直接用于 matmul）│
│  (scaleLen) │        (commLen)         │
└─────────────────────────────────────────────────┘

总大小 ≈ scaleLen + commLen（无需 transOut / transX1Scale 空间）
```

---

## 七、性能收益

| 收益项 | 说明 |
|-------|------|
| **消除 fixpipe** | 去除 rankDim-1 次 PIPE_FIX 排空，K 块间无 barrier |
| **K 轴流水加深** | 全部 rank 的 K 块在单次 BlockMmad 中无缝流水 |
| **无 SetAtomicAdd 开销** | C 累加在 L0C 内部完成，不触发 GM 原子操作 |
| **内存节省** | 无需 transOut / transX1Scale 缓存 |

---

## 八、相关代码位置索引

| 代码位置 | 文件 | 说明 |
|---------|------|------|
| Tiling fullK 计算 | `allto_all_mx_quant_matmul_tiling_base.cpp:405` | `fullK = orgKValue * rankDim` |
| kL1 约束 | `allto_all_mx_quant_matmul_tiling_base.cpp:412-424` | 整除 rankK 检查 |
| SetMultiRankInfo | `mx_block_mmad_swat.h:195-200` | BlockMmad 扩展接口 |
| rankIdx/kInRank 计算 | `mx_block_mmad_swat.h:212-213` | K 循环内 rank 映射 |
| ScaleA multi-rank 拷贝 | `mx_block_mmad_swat.h:229-241` | rankDim_>1 分支 |
| A multi-rank 拷贝 | `mx_block_mmad_swat.h:261-275` | rankDim_>1 分支 |
| per-rank 地址预计算 | `tensor_mx_matmul_stage.h:92-97` | 无 M 偏移的基础地址 |
| mPos 偏移 + SetMultiRankInfo | `tensor_mx_matmul_stage.h:126-132` | scheduler 循环内 |
| arch35 fullK 赋值 | `allto_all_mx_tensor_matmul_arch35.h:165,219` | tile + tail |

---

## 九、已修复的精度问题

### 9.1 B 矩阵 Layout 维度反转（根因）

`tensor_mx_matmul_stage.h` 中 B 矩阵 GM layout 原为 `MakeNDLayout<BType>(n_, k_)`（NDLayout [N,K]），但 BlockMmad 内部 `MakeCoord(kL1Offset, 0)` 是按 DNLayout [K,N] 设计的——第一维应为 K。

ND→ZN DMA 固定映射：源行→ZN K、源列→ZN N。NDLayout 下源行是 N 维数据，导致 ZN 的 K 维实际存储了 N 维数据，MMAD 累加语义反转。

**修复**：改为 `MakeDNLayout<BType>(k_, n_)`。DNLayout strides 为 `(1, K_size)`，物理偏移 `MakeCoord(K=k, N=n) = n*K_size + k` 与 NDLayout `MakeCoord(N=n, K=k) = n*K_size + k` 一致，只是逻辑维度标签从 [N,K] 变为 [K,N]，不改变物理数据解读。配套修改 gmBlockB 子 tensor 创建：`MakeCoord(kPos, nPos), MakeShape(k_, curN_tile)`。

**注意**：原版非 MX 代码 (`tensor_matmul_stage.h:68`) 一直使用 `MakeDNLayout<BType>(k_, n_)`，此 bug 是 MX 版重构时引入的。

### 9.2 ScaleA stride 公式

`scaleKPerRank = CeilDiv(rankK, 32)` 在 `rankK mod 64 ∈ [1,32]` 时与正确值 `CeilDiv(rankK, 64)*2` 不一致（差 1 byte），导致 mPos>0 时 ScaleA 地址偏移错误。rankK 为 64 整倍数时两者数值相等（如 768→24），当前测试不触发。

**修复**：改为 `((rankK_ + 63U) / 64U) * 2U`。同步修正 `kScaleGrp`。

### 9.3 ScaleA multi-rank Layout

`mx_block_mmad_swat.h` 中 ScaleA multi-rank 路径原用 `MakeNDLayout<ScaleAType>(curM, CeilDiv(rankK_, MXFP_GROUP_SIZE))`，fp8_e8m0_t 的 NDLayout 有 MX_SCALE_K0=2 的特殊 stride（48 bytes/row），与 ScaleA 实际 per-row stride 24 bytes 不匹配。

**修复**：改为 `MakeScaleANDLayout<ScaleAType>(curM, CeilDiv(rankK_, MXFP_GROUP_SIZE))`，使用 `Std::ignore_t` 调度器生成标准 stride。

**注意**：全 1 输入下 ScaleA 值恒为 1.0，此 stride 错误对当前测试不可见，但在非全 1 输入 + 多 mPos 场景下会读错行。

---

**文档版本**: v1.1  
**创建日期**: 2026-05-21  
**前置文档**: `remove_transpose_optimization.md`  
**状态**: 已实施，精度 PASS
