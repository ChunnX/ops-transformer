# AllGather + QuantMatmul CCU 方案设计

## 1. 背景

### 1.1 问题描述

在 apace 仓库中实现基于 CCU 通信的 AllGather + MX FP8 量化矩阵乘融合算子。每张卡持有本地 A 矩阵的一部分（M 轴切分），通过 AllGather 收集所有卡的 A 数据后，与共享的 B 矩阵做 MX 量化矩阵乘，输出完整的 C 矩阵。本方案须充分利用 AIC-only 全 C 核执行模型的优势，实现通信与计算的流水掩盖。

### 1.2 现有方案

当前 `all_gather_matmul_v2` 仓中的 `all_gather_quant_bmm.h` 已实现基于 CCU 通信的 AllGather + 量化矩阵乘：

| 维度 | 现有方案（all_gather_quant_bmm.h） |
|------|-------------------------------------|
| 仓库位置 | mc2/all_gather_matmul_v2/op_kernel/arch35/ |
| 通信方式 | Hccl<HCCL_SERVER_TYPE_CCU> AllGather |
| 执行模型 | AIC-only（AIV 不参与） |
| 通信提交 | per-tile 逐个提交（handleList_ 数组） |
| 计算内核 | Mc2QuantBatchMatmulASWKernel（标准 QBMM kernel） |
| 区域调度 | local compute + gather compute 两段式 |
| Tiling | Mc2Tiling::AllGatherMatmulTilingDataFp8 |

### 1.3 新方案目标

在 apace 仓库中实现新的 AllGather + QuantMatmul CCU 方案，核心改进：

| 维度 | 新方案（apace CCU） |
|------|---------------------|
| 仓库位置 | mc2/common/op_kernel/apace/kernel/all_gather_quant_matmul/ |
| 通信方式 | Hccl<HCCL_SERVER_TYPE_CCU> AllGather（保持 CCU） |
| 执行模型 | AIC-only（全 C 核，重要优势） |
| 通信提交 | head/tail 批量提交（3 个 handle） |
| 计算内核 | QmmMxKernelAgUdma + CommPolicy（FragmentTensor） |
| 区域调度 | HEAD/MAIN/TAIL 三区域 FragmentTensor 调度 |
| Tiling | ccuAllGatherMxMatmulTilingData（apace 统一 tiling） |

### 1.4 设计约束

- 目标芯片：Ascend 950（dav-3510）
- 数据类型：A/B = fp8_e4m3fn_t 或 fp8_e5m2_t（任意组合），ScaleA/ScaleB = fp8_e8m0_t，C = bfloat16_t
- 执行模型：AIC-only（全 C 核），这是本方案的重要优势
- 通信风格参考 `all_to_all_mx_quant_matmul_hcomm_impl.h`（HcommCommState + HcommCommWaitPolicy 模式）
- AllGather API 用法参考 `all_gather_matmul_fp16_bf16.h` 和 `all_gather_quant_bmm.h`（hccl_.AllGather<true> 签名）
- FragmentTensor 计算能力参考 `all_gather_mx_matmul_udma_impl.h`（HEAD/MAIN/TAIL 区域调度）
- 约束：tileCnt > 0（确保至少有一个 head tile，空 tensor 场景由上层拦截不进入此 kernel）

### 1.5 与现有方案的核心差异

| 差异点 | 现有方案（all_gather_quant_bmm.h） | 新方案（apace CCU） |
|--------|-------------------------------------|---------------------|
| 通信提交 | per-tile 逐个提交 | head/tail 批量提交 |
| 计算内核 | Mc2QuantBatchMatmulASWKernel | QmmMxKernelAgUdma + CommPolicy（FragmentTensor） |
| 区域调度 | local + gather 两段式 | HEAD/MAIN/TAIL 三区域 FragmentTensor |
| B 矩阵处理 | kernel 内按 rank 偏移 K 轴 | B 全 rank 共享，不偏移 |
| CommPolicy | 无（直接 hccl_.Wait） | HcommCommWaitPolicy 策略注入 |
| Handle 管理 | handleList_ 数组（per-tile） | 3 个 handle（scale + dataHead + dataTail） |
| Tiling 系统 | Mc2Tiling（all_gather_matmul_v2） | apace 统一 tiling |

> FragmentTensor 内核的优势：B 矩阵全 rank 共享时，标准 kernel 的 `rank * K` 偏移会导致越界。FragmentTensor 通过显式地址列表管理各 rank 的 A 数据片段，B 统一从基址切片，避免了偏移问题。

### 1.6 与 AllToAll CCU 方案的核心差异

| 差异点 | AllToAll | AllGather |
|--------|----------|-----------|
| B 矩阵 | 按 rank 切分 K 轴 | 全 rank 共享同一 B |
| 通信原语 | AlltoAll | AllGather |
| 输出 C | 仅本 rank 的 M 行 | 全 rankSize × M 行 |
| 计算内核 | QuantMatmulMxKernel（标准） | QmmMxKernelAgUdma + CommPolicy（FragmentTensor） |
| B 切片 | gmB.Slice(rank * K, ...) | gmB.Slice(0, ...)（不按 rank 偏移） |

> AllToAll 可直接复用标准 `QuantMatmulMxKernel`，因为其 B 按 rank 切分 K 轴，与 kernel 内 `rank * K` 偏移逻辑一致。AllGather 的 B 全 rank 共享，标准 kernel 的 B 偏移会导致越界，因此需使用 FragmentTensor 内核。

---

## 2. AllGather 数据排布与提交策略

### 2.1 Gather 缓冲区数据排布

Gather 缓冲区位于 workspace，采用 rank-major 布局，data 段和 scale 段连续存放：

```
workspaceGM
├── Data 段 [0, dataRegionBytes)
│   ├── rank0: [tile0 | tile1 | ... | tail]   <- M 轴顺序，每 tile = tileM × K
│   ├── rank1: [tile0 | tile1 | ... | tail]
│   └── ...
│
├── Scale 段 [dataRegionBytes, +scaleRegionBytes)
│   ├── rank0: [tile0 | tile1 | ... | tail]   <- 每 tile = tileM × scaleKLen
│   ├── rank1: [tile0 | tile1 | ... | tail]
│   └── ...
│
└── HCCL 内部 workspace [dataRegionBytes + scaleRegionBytes, ...)
    起始偏移 = dataRegionBytes + scaleRegionBytes，大小由 mc2InitTiling 独立配置，不与 data/scale 段重叠
```

关键尺寸：
```
m              = tileCnt * tileM + tailCnt * tailM        // 每 rank M 行数
scaleKLen      = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE
dataRegionBytes  = rankSize * m * k * sizeof(AType)
scaleRegionBytes = rankSize * m * scaleKLen * sizeof(fp8_e8m0_t)
```

- `stride`（rank 间步长）= m × k（data）或 m × scaleKLen（scale）
- 每 rank 内 tile 按 M 轴顺序排列：head tiles（tileCnt × tileM）+ tail tiles（tailCnt × tailM）

---

## 3. 详细设计

### 3.1 通信层设计

#### HcommCommState

通信状态容器，直接持有 HCCL 对象和 handle（与 `all_to_all_mx_quant_matmul_hcomm_impl.h` 风格一致）：

```cpp
struct HcommCommState {
    static constexpr HcclServerType ServerType = HCCL_SERVER_TYPE_CCU;
    Hccl<ServerType> hccl_;
    HcclHandle scaleHandle_{0};
    HcclHandle dataHeadHandle_{0};
    HcclHandle dataTailHandle_{0};
    uint32_t tileCnt_{0};
};
```

#### HcommCommWaitPolicy

等待策略通过 `state_` 指针引用 HcommCommState，kernel 内逐 tile 调用 `WaitTile()`：

```cpp
struct HcommCommWaitPolicy {
    HcommCommState *state_{nullptr};
    __aicore__ inline void WaitTile(uint32_t tileIdx) {
        if (tileIdx == 0) return;                    // HEAD: 本 rank 数据，无需等待
        if (tileIdx == 1) {                          // 首次远端等待: 等 scale（全量）
            state_->hccl_.Wait(state_->scaleHandle_);
        }
        if (tileIdx <= state_->tileCnt_) {           // MAIN: 等 data head
            state_->hccl_.Wait(state_->dataHeadHandle_);
        } else {                                      // TAIL: 等 data tail
            state_->hccl_.Wait(state_->dataTailHandle_);
        }
    }
};
```

dependTileIdx 与 handle 的映射关系：

| dependTileIdx | Region | WaitTile 动作 | HCCL Handle |
|---------------|--------|---------------|-------------|
| 0 | HEAD | return（无需等待） | 无 |
| 1 | MAIN r0 | Wait(scale) + Wait(dataHead) | scaleHandle_, dataHeadHandle_ |
| 2..tileCnt | MAIN rN | Wait(dataHead) | dataHeadHandle_（已 wait 则立即返回） |
| commTurn | TAIL | Wait(dataTail) | dataTailHandle_ |

> MAIN 区域所有 round 共享 dataHeadHandle_，kernel 的 waitedMask 保证只 wait 一次。TAIL 区域同理。

#### Init 阶段

> **参考代码**：`all_to_all_mx_quant_matmul_hcomm_impl.h` 的 `Init()` 方法
> - HCCL 初始化（`InitV2` / `SetCcTilingV2`）→ 直接参考
> - 批量提交通信任务（`AllGather<true>` + repeat 参数）→ 参考 alltoall 的 `AlltoAll<true>` 批量提交模式，替换为 AllGather API
> - 绑定 CommPolicy（`GetCommPolicy().state_ = &commState_`）→ 直接参考

```cpp
// 以下代码中 hccl_、scaleHandle_ 等均为 commState_ 的成员，省略 commState_. 前缀

// 0. HCCL dataType 映射（参考 all_to_all_mx_quant_matmul_hcomm_impl.h）
//    scale 固定 FP8E8M0，data 按 AType 映射（e4m3 → FP8E4M3, e5m2 → FP8E5M2）
static constexpr uint64_t scaleDataType = AscendC::HCCL_DATA_TYPE_FP8E8M0;
static constexpr uint64_t dataDataType = GetHcclDataType<AType>();  // 模板映射 helper

// 1. HCCL 初始化
commState_.hccl_.InitV2(GetHcclContext<0>(), &mc2InitTiling);
commState_.hccl_.SetCcTilingV2(offsetof(TilingDataType, mc2CcTiling));
rankId_ = commState_.hccl_.GetRankId();
rankSize_ = commState_.hccl_.GetRankDim();

// 2. 计算 Gather 缓冲区地址
gatherDataAddr_  = workspaceGM;
gatherScaleAddr_ = workspaceGM + dataRegionBytes;

// 3. 提交 AllGather（非阻塞 <true>），Scale 先于 Data
//    Scale: 全量一次性提交
commState_.scaleHandle_ = commState_.hccl_.AllGather<true>(
    aScaleGM, gatherScaleAddr, m * scaleKLen,
    static_cast<HcclDataType>(scaleDataType), m * scaleKLen, 1);

//    Data Head: 通过 repeat 批量提交所有 head tile
commState_.dataHeadHandle_ = commState_.hccl_.AllGather<true>(
    aGM, gatherDataAddr, tileM * k,
    static_cast<HcclDataType>(dataDataType), m * k, tileCnt);

//    Data Tail: 通过 repeat 批量提交所有 tail tile
if (tailCnt > 0) {
    commState_.dataTailHandle_ = commState_.hccl_.AllGather<true>(
        aGM + headRows * k * sizeof(AType),
        gatherDataAddr + headRows * k * sizeof(AType),
        tailM * k, static_cast<HcclDataType>(dataDataType), m * k, tailCnt);
}

// 4. 绑定 CommPolicy -> CommState
quantMatmulKernelImpl_.GetCommPolicy().state_ = &commState_;
```

### 3.2 通算同步设计

参考 `all_to_all_mx_quant_matmul_hcomm_impl.h` 和 `all_to_all_mx_quant_matmul_udma_impl.h` 的设计，通信与计算的同步通过 **CommPolicy 策略模式** 解耦：kernel 通过模板参数注入 WaitPolicy，不感知底层是 UDMA 还是 CCU 通信。

**两种 WaitPolicy**：

```cpp
// UDMA 版本：跨核 flag 同步（AIV 通信 → AIC 计算）
struct UdmaCommWaitPolicy {
    __aicore__ inline void WaitTile(uint32_t tileIdx) {
        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(tileIdx);
    }
};

// CCU 版本：核内 hccl_.Wait 同步（AIC 内通信+计算）
// HcommCommWaitPolicy 定义见第 3.1 节
```

kernel 通过 `CommPolicy` 模板参数选择策略，同步点在 `Process()` 核心循环中：

```cpp
// kernel Process() 核心循环 — CommPolicy 无关的通用逻辑
uint32_t waitedMask = 0U;
while (sch.GetTileIdx(blockIdx)) {
    sch.GetTileCoord(blockIdx, mPos, nPos);
    auto ctx = ResolveTileCtx(mPos, ...);  // 解析 region + dependTileIdx

    // 按位去重：同一 dependTileIdx 只 wait 一次
    if ((waitedMask & (1U << ctx.dependTileIdx)) == 0U) {
        commPolicy_.WaitTile(ctx.dependTileIdx);   // ← 同步点（UDMA: CrossCoreWaitFlag / CCU: hccl_.Wait）
        waitedMask |= (1U << ctx.dependTileIdx);
    }

    // 执行 MMAD 计算
    mmadFrag(blockA, gmBlockB, blockScaleA, gmBlockScaleB, ...);
}

// 收尾：确保所有 dependTileIdx 均已 wait
for (uint32_t t = 0; t <= commTurn; ++t) {
    if ((waitedMask & (1U << t)) == 0U) {
        commPolicy_.WaitTile(t);
    }
}
```

**CCU 同步流程**（HcommCommWaitPolicy）：

1. **HEAD**（dependTileIdx=0）：本 rank 数据，直接 return，计算与通信并行启动
2. **MAIN**（dependTileIdx=1..tileCnt）：首次 `WaitTile(1)` 等 scale + dataHead，后续因 `waitedMask` 去重立即返回
3. **TAIL**（dependTileIdx=commTurn）：`WaitTile` 等 dataTail，被 MAIN 尾部计算掩盖

**UDMA vs CCU 同步对比**：

| 维度 | UdmaCommWaitPolicy | HcommCommWaitPolicy |
|------|---------------------|----------------------|
| 同步机制 | CrossCoreWaitFlag（AIV→AIC 跨核） | hccl_.Wait（AIC 核内） |
| 通信方 | AIV 下发 UDMA | AIC 直接 HCCL 提交 |
| AIV 参与 | 需要 | 不需要 |
| 同步开销 | 跨核 flag 开销 | 核内 wait，无跨核开销 |

`waitedMask` 去重机制和收尾 wait 循环对两种策略通用，保证同一依赖只 wait 一次，防止通信资源泄漏。

### 3.3 计算层设计

#### Kernel 复用策略

复用已有 `qmm_mx_kernel_ag_udma.h` 中的 `QmmMxKernelAgUdma`，为其新增 `CommPolicy` 模板参数：

```cpp
template <typename AType, typename BType, typename CType, typename CommPolicy = UdmaCommWaitPolicy>
class QmmMxKernelAgUdma {
    CommPolicy commPolicy_;
public:
    CommPolicy& GetCommPolicy() { return commPolicy_; }
    // ...
};
```

- UDMA 版本：`CommPolicy = UdmaCommWaitPolicy`（默认，内部调用 `CrossCoreWaitFlag`）
- CCU 版本：`CommPolicy = HcommCommWaitPolicy`（内部调用 `hccl_.Wait(handle)`）

共 2 处改动：将 kernel `Process()` 中的 `CrossCoreWaitFlag<0x2, PIPE_MTE2>` 替换为 `commPolicy_.WaitTile`：
- 主循环内（行 371）：`CrossCoreWaitFlag<0x2, PIPE_MTE2>(ctx.dependTileIdx)` → `commPolicy_.WaitTile(ctx.dependTileIdx)`
- 收尾循环（行 412）：`CrossCoreWaitFlag<0x2, PIPE_MTE2>(t)` → `commPolicy_.WaitTile(t)`

其余所有逻辑（FragmentTensor 构建、区域调度、BlockScheduler、BlockMmadFragC 计算、L2 Cache 优化）完全不变。

#### FragmentTensor 原理

FragmentTensor 是 apace 提供的多地址聚合 tensor，通过显式地址列表将多个不连续的 GM 区域映射为逻辑上连续的 M 轴，避免标准 kernel 对 B 矩阵按 rank 偏移 K 轴的越界问题。

**数据结构**：
```cpp
template <int Dim, int MaxFrag, class MakeLayout, typename T>
class FragmentTensor {
    FragmentParam<Dim> param_;       // assembledShape, fragmentSize, fragmentCnt
    GM_ADDR addrList_[MaxFrag];      // 各 fragment 的 GM 基址
};
```

- `fragmentCnt`：fragment 数量（HEAD=1, MAIN=rankSize-1, TAIL=rankSize）
- `fragmentSize`：每个 fragment 的 M 行数（HEAD=headRows, MAIN=tileM, TAIL=paddedTailM）
- `assembledShape[0]`：逻辑 M = fragmentSize × fragmentCnt
- `addrList_[i]`：第 i 个 fragment 的 GM 基址

**Slice 语义**：`frag.Slice(coord, shape)` 返回一个逻辑连续的 tensor view，底层自动路由到正确的 fragment 地址。例如 MAIN 区域 `mPos=0` 路由到 `addrList_[0]`（rank 0 的远端数据），`mPos=tileM` 路由到 `addrList_[1]`（rank 1 的远端数据）。

**与标准 kernel 的区别**：

| 维度 | 标准 QuantMatmulMxKernel | FragmentTensor Kernel |
|------|--------------------------|------------------------|
| A 地址管理 | 单一 GM 基址 + `rank * oriM` 偏移 | 地址列表，每 rank 独立基址 |
| B 地址管理 | `gmB.Slice(rank * K, nPos)` | `gmB.Slice(0, nPos)`（不偏移） |
| M 轴映射 | `actualMPos = rank * oriM + mPos` | FragmentTensor 自动路由 |
| 适用场景 | B 按 rank 切分 K（AllToAll） | B 全 rank 共享（AllGather） |

#### 区域调度（HEAD / MAIN / TAIL）

FragmentTensor kernel 将总 M 轴划分为三个区域，每区域对应不同的数据来源和通信依赖：

```
总 logical M = headRows + mainSectionRows + tailRows

├── HEAD ──────────────┤<- headRows = tileCnt x tileM（本 rank 数据）
│   dependTileIdx = 0  │   读本地 A（aGM），无需通信
│   FragmentTensor:    │   headFragA_（1 个 fragment，地址 = aGM）
│
├── MAIN ──────────────┤<- mainSectionRows = tileCnt x (rankSize-1) x tileM
│   dependTileIdx =    │   每 round 对应一个通信 tile
│     roundIdx + 1     │   读 Gather 缓冲区远端 rank 数据
│   FragmentTensor:    │   curMainA_（rankSize-1 个 fragment）
│   基址 = gatherDataAddr + r * mPerRank * dataBytesPerMRow + roundOffset
│
├── TAIL ──────────────┤<- tailRows = rankSize x paddedTailM
│   dependTileIdx =    │   读 Gather 缓冲区所有 rank 的 tail
│     commTurn         │   tailFragA_（rankSize 个 fragment）
└──────────────────────┘
```

**ResolveTileCtx**：调度器给出 `mPos` 后，解析为 `(region, dependTileIdx, regionMPos, fragA, fragC, rankCnt)`：
- `mPos < headRows` → HEAD，dependTileIdx=0，使用 headFragA_
- `mPos < headRows + mainSectionRows` → MAIN，dependTileIdx=roundIdx+1，使用 curMainA_（按 roundIdx 延迟构建/更新地址）
- `else` → TAIL，dependTileIdx=commTurn，使用 tailFragA_（首次命中时构建）

**Fragment 延迟构建**：MAIN 和 TAIL 的 FragmentTensor 不在 Init 时构建，而是在首次命中该区域时才构建地址列表，构建开销被通信 wait 掩盖。MAIN 区域在 roundIdx 变化时通过 `UpdateMainRoundAddrs` 增量更新地址（仅更新偏移，不重建 tensor 对象）。

#### B 矩阵处理

B 矩阵全 rank 共享，不按 rank 偏移 K 轴：

```cpp
// AllGather: B 共享，所有 rank 使用同一 B
auto gmBlockB = gmB.Slice(MakeCoord(0, nPos), MakeShape(Ki, curNtile));
auto gmBlockScaleB = gmScaleB.Slice(MakeCoord(0, nPos), MakeShape(scaleKLen, curNtile));
```

#### C 矩阵输出

C 按 rank 分段写入（每 rank 的 M 行连续）：

```cpp
cFragAddrs_[r] = cGM + r * mPerRank * cBytesPerM;  // rank r 的 C 输出基址
```

FragmentTensor 的 C 地址列表 `fragC` 同样按 rank 组织，HEAD 区域写入 `cFragAddrs_[rankId]`，MAIN/TAIL 区域写入对应远端 rank 的 C 地址。

### 3.4 通信计算双缓冲掩盖

#### 流水线设计

CCU 方案的通算掩盖通过 HCCL 异步提交 + 本地计算先行 + batch 等待三重机制实现：

```
阶段 1: Init 中批量提交所有 AllGather（非阻塞），HCCL CCU 硬件开始通信
阶段 2: kernel 先执行 HEAD 区域（本地 A × B），与 HCCL 通信并行
阶段 3: kernel 执行 MAIN 区域，per-tile WaitTile 等待对应通信完成
阶段 4: kernel 执行 TAIL 区域，等待 tail 通信完成
```

#### 时序图

```
AIC:
  ┌─Init─────────┐┌─HEAD(本地MM)──────┐┌─MAIN r0(远端MM)─┐┌─MAIN r1──┐┌─TAIL──┐
  │ AG提交x3     ││ A_self x B        ││ Wait->A_r0 x B   ││ Wait->... ││ Wait  │
  └──────────────┘└───────────────────┘└─────────────────┘└──────────┘└───────┘
        │                ↕ 并行               │                  │
        │                ↕ 掩盖               │                  │
HCCL:   ├─Scale(全量)─┐┌─DataHead(repeat)──┐┌─DataTail(repeat)─┐
        │ (小,快)      ││ (head 所有 tile)  ││ (tail 所有 tile)  │
        └──────────────┘└───────────────────┘└───────────────────┘
                         ↑                              ↑
                    HEAD MM 掩盖                   MAIN MM 掩盖
                    DataHead 通信                  DataTail 通信
```

- HEAD 本地计算（不依赖通信）直接掩盖 Scale + DataHead 通信
- MAIN r0 的 WaitTile 仅在 DataHead 未完成时阻塞，否则立即返回
- MAIN r0 → MAIN r1 → ... 逐 tile 流水，后续 tile 通信被前序计算掩盖
- TAIL 的 WaitTile 等待 DataTail，被 MAIN 尾部计算掩盖

#### 掩盖效果

| 通信阶段 | 掩盖来源 | 预期效果 |
|----------|----------|----------|
| Scale（全量） | HEAD 本地 MM 头开销 | 完全掩盖（scale 数据量小） |
| DataHead | HEAD 本地 MM 全部 | 大部分掩盖（取决于 HEAD 计算量 vs DataHead 通信量） |
| DataTail | MAIN 尾部 MM + TAIL 等待 | 部分掩盖 |

**最优条件**：HEAD 计算时间 >= DataHead 通信时间时，通信开销被完全掩盖。

**退化条件**：M 很小（HEAD 计算量不足）时，通信无法完全掩盖，WaitTile 会阻塞。

#### AIC-only 优势

全 C 核执行是本方案的重要优势：通信下发、计算、per-tile wait 均在 AIC 核内完成，无需 AIV 参与，省去了 AIV↔AIC 跨核 flag 同步开销。HCCL CCU 硬件内部有自身的流水调度，可进一步重叠通信与计算。

---

## 5. 各模块改动说明

### 5.1 Tiling Data（新增）

**文件**：`all_gather_mx_matmul_ccu_tiling_data.h`

**基于**：`all_to_all_matmul_tiling_data.h` 的 `ccuAllToAllMatmulTilingData`

```cpp
struct ccuAllGatherMxMatmulTilingData {
    Mc2InitTiling mc2InitTiling;       // HCCL 初始化参数
    Mc2CcTiling mc2CcTiling;           // HCCL CC task 配置
    CommTilingData commTile;           // Data AllGather 切分参数
    CommTilingData scaleCommTile;      // Scale AllGather 切分参数
    QuantMatmulTilingData mmTile;      // Matmul tiling 参数
};
```

**与现有方案 tiling 的差异**：
- 新增 `mc2InitTiling` / `mc2CcTiling`（CCU HCCL 初始化必需，与 `all_gather_quant_bmm.h` 中 `Mc2Tiling::AllGatherMatmulTilingDataFp8` 的字段对应）
- `commTile` 和 `scaleCommTile` 分离（现有方案中 scale 和 data 混在一次通信中，这里独立切分）
- 不需要 `localMatmul` 字段：FragmentTensor kernel 的 HEAD 区域天然是 local-first 语义，不需要独立的 LOCAL/REMOTE 模式

### 5.2 Kernel（复用改造）

**文件**：`qmm_mx_kernel_ag_udma.h`（已有文件，修改）

**策略**：不新建 kernel 文件，复用已有的 `QmmMxKernelAgUdma`，为其新增 `CommPolicy` 模板参数：

```cpp
template <typename AType, typename BType, typename CType, typename CommPolicy = UdmaCommWaitPolicy>
class QmmMxKernelAgUdma {
    CommPolicy commPolicy_;
public:
    CommPolicy& GetCommPolicy() { return commPolicy_; }
    // ...
};
```

改动点：

| 改动 | 改动前 | 改动后 |
|------|--------|--------|
| 模板参数 | `<AType, BType, CType>` | `<AType, BType, CType, CommPolicy = UdmaCommWaitPolicy>` |
| 通信等待 | `CrossCoreWaitFlag<0x2, PIPE_MTE2>(idx)`（2 处：主循环 + 收尾循环） | `commPolicy_.WaitTile(idx)`（2 处） |
| 新增成员 | 无 | `CommPolicy commPolicy_` + `GetCommPolicy()` |

**不变部分**：FragmentTensor 构建、ResolveTileCtx 区域调度、BlockScheduler、BlockMmadFragC 计算、L2 Cache 优化、B 矩阵共享处理。

**UDMA 兼容性**：默认模板参数 `UdmaCommWaitPolicy` 保持原有 UDMA 行为不变，现有 UDMA 调用方无需修改。

**UdmaCommWaitPolicy 安置**：`UdmaCommWaitPolicy` 当前定义在 `all_to_all_mx_quant_matmul_udma_impl.h` 中，`all_gather_mx_matmul_udma_impl.h` 未包含此头文件。修改 `qmm_mx_kernel_ag_udma.h` 时需在文件内内联定义默认 `UdmaCommWaitPolicy`，或将该类型提取到共享头文件。

### 5.3 Impl（新增）

**文件**：`all_gather_mx_matmul_ccu_impl.h`

**基于**：`all_to_all_mx_quant_matmul_hcomm_impl.h`（CCU 通信风格）+ `all_gather_mx_matmul_udma_impl.h`（AllGather 语义 + FragmentTensor kernel 调用）

**新增组件**：HcommCommState（详见第 3.1 节）、HcommCommWaitPolicy（详见第 3.1 节）、AllGatherMxMatmulCcuImpl。

**Init 流程**：
1. 解析 tiling（tileCnt, tileM, tailCnt, tailM, k, n）
2. HCCL dataType 映射：scale → `HCCL_DATA_TYPE_FP8E8M0`，data → `GetHcclDataType<AType>()`（e4m3/e5m2 自动映射）
3. `commState_.hccl_.InitV2(GetHcclContext<0>(), &mc2InitTiling)`
4. `commState_.hccl_.SetCcTilingV2(offsetof(..., mc2CcTiling))`
5. 获取 rankId, rankSize
6. 计算 Gather 缓冲区地址（workspaceGM 偏移）
7. 批量提交 AllGather（scale → data head → data tail）
8. 绑定 `commPolicy_.state_ = &commState_`
9. 构造 kernel params（FragmentParams + GM 指针 + win 基址）

**Run 流程**：
1. `quantMatmulKernelImpl_(params)` — FragmentTensor kernel 执行
2. `SyncAll()`
3. `commState_.hccl_.Finalize()`

### 5.4 Kernel Entry（新增）

```cpp
__global__ __aicore__ void AllGatherQuantMatmulCcuKernel(
    GM_ADDR aGM, GM_ADDR aScaleGM,
    GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM,
    GM_ADDR workspaceGM,
    ccuAllGatherMxMatmulTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_0);  // AIC-only
    AllGatherQuantMatmulImpl::AllGatherMxMatmulCcuImpl<
        fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t> impl(&tilingData);
    impl.Init(aGM, aScaleGM, bGM, bScaleGM, cGM, workspaceGM);
    impl.Run();
}
```

> Kernel entry 中的模板参数（A/B 类型）由上层 dispatch 根据输入 dtype 选择，支持 fp8_e4m3fn_t / fp8_e5m2_t 任意组合。Scale 固定为 fp8_e8m0_t，C 固定为 bfloat16_t。
>
> Context 传递方式见「设计决策记录」节：在 impl Init 内调用 `GetHcclContext<0>()`（已验证可行）。

### 5.5 改动汇总

| 文件 | 操作 | 说明 |
|------|------|------|
| `all_gather_mx_matmul_ccu_tiling_data.h` | 新增 | CCU tiling 结构体 |
| `qmm_mx_kernel_ag_udma.h` | 修改 | 新增 CommPolicy 模板参数，默认值保持 UDMA 兼容 |
| `all_gather_mx_matmul_ccu_impl.h` | 新增 | CCU impl（HcommCommState + HcommCommWaitPolicy + AllGatherMxMatmulCcuImpl + kernel entry） |
| `design.md` | 新增 | 本设计文档 |

**不改动**的已有文件：
- `all_gather_mx_matmul_udma_impl.h`
- `all_gather_mx_matmul_udma_tiling_data.h`

---

## 6. 参考资料

| 文件 | 用途 |
|------|------|
| `all_gather_quant_bmm.h` | 现有方案：CCU AllGather + 量化矩阵乘的现有实现，通信提交、计算流程参考 |
| `all_gather_mx_matmul_udma_impl.h` | AllGather 语义参考：FragmentTensor kernel 调用、HEAD/MAIN/TAIL 区域调度、win buffer 布局 |
| `qmm_mx_kernel_ag_udma.h` | Kernel 复用基底：FragmentTensor 构建、ResolveTileCtx、BlockScheduler 集成 |
| `all_to_all_mx_quant_matmul_hcomm_impl.h` | CCU 通信风格参考：HcommCommState、HcommCommWaitPolicy、InitV2/SetCcTilingV2、批量提交+batch wait |
| `all_to_all_mx_quant_matmul_udma_impl.h` | UDMA 同步对照参考：UdmaCommWaitPolicy（CrossCoreWaitFlag）vs HcommCommWaitPolicy（hccl_.Wait），CommPolicy 策略模式解耦（详见第 3.2 节） |
| `all_to_all_matmul_tiling_data.h` | CCU tiling 结构参考：ccuAllToAllMatmulTilingData（Mc2InitTiling + Mc2CcTiling） |
| `all_gather_matmul_fp16_bf16.h` | AllGather API 用法参考：hccl_.AllGather<true> 签名、stride/repeat 参数、handle 数组管理 |
| `quant_matmul_mx_kernel.h` | 标准 QuantMatmulMxKernel 参考：CommPolicy 注入模式、LOCAL/REMOTE/DEFERRED_SYNC 模式 |
| `quant_matmul_tiling_data.h` | QuantMatmulTilingData 字段定义 |
| `comm_tiling_data.h` | CommTilingData 字段定义 |
| `all_gather_matmul_v2_apt.cpp` | Kernel entry 参考：GetHcclContext<0>() 用法、KERNEL_TASK_TYPE_DEFAULT、dtype dispatch |

---

## 设计决策记录

### Context 传递方式

**已确认**：在 impl 的 Init 内调用 `GetHcclContext<0>()`（参考 `all_to_all_mx_quant_matmul_hcomm_impl.h` 行 177，已验证可行）。kernel entry 不传 context 参数，impl 内部自行获取。

### AllGather repeat > 1 验证

**说明**：AllGather 批量提交（`repeat = tileCnt`）为新用法，现有 AllGather 实现（`all_gather_quant_bmm.h`、`all_gather_matmul_fp16_bf16.h`）均使用 `repeat = 1` + 手动循环。AlltoAll 的 repeat 批量提交已验证可行，AllGather 的 repeat 语义需在实现阶段通过单测验证 stride 和地址推进与 AlltoAll 一致。


