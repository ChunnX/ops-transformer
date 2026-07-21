# QuantGroupedMatMulAlltoAllv 算子 Permute 消除优化方案

## 一、算子概述

### 1.1 算子功能

`QuantGroupedMatMulAlltoAllv` 是 MoE（混合专家）场景下的融合算子，运行在 Ascend 950PR/950DT（arch35）上，采用**先计算后通信**模式。

核心数据流：
- 路由专家：`gmmX @ gmmWeight → gmmY → Unpermute → AlltoAllv → y`
- 共享专家：`mmX @ mmWeight → mmY`（与路由专家并行）

### 1.2 关键参数

| 参数 | 说明 |
|------|------|
| gmmX | shape (A, H1)，A 为 sendCounts 之和 |
| gmmWeight | shape (e, H1, N1)，e 为单卡专家数 |
| y | shape (BSK, N1)，BSK 为 recvCounts 之和 |
| epWorldSize | 通信域大小，支持 2/4/8/16/32/64/128/256 |
| sendCounts/recvCounts | 按 `sendCounts[fromRank][expertId]` 一维展开 |
| expertNum | 每次合并计算的专家数量（影响 permute 触发） |

### 1.3 代码结构

```
quant_grouped_mat_mul_allto_allv/
├── op_kernel/
│   ├── quant_grouped_mat_mul_allto_allv.cpp          # Kernel 入口
│   └── arch35/quant_grouped_mat_mul_allto_allv_tiling.h  # TilingData 定义
├── op_host/
│   └── op_tiling/arch35/
│       ├── quant_grouped_mat_mul_allto_allv_tiling_common.cpp  # Tiling 通用逻辑
│       ├── tt_quant_grouped_mat_mul_allto_allv_tiling.cpp      # Pertensor 量化 Tiling
│       └── mx_quant_grouped_mat_mul_allto_allv_tiling.cpp      # MX 量化 Tiling
├── allto_allv_quant_grouped_mat_mul/op_kernel/mc2_templates/
│   ├── scheduler/gmm_a2av_scheduler.h               # 后通信模式 Scheduler
│   ├── communication/hccl_a2av_op.h                  # HCCL AlltoAllV 封装
│   ├── communication/a2av_permute_engine.h           # Permute 实现
│   ├── communication/a2av_comm_params.h              # 通信参数计算
│   └── compute/quant_grouped_matmul.h                # GMM 计算封装
└── 3rd/grouped_matmul/op_kernel/
    ├── gqmm_cube_on_the_fly.h                        # GMM Cube 底层实现
    └── quant_block_sch.h                             # GMM Block 调度与地址累积
```

---

## 二、Permute 的产生原因与耗时分析

### 2.1 数据布局矛盾

```
gmmX (expert-first, 内部rank-first):
  [exp0: rank0_toks | rank1_toks | ... | exp1: rank0_toks | rank1_toks | ...]

     ↓ GMM (groupList按expert分组, groupNum=e)

gmmOutput (expert-first, 内部rank-first):
  [exp0: rank0_out | rank1_out | ... | exp1: rank0_out | rank1_out | ...]

     ↓ PermuteSend (expert-first → rank-first)   ← 要消除的步骤

sendBuffer (rank-first, 内部expert-first):
  [rank0: exp0_out | exp1_out | ... | rank1: exp0_out | exp1_out | ...]

     ↓ AlltoAllv (sendCnt[i]/sendOffset[i] 按rank定位)
```

**根因**：AlltoAllv 的 `sendOffset[i]` 要求同一 rank 的数据在 sendBuffer 中连续。GMM 按 expert 分组输出，同一 rank 的数据分散在各 expert 区域中，必须 permute 重排。

### 2.2 Permute 触发条件

- 代码位置：`gmm_a2av_scheduler.h:39-56`
- 触发条件：`expertNum > 1`（多专家批量计算时）
- 当 `expertNum == 1` 时直接用 GMM 输出作为 sendBuffer，跳过 permute

### 2.3 Permute 实现与耗时瓶颈

Permute 实现（`a2av_permute_engine.h`）通过 GM→UB→GM 的 DataCopyPad 双缓冲搬运，对每个 (rank, expert) 数据块执行重排。

**真实瓶颈不在数据搬运本身，而在 AIC-AIV 串行同步**：

```
每轮 (start += expertNum):
  AIC:  GMM计算 → SetFlag(8) → WaitFlag(8) → SetFlag(9)   ← AIC被阻塞!
  AIV:                        WaitFlag(9) → Permute → SyncAll → LaunchComm
```

- AIC 在 GMM 计算后执行 `WaitFlag(8)`，等待 AIV 完成 Permute + SyncAll + LaunchComm 后才能进入下一轮
- GMM 计算与 Permute 完全串行，AIC 空等 AIV 做 permute
- 每轮 `SyncAll<true>()` 全核 barrier，负载不均时快核空等慢核
- 轮间无重叠，第 i 轮通信与第 i+1 轮计算没有流水

### 2.4 已验证无效的方案

| 方案 | 失败原因 |
|------|---------|
| 单专家粒度 AlltoAllv | HCCL 每次调用有固定通信头开销，逐专家调用导致开销线性累积 |
| 增大 UB 缓冲区 | Permute 瓶颈不在 tile 迭代次数，数据搬运本身非主要矛盾 |

---

## 三、消除 Permute 方案设计

### 3.1 核心思路

将 GMM 的 group 从 expert 粒度细化为 (expert, rank) 粒度，修改输出地址累积逻辑，使 GMM 直接输出 rank-first 布局，从根源消除 expert-first 与 rank-first 的布局矛盾。

### 3.2 GMM 底层 group 地址机制（技术基础）

GMM 底层（`gqmm_cube_on_the_fly.h` + `quant_block_sch.h`）的地址计算机制：

**group 遍历与地址累积**（`gqmm_cube_on_the_fly.h:283-314`）：
```cpp
for (uint32_t groupIdx = 0; groupIdx < groupNum_; ++groupIdx) {
    SetMNK(groupIdx, mSize, nSize, kSize);           // 从groupList读取第groupIdx组的M
    block_.UpdateGroupOffset(m, n, k, groupIdx);      // 累积偏移
    UpdateMMGlobalAddr(groupIdx);                      // 设置GM地址
    // ... Cube计算，输出写入 yGlobal[offsetC]
}
```

**输出地址累积**（`quant_block_sch.h:145`）：
```cpp
params_.cGroupAddrOffset += params_.m * params_.n;  // 按group顺序连续累积 → expert-first
```

**输入地址累积**（`quant_block_sch.h:132`）：
```cpp
params_.aGroupAddrOffset += params_.m * params_.k;  // 按group顺序连续读取gmmX
```

**Weight 偏移累积**（`quant_block_sch.h:142`）：
```cpp
params_.bGroupAddrOffset += params_.n * params_.k;  // 每个group累积一次
```

### 3.3 方案详细设计

#### 步骤 1：groupList 按 (expert, rank) 细化

当前 groupList（`quant_grouped_matmul.h:153-155`）：
```cpp
// groupNum = e, 每个 group = 一个 expert 的所有 rank token
groupList = [expert0_total_tokens, expert1_total_tokens, ..., expertE_total_tokens]
```

改为：
```cpp
// groupNum = e * rankDim, 每个 group = 一个 (expert, rank) 的 token
// gmmX 是 expert-first 内部 rank-first，连续切分即可正确匹配
groupList = [exp0_rank0_cnt, exp0_rank1_cnt, ..., exp0_rankR_cnt,
             exp1_rank0_cnt, exp1_rank1_cnt, ..., exp1_rankR_cnt,
             ...]
```

gmmX 的排列（expert-first，内部 rank-first）与这个 groupList 天然匹配，GMM 连续切分 gmmX 即可正确分组。

#### 步骤 2：Weight 偏移跨 expert 边界判断

group 数从 `e` 增至 `e × rankDim`，但 weight 只有 `e` 个。需同一 expert 的不同 rank group 复用同一 weight。

修改 `quant_block_sch.h` 的 `UpdateGroupOffset`：
```cpp
// 当前: params_.bGroupAddrOffset += params_.n * params_.k;
// 改为: 仅在跨 expert 边界时累积 weight 偏移
if (groupIdx > 0 && (groupIdx % rankDim == 0)) {
    params_.bGroupAddrOffset += params_.n * params_.k;  // 跨expert时才累积
}
```

weight scale 偏移 `wScaleGroupAddrOffset` 需同步修改。

pertensor 模式下 weight scale 索引（`gqmm_cube_on_the_fly.h:150`）需改为按 `groupIdx / rankDim` 索引。

#### 步骤 3：输出地址按 rank-first 布局重映射（核心改动）

当前输出地址连续累积（`quant_block_sch.h:145`）：
```cpp
params_.cGroupAddrOffset += params_.m * params_.n;  // 连续累积 → expert-first
```

改为按 rank-first 布局计算偏移：

```
对于 group(expertIdx, rankIdx)，其在 rank-first 布局中的偏移：
  cGroupAddrOffset = rankBaseOffset[rankIdx] + expertOffsetInRank[rankIdx][expertIdx]
```

其中：
- `rankBaseOffset[r]` = rank 0~r-1 的所有 expert token 总数 × N1
- `expertOffsetInRank[r][e]` = rank r 中 expert 0~e-1 的 token 总数 × N1

实现方式：在 kernel 中从 groupList/sendCounts 动态计算，或由 tiling 阶段预计算偏移表传入。

#### 步骤 4：Scheduler 保持多轮流水，移除 permute

```cpp
// gmm_a2av_scheduler.h Process() 修改后
__aicore__ inline void Process() {
    uint32_t e = taskTilingInfo_->e;
    uint32_t expertNum = taskTilingInfo_->expertNum;

    for (uint32_t start = 0; start < e; start += expertNum) {
        uint32_t actualExpertNum = (start + expertNum > e) ? (e - start) : expertNum;
        if ASCEND_IS_AIC {
            computeOp_.Process(start, actualExpertNum);
            AscendC::CrossCoreSetFlag<0, PIPE_FIX>(8);
            AscendC::CrossCoreWaitFlag(8);
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(9);
        }
        if ASCEND_IS_AIV {
            AscendC::CrossCoreWaitFlag(9);
            // 移除 PermuteSendAfterCompute 和 SyncAll
            hcclOp_.LaunchCommData(start, actualExpertNum);  // 异步, 只发这批expert的数据
        }
    }

    if (IsNeedMM) {
        shareComputeOp_.Process(0, 1);
    }

    for (uint32_t start = 0; start < e; start += expertNum) {
        hcclOp_.Wait(start);  // 等每轮通信完成
    }
    this->End();
}
```

与当前代码相比，仅删除 permute 和 SyncAll 两行，循环结构和多轮通信完全保留。

#### 步骤 5：通信参数计算改为 rank-first 布局

`a2av_comm_params.h` 的 `CalcA2avCommAfterParams` 修改 offset 计算：

```cpp
// 修改前 (expert-first布局, sendOffset按expert连续累积):
params.sendOffset[0] = expertOffset * axis;
for (i = 1; i < rankDim; i++) {
    params.sendOffset[i] = params.sendOffset[i-1];
    for (expertIdx = start; expertIdx < start+expertNum; expertIdx++) {
        params.sendOffset[i] += rawSendCounts[expertIdx + (i-1)*e] * axis;
    }
}

// 修改后 (rank-first布局, sendOffset按rank区域定位):
for (i = 0; i < rankDim; i++) {
    params.sendOffset[i] = (rankBaseOffset[i] + batchExpertOffsetInRank[i]) * axis;
    // batchExpertOffsetInRank[i] = rank i 中 exp 0~start-1 的 token 总数
    // 每轮结束后累加 actualExpertNum 对应的 token 数
}
```

`rankBaseOffset` 和 `batchExpertOffsetInRank` 可在 kernel 中动态计算（从 sendCounts 前缀和推导），或由 tiling 阶段预计算传入。

#### 步骤 6：Kernel 层 sendBuffer 直接用 gmmOutput

```cpp
// quant_grouped_mat_mul_allto_allv.cpp:115
// 修改前: expertNum > 1 时用 permuteSendGM
// 修改后: 永远用 gmmOutputGM (已是rank-first布局)
GM_ADDR sendBufferAddr = gmmOutputGM;
```

#### 步骤 7：Tiling 层

```cpp
// wsPermuteOutSize = 0, 节省 workspace
localTilingData_.workspaceInfo.wsPermuteOutSize = 0;
```

### 3.4 数据流验证

```
gmmX: [exp0: r0, r1, ... | exp1: r0, r1, ... | ...]
        ↓ groupList = [exp0_r0, exp0_r1, ..., exp1_r0, exp1_r1, ...]
        ↓ group 0(exp0,r0) → 输出写到 rank0区域开头
        ↓ group 1(exp0,r1) → 输出写到 rank1区域开头
        ↓ group R(exp1,r0) → 输出写到 rank0区域exp0之后
        ↓ group R+1(exp1,r1) → 输出写到 rank1区域exp0之后
        ↓ ...
gmmOutput: [rank0: exp0, exp1, ... | rank1: exp0, exp1, ... | ...]  ← rank-first!
        ↓ AlltoAllv 直接使用，无需 permute
yGM
```

### 3.5 流水时序对比

**当前（有 permute）**：
```
AIC:  |GMM_0 |Wait|GMM_1 |Wait|GMM_2 |
AIV:        |Perm_0|SyncAll|Launch0|Perm_1|SyncAll|Launch1|Perm_2|SyncAll|Launch2|
HCCL:                              |====Comm0====|====Comm1====|====Comm2====|
```
AIC 被 permute+SyncAll 阻塞，计算与通信重叠受限。

**消除 permute 后**：
```
AIC:  |GMM_0 |GMM_1   |GMM_2   |
AIV:        |Launch0  |Launch1 |Launch2  |Wait0|Wait1|Wait2|
HCCL:       |==Comm0==|==Comm1==|==Comm2==|
```
- AIC 连续计算，不被 permute 阻塞
- AIV 每轮只做轻量 LaunchComm（微秒级），立即让 AIC 进入下一轮
- 通信仍多轮异步，与后续计算重叠

---

## 四、改动点汇总

| 文件 | 改动内容 |
|------|---------|
| `quant_block_sch.h:124-164` | `UpdateGroupOffset`：cGroupAddrOffset 改为按 rank-first 布局计算；bGroupAddrOffset/wScaleGroupAddrOffset 增加跨 expert 边界判断 |
| `quant_grouped_matmul.h:126-132` | `expertTokenNum_` 计算改为按 (expert, rank) 细化 |
| `quant_grouped_matmul.h:151-155` | groupList 填充改为 (expert, rank) 粒度 |
| `quant_grouped_matmul.h:191` | `localQuantParams.groupNum = e * rankDim` |
| `gmm_a2av_scheduler.h:39-57` | 移除 PermuteSendAfterCompute 和 SyncAll |
| `a2av_comm_params.h:66-104` | `CalcA2avCommAfterParams` 改为 rank-first 布局 offset 计算 |
| `quant_grouped_mat_mul_allto_allv.cpp:115` | sendBuffer 永远用 gmmOutputGM |
| `quant_grouped_mat_mul_allto_allv_tiling_common.cpp:878` | `wsPermuteOutSize = 0` |
| tiling 层 | 预计算 rank-first 偏移表或 kernel 中动态计算 |

---

## 五、关键技术风险与应对

### 5.1 group 数量增加导致 Cube 利用率下降

group 从 `e` 增至 `e × rankDim`，每个 group 的 M（token 数）更小，Cube 单次计算的 M tile 可能不饱满。

**应对**：
- GMM 底层有 `CalcTailTile`（`gqmm_cube_on_the_fly.h:220-254`）机制，最后一个 group 会自动切分以利用更多核
- MoE 场景中 e 通常 4~32，rankDim 通常 2~8，group 数 8~256，在 GMM 支持范围内
- 如果某些 (expert, rank) 的 token 数为 0，GMM 会跳过（`gqmm_cube_on_the_fly.h:290-292`：`if (mSize <= 0) continue`）

### 5.2 weight 偏移修改的复杂性

`bGroupAddrOffset` 的累积逻辑改动涉及 weight 和 weight scale 两个维度，需确保 pertensor 和 mx 两种量化模式都正确。

**应对**：
- pertensor 模式：weight scale 是标量，按 `groupIdx / rankDim` 索引
- mx 模式：weight scale 偏移 `wScaleGroupAddrOffset` 需同样增加跨 expert 边界判断

### 5.3 tiling 阶段偏移表预计算

rank-first 偏移表大小 = `e × rankDim` 个 uint64，需加入 tilingData 或 workspace。

**应对**：
- 偏移表可通过 groupList 间接推导，无需额外存储——在 kernel 中动态计算
- 或在 workspace 中预计算（类似当前 groupList 的处理方式）

---

## 六、方案对比

| 维度 | 当前（有 permute） | 消除 permute + 多轮流水 |
|------|-------------------|----------------------|
| Permute | 每轮 GM→UB→GM 搬运 | **消除** |
| SyncAll | 每轮一次 | **消除** |
| 通信次数 | CeilDiv(e, expertNum) | CeilDiv(e, expertNum)（不变） |
| 计算通信流水 | 有，但受 permute 阻塞 | **有，且更充分**（AIC 不被阻塞） |
| 额外 workspace | A×N1×dtypeSize | **0** |
| GMM group 数 | e | e×rankDim |
| Cube 利用率 | 高 | 中（group 变小，有 tail tile 兜底） |

---

## 七、总结

消除 permute 方案从根源解决了 expert-first 与 rank-first 的布局矛盾：

1. **GMM group 细化**：从 expert 粒度到 (expert, rank) 粒度，gmmX 的排列天然匹配
2. **输出地址重映射**：`cGroupAddrOffset` 按 rank-first 布局计算，GMM 直接输出 rank-first
3. **Weight 偏移条件累积**：同一 expert 内的 group 复用同一 weight
4. **多轮流水保持**：通信仍按 expertNum 分批异步，与后续计算重叠
5. **AIC 不再被阻塞**：移除 permute 和 SyncAll，AIC 连续计算，流水更充分
