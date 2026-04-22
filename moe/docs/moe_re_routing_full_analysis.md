# moe_re_routing 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeReRouting`
**算子类**: MoE Token 重排，AlltoAll 后处理算子
**功能描述**: 在分布式 MoE 推理/训练中，AlltoAll 通信完成后，当前卡上收到的 token 按 **rank(卡)序→expert(专家)序** 排列，而后续计算需要按 **expert(专家)序→rank(卡)序** 排列。本算子完成这种二维矩阵视角的行列转置重排：将按 `[rank][expert]` 布局的 token 重排为按 `[expert][rank]` 布局的 token。

**核心计算公式**:
- **SrcOffset** (rank-major): `sum_{r<cur_rank, all e}(count(r,e)) + sum_{e<cur_expert}(count(cur_rank,e))`
- **DstOffset** (expert-major): `sum_{all r, e<cur_expert}(count(r,e)) + sum_{r<cur_rank}(count(r,cur_expert))`
- **expert_token_num[e]**: `sum_{all r}(count(r,e))`

**支持平台**: ascend910b, ascend910_93, ascend950, kirinx90, kirin9030
**支持数据类型**: FP16/BF16/INT8（通用）; FP8_E5M2/FP8_E4M3FN（ascend950）; expert计数: INT32/INT64

---

## 2. 目录结构与文件职责

```
moe_re_routing/
├── CMakeLists.txt, README.md
├── op_host/
│   ├── moe_re_routing_def.cpp                 # 算子定义（3输入/4输出/2属性）
│   ├── moe_re_routing_tiling.h                # Tiling数据结构 + CompileInfo
│   ├── moe_re_routing_tiling.cpp              # Tiling入口（注册+Prepare）
│   ├── moe_re_routing_tiling_base.h/.cpp      # Tiling基类（校验/Shape解析）
│   ├── moe_re_routing_membase_tiling.h/.cpp   # membase策略（910b/910_93）
│   ├── moe_re_routing_re_tiling.h/.cpp        # Re策略（rank×expert二维分核）
│   ├── moe_re_routing_r_tiling.h/.cpp         # R策略（仅rank分核）
│   ├── moe_re_routing_infershape.cpp          # Shape推导
│   └── config/ {ascend910b, ascend910_93, ascend950, kirinx90, kirin9030}
├── op_kernel/
│   ├── moe_re_routing.h                       # membase kernel模板类
│   ├── moe_re_routing.cpp                     # membase kernel入口（6分支）
│   ├── moe_re_routing_apt.cpp                 # regbase kernel入口（6分支）
│   └── arch35/
│       ├── platform.h                         # 平台适配
│       ├── moe_re_routing_re_regbase.h        # Re模式kernel（rank+expert二维）
│       └── moe_re_routing_r_regbase.h         # R模式kernel（仅rank分核）
├── op_graph/
│   └── moe_re_routing_proto.h                 # 图模式原型
├── examples/ test_geir_moe_re_routing.cpp
└── tests/ut/ {op_host, op_kernel}
```

---

## 3. Host 侧分析

### 3.1 算子定义 (moe_re_routing_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | tokens | REQUIRED | FP16/BF16/INT8(+FP8) | 2D: [A, H] |
| 输入 | expert_token_num_per_rank | REQUIRED | INT32/INT64 | 2D: [N, E] |
| 输入 | per_token_scales | OPTIONAL | FLOAT(FP8:FLOAT8_E8M0) | 1D/2D: [A]或[A,S] |
| 输出 | permute_tokens | REQUIRED | 同 tokens | 2D: [A, H] |
| 输出 | permute_per_token_scales | REQUIRED | 同 scales | 同 per_token_scales |
| 输出 | permute_token_idx | REQUIRED | INT32 | 1D: [A] |
| 输出 | expert_token_num | REQUIRED | 同 expert输入 | 1D: [E] |

**属性**:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| expert_token_num_type | Int | 1 | 输出模式: 0=cumsum(暂不支持), 1=count |
| idx_type | Int | 0 | 索引类型: 0=gather, 1=scatter |

**SoC 配置差异**:
- **ascend910b/910_93**: opFile="moe_re_routing"(membase)，6种dtype组合
- **ascend950**: opFile="moe_re_routing_apt"(regbase)，14种dtype组合(含FP8)
- **kirinx90/kirin9030**: 独立 `GetKirinCoreConfig()`，4种组合(FP16/INT8)

### 3.2 Tiling 策略

#### 3.2.1 TilingData 结构（tiling.h）

**MoeReRoutingTilingData**（第30-38行，membase kernel使用）:
```
├── coreNum          : int64_t   # 使用核数
├── ubFactor         : int64_t   # UB每次处理token数
├── tokensNum        : int64_t   # token总数(A)
├── tokensSize       : int64_t   # token长度(H)
├── rankNum          : int64_t   # rank总数(N)
├── expertNumPerRank : int64_t   # 每rank的expert数
└── hasScale         : int64_t   # 是否有scale
```

**MoeReRoutingReTilingData**（第42-54行，Re模式）:
```
├── tokenSize     : int64_t   ├── blockNumR      : int64_t
├── scaleSize     : int64_t   ├── blockFactorR   : int64_t
├── rankNum       : int64_t   ├── blockNumE      : int64_t
├── expertNum     : int64_t   ├── blockFactorE   : int64_t
├── coreNum       : int64_t   ├── ubFactor       : int64_t
└── idxType       : int64_t
```

**MoeReRoutingRTilingData**（第60-69行，R模式）:
```
├── tokenSize   : int64_t   ├── coreNum      : int64_t
├── scaleSize   : int64_t   ├── blockFactor   : int64_t
├── rankNum     : int64_t   ├── ubFactor      : int64_t
├── expertNum   : int64_t   └── idxType       : int64_t
```

**CompileInfo**（第83-87行）: `coreNum`, `ubSize`, `socVersion`。

#### 3.2.2 三策略注册与优先级

| 优先级 | 类名 | 注册值 | 适用条件 |
|--------|------|--------|----------|
| 1 (最高) | MoeReRoutingReTiling | 10000 | regbase SoC 且 rankNum<5 且 expertNum!=1 |
| 2 | MoeReRoutingRTiling | 11000 | regbase SoC (Re不适用时的回退) |
| 3 (最低) | MoeReRoutingTiling | 20000 | 始终 capable (membase兜底) |

TilingRegistry 按优先级从高到低尝试，第一个 `IsCapable() == true` 的策略被采用。

#### 3.2.3 membase 策略 (membase_tiling.cpp)

适用: ascend910b, ascend910_93。

- **IsCapable()**: 始终 true（兜底）
- **DoOpTiling()**（第153-193行）: 计算 ubFactor（预留 12288 字节，双缓冲除以 2），按 rank 分核
- **约束**: idx_type 必须为 0（不支持 scatter）
- **TilingKey**: `100000`(INT8+INT32), `100010`(INT8+INT64), `100100`(FP16+INT32), `100110`(FP16+INT64), `100200`(BF16+INT32), `100210`(BF16+INT64)

#### 3.2.4 Re 策略 (re_tiling.cpp)

适用: ascend950, kirinx90, kirin9030（regbase）。

- **IsCapable()**: `IsRegbaseSocVersion() && rankNums_ < 5 && expertNum_ != 1`
- **SplitCore()**（第29-45行）: rank+expert二维分核 — 先按rank分，剩余核分配给expert维
- **SplitUb()**: 预留 `INDEX_UB_SIZE(256)*2*sizeof(expertDtype)` 字节索引空间
- **TilingKey**: `200000`(无scale), `200100`(FLOAT scale), `200200`(FP8/E8M0 scale)
- **Workspace**: 16 MB

#### 3.2.5 R 策略 (r_tiling.cpp)

适用: ascend950, kirinx90, kirin9030（regbase，Re不适用时的回退）。

- **IsCapable()**: `IsRegbaseSocVersion()`（regbase平台均可）
- **SplitCore()**: 仅 rank 维度分核
- **TilingKey**: `210000`(无scale), `210100`(FLOAT scale), `210200`(FP8/E8M0 scale)
- **Workspace**: 16 MB

#### 3.2.6 Tiling 基类 (tiling_base.cpp)

`MoeReRoutingTilingBase` 继承 `TilingBaseClass`，Re和R策略的公共基类。

- **GetShapeAttrsInfo()**（第224-271行）: 解析 tokens [A,H], expert [N,E], scale, 属性
- **CheckParam()**（第201-222行）: 校验 dtype 组合、维度合法性
- **CheckDtypeAndAttr()**（第91-147行）: 支持 INT8/BF16/FP16/FP8_E4M3FN/FP8_E5M2 × INT32/INT64，FP8 时 scale 为 FLOAT8_E8M0

### 3.3 Shape 推导 (moe_re_routing_infershape.cpp)

| 输出 | 形状 | dtype |
|------|------|-------|
| permute_tokens | 同 tokens [A, H] | 同 tokens |
| permute_per_token_scales | 同 per_token_scales 或 [A] | 同 scales / FLOAT |
| permute_token_idx | [A] | INT32 |
| expert_token_num | [E] | 同 expert_token_num_per_rank |

支持 UnknownRank (-2) 和动态 Shape (-1) 输入。

### 3.4 ACLNN API

无独立 ACLNN 两段式 API 文档。通过 L0 接口直接调用（GEIR 模式）。算子定义中声明 `aclnnSupport` 支持 ACLNN 调用。

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与初始化

两套独立 Kernel 实现，按 SoC 架构分发：

```
moe_re_routing (算子入口)
├── moe_re_routing.cpp (membase: 910b/910_93)
│   └── KernelMoeReRouting<T1, T2, TScale> (单文件模板类)
└── moe_re_routing_apt.cpp (regbase: 950/kirin)
    ├── MoeReRoutingReRegbase (Re模式: rank*expert二维)
    └── MoeReRoutingRRegbase (R模式: 仅rank分核)
```

**membase 入口**（moe_re_routing.cpp 第26-97行）: 6 个 TilingKey 分支，任务类型 `KERNEL_TYPE_AIV_ONLY`。

**regbase 入口**（moe_re_routing_apt.cpp 第22-119行）: 6 个 TilingKey 分支，使用 `DTYPE_TOKENS`/`DTYPE_EXPERT_TOKEN_NUM_PER_RANK` 编译时宏，AIC核直接返回。

### 4.2 核心计算流程

#### membase Kernel (moe_re_routing.h)

**模板参数**: `<T1(token类型), T2(expert计数类型), TScale(scale类型)>`

```
Process() (第85-146行):
  for 每个分配的 rank (interleaved by coreNum):
    计算 tokenTotalNumsSrc (rank-major累加)
    for 每个expert:
      tokenNums = srcExpertTokenGm(rank, expert)
      计算 tokenTotalNumsDst (expert-major累加)
      for UB分片:
        CopyIn(srcOffset) → ComputeAndCopyOutIndex(dstOffset, srcOffset)
        → CopyOut(dstOffset)
      tokenTotalNumsSrc += tokenNums
      补全 tokenTotalNumsDst (当前rank及之后的rank)
    if blockId==0: 计算 expert_token_num (列方向求和)
```

#### Re 模式 Kernel (arch35/moe_re_routing_re_regbase.h)

**模板参数**: `<T, TIndex, TScale, bool hasScales>`

```
PrepareParams() (第99-112行):
  rIdx_ = blockIdx_ % blockNumR; eIdx_ = blockIdx_ / blockNumR
  尾部核修正 rNum_/eNum_

Process() (第145-166行):
  for rIdx in [rIdxStart, rIdxStart+rNum_):
    CalSrcOffSet(rIdx, eIdxStart)
    for eIdx in [eIdxStart, eIdxStart+eNum_):
      CalDstOffSet(rIdx, eIdx)
      ProcessTokenScale(currTokenNum, tokenSize_, false)  // 搬移token
      if hasScales: ProcessTokenScale(currTokenNum, scaleSize_, true)
  CalcExpertTokenNum()
```

偏移计算:
- **CalSrcOffSet**（第115-127行）: rank-major累加到 (rIdx, eIdx)
- **CalDstOffSet**（第130-142行）: expert-major累加到 (eIdx, rIdx)

#### R 模式 Kernel (arch35/moe_re_routing_r_regbase.h)

与 Re 模式类似但仅按 rank 分核，每核遍历所有 expert:

```
Process() (第90-121行):
  for rankLoopIdx in [0, rankLoopNum):
    currRankId = rankLoopIdx * coreNum + blockIdx
    计算 tokensSrc_ (rank-major)
    for expertIdx in [0, expertNum):
      计算 tokensDst_ (expert-major)
      ProcessTokenScale(currTokenNum)
```

#### ProcessTokenScale 双路径（Re/R共用）

**路径1: UB拆分** (ubFactor < tokSclSize，单个token超UB容量):
```
逐token逐分片: CopyIn(分片) → CopyOut(分片)
索引单独 CopyOutIndex
```

**路径2: UB满载** (ubFactor >= tokSclSize):
```
tokenFactor = ubFactor / tokSclSize  // UB可容纳token数
批量: CopyIn → CopyOutIndex → CopyOut
```

### 4.3 数据结构与内存布局

**membase Kernel 数据成员**（moe_re_routing.h 第39-59行）:

| 成员 | 类型 | 用途 |
|------|------|------|
| `queBind` | `TQueBind` | token+scale数据队列(VECIN/VECOUT) |
| `idxOutQue` | `TQue` | 索引输出队列 |
| `srcOffset`, `tokenTotalNumsSrc` | 运行时变量 | 源端偏移追踪 |
| `tokenTotalNumsDst` | 运行时变量 | 目的端偏移追踪 |
| `ubLoop/ubTail` | 运行时变量 | UB循环分片 |
| `rankLoopNum/rankTailNum` | 运行时变量 | rank循环分片 |

**regbase Kernel 额外成员**: `rIdx_/eIdx_`(rank/expert索引), `rNum_/eNum_`(负责的rank/expert数), `tokensSrc_/tokensDst_`(源/目的偏移)。

### 4.4 多核并行策略

**membase**: 按 rank 分核（interleaved），核 `blockId` 处理 rank `blockId, blockId+coreNum, ...`。

**Re模式**: rank×expert二维分核。先按rank切 `blockNumR = ceil(N, blockFactorR)`，剩余核分配给expert维 `blockNumE = min(coreNum/blockNumR, E)`。总核数 = `blockNumR × blockNumE`。

**R模式**: 仅rank维度分核，每核处理所有expert。

### 4.5 通信模式

三种策略均无 `SyncAll`。各核独立处理自己负责的 (rank, expert) 组合，写入输出的不同区域，无写冲突。

`expert_token_num` 输出: membase 由核0计算（其他核跳过）；regbase 由所有核按步长 `coreNum` 并行计算不同 expert 行。

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

见 3.2.1 节三种 TilingData 结构。Host 侧 `PostTiling()` 通过 `SaveToBuffer` 序列化，Kernel 侧 membase 用 `GET_TILING_DATA` 解包，regbase 用 `GET_TILING_DATA_WITH_STRUCT` 解包。

**TilingKey 映射总表**:

| TilingKey | Kernel类型 | 数据类型 | 适用场景 |
|-----------|-----------|---------|----------|
| 100000 | membase | INT8+INT32 | 910b/910_93 |
| 100010 | membase | INT8+INT64 | 910b/910_93 |
| 100100 | membase | FP16+INT32 | 910b/910_93 |
| 100110 | membase | FP16+INT64 | 910b/910_93 |
| 100200 | membase | BF16+INT32 | 910b/910_93 |
| 100210 | membase | BF16+INT64 | 910b/910_93 |
| 200000 | regbase Re | 通用+无scale | 950, rank<5 |
| 200100 | regbase Re | 通用+FLOAT scale | 950, rank<5 |
| 200200 | regbase Re | FP8+E8M0 scale | 950, rank<5 |
| 210000 | regbase R | 通用+无scale | 950, 回退 |
| 210100 | regbase R | 通用+FLOAT scale | 950, 回退 |
| 210200 | regbase R | FP8+E8M0 scale | 950, 回退 |

### 5.2 Workspace 布局

| 策略 | Workspace 大小 | 用途 |
|------|---------------|------|
| membase | 32 字节 | 系统最小 workspace |
| Re / R (regbase) | 16 MB | 预留（当前kernel未实际使用） |

---

## 6. 完整数据流

### 6.1 端到端数据流

```
  tokens [A, H]   expert_token_num_per_rank [N, E]   per_token_scales [A]
       │                     │                              │
       v                     v                              v
  ┌--------------------------------------------------------------------┐
  │ Host: Tiling策略选择 → TilingKey + TilingData + BlockDim          │
  └──────────────────────────────┬─────────────────────────────────────┘
                                 │
  ┌──────────────────────────────▼─────────────────────────────────────┐
  │ Kernel: TilingKey路由                                              │
  │   ├── membase (100xxx): 按 rank 分核，标量偏移计算                  │
  │   ├── regbase Re (200xxx): rank×expert二维分核                     │
  │   └── regbase R (210xxx): 仅 rank 分核                            │
  │                                                                    │
  │   per (rank, expert) pair:                                         │
  │     CalSrcOffSet → CalDstOffSet                                    │
  │     ProcessTokenScale: CopyIn(GM→UB) → CopyOut(UB→GM)             │
  │     CopyOutIndex: CreateVecIndex → permute_token_idx               │
  │   CalcExpertTokenNum: 沿rank维求和 → expert_token_num              │
  └──────────────────────────────┬─────────────────────────────────────┘
                                 │
       ┌─────────────────────────┼──────────────────────────┐
       v                         v                          v
  permute_tokens [A,H]   permute_token_idx [A]   expert_token_num [E]
  (expert-major排列)      (gather/scatter索引)     (每expert总token数)
```

### 6.2 核心 Token 搬移示意

以 2 rank、3 expert 为例:

```
输入布局 (rank-major):              输出布局 (expert-major):
  expert_token_num_per_rank [2,3]:   expert_token_num [3]:
  R0: [E0:2, E1:1, E2:3]            [2+1=3, 1+2=3, 3+1=4]
  R1: [E0:1, E1:2, E2:1]

  tokens (按rank序):                  permute_tokens (按expert序):
  [R0E0,R0E0,R0E1,R0E2,              [R0E0,R0E0,R1E0,              ← expert 0
   R0E2,R0E2,R1E0,R1E1,               R0E1,R1E1,R1E1,              ← expert 1
   R1E2]                               R0E2,R0E2,R0E2,R1E2]        ← expert 2

  permute_token_idx (gather模式):
  [0, 1, 6,                           ← expert 0 的源位置
   2, 7, 8,                           ← expert 1 的源位置
   3, 4, 5, 9]                        ← expert 2 的源位置
```

---

## 7. 性能相关设计

1. **三策略自适应选择**: 根据平台（membase/regbase）和数据规模（rank数、expert数）自动选择最优策略。rank<5 且 expert!=1 时用 Re（二维分核），否则用 R（一维分核），membase 平台走独立路径。

2. **二维分核优化（Re模式）**: 当 rank 数较少时，仅按 rank 分核会导致大量核空闲。Re 模式将剩余核分配给 expert 维度，形成 rank×expert 二维分核，充分利用所有可用核。

3. **双路径 UB 搬移**: 根据 tokenSize 与 ubFactor 的关系选择搬移路径。单个token小于UB时批量搬移多个token；单个token超过UB时逐token逐分片搬移，避免 UB 溢出。

4. **无 SyncAll 设计**: 三种策略均无需核间同步。各核处理不同 (rank, expert) 区域，写入输出不同位置。`expert_token_num` 的计算也通过步长访问避免写冲突。

5. **FP8 支持（ascend950）**: 针对 FP8 量化场景，支持 FLOAT8_E5M2/E4M3FN token 和 FLOAT8_E8M0 scale，减少量化推理的 dtype 转换开销。

6. **双缓冲队列**: membase 使用 `TQueBind`（VECIN/VECOUT 双缓冲），regbase 使用 `TQue` 配合 `DOUBLE_BUFFER=2`，隐藏 DMA 延迟。

---

## 8. 依赖关系

```
op_host/moe_re_routing_def.cpp
└── register/op_def_registry.h

op_host/moe_re_routing_tiling.cpp
├── moe_re_routing_tiling.h
└── tiling_base/tiling_templates_registry.h

op_host/moe_re_routing_tiling_base.cpp
├── moe_re_routing_tiling_base.h
│   └── tiling_base/tiling_base.h
└── moe_re_routing_tiling.h

op_host/moe_re_routing_membase_tiling.cpp
├── moe_re_routing_membase_tiling.h
│   ├── moe_re_routing_tiling.h
│   └── tiling_base/tiling_base.h
└── platform_ascendc::PlatformAscendC

op_host/moe_re_routing_re_tiling.cpp
├── moe_re_routing_re_tiling.h
│   ├── moe_re_routing_tiling_base.h
│   └── tiling_base/tiling_base.h
└── IsRegbaseSocVersion()

op_host/moe_re_routing_r_tiling.cpp
├── moe_re_routing_r_tiling.h
│   ├── moe_re_routing_tiling_base.h
│   └── tiling_base/tiling_base.h
└── IsRegbaseSocVersion()

op_kernel/moe_re_routing.cpp
└── moe_re_routing.h
    └── kernel_operator.h

op_kernel/moe_re_routing_apt.cpp
├── arch35/moe_re_routing_re_regbase.h
│   ├── arch35/platform.h
│   └── kernel_operator.h
└── arch35/moe_re_routing_r_regbase.h
    ├── arch35/platform.h
    └── kernel_operator.h

op_host/moe_re_routing_infershape.cpp
├── log/log.h
└── register/op_impl_registry.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| expert_token_num_type | 必须为 1（仅 count 模式，cumsum 暂不支持） |
| idx_type (membase) | 必须为 0（仅 gather），scatter 仅 regbase 支持 |
| tokenSize 上限 | membase: H <= 16384（TOKENS_LENGTH_UPPER_BOUND） |
| BF16 平台排除 | arch3003/arch3113 不可用 BF16 |
| Re 策略激活条件 | rankNum < 5 且 expertNum != 1，且必须 regbase SoC |
| expert_token_num 写入 | membase 仅核0计算；regbase 多核步长并行 |
| 空token处理 | tokenNum == 0 时强制 numBlocks = 1 |

---

## 10. 与同族变体的核心差异

本算子为独立的 MoE 重排算子，无直接同族变体。在 MoE 流水线中位于 AlltoAll 通信之后、expert 计算之前，完成从 rank-major 到 expert-major 的布局转换。与 `moe_token_permute`（单卡内按索引重排）不同，本算子处理的是跨卡通信后的全局重排场景。
