# moe_init_routing_v2_grad 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeInitRoutingV2Grad`
**算子类**: MoE 路由初始化反向算子（第二代）
**功能描述**: moe_init_routing_v2 的反向传播。将经过 MoE 专家处理后的梯度 `grad_expanded_x` 通过 `expanded_row_idx` 映射回原始 token 位置，累加到 `grad_x`。核心操作为 scatter-add——将展开后的梯度"折叠"回原始形状。

**支持平台**: ascend910b, ascend910_93, ascend950（regbase，使用 APT 内核）
**支持数据类型**: FP16/BF16/FP32 × INT32（expanded_row_idx）

---

## 2. 目录结构与文件职责

```
moe_init_routing_v2_grad/
├── CMakeLists.txt, README.md
├── op_host/
│   ├── moe_init_routing_v2_grad_def.cpp             # 算子定义
│   ├── moe_init_routing_v2_grad_tiling.h            # Tiling数据结构（4种TilingData）
│   ├── moe_init_routing_v2_grad_tiling_base.cpp     # Tiling基础（校验/平台/公共）
│   ├── moe_init_routing_v2_grad_tiling.cpp          # Tiling策略（membase, 9 TilingKey）
│   ├── moe_init_routing_v2_grad_tiling_arch35.cpp   # regbase标准Tiling
│   ├── moe_init_routing_v2_grad_full_load_tiling_arch35.cpp  # regbase FullLoad Tiling
│   ├── moe_init_routing_v2_grad_split_h_tiling_arch35.cpp    # regbase SplitH Tiling
│   ├── moe_init_routing_v2_grad_infershape.cpp      # Shape推导
│   ├── op_api/
│   │   ├── aclnn_moe_init_routing_v2_grad.h         # ACLNN API声明
│   │   └── aclnn_moe_init_routing_v2_grad.cpp       # ACLNN API（转发内部实现）
│   └── config/ {ascend910b, ascend910_93, ascend950}
├── op_kernel/
│   ├── moe_init_routing_v2_grad.cpp                 # membase Kernel入口（9分支）
│   ├── moe_init_routing_v2_grad_apt.cpp             # regbase Kernel入口（9分支）
│   ├── moe_init_routing_v2_grad_base.h              # Grad计算基类（BinaryAdd树）
│   ├── moe_init_routing_v2_grad_with_dropless.h     # Dropless模式
│   ├── moe_init_routing_v2_grad_with_activate.h     # Activate模式
│   ├── moe_init_routing_v2_grad_with_pos_drop_and_pad_zero.h  # DropPad模式
│   └── arch35/                                      # ascend950 regbase专用
│       ├── moe_init_routing_v2_grad.h               # regbase标准路径
│       ├── moe_init_routing_v2_grad_base.h          # MicroAPI基类（SequenceReduceSum）
│       ├── moe_init_routing_v2_grad_full_load.h     # FullLoad路径
│       └── moe_init_routing_v2_grad_split_h.h       # SplitH路径
├── op_graph/ moe_init_routing_v2_grad_proto.h
├── docs/ aclnnMoeInitRoutingV2Grad.md
├── examples/ test_aclnn_moe_init_routing_v2_grad.cpp
└── tests/ut/ {op_host, op_kernel}, tests/pytest/
```

---

## 3. Host 侧分析

### 3.1 算子定义 (moe_init_routing_v2_grad_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | grad_expanded_x | REQUIRED | FP16/BF16/FP32 | Dropless: [N*K, cols]; Activate: [activeNum, cols]; DropPad: [E, C, cols] |
| 输入 | expanded_row_idx | REQUIRED | INT32 | 1D: [N*K] |
| 输出 | grad_x | REQUIRED | 同 grad_expanded_x | 2D: [N, cols] |

**属性**:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| top_k | Int（必填） | - | TopK 专家数 |
| drop_pad_mode | Int | 0 | 0=dropless, 1=drop/pad |
| active_num | Int | 0 | 活跃 token 行数 |

**平台配置**: ascend910b/910_93 使用 `moe_init_routing_v2_grad` 内核；ascend950 使用 `moe_init_routing_v2_grad_apt` 内核。

### 3.2 Tiling 策略

**四种 TilingData 结构**（tiling.h）:

```
1. MoeInitRoutingV2GradTilingData (membase标准路径)
├── coreNum, n, e, c, cols, k, activeNum
└── MoeV2GradComputeParamsOp
    ├── needCoreNum, perCoreElements, lastCoreElements
    ├── elementCopyLoops, elementPerCopyCols, elementLastCopyCols
    ├── binaryAddBufferNum, tmpBufferNum, exponentOfBinary, copyBufferSize
    └── tokensFormer, perCoreTokensLoop, perCoreTailTokensFormer, lastCoreTokensLoop, lastCoreTailTokensFormer

2. MoeInitRoutingV2GradRegbaseSplitHTilingData (regbase SplitH)
├── n, kUbFactor, k, activeNum, h, hBlockFactor, hUbFactor, numBlocks

3. MoeInitRoutingV2GradRegbaseFullLoadTilingData (regbase FullLoad)
├── n, nBlockFactor, nUbFactor, k, activeNum, h, hUbFactor, numBlocks

4. MoeInitRoutingV2GradRegbaseTilingData (regbase标准)
├── n, nBlockFactor, nUbFactor, k, kUbFactor, activeNum, h, hUbFactor, numBlocks
```

**关键常量**（tiling.cpp）:

| 常量 | 值 | 说明 |
|------|----|------|
| MAX_BINARY_ADD_BUFFER_CNT | 64 | BinaryAdd 最大缓冲数 |
| MAX_EXPONENT_OF_BINARY | 6 | BinaryAdd 最大指数 |
| BOUND_K_FOR_BINARY | 128 | BinaryAdd K 阈值 |
| ELEMENT_ALIGN_SIZE | 32 | 对齐粒度 |
| MAX_DATACOPY_SIZE | 65536 | 单次最大拷贝元素数 |

**Membase TilingKey 编码**（tiling.cpp 第 110-126 行）:

```
Key = 1000 + mode_offset + dtype_offset
mode_offset: 0=Dropless, 10=Activate, 100=DropPad
dtype_offset: 0=FP32, 1=FP16, 2=BF16
```

| Key | 模式 | 数据类型 |
|-----|------|----------|
| 1000/1001/1002 | Dropless | FP32/FP16/BF16 |
| 1010/1011/1012 | Activate | FP32/FP16/BF16 |
| 1100/1101/1102 | DropPad | FP32/FP16/BF16 |

**Regbase TilingKey 编码**:

| Key | 策略 | 模式 |
|-----|------|------|
| 200001/200002/200003 | SplitH | Dropless/DropPad/Active |
| 300001/300002/300003 | FullLoad | Dropless/DropPad/Active |
| 400001/400002/400003 | Regbase标准 | Dropless/DropPad/Active |

**Tiling 模板优先级**（从高到低）: SplitH(20000) → FullLoad(30000) → Regbase标准(40000) → Membase(10000)。

**Membase BinaryAdd 树参数计算**（tiling.cpp 第 168-228 行）:
- `K >= 128`: `exponentOfBinary=6`, `binaryAddBufferNum=64`, `tmpBufferNum=64`
- `K < 128`: `exponentOfBinary=floor(log2(K))-1`, `binaryAddBufferNum=2^exp`, `tmpBufferNum=1`

### 3.3 Shape 推导 (moe_init_routing_v2_grad_infershape.cpp)

```
grad_x: [expandedRowIdxDim0 / topK, gradExpandedXDim1]
```

**校验约束**: `dropPadMode` 为 0 或 1，`topK > 0`，`activeNum >= 0`；Dropless 时输入 2D，DropPad 时输入 3D；输出 dim0 = rowIdxDim0 / topK。

### 3.4 ACLNN API (op_api/aclnn_moe_init_routing_v2_grad.cpp)

**两段式接口**:

```cpp
// 第一段
aclnnStatus aclnnMoeInitRoutingV2GradGetWorkspaceSize(
    const aclTensor* gradExpandedX, const aclTensor* expandedRowIdx,
    int64_t topK, int64_t dropPadMode, int64_t activeNum,
    const aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

// 第二段
aclnnStatus aclnnMoeInitRoutingV2Grad(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);
```

ACLNN 实现为薄转发层，直接调用 `aclnnInnerMoeInitRoutingV2Grad` 内部实现。

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与初始化

**Membase 入口**（moe_init_routing_v2_grad.cpp）: AIV-only 内核，9 个 TilingKey 分支:
```
Keys 1000-1002: MoeInitRoutingV2GradDroplessCompute<T>
Keys 1010-1012: MoeInitRoutingV2GradActivateCompute<T>
Keys 1100-1102: MoeInitRoutingV2GradPositionPad0Compute<T>
```

**Regbase 入口**（moe_init_routing_v2_grad_apt.cpp）: 9 个 TilingKey 分支:
```
Keys 200001-200003: MoeInitRoutingV2GradSplitHCompute<T, Mode>
Keys 300001-300003: MoeInitRoutingV2GradFullLoadCompute<T, Mode>
Keys 400001-400003: MoeInitRoutingV2GradCompute<T, Mode>
```
Mode 模板参数: 0=Dropless, 1=DropPad, 2=Active。

### 4.2 核心计算流程

#### Membase 路径 — BinaryAdd 树

**基类** `MoeInitRoutingV2GradBase<T>`（moe_init_routing_v2_grad_base.h）:

核心梯度计算为四阶段 BinaryAdd 树:
```
S1: 加载初始 binaryAddBufferNum 行到 bin 缓冲（baseStride=2 隔行访问）
S2: 累加剩余偶数行直接到对应 bin 缓冲
S3: 累加所有奇数行（偏移1）到对应 bin 缓冲
S4: 二分归约树跨 bin 缓冲（stride 每层翻倍, expNum 层）
```

数据流: `CopyInWithCastFloat`(FP16/BF16→FP32) → `GradAdd`(向量加) → `CopyOut`(FP32→原类型)。

**三种模式的差异**:

| 模式 | 输入形状 | 无效行判断 | 初始化 |
|------|----------|-----------|--------|
| Dropless | [N*K, H] | 无，所有行有效 | 全部加载 |
| Activate | [activeNum, H] | `xRow >= activeNum` 时置零 | 越界零填充 |
| DropPad | [E, C, H] | `xRow == -1` 时置零 | padding 零填充 |

#### Regbase 路径 — SequenceReduceSum

**基类** `MoeInitRoutingV2GradBase`（arch35/）: MicroAPI 实现。

核心函数 `SequenceReduceSum<T, U, withInit>`:
```
4-way H 维展开 × 4-way K 维展开 = 16 寄存器操作数
主循环: 处理 h / (VL_F32 * 4) 个完整块
K 按 4 分组处理（含余数处理）
尾循环: 处理非对齐 H 余量
```

**三种策略**:

| 策略 | 核心切分 | 适用场景 |
|------|----------|----------|
| FullLoad | 按 N 分核，H 全量加载 | 通用场景 |
| SplitH | 按 H 分核 | 小 topK, 少 token, 大 H |
| Regbase标准 | 按 N 分核，K/H 分段 | FullLoad 失败时回退 |

### 4.3 数据结构与内存布局

**Membase 基类**（moe_init_routing_v2_grad_base.h）:
- `TBuf<VECIN> binBuff, tmpBuff` — UB 缓冲，BinaryAdd 树的缓冲对
- `GlobalTensor<T> gradExpandedXGm, gradXGm` — GM 张量
- `baseStride=2` — 隔行访问步长
- `multiTmpBuff` — K>=128 时 true，每个 bin 对应一个 tmp 缓冲

**Regbase 基类**（arch35/）:
- `VL_F32 = GetVRegSize() / sizeof(float)` — 向量寄存器宽度
- `BLOCK_SIZE = GetUbBlockSize()` — UB 块对齐
- Cast traits: `castTraitB162B32`(B16→FP32), `castTraitB322B16`(FP32→B16)

### 4.4 多核并行策略

**Membase**: 按 N 均匀分核，`perCoreElements = CeilDiv(N, aivNum)`。核内按 token 组并行，每组 `tokensFormer` 个 token 共享 UB 缓冲。

**Regbase FullLoad**: 按 N 分核，每个核处理 `nBlockFactor` 个 token。
**Regbase SplitH**: 按 H 分核，每个核处理 `hBlockFactor` 个元素。
**Regbase 标准**: 按 N 分核，K 和 H 在 UB 内分段处理。

### 4.5 通信模式

**Membase**: 无 SyncAll。核间通过显式硬件事件同步（`SetFlag<HardEvent::V_MTE2>`/`WaitFlag`）实现细粒度流水重叠。

**Regbase**: 无 SyncAll。各核独立处理分配的 N/H 分片。

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

见 3.2 节四种 TilingData 结构。Membase 使用 `BEGIN_TILING_DATA_DEF` 宏，Regbase 三种策略各有独立 TilingData。Kernel 侧 `GET_TILING_DATA` 解包。

**CompileInfo**: 空结构（平台信息通过 `PlatformAscendC` 运行时获取）。

### 5.2 Workspace 布局

```
┌──────────────────────────────────────────────────────────────┐
│ System Workspace = 16 MB                                      │
│ (固定大小，无额外算子级 workspace)                              │
└──────────────────────────────────────────────────────────────┘
```

所有中间计算在 UB 内完成（BinaryAdd 树或 SequenceReduceSum），不需要 GM 级中间缓冲。

---

## 6. 完整数据流

### Dropless 模式

```
  grad_expanded_x [N*K, H]       expanded_row_idx [N*K]
       │                                │
       └──────────┬─────────────────────┘
                  │
          [BinaryAdd树 / SequenceReduceSum]
          (按row_idx累加K路梯度到同一行)
                  │
                  │  对于每个输出位置 i:
                  │  grad_x[i,:] = Σ grad_expanded_x[row_idx[i*k+j],:]
                  │                for j in 0..k
                  │
          ┌───────┴────────┐
          │                │
      grad_x [N, H]
```

### DropPad 模式

```
  grad_expanded_x [E, C, H]      expanded_row_idx [N*K]
       │                                │
       └──────────┬─────────────────────┘
                  │
          [BinaryAdd树 / SequenceReduceSum]
          (跳过 row_idx==-1 的 padding 位置)
                  │
          ┌───────┴────────┐
          │                │
      grad_x [N, H]
```

---

## 7. 性能相关设计

1. **BinaryAdd 树（Membase）**: 当 `K >= 128` 时使用 64 缓冲全展开树（6 层二分归约），`K < 128` 时按 `2^(floor(log2(K))-1)` 缓冲。隔行访问（baseStride=2）避免连续行写冲突。

2. **SequenceReduceSum（Regbase）**: MicroAPI 4-way H × 4-way K 展开实现高吞吐，自动处理 FP16/BF16 到 FP32 的精度转换。单次循环处理 16 个数据块。

3. **三种策略自适应（Regbase）**: SplitH 适用于大 H 少 token 场景（按 H 分核），FullLoad 适用于通用场景（按 N 分核全量加载），标准路径为 FullLoad 失败（K>10 且 H 无法全量加载）时的回退。

4. **显式硬件事件同步**: Membase 路径使用 `SetFlag`/`WaitFlag` 替代 `PipeBarrier`，实现搬运与计算的细粒度重叠。

5. **混合精度累加**: 所有累加在 FP32 空间进行（`CopyInWithCastFloat`），减少精度损失，最终写回时 cast 到原始数据类型。

6. **tokensFormer 并行**: 多个 token 共享同一组 UB 缓冲，减少缓冲分配开销。

---

## 8. 依赖关系

```
op_host/moe_init_routing_v2_grad_def.cpp
└── register/op_def_registry.h

op_host/moe_init_routing_v2_grad_tiling_base.cpp
├── moe_init_routing_v2_grad_tiling.h
│   ├── register/op_impl_registry.h
│   ├── register/tilingdata_base.h
│   └── tiling_base/tiling_base.h
└── platform_ascendc::PlatformAscendC

op_host/moe_init_routing_v2_grad_tiling.cpp (membase)
├── moe_init_routing_v2_grad_tiling.h
└── tiling_base/tiling_base.h

op_host/moe_init_routing_v2_grad_tiling_arch35.cpp (regbase标准)
op_host/moe_init_routing_v2_grad_full_load_tiling_arch35.cpp (regbase FullLoad)
op_host/moe_init_routing_v2_grad_split_h_tiling_arch35.cpp (regbase SplitH)

op_host/moe_init_routing_v2_grad_infershape.cpp
├── log/log.h
└── register/op_impl_registry.h

op_host/op_api/aclnn_moe_init_routing_v2_grad.cpp
├── aclnn_moe_init_routing_v2_grad.h
└── aclnnInnerMoeInitRoutingV2Grad (内部符号)

op_kernel/moe_init_routing_v2_grad.cpp (membase)
├── moe_init_routing_v2_grad_with_dropless.h
│   └── moe_init_routing_v2_grad_base.h
│       └── kernel_operator.h
├── moe_init_routing_v2_grad_with_activate.h
│   └── moe_init_routing_v2_grad_base.h
├── moe_init_routing_v2_grad_with_pos_drop_and_pad_zero.h
│   └── moe_init_routing_v2_grad_base.h
└── kernel_operator.h

op_kernel/moe_init_routing_v2_grad_apt.cpp (regbase)
├── arch35/moe_init_routing_v2_grad.h
│   └── arch35/moe_init_routing_v2_grad_base.h
├── arch35/moe_init_routing_v2_grad_full_load.h
│   └── arch35/moe_init_routing_v2_grad_base.h
├── arch35/moe_init_routing_v2_grad_split_h.h
│   └── arch35/moe_init_routing_v2_grad_base.h
└── kernel_operator.h

op_graph/moe_init_routing_v2_grad_proto.h
└── graph/operator_reg.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| top_k | 必填属性，必须 > 0 |
| drop_pad_mode | 0 或 1 |
| active_num | 必须 >= 0 |
| 输入维度 | Dropless/Activate: 2D; DropPad: 3D |
| expanded_row_idx | 必须为 1D INT32 |
| 输出维度 | 始终 2D: [N, cols] |
| SplitH 条件 | topK<=64, N<aivNum, N<(H/512/typeSize) |
| FullLoad 限制 | regbase K>10 且 H 不能全量加载时回退到标准路径 |
| BinaryAdd 缓冲 | 最大 64 个（K>=128 时） |
| dtype 一致 | grad_x 与 grad_expanded_x 同类型 |

---

## 10. 与同族变体的核心差异

### vs moe_init_routing_v2（正向）

| 维度 | v2（正向） | v2_grad（反向） |
|------|-----------|----------------|
| 操作方向 | token → 专家（Scatter） | 专家梯度 → token（Gather+Reduce） |
| 核心操作 | Sort→SrcToDst→Gather | BinaryAdd 树 / SequenceReduceSum |
| 排序 | 有（单核/多核/HistRegBase） | 无 |
| DropPad 输出形状 | [E, C, cols] 3D | 输入 3D，输出 [N, cols] 2D |
| workspace | 排序+scatter+sync | 仅系统 16MB |
| Expert Token Count | 有 | 无 |

### vs moe_init_routing_grad（v1 反向）

| 维度 | v1_grad | v2_grad |
|------|---------|---------|
| Activate 模式 | 无 | 有 |
| DropPad 模式 | 无 | 有 |
| Regbase 路径 | 无 | 有（SplitH/FullLoad/标准） |
| BinaryAdd 树 | 基础版 | 优化版（64缓冲全展开） |
