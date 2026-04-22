# moe_finalize_routing 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeFinalizeRouting`
**算子类**: MoE 路由终结（第一代）
**功能描述**: MoE 推理的最后一步——将各专家的输出加权合并回原始 token 顺序。对每行 token，按 K 个选中专家的权重（scales）加权求和 `expanded_x + bias`，再加上残差连接（skip1 + skip2），输出最终结果。

**支持平台**: ascend910b, ascend910_93, kirinx90, kirin9030（Kirin 不支持 BF16）
**支持数据类型**: FP16/BF16/FP32 × INT32（索引）

---

## 2. 目录结构与文件职责

```
moe_finalize_routing/
├── CMakeLists.txt, README.md
├── op_host/
│   ├── moe_finalize_routing_def.cpp             # 算子定义
│   ├── moe_finalize_routing_tiling.h            # TilingData结构（21字段）
│   ├── moe_finalize_routing_tiling.cpp          # Tiling策略（19档位）
│   ├── moe_finalize_routing_infershape.cpp      # Shape推导
│   └── config/ {ascend910b, ascend910_93, kirinx90, kirin9030}
├── op_kernel/
│   ├── moe_finalize_routing.cpp                 # Kernel入口（19分支）
│   ├── moe_finalize_routing_common.h            # 公共常量/工具
│   ├── moe_finalize_routing_fp.h                # FP32基础模板
│   ├── moe_finalize_routing_fp_db.h             # 双缓冲FullLoad
│   ├── moe_finalize_routing_fp_db_all_bias.h    # 双缓冲全Bias预加载
│   ├── moe_finalize_routing_fp_cutk.h           # CutK（K>256）
│   ├── moe_finalize_routing_fp_cuth.h           # CutH（通用K）
│   ├── moe_finalize_routing_fp_cuth_k2.h        # CutH + K=2展开
│   ├── moe_finalize_routing_fp_cuth_k4.h        # CutH + K=4展开
│   ├── moe_finalize_routing_fp_cuth_k2_optimized.h  # H=5120优化
│   ├── moe_finalize_routing_bf16.h              # BF16（Cast到FP32计算）
│   ├── moe_finalize_routing_bf16_cutk.h         # BF16 CutK
│   ├── moe_finalize_routing_bf16_cuth.h         # BF16 CutH
│   ├── moe_finalize_routing_bf16_cuth_k2.h      # BF16 CutH+K2
│   ├── moe_finalize_routing_bf16_cuth_k4.h      # BF16 CutH+K4
│   └── moe_finalize_routing_bf16_all_bias.h     # BF16全Bias预加载
├── docs/ aclnnMoeFinalizeRouting.md
├── examples/ test_aclnn_moe_finalize_routing.cpp
└── tests/ut/ {op_host, op_kernel}
```

---

## 3. Host 侧分析

### 3.1 算子定义 (moe_finalize_routing_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | expanded_x | REQUIRED | FP16/BF16/FP32 | 2D: [N*K, H] |
| 输入 | x1 (skip1) | REQUIRED | FP16/BF16/FP32 | 2D: [N, H] |
| 输入 | x2 (skip2) | OPTIONAL | FP16/BF16/FP32 | 2D: [N, H] |
| 输入 | bias | REQUIRED | FP16/BF16/FP32 | 2D: [E, H] |
| 输入 | scales | REQUIRED | FP16/BF16/FP32 | 2D: [N, K] |
| 输入 | expanded_row_idx | REQUIRED | INT32 | 1D: [N*K] |
| 输入 | expanded_expert_idx | REQUIRED | INT32 | 2D: [N, K] |
| 输出 | y | REQUIRED | 同 x1 | 2D: [N, H] |

### 3.2 Tiling 策略 (moe_finalize_routing_tiling.cpp)

**TilingData**（tiling.h）: 21 个 int64_t 字段，含核分配（`totalCoreNum/usedCoreNum`）、H/K 切分（`normalH/unnormalH/hSliceNum/normalK/unnormalK/kSliceNum`）、行分配（`normalCoreHandleNum/tailCoreHandleNum` 及循环参数）。

**TilingKey 编码**（19 个 Key，范围 20000-20018）:

| Key | 策略 | 数据类型 |
|-----|------|----------|
| 20000-20002 | CutK（K>256） | FP32/FP16/BF16 |
| 20003 | FullH+FullK（BF16 Cast） | BF16 |
| 20004-20005 | 双缓冲 FullH | FP32/FP16 |
| 20006-20008 | CutH + K=2 展开 | FP32/FP16/BF16 |
| 20009-20011 | CutH + K=4 展开 | FP32/FP16/BF16 |
| 20012-20014 | CutH（通用 K） | FP32/FP16/BF16 |
| 20015-20016 | 双缓冲全Bias预加载 | FP32/FP16 |
| 20017 | BF16全Bias预加载 | BF16 |
| 20018 | CutH优化（H=5120） | FP32 |

**关键常量**: `BOUND_K=256`（CutK 阈值）, `NETWORKSIZE2=5120`（优化触发 H 值）, `TRIPLE_BUFFER=3`（BF16→FP32 空间系数）。

**Tiling 策略选择优先级**: 优化CutH(H=5120) → 全Bias预加载(K<=256) → 双缓冲FullH → CutH（按 K=2/K=4/通用）。

### 3.3 Shape 推导 (moe_finalize_routing_infershape.cpp)

```
y: [x1.dim(0), x1.dim(1)]  (同 x1/skip1 的形状)
```

校验: 所有 2D 输入的 dim-1 必须匹配（=H），dim-0 一致性校验。

### 3.4 ACLNN API

本算子支持 ACLNN 发布接口（`aclnnSupport.value = "support_aclnn"`）。

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口 (moe_finalize_routing.cpp)

19 个 TilingKey 分支，全部为 AIV-only 内核。

### 4.2 核心计算流程

**数学公式**:
```
y[i, h] = x1[i, h] + x2[i, h] + Σ(k=0..K-1) scales[i, k] * (expanded_x[row_idx[i*K+k], h] + bias[expert_idx[i, k], h])
```

**数据流**:
```
对每行 i:
  result = skip1[i] + skip2[i]              # 残差连接
  对每个 k in 0..K-1:
    idx = expanded_row_idx[i*K+k]           # 查找展开后的行
    expert = expert_idx[i, k]               # 查找专家ID
    result += scales[i,k] * (expanded_x[idx] + bias[expert])  # 加权累加
  y[i] = result
```

**BF16 处理**: 所有 BF16 输入先 Cast 到 FP32 进行累加，最终 Cast 回 BF16 输出。额外 UB 缓冲: `skip1CastBuf_`, `skip2CastBuf_`, `biasCastBuf_`, `expandedPermutedRowsCastBuf_`。

### 4.3 数据结构与内存布局

**公共常量**（moe_finalize_routing_common.h）: `ONE_BLK_SIZE=32`, `ONCE_ALGN_NUM_INT32=8`, `BUFFER_NUM=1`, `PARALLEL_NUM=2`（双缓冲乒乓）。

**13 个 Kernel 类**:
- FP32/FP16: `FpDb`, `FpCutK`, `FpCuth`, `FpCuthK2`, `FpCuthK4`, `FpDbAllBias`, `FpCuthK2Optimized`
- BF16: `BF16`, `Bf16CutK`, `Bf16Cuth`, `Bf16CuthK2`, `Bf16CuthK4`, `BF16AllBias`

### 4.4 多核并行策略

按行均匀分核: `normalCoreHandleNum = CeilDiv(N, coreNum)`。核内按 H/K 切分:
- **CutK**: K>256 时按 256 切片
- **CutH**: H 过大时切片，支持 K=2/K=4 展开优化
- **双缓冲**: `PARALLEL_NUM=2` 乒乓加载 expanded_x 和 bias

### 4.5 通信模式

无 SyncAll。双缓冲使用硬件事件标志（`EVENT_ID0`-`EVENT_ID3`）实现细粒度流水:
- V_S → S_MTE2 → MTE2_V → V_S（标量读 → GM 加载 → 向量计算）

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

21 个 int64_t 字段（见 3.2 节）。CompileInfo 含 `totalCoreNum`, `ubSize`。

### 5.2 Workspace 布局

```
┌──────────────────────────────────────────────────────────────┐
│ System Workspace = 16 MB (固定大小)                            │
└──────────────────────────────────────────────────────────────┘
```

无额外算子级 workspace，所有中间计算在 UB 内完成。

---

## 6. 完整数据流

```
  expanded_x [N*K, H]        scales [N, K]
  expanded_row_idx [N*K]     bias [E, H]
  expanded_expert_idx [N, K]
  x1 [N, H]  x2 [N, H](可选)
       │           │              │
       └───────────┼──────────────┘
                   │
         对每行 i (核间并行):
           result = x1[i] + x2[i]  (残差)
           对每个 k:
             idx = expanded_row_idx[i*K+k]
             expert = expert_idx[i, k]
             weighted = scales[i,k] * (expanded_x[idx] + bias[expert])
             result += weighted
           y[i] = result
                   │
           ┌───────┴───────┐
           │               │
       y [N, H]
```

---

## 7. 性能相关设计

1. **K=2/K=4 完全展开**: 当 K 恰好为 2 或 4 时，完全展开循环消除分支开销。

2. **H=5120 专用优化**: 特定网络配置的 CutH 优化，使用双缓冲队列合并 skip1/skip2/scales 为连续缓冲。

3. **全 Bias 预加载**: K<=256 时将整个 bias 表预加载到 UB（`PrepareData()`），消除逐行 bias 读取延迟。

4. **双缓冲乒乓**: `PARALLEL_NUM=2` 硬件事件标志实现 expanded_x 和 bias 的加载与计算重叠。

5. **BF16→FP32 精度累加**: 所有 BF16 累加在 FP32 空间完成，减少精度损失。

6. **CutK 切片**: K>256 时按 256 切片，避免 UB 溢出。

---

## 8. 依赖关系

```
op_host/moe_finalize_routing_def.cpp
└── register/op_def_registry.h

op_host/moe_finalize_routing_tiling.cpp
├── moe_finalize_routing_tiling.h
│   ├── register/op_impl_registry.h
│   └── tiling_base/tiling_base.h
└── platform_ascendc::PlatformAscendC

op_kernel/moe_finalize_routing.cpp
├── moe_finalize_routing_fp.h / _fp_db.h / _fp_cutk.h
├── moe_finalize_routing_fp_cuth.h / _k2.h / _k4.h / _k2_optimized.h
├── moe_finalize_routing_fp_db_all_bias.h
├── moe_finalize_routing_bf16.h / _cutk.h / _cuth.h / _k2.h / _k4.h
├── moe_finalize_routing_bf16_all_bias.h
├── moe_finalize_routing_common.h
└── kernel_operator.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| x2 | OPTIONAL，null 时 `skip2IsNull=1` |
| dim-1 一致性 | x1, x2, expanded_x, bias 的最后一维必须相同（=H） |
| dim-0 一致性 | x1, x2, scales, expert_idx 的 dim-0 必须相同（=N） |
| expanded_row_idx | 必须为 1D |
| Kirin 平台 | 不支持 BF16 |
| BF16 内核 | NPU arch 3003/3113 不支持 BF16（`#if` 排除） |
| STRIDE_BLOCK_LIMIT | ~1.07 billion rows（DataCopy stride 上限） |

---

## 10. 与同族变体的核心差异

| 维度 | finalize_routing (v1) | finalize_routing_v2 |
|------|----------------------|-------------------|
| x1 (skip1) | REQUIRED | OPTIONAL |
| bias | REQUIRED | OPTIONAL |
| scales | REQUIRED | OPTIONAL |
| expert_idx | REQUIRED (expanded_expert_idx) | OPTIONAL |
| Drop/Pad | 不支持 | 支持（4种模式） |
| regbase | 不支持 | 支持（ascend950, MicroAPI） |
| 310P | 不支持 | 支持 |
| TilingKey 范围 | 20000-20018（19个） | 20000-20049（~50个） |
| 混合精度 | 不支持 | 支持（BF16输入+FP32 scale） |
| ACLNN API | 支持 | 支持 |
