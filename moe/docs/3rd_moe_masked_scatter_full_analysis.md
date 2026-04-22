# 3rd_moe_masked_scatter 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeMaskedScatter`
**算子类**: 第三方算子（3rd），掩码散射
**功能描述**: 根据 bool 掩码将 updates 中的元素逐个填入输出张量 y。mask 为 True 的位置写入 updates 值，False 的位置复制 x 值。等价于 PyTorch 的 `torch.masked_scatter`。

**核心语义**:
```
if mask[i] == True:  y[i] = updates[前i个位置中True的个数]
else:                y[i] = x[i]
```

**两阶段执行**: Phase 1 计算前缀和 → SyncAll → Phase 2 Scatter

**支持平台**: ascend950, ascend910_93
**支持数据类型**: FP32/FP16/DOUBLE/UINT8/INT8/INT16/INT32/INT64/BOOL/BF16

---

## 2. 目录结构与文件职责

```
3rd/moe_masked_scatter/
├── CMakeLists.txt
├── op_host/
│   ├── moe_masked_scatter_def.cpp              # 算子定义
│   ├── arch35/
│   │   ├── moe_masked_scatter_tiling.h         # Tiling数据结构
│   │   └── moe_masked_scatter_tiling.cpp       # Tiling策略
│   └── op_api/
│       ├── moe_masked_scatter.h                # L0 接口声明
│       └── moe_masked_scatter.cpp              # AiCore 路径选择
└── op_kernel/
    ├── moe_masked_scatter.cpp                  # Kernel入口（8个分支）
    └── arch35/
        └── moe_masked_scatter.h                # 核心kernel实现（420行）
```

---

## 3. Host 侧分析

### 3.1 算子定义 (moe_masked_scatter_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 说明 |
|------|------|----------|----------|------|
| 输入 | x | REQUIRED | FP32/FP16/DOUBLE/UINT8/INT8/INT16/INT32/INT64/BOOL/BF16 | 输入张量 |
| 输入 | mask | REQUIRED | BOOL | 布尔掩码（与 x 同 shape） |
| 输入 | updates | REQUIRED | 与 x 相同 | 更新值（一维展开） |
| 输出 | y | REQUIRED | 与 x 相同 | 输出张量 |

所有输入设置 `AutoContiguous()`。

### 3.2 Tiling 策略 (arch35/moe_masked_scatter_tiling.cpp)

**TilingData 结构**（tiling.h 第29-40行）:

```
MoeMaskedScatterTilingData
├── normalCoreData        : int64_t   # 整核处理数据数
├── tailCoreData          : int64_t   # 尾核处理数据数
├── ubFactor              : int64_t   # UB 每次处理元素数
├── blockFactor           : int64_t   # 整核搬运循环次数
├── tailBlockFactor       : int64_t   # 尾核搬运循环次数
├── tailUbFactor          : int64_t   # 整核尾循环元素数
├── tailCoreTailUbFactor  : int64_t   # 尾核尾循环元素数
├── updateEleNums         : int64_t   # updates 总元素数
├── bufferSize            : int64_t   # 低精度缓冲区大小
└── prefixSumUbSize       : int64_t   # 前缀和 UB 空间
```

**Tiling 计算**（tiling.cpp 第119-146行）:
```
isInt64Mask = xEleNums > INT32_MAX
prefixSumDtypeSize = isInt64Mask ? 8 : 4
maxUbAvailable = ubSize / (2 + 2*prefixSumDtypeSize)
normalCoreData = max(ceil(xEleNums, cores), 1024)
usedCoreNum = ceil(xEleNums, normalCoreData)
ubFactor = 对齐到 VF 寄存器长度
```

**TilingKey 编码**: `dataTypeSize * 10 + isInt64Mask`

| Key | 类型 | 前缀和类型 |
|-----|------|-----------|
| 10/11 | int8_t/uint8_t/bool | int32/int64 |
| 20/21 | int16_t/FP16/BF16 | int32/int64 |
| 40/41 | int32_t/FP32 | int32/int64 |
| 80/81 | int64_t | int32/int64 |

### 3.3 Shape 推导

无独立 infershape 文件。`y` shape 与 `x` 相同。

### 3.4 ACLNN API

L0 接口声明在 `op_api/moe_masked_scatter.h`。L2 层双路径: arch3510 使用自定义 kernel，其他平台回退到 CANN 内置 `MaskedScatter`。

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口 (moe_masked_scatter.cpp)

8 个 TilingKey 分支（第30-77行），按数据类型和前缀和类型分发到 `MoeMaskedScatterImpl<T, U>`。

### 4.2 核心计算流程

**Phase 1 — 前缀和计算**:
```
for 每个UB块:
  CopyInMask → ComputePrefixSum (并行前缀和) → CopyOutMask (写出到 workspace)
SyncAll()
```

前缀和算法: bool→int32 → 行扫描 → 列方向 ReduceSum → 广播加

**Phase 2 — 数据 Scatter**:
```
ComputePreCoreMaskSum()  -- 计算前面核的 mask True 总数

低精度(1/2字节): ProcessLowBit — 在 UB 上 scatter
正常精度(4/8字节): SIMT VF_CALL — 在 GM 上 scatter

per position i:
  if mask[i] == True:  y[i] = updates[prefixSum[i] + preCoreMaskSum]
  else:                y[i] = x[i]
```

### 4.3 数据结构与内存布局

| 成员 | 类型 | 用途 |
|------|------|------|
| `maskQueue_`, `xQueue_` | `TQue<VECIN, 2>` | 双缓冲输入 |
| `tmpBuf_` | `TBuf<VECCALC>` | 临时计算缓冲 |
| `prefixSumQueue_`, `yQueue_` | `TQue<VECOUT, 1>` | 输出队列 |
| `prefixSumGm_` | `GlobalTensor<U>` | 前缀和（workspace） |
| `curMaskSum_`, `preCoreMaskSum_` | `U` | 当前核/前面核的 maskSum |

### 4.4 多核并行策略

按 `normalCoreData = max(ceil(xEleNums, cores), 1024)` 分配核。每核处理 `normalCoreData` 或 `tailCoreData` 个元素。

### 4.5 通信模式

Phase 1 结束后 `SyncAll()` 确保所有核的前缀和结果已写出。Phase 2 开始时每核通过 `ComputePreCoreMaskSum()` 读取 workspace 中其他核的 maskSum，计算全局偏移。

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

见 3.2 节 TilingData 结构（10 个 int64_t 字段）。

**CompileInfo**（tiling.h 第24-27行）:
```
MoeMaskedScatterCompileInfo
├── coreNum : int64_t   # 总核数
└── ubSize  : int64_t   # UB 大小
```

### 5.2 Workspace 布局

```
workspaceSize = 16MB (AscendC工具空间)
             + usedCoreNum * prefixSumDtypeSize  (每核的 maskSum)
             + xEleNums * prefixSumDtypeSize      (全局 prefixSum 数组)
```

workspace 用于存储前缀和中间结果和每核的 maskSum。

---

## 6. 完整数据流

```
  x [N]            mask [N]           updates [M]
    |                  |                  |
    |         Phase 1: 前缀和计算         |
    |         (分核+分UB块)               |
    |                  |                  |
    |         prefixSum [N] (workspace)   |
    |                  |                  |
    |         SyncAll()                   |
    |         ComputePreCoreMaskSum       |
    |                  |                  |
    v                  v                  v
  +-----------------------------------------------+
  | Phase 2: Scatter                              |
  |                                               |
  | for 每个位置 i:                                |
  |   if mask[i] == True:                         |
  |     y[i] = updates[prefixSum[i]+preCoreMask]  |
  |   else:                                       |
  |     y[i] = x[i]                               |
  |                                               |
  | 低精度(1/2字节): 在 UB 上 scatter             |
  | 正常精度(4/8字节): SIMT VF 在 GM 上 scatter   |
  +-----------------------------------------------+
                    |
                    v
               y [N] (输出)
```

---

## 7. 性能相关设计

1. **并行前缀和**: 使用 MicroAPI 实现高效并行前缀和（行扫描+列 ReduceSum+广播加），避免逐元素条件分支开销。

2. **SIMT VF 调用**: Scatter 阶段使用 `Simt::VF_CALL` 启动 1024 线程 SIMT Vector Function，充分利用 arch35 的 SIMT 单元。

3. **低精度 UB Scatter**: 1/2 字节类型先在 UB 完成 scatter 再写回 GM，避免 GM 上随机访问开销。

4. **双精度前缀和**: 根据数据量自动选择 int32 或 int64，避免溢出。

5. **全核同步仅一次**: 仅在 Phase 1→Phase 2 过渡时 SyncAll 一次，Phase 2 各核独立执行。

---

## 8. 依赖关系

```
op_host/arch35/moe_masked_scatter_tiling.cpp
└── moe_masked_scatter_tiling.h
    ├── register/op_impl_registry.h
    └── tiling_base/tiling_base.h

op_host/op_api/moe_masked_scatter.cpp
├── moe_masked_scatter.h
└── external/aclnn_kernels/aclnn_platform.h

op_kernel/moe_masked_scatter.cpp
└── arch35/moe_masked_scatter.h
    └── kernel_operator.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| 平台限制 | 仅 ascend950（def.cpp 第66行） |
| mask 类型 | 仅支持 BOOL |
| mask shape | 必须与 x 相同 |
| updates 大小 | 必须 >= mask 中 True 的总数 |
| 前缀和精度 | xEleNums > INT32_MAX 时自动切换 int64 |
