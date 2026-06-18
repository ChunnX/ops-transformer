# ReLU 从 Vector 核移至 Fixpipe 随路执行方案

---

## 一、背景与动机

当前 QLI 算子 arch35 路径中，ReLU 在 Vector 核的 `MulWeightAndReduceSum*` 函数内以 VF 指令逐元素执行。
Fixpipe 硬件支持随路 ReLU（`reluEn`），可以在 L0C → UB 搬运过程中零开销完成 ReLU。

**但旧芯片（DAV_3510）中 `dualDstCtl != 0` 与 `reluEn` 互斥**：

```cpp
// dav_c310/kernel_operator_fixpipe_impl.h
ASCENDC_ASSERT(!(intriParams.dualDstCtl != 0b00 &&
    (... || intriParams.reluEn)), ...);
```

当前代码使用 `dualDstCtl=1`（按 M 维度拆分给两个 AIV 核），因此无法启用 `reluEn`。

**新芯片 CV 比例改为 1:1**（一个 AIC 对应一个 AIV），不再需要 `dualDstCtl=1` 拆分 M。
改用 `dualDstCtl=0`（单目标模式）后，`reluEn=true` 可用。

### 收益

1. **减少 Vector 核指令数**：每个 G-loop 迭代省去 2~4 条 Relu VF 指令
2. **利用 Fixpipe 空闲算力**：ReLU 是随路操作，零额外 cycle
3. **缩短 Vector 核关键路径**：Vector 核读到 UB 数据时已经是 ReLU 后的值

---

## 二、dualDstCtl 参数详解

### 2.1 定义

`dualDstCtl` 是 `FixpipeParamsC310` 的字段，控制 Fixpipe 的**双目标写入模式**，用于将 L0C 矩阵乘结果自动拆分写入 UB 的两个不同地址。

### 2.2 三种模式

```cpp
enum class McgShfMode {
    SINGLE_DST_MODE = 0,  // dualDstCtl=0b00: 单目标，全部数据写入一个 dst
    DUAL_DST_SPLIT_M,     // dualDstCtl=0b01: 按 M 维度拆半，各写 M/2×N 到两个 dst
    DUAL_DST_SPLIT_N,     // dualDstCtl=0b10: 按 N 维度拆半，各写 M×N/2 到两个 dst
    RESERVED              // dualDstCtl=0b11: 保留
};
```

### 2.3 在 QLI 中的用法

当前 QLI 使用 `dualDstCtl=1`（`DUAL_DST_SPLIT_M`），效果：

```
L0C 矩阵 [M × N] (NZ 格式)
         │
         │  Fixpipe, dualDstCtl=1
         │
    ┌────┴────┐
    ▼         ▼
 dst0       dst1
 M/2 × N   M/2 × N
 (前一半行)  (后一半行)
```

对应代码：
```cpp
fixpipeParams.dualDstCtl = 1;
// dst 地址 = mm1ResUB_[pingpong * s2BaseSize/2]
// dstNdStride = s2BaseSize * mBaseSize / 2  ← 两个 dst 之间的间距
```

### 2.4 为什么需要拆 M

QLI 中 `mBaseSize = 256`（最大），一个 AIC 核配两个 AIV 核（CVRATIO=2）。拆 M 后：

- **AIV0** 读 `dst0`（前 128 行）
- **AIV1** 读 `dst1`（后 128 行）

硬件一次 Fixpipe 就完成拆分，两个 AIV 核各取各的，无需额外搬运。

### 2.5 旧芯片限制

```cpp
// dualDstCtl != 0 时，以下功能全部禁用：
ASCENDC_ASSERT(!(dualDstCtl != 0 && (
    format == COLUMN_MAJOR ||    // 不能 COLUMN_MAJOR 输出
    quantPre != NoQuant ||       // 不能做量化
    reluEn                       // 不能做 ReLU
)), ...);
```

所以旧芯片上 `dualDstCtl` 和 `reluEn` 互斥。

### 2.6 新芯片解决方案

新芯片 CV 比例改为 1:1，一个 AIC 对应一个 AIV，**不再需要拆 M**。改用 `dualDstCtl=0`（单目标模式）后，`reluEn=true` 自然可用，绕过了旧芯片的互斥限制。

---

## 三、当前数据流（CVRATIO=2, dualDstCtl=1）

**注意**：以下为旧芯片配置，新芯片 CVRATIO=1 后不再使用 `dualDstCtl=1`。

```
L0C (cL0_, float, NZ, M×N)
  │
  │  Fixpipe (dualDstCtl=1, 无 ReLU)
  │  硬件自动拆 M/2:
  │    前 M/2 行 → dst0
  │    后 M/2 行 → dst1 = dst0 + dstNdStride
  ▼
UB (mm1ResUB_, float, ND)
  │
  │  ProcessVec1 (两个 AIV 核各读一半)
  │  AIV0: qkBase = dst0, 读前 M/2 行
  │  AIV1: qkBase = dst1, 读后 M/2 行
  │  ├─ Load QK → regQK
  │  ├─ Relu(regQK)              ← ReLU 在这里（VF 指令）
  │  ├─ MulAddDst(accum, regQK, weight)
  │  └─ Cast → SortableKey → Store
  ▼
scoreGm (GM)
```

---

## 四、目标数据流（CVRATIO=1, dualDstCtl=0, reluEn=true）

```
L0C (cL0_, float, NZ, M×N)
  │
  │  Fixpipe (dualDstCtl=0, reluEn=true)
  │  所有 M 行连续写入一个 dst，随路 ReLU
  ▼
UB (mm1ResUB_, float, ND)  ← 数据已经过 ReLU
  │
  │  ProcessVec1 (一个 AIV 核读全部 M 行)
  │  ├─ Load QK → regQK    ← 已是 ReLU 后的值
  │  ├─ MulAddDst(accum, regQK, weight)  ← 无需再 Relu
  │  └─ Cast → SortableKey → Store
  ▼
scoreGm (GM)
```

---

## 五、数学等价性

ReLU 与正数缩放可交换：

```
Relu(x * c) = Relu(x) * c    (当 c > 0)
```

- **MXFP8 路径**：QK 结果已含 qScale×kScale（mmad_mx 融合），scale 均为正数 → 等价
- **非 MXFP8 路径**：qScale、kScale 为 dequant scale，均为正数 → 等价

**将 ReLU 提前到 Fixpipe 对所有量化模式数学等价。**

---

## 六、修改清单

### 修改 1：Fixp 函数 — dualDstCtl=0 + reluEn=true

**文件**：`op_kernel/arch35/quant_lightning_indexer_service_cube.h`  
**函数**：`QLIMatmul<QLIT>::Fixp` (lines 637-674)

#### 当前代码

```cpp
FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
fixpipeParams.mSize = mSize;
fixpipeParams.srcStride = mSize;
fixpipeParams.dstStride = UB_BANK_DEPTH_STRIDE / sizeof(float);
fixpipeParams.dualDstCtl = 1;  // ← 拆 M 给两个 AIV
```

#### 修改后代码

```cpp
FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
fixpipeParams.mSize = mSize;
fixpipeParams.srcStride = mSize;
fixpipeParams.dstStride = UB_BANK_DEPTH_STRIDE / sizeof(float);
fixpipeParams.dualDstCtl = 0;  // ← 单目标（CVRATIO=1，不需要拆 M）
fixpipeParams.reluEn = true;   // ← 启用随路 ReLU
```

**只需改 1 行 + 加 1 行**。nSize 分支逻辑、dst 地址、dstNdStride 全部保持不变。

---

### 修改 2：Vector 核 qkVLstride 重新计算

**文件**：`op_kernel/arch35/quant_lightning_indexer_service_vector.h`  
**函数**：`QLIVector<QLIT>::ProcessVec1` (line 413)

#### 问题分析

当前 `dualDstCtl=1` 时，后 M/2 行写入 `dst + dstNdStride`，Vector 核用固定的 `qkVLstride` 定位：

```cpp
// 当前（dualDstCtl=1）：
auto qkVLstride = (UB_BANK_DEPTH_STRIDE / sizeof(float)) / 2 * constInfo_.mBaseSize;
//                = 128 / 2 * 256 = 16384
```

改为 `dualDstCtl=0` 后，所有 M 行连续写入一个 dst。后 M/2 行的起始位置变为 `(M/2) * dstStride`，与 `dualDstCtl=1` 时的 `dstNdStride` 可能不同。

#### 需要确认

| 问题 | 说明 |
|------|------|
| dualDstCtl=0 时 Fixpipe 的 UB 布局 | 所有 M 行是否连续？后 M/2 行的偏移是 `(M/2) * dstStride` 还是其他？ |
| qkVLstride 是否需要改为动态值 | 如果 M 总是等于 mBaseSize（256），则 `(M/2) * dstStride = 128 * 128 = 16384`，与当前值相同，**无需修改** |
| 最后一块 M < mBaseSize 的情况 | 如果 M 可变，qkVLstride 需要动态计算 |

#### 如果 M 总是等于 mBaseSize（无需修改）

```cpp
// dualDstCtl=0 时，后 M/2 行偏移 = (mBaseSize/2) * dstStride = 128 * 128 = 16384
// 与当前 qkVLstride = 16384 相同 → 无需修改
auto qkVLstride = (UB_BANK_DEPTH_STRIDE / sizeof(float)) / 2 * constInfo_.mBaseSize;
```

#### 如果 M 可变（需要修改）

```cpp
// 修改前：
auto qkVLstride = (UB_BANK_DEPTH_STRIDE / sizeof(float)) / 2 * constInfo_.mBaseSize;

// 修改后：需要传入实际的 s1gL0RealSize
// 具体值取决于 Fixpipe dualDstCtl=0 时的 UB 布局，需实测确认
```

---

### 修改 3：Vector1 — 移除所有 ReLU 操作

**文件**：`op_kernel/arch35/vf/quant_lightning_indexer_vector1.h`

#### 3A. S1=1 变体：WeightedAccum 传入 ApplyRelu=false

共有 **10 处** `WeightedAccum` 调用需要修改：

| 行号 | 所在函数 | Weight 类型 | Scale |
|------|---------|------------|-------|
| 259 | `MulWeightAndReduceSum` (float weight) | `float*` | 有 |
| 270 | 同上 | `float*` | 有 |
| 354 | `MulWeightAndReduceSum` (half weight) | `half*` | 有 |
| 365 | 同上 | `half*` | 有 |
| 444 | `MulWeightAndReduceSum` (bf16 weight) | `bfloat16_t*` | 有 |
| 455 | 同上 | `bfloat16_t*` | 有 |
| 923 | `MulWeightAndReduceSum_MxFP8` (float weight, MXFP8) | `float*` | 无 |
| 934 | 同上 | `float*` | 无 |
| 999 | `MulWeightAndReduceSum_MxFP8` (bf16 weight, MXFP8) | `bfloat16_t*` | 无 |
| 1010 | 同上 | `bfloat16_t*` | 无 |

**修改方式**（以 line 923 为例）：

```cpp
// 修改前：
WeightedAccum(regSum0, regQK, regwBrc, maskAllB32);

// 修改后：
WeightedAccum<float, 2, false>(regSum0, regQK, regwBrc, maskAllB32);
```

#### 3B. S1=2 变体：删除内联 Relu 调用

共有 **5 个函数**，每个函数内有 **4 条** `Relu` 调用需要删除：

| 行号 | 所在函数 | Weight 类型 | Scale |
|------|---------|------------|-------|
| 556-559 | `MulWeightAndReduceSum2` (float weight) | `float*` | 有 |
| 674-677 | `MulWeightAndReduceSum2` (bf16 weight) | `bfloat16_t*` | 有 |
| 798-801 | `MulWeightAndReduceSum2` (half weight) | `half*` | 有 |
| 1088-1091 | `MulWeightAndReduceSum2_MxFP8` (float weight, MXFP8) | `float*` | 无 |
| 1181-1184 | `MulWeightAndReduceSum2_MxFP8` (bf16 weight, MXFP8) | `bfloat16_t*` | 无 |

**修改方式**（以 lines 1088-1091 为例）：

```cpp
// 修改前：
AscendC::MicroAPI::Relu(regQK0[0], regQK0[0], maskAllB32);
AscendC::MicroAPI::Relu(regQK0[1], regQK0[1], maskAllB32);
AscendC::MicroAPI::Relu(regQK1[0], regQK1[0], maskAllB32);
AscendC::MicroAPI::Relu(regQK1[1], regQK1[1], maskAllB32);
AscendC::MicroAPI::MulAddDst(regSum0[0], regQK0[0], regwBrc[0], maskAllB32);

// 修改后（删除 4 行 Relu）：
AscendC::MicroAPI::MulAddDst(regSum0[0], regQK0[0], regwBrc[0], maskAllB32);
```

---

### 修改 4：CVRATIO 相关适配

CVRATIO 从 2 改为 1 涉及的其他修改（参考 qli-change-to-1-1 skill），与 ReLU 迁移独立，但需要同步完成：

| 文件 | 修改点 | 说明 |
|------|--------|------|
| `quant_lightning_indexer_common.h` | `#define CVRATIO 1` | 核数比值宏 |
| `quant_lightning_indexer_kernel.h` | `aiCoreIdx = tmpBlockIdx / CVRATIO` | 核 ID 映射 |
| `quant_lightning_indexer_service_vector.h` | S1 任务分配、InitBuffers | 不再拆 S1 给两个 AIV |
| `quant_lightning_indexer_service_cube.h` | Event ID 同步（条件编译） | 不再等第二个 AIV |
| `quant_lightning_indexer.cpp` | `KERNEL_TYPE_MIX_AIC_1_1` | 核类型常量 |

---

## 七、修改统计

| 文件 | 修改类型 | 修改处数 |
|------|---------|---------|
| `service_cube.h` (Fixp) | `dualDstCtl=0` + `reluEn=true` | **2 行** |
| `service_vector.h` (ProcessVec1) | qkVLstride（可能无需修改） | **0~1 处** |
| `vector1.h` (S1=1 变体) | WeightedAccum 添加 `<..., false>` | **10 处** |
| `vector1.h` (S1=2 变体) | 删除内联 Relu 调用 | **5 处 × 4 行 = 20 行删除** |
| **总计（ReLU 相关）** | | **~32 行改动** |

---

## 八、性能影响分析

### 7.1 收益

| 项目 | 影响 |
|------|------|
| Vector 核指令减少 | S1=1: 每次 G-loop 省 2 条 Relu VF；S1=2: 省 4 条 Relu VF |
| Fixpipe 开销 | ReLU 是随路操作，**零额外 cycle** |
| Fixpipe 调用次数 | **不变**（仍为 1 次） |
| CVRATIO=1 额外收益 | 减少核间同步开销，简化任务分配 |

### 7.2 风险评估

| 风险项 | 等级 | 说明 |
|--------|------|------|
| UB 布局变化 | 🟡 需验证 | dualDstCtl=0 时后 M/2 行的偏移可能与 dualDstCtl=1 不同，需 Dump UB 确认 |
| qkVLstride 不匹配 | 🟡 需验证 | 如果 M 总是等于 mBaseSize=256，则 qkVLstride=16384 不变；否则需动态计算 |
| 精度等价性 | 🟢 低 | Relu(x*c) = Relu(x)*c (c>0)，数学等价 |
| CVRATIO 适配遗漏 | 🟡 中 | CVRATIO 改为 1 涉及多处修改，需按 qli-change-to-1-1 skill 检查清单逐项完成 |

---

## 九、验证要点

### 8.1 UB 布局验证（最关键）

在 Fixpipe 后 Dump UB 数据，对比 `dualDstCtl=0` 和 `dualDstCtl=1` 的输出：

```
dualDstCtl=1 时：
  dst0[0..M/2-1, 0..N-1]     ← 前 M/2 行
  dst0+dstNdStride[0..M/2-1, 0..N-1]  ← 后 M/2 行

dualDstCtl=0 时（预期）：
  dst[0..M-1, 0..N-1]        ← 所有 M 行连续
  后 M/2 行起始 = dst + (M/2) * dstStride
```

如果 `(M/2) * dstStride == dstNdStride`（当 M=mBaseSize 时成立），则 qkVLstride 无需修改。

### 8.2 边界情况验证

当最后一块 `s1gL0RealSize < mBaseSize` 时：
- `dualDstCtl=1`：硬件仍按 dstNdStride 拆分，Vector 核用固定 qkVLstride 读取
- `dualDstCtl=0`：后 M/2 行偏移 = `(s1gL0RealSize/2) * dstStride`，可能不等于固定 qkVLstride

**如果存在此边界情况，需要在 ProcessVec1 中动态计算 qkVLstride。**

---

## 十、分步实施建议

### Step 1：CVRATIO 适配（前置条件）

按 qli-change-to-1-1 skill 完成 CVRATIO=1 的全部修改，确保算子在 1:1 核配比下正常运行。

### Step 2：仅修改 MXFP8 路径的 ReLU

在 Fixp 中用 `if constexpr (IS_MXFP8)` 分支：
- MXFP8：`dualDstCtl=0` + `reluEn=true`
- 非 MXFP8：保持 `dualDstCtl=0`（CVRATIO=1 已不需要 dualDstCtl=1），`reluEn` 默认 false

对应地，只修改 `*_MxFP8` 函数中的 Relu（6 处 WeightedAccum + 2 处内联 Relu）。

### Step 3：扩展到全路径

MXFP8 验证通过后，去掉 `if constexpr` 分支，所有路径统一启用 `reluEn=true`。

---

## 十一、相关文件索引

| 文件 | 修改区域 | 作用 |
|------|---------|------|
| `op_kernel/arch35/quant_lightning_indexer_service_cube.h` | `Fixp()` L637-674 | Fixpipe 参数 |
| `op_kernel/arch35/quant_lightning_indexer_service_vector.h` | `ProcessVec1()` L413 | qkVLstride（可能需改） |
| `op_kernel/arch35/vf/quant_lightning_indexer_vector1.h` | 全部 12 个计算函数 | 移除 ReLU |
| `op_kernel/quant_lightning_indexer_common.h` | CVRATIO 宏定义 | 核数比值 |
| `op_kernel/quant_lightning_indexer.cpp` | KERNEL_TYPE_MIX_AIC | 核类型常量 |

---

**文档版本**：3.0  
**创建日期**：2026-06-17  
**变更说明**：v3.0 — CVRATIO=1:1，dualDstCtl=0 自然兼容 reluEn，新增 qkVLstride 验证要点和 CVRATIO 适配说明
