# PR 代码概要

PR: #6276 | 算子: attention (common memcopy) | 侧别: Kernel

## 变更文件概览

| # | 文件 | 侧别 | 变更类型 | 说明 |
|---|------|------|---------|------|
| 1 | `attention/common/op_kernel/memcopy/copy_gm_to_l1.h` | Kernel | 修改 | 优化 `CopySingleMatrixNDToNZ` 函数的 else 分支（srcDValue >= 65536 场景），替换硬编码偏移为类型感知计算，简化局部变量 |

## 文件1: `attention/common/op_kernel/memcopy/copy_gm_to_l1.h`（Kernel侧）

### 变更概要

修改 `CopySingleMatrixNDToNZ` 模板函数中 `else` 分支（当 `srcDValue >= 65536` 时触发逐行搬运逻辑）。改动涉及三个维度：

1. **L1 偏移计算泛化**：将硬编码的 `dstNzMatrixStride = 16` 替换为编译期类型感知常量 `BLOCK_ELEMENT_CNT`（`int4b_t` 时为 64，其他类型为 `32 / sizeof(T)`），使代码对 `float`（4字节）和 `int4b_t` 等不同类型的 L1 偏移计算正确。
2. **移除冗余局部变量**：删除 `srcNdMatrixNum`、`srcNdMatrixStride`、`dstNzMatrixStride` 三个中间变量，循环直接使用 `nValue` 作为迭代上限，GM 偏移直接递增 `srcDValue`。
3. **Nd2NzParams.srcDValue 语义修正**：循环体内，每行单次 ND→NZ 搬运时，`nd2nzPara.srcDValue` 从 `srcDValue`（原始大步长）改为 `dValue`（当前行宽度）。由于每轮只搬一行（`nValue=1`），该字段不影响实际数据搬运结果，但语义上更合理——表示单行内"到下一行"的步长为行宽 `dValue`。

### 代码脉络

**入口**: `CopySingleMatrixNDToNZ<T>(l1Tensor, gmTensor, nValue, dValue, srcDValue, dstNzC0Stride)` — 内联模板函数，由多个上层类调用

**数据流**:
```
GlobalMemory (gmTensor) → ND→NZ DataCopy → Local Memory L1 (l1Tensor)
```

**分支覆盖**:
| 分支条件 | 位置(行) | 触发场景 | 处理逻辑 | 涉及 API |
|---------|---------|---------|---------|----------|
| `srcDValue < 65536` | copy_gm_to_l1.h:30 | ND 矩阵步长在硬件单次 DataCopy 允许范围内 | 单次 ND→NZ DataCopy 完成全部搬运 | `DataCopy` |
| `srcDValue >= 65536` (else) | copy_gm_to_l1.h:46 | ND 矩阵步长超限，必须逐行搬运 | 循环 `nValue` 次，每次搬一行（`nValue=1`），逐行更新 `gmOffset`(+=`srcDValue`) 和 `l1Offset`(+=`BLOCK_ELEMENT_CNT`) | `DataCopy` |

**计算核心**: for 循环逐行 ND→NZ DataCopy

**关键变量流转**:
| 变量 | 来源 | 用途 | 流转路径 |
|------|------|------|----------|
| `BLOCK_ELEMENT_CNT` | constexpr 常量（新增） | L1 每次搬运后的偏移增量 | `int4b_t → 64`；其他类型 → `32 / sizeof(T)` |
| `NUM32` | constexpr 常量（新增） | 32B 块大小，用于计算 BLOCK_ELEMENT_CNT | 固定值 32 |
| `nValue` | 函数参数，上层调用者传入 | 循环迭代次数（总行数） | 保持不变，直接用于 `for (i < nValue)` |
| `dValue` | 函数参数 | nd2nzPara.dValue（列数）和 nd2nzPara.srcDValue（行步长） | 传入 `Nd2NzParams`，int4b 时除以 2 |
| `srcDValue` | 函数参数 | gmOffset 增量（GM 中相邻行起始地址间的偏移） | 用于 `gmOffset += srcDValue` |
| `dstNzC0Stride` | 函数参数 | NZ 矩阵 C0 维度步长 | 直接赋给 `nd2nzPara.dstNzC0Stride` |
| `nd2nzPara` | 局部变量（Nd2NzParams 结构体） | ND→NZ 搬运参数封装 | 每轮循环填充后传给 `DataCopy` |

**核心 API**: `DataCopy` (ND→NZ 格式转换搬运)

**输出**: 结果写入 `l1Tensor`（LocalTensor），通过 `l1Offset` 偏移索引逐行写入。无 EnQue/DeQue 同步（本函数是同步 DataCopy，由调用者管理流水线同步）。

### 变量溯源

| 变量 | 声明(文件:行) | 初始化(文件:行) | 校验(文件:行) | 来源类型 |
|------|-------------|----------------|-------------|---------|
| `BLOCK_ELEMENT_CNT` | copy_gm_to_l1.h:48 | 同声明行 `= IsSameType<T, int4b_t>::value ? 64 : NUM32 / sizeof(T)` | 无 | 编译期常量 |
| `NUM32` | copy_gm_to_l1.h:47 | 同声明行 `= 32` | 无 | 编译期常量 |
| `nValue` | copy_gm_to_l1.h:25（函数参数） | 调用者传入 | 无（函数内部无校验，依赖调用者） | 外部输入（上层调用者） |
| `dValue` | copy_gm_to_l1.h:25（函数参数） | 调用者传入 | 无 | 外部输入（上层调用者） |
| `srcDValue` | copy_gm_to_l1.h:26（函数参数） | 调用者传入 | 函数入口 `unlikely(srcDValue < SRCDVALUE_LIMIT)` 判定大小（行 30） | 外部输入，函数入口做范围分支 |
| `dstNzC0Stride` | copy_gm_to_l1.h:26（函数参数） | 调用者传入 | 无 | 外部输入 |
| `HALF_SIZE_DIVISOR` | 未声明（内置）或 copy_gm_to_l1.h:55 | `= 2` (constexpr) | 无 | 编译期常量 |
| `nd2nzPara` | copy_gm_to_l1.h:29 | 局部默认构造 | 无 | 局部变量 |

> 来源类型判定：编译期常量（constexpr）→ 编译期固定；外部输入 → 上层调用者经硬件配置/Offload 数据传递，函数内仅做范围判断分支

## 跨文件关联

### 调用者 → CopySingleMatrixNDToNZ 数据流

本次 PR 不改变函数签名，所有调用者不受影响。但函数内部行为变更会影响所有**触发 else 分支**（`srcDValue >= 65536`）的调用场景。

调用者列表（均位于 Kernel 侧）：
| 调用位置（文件:行） | 调用上下文 | srcDValue 来源 |
|---|---|---|
| `copy_gm_to_l1.h:84` | `CopyMultiMatrixNDToNZ` 内递归调用 | 函数参数 `srcDValue` |
| `copy_gm_to_l1.h:145` | `CopyQueryGmToL1::ProcessS1G` | `offsetCalculator.GetStrideS1()` |
| `copy_gm_to_l1.h:159` | `CopyQueryGmToL1::ProcessS1G` | `offsetCalculator.GetStrideG()` |
| `copy_gm_to_l1.h:178` | `CopyQueryGmToL1::ProcessS1G`（尾块） | `offsetCalculator.GetStrideG()` |
| `copy_gm_to_l1.h:194` | `CopyQueryGmToL1::ProcessContinuous` | `offsetCalculator.GetDimD()` |
| `copy_gm_to_l1.h:232` | `CopyQueryGmToL1::ProcessGS1` | `offsetCalculator.GetStrideS1()` |
| `copy_gm_to_l1.h:251` | `CopyQueryGmToL1::ProcessGS1`（尾块） | `offsetCalculator.GetStrideS1()` |
| `copy_gm_to_l1.h:292` | `CopyKvGmToL1::ProcessContinuousOrTensorlist` | `offsetCalculator.GetStrideS2()` |
| `copy_gm_to_l1.h:327` | `CopyKvGmToL1::ProcessPageAttention` | `offsetCalculator.GetStrideBlockSize()` |
| `copy_gm_to_l1.h:430` | `CopyKKropePAGmToL1::ProcessPageAttention` | `offsetCalculator.GetStrideBlockSize()` |
| `copy_gm_to_l1.h:432` | `CopyKKropePAGmToL1::ProcessPageAttention` | `offsetCalculator.GetStrideBlockSize()` |
| `fia_block_cube_nonquant_gqa.h:822` | FIA 非量化 GQA | `gmStride` |

### Include 链

```
copy_gm_to_l1.h
  ├── gm_layout.h          → GmFormat 枚举、GmLayout 模板特化
  ├── fa_l1_tensor.h       → FaL1Tensor 包装类
  ├── gm_coord.h           → GmCoord / GmKvCoord 坐标结构
  └── offset_calculator_v2.h → OffsetCalculator（提供 GetStrideS1/G/S2/BlockSize 等 stride API）
       ├── gm_layout.h      → GmLayoutParams（FormatCategory 枚举）
       └── parser.h         → ActualSeqLensParser 等（实际序列长度解析）
```

### 常量定义

| 定义位置 | 值 | 引用位置 | 影响范围 |
|---------|-----|---------|---------|
| `copy_gm_to_l1.h:28` `SRCDVALUE_LIMIT` | 65536 | `copy_gm_to_l1.h:30` | 分支条件：决定走单次还是逐行搬运 |
| `copy_gm_to_l1.h:47` `NUM32`（新增） | 32 | `copy_gm_to_l1.h:48` | BLOCK_ELEMENT_CNT 计算 |
| `copy_gm_to_l1.h:48` `BLOCK_ELEMENT_CNT`（新增） | 类型相关 | `copy_gm_to_l1.h:69` | L1 偏移步长控制 |
