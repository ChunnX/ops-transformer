# quant_all_reduce & quant_reduce_scatter 算子全面分析报告

## 1. 算子概述

`quant_all_reduce` 和 `quant_reduce_scatter` 实现 **量化 AllReduce / ReduceScatter**，仅使用 AIV 核 (无 AIC 参与)。通过 MTE (Memory Transaction Engine) 基于 HCCL Window 的 P2P 通信实现跨 Rank 归约，支持 TG (Tensor-Group, 128:1) 和 MX (32:1, FP8_E8M0 scale) 两种量化模式。

```
quant_all_reduce:
  x[BS, H] (量化: INT8/FP8/FP4) + scales[BS, H/128 或 BS, H/64, 2]
      → Dequant (Cast→FP32, Mul with scale)
      → AllReduce (MTE Window P2P, 逐块累加所有Rank)
  输出 out[BS, H] (FP16/BF16/FP32)

quant_reduce_scatter:
  x[BS, H] (量化) + scales
      → Dequant + ReduceScatter (MTE Window, AlltoAll切片后累加)
  输出 out[BS/rankSize, H]
```

**平台**: 仅 ascend950 | **HCCL组**: `"group"` | **数据类型**: 30种x-scales-output组合

---

## 2. 目录结构与文件职责

```
quant_all_reduce/
├── op_kernel/
│   ├── quant_all_reduce.cpp               (1.7K) 内核入口: 模板分发MTE_ONE_SHOT
│   ├── quant_all_reduce_mte_one_shot.h    (7.9K) QuantAllReduceMteOneShot类
│   ├── quant_all_reduce_tiling_data.h     (1.4K) Tiling数据结构
│   └── quant_all_reduce_tiling_key.h      (1.4K) 模板参数: MTE_ONE_SHOT=1
├── op_host/
│   ├── quant_all_reduce_def.cpp           (4.7K) 定义: 2输入/1输出/4属性
│   ├── quant_all_reduce_infershape.cpp    (6.4K) Shape推导
│   └── op_tiling/
│       └── quant_all_reduce_tiling.cpp    (7.3K) Tiling策略
└── op_api/, op_graph/, tests/

quant_reduce_scatter/
├── op_kernel/
│   ├── quant_reduce_scatter.cpp           (1.8K) 内核入口: 模板分发MTE_COMM
│   ├── quant_reduce_scatter_mte.h         (9.1K) QuantReduceScatterMte类
│   ├── mte_comm.h                         (20K)  MTE通信基础设施 (共享)
│   ├── vec_comp.h                         (8.7K) 向量计算引擎 (共享)
│   ├── utils.h                            (2.7K) 工具函数
│   ├── quant_reduce_scatter_tiling_data.h (1.3K) Tiling数据结构
│   └── quant_reduce_scatter_tiling_key.h  (1.4K) 模板参数: MTE_COMM=1
├── op_host/
│   ├── quant_reduce_scatter_def.cpp       (4.7K) 定义: 与all_reduce相同
│   ├── quant_reduce_scatter_infershape.cpp (7.2K) Shape推导
│   └── op_tiling/
│       ├── quant_reduce_scatter_tiling.cpp     (7.4K) Tiling策略
│       └── common/
│           ├── quant_reduce_scatter_util_tiling.h  (3.4K) 共享Tiling声明
│           └── quant_reduce_scatter_util_tiling.cpp (34K)  共享校验逻辑
└── op_api/, op_graph/, tests/

代码复用关系:
  quant_all_reduce → 引用 quant_reduce_scatter/ 下的 mte_comm.h, vec_comp.h, utils.h
```

---

## 3. Host 侧分析

### 3.1 算子定义 (两个算子完全相同)

`quant_all_reduce_def.cpp`（第18行类定义，第22-61行输入声明，第63-67行输出声明，第69-73行属性声明），`quant_reduce_scatter_def.cpp`（结构相同）

```
输入 (2):
  x        REQUIRED  [BS, H]    INT8/HIF8/FP8_E5M2/FP8_E4M3FN/FP4  (30种dtype组合)
  scales   REQUIRED  [BS, ?]    FP32(TG模式) 或 FP8_E8M0(MX模式)    (30种dtype组合)

输出 (1):
  out_put  REQUIRED  [BS, H]    FP16/BF16/FP32  (循环对应x的10组×3种输出类型)

属性 (4):
  group        REQUIRED  String           通信域名称
  reduce_op    OPTIONAL  String("sum")    归约操作
  output_dtype OPTIONAL  Int(DT_BF16)     输出数据类型
  world_size   REQUIRED  Int              通信域大小

MC2配置: HcclGroup("group")
平台: ascend950 (仅此平台)
```

**30种dtype组合**（第22-61行）:
```
每3种输出类型为一组, 共10组:
  INT8 + FP32_scale → FP16/BF16/FP32
  HIFLOAT8 + FP32_scale → FP16/BF16/FP32
  FP8_E5M2 + FP32_scale → FP16/BF16/FP32
  FP8_E4M3FN + FP32_scale → FP16/BF16/FP32
  FP8_E5M2 + FP8_E8M0_scale(MX) → FP16/BF16/FP32
  FP8_E4M3FN + FP8_E8M0_scale(MX) → FP16/BF16/FP32
  FP4_E2M1 + FP32_scale → FP16/BF16/FP32
  FP4_E1M2 + FP32_scale → FP16/BF16/FP32
  FP4_E2M1 + FP8_E8M0_scale(MX) → FP16/BF16/FP32
  FP4_E1M2 + FP8_E8M0_scale(MX) → FP16/BF16/FP32
```

### 3.2 共享校验逻辑

`quant_reduce_scatter_util_tiling.cpp`（34K，两个算子共用）

```
CheckTilingFunc():
  ├── CheckAttrsInfo(): group有效, reduce_op="sum", output_dtype∈{FP16,BF16,FP32}
  ├── SetRankSize(): world_size ∈ {2, 4, 8}
  ├── SetQuantMode():
  │   ├── TG模式: scales.dtype == DT_FLOAT, 128个x共享1个scale
  │   └── MX模式: scales.dtype == DT_FLOAT8_E8M0, 32个x共享1个scale
  ├── CheckTensorDataType(): x和scale类型匹配量化模式
  ├── CheckXShapeValid():
  │   ├── BS % rankSize == 0 (仅ReduceScatter)
  │   ├── H ∈ [1024, 8192], H % 128 == 0
  │   └── 总元素数可被AIV核数整除
  ├── CalculateExpectedScalesShape():
  │   ├── TG: [BS, H/128]
  │   └── MX: [BS, H/64, 2]
  ├── CheckWindowSize(): HCCL buffer足够大
  └── SetWorkSpace(): 16MB
```

### 3.3 Tiling 策略

两个算子的 Tiling 策略由 `quant_reduce_scatter_util_tiling.cpp`（34K）统一提供。核心流程包括参数校验、量化模式检测（TG/MX）、HCCL Window 配置和 Workspace 分配（16MB）。详细的 Tiling 数据结构见 §5.1。

---

## 4. Kernel 侧分析

### 4.1 MTE 通信基础设施

`mte_comm.h` (20K, `QuantMTECommImpl` 命名空间)

```
常量:
  UB_ALIGN_BYTES = 32          // UB对齐
  X_BLOCK_BYTES = 1024         // 每块x数据大小
  STATE_WIN_SIZE = 1MB          // 状态区大小
  SINGLE_STATE_REGION_SIZE = 462KB  // 单个状态区大小
  WIN_ADDR_ALIGN = 512          // Window地址对齐
  BUFFER_NUM = 2                // 双缓冲

MTECommunication<XType, ScalesType, OutputType>:

InitHcclContext():
  ├── hcclContext_ = GetHcclContext<HCCL_GROUP_ID_0>()
  ├── rankId = context->rankId
  ├── rankDim = context->rankDim
  └── 读取Window地址数组

InitParams():
  └── 分配块索引和偏移量给每个AIV核 (aivId_, round_, tailBlockNums_)

InitGMTensor(x, scales, output, alignedXSize, dataSpaceGmSize):
  ├── 设置x/scales/output的GM地址
  ├── 读取并翻转0/1分区标志 (winFlag)
  └── 设置Window数据区和状态区地址

CopyDataToWin<isReduceScatter>():
  ├── [AllReduce=false]: 直接复制全部x+scale到本地Window
  └── [ReduceScatter=true]: AlltoAll模式, 逐Rank切片复制

WriteStatusToWin():
  └── 向所有Rank的Status Area写1 (通过DataCopy)

ReadStatus():
  └── 自旋等待: Sum(Status[0..rankDim-1]) == rankDim

CopyResultToOutput(outOffsetGM, localResultTensor, count):
  └── 从UB复制FP32结果到GM输出 (可选Cast到FP16/BF16)
```

### 4.2 向量计算引擎

`vec_comp.h` (8.7K, `VectorComputeImpl` 命名空间)

```
常量:
  KG_SCALE_TRANS_NUM = 8    // TG量化: fp32 scale, 32B对齐=8个元素
  MX_SCALE_TRANS_NUM = 32   // MX量化: fp8_e8m0 scale, 32B对齐=32个元素
  PER_GROUP_SIZE = 128      // TG: 128个x共享1个scale
  MX_SIZE = 32              // MX: 32个x共享1个scale

VectorCompute<XType, ScalesType, OutputType>:

InitBuffer(TPipe*):
  ├── 分配cast缓冲: xCastBuf_(FP32), xCastBf16Buf_(BF16), xCastfp16Buf_(FP16)
  └── 分配scale缓冲:
      ├── MX模式: tempBf16ScaleBuf_ + castScaleBuf_ (FP8→BF16→FP32路径)
      └── TG模式: castScaleBuf_ (直接FP32)

CastToFloat(xTensor, scaleTensor):
  ├── x转换路径:
  │   ├── FP4 → BF16 → FP32
  │   ├── INT8 → FP16 → FP32
  │   └── FP8 → 直接FP32
  └── scale转换路径:
      ├── MX (FP8_E8M0): CastVf → BF16 → FP32, Brcb广播到32个元素
      └── TG (FP32): ReinterpretCast, Brcb广播到128个元素

DequantReduceSum(xTensor, scaleTensor, sumTensor):
  ├── CastToFloat() (x和scale都转FP32)
  ├── Mul: dequant = x_fp32 × scale_fp32
  └── Add: sumTensor += dequant (跨Rank累加)
```

### 4.3 quant_all_reduce Kernel

`quant_all_reduce.cpp`（第31-43行入口分发到 `QuantAllReduceMteOneShot<XType, ScalesType, OutputType>`）

`quant_all_reduce_mte_one_shot.h` (`QuantAllReduceImpl` 命名空间):

```
常量: X_PRE_BLOCK_NUM = 1024  // 每块处理1024字节x数据

Init()（第68-100行）:
  ├── mteComm_.InitHcclContext()  ← HCCL上下文
  ├── 读取tiling: bs, hiddenSize, scaleHiddenSize, aivNum, totalWinSize
  ├── 计算分块:
  │   ├── xSize_ = bs * hiddenSize * sizeof(XType)
  │   ├── totalBlockNums_ = Ceil(xSize_, X_PRE_BLOCK_NUM * aivNum)
  │   └── tailXNums_ = 余数处理
  ├── UB缓冲初始化:
  │   ├── xInQueue: 1024B (双缓冲)
  │   ├── scaleInQue: 32B (双缓冲)
  │   └── sumBuf_: 4096B (FP32累加缓冲)
  ├── mteComm_.InitParams()     ← 核分配
  ├── mteComm_.InitGMTensor()   ← GM地址
  ├── mteComm_.InitBuffer()     ← Window通信缓冲
  ├── vecComp_.InitBuffer()     ← 向量计算缓冲
  └── vecComp_.SetBlockSize()   ← 设置每块元素数

Process()（第102-125行）:
  ├── mteComm_.CopyDataToWin<false>()   ← 全量复制到Window
  ├── mteComm_.WriteStatusToWin()        ← 通知所有Rank
  ├── mteComm_.ReadStatus()              ← 等待所有Rank完成
  ├── ClearSumTensor()                   ← 清零FP32累加器
  ├── ExecuteAllReduce():
  │   └── for block = 0..totalBlockNums:
  │       └── ReadDataBlockReduceSum(curXOffset, curScaleOffset):
  │           ├── for rank = 0..rankDim (staggered start):
  │           │   ├── 从Rank的Window读取x和scale (1024B x + 32B scale)
  │           │   ├── vecComp_.DequantReduceSum(x, scale, sumTensor)
  │           │   │   └── CastToFloat → Mul(x×scale) → Add(sum)
  │           │   └── 释放队列缓冲
  │           └── mteComm_.CopyResultToOutput(offset, sumTensor, count)
  │               └── FP32 → 输出类型, 写入GM
  └── 结束
```

### 4.4 quant_reduce_scatter Kernel

`quant_reduce_scatter.cpp`（第31-43行入口分发到 `QuantReduceScatterMte<XType, ScalesType, OutputType>`）

`quant_reduce_scatter_mte.h` (`QuantReduceScatterImpl` 命名空间):

```
常量: X_PRE_BLOCK_NUM = 1024, MX_SCALES_LAST_DIM = 2

Init()（第77-100行）:
  ├── 与AllReduce相同的HCCL/tiling/缓冲初始化
  ├── 关键差异 — 计算per-Rank切片:
  │   ├── xSliceSize_ = xSize_ / rankDim         ← 每Rank的数据切片大小
  │   ├── xSliceSizeNums_ = 元素级切片大小
  │   ├── scaleSliceNums_ = scale切片大小
  │   └── MX特有: scaleSize_ *= MX_SCALES_LAST_DIM
  └── 基于切片计算totalBlockNums_

Process():
  ├── mteComm_.CopyDataToWin<true>()    ← AlltoAll切片复制
  │   └── 逐Rank: 将属于该Rank的切片复制到其Window区域
  ├── mteComm_.WriteStatusToWin()         ← 通知所有Rank
  ├── mteComm_.ReadStatus()               ← 等待所有Rank完成
  ├── ClearSumTensor()
  ├── ExecuteReduceScatter():
  │   └── for block = 0..totalBlockNums:
  │       └── ReadDataBlockReduceSum(curXOffset, curScaleOffset):
  │           ├── for rank = 0..rankDim:
  │           │   ├── 只读取属于本Rank的切片:
  │           │   │   offset = curXOffset + rankId * xSliceSize
  │           │   ├── 从Window读取x和scale
  │           │   └── DequantReduceSum: 反量化并累加
  │           └── CopyResultToOutput: 写入输出 [BS/rankSize, H]
  └── 结束
```

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构

**quant_all_reduce_tiling_data.h**（第21-33行）:
```cpp
struct QuantAllReduceTilingInfo {
    uint64_t bs;              // Batch/序列维度
    uint64_t hiddenSize;      // 隐藏层维度
    uint64_t scaleHiddenSize; // Scale隐藏维度
    uint64_t aivNum;          // AIV核数
    uint64_t totalWinSize;    // Window总大小 (=HCCL_BUFFER_SIZE)
};

struct QuantAllReduceTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    QuantAllReduceTilingInfo quantAllReduceTilingInfo;
};
```

`quant_reduce_scatter` 结构完全对称，字段相同。

### 5.2 模板 Key 编码

**quant_all_reduce**: `ASCENDC_TPL_UINT_DECL(quantAllReduceCommMode, 8bit, LIST, MTE_ONE_SHOT=1)`
**quant_reduce_scatter**: `ASCENDC_TPL_UINT_DECL(quantReduceScatterCommMode, 8bit, LIST, MTE_COMM=1)`

各仅有一种组合，分别为 MTE_ONE_SHOT 和 MTE_COMM。

### 5.3 Workspace 布局

```
Workspace 总布局 (16MB):
┌─────────────────────────────────────────────────────────┐
│ HCCL Window 内存 (由 HCCL 框架分配)                      │
│ ├── 每个Rank的Window:                                    │
│ │   ├── Status Region 0 (462KB)                         │
│ │   ├── Status Region 1 (462KB)                         │
│ │   ├── Flag Area (100KB)                               │
│ │   └── Data Area (剩余空间: x_data + scale_data)        │
│ └── Window地址通过 hcclInitTiling 传递给Kernel           │
├─────────────────────────────────────────────────────────┤
│ 系统工作区 (SetSysWorkspace)                              │
└─────────────────────────────────────────────────────────┘
```

Kernel 内部通过 `InitBuffer()` 分配 UB 缓冲（xInQueue 1024B 双缓冲、scaleInQue 32B 双缓冲、sumBuf_ 4096B FP32 累加缓冲）。

---

## 6. 通信架构: MTE Window P2P

### 6.1 Window内存布局

```
每个Rank的Window Memory (共享给其他Rank可见):
┌─────────────────────────────────────────────────────────┐
│ 0x00000                                                 │
│ ┌─── Status Region 0 (462KB) ────────────────────┐     │
│ │ [rank0_status][rank1_status]...[rankN_status]   │     │
│ └─────────────────────────────────────────────────┘     │
│                                                         │
│ ┌─── Status Region 1 (462KB) ────────────────────┐     │
│ │ [rank0_status][rank1_status]...[rankN_status]   │     │
│ └─────────────────────────────────────────────────┘     │
│                                                         │
│ ┌─── Flag Area (100KB) ──────────────────────────┐     │
│ │ 0/1 分区标志 + 完成标志                          │     │
│ └─────────────────────────────────────────────────┘     │
│ 0xFFFF (1MB)                                            │
│                                                         │
│ ┌─── Data Area (剩余空间) ───────────────────────┐     │
│ │ [x_data (量化)][scale_data]                     │     │
│ └─────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘

双缓冲机制:
  第N次调用使用Region 0 (winFlag=0)
  第N+1次调用使用Region 1 (winFlag=1)
  交替使用, 防止数据竞争
```

### 6.2 软同步协议

```
同步流程 (以4 Rank为例):

Step 1: 每个AIV核将本地数据写入Window Data Area
Step 2: 每个AIV核向所有Rank的Status Area写"1"
  Core_X → Rank_0_Status[X] = 1
  Core_X → Rank_1_Status[X] = 1
  Core_X → Rank_2_Status[X] = 1
  Core_X → Rank_3_Status[X] = 1

Step 3: 每个AIV核自旋等待自己的Status Area
  Rank_0 等待: Status[0]+Status[1]+Status[2]+Status[3] == 4

Step 4: 清零Status Area, 开始读取其他Rank的数据
```

### 6.3 AllReduce vs ReduceScatter 数据流对比

```
AllReduce:
  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
  │ Rank 0  │     │ Rank 1  │     │ Rank 2  │     │ Rank 3  │
  │ x[BS,H] │     │ x[BS,H] │     │ x[BS,H] │     │ x[BS,H] │
  └────┬────┘     └────┬────┘     └────┬────┘     └────┬────┘
       │               │               │               │
       ▼               ▼               ▼               ▼
  CopyDataToWin: 每个Rank将全部数据写入本地Window
       │               │               │               │
  WriteStatus → ReadStatus (同步等待)
       │               │               │               │
       ▼               ▼               ▼               ▼
  ExecuteAllReduce:
  对每个block, 从所有Rank的Window读取同一位置 → DequantReduceSum → 累加
       │               │               │               │
       ▼               ▼               ▼               ▼
  out[BS, H]       out[BS, H]       out[BS, H]       out[BS, H]

ReduceScatter:
  ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
  │ Rank 0  │     │ Rank 1  │     │ Rank 2  │     │ Rank 3  │
  │ x[BS,H] │     │ x[BS,H] │     │ x[BS,H] │     │ x[BS,H] │
  └────┬────┘     └────┬────┘     └────┬────┘     └────┬────┘
       │               │               │               │
       ▼               ▼               ▼               ▼
  CopyDataToWin<true>: AlltoAll切片, 每个Rank将第i切片写入Rank i的Window
       │               │               │               │
  WriteStatus → ReadStatus (同步等待)
       │               │               │               │
       ▼               ▼               ▼               ▼
  ExecuteReduceScatter:
  Rank 0只读所有Rank中属于Rank 0的切片 → DequantReduceSum → 累加
       │               │               │               │
       ▼               ▼               ▼               ▼
  out[BS/4,H]      out[BS/4,H]      out[BS/4,H]      out[BS/4,H]
```

---

## 7. 性能相关设计

### 7.1 纯AIV架构

两个算子都使用 `KERNEL_TYPE_AIV_ONLY`:
- 无AIC参与, 所有计算通过Vector核完成
- 反量化使用Cast指令链 (FP4→BF16→FP32, INT8→FP16→FP32)
- 累加在FP32精度下进行, 避免精度损失

### 7.2 双缓冲Window通信

- 状态区分为Region 0和Region 1, 通过`winFlag`自动切换
- 避免了连续两次调用间的数据竞争
- Flag Area存储per-core的双缓冲标识

### 7.3 块级并行

```
每个AIV核处理:
  round_ = Ceil(totalBlocks, aivNum) 个block
  每个block:
    读取1024B x数据 + 32B scale数据
    → DequantReduceSum → 累加到sumBuf_(4096B)
    → 写出结果
```

### 7.4 Staggered Rank访问

AllReduce中跨Rank读取时使用交错起始偏移:
- 不同AIV核从不同Rank开始读取, 避免所有核同时访问同一Rank的Window

### 7.5 统一量化框架

支持5种x数据类型 + 2种量化粒度:
```
TG (Tensor-Group): 128个元素共享1个FP32 scale
  → x: INT8/HIF8/FP8_E5M2/FP8_E4M3FN → scales: FP32

MX (Microscaling): 32个元素共享1个FP8_E8M0 scale
  → x: FP8_E5M2/FP8_E4M3FN/FP4_E2M1/FP4_E1M2 → scales: FP8_E8M0
```

---

## 8. 依赖关系

```
quant_all_reduce/
├── op_kernel/
│   ├── quant_all_reduce.cpp → 引用 quant_reduce_scatter/ 下的共享代码
│   │   ├── ../quant_reduce_scatter/op_kernel/mte_comm.h  ← MTECommunication
│   │   ├── ../quant_reduce_scatter/op_kernel/vec_comp.h  ← VectorCompute
│   │   └── ../quant_reduce_scatter/op_kernel/utils.h     ← 工具函数
│   └── quant_all_reduce_mte_one_shot.h
│       └── 使用 MTECommunication + VectorCompute 组合
└── op_host/
    └── op_tiling/quant_all_reduce_tiling.cpp
        └── 引用 quant_reduce_scatter/op_tiling/common/ 下的共享校验

quant_reduce_scatter/
├── op_kernel/
│   ├── mte_comm.h        ← MTE通信基础设施 (被quant_all_reduce复用)
│   ├── vec_comp.h         ← 向量计算引擎 (被quant_all_reduce复用)
│   ├── utils.h            ← 工具函数
│   └── quant_reduce_scatter_mte.h
└── op_host/
    └── op_tiling/common/
        └── quant_reduce_scatter_util_tiling.cpp  ← 共享校验 (34K)

两个算子对比:
  ┌────────────────┬───────────────────┬──────────────────────┐
  │ 特性           │ quant_all_reduce  │ quant_reduce_scatter  │
  ├────────────────┼───────────────────┼──────────────────────┤
  │ 通信模式       │ AllReduce(全量)    │ ReduceScatter(分片)  │
  │ 输出形状       │ 同输入 [BS, H]     │ [BS/rankSize, H]     │
  │ Window写入     │ 全量复制           │ AlltoAll切片复制      │
  │ Window读取     │ 同偏移读所有Rank   │ 只读本Rank切片       │
  │ 模板常量       │ MTE_ONE_SHOT=1    │ MTE_COMM=1           │
  │ 计算核心       │ 纯AIV             │ 纯AIV                │
  │ HCCL配置       │ AllToAll          │ AllToAll             │
  │ MX scale处理   │ 标准路径          │ 额外MX_SCALES_LAST_DIM│
  └────────────────┴───────────────────┴──────────────────────┘
```

---

*分析完成时间: 2026-04-20，修订时间: 2026-04-21（修正章节编号规范、补充行号引用、新增§5 Host-Kernel接口章节）*
