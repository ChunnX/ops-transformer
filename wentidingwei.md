# AllGatherMatmulV2 与 MatmulReduceScatterV2 交替批跑报错根因分析

## 一、问题现象

在 **动/静态图 + perblock量化(hif8)** 场景下，AllGatherMatmulV2 与 MatmulReduceScatterV2 交替批跑时，左矩阵为 `[3200, 18507]` 会报错，而 `[1024, 18507]` 和 `[4096, 18507]` 不会报错。

## 二、根因定位

**根因在 MatmulReduceScatterV2 的 A2A+VecReduce perblock 串行路径中，AlltoAll通信量与 PostProcess ReduceSum 处理量之间存在 8 倍不匹配，导致缓冲区越界访问。**

### 2.1 关键数学关系

perblock量化要求 M 对齐到 128。两个算子对 M 的对齐要求不同：

| 算子 | 对齐要求 | 检查位置 |
|------|---------|---------|
| AllGatherMatmulV2 | `M_local % 128 == 0` | `all_gather_quant_bmm_tiling.cpp:144` (`CheckX1Input`) |
| MatmulReduceScatterV2 | `M_full % (128 × rankDim) == 0` | `quant_bmm_reduce_scatter_tiling.cpp:436` (`CheckPerblockM`) |

对于不同的 M（假设 rankDim=8，A2A通路用于4P/8P on A5）：

| M | M % 128 | M % (128×8) | 是否走串行路径 | 是否报错 |
|---|---------|-------------|--------------|---------|
| 1024 | 0 ✓ | 0 ✓ | 否（正常路径） | 不报错 |
| 3200 | 0 ✓ | **128 ≠ 0** ✗ | **是（串行路径）** | **报错** |
| 4096 | 0 ✓ | 0 ✓ | 否（正常路径） | 不报错 |

- `3200 = 25×128`，满足 AllGatherMatmulV2 的对齐要求（`3200%128=0`），所以 AllGatherMatmulV2 正常工作。
- 但 `3200 % (128×8) = 128 ≠ 0`，不满足 MatmulReduceScatterV2 perblock 的 `CheckPerblockM()` 条件，触发**串行降级路径**。

### 2.2 串行路径触发逻辑

`quant_bmm_reduce_scatter_tiling.cpp:447-458`:

```cpp
// 在perblock量化场景，orgMValue不满足PERBLOCK_SIZE * args_.rankDim整数倍时，
// 不做公式化切分，走计算通信串行
if (...(quantMode_ == PERBLOCK_MODE && CheckPerblockM())) {
    MC2_CHECK_LOG_RET(opName_, DoSplitMTiling(MutableRCSTilingDataA5()));
}
// ...
if ((GetQuantMode() == PERBLOCK_MODE) && (tileCnt == 0) && (tailCnt == 0)) {
    MutableRCSTilingDataA5().tileCnt = 1;  // 串行：仅1轮
}
```

当 `CheckPerblockM()` 返回 false 时：
1. `DoSplitMTiling()` **不被调用** → `tileMValue_` 保持 0
2. `DoAdaptSlidWindowTiling()` 中 `tempMValue = rankM = orgMValue = 3200`（完整M）
3. `tileCnt = 1`（串行单轮）
4. matmul tiling 的 `tiling.M = 3200`（完整M）

### 2.3 缓冲区越界的具体机制

在 A2A+VecReduce 内核 `quant_bmm_a2a_vec_reduce_fp8_hif8.h` 中，串行路径的执行流程为：

**步骤1：MatMul + AlltoAll**（`MatMulReduceScatterSerial()`, L216-248）

```cpp
// AlltoAll 每张卡发给每张卡的元素数 = (M/rankDim) * N
uint64_t recvCount = static_cast<uint64_t>(tiling.M / cfg.rankDim)
                   * static_cast<uint64_t>(tiling.N);  // = (3200/8)*N = 400*N
// ...
handles_[0] = hccl_.template AlltoAll<true>(
    sendBuffer, recvBuffer, recvCount, ...);  // 每卡收发 400*N
```

AlltoAll 后，`recvBuffer` 中共有 `rankDim × 400×N = 8×400×N = 3200×N` 个元素。

**步骤2：PostProcess ReduceSum**（L148-199）

```cpp
const uint64_t tileElemCount = static_cast<uint64_t>(tiling.M)
                             * static_cast<uint64_t>(tiling.N);  // = 3200*N  ← 错误！
// ...
reduceSum_.Init(tileElemCount, 0, cfg.rankDim, aivNum_, curSrcPtr, curOutPtr, tPipe_);
//                ↑ 3200*N          ↑ 8      → 期望读取 8 × 3200×N = 25600×N 个元素
```

**不匹配对比：**

| 项目 | AlltoAll 实际产生 | ReduceSum 期望读取 | 倍数差 |
|------|------------------|-------------------|-------|
| 元素数 | `rankDim × (M/rankDim) × N = 3200×N` | `rankDim × M × N = 8×3200×N = 25600×N` | **8倍** |

ReduceSum 尝试从 `recvBuffer`（仅有 `3200×N` 个元素）中读取 `25600×N` 个元素，造成 **8 倍缓冲区越界读取**。

### 2.4 正常路径为何不出错

在正常（非串行）perblock 路径中，`tiling.M = tileMValue_`（按卡切分后的M，如128或512）：

```cpp
// AlltoAll: recvCount = tiling.M * N = tileMValue_ * N（不除以rankDim）
const uint64_t recvCount = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N);
// PostProcess: tileElemCount = tiling.M * tiling.N = tileMValue_ * N
```

AlltoAll 产生 `rankDim × tileMValue_ × N`，ReduceSum 读取 `rankDim × tileMValue_ × N`，**完全匹配**。

### 2.5 非 A2A 通路（ReduceScatter）串行路径无此问题

`quant_bmm_reduce_scatter_fp8_hif8.h` 的串行路径使用原生 `ReduceScatter`（L154-156），PostProcess 仅 Wait + Finalize（L111-122），**无 ReduceSum 步骤**，因此不存在越界问题。但 4P/8P 在 A5 上走 A2A 通路（`mc2_tiling_utils.h:292-296`），故此场景必现。

### 2.6 为何"两算子交替批跑"才复现

- 串行路径的越界读取是否导致可见报错，取决于 `recvBuffer` 之后的内存是否被映射。
- 单独运行 MatmulReduceScatterV2 时，越界读取可能落在 workspace 的合法分配区域内（`workspaceGM_` 紧随 `recvBuf_` 之后），不一定立即崩溃。
- **交替批跑**时，AllGatherMatmulV2 与 MatmulReduceScatterV2 的 workspace 交替分配/释放，整体内存布局变化，使越界读取更容易命中未映射/受保护内存，从而触发报错。

## 三、问题代码位置

| 文件 | 行号 | 问题描述 |
|------|------|---------|
| `quant_bmm_a2a_vec_reduce_fp8_hif8.h` | **L226** | 串行路径 AlltoAll 使用 `tiling.M / cfg.rankDim`（每卡切片大小） |
| `quant_bmm_a2a_vec_reduce_fp8_hif8.h` | **L168** | PostProcess ReduceSum 使用 `tiling.M`（完整M），与 AlltoAll 不匹配 |
| `quant_bmm_a2a_vec_reduce_fp8_hif8.h` | **L259-263** | 串行路径判断条件 `tiling.M == cfg.rankM` |

## 四、修复建议

在 `quant_bmm_a2a_vec_reduce_fp8_hif8.h` 的 **PostProcess** 中，针对串行路径使用正确的元素计数：

```cpp
// 修复方案：串行路径下 ReduceSum 的 elemCount 应为 (tiling.M / rankDim) * tiling.N
const uint64_t tileElemCount = (tiling.M == cfg.rankM)  // 串行路径
    ? static_cast<uint64_t>(tiling.M / cfg.rankDim) * static_cast<uint64_t>(tiling.N)
    : static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.N);
```

或者更彻底的方案：在串行路径中不使用通用的 PostProcess，而是单独处理 ReduceSum 参数，确保与 AlltoAll 的 `recvCount` 一致。

## 五、总结

```
根因链路：
M=3200, rankDim=8
  → 3200 % (128×8) = 128 ≠ 0
  → CheckPerblockM() 返回 false
  → 跳过 DoSplitMTiling，进入串行路径 (tileCnt=1)
  → A2A内核串行路径: AlltoAll 用 (M/rankDim)×N 通信
  → PostProcess ReduceSum 用 M×N 计算 (8倍于实际数据)
  → 缓冲区越界读取 8 倍数据
  → 交替批跑时内存布局变化，越界访问触发报错

为何 1024/4096 不出错：
  → 1024 % 1024 = 0, 4096 % 1024 = 0
  → 走正常 perblock 路径
  → AlltoAll 与 ReduceSum 参数一致，无越界
```

---

## 六、深入分析：为何单算子执行无问题，仅图模式报错

### 6.1 ReduceSum 越界读取已确认

通过阅读 `ReduceSumForAlltoAll` 的完整实现（`reduce_sum_cast_fp32.h`），确认了越界读取的具体机制：

```cpp
// reduce_sum_cast_fp32.h:210-239  ExecuteReduceSum()
for (uint64_t curBlock = 0; curBlock < assignedBlockNums_; ++curBlock) {
    Duplicate<float>(sumTensor_, 0.0, perBlockNum_);  // 清零累加器
    for (uint64_t curRank = 0; curRank < rankDim_; ++curRank) {  // ← 遍历所有 rank
        uint64_t srcSliceOffset = curRank * strideSize_;  // stride=0时, strideSize_=sliceSize_=tiling.M*N
        uint64_t curOffset = srcSliceOffset + curBlockOffset;
        ReadDataBlockReduceSum(curOffset, copyBlockNum);  // 从 recvBuf_ 读取数据
    }
    CopyResultToOutput(curBlockOffset, sumTensor_, copyBlockNum, isTailBlock);
}
```

- `sliceSize_ = outputSize = tileElemCount = tiling.M * N`（Init 参数中传入）
- `strideSize_ = sliceSize_`（因 stride=0，见 `InitParams` L132）
- 内层循环遍历 `rankDim_`(=8) 个 rank，每个 rank 读取 `sliceSize_` 个元素
- **总读取量 = `rankDim_ × sliceSize_ = 8 × tiling.M × N`**

而串行路径 AlltoAll 仅向 `recvBuf_` 写入 `tiling.M × N` 个元素（总量），因此越界读取 `7 × tiling.M × N` 个元素。

### 6.2 Workspace 缓冲区布局

内核 `Init()` 中（`quant_bmm_a2a_vec_reduce_fp8_hif8.h:134-137`）将 workspace 三等分：

```
workspaceGM 布局 (A2A路径, factor=3):
┌─────────────────────────┬─────────────────────────┬─────────────────────────┐
│      sendBuf_           │      recvBuf_           │      workspaceGM_       │
│  [0, cToFloatLen)       │  [cToFloatLen,          │  [2*cToFloatLen,        │
│                         │   2*cToFloatLen)        │   3*cToFloatLen)        │
│  MatMul输出/All2All发送  │  All2All接收/ReduceSum源 │  MatMul内部workspace     │
└─────────────────────────┴─────────────────────────┴─────────────────────────┘

其中 cToFloatLen = AlignUp(rankM * rankN * outputDtypeSize, 512)
     ≈ tiling.M * N * sizeof(CType)   (串行路径下 tiling.M = rankM = 3200)
```

**越界范围分析（M=3200, rankDim=8）：**

| 区域 | 偏移范围 | 大小 | ReduceSum 是否访问 |
|------|---------|------|------------------|
| recvBuf_ 有效数据 | `[cToFloatLen, 2*cToFloatLen)` | 1×cToFloatLen | ✅ 合法（rank0 切片） |
| recvBuf_ 尾部对齐 | 约 0~512 字节 | ~0 | ✅ 合法 |
| workspaceGM_ | `[2*cToFloatLen, 3*cToFloatLen)` | 1×cToFloatLen | ⚠️ 越界（rank1 切片） |
| **workspace 之后** | `[3*cToFloatLen, 8*cToFloatLen)` | **5×cToFloatLen** | ❌ **严重越界（rank2~7 切片）** |

ReduceSum 期望从 `recvBuf_` 读取 `8 × cToFloatLen` 字节，但 `recvBuf_` 仅含 `1 × cToFloatLen` 字节有效数据。越界的 `7 × cToFloatLen` 字节中，仅 `1 × cToFloatLen` 落在 workspaceGM_ 内，其余 **`5 × cToFloatLen` 完全超出 workspace 边界**。

### 6.3 单算子执行（aclnn）为何不报错

单算子执行通过 aclnn 两段式 API 调用，其内存管理特征如下：

**（1）Workspace 独立分配与释放**

```
单算子执行流程:
  aclnnMatmulReduceScatterV2GetWorkspaceSize()  → 获取 workspaceSize
  aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST)  → 独立分配
  aclnnMatmulReduceScatterV2(workspace, ...)  → 执行
  aclrtFree(workspace)  → 独立释放
```

每次调用独立分配/释放 workspace，**两个算子之间不存在内存复用**。

**（2）HCCL 上下文隔离**

aclnn 模式下，HCCL engine context 的 tag 为 `groupEp + opName`（`mc2_context.cpp:506`）。由于 AllGatherMatmulV2 和 MatmulReduceScatterV2 的 opName 不同，它们获得**不同的 HCCL 上下文**，互不影响。

**（3）越界读取的后果：静默数据损坏**

在单算子模式下，越界的 `5 × cToFloatLen` 字节读取超出 workspace 的内存。在 NPU 设备上，这种越界读取的后果取决于后续内存的映射状态：

- **若后续内存被映射（如大页余量、其他分配块）**：ReduceSum 读取到垃圾数据，累加后输出到 `cGM_`（算子输出）。结果是**精度错误但不会崩溃**。若调用方未校验精度，则表现为"无问题"。
- **若后续内存未映射**：理论上应触发硬件异常。但 NPU 的内存保护机制与 CPU 不同，部分情况下越界访问可能被硬件静默处理（返回 0 或随机值），不产生显式错误码。

**关键结论**：单算子模式下，越界读取产生的垃圾数据仅污染**当前算子的输出**，不影响其他算子。若不校验精度，表现為"执行成功"。

### 6.4 图模式为何报错

图模式下，GE（Graph Engine）统一管理内存和调度，存在三个关键差异：

**（1）内存复用导致跨算子数据损坏**

GE 的图内存管理器会分析算子间的生命周期重叠关系，对生命周期不重叠的算子**复用同一块物理内存**。当 AllGatherMatmulV2 和 MatmulReduceScatterV2 交替执行时：

```
图模式内存布局（简化）:
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ AG-MM output │ MM-RS input  │ MM-RS ws     │ AG-MM ws     │
│ (=MM-RS input)│(=AG-MM out) │ (含recvBuf_) │ (可能被复用) │
└──────────────┴──────────────┴──────────────┴──────────────┘
                                ↑
                         ReduceSum 从 recvBuf_ 越界读取
                         5×cToFloatLen，覆盖右侧所有区域
```

ReduceSum 的越界读取直接覆盖了：
- **AG-MM 的 workspace** → 下一次 AG-MM 执行时读取到被污染的数据
- **AG-MM 的输出缓冲** → 下一次 AG-MM 写入时地址已被破坏
- **HCCL 通信缓冲区** → 通信数据被污染

这导致后续算子执行时出现**计算异常、通信错误或地址非法访问**，从而产生显式报错。

**（2）共享通信流导致错误传播**

两个算子使用相同的 `reuse_key`（`"ccu_stream"`，见 `mc2_gen_task_ops_utils.cpp:74`），GE 为它们分配**同一条附加通信流**。HCOM 通信句柄也作为 hidden input 在图级别共享（`mc2_gen_task_ops_utils.cpp:116`）。

当 MM-RS 的越界读取污染了通信相关内存后，同一流上的 AG-MM 通信操作也会受到影响，错误在两个算子间传播。

**（3）SetScheduleMode(1) 的独占全核机制**

两个算子均设置 `SetScheduleMode(1)`（独占全核模式），意味着它们**严格串行执行**，不能并行重叠。这反而使得越界损坏的数据**必定被下一个算子读取到**——因为下一个算子必须等待当前算子完成后才启动，此时被污染的内存尚未被恢复。

### 6.5 对比总结

| 维度 | 单算子执行 (aclnn) | 图模式 (GE) |
|------|-------------------|-------------|
| **Workspace 分配** | 每次独立 `aclrtMalloc`/`aclrtFree` | GE 统一管理，支持跨算子内存复用 |
| **内存布局** | workspace 独立隔离，越界读取不影响其他算子 | workspace 紧邻排列，越界读取直接覆盖相邻算子数据 |
| **HCCL 上下文** | 按 `groupEp + opName` 隔离，两算子独立 | 共享 `reuse_key` 和 HCOM 句柄，错误可传播 |
| **越界后果** | 垃圾数据仅污染当前算子输出，不校验精度则"无问题" | 垃圾数据污染相邻算子的输入/workspace，导致后续算子报错 |
| **错误可见性** | 低（精度错误不崩溃） | 高（数据损坏导致后续算子计算/通信异常） |

### 6.6 AllGatherMatmulV2 perblock 路径无此问题

AllGatherMatmulV2 的 perblock 实现（`all_gather_quant_bmm_perblock.h`）不存在类似串行路径缺陷：

- **无串行降级路径**：AllGatherMatmulV2 使用 `PostDoSplitMTiling()`（`all_gather_matmul_tiling_base.cpp:550-566`）对 tileM 进行 128 对齐调整，始终执行 M 切分，不会出现"不切分、全M串行"的情况。
- **通信方式不同**：AllGatherMatmulV2 使用 AllGather 通信（`HcclStart`/`HcclWait`），通信后直接进行 MatMul 计算，**无 PostProcess ReduceSum 步骤**，不存在通信量与后处理量不匹配的问题。
- **对齐要求更宽松**：仅要求 `M % 128 == 0`（`CheckX1Input`），不要求 `M % (128 × rankDim) == 0`。M=3200 满足此条件，走正常 perblock 路径。

因此，**报错根因完全在 MatmulReduceScatterV2 的 A2A+VecReduce 串行路径**，AllGatherMatmulV2 本身工作正常，但作为交替执行的"受害者"被波及。
