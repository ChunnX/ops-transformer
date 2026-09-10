# CCU AllGatherMatmul 算子方案设计（三阶段）

> **场景**：基于 CCU 通信的 AllGather + Matmul 通算融合算子，`<<<>>>` 直调，两卡开发。
> **核心参考**：`/home/wuhua/code/asc-comm/examples/ccu/ccu_direct/02_allgather_add/`（CCU 双 kernel 架构，通信层+Host编排+同步机制整体复用，matmul 替换其中的 add）。
> **matmul 计算**：Blaze 库纯 matmul（FP16→BF16），baseM=baseN=baseK=256。
> **状态**：第一阶段编译通过 + 运行验证 PASS（commit `f7c94cf15`，2025-08-17）。2 卡 AllGather 通信正确，recvBuf 前 65536==1.0 后 65536==2.0，两卡一致。

---

## 交付阶段总览

| 阶段 | 步骤 | 目标 | 核验内容 | 同步架构 | 交付物 |
|------|------|------|---------|---------|--------|
| **一** | Step 1 | CCU AllGather 通信跑通 | 不崩溃、不卡死（不关注精度） | CCU + AIV 同步桥 | `ccu_all_gather_matmul_kernel.asc` |
| | Step 2 | 通信精度验证 | recvBuf 拼接顺序与内容正确 | （同上） | run.sh + 可选 verify_result.py |
| **二** | — | AllGather + Matmul 串联端到端精度 | C = concat(A) @ B，BF16 rtol/atol=1e-3 | CCU + AIV 同步桥 + CrossCoreSetFlag 唤醒 AIC | 叠加 Blaze matmul kernel |
| **三** | — | 通算并行流水 + AIC 直轮询 | 两次 allgather + 两次 matmul overlap；与二阶段精度对齐 + 性能对比 | CCU + AIC 直轮询（去 AIV 桥） | AIC 直轮询 + 多 round 流水 kernel |

> **严格串行**：每阶段/每步验证通过才进下一阶段/步，禁止跨阶段开发。前两步仅针对极简 case，不要求泛化。
> **极简 case**：2 卡，M=256, K=256，第一阶段 dtype=FP16；第二阶段起 N=256，FP16→BF16。

---

## 0. 关键决策与边界澄清

### 0.1 通信路线选择：CCU 直调（非 apace 现有 UDMA）

**重要发现**：当前工作区 `apace/` 已有 `all_gather_mx_matmul_udma_impl.h`，它走的是 **UDMA/SHMEM（hcomm）** 通信路线，由 AIV 核发起通信，AIV/AIC 同核 MIX 协同。本方案需求与之**不同**：

| 维度 | apace 现有 UDMA 路线 | 本方案 CCU 路线 |
|------|---------------------|----------------|
| 通信发起单元 | AIV 核（CollectiveComm API） | **CCU 微核**（`__ccu_host__` kernel） |
| 通信库 | hcomm UDMA + asccomm | **hcomm CCU + asccomm_ccu_dataplane** |
| kernel 拉起 | 单个 MIX kernel（`__global__ __aicore__`） | **双 kernel**（CCU `<<<>>>` + AIV/AIC `<<<>>>` 双流） |
| 同步 | CrossCoreSetFlag/WaitFlag（核内） | **GM 轮询 + CKE Event**（跨异构单元） |
| 参考样例 | apace 自身 ST | `02_allgather_add/main.asc` |

**结论**：本方案以 `02_allgather_add` 为骨架，CCU 通信侧整体复用其通信 kernel + Host 编排 + 同步机制；计算侧用 Blaze matmul 替换 `vector_add`。apace 现有 UDMA 代码**仅作 matmul kernel 实现参考**，通信层不复用。

### 0.2 matmul kernel 选型：Blaze GemmUniversal（纯 matmul FP16→BF16，无量化）

**已确认 P1**：走 Blaze 库纯 matmul 路线，**FP16 输入 A/B，FP32 累加，BF16 输出 C**（Blaze 标准 dtype 组合）。无 scale 通信。

- `DispatchPolicy = Matmul`（非 `MatmulWithScaleMx`，无需 scale 输入）
- `BlockMmad`、`BlockScheduler`、`GemmUniversal` 来自 `blaze/gemm/`
- matmul kernel 类型链参考 Blaze matmul 模板（`matmul_blaze_template/op_kernel/matmul_blaze_kernel.h`）+ 工作区 apace `all_gather_qbmm_mx_kernel.h` 的 Blaze 用法
- baseM=baseN=baseK=256（用户指定，host tiling 填充）

> **参考边界**：通信层（CCU kernel + Host 编排 + 同步）以 `02_allgather_add` 为核心参考，逐函数复用；matmul 计算层以 Blaze 库为参考（Blaze 模板 + apace 现有 Blaze 用法）。两者通过 GM（recvBuf）解耦。

### 0.3 三核协同架构：CCU 通信 + AIV 同步桥 + AIC 计算

**已确认**：保留 AIV 做同步桥（与参考样例 02_allgather_add 完全一致），不尝试 AIC 直接轮询。

`02_allgather_add` 中 AIV 承担"写参数→唤醒CCU→轮询→计算"全链路。本方案 matmul 走 AIC（cube 核），但**同步桥仍由 AIV 承担**，AIV 通信完成后通过 `CrossCoreSetFlag` 唤醒 AIC matmul。原因：
- CCU Variable 的 GM 地址**必须从 AIV 核上下文写入**（同一 DIE 地址空间，Host 无法直接写）—— ARCHITECTURE.md §5 设计要点第 1 条
- `WriteGmByPassDCache`/`ReadGmByPassDCache`/`Nop`/`DataSyncBarrier` 均为 AIV 已验证 API，AIC 上未验证（原风险 P7）

**为何必须引入 AIC（而非在 AIV 上跑 matmul）**：matmul 是 Cube 运算，Blaze 库的 `BlockMmad`/`GemmUniversal` 基于 AIC cube 核的 MMAD 指令，AIV（向量核）无法执行 Cube 指令。因此 matmul 必须在 AIC 上运行，引入第三核是计算需求决定的，非可选。

**同步方式分阶段演进（消除矛盾，明确路径）**：
- **第一/二阶段**：AIV 做同步桥（写参数→唤醒CCU→轮询→CrossCoreSetFlag 唤醒 AIC），与参考样例 AIV 同步链路一致，仅新增 AIV→AIC 唤醒一步。AIC 不做轮询，仅 `CrossCoreWaitFlag` + matmul。
- **第三阶段**：切换为 AIC 直轮询（AIC 接管参数桥+唤醒CCU+轮询全链路，去掉 AIV 同步桥）。这是对参考样例同步机制的最大偏离，但用户已确认 AIC 支持相关 API，且通算并行流水下 AIV 中转的每 round 延迟会吃掉 overlap 收益——AIC 直轮询收益此时才显现。第三阶段引入是性能驱动，前两阶段已建立正确性基线，风险可控。

**受控偏离说明**：第二阶段引入的 `CrossCoreSetFlag`/`CrossCoreWaitFlag`（AIV→AIC 唤醒）是参考样例没有的新增机制（参考样例是纯双核双流 CCU+AIV，无 AIC）。此偏离的必要性在于 matmul 必须走 AIC，而 AIV 同步桥完成后需通知 AIC。可行性佐证：apace UDMA 路线 `all_gather_mx_matmul_udma_impl.h` L248 已用 `CrossCoreSetFlag`（P7）。此为受控偏离，非创新——机制本身是 AscendC 标准跨核同步 API。

三流三核结构：

| 流 | Kernel | 执行单元 | 职责 |
|---|---|---|---|
| `streamCcu` | `CcuAllGatherKernel` | CCU 微核 | AllGather 通信 |
| `streamAiv` | `comm_sync_kernel` | AIV 向量核 | 写参数桥 → 唤醒CCU → 轮询通信完成 → `CrossCoreSetFlag` 唤醒AIC |
| `streamAic` | `matmul_kernel` | AIC 立方核 | `CrossCoreWaitFlag` 等待 → Blaze matmul |

> 第一阶段（无 matmul）：AIV 同步桥（comm_sync）轮询到通信完成后直接结束，不做数据搬运。Host 从 recvBuf 读回数据落地 bin 文件校验。第二阶段起引入 AIC matmul，AIV 同步桥额外做 `CrossCoreSetFlag` 唤醒 AIC。

### 0.4 编译验证状态

第一阶段已编译通过（2025-08-17，commit `f7c94cf15`）。开发过程中的关键编译踩坑：

- **half 软浮点符号缺失**：Host 侧用 `half` 类型导致 `__truncsfhf2`/`__extendhfsf2` 链接错误。`--rtlib=compiler-rt` 链接失败（环境无此库）。最终方案：Host 侧全程 `uint16_t` 承载 FP16 数据（对齐 apace ST），不打印数值，结果落 bin 文件用 Python 校验。
- **UB 容量超限**：原始 verify_copy 设计的 UB 搬运（256KB）超 248KB 上限。改为 comm_sync（不做数据搬运），从根源消除。

> 详见 `CCU_ALLGATHER_PHASE1_DESIGN.md` §0 实现演进记录。

---

## 1. 整体架构（目标态，三阶段完成后）

### 1.1 数据流（第二阶段起，含 matmul）

```
各 rank 的 sendBuf(M×K 分片)
        │
   CCU AllGather (streamCcu, __ccu_host__)
        ▼
recvBuf[rank0 M×K | rank1 M×K | ... ]  (rankSize×M 行 × K 列，拼接)
        │  (AIV 轮询 ckeAddr1==1 同步)
        ▼
   AIV 同步桥 (streamAiv, __vector__)
        │  CrossCoreSetFlag 唤醒 AIC
        ▼
   AIC matmul (streamAic, __cube__)
    A = recvBuf 拼接结果 (rankSize*M, K)
    B = 本 rank 权重 (K, N)
    C = 输出 (rankSize*M, N)
        ▼
   computeBuf / cGM
```

> 第一阶段（无 matmul）：AIV 同步桥轮询到通信完成后直接结束（comm_sync，不做数据搬运）。Host 从 recvBuf 读回数据落地 bin 文件校验，不唤醒 AIC。

### 1.2 三核同步机制（AIV 同步桥，复用 02_allgather_add）

```
AIV 同步桥 (streamAiv)            CCU (streamCcu)              AIC (streamAic)
──────────────────                ──────────────                ──────────────
launch comm_sync_kernel           launch ccu_allgather_kernel   launch matmul_kernel
  写 taskArgs[0..9] → GM          EventWait(events[0]) ←阻塞┐    CrossCoreWaitFlag ←阻塞┐
  DataSyncBarrier                   │                        │                          │
  写 ckeAddr0 = 1 ───────────────→ │ (唤醒)                  │                          │
                                    ↓                        │                          │
  轮询 ckeAddr1 ←─┐                AllGather 传输            │                          │
    Nop<800>      │                EventRecord(events[1])    │                          │
    Read again    │                (CCU 结束)                │                          │
  ckeAddr1==1 ────┘─────────────── 写 eventVa[1]=1           │                          │
  CrossCoreSetFlag ─────────────────────────────────────────────────────────────────→ (唤醒)
  [第一阶段: 无数据搬运，AIV 直接结束]                          Blaze matmul(A,B→C)
  (AIV 结束)                                                    (AIC 结束)
```

**同步链路（4 步）**：
1. AIV 写 taskArgs → GM（参数桥：通信参数送 CCU）+ `DataSyncBarrier`
2. AIV 写 ckeAddr0=1 → 唤醒 CCU（CCU 侧 `EventWait(events[0])` 解阻）
3. CCU 执行 AllGather → `EventRecord(events[1])`（写 eventVa[1]=1）
4. AIV 轮询 ckeAddr1==1 → 通信完成 → `CrossCoreSetFlag` 唤醒 AIC（第一阶段无此步，AIV 直接结束，Host 读回 recvBuf 校验）

> 第一阶段仅需 AIV + CCU 双核双流，AIC matmul 第二阶段引入。同步 API 全部为 AIV 已验证（`WriteGmByPassDCache`/`ReadGmByPassDCache`/`Nop`/`DataSyncBarrier`/`CrossCoreSetFlag`），不触碰 AIC 未验证 API。

### 1.3 工程目录目标设计

新建独立直调工程，结构对齐参考样例 `02_allgather_add/`（单目录含 .asc + CMakeLists.txt + run.sh），不嵌入 apace 的 `kernel/fusions/` 和 `tests/st/` 多层结构：

```
apace/examples/ccu_all_gather_matmul/         ← 独立工程目录（对齐 02_allgather_add 结构）
├── ccu_all_gather_matmul_kernel.asc           # 单文件：CCU kernel + AIV/AIC kernel + Host 编排 + main（仿 02_allgather_add/main.asc）
├── ccu_all_gather_matmul_tiling_data.h        # Tiling POD 结构
├── CMakeLists.txt                             # 对齐 02_allgather_add/CMakeLists.txt
├── run.sh                                     # 编译+运行+设置 CCU_SCHED 环境变量
├── verify_result.py                           # 可选：脚本化校验
└── README.md
```

> 第一阶段单文件 `ccu_all_gather_matmul_kernel.asc` 完全仿 `02_allgather_add/main.asc`，CMake 逐字节对齐参考样例。第二阶段引入 Blaze 时，blaze/tensor_api 拉取到本目录 `op_kernel/include/` 下（仿 Blaze 模板 run.sh 自动拉取）。

---

## 2. 第一阶段：x1 AllGather（通信先行验证）

> **详细设计见**：[`CCU_ALLGATHER_PHASE1_DESIGN.md`](CCU_ALLGATHER_PHASE1_DESIGN.md)（含完整伪代码、函数级复用清单、CMake、测试脚本、风险点）

### 2.1 目标
完成单次 AllGather（CCU 通信），**不含 matmul**。输出 buffer 预分配满足 allgather 后的数据量（rankSize × perRank）。验证通信正确后再进第二阶段。

**第一阶段内部分两步**（均仅针对极简 case，不要求泛化）：

| 步骤 | 目标 | 验收标准 |
|------|------|---------|
| **Step 1：跑通** | kernel 能拉起、双流跑完、正常退出 | **不崩溃、不卡死**即可，不关注精度（输出可为任意值/全零） |
| **Step 2：精度** | 通信结果正确 | recvBuf 拼接顺序与内容正确，computeBuf == recvBuf |

> Step 1 与 Step 2 串行：Step 1 跑通后才做 Step 2 精度校验。两步都用同一个极简 case（§2.4），不额外造泛化用例。

### 2.2 范围
- CCU 通信 kernel：**整体复用** `02_allgather_add` 的 `CcuAllGatherMesh1DMem2MemKernel` + `ExecuteAllGatherTransfer` + `GroupCopy` + `CalGoSize` + `AcquireCcuChannels` + `CreateCcuInsWithEnabledDies`。
- AIV 同步桥 kernel：复用 02_allgather_add 的 `vector_add` 骨架（写参数桥→唤醒CCU→轮询），但**计算部分替换为纯验证拷贝**（DataCopy recvBuf→UB→computeBuf，不做 Adds），无 matmul、无 AIC。
- Host 编排：`AllGatherCustom` 入口（双流：streamCcu + streamAiv，无 streamAic）。
- 测试：2 卡，每卡发 M×K 的 FP16 数据，验证 recvBuf 拼接顺序与内容。

### 2.3 关键改造点（相对 02_allgather_add）
1. **数据规模**：原样例固定 `sendCount=256 FP32`（1D 向量）。本方案数据是 2D 矩阵分片（M×K）。CCU 通信按**字节连续块**搬运，不关心 2D 语义——只要把 `inputSize = M*K*sizeof(T)` 字节作为 sendCount 等价量传入 `CalGoSize` 即可。`recvBuf` 按 rank 顺序拼接，每 rank 占 `inputSize` 字节。
2. **数据类型**：放开 `dataType != HCCL_DATA_TYPE_FP32` 的硬校验，支持 FP16/FP32。`SIZE_TABLE` 已有映射。第一阶段用 FP16。
3. **sendCount 硬编码移除**：移除 `sendCount != 256` 校验，改为动态。
4. **验证 kernel**：AIV kernel 改为 `verify_copy<T>`：等通信完成 → `DataCopy(recvBuf → UB → computeBuf)`，不做 Adds。Host 侧 `aclrtMemcpy` 读回校验。

### 2.4 测试 shape（第一阶段，极简 case）
- 2 卡，每卡 `M=256, K=256, dtype=FP16`，`inputSize = 256*256*2 = 128KB`。
- recvBuf = `2 * 128KB = 256KB`（输出 buffer 按聚合后大小分配）。
- 期望（Step 2）：rank i 的 recvBuf = `[rank0数据 | rank1数据]`，computeBuf 与 recvBuf 一致（纯拷贝验证）。
- **不要求泛化**：仅此一个 case，不扫描多 shape / 多 dtype。

### 2.5 第一阶段交付
- `ccu_all_gather_matmul_kernel.asc`（仅通信 + 验证拷贝，单文件仿 02_allgather_add）
- `examples/ccu_all_gather_matmul/`（CMakeLists + run.sh + 可选 verify_result.py）
- **Step 1 验收**：上板跑完不崩溃不卡死
- **Step 2 验收**：recvBuf 拼接正确
- 两步均通过 → 进入第二阶段

---

## 3. 第二阶段：x1 AllGather + x1 Matmul 串联（端到端精度）

### 3.1 目标
在第一阶段 AllGather 通信基础上，**串联 matmul 计算**：单次 AllGather 完成后，AIC matmul 消费聚合后的 A（rankSize*M × K）与权重 B（K × N），输出 C（rankSize*M × N）。验证端到端精度。**无 scale 通信**（纯 matmul 路线）。

### 3.2 依赖第一阶段的确认项
- 第一阶段通信 kernel 稳定可用，recvBuf 拼接正确

### 3.3 实现内容（AIV 同步桥 + AIC matmul，三核三流）
- CCU 通信 kernel：复用第一阶段，不变。
- AIV 同步桥 kernel：复用第一阶段骨架（写参数桥→唤醒CCU→轮询通信完成），通信完成后改为 **`CrossCoreSetFlag` 唤醒 AIC**（第一阶段是做验证拷贝，第二阶段改为唤醒 AIC）。
- AIC matmul kernel：**新增**。结构：
  1. 头部 `CrossCoreWaitFlag` 等待 AIV 唤醒（通信已完成的信号）
  2. 调用 Blaze `GemmUniversal` matmul：A=recvBuf (rankSize*M, K)，B=权重B (K, N)，C=输出 (rankSize*M, N)
- Host 编排：`AllGatherMatmulCustom` 入口，三流：streamCcu + streamAiv + streamAic。
- matmul tiling：baseM=baseN=baseK=256，host 侧计算 `usedCoreNum`、`mTailTile` 等填充 tiling POD 结构（参考 apace `quant_matmul_tiling_data.h` 字段定义，或简化为固定 tiling stub）。

> 第二阶段仍用 AIV 同步桥（已验证路径），AIC 仅做 `CrossCoreWaitFlag` + matmul，不触碰 AIC 未验证的 GM 轮询 API。AIC 直轮询延至第三阶段引入（见 §4）。

### 3.4 matmul kernel 类型链（FP16→BF16，已确认）

```cpp
// 纯 matmul 组件（ops-tensor examples/mat_mul/mat_mul_basic 确认）
using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<>;  // 非 MX/Quant
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, half, LayoutA, half, LayoutB, bfloat16_t, LayoutC, float, LayoutC>;
using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape>;
using KernelImpl = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```

> **P6 闭环**：纯 matmul 组件名已从 ops-tensor `examples/mat_mul/mat_mul_basic/mat_mul_basic.cpp` 确认。关键头文件：`block_mmad_matmul_basic.h` + `block_scheduler_matmul_basic.h` + `kernel_matmul_basic.h`（非 QBMM 的 `kernel_qbmm_mx.h`）。`GemmUniversal` 类名相同但特化头文件不同。详细设计见 [`CCU_ALLGATHER_PHASE2_DESIGN.md`](CCU_ALLGATHER_PHASE2_DESIGN.md)。

### 3.5 第二阶段交付
- 通信 + matmul 串联（无并行），端到端精度通过（BF16 rtol/atol=1e-3）
- 进入第三阶段引入并行

---

## 4. 第三阶段：通信计算并行（两次 AllGather + 两次 matmul）+ AIC 直轮询

### 4.1 目标
引入通信计算 overlap，**同时引入 AIC 直轮询**（去掉 AIV 同步桥中转）：
- **两次 AllGather**（按 M 轴分 tile，CCU kernel 内循环 commTurn 次）
- **单次 AllGather 完成即启动对应 matmul tile**
- 两次 allgather + 两次 matmul，形成流水

### 4.2 为何在第三阶段引入 AIC 直轮询

AIC 直轮询（AIC 头部直接 `WriteGmByPassDCache`/`ReadGmByPassDCache`/`Nop` 轮询 CKE Event，去掉 AIV 中转）是未验证路径，引入时机考量：

| 阶段 | 同步方式 | 为何不在更早阶段引入 AIC 直轮询 |
|------|---------|------------------------------|
| 第一阶段 | AIV 同步桥 | 目标跑通+精度，用已验证路径最快达成，不引入新风险 |
| 第二阶段 | AIV 同步桥 + CrossCoreSetFlag | 单次串联，AIV 中转只多一跳固定延迟，不影响正确性；前两阶段验证通信+matmul 各自正确 |
| **第三阶段** | **AIC 直轮询** | 通算并行流水下，AIV 中转每 round 多一跳延迟会累积吃掉 overlap 收益——AIC 直轮询收益此时才显现；且出问题时可二分定位（前两阶段已隔离通信/matmul 正确性） |

> 用户已确认 AIC 支持这些 API。第三阶段是引入 AIC 直轮询的最佳时机：性能收益最大，且风险可控（前两阶段已建立正确性基线）。

### 4.3 流水设计（AIC 直轮询，同步机制复用 02_allgather_add 的 CKE Event，分 round 扩展）

```
Round 0: CCU AllGather tile0 ──EventRecord──▶ AIC 轮询 ckeAddr ──▶ matmul tile0
Round 1: CCU AllGather tile1 ──EventRecord──▶ AIC 轮询 ckeAddr ──▶ matmul tile1（与 tile0 计算并行）
```

- CCU 侧：循环 `commTurn` 次，每次 AllGather 一个 tileM 行，完成后 `EventRecord`（复用 02_allgather_add 的 `ExecuteAllGatherTransfer` Event 机制，扩展为多 round）。
- AIC 侧：Blaze matmul 调度器按 tile 遍历，**每 tile 前直接轮询对应 round 的 CKE Event**（复用 02_allgather_add 的 `ReadGmByPassDCache`/`Nop` 轮询，从 AIV 迁移到 AIC）。
- 同步：每 round 用独立 CKE Event pair（events[2*round], events[2*round+1]），或复用 flag 计数。

### 4.4 关键改造（相对第二阶段）
- **AIV 同步桥移除**：AIC matmul kernel 头部接管参数桥 + 唤醒CCU + 轮询全链路（AIC 直轮询，机制复用 02_allgather_add 的 AIV 轮询逻辑，迁移到 AIC）。
- **参数桥写入**：AIC 写 taskArgs 到 GM（复用 02_allgather_add 的 `WriteGmByPassDCache` 参数桥，需验证 AIC 可写 CCU Variable 地址空间）。
- CKE Event 数量：从 2 增至 `2 * commTurn`（或用 flag 滚动复用）。
- Variable 数量：每 round 的地址参数（若 tile 间地址不同需多组 Variable；若地址连续递增可用 offset 计算）。
- matmul 调度器：需支持"按 round 等待"——AIC 轮询对应 round 的 CKE Event 后启动该 round 的 matmul tile。

### 4.5 第三阶段交付
- AIC 直轮询 + 通算并行 kernel，性能可测（msprof）
- 两次 allgather + 两次 matmul 流水验证
- 与第二阶段（AIV 桥串联）精度对齐，性能对比

---

## 5. CMake 构建方案（关键，提前考虑）

### 5.1 依赖矩阵

| 依赖 | 来源 | 用途 |
|------|------|------|
| CANN Toolkit | `$ASCEND_HOME_PATH` | ACL/HCCL/AscendC runtime |
| blaze + tensor_api | ops-tensor 仓库（git clone） | matmul Blaze 库 |
| hcomm CCU 头文件 | CANN `$ASCEND_HOME_PATH/include/hccl/hcomm/ccu` | CCU 通信 API |
| asccomm_ccu_dataplane | CANN lib64 | CCU 数据面库 |

### 5.2 CMakeLists.txt 要点（以 02_allgather_add 为基底，第二阶段起叠加 Blaze）

**第一阶段 CMake 逐字节对齐参考样例**，详见 [`CCU_ALLGATHER_PHASE1_DESIGN.md` §8](CCU_ALLGATHER_PHASE1_DESIGN.md)（7 条 include 路径 + 6 个链接库 + 参考样例标准 target 级 API）。

第二阶段起叠加 Blaze/tensor_api 依赖，目标态 CMake（在第一阶段基础上追加 Blaze 路径，其余不变）：

```cmake
cmake_minimum_required(VERSION 3.16.0)
find_package(ASC REQUIRED)
project(ccu_all_gather_matmul LANGUAGES CXX ASC)

# ASCEND_CANN_PACKAGE_PATH 推导（对齐 02_allgather_add）
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
    elseif(DEFINED ENV{ASCEND_OPP_PATH})
        get_filename_component(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_OPP_PATH}/.." ABSOLUTE)
    else()
        message(FATAL_ERROR "ASCEND_CANN_PACKAGE_PATH or ASCEND_HOME_PATH must be set")
    endif()
endif()

add_executable(ccu_all_gather_matmul
    ccu_all_gather_matmul_kernel.asc
)

target_include_directories(ccu_all_gather_matmul PRIVATE
    # 参考样例 7 条路径（逐字节对齐 02_allgather_add）
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl/hcomm
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl/hcomm/ccu
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu
    # 第二阶段新增：Blaze + tensor_api（第一阶段无此两行）
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/include
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/include/tensor_api
)

target_compile_features(ccu_all_gather_matmul PRIVATE cxx_std_14)
target_compile_options(ccu_all_gather_matmul PRIVATE
    -Werror -fstack-protector-strong -fPIE -O2
)
set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "")
target_compile_options(ccu_all_gather_matmul PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
target_compile_definitions(ccu_all_gather_matmul PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_link_options(ccu_all_gather_matmul PRIVATE
    -pie -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack
)
target_link_directories(ccu_all_gather_matmul PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)
target_link_libraries(ccu_all_gather_matmul PRIVATE
    # 参考样例 6 个库（逐字节对齐 02_allgather_add）
    hcomm
    asccomm_ccu_dataplane
    c_sec
    ascendcl
    acl_rt
    ascendc_runtime
    # 第二阶段新增：Blaze matmul 所需库（待开发时确认具体库名）
    # platform tiling_api register
)
```

> **对齐说明**：目标态 CMake 以参考样例为基底，仅追加 Blaze include 路径（2 行）和可能的 Blaze 链接库。第一阶段去掉 Blaze 相关行即为第一阶段 CMake（= PHASE1 §8）。不采用 Blaze 模板的 `set_source_files_properties` 写法，统一用参考样例的 `target_include_directories`/`target_compile_options` 标准 target 级 API。

### 5.3 blaze/tensor_api 依赖拉取（run.sh 自动）

```bash
# 仿 Blaze 模板 run.sh
if [ ! -d op_kernel/include/blaze ]; then
    git clone --depth 1 https://gitcode.com/cann/ops-tensor.git /tmp/ops-tensor
    cp -r /tmp/ops-tensor/include/blaze op_kernel/include/
    cp -r /tmp/ops-tensor/include/tensor_api op_kernel/include/
fi
```

### 5.4 预期编译风险点（提前排查）
1. **CCU 头文件路径**：`02_allgather_add` 用了 `${ASCEND_CANN_PACKAGE_PATH}/include/hccl/hcomm/ccu` 和 `pkg_inc/hcomm/ccu` 两套路径——说明 CCU 头文件分布可能跨目录，CMake 需都包含。
2. **asccomm_ccu_dataplane 库**：此库是 CCU 专有，Blaze 纯 matmul 模板未链接。需确认 CANN 安装时此库存在。
3. **ASC 语言编译 .asc 文件**：完全对齐 02_allgather_add——`add_executable(demo main.asc)` + `find_package(ASC)`，单文件 `.asc` 全走 ASC 编译器（含 main/kernel/host）。不采用 Blaze 模板的 `.cpp` 设 LANGUAGE ASC 混编方式。
4. **dav-3510 架构**：CCU + Blaze 均要求 `--npu-arch=dav-3510`（Ascend 950）。
5. **第二阶段叠加 Blaze**：第一阶段无 Blaze 依赖，第二阶段引入 matmul 时需叠加 `op_kernel/include/blaze` + `tensor_api` include 路径和 ops-tensor 依赖拉取（见 §5.3）。

---

## 6. 测试方案

### 6.1 多 rank 启动
采用 `02_allgather_add` 的 **`std::thread` 多线程模式**（L909-943），每设备一个线程，rootInfo 指针同进程共享，无需 TCP 交换。完全对齐核心参考。

### 6.2 数据生成与校验
- **进程内生成**：数据在 Sample() 内生成（rank i 填充 i+1），进程内打印校验。第一阶段无需外部 gen_data.py。
- **第二阶段起**（含 matmul）：权重 B 需预生成，可选用 gen_data.py 生成 B + golden（`concat(A_0..A_R) @ B`），verify_result.py 校验 BF16 rtol/atol=1e-3。
- 第一阶段极简 case：进程内打印人工校验为主，verify_result.py 为可选增强。

### 6.3 测试 shape
- 第一阶段：`M=256, K=256, rank=2, dtype=FP16`（最小验证，无 N）
- 第二阶段：`M=256, K=256, N=256, rank=2`（端到端精度）
- 第三阶段：`M=512, K=256, N=256, rank=2, tileM=256`（2 tile 流水）

---

## 7. 风险与待确认清单

| 编号 | 问题 | 结论 | 状态 |
|------|------|------|------|
| P1 | matmul 数据类型（量化 vs 纯 matmul） | **纯 matmul FP16→BF16** | 已确认 |
| P2 | 第一阶段是否做 scale 通信 | **不做 scale 通信**（纯 matmul 无 scale） | 已确认 |
| P3 | CCU→计算核同步方式 | **第一/二阶段用 AIV 同步桥**（已验证，CrossCoreSetFlag 唤醒 AIC）；**第三阶段引入 AIC 直轮询**（用户确认 AIC 支持，性能收益在并行流水下最大） | 已确认 |
| P4 | `asccomm_ccu_dataplane` 库在目标 CANN 是否预装 | 需环境就绪后确认 | 待环境 |
| P5 | CCU kernel 内 `std::vector`/`std::map` 在 `__ccu_host__` 是否可用 | 02_allgather_add 已用，可行 | 已验证（参考） |
| P6 | 纯 matmul（非 QBMM）的 Blaze scheduler/kernel 组件名 | **已确认**：`MatmulMultiBlockBasic` + `BlockSchedulerMatmulBasic` + `kernel_matmul_basic.h`（来自 ops-tensor examples/mat_mul/mat_mul_basic） | ✅ 已确认 |
| P7 | AIV→AIC 的 `CrossCoreSetFlag`/`CrossCoreWaitFlag` 跨核同步（第二阶段用） | apace UDMA 路线已用（all_gather_mx_matmul_udma_impl.h L248），可行 | 已验证（参考） |
| P8 | AIC 直轮询（第三阶段）：AIC 能否用 `WriteGmByPassDCache`/`ReadGmByPassDCache`/`Nop` + 能否写 CCU Variable 地址空间 | 用户确认 AIC 支持；上板验证 | 待第三阶段验证 |

---

## 8. 执行节奏

1. **本方案持久化** ✅ → 用户确认关键决策 ✅（P1=纯matmul FP16→BF16，P2=无scale，P3=AIV桥→第三阶段AIC直轮询，第一阶段内部两步制）
2. **第一阶段开发**（待用户最终批准方案后开始）：
   - **Step 1**：CCU 通信 + AIV 同步桥验证拷贝代码跑通，极简 case 上板**不崩溃不卡死**（不关注精度）
   - **Step 2**：同 case 精度校验，recvBuf 拼接正确
3. **第二阶段开发**（第一阶段 Step1+Step2 验证后）：+ AIC matmul（AIV 同步桥 CrossCoreSetFlag 唤醒 AIC），端到端精度
4. **第三阶段开发**（第二阶段验证后）：**引入 AIC 直轮询**（去 AIV 桥）+ 通算并行流水（两次 allgather + 两次 matmul）

**严格串行**：每阶段/每步验证通过才进下一阶段/步，禁止跨阶段开发。两步均仅针对极简 case，不要求泛化。
