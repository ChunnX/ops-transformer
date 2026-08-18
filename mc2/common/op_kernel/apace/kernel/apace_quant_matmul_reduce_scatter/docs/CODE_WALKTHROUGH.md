# quantMatmulReduceScatter 代码讲解文档

> 本文档面向初学者，从整体架构到逐文件、逐代码块、逐参数，完整讲解 `quantMatmulReduceScatter` 算子的实现逻辑与数据流程。

---

## 目录

- [1. 算子概述](#1-算子概述)
  - [1.1 一句话定义](#11-一句话定义)
  - [1.2 数学公式](#12-数学公式)
  - [1.3 为什么叫"通算融合"（MC2）](#13-为什么叫通算融合mc2)
  - [1.4 硬件架构背景](#14-硬件架构背景)
  - [1.5 关键设计决策](#15-关键设计决策)
- [2. 目录结构与文件职责](#2-目录结构与文件职责)
- [3. 端到端执行流程](#3-端到端执行流程)
- [4. M 轴两级 Tiling 切分](#4-m-轴两级-tiling-切分)
- [5. 逐文件详细讲解](#5-逐文件详细讲解)
  - [5.1 src/main.cpp — Host 侧主程序](#51-srcmaincpp--host-侧主程序)
  - [5.2 src/kernel_launcher.h — Kernel 入口](#52-srckernel_launcherh--kernel-入口)
  - [5.3 kernel/quant_matmul_reduce_scatter_tiling_data.h — Tiling 数据结构](#53-kernelquant_matmul_reduce_scatter_tiling_datah--tiling-数据结构)
  - [5.4 kernel/quant_matmul_reduce_scatter_impl.h — 核心实现类](#54-kernelquant_matmul_reduce_scatter_implh--核心实现类)
  - [5.5 kernel/qmm_mx_kernel_rs_frag.h — FragmentTensor MatMul 内核](#55-kernelqmm_mx_kernel_rs_fragh--fragmenttensor-matmul-内核)
  - [5.6 kernel/reduce_sum_ref.h — AIV reduceSum 实现](#56-kernelreduce_sum_refh--aiv-reducesum-实现)
  - [5.7 kernel/reduce_sum_utils.h — 数学工具](#57-kernelreduce_sum_utilsh--数学工具)
  - [5.8 kernel/mc2_kernel_utils.h — SyncFunc](#58-kernelmc2_kernel_utilsh--syncfunc)
  - [5.9 src/utils.h — Host 工具](#59-srcutilsh--host-工具)
  - [5.10 src/root_info_exchanger.h — TCP rootInfo 交换](#510-srcroot_info_exchangerh--tcp-rootinfo-交换)
  - [5.11 scripts/gen_data.py — CPU golden 数据生成](#511-scriptsgen_datapy--cpu-golden-数据生成)
  - [5.12 scripts/verify_result.py — 精度比对](#512-scriptsverify_resultpy--精度比对)
  - [5.13 CMakeLists.txt — 构建配置](#513-cmakeliststxt--构建配置)
  - [5.14 run.sh — 一键脚本](#514-runsh--一键脚本)
  - [5.15 cases.csv — 测试用例](#515-casescsv--测试用例)
- [6. Kernel 内部 AIC ↔ AIV 流水线](#6-kernel-内部-aic--aiv-流水线)
- [7. 数据内存布局](#7-数据内存布局)
- [8. FragmentTensor 消除 R 循环](#8-fragmenttensor-消除-r-循环)
- [9. AllToAll PUT 通信数据流](#9-alltoall-put-通信数据流)
- [10. reduceSum pingpong 流水](#10-reducesum-pingpong-流水)
- [11. 完整数据流总结](#11-完整数据流总结)

---

## 1. 算子概述

### 1.1 一句话定义

每张卡（rank）独立做一次完整的量化矩阵乘法 `C_i = A_i × B`，然后通过 **AlltoAll** 通信把 `C_i` 按 M 轴切分发给所有卡，每张卡收到来自所有卡的片段后做**逐元素求和（reduceSum）**，最终每卡输出 `[m/rankSize, n]` 的结果。

### 1.2 数学公式

```
output = reduceSum( AlltoAll( Dequant(A_i) × Dequant(B) ) )
```

展开说明：
- `Dequant(A_i)`：将 FP8 量化矩阵 A 反量化为 float32，每卡持有不同的 A_i
- `Dequant(B)`：将 FP8 量化矩阵 B 反量化为 float32，所有卡共享相同 B
- `C_i = Dequant(A_i) × Dequant(B)`：每卡独立计算完整 `[m, n]` 结果
- `AlltoAll(C_i)`：每卡将 C_i 按 M 轴切分为 R 个 chunk，chunk j 发送到 rank j
- `reduceSum`：rank r 收到所有卡的 chunk r，逐元素求和
- 最终每卡输出 `[m/rankSize, n]` bfloat16

### 1.3 为什么叫"通算融合"（MC2）

传统做法是：先单独做 MatMul → 再单独做通信 → 再单独做 ReduceSum，三次 GM 读写。

本算子把 **计算（AIC 核）+ 通信（AIV 核）+ 求和（AIV 核）** 融合进**同一个 Kernel**，AIC 边算边通知 AIV，AIV 边收边累加，减少 GM 往返。MC2 = Multi-Cards Communication + Compute fusion。

### 1.4 硬件架构背景

| 概念 | 说明 |
|------|------|
| **Ascend 950 / dav-3510** | 目标芯片，AIC（Cube 矩阵核）+ AIV（Vector 向量核）分离 |
| **AIC** | 负责矩阵乘法（MatMul），使用 Blaze/tensor_api 框架 |
| **AIV** | 负责通信（AllToAll PUT）+ 向量运算（reduceSum） |
| **KERNEL_TYPE_MIX_AIC_1_1** | 混合核类型：1 个 AIC 配 1 个 AIV，组成一个 block |
| **APACE 框架** | CANN 内置的通算融合框架，提供通信 API（CollectiveComm 等） |
| **Blaze 框架** | CANN 内置的 MatMul 高性能框架，提供 BlockMmad/BlockScheduler |
| **PUT 模式** | 通信模式之一：AIV 主动把自己的数据"推"到其他卡的窗口区 |

### 1.5 关键设计决策

与参考算子 `all_to_all_quant_matmul`（通信在前计算在后）的关键区别：

| 决策 | 说明 |
|------|------|
| **计算在前，通信在后** | all_to_all 是通信在前计算在后；本算子先算完 C 再通信 |
| **通信 C 数据（bfloat16）** | 不是通信 A 数据（fp8），因为通信的是计算结果 |
| **无 ScaleA 通信** | Scale 只在计算时用，不需要通信 |
| **FragmentTensor 消除 R 循环** | 把 R 个 rank 的数据段打包成 FragmentTensor，一次 matmul 调用覆盖所有 rank 段，不用 for r 循环 |
| **AIV 需要 reduceSum** | AlltoAll 后每卡收到 R 份数据，需逐元素求和 |
| **仅 LOCAL matmul 模式** | splitKNum=1，无 L0C 原子累加 |

---

## 2. 目录结构与文件职责

```
quantMatmulReduceScatter/
├── docs/                    # 设计文档
│   ├── DESIGN.md            # 技术设计
│   ├── PLAN.md              # 开发计划
│   ├── environment.md       # 环境检查
│   ├── DESIGN_VS_CODE_REVIEW.md
│   └── CODE_WALKTHROUGH.md  # 本文件（代码讲解）
├── kernel/                  # Kernel 侧实现（头文件）
│   ├── quant_matmul_reduce_scatter_tiling_data.h  # Tiling 数据结构
│   ├── quant_matmul_reduce_scatter_impl.h          # 核心实现类（Init/Run/通信/reduceSum）
│   ├── qmm_mx_kernel_rs_frag.h                    # FragmentTensor MatMul 内核
│   ├── reduce_sum_ref.h                           # AIV reduceSum 实现
│   ├── reduce_sum_utils.h                         # 数学工具（CeilDiv/CeilAlign...）
│   └── mc2_kernel_utils.h                         # SyncFunc 工具
├── src/                     # Host 侧 + Kernel 入口
│   ├── kernel_launcher.h    # __global__ Kernel 入口函数
│   ├── main.cpp             # Host 主程序（fork/ACL/HCCL/Tiling/启动）
│   ├── utils.h              # 文件读写/日志宏
│   └── root_info_exchanger.h  # TCP rootInfo 交换
├── scripts/                 # 测试脚本
│   ├── gen_data.py          # CPU golden 数据生成
│   └── verify_result.py     # 精度比对
├── build/                   # 编译输出
├── input/                   # 生成的输入数据
├── output/                  # NPU 输出 + golden 输出
├── CMakeLists.txt           # 构建配置
├── run.sh                   # 一键编译运行脚本
├── cases.csv                # 测试用例参数表
└── README.md                # 算子说明
```

---

## 3. 端到端执行流程

```
┌──────────────────────────────────────────────────────────────────┐
│                        run.sh (bash)                             │
│                                                                  │
│  [1] python3 gen_data.py m k n rankNum                           │
│       │                                                          │
│       ▼                                                          │
│      生成 input/{r}/input_a.bin, input_b.bin,                    │
│            input_scaleA.bin, input_scaleB.bin                    │
│      生成 output/{r}/cpu_output.bin (golden)                     │
│                                                                  │
│  [2] cmake .. && make -j4                                        │
│       │                                                          │
│       ▼  quant_matmul_reduce_scatter_st (可执行)                  │
│                                                                  │
│  [3] ./quant_matmul_reduce_scatter_st m k n rankNum precision h  │
│       │                                                          │
│       ├─ fork() ── rank 0 进程 ── 绑 NPU 0                       │
│       ├─ fork() ── rank 1 进程 ── 绑 NPU 1                       │
│       └─ fork() ── rank R-1                                       │
│                                                                  │
│       每进程: aclInit → setDevice → HCCL建链 →                    │
│              Tiling计算 → 加载数据 → 启动Kernel → 拷回结果        │
│                                                                  │
│  [4] python3 verify_result.py m n rankNum ./output               │
│       │                                                          │
│       ▼                                                          │
│      npu_out.bin  vs  cpu_output.bin  (atol=rtol=1e-2)          │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. M 轴两级 Tiling 切分

以 `m=2048, rankSize=2, headMSize=512` 为例：

```
M 轴 (2048 行)
│
├──────── 第一级：按 rankSize 切 (用于 AlltoAll) ────────────┤
│                                                            │
│   chunkM = m / rankSize = 1024                             │
│                                                            │
│   rank 0 段 [0 : 1024]      rank 1 段 [1024 : 2048]        │
│   ├──────────────────┤      ├──────────────────┤           │
│   │    chunk 0       │      │    chunk 1       │           │
│   │   (1024 行)      │      │   (1024 行)      │           │
│   └──────────────────┘      └──────────────────┘           │
│                                                            │
├────── 第二级：按 headMSize 切 chunkM (通信粒度) ──────────┤
│                                                            │
│   headMSize=512, headTileCnt = 1024/512 = 2                │
│   tailMSize = 1024 % 512 = 0  → tailTileCnt = 0            │
│   commTurn = headTileCnt + tailTileCnt = 2                 │
│                                                            │
│   rank 0 段:                                                │
│   ├── tile 0 [0:512]   ├── tile 1 [512:1024]               │
│   │   headTile          │   headTile                        │
│                                                            │
│   rank 1 段:                                                │
│   ├── tile 0 [1024:1536] ├── tile 1 [1536:2048]            │
│                                                            │
│   每个 commTurn (t=0,1):                                    │
│     AIC 计算 R*curTileM = 2*512 = 1024 行                   │
│     AIV 通信 + reduceSum 1024 行                            │
└────────────────────────────────────────────────────────────┘
```

**关键概念**：
- `chunkM = m / rankSize`：每个 rank 分到的 M 行数
- `headMSize`：通信头块大小，控制 AlltoAll 切分粒度
- `headTileCnt = chunkM / headMSize`：头块数量
- `tailMSize = chunkM % headMSize`：尾块大小（可能为 0）
- `commTurn = headTileCnt + tailTileCnt`：总通信轮次，决定 AIC/AIV 循环次数

---

## 5. 逐文件详细讲解

### 5.1 `src/main.cpp` — Host 侧主程序

整个算子的**入口程序**，运行在 CPU 上，负责：参数解析 → fork 多进程 → NPU 初始化 → HCCL 建链 → Tiling 计算 → 数据加载 → Kernel 启动 → 结果拷回。

#### 5.1.1 `parseArguments()` (L80-122)

解析命令行参数 `m k n rankNum [mode] [headMSize]`：

| 参数 | 含义 |
|------|------|
| `m` | 矩阵 A 的行数（必须被 rankNum 整除） |
| `k` | 矩阵 A 的列数 / 矩阵 B 的行数（完整 K，**不切分**） |
| `n` | 矩阵 B 的列数 |
| `rankNum` | rank 数量（卡数） |
| `mode` | `precision`（精度验证）或 `perf`（性能测试） |
| `headMSize` | 通信头块 M 大小（默认 512），控制 AlltoAll 切分粒度 |

两个约束检查：
- `m % rankNum == 0`：M 轴按 rankSize 切分用于 AlltoAll
- `CeilDiv(k, 64) % 2 == 0`：MXFP8 scale 对齐要求（每 64 个元素一组，2 字节，需偶数组）

#### 5.1.2 `runQuantMatmulReduceScatter()` (L124-417) — 核心

这是每个 rank 进程执行的函数。按步骤讲解：

**Step A: Tiling 计算 (L130-192)**

```cpp
uint32_t chunkM = m / rankNum;  // 每个 rank 分到的 M 行数
```

Tiling 有三套 MatMul tiling：

| Tiling | 用途 | m 值 |
|--------|------|------|
| `mmTilingData` | T=1 退化路径（单 tile） | 完整 m |
| `subMmTilingData` | T>1 的 head tile | headMSize |
| `tailMmTilingData` | T>1 的 tail tile | rankSize × tailMSize |

为什么需要三套？因为 Blaze 的 `baseM`（MatMul tile 的 M 粒度）会根据输入 m 自动选择，不同 m 大小可能得到不同的 baseM，影响性能。当 `subMmTilingData.baseM == 256 && headMSize % 128 == 0` 时，会把 baseM 减半到 128，让 tile 分配更均匀（L174-180）。

通信 tiling 的关键字段：
```cpp
tilingData.commTilingData.splitAxisTileCnt   = headTileCnt;    // 头块数量
tilingData.commTilingData.splitAxisTileSize  = headMSize;      // 头块 M 大小
tilingData.commTilingData.splitAxisTailSize  = tailMSize;      // 尾块 M 大小
tilingData.commTilingData.splitAxisTailCnt   = tailTileCnt;    // 尾块数量 (0或1)
tilingData.commTilingData.nonSplitAxisSize   = n;              // 非切分轴 (N)
```

**Step B: ACL 初始化 (L233-239)**
```cpp
aclInit(nullptr);           // 初始化 ACL 框架
aclrtSetDevice(deviceId);   // 绑定 NPU 设备（deviceId = rankId）
aclrtCreateStream(&stream); // 创建计算流
```

**Step C: HCCL 建链 (L241-261)**
```cpp
RootInfoExchanger exchanger(rankId, rankNum, "tcp://127.0.0.1:8998");
exchanger.Exchange(rootInfo);  // TCP 交换 rootInfo
HcclCommInitRootInfoConfig(rankNum, &rootInfo, rankId, &config, &comm);
```
- rank 0 调 `HcclGetRootInfo` 生成 rootInfo，通过 TCP 广播给其他 rank
- 所有 rank 用相同的 rootInfo 初始化 HCCL 通信域

**Step D: Apace 通信上下文 (L263-279)**
```cpp
CommChannelBuilder builder(comm);
CommContext *devContext = builder.CreateDeviceContext(...);
```
这一步建立 Apace 框架需要的设备侧通信上下文（`udmaCtx` 用于 UDMA 通信，`ubmemCtx` 用于 UB 同步），后续 Kernel 会用到。

**Step E: 数据加载 (L281-338)**

数据布局：

| 数据 | shape | dtype | 说明 |
|------|-------|-------|------|
| A | `[m, k]` | fp8_e4m3fn | 每卡不同，量化矩阵 A |
| ScaleA | `[m, ceil(k/64)*2]` | fp8_e8m0 | A 的 scale，每 64 元素一组，2 字节 |
| B | `[n, k]` | fp8_e4m3fn | **所有 rank 共享相同 B**，TransB=true（列主序存储） |
| ScaleB | `[n, ceil(k/64)*2]` | fp8_e8m0 | B 的 scale |
| workspace | `[m, n]` | bfloat16 | AIC 写入 C_i 的缓冲区 |
| output | `[chunkM, n]` | bfloat16 | 最终每卡输出 |

数据从 Host `input/{rankId}/input_*.bin` 读入 → `aclrtMalloc` 分配 GM → `aclrtMemcpy` 拷到设备。

workspace 大小计算：
```cpp
size_t sizeWorkspace = static_cast<size_t>(m) * n * sizeof(uint16_t);  // [m, n] bf16
```
workspace 用于 AIC 写入完整 C_i `[m, n]`，AIV 从此处读取并 PUT 到其他 rank。

**Step F: Kernel 启动 (L340-355)**
```cpp
QuantMatmulReduceScatterKernelE4M3E4M3<<<
    tilingData.mmTilingData.usedCoreNum, nullptr, stream>>>(
        devContext, deviceA, deviceScaleA, deviceB, deviceScaleB,
        deviceWorkspace, deviceOutput, tilingData);
```
- `<<<usedCoreNum>>>`：启动 usedCoreNum 个 block（block 数由 Tiling 引擎根据 m/n/k 决定）
- 每个 block 是一个 AIC+AIV pair

**Step G: 性能模式 (L356-401)**

`mode == "perf"` 时：循环跑 10 次，用 boost 数据刷 cache，记录耗时，输出 avg/min/max ms。

#### 5.1.3 `main()` (L419-462) — fork 多进程

```cpp
for (rankId = 0..rankNum) {
    pid = fork();
    if (pid == 0) {  // 子进程
        runQuantMatmulReduceScatter(rankNum, rankId, ...);
        exit(ret);
    }
}
// 父进程 waitpid 所有子进程
```
每个 rank = 一个独立进程，绑定不同的 NPU 设备。

---

### 5.2 `src/kernel_launcher.h` — Kernel 入口

```cpp
__global__ __aicore__ void QuantMatmulReduceScatterKernelE4M3E4M3(
    __gm__ CommContext *hcommCtx,
    GM_ADDR aGM, GM_ADDR scaleAGM,
    GM_ADDR bGM, GM_ADDR scaleBGM,
    GM_ADDR workspaceGM,
    GM_ADDR cGM,
    QuantMatmulReduceScatterTilingData tilingData)
{
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
  Apace::QuantMatmulReduceScatterImpl<fp8_e4m3fn_t, fp8_e4m3fn_t, bfloat16_t, false, true> impl;
  impl.Init(hcommCtx, aGM, scaleAGM, bGM, scaleBGM, workspaceGM, cGM, &tilingData);
  impl.Run();
}
```

| 参数 | 含义 |
|------|------|
| `hcommCtx` | Apace 通信上下文（含 UDMA/UBmem 通道） |
| `aGM` / `scaleAGM` | A 矩阵及其 scale 的 Global Memory 地址 |
| `bGM` / `scaleBGM` | B 矩阵及其 scale |
| `workspaceGM` | AIC 写 C_i 的缓冲区，也是 AIV PUT 的源数据 |
| `cGM` | 最终输出地址 |
| `tilingData` | Tiling 参数（值传递，Kernel 栈上） |

模板参数：
- `AType = fp8_e4m3fn_t`：A 是 FP8 E4M3 格式
- `BType = fp8_e4m3fn_t`：B 也是 FP8 E4M3
- `CType = bfloat16_t`：输出 BF16
- `TransA = false`：A 不转置（行主序）
- `TransB = true`：B 转置（存储为 `[n,k]` 列主序）

`KERNEL_TYPE_MIX_AIC_1_1`：声明这是混合核任务，1 AIC : 1 AIV。

---

### 5.3 `kernel/quant_matmul_reduce_scatter_tiling_data.h` — Tiling 数据结构

```cpp
struct QuantMatmulReduceScatterTilingData {
    CommTilingData commTilingData;          // 通信切分参数
    QuantMatmulTilingData mmTilingData;     // 全量 MatMul tiling (T=1)
    QuantMatmulTilingData subMmTilingData;  // 子问题 tiling (head tile)
    QuantMatmulTilingData tailMmTilingData; // tail tile tiling
};
```

| 字段 | 作用 |
|------|------|
| `commTilingData` | AlltoAll 通信的 M 轴切分：`splitAxisTileSize`(头块大小) / `splitAxisTileCnt`(头块数) / `splitAxisTailSize`(尾块大小) / `splitAxisTailCnt`(尾块数，0或1) / `nonSplitAxisSize`(N) |
| `mmTilingData` | Blaze MatMul 的完整 tiling：m/n/k/baseM/baseN/baseK/stepK/usedCoreNum/dbL0c 等 |
| `subMmTilingData` | 当通信分多 tile 时，head tile 用的子问题 tiling（m=headMSize） |
| `tailMmTilingData` | tail tile 用的 tiling（m=rankSize×tailMSize） |

---

### 5.4 `kernel/quant_matmul_reduce_scatter_impl.h` — 核心实现类

这是算子的**灵魂文件**。`QuantMatmulReduceScatterImpl` 模板类包含 Init / Run / RunMatmul / RunAllToAllAndReduceSum 四个核心方法。

#### 5.4.1 类成员定义

**类型定义 (L87-107)**：
```cpp
using TypeA = AType;          // fp8_e4m3fn
using TypeB = BType;          // fp8_e4m3fn
using TypeC = CType;          // bfloat16
using TypeScaleA = fp8_e8m0_t;  // MX scale 类型
using TypeScaleB = fp8_e8m0_t;
static constexpr int32_t SCALE_C0 = 2;  // scale 的 C0 维度=2字节
```

**Layout 定义 (L95-107)**：
- `LayoutA = NDExtLayoutPtn`（A 不转置，ND 扩展布局）
- `LayoutB = DNExtLayoutPtn`（B 转置，DN 布局）
- `LayoutC = NDExtLayoutPtn`（C 不转置）
- ScaleA/ScaleB 用 `ScaleAND/ScaleBDN` + `FrameLayoutFormat` 包装

**通信组件 (L118-123)**：
```cpp
CollectiveComm<CommCollectiveOp::AllToAll, CommMode::PUT, CType, TeamBarrier> allToAllC_;
TeamBarrier teamBarrier_;
```
- `allToAllC_`：APACE 的 AllToAll 通信器，PUT 模式，数据类型 CType(bfloat16)
- `teamBarrier_`：团队同步屏障

**BaseParams 结构 (L125-147)** — 运行时参数：

| 字段 | 含义 |
|------|------|
| `selfWinAddr` | 本卡通信窗口地址（AIV 收到数据的落点） |
| `aGm/scaleAGm/bGm/scaleBGm` | 输入数据 GM 地址 |
| `workspaceGm` | AIC 写 C_i 的缓冲区 |
| `cGm` | 最终输出 GM |
| `rankId` / `rankSize` | 本卡 ID / 总卡数 |
| `commTurn` | 总通信轮次 = headTileCnt + tailTileCnt |
| `axisM/N/K` | 矩阵维度 |
| `headMSize` / `tailMSize` | 头块/尾块 M 大小 |
| `headTileCnt` / `tailTileCnt` | 头块/尾块数量 |
| `chunkM` | 每个 rank 的 M 分片 = m / rankSize |
| `chunkBytes` | 每个 rank 的 C 数据块字节数 |
| `scaleARowBytes` | ScaleA 每行字节数 = CeilDiv(k,64)×2 |

#### 5.4.2 `Init()` (L165-201)

```cpp
tilingData_ = tilingData;
// 保存所有 GM 地址
syncBuffer_ = &(hcommCtx->ubmemCtx);   // UB 同步缓冲
udmaCtx_ = &(hcommCtx->udmaCtx);       // UDMA 通信上下文
baseParams_.rankId = udmaCtx_->rankId;
baseParams_.rankSize = udmaCtx_->rankSize;
InitBaseParams(tilingData);
baseParams_.selfWinAddr = udmaCtx_->commBufferAddrs[rankId];  // 本卡 win 区

// UB 分配: commBuf(通信) + barrierBuf(同步)
auto commBuf = MakeMemPtr<UB>(ubOffset);
auto barrierBuf = MakeMemPtr<UB>(ubOffset + COMM_WORKSPACE_SIZE);

// AllToAll 初始化: totalJobs=1 (block 0 处理所有)
allToAllC_.Init(udmaCtx_, teamBarrier_, commTilingData, workspaceGm, commBuf, 1, blockIdx);
```

**UB 内存分配**：
- `commBuf`：AllToAll 通信缓冲区，大小 `COMM_WORKSPACE_SIZE`
- `barrierBuf`：TeamBarrier 同步用，紧跟在 commBuf 后面

`allToAllC_.Init` 参数说明：
- `totalJobs=1`：只有 block 0 执行通信，一个 job 覆盖所有 target rank
- `workspaceGm`：PUT 模式下作为 localAddr（源数据地址）

#### 5.4.3 `Run()` (L204-215) — AIC/AIV 分支

```cpp
if ASCEND_IS_AIC {
    RunMatmul();  // AIC: 计算
}
if ASCEND_IS_AIV {
    RunAllToAllAndReduceSum();  // AIV: 通信+求和
}
```

**同一个函数在 AIC 和 AIV 上都会执行，但各自只走自己的分支。** 这是混合核编程的典型模式：用 `ASCEND_IS_AIC` / `ASCEND_IS_AIV` 宏在编译期区分。

#### 5.4.4 `RunMatmul()` (L253-314) — AIC 侧

```cpp
for (uint32_t t = 0; t < commTurn; ++t) {
    // 1. 判断 head/tail tile，选 tiling
    bool isHeadTile = (t < splitAxisTileCnt);
    uint32_t curTileM = isHeadTile ? splitAxisTileSize : splitAxisTailSize;
    const auto& tileT = (commTurn == 1) ? mmT : (isHeadTile ? subMmT : tailMmT);

    // 2. 构建 Fragment 参数
    FragFragmentParams fragParams;
    fragParams.tileM = curTileM;
    fragParams.headRows = R * curTileM;  // 跨所有 rank 段的总 M
    fragParams.rankId = rankId;
    fragParams.rankSize = R;
    fragParams.mPerRank = chunkM;
    // ...

    // 3. 计算 A/C 地址偏移
    uint64_t tileMOffset = isHeadTile ? t * splitAxisTileSize : ...;
    params.aGM = aGm + tileMOffset * K;
    params.cGM = workspaceGm + tileMOffset * N * sizeof(CType);

    // 4. 调用 Fragment Kernel 做一次 matmul
    fragKernelImpl_(params);

    // 5. 通知 AIV 本 tile 计算完成
    CrossCoreSetFlag<0x2, PIPE_FIX>(0);
}
```

**核心思想**：
- 每个 tile 调用一次 `fragKernelImpl_`，覆盖 `R × curTileM` 行（所有 rank 段）
- `FragmentTensor` 让一次调用自动处理跨 rank 边界的 tile，无需 for r 循环
- 计算完一个 tile → `SetFlag` 通知 AIV

**Fragment 参数详解**：

| 字段 | 值 | 含义 |
|------|-----|------|
| `tileM` | curTileM | 当前 tile 每个 rank 段的行数 |
| `headRows` | R × curTileM | 当前 tile 跨所有 rank 段的总 M |
| `rankId` | 本卡 ID | 用于 Fragment 地址计算 |
| `rankSize` | R | fragment 数量 |
| `mPerRank` | chunkM | 每个 rank 的完整 M 分片 |
| `commTurn` | 1 | per-tile 调用，commTurn 恒为 1 |

**地址偏移计算**：
- `tileMOffset`：当前 tile 在 chunkM 内的 M 偏移
  - head tile: `t * splitAxisTileSize`
  - tail tile: `splitAxisTileCnt * splitAxisTileSize`（尾块在所有头块之后）
- `aGM = aGm + tileMOffset * K`：A 矩阵带 tile 偏移
- `cGM = workspaceGm + tileMOffset * N * sizeof(CType)`：workspace 带 tile 偏移

#### 5.4.5 `RunAllToAllAndReduceSum()` (L321-374) — AIV 侧

```cpp
for (uint32_t t = 0; t < commTurn; ++t) {
    // 1. 等 AIC 算完本 tile
    CrossCoreWaitFlag<0x2, PIPE_MTE2>(0);
    SyncAll<true>();

    // 2. AllToAll PUT (仅 block 0)
    if (GetBlockIdx() == 0) {
        allToAllC_.Commit<BARRIER_NONE>();  // 推送 C_i 到所有 rank win 区
        allToAllC_.Wait<BARRIER_DEVICE>();   // 等推送完成
    }
    SyncAll<true>();

    // 3. 计算 tile 参数 (head/tail)
    uint32_t tileM = ...;
    uint64_t tileMOffset = ...;

    // 4. reduceSum (所有 block 并行)
    //    srcAddr = win区 + tile偏移 (收到的数据)
    //    selfAddr = workspace + tile偏移 (自己的数据)
    //    dstAddr = cGm + tile偏移 (输出)
    reduceSum_.Init(outputSize, stride, rankSize, rankId, usedCoreNum, srcAddr, selfAddr, dstAddr);
    reduceSum_.ExecuteReduceSum(tileM, nSize);
}
```

**reduceSum 的三个地址**：
- `srcAddr`（win 区）：AlltoAll 后收到的其他 rank 的数据，布局是 rank-major
- `selfAddr`（workspace）：自己计算的 C_i，同样是 rank-major 布局
- `dstAddr`（cGm）：最终输出
- `stride = chunkM × N`：rank 间的元素偏移

**同步机制**：
- `CrossCoreWaitFlag<0x2, PIPE_MTE2>`：等待 AIC 的 SetFlag（跨核 AIC→AIV 同步）
- `SyncAll<true>()`：AIV 间同步（所有 AIV block 到齐）
- `allToAllC_.Wait<BARRIER_DEVICE>()`：等通信完成

---

### 5.5 `kernel/qmm_mx_kernel_rs_frag.h` — FragmentTensor MatMul 内核

这是 Blaze MatMul 的封装，核心是**用 FragmentTensor 消除 R 循环**。

#### 5.5.1 FragmentTensor 是什么？

FragmentTensor 是 APACE 提供的一种数据结构，它把**多个不连续（或连续但逻辑分离）的 GM 地址**打包成一个逻辑张量。Blaze 的 BlockMmad 遍历 tile 时，遇到 fragment 边界会自动切换数据源地址，同时保持每个 fragment 独立的 L1 缓存上下文。

```
传统 R 循环:                    FragmentTensor:
for r = 0..R:                    一次 matmul(R*chunkM)
  matmul(chunkM)                   BlockScheduler 遍历 R*chunkM/baseM 个 tile
  → R 次 matmul 调用               tile 跨 rank 边界时自动切换 fragment
  → R 次 L1 加载                   → 1 次调用，per-fragment L1 隔离
```

#### 5.5.2 核心类型 (L67-97)

```cpp
using BlockMmadFragC = QmmMxBlockMmadFragment<...>;  // 支持 Fragment 的 BlockMmad
using FragTensorA = FragmentTensor<2, MAX_FRAGMENT_COUNT, MakeLayoutA, AType>;
using FragTensorC = FragmentTensor<2, MAX_FRAGMENT_COUNT, MakeLayoutC, CType>;
```

- `BlockMmadFragC`：Blaze 的量化 MatMul block，支持 Fragment 输入
- `FragTensorA/C`：2 维 FragmentTensor，最多 `MAX_FRAGMENT_COUNT` 个 fragment

#### 5.5.3 `Run()` (L240-279)

```cpp
// 计算 totalLogicalM = headRows + mainSectionRows + tailRows
// = R*curTileM (所有 rank 段当前 tile 的总行数)
ProblemShape problemShape{totalLogicalM, Ni, Ki, 1};
BlockScheduler sch(problemShape, schParams);    // tile 调度器
BlockMmadFragC mmadFrag;
mmadFrag.Init(problemShape, l0TileShape, fragL1, false, dbL0c > 1);
BuildFragmentTensors(params);   // 构建 FragmentTensor
Process(params, problemShape, sch, mmadFrag);  // 遍历 tile 执行 matmul
```

**totalLogicalM 计算**：
- `headMainRows = headRows = R * curTileM`（HEAD region 行数）
- `mainSectionRows = tileCnt * R * tileM`（MAIN region 行数，本算子 tileCnt=0）
- `tailRows = R * paddedTailM`（TAIL region 行数）
- `totalLogicalM = headMainRows + mainSectionRows + tailRows`

#### 5.5.4 `BuildFragmentTensors()` (L368-406)

```cpp
for (uint32_t r = 0; r < rankSize; ++r) {
    addrListA[r] = params.aGM + r * mPerRank * dataBytesPerMRow;    // A 的第 r 个 rank 段
    addrListC[r] = cFragAddrs_[r];                                   // C 的第 r 个 rank 段
}
headFragA_ = MakeFragmentTensor(fragM, fragM, rankSize, k, addrListA);
```

把 R 个 rank 段的 GM 地址打包成 FragmentTensor，`fragmentSize = curTileM`（每个 rank 段的行数）。

**地址计算**：
- A 的 rank r 段地址：`aGM + r * chunkM * K * sizeof(AType)`
- C 的 rank r 段地址：`cGM + r * chunkM * N * sizeof(CType)`
- 虽然各段在 GM 中连续，但 FragmentTensor 保持逻辑隔离，per-fragment L1 独立

#### 5.5.5 `Process()` (L282-350) — tile 循环

```cpp
while (sch.GetTileIdx(blockIdx)) {
    auto singleShape = sch.GetBlockShape(blockIdx);  // 当前 tile 的 M/N
    sch.GetTileCoord(blockIdx, mPos, nPos);          // tile 坐标

    auto ctx = ResolveTileCtx(mPos, ...);  // 根据 mPos 判断 HEAD/MAIN/TAIL region

    auto blockA = ctx.fragA->Slice(coordA, shapeA);    // 从 fragment 切片
    auto blockC = ctx.fragC->Slice(coordC, shapeC);
    mmadFrag(blockA, gmBlockB, blockScaleA, ...);  // 执行一个 tile 的 matmul
}
```

`ResolveTileCtx()` (L496-529) 根据当前 tile 的 M 位置判断属于哪个 region（HEAD/MAIN/TAIL），选择对应的 FragmentTensor。

**Region 划分逻辑**：
- `mPos < headMainRows` → HEAD region，用 `headFragA_/headFragC_`
- `mPos < headMainRows + mainSectionRows` → MAIN region，用 `curMainA_/curMainC_`（需按 roundIdx 更新地址）
- 否则 → TAIL region，用 `tailFragA_/tailFragC_`

**B 矩阵处理**：
B 矩阵所有 rank 共享，不需要 FragmentTensor，直接从 GM 切片：
```cpp
auto gmBlockB = gmB.Slice(MakeCoord(0L, nPos), MakeShape(Ki, curNtile));
```

---

### 5.6 `kernel/reduce_sum_ref.h` — AIV reduceSum 实现

这个文件实现 AlltoAll 后的逐元素求和。

#### 5.6.1 算法思路

```
对于每行 row ∈ [tileM]:
  sum = 0
  for rank = 0..rankSize:
    data = (rank == rankId) ? workspace[row] : win区[rank*stride + row]
    sum += data
  output[row] = sum
```

#### 5.6.2 UB 布局 — pingpong 双 buffer + Te::MakeTensor

```
UB offset:
0                    sumSlotBytes       sumSlotBytes+inputSlotBytes
├─────────────────────┬──────────────────┬──────────────────┤
│  sum (float-sized,  │ inputBF16[0]     │ inputBF16[1]     │
│  BF16 view) (1份)   │ (ping)           │ (pong)           │
├─────────────────────┴──────────────────┴──────────────────┘
```

- `sum`：按 float 大小分配（用于 MulScalar 清零），BF16 视图用于 Add（950 硬件级精度）
- `inputBF16`：2 份（pingpong），`pingPongCount_ & 1` 交替使用
- Te::MakeTensor + Te::Copy + Te::Transform 创建/操作 tensor（不用 LocalTensor/VECIN）

#### 5.6.3 `ExecuteReduceSum()` (L61-137)

```cpp
// Te::MakeTensor 创建 sum tensor (float 视图清零, BF16 视图 Add)
auto sumF32Tensor = Te::MakeTensor(MakeMemPtr<UB, float>(sumOffset), ...);
auto sumUbTensor = Te::MakeTensor(MakeMemPtr<UB, DataType>(sumOffset), ...);

for (tileIdx = 0..rowNum) {
    // 1. 清零 sum (float MulScalar, float 0.0 二进制全 0 = BF16 0)
    Te::Transform<Inst::MulScalar>(sumF32Tensor, sumF32Tensor, 0.0f);
    PipeBarrier<PIPE_V>();

    // 2. 逐 rank 累加 (pingpong, BF16 直接 Add)
    for (curRank = 0..rankDim) {
        ppId = pingPongCount_ & 1;

        // GM → UB (Te::Copy)
        Te::Copy(copyGM2UB_, ubTensor, gmTensor);
        SetFlag<MTE2_V>(1); WaitFlag<MTE2_V>(1);

        // Add sumBF16 += inputBF16 (BF16 直接累加, 无 Cast)
        Te::Transform<Inst::Add>(sumUbTensor, sumUbTensor, ubTensor);
        SetFlag<V_MTE2>(1); WaitFlag<V_MTE2>(1);

        pingPongCount_++;
    }

    // 3. UB → GM (直接输出 BF16, 无 Cast)
    Te::Copy(copyUB2GM_, gmOutTensor, sumUbTensor);
}
```

**为什么用 BF16 直接累加？** 950 硬件支持 BF16 Add，直接累加无需 Cast float32 中间步骤，V 指令数从 3 条/rank 减少到 1 条/rank。精度验证通过（5/5 代表用例 ALL PASS）。

**pingpong 的作用**：读 rank N+1 的数据（MTE2）和算 rank N 的累加（V）可以并行，隐藏 GM 读取延迟。

**self/received 分离读取**：
- `curRank == rankId`：从 `selfAddr_`（workspace）读自己的数据
- `curRank != rankId`：从 `srcAddr_`（win 区）读收到的数据

#### 5.6.4 `SplitToCore()` (L167-180) — 多核行分配

```cpp
rowNum = totalRows / coreNum;      // 基础行数
remainder = totalRows % coreNum;   // 余数
startId = rowNum * coreId;
if (coreId < remainder) {
    rowNum += 1;        // 前 remainder 个核多分一行
    startId += coreId;
} else {
    startId += remainder;
}
```
典型的均匀切分：前 `remainder` 个核各多处理 1 行。

---

### 5.7 `kernel/reduce_sum_utils.h` — 数学工具

提供以下基础运算：

| 函数 | 功能 |
|------|------|
| `CeilDiv(a, b)` | 向上取整除法 `(a+b-1)/b` |
| `FloorDiv(a, b)` | 向下取整除法 `a/b` |
| `CeilAlign(a, b)` | 向上对齐到 b 的倍数 |
| `FloorAlign(a, b)` | 向下对齐到 b 的倍数 |
| `BlockAlignMod(a, b)` | 尾块模运算，模为 0 返回 b |
| `MIN(x, y)` | 取最小值 |

---

### 5.8 `kernel/mc2_kernel_utils.h` — SyncFunc

```cpp
template<HardEvent event>
void SyncFunc() {
    SetFlag<event>(eventID);
    WaitFlag<event>(eventID);
}
```

一个简单的同步原语工具：SetFlag + WaitFlag 配对使用，确保流水线阶段间正确同步。

---

### 5.9 `src/utils.h` — Host 工具

- `INFO_LOG / WARNING_LOG / ERROR_LOG`：日志宏，带 `fflush(stdout)`
- `ACL_CHECK / RT_CHECK`：错误检查宏，失败时输出文件名和行号
- `ReadFile(filePath, buffer, bufferSize)`：二进制文件读取，带 `stat` 检查和 `ifstream`
- `WriteFile(filePath, buffer, size, offset)`：二进制文件写入，带文件锁 `flock(LOCK_EX)`

---

### 5.10 `src/root_info_exchanger.h` — TCP rootInfo 交换

由于多进程 fork 后没有共享内存，rank 间需要通过 TCP 交换 HCCL 的 `HcclRootInfo`。

**工作流程**：
1. **ConnectPeers()**：rank 0 监听 TCP 端口，其他 rank 连接 rank 0
2. **ExchangeRootInfo()**：
   - rank 0：`HcclGetRootInfo` 生成 rootInfo → 发给所有 rank
   - rank > 0：接收 rootInfo
3. **TcpBarrier()**：简单的 TCP 同步屏障
   - rank 0：recv all → send all
   - rank > 0：send → recv

**端点格式**：`tcp://127.0.0.1:8998`，解析出 IP 和端口。

---

### 5.11 `scripts/gen_data.py` — CPU golden 数据生成

```python
# 每卡独立 A_i, 共享 B
for rank_id in range(rank_num):
    a_local = random fp8 [m, k]
    a_scale = random fp8_e8m0 [m, ceil(k/64), 2]
    a_dequant = a_local.float() * a_scale_broadcast   # 反量化
    c_i = matmul(a_dequant, b_dequant).to(bf16)       # [m, n]

# AlltoAll + reduceSum
for dst_rank in range(rank_num):
    output = zeros([chunk_m, n], float32)
    for src_rank in range(rank_num):
        chunk = c_list[src_rank][dst_rank*chunk_m : (dst_rank+1)*chunk_m]
        output += chunk.float()    # BF16 直接累加
    output = output.to(bf16)
    write to output/{dst_rank}/cpu_output.bin
```

**反量化过程**：
- Scale shape `[m, ceil(k/64), 2]` → reshape `[m, k/32]` → repeat 32 倍 → `[m, k]`
- `a_dequant = a_local.float() * a_scale_broadcast`
- 同理反量化 B

**累加精度**：在 float32 中累加，最后转 bfloat16，减少 BF16 累加误差。

---

### 5.12 `scripts/verify_result.py` — 精度比对

```python
for rank_id in range(rank_num):
    npu_out = np.fromfile(npu_out.bin, uint16).view(bfloat16).reshape(chunk_m, n)
    golden  = np.fromfile(cpu_output.bin, uint16).view(bfloat16).reshape(chunk_m, n)
    torch.allclose(golden, npu_out, rtol=1e-2, atol=1e-2)
```

- 精度标准：`atol=1e-2, rtol=1e-2`
- 逐 rank 比对，全部 PASS 才算通过

---

### 5.13 `CMakeLists.txt` — 构建配置

关键点：

| 配置项 | 值 | 说明 |
|--------|-----|------|
| 编译器 | `bisheng` | 昇腾 C/C++ 编译器 |
| 编译模式 | `-xasc` | Ascend kernel + host 统一编译 |
| 架构 | `--npu-arch=dav-3510` | 目标芯片 Ascend 950 |
| 优化 | `-O3` | 最高优化级别 |
| include 路径 | Apace / Blaze / tensor_api / 本算子 | 5 个 include 根 |
| 链接库 | `hccl_fwk, ascendcl, runtime, tiling_api` | HCCL 内部符号 + ACL 运行时 |

**三个 include 根**（覆盖所有 #include 形式）：
- `${OPS_TENSOR_INCLUDE_DIR}` → `blaze/...`
- `${OPS_TENSOR_INCLUDE_DIR}/tensor_api/include` → `tensor_api/tensor.h`
- `${OPS_TENSOR_INCLUDE_DIR}/tensor_api` → `include/tensor_api/...`

**`-Wl,--no-as-needed` hccl_fwk**：强制链接 `libhccl_fwk.so`，提供 HCCL 内部符号（HcclChannelAcquire 等）。

---

### 5.14 `run.sh` — 一键脚本

流程：`gen_data.py → cmake/make → 多 rank 运行 → verify_result.py`

支持的模式：
- `bash run.sh`：运行 cases.csv 全部用例
- `bash run.sh 1`：运行第 1 个用例
- `bash run.sh --cli m k n r h`：命令行模式
- `bash run.sh --skip-build`：跳过编译
- `bash run.sh --gen-only`：仅生成 CPU golden
- `bash run.sh --verify-only`：仅精度比对

CSV 模式下多 case 串行运行，第 2 个 case 起 `SKIP_BUILD=1` 复用编译产物。

---

### 5.15 `cases.csv` — 测试用例

34 个用例，覆盖不同 m/k/n/rankNum/headMSize 组合：

| 测试维度 | 示例 |
|----------|------|
| 不同 rankNum | 2 卡 / 4 卡 |
| 不同 M | 256 / 512 / 1024 / 2048 / 4096 / 8192 |
| 不同 K | 128 / 512 / 2048 / 3584 / 8192 |
| 不同 N | 256 / 1024 / 4096 / 8192 |
| 不同 headMSize | 128 / 256 / 512 / 1024 / 2048 |
| 非 2 的幂 | 4032 / 4160 / 800 |
| 小矩阵边界 | 256×3584×256, 512×128×1024 |

---

## 6. Kernel 内部 AIC ↔ AIV 流水线

以 `commTurn=2` 为例，展示 AIC 和 AIV 的流水线时序：

```
时间轴 ──────────────────────────────────────────────────────►

 AIC 侧 (RunMatmul):
 ┌───────────────────┐   ┌───────────────────┐
 │   t=0 matmul      │   │   t=1 matmul      │
 │ R*512=1024行 A×B  │   │ R*512=1024行 A×B  │
 │ → workspace[0:1024]│   │→workspace[1024:]  │
 └────────┬──────────┘   └────────┬──────────┘
          │                       │
          │ SetFlag<0x2>          │ SetFlag<0x2>
          ▼                       ▼

 AIV 侧 (RunAllToAllAndReduceSum):
          ┌───────────────────┐   ┌───────────────────┐
          │   t=0             │   │   t=1             │
          │ WaitFlag<0x2>     │   │ WaitFlag<0x2>     │
          │ ↓                 │   │ ↓                 │
          │ SyncAll           │   │ SyncAll           │
          │ ↓                 │   │ ↓                 │
          │ AllToAll PUT      │   │ AllToAll PUT      │
          │ (block 0 推送)    │   │ (block 0 推送)    │
          │ ↓                 │   │ ↓                 │
          │ Wait<BARRIER_DEV> │   │ Wait<BARRIER_DEV> │
          │ ↓                 │   │ ↓                 │
          │ SyncAll           │   │ SyncAll           │
          │ ↓                 │   │ ↓                 │
          │ reduceSum         │   │ reduceSum         │
          │ (所有block并行)   │   │ (所有block并行)   │
          │ R份求和→cGm       │   │ R份求和→cGm       │
          └───────────────────┘   └───────────────────┘

流水关系: AIC t=0 ──SetFlag──► AIV t=0 ──SyncAll──► AIC t=1 ──SetFlag──► AIV t=1
         (AIC t=1 可与 AIV t=0 部分重叠，因为 AIC 不依赖 AIV)
```

**要点**：
- AIC 计算不依赖通信，所以 AIC t=1 可以在 AIV t=0 执行时就开始算
- AIV 必须等 AIC SetFlag + SyncAll 后才能通信
- `CrossCoreSetFlag/WaitFlag` 是 AIC→AIV 的跨核通知，`SyncAll` 是 AIV 间的同步
- 每个 tile 的 AIC 计算 + AIV 通信/reduceSum 形成流水

---

## 7. 数据内存布局

### 7.1 A 矩阵 GM 布局

```
A [m, k] 连续存储, 按 rank 段排列 (m=2048, k=3584 为例):

┌──────────────────────┬──────────────────────┐
│   rank 0 的 A 段     │   rank 1 的 A 段     │
│   [0 : 1024, :]      │   [1024 : 2048, :]   │
│   chunkM=1024 行     │   chunkM=1024 行     │
└──────────────────────┴──────────────────────┘
↑                      ↑                      ↑
aGm+0                  aGm+1024*K            aGm+2048*K

FragmentTensor 打包:
frag[0] → aGm + 0 * chunkM * K    (rank 0 段)
frag[1] → aGm + 1 * chunkM * K    (rank 1 段)
```

### 7.2 workspace GM 布局

```
C_i = A_i × B, 按 rank 段排列 (m=2048, n=4096, bf16):

┌──────────────────────┬──────────────────────┐
│  rank 0 的 C 段      │  rank 1 的 C 段      │
│  [0:1024, :] bf16    │  [1024:2048, :] bf16 │
└──────────────────────┴──────────────────────┘
↑                      ↑
workspace+0            workspace+chunkM*N*2

cFragAddrs_[0] = workspace + 0 * chunkM * cBytesPerM
cFragAddrs_[1] = workspace + 1 * chunkM * cBytesPerM
```

### 7.3 win 区（AlltoAll 后收到的数据）

```
AlltoAll PUT 后, 本卡 win 区包含所有 rank 推送来的 chunk:

┌──────────────┬──────────────┬──────────────┐
│ rank 0 推来的 │ rank 1 推来的 │  ... rank R  │
│ C_0[本rank段] │ C_1[本rank段] │              │
└──────────────┴──────────────┴──────────────┘
↑              ↑              ↑
win+0          win+stride      win+2*stride
stride = chunkM * N (元素单位)
```

### 7.4 cGm 最终输出

```
┌──────────────────────┐
│  Σ_{r=0}^{R-1} C_r   │
│  [本rank段, :]        │
│  [chunkM, n]          │
└──────────────────────┘
↑
cGm + 0
```

### 7.5 reduceSum 地址关系

```
srcAddr  = win区 + tileMOffset * rowBytes   (收到的其他rank数据)
selfAddr = workspace + tileMOffset * rowBytes (自己的C_i)
dstAddr  = cGm + tileMOffset * rowBytes      (输出)

stride = chunkM * N  (rank间偏移)
对每行 row:
  sum = workspace[row] (self) + win[0*stride+row] + win[1*stride+row] + ...
  → cGm[row]
```

---

## 8. FragmentTensor 消除 R 循环

### 8.1 传统 R 循环 vs FragmentTensor

```
=== 传统 R 循环做法 ===

for r = 0..R-1:
  ┌──────────────────┐
  │ matmul(chunkM)   │  ← 每次只算一个 rank 段
  │ A[r*chunkM] × B  │
  │ → C[r*chunkM]    │
  └──────────────────┘
  R 次调用, R 次 L1 加载, R 次 SetFlag

=== FragmentTensor 做法 (本算子) ===

一次 matmul(R * chunkM):
┌──────────────────────────────────────────────┐
│  BlockScheduler 遍历 totalLogicalM / baseM   │
│                                              │
│  tile 0    tile 1    tile 2    tile 3  ...   │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐     │
│  │baseM │  │baseM │  │baseM │  │baseM │     │
│  └──────┘  └──────┘  └──────┘  └──────┘     │
│  ├─ rank 0 段 ──┤├─ rank 1 段 ──┤            │
│                  ↑                             │
│           tile 跨越 rank 边界时               │
│           BlockMmadFragC 自动切换 fragment    │
│           保持 per-fragment L1 隔离           │
└──────────────────────────────────────────────┘
1 次调用, FragmentTensor 打包 R 个地址
```

### 8.2 FragmentTensor 地址打包

```
fragA:
┌─────────┬─────────┬─────────┐
│ frag[0] │ frag[1] │ frag[2] │  (R 个 fragment)
│ rank0 A │ rank1 A │ rank2 A │
│ addr    │ addr    │ addr    │
└─────────┴─────────┴─────────┘
     │        │        │
     ▼        ▼        ▼
GM:  [rank0 A段][rank1 A段][rank2 A段]  (连续, 但逻辑隔离)

BlockMmadFragC 遍历时:
- tile 在 frag[0] 内 → 从 frag[0] 地址加载 A/L1
- tile 跨越 frag[0]→frag[1] → 自动切换, 独立 L1
- tile 在 frag[1] 内 → 从 frag[1] 地址加载
```

**好处**：1 次 matmul 调用覆盖所有 rank 段，L1 缓存按 fragment 隔离，性能与 R 循环等价但调用开销更小。

---

## 9. AllToAll PUT 通信数据流

以 `rankSize=2` 为例，展示 t=0 这一轮的通信：

### 9.1 通信前

AIC 已算完 C_i 写入 workspace：

```
rank 0 workspace:                rank 1 workspace:
┌────────┬────────┐              ┌────────┬────────┐
│C_0[r0] │C_0[r1] │              │C_1[r0] │C_1[r1] │
│chunk0  │chunk1  │              │chunk0  │chunk1  │
└────────┴────────┘              └────────┴────────┘
```

### 9.2 AllToAll PUT 执行

PUT = 主动推送（AIV 把自己的数据写到其他卡的 win 区）：

```
rank 0 推送:                     rank 1 推送:
┌────────┐─────► rank 0 win      ┌────────┐─────► rank 0 win
│C_0[r0] │  自留                  │C_1[r0] │  推到 rank 0
└────────┘                       └────────┘
┌────────┐─────► rank 1 win      ┌────────┐─────► rank 1 win
│C_0[r1] │  推到 rank 1           │C_1[r1] │  自留
└────────┘                       └────────┘
```

### 9.3 通信后

每卡 win 区包含所有 rank 的对应 chunk：

```
rank 0 win 区:                   rank 1 win 区:
┌────────┬────────┐              ┌────────┬────────┐
│C_0[r0] │C_1[r0] │              │C_0[r1] │C_1[r1] │
│(自己)  │(收到)  │              │(收到)  │(自己)  │
└────────┴────────┘              └────────┴────────┘
     │      │                         │      │
     │      └── srcAddr               │      └── srcAddr
     └── selfAddr (workspace)         └── selfAddr (workspace)
```

### 9.4 reduceSum 求和

```
rank 0: cGm[0:chunkM] = C_0[r0] + C_1[r0]
        (workspace[0] + win[1*stride])
                          ↑ srcAddr

rank 1: cGm[0:chunkM] = C_0[r1] + C_1[r1]
        (win[0*stride] + workspace[1*stride])
          ↑ srcAddr       ↑ selfAddr (workspace 偏移 chunkM)

注意: curRank==rankId 时从 selfAddr(workspace) 读
      其他 rank 从 srcAddr(win区) 读
```

---

## 10. reduceSum pingpong 流水

### 10.1 逐 rank 累加流水（rankSize=2）

```
rank 0 (ppId=0, ping):          rank 1 (ppId=1, pong):
┌────────────────────┐          ┌────────────────────┐
│ MTE2: GM→UB[BF16_0]│          │ MTE2: GM→UB[BF16_1]│
│   读 win/workspace │          │   读 win/workspace │
└─────────┬──────────┘          └─────────┬──────────┘
          │ SetFlag<MTE2_V>               │ SetFlag<MTE2_V>
          ▼                               ▼
┌────────────────────┐          ┌────────────────────┐
│ V: Cast BF16→F32   │          │ V: Cast BF16→F32   │
│   ubF32[0]         │          │   ubF32[1]         │
└─────────┬──────────┘          └─────────┬──────────┘
          │                               │
          ▼                               ▼
┌────────────────────┐          ┌────────────────────┐
│ V: Add sumF32 +=   │          │ V: Add sumF32 +=   │
│   ubF32[0]         │          │   ubF32[1]         │
└─────────┬──────────┘          └─────────┬──────────┘
          │ SetFlag<V_MTE2>               │ SetFlag<V_MTE3>
          ▼                               ▼
(允许 MTE2 读下一 rank)         ┌────────────────────┐
                                 │ V: Cast F32→BF16   │
                                 │   outBF16          │
                                 └─────────┬──────────┘
                                            │
                                            ▼
                                 ┌────────────────────┐
                                 │ MTE3: UB→GM        │
                                 │   写 cGm           │
                                 └────────────────────┘
```

**pingpong 优势**：rank 0 的 MTE2（读）和 rank 1 的 V（算）可以并行，隐藏 GM 延迟。

### 10.2 完整执行步骤

1. **清零累加器**：`Duplicate<float>(sumF32, 0.0, ...)`
2. **逐 rank 累加**（pingpong 交替）：
   - MTE2：GM → UB[BF16_pp]（读数据）
   - V：Cast BF16 → F32（类型转换）
   - V：Add sumF32 += F32（累加）
3. **Cast 输出**：Cast F32 → BF16
4. **写回 GM**：MTE3：UB → cGm

---

## 11. 完整数据流总结

```
gen_data.py
    │
    │  生成 input/{r}/input_a.bin (fp8), input_b.bin (fp8),
    │         input_scaleA.bin (e8m0), input_scaleB.bin (e8m0)
    │  生成 output/{r}/cpu_output.bin (bf16 golden)
    ▼
main.cpp (per rank process)
    │
    ├─ aclInit, setDevice(rankId)
    ├─ TCP 交换 rootInfo → HCCL 建链
    ├─ CommChannelBuilder → Apace CommContext
    ├─ ReadFile → hostA/B/ScaleA/ScaleB
    ├─ aclrtMalloc → deviceA/B/ScaleA/ScaleB/workspace/output
    ├─ aclrtMemcpy H2D
    ├─ Tiling 计算 (mmTiling + commTiling + subMmT + tailMmT)
    │
    ▼
Kernel <<<usedCoreNum>>>
    │
    ├─ AIC: for t in commTurn:
    │        ├─ 选 tiling (mmT/subMmT/tailMmT)
    │        ├─ Build FragmentTensor (R 个 rank 段地址)
    │        ├─ Blaze BlockMmad: A×B → workspace
    │        │   (FragmentTensor 消除 R 循环, 一次覆盖 R*curTileM)
    │        └─ CrossCoreSetFlag → AIV
    │
    ├─ AIV: for t in commTurn:
    │        ├─ CrossCoreWaitFlag ← AIC
    │        ├─ SyncAll (AIV 间同步)
    │        ├─ AllToAll PUT (block 0):
    │        │   workspace C_i → 各 rank win 区
    │        ├─ Wait<BARRIER_DEVICE>
    │        ├─ SyncAll
    │        └─ reduceSum (所有 block 并行):
    │            for rank in R:
    │              MTE2: win/workspace → UB[BF16_pp]
    │              V:    Cast BF16→F32
    │              V:    Add sumF32 += F32
    │            V:     Cast F32→BF16
    │            MTE3:  UB → cGm
    │
    ▼
main.cpp
    ├─ aclrtSynchronizeStream
    ├─ aclrtMemcpy D2H → hostOutput [chunkM, n] bf16
    └─ WriteFile → output/{r}/npu_out.bin
    │
    ▼
verify_result.py
    ├─ npu_out.bin vs cpu_output.bin
    ├─ atol=1e-2, rtol=1e-2
    └─ torch.allclose → PASS/FAIL
```

---

## 附录：关键术语速查

| 术语 | 全称 | 含义 |
|------|------|------|
| MC2 | Multi-Cards Communication + Compute fusion | 通算融合 |
| AIC | AI Cube | 矩阵计算核 |
| AIV | AI Vector | 向量计算核 |
| APACE | Ascend Parallel Async Compute Engine | 通算融合框架 |
| Blaze | - | MatMul 高性能框架 |
| FragmentTensor | - | 多地址打包逻辑张量 |
| PUT | Push to Unifying Target | 主动推送通信模式 |
| MXFP8 | MX-scale FP8 | 带 block scale 的 FP8 量化 |
| chunkM | chunk M | 每个 rank 的 M 分片大小 |
| commTurn | communication turn | 通信轮次 |
| win 区 | window area | 通信窗口区 |
| UB | Unified Buffer | 统一缓冲区 |
| GM | Global Memory | 全局显存 |
| HCCL | Huawei Collective Communication Library | 华为集合通信库 |

---

> 本文档基于代码实现逐行分析整理，覆盖从 Host 入口到 Kernel 执行的完整链路。如需了解设计决策的详细推演过程，请参阅 `docs/DESIGN.md`。
