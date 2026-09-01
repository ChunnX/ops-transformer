# 第二阶段详细设计：x1 AllGather + x1 Matmul 串联（端到端精度）

> 隶属：CCU_ALLGATHER_MATMUL_PLAN.md 第二阶段
> 目标：AllGather 通信 + Blaze matmul 串联，验证 C = concat(A) @ B 端到端精度
> 同步架构：CCU 通信 + AIV/AIC 融合 kernel（MIX_AIC_1_1，AIV 同步桥 + AIC matmul）
> 流设计：双流 streamCcu（CCU）+ streamAivAic（AIV 同步桥 + AIC matmul 融合 kernel）
> 参考：apace UDMA `all_gather_mx_matmul_udma_impl.h` 的 MIX_AIC_1_1 模式
> 前置条件：第一阶段已验证通过（commit `130afa1ef`，2025-08-17）
> 参考：通信层复用 02_allgather_add；matmul 层复用 ops-tensor `examples/mat_mul/mat_mul_basic`

---

## 0. Blaze 纯 matmul 组件确认（P6 闭环）

### 0.1 类型链（FP16 A/B → FP32 累加 → BF16 C）

```cpp
// 头文件（来自 ops-tensor/include/blaze/gemm/）
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/kernel/kernel_matmul_basic.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/utils/common_utils.h"

// 类型
using TypeA = half;          // FP16
using TypeB = half;          // FP16
using TypeC = bfloat16_t;    // BF16
using BiasType = float;
using LayoutA = AscendC::Te::NDExtLayoutPtn;  // ND row-major
using LayoutB = AscendC::Te::NDExtLayoutPtn;
using LayoutC = AscendC::Te::NDExtLayoutPtn;

// 组件
using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, TypeA, LayoutA, TypeB, LayoutB, TypeC, LayoutC, BiasType, LayoutC>;
using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape>;
using KernelImpl = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```

> **关键确认**：纯 matmul 与 QBMM 都用 `GemmUniversal` 类名，但特化头文件不同。纯 matmul 必须 `#include "kernel_matmul_basic.h"`（非 `kernel_qbmm_mx.h`），否则命中未特化 `static_assert`。

### 0.2 融合 kernel 签名（MIX_AIC_1_1，对齐 apace UDMA 风格）

```cpp
// TilingData POD 结构（host/device 共用，参照 apace quant_matmul_tiling_data.h 风格）
#pragma pack(push, 8)
struct alignas(8) MatmulTilingData {
    uint32_t m{0};          // = rankSize * M（AllGather 后总行数）
    uint32_t n{0};          // = N
    uint32_t k{0};          // = K
    uint32_t baseM{0};      // = 256
    uint32_t baseN{0};      // = 256
    uint32_t baseK{0};      // = 64（mat_mul_basic 默认，若 256 可行则改）
    uint32_t mL1{0};        // = baseM
    uint32_t nL1{0};        // = baseN
    uint32_t kL1{0};        // = baseK * 2
    uint32_t mTailCnt{1};   // 尾块拆分
    uint32_t nTailCnt{1};
    uint32_t usedCoreNum{0};// AIC 核数
    uint8_t  isHf32{0};     // FP16 输入时 = 0（FP32 累加）
};
#pragma pack(pop)

// 融合 kernel：AIV 同步桥 + AIC matmul（参照 apace UDMA all_gather_mx_matmul_udma_impl.h）
__global__ __aicore__ void all_gather_matmul_kernel(
    GM_ADDR sendBuf, GM_ADDR recvBuf, GM_ADDR weightBuf, GM_ADDR outputBuf,
    CcuMappedAddrs mappedAddrs,
    uint64_t token, uint64_t inputOffset, uint64_t outputOffset, uint64_t sliceSize,
    uint64_t goSize0, uint64_t goSize1, uint64_t goSize2, uint64_t goSize3,
    MatmulTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    AscendC::InitSocState();
    if ASCEND_IS_AIV { SyncBridgeProcess(...); }
    if ASCEND_IS_AIC { MatmulProcess(recvBuf, weightBuf, outputBuf, tilingData); }
}
```

> **对齐 apace UDMA**：apace `AllGatherQuantMatmulKernel`（L271-283）就是 `KERNEL_TYPE_MIX_AIC_1_1` + `if ASCEND_IS_AIV / if ASCEND_IS_AIC` 分支。区别是 apace AIV 走 UDMA 通信，我们 AIV 走 CCU 同步桥（参数桥+唤醒+轮询）。

### 0.3 Tiling（参照 mat_mul_basic.cpp L90-140）

```cpp
struct TilingConfig {
    int64_t mL1, nL1, kL1;
    int64_t baseM, baseN, baseK;
    static constexpr int L1_BUFFER_NUM = 2;
    static constexpr int L0C_DB = 1;
};

// FP16/BF16 默认 tiling
TilingConfig = {256, 256, 256, 256, 256, 64};
// 即 baseM=256, baseN=256, baseK=64, mL1=256, nL1=256, kL1=128(=baseK*2)
```

> **用户指定 baseM=baseN=baseK=256**，但 mat_mul_basic 示例的 baseK=64。需确认 baseK=256 是否可行（kL1=baseK*2=512，需确认 L1 容量）。若不行则用示例默认值 baseK=64。

### 0.4 launch 方式（融合 kernel，MIX_AIC_1_1）

```cpp
all_gather_matmul_kernel<<<tilingData.usedCoreNum, 0, streamAivAic>>>(
    sendBuf, recvBuf, weightBuf, outputBuf, mappedAddrs,
    taskArgs[2], taskArgs[3], taskArgs[4], taskArgs[5],
    taskArgs[6], taskArgs[7], taskArgs[8], taskArgs[9],
    tilingData);
```

> `usedCoreNum` = AIV+AIC 核数（MIX 1:1，每对 AIV+AIC 算 1 个 block）。tilingData 按值传递。
> **对比第一阶段**：第一阶段是 `comm_sync<<<1, nullptr, streamAiv>>>`（单 AIV 核），第二阶段改为融合 kernel `<<<usedCoreNum, 0, streamAivAic>>>`（多 AIV+AIC 对）。

---

## 1. 极简 case 定义

| 参数 | 值 | 说明 |
|------|-----|------|
| rankSize | 2 | 两卡 |
| M | 256 | 每 rank A 分片行数（AllGather 后总 M = rankSize*M = 512） |
| K | 256 | A 列数 = B 行数 |
| N | 256 | B 列数 = C 列数 |
| dtype A/B | FP16 (half) | Device 侧用 half；Host 侧用 uint16_t 承载 |
| dtype C | BF16 (bfloat16_t) | Device 侧用 bfloat16_t；Host 侧用 uint16_t 承载 |
| inputSize | `M*K*2 = 128KB` | 每 rank A 分片 |
| recvSize | `rankSize*M*K*2 = 256KB` | AllGather 后 A 总量（= matmul 的 A 输入） |
| weightSize | `K*N*2 = 128KB` | B 矩阵（每 rank 本地，不通信） |
| outputSize | `rankSize*M*N*2 = 256KB` | C 输出 |

**数据流**：
```
各 rank sendBuf (M×K, FP16)
    │ CCU AllGather
    ▼
recvBuf (rankSize*M × K, FP16)  ← matmul 的 A
    │ AIC matmul: A @ B
    ▼
outputBuf (rankSize*M × N, BF16)  ← C
```

---

## 2. 双流架构（CCU + MIX kernel）

| 流 | Kernel | 执行单元 | 职责 |
|---|---|---|---|
| `streamCcu` | `CcuAllGatherMesh1DMem2MemKernel` | CCU 微核 | AllGather 通信（第一阶段复用，零改动） |
| `streamAivAic` | `all_gather_matmul_kernel` | AIV + AIC（MIX_AIC_1_1） | AIV：参数桥→唤醒CCU→轮询→CrossCoreSetFlag；AIC：CrossCoreWaitFlag→Blaze matmul |

> **MIX_AIC_1_1 融合 kernel**：AIV 同步桥和 AIC matmul 融合在**同一个 `__global__ __aicore__` kernel** 里，通过 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)` 声明任务类型，内部用 `if ASCEND_IS_AIV { syncBridge(); } if ASCEND_IS_AIC { matmul(); }` 分支（参照 apace UDMA `all_gather_mx_matmul_udma_impl.h` L271-283）。AIV 和 AIC 是同一 kernel launch 的不同核，不是两个独立 kernel。

### 2.1 同步链路（5 步）

```
streamAivAic (MIX kernel)              streamCcu
──────────────────────────             ──────────────
launch all_gather_matmul_kernel        launch ccu_kernel
  [AIV 核]                              [CCU 微核]
  写 taskArgs → GM                     EventWait(events[0]) ←阻塞
  写 ckeAddr0=1 ──────────────────────→ (唤醒)
                                        AllGather 传输
  轮询 ckeAddr1                         EventRecord(events[1])
  ckeAddr1==1 ───────────────────────── (CCU 结束)
  CrossCoreSetFlag<0x2, PIPE_MTE3>
  (AIV 核结束)

  [AIC 核]
  CrossCoreWaitFlag<0x2, PIPE_MTE2>
  Blaze matmul(A=recvBuf, B=weight → C=output)
  (AIC 核结束)
```

1. AIV 写 taskArgs → GM（参数桥）+ DataSyncBarrier
2. AIV 写 ckeAddr0=1 → 唤醒 CCU
3. CCU 执行 AllGather → EventRecord(events[1])
4. AIV 轮询 ckeAddr1==1 → 通信完成
5. **AIV CrossCoreSetFlag<0x2, PIPE_MTE3> → 唤醒 AIC**（同一 kernel 内 AIC 核 CrossCoreWaitFlag<0x2, PIPE_MTE2> 等待，参照 apace UDMA L248/L351）

> CrossCoreSetFlag/WaitFlag 的 flag ID=0x2、PIPE 设置参照 apace UDMA 路线（`all_gather_mx_matmul_udma_impl.h` L248 + `all_gather_qbmm_mx_kernel.h` L351）。

### 2.2 CrossCoreSetFlag 参数（参照 apace UDMA L248）

```cpp
// AIV 侧（SyncBridgeProcess 末尾）：
AscendC::CrossCoreSetFlag<0x2, AscendC::HardEvent::PIPE_MTE3>(0);

// AIC 侧（MatmulProcess 头部）：
AscendC::CrossCoreWaitFlag<0x2, AscendC::HardEvent::PIPE_MTE2>(0);
```

> flag ID=0x2（避开 CCU Event 使用的资源）。AIC 用 `PIPE_MTE2` 等待（matmul 数据搬入前），AIV 用 `PIPE_MTE3` 设置（AIV 结束前）。参数 (0) 对齐 apace UDMA L248/L351。
> **多 AIV 核门控**：SyncBridgeProcess 开头 `if (GetBlockIdx() == 0)` 包裹参数桥/唤醒/轮询，`SyncAll<true>()` 后所有 AIV 核一起 `CrossCoreSetFlag`（参照 apace UDMA L251/L256）。
> **rankSize==1 短路保护**：传空 `CcuMappedAddrs{}` 时（startVa[0]==0），所有 AIV 核直接 `CrossCoreSetFlag(0)` + return，不走参数桥/门控/SyncAll。

---

## 3. 融合 kernel（all_gather_matmul_kernel，MIX_AIC_1_1）

### 3.1 整体结构（参照 apace UDMA all_gather_mx_matmul_udma_impl.h）

```cpp
__global__ __aicore__ void all_gather_matmul_kernel(
    GM_ADDR sendBuf, GM_ADDR recvBuf, GM_ADDR weightBuf, GM_ADDR outputBuf,
    CcuMappedAddrs mappedAddrs,
    uint64_t token, uint64_t inputOffset, uint64_t outputOffset, uint64_t sliceSize,
    uint64_t goSize0, uint64_t goSize1, uint64_t goSize2, uint64_t goSize3,
    MatmulTilingData tilingData)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    AscendC::InitSocState();

    if ASCEND_IS_AIV {
        SyncBridgeProcess(sendBuf, recvBuf, mappedAddrs,
                          token, inputOffset, outputOffset, sliceSize,
                          goSize0, goSize1, goSize2, goSize3);
    }
    if ASCEND_IS_AIC {
        MatmulProcess(recvBuf, weightBuf, outputBuf, tilingData);
    }
}
```

> 参照 apace UDMA `all_gather_mx_matmul_udma_impl.h` L271-283：单 kernel + `KERNEL_TYPE_MIX_AIC_1_1` + `if ASCEND_IS_AIV / if ASCEND_IS_AIC` 分支。

### 3.2 AIV 分支：SyncBridgeProcess（复用第一阶段 comm_sync + CrossCoreSetFlag）

```cpp
__aicore__ inline void SyncBridgeProcess(...) {
    // ---- 第一阶段 comm_sync 完全复用 ----
    // 参数桥：写 taskArgs[0..9] → GM + DataSyncBarrier
    // 唤醒 CCU：写 ckeAddr0=1 + DataSyncBarrier
    // 轮询：读 ckeAddr1==1 + Nop<800>

    // ---- 第二阶段新增 ----
    AscendC::CrossCoreSetFlag<0x2, AscendC::HardEvent::PIPE_MTE3>();
}
```

> 相对第一阶段 comm_sync：仅末尾新增一行 `CrossCoreSetFlag`。参数桥/唤醒/轮询三段逻辑逐字节复用。

### 3.3 AIC 分支：MatmulProcess（Blaze matmul + CrossCoreWaitFlag）

```cpp
__aicore__ inline void MatmulProcess(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM,
                                     MatmulTilingData tilingData) {
    // ---- 等待 AIV 唤醒 ----
    AscendC::CrossCoreWaitFlag<0x2, AscendC::HardEvent::PIPE_MTE2>();

    // ---- Blaze matmul（复用 mat_mul_basic 组装方式，tilingData 解包）----
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, half, LayoutA, half, LayoutB, bfloat16_t, LayoutC, float, LayoutC>;
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;

    using Params = typename MatmulKernel::Params;
    Params params = {
        {tilingData.m, tilingData.n, tilingData.k, 1L},                          // ProblemShape
        {aGM, bGM, cGM, nullptr, nullptr, nullptr,                                // BlockMmadParams
         tilingData.mL1, tilingData.nL1, tilingData.kL1,
         tilingData.baseM, tilingData.baseN, tilingData.baseK,
         TilingConfig::L1_BUFFER_NUM, TilingConfig::L0C_DB},
        {},                                                                       // BlockEpilogueParams (empty)
        {tilingData.mL1, tilingData.nL1, tilingData.kL1,                          // BlockSchedulerParams
         tilingData.baseM, tilingData.baseN, tilingData.baseK,
         tilingData.mTailCnt, tilingData.nTailCnt,
         1, 1, 1, 1, tilingData.isHf32, Blaze::Gemm::L2_CACHE_DEFAULT}
    };
    MatmulKernel kernel;
    kernel(params);
}
```

### 3.4 matmul 输入

- **A** = recvBuf（AllGather 结果，`rankSize*M × K`，FP16）
- **B** = weightBuf（本 rank 本地权重，`K × N`，FP16，Host 预加载）
- **C** = outputBuf（输出，`rankSize*M × N`，BF16）
- **bias** = nullptr（无 bias）

---

## 5. Host 编排（AllGatherMatmulCustom）

### 5.1 相对第一阶段的改动

1. 新增 `weightBuf`（B 矩阵）和 `outputBuf`（C 输出）分配
2. `streamAiv` 改名为 `streamAivAic`（AIV+AIC 融合 kernel）
3. AIV `comm_sync` 和 AIC `matmul_kernel` 合并为单个 `all_gather_matmul_kernel`（MIX_AIC_1_1）
4. `AllGatherCustom` → `AllGatherMatmulCustom`（新增 matmul 参数）
5. Host launch 从两个独立 kernel → 一个融合 kernel

### 5.2 流程

```
1-10. CCU 通信编排（完全复用第一阶段 AllGatherCustom 步骤 1-10）
11. 下发融合 kernel all_gather_matmul_kernel（MIX_AIC_1_1，streamAivAic）
     all_gather_matmul_kernel<<<usedCoreNum, 0, streamAivAic>>>(
         sendBuf, recvBuf, weightBuf, outputBuf, mappedAddrs,
         taskArgs[2..9], tilingData)
```

> **A = recvBuf**：matmul 的 A 输入直接用 AllGather 的 recvBuf，零拷贝。M = rankSize*M（AllGather 后的总行数）。
> **融合 kernel**：AIV 同步桥 + AIC matmul 在同一 kernel launch（KERNEL_TYPE_MIX_AIC_1_1），参照 apace UDMA。

### 5.3 权重 B 加载

```cpp
// Host 侧预加载权重 B（每 rank 相同或不同，第一阶段极简 case 用相同值）
void *weightBuf = nullptr;
ACL_CHECK(aclrtMalloc(&weightBuf, weightSize, ACL_MEM_MALLOC_HUGE_ONLY));
// Host 侧用 uint16_t 填充 FP16 数据（避免 half 软浮点符号）
ACL_CHECK(aclrtMemcpy(weightBuf, weightSize, hostWeight, weightSize, ACL_MEMCPY_HOST_TO_DEVICE));
```

---

## 6. Sample / main（基于第一阶段增强）

### 6.1 相对第一阶段的改动

1. 新增 N=256 参数
2. 新增 weightBuf 分配 + B 数据填充
3. 新增 outputBuf 分配
4. `streamAiv` 改名为 `streamAivAic`（AIV+AIC 融合 kernel）
5. 调用 `AllGatherMatmulCustom`（替代 `AllGatherCustom`）
6. 结果输出改为 outputBuf → bin 文件（BF16，uint16_t 承载）

### 6.2 数据填充

| Buffer | 类型 | 填充值 | 说明 |
|--------|------|--------|------|
| sendBuf (A 分片) | FP16 | rank0=1.0(0x3C00), rank1=2.0(0x4000) | 与第一阶段一致 |
| weightBuf (B) | FP16 | 全 1.0(0x3C00) | 简化校验：A@B = A×1 = A（BF16） |
| outputBuf (C) | BF16 | - | 输出，Host 读回校验 |

> **校验简化**：B 全 1.0 时，C = A @ B = A（每行求和×1=A 的列和，但 B 是 K×N 全 1，所以 C[i][j] = sum(A[i][0..K])。若 A 每行全相同值 v，则 C[i][j] = K×v）。需重新设计 golden。

### 6.3 Golden 计算

A = `[rank0: 65536 个 1.0 | rank1: 65536 个 2.0]`，shape = (512, 256)
B = 全 1.0，shape = (256, 256)
C = A @ B，shape = (512, 256)，BF16

C[i][j] = sum(A[i][0..255]) = 256 × A[i][0]
- rank0 行（i < 256）：C[i][j] = 256 × 1.0 = 256.0
- rank1 行（i >= 256）：C[i][j] = 256 × 2.0 = 512.0

Python 校验：
```python
import numpy as np
data = np.fromfile("output/0/npu_out.bin", dtype=np.float16)  # BF16 读回需特殊处理
# BF16 → float32：uint16 左移 16 位
raw = np.fromfile("output/0/npu_out.bin", dtype=np.uint16)
c = np.frombuffer((raw.astype(np.uint32) << 16).tobytes(), dtype=np.float32)
assert np.all(c[:256*256] == 256.0)   # rank0 行
assert np.all(c[256*256:] == 512.0)   # rank1 行
```

---

## 7. CMake 构建

### 7.1 相对第一阶段的改动

1. 新增 Blaze + tensor_api include 路径（3 条）
2. 新增 Blaze 链接库（`platform tiling_api register`）
3. 新增 ops-tensor FetchContent 拉取

### 7.2 CMakeLists.txt（第一阶段基础上增量）

```cmake
# ---- 第一阶段完全保留 ----
# ...（7 条 CANN include + 6 个 CANN 链接库 + pthread + dav-3510 + 编译选项）...

# ---- 第二阶段新增：Blaze + tensor_api ----
# ops-tensor 拉取（仿 apace ST cmake/third_party/ops-tensor.cmake）
include(FetchContent)
FetchContent_Declare(ops-tensor
    GIT_REPOSITORY https://gitcode.com/cann/ops-tensor.git
    GIT_TAG 602d648ae62e0bb7b51494451e32e5c7bd50c1c3
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(ops-tensor)

# Blaze include 路径（3 条）
target_include_directories(ccu_all_gather_matmul PRIVATE
    ${ops-tensor_SOURCE_DIR}/include
    ${ops-tensor_SOURCE_DIR}/include/tensor_api/include
    ${ops-tensor_SOURCE_DIR}/include/tensor_api
)

# Blaze 链接库
target_link_libraries(ccu_all_gather_matmul PRIVATE
    platform
    tiling_api
    register
)
```

> tensor_api 是 ops-tensor 的 submodule，需 `git submodule update --init --recursive`。

---

## 8. 工程结构

```
apace/examples/ccu_all_gather_matmul/
├── ccu_all_gather_matmul_kernel.asc   # 单文件：CCU kernel + AIV comm_sync + AIC matmul + Host 编排 + main
├── CMakeLists.txt                      # 第一阶段 + Blaze 增量
├── run.sh                              # 编译+运行+CCU_SCHED
└── README.md
```

> 保持 .asc 单文件（用户确认）。Blaze kernel 写在同一 .asc 中，CMake 仍用 `add_executable(... .asc)` + `find_package(ASC)`。

---

## 9. 验证

### 9.1 验收标准

- 程序正常退出，不崩溃、不卡死
- `output/{rank}/npu_out.bin` 生成（各 256KB，BF16）
- Python 校验：C 前 256×256 元素 == 256.0，后 256×256 == 512.0
- 两卡 output 内容相同（A 相同、B 相同 → C 相同）

### 9.2 风险点

| 风险 | 影响 | 缓解 | 状态 |
|------|------|------|------|
| baseK=256 是否可行（kL1=512） | L1 容量不足 | 若不行用 mat_mul_basic 默认 baseK=64 | ⏳ 待验证 |
| CrossCoreSetFlag flag ID 冲突 | 同步异常 | 用 0x2 避开 CCU Event | ⏳ 待验证 |
| BF16 Host 侧读回 | uint16_t 承载 + Python 转换 | 已规划方案（§6.3） | ✅ 已规划 |
| .asc 编译 Blaze kernel | 编译报错 | 若不行拆分 .cpp | ⏳ 待验证 |
| ops-tensor FetchContent 网络 | 拉取失败 | 备选手动 clone + cp | ⏳ 待验证 |

---

## 10. 编译与运行问题记录（实测修复）

> 阶段二在多环境编译验证过程中遇到的问题及修复，按发现顺序记录。修复 commit：`2a8aad5ab`。

### 10.1 `unknown type name 'CcuMappedAddrs'`

**根因**：文件拆分后 include 顺序导致前向引用。`CcuMappedAddrs` 结构体定义在 `.asc` L185，但 `matmul_fusion_kernel.h`（L37 include）的 `SyncBridgeProcess` 和 `host_launcher.h`（L670 include）的 `AllGatherMatmulContext` 都引用了它——include 发生在定义之前。

**修复**：把 `CcuMappedAddrs` 及其依赖常量（`HCOMM_HOST_TASK_ARG_NUM`、`CKE_NUM`）前移到 `#include "matmul_fusion_kernel.h"` 之前（L39-43），删除原 L107/L109 和 L191-194 的重复定义。

**文件**：`ccu_all_gather_matmul_kernel.asc`

### 10.2 `no matching function for call to MatmulProcess`

**根因**：类型不匹配。kernel 入参 `__gm__ void *recvBuf`（继承阶段一 `comm_sync` 风格），但 `MatmulProcess` 签名用 `GM_ADDR`（Blaze `BlockMmadParams` 字段类型）。`__gm__ void *` 不能隐式转换为 `GM_ADDR`。

参考代码 `mat_mul_basic.cpp` 的 kernel 入参直接用 `GM_ADDR`，不存在此问题。但我们的 kernel 入参继承了阶段一的 `__gm__ void *`（`SyncBridgeProcess` 需要它做 `reinterpret_cast<uint64_t>` 取地址值写 Variable）。

**修复**：`MatmulProcess` 参数从 `GM_ADDR` 改为 `__gm__ void *`（与 kernel 入参一致），函数内部两步 cast 供 Blaze Params 使用：

```cpp
// __gm__ void* → uint64_t（阶段一 comm_sync 已验证）→ GM_ADDR（apace L235 先例）
GM_ADDR aGmAddr = reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(aGM));
```

**文件**：`matmul_fusion_kernel.h`

### 10.3 `CrossCoreWaitFlag/CrossCoreSetFlag` 编译错误

**根因**：原代码使用 `AscendC::HardEvent::PIPE_MTE2` / `AscendC::HardEvent::PIPE_MTE3`，编译器未识别 `AscendC::HardEvent` 命名空间。

**修复**：简化为 `PIPE_MTE2` / `PIPE_MTE3`（宏已在 `kernel_operator.h` 中定义）。

**文件**：`matmul_fusion_kernel.h`

### 10.4 `aclrtFinalize` 未定义

**根因**：`aclrtFinalize` API 不存在，正确的 API 是 `aclFinalize`。

**修复**：`aclrtFinalize()` → `aclFinalize()`。

**文件**：`sample_main.cpp`

### 10.5 `ld.ldd: error: undefined symbol: dlopen`

**根因**：`dlopen`/`dlsym`/`dlclose` 在 `libdl` 中。CANN 的 `ascendcl` / `acl_rt` 等库内部调用 `dlopen` 动态加载 HCCL 插件，但不传递依赖，可执行文件需显式链接 `dl`。

**修复**：CMakeLists.txt `target_link_libraries` 新增 `dl`。

**文件**：`CMakeLists.txt`

### 10.6 `ops-tensor not found at /root/code/ops-tensor/include/blaze`

**根因**：CMakeLists.txt 中 `OPS_TENSOR_ROOT` 硬编码为 `$ENV{HOME}/code/ops-tensor`。换环境（root 用户，HOME=/root）时路径变为 `/root/code/ops-tensor`，但 ops-tensor 不在此处。

**修复**：暂保持硬编码 `$ENV{HOME}/code/ops-tensor`，跨环境时手动修改 `OPS_TENSOR_ROOT`。

**待解决**：后续考虑改为环境变量 `OPS_TENSOR_ROOT` 覆盖或自动查找。曾尝试多候选路径查找方案（环境变量 > ~/code/ops-tensor > /root/code/ops-tensor），但用户认为未真正解决问题，已回退。

**文件**：`CMakeLists.txt`

### 10.7 `tensor_api submodule` 初始化失败

**根因**：部分 WSL 环境下 `.git` 目录虽然显示正确属主和权限（`drwxr-xr-x wuhua:wuhua`），但实际无法写入（`touch` 测试失败）。`git submodule update --init` 因无法写 `.git/config` 和 `.git/objects` 而失败。

**修复**：需 `sudo` 执行或修复 `.git` 目录权限（`sudo chown -R $(whoami) .git/`）。

**文件**：`~/code/ops-tensor`（非本仓库）

### 10.8 运行报错 `return err retcode 1`

**根因**：新环境未设置 `HCCL_OP_EXPANSION_MODE=CCU_SCHED` 环境变量。CCU 相关 API（`AcquireCcuChannels`、`CreateCcuInsWithEnabledDies` 等）在非 CCU_SCHED 模式下返回 `HCCL_E_PARA=1`。

**修复**：`run.sh` 已包含 `export HCCL_OP_EXPANSION_MODE=CCU_SCHED`。手动运行编译产物时需自行 export。同时在 `sample_main.cpp` 的 `main()` 中新增环境变量检查日志，启动时打印 CCU_SCHED 状态。

**文件**：`sample_main.cpp`
