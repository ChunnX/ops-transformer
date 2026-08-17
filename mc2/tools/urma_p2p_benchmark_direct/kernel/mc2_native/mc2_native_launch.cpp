// ============================================================================
// mc2_native_launch.cpp — mc2_native 模式的 kernel 入口 + <<<>>> launch 函数
//
// 两部分:
//   1. __global__ mc2_native_bench_kernel: kernel 入口, 由 GPU/NPU 调度器调用
//      - KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY): 仅 AIV 核心
//      - 从参数构造 UrmaP2pBenchTiling
//      - 创建 Mc2NativeBenchOp 实例, Init + Process
//
//   2. mc2_native_bench_launch: Host 侧调用, 通过 <<<>>> 语法启动 kernel
//      - blockDim: 核心数 (当前固定 1)
//      - stream: ACL stream
//      - hcclContextGM: HcclCombinOpParam 的 GM 地址 (Host 侧 build_hccl_context 构造)
//      - workspace: 本端 URMA workspace 地址
//      - timingGm: timing 输出 GM 地址
//      - tiling: tiling 参数结构体
// ============================================================================

#include "kernel_operator.h"
#include "../common/urma_p2p_bench_tiling.h"
#include "mc2_native_kernel.h"
#include "mc2_native_launch.h"

using namespace AscendC;
using namespace UrmaP2pBench;

// ============================================================================
// mc2_native_bench_kernel — kernel 入口
//
// 参数由 <<<>>> 语法传入, Host 侧 mc2_native_bench_launch 负责拆包。
// kernel 内部:
//   1. KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY): 声明为纯 AIV 任务
//   2. TPipe pipe: 管理 UB 缓冲区分配
//   3. 从标量参数构造 UrmaP2pBenchTiling
//   4. Mc2NativeBenchOp op: 创建 kernel 类实例
//   5. op.Init: 绑定 GM 地址 + 分配 UB
//   6. op.Process: InitRankComm → Write/Drain 循环 → 打点
// ============================================================================
extern "C" __global__ __aicore__ void mc2_native_bench_kernel(
    GM_ADDR hcclContextGM, GM_ADDR workspace, GM_ADDR timingGm,
    uint32_t peer, uint32_t numSqe, uint64_t sqeBytes, uint64_t bufBytes,
    uint32_t dbInterval, uint32_t channelSlot, uint32_t cqOwnerSlot, uint32_t stateStride,
    uint32_t loopCount, uint32_t benchMode)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    UrmaP2pBenchTiling tiling;
    // 从标量参数构造 tiling (Host 侧通过 <<<>>> 传入)
    tiling.peer = peer;
    tiling.numSqe = numSqe;
    tiling.sqeBytes = sqeBytes;
    tiling.bufBytes = bufBytes;
    tiling.dbInterval = dbInterval;
    tiling.channelSlot = channelSlot;
    tiling.cqOwnerSlot = cqOwnerSlot;
    tiling.stateStride = stateStride;
    tiling.loopCount = loopCount;
    tiling.benchMode = benchMode;
    // 创建 kernel 实例并执行
    Mc2NativeBenchOp op;
    op.Init(hcclContextGM, workspace, timingGm, &pipe, &tiling);
    op.Process();
}

// ============================================================================
// mc2_native_bench_launch — Host 侧 launch 函数
//
// 被 test_urma_p2p_bench.cpp 的 Sample() 调用 (sqeMode==3 分支):
//   mc2_native_bench_launch(BLOCK_NUM, stream,
//       (uint64_t)ctx->hcclContextDev, (uint64_t)ctx->workspaceDev,
//       (uint64_t)timingGm, &tilingHost);
//
// 通过 <<<>>> 语法启动 kernel:
//   mc2_native_bench_kernel<<<blockDim, nullptr, stream>>>(...)
//   - blockDim: 核心数 (当前固定 1, 单核执行)
//   - nullptr: l2Ctrl (不使用)
//   - stream: ACL stream
// ============================================================================
void mc2_native_bench_launch(uint32_t blockDim, void *stream,
                             uint64_t hcclContextGM, uint64_t workspace,
                             uint64_t timingGm, UrmaP2pBenchTiling *tiling)
{
    mc2_native_bench_kernel<<<blockDim, nullptr, stream>>>(
        (void*)hcclContextGM, (void*)workspace, (void*)timingGm,
        tiling->peer, tiling->numSqe, tiling->sqeBytes, tiling->bufBytes,
        tiling->dbInterval, tiling->channelSlot, tiling->cqOwnerSlot, tiling->stateStride,
        tiling->loopCount, tiling->benchMode);
}
