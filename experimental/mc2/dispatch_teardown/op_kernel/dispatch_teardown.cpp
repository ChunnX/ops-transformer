/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 *
 *  dispatch_teardown.cpp —— DispatchTeardown 的 AIV 核函数。
 *  语义：轮询本卡状态区 status[0]，等到 == 1（由 CCU dispatch_setup 的 LocalCopyNb 写入）
 *        后写 done[0] = 1 并结束。这就是“等状态位”的接收端。
 *
 *  与 dispatch_setup 的字节级契约：
 *    - status 是 int32 GM，slot 0 为就绪标志，值 1 表示本卡数据齐；
 *    - 状态区按 32B 对齐分配（≥ 8 个 int32），因为下面 DataCopy 一次搬 32B（8 int32），
 *      若只分配 4 字节会 MTE2 越界（aicore error）。单 die 只有 slot 0 有意义，其余清 0。
 *
 *  轮询范式裁剪自 quant_all_reduce/kernel/mte_comm.h 的 ReadStatus 忙等循环，
 *  但 dtype 由 float 改为 int32（与 CCU LocalCopyNb 写入的 int32 对齐），且只判单 flag。
 *
 *  a6-TODO：
 *    [T3] 跨流可见性 / cache：AIV 每轮重发 DataCopy 从 GM 取数；若 a6 上 CCU 的写
 *         不能被 AIV 及时观测（L2 cache），需配合 dcci / 非 cache 读。all_gather 编译用了
 *         -mllvm -cce-aicore-dcci-before-kernel-end=false，本类同步对 cache 敏感，需上板核实。
 *    [T4✓] 忙等超时：已加 sanity 上限熔断（见下 poll 段，超限带诊断退出、不写 done）；§24 host 侧再兜底。
 *    [T12] PRINTF 设备符号若 a6 不同（AscendC::printf vs PRINTF），按 a6 头调整。
 */
#include "kernel_operator.h"

namespace {
constexpr int32_t STATUS_BLOCK_INT32 = 8;   // 一次搬 32B = 8 个 int32（32B 对齐）
constexpr int32_t STATUS_READY_SLOT = 0;    // 就绪标志在 slot 0

// 与 op_host/op_tiling/dispatch_teardown_tiling.h 的 TilingData 字段【同序同类型】，
// 直接 POD cast 读（SaveToBuffer 从偏移 0 起按声明序列化，两 uint32 连续无填充）。
struct DispatchTeardownTilingPod {
    uint32_t totalCnt;
    uint32_t expectVal;
};
}  // namespace

extern "C" __global__ __aicore__ void dispatch_teardown(
    GM_ADDR status, GM_ADDR done, GM_ADDR workspace, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (AscendC::GetBlockIdx() != 0) {
        return;  // blockDim=1，仅 0 号核工作
    }

    __gm__ DispatchTeardownTilingPod *tiling = (__gm__ DispatchTeardownTilingPod *)tilingGM;
    const int32_t expectVal = static_cast<int32_t>(tiling->expectVal);  // = 1

    AscendC::GlobalTensor<int32_t> statusGt;
    statusGt.SetGlobalBuffer((__gm__ int32_t *)status, STATUS_BLOCK_INT32);
    AscendC::GlobalTensor<int32_t> doneGt;
    doneGt.SetGlobalBuffer((__gm__ int32_t *)done, STATUS_BLOCK_INT32);

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf;
    pipe.InitBuffer(ubBuf, STATUS_BLOCK_INT32 * sizeof(int32_t));  // 32B
    AscendC::LocalTensor<int32_t> ub = ubBuf.Get<int32_t>();

    AscendC::printf("[dispatch_teardown] blk=%d start poll, expect=%d\n",
                    AscendC::GetBlockIdx(), expectVal);

    // ============ 忙等：每轮重新从 GM 搬 32B 到 UB，读 slot0，直到 == expectVal ============
    //   [T4] sanity 上限：CCU 若始终没写 1（建链失败 / 地址错 / DMA 没落）就不死等——超限带诊断退出，
    //   让 host 校验 done!=1 显式 FAIL，而非静默卡死成 aicore timeout（核内兜底，与 §24 host 侧熔断互补）。
    constexpr uint64_t POLL_MAX_SPINS = 1000000000UL;   // 远超任何合法等待，仅作死等熔断阈值
    int32_t v = 0;
    bool ready = false;
    for (uint64_t spins = 0; spins < POLL_MAX_SPINS; ++spins) {
        AscendC::DataCopy(ub, statusGt, STATUS_BLOCK_INT32);  // GM -> UB（MTE2，每轮重取）
        AscendC::PipeBarrier<PIPE_ALL>();                     // 等 DMA 落地，scalar 再读 UB（[T3] cache 敏感）
        v = ub.GetValue(STATUS_READY_SLOT);
        if (v == expectVal) { ready = true; break; }
    }
    if (!ready) {
        AscendC::printf("[dispatch_teardown] blk=%d TIMEOUT: status[0]=%d != expect=%d —— CCU 未写就绪位，不写 done（host 将判 FAIL）\n",
                        AscendC::GetBlockIdx(), v, expectVal);
        return;   // 不写 done → done 保持 0 → host 校验 done!=1 → 显式 FAIL，而非死等
    }

    AscendC::printf("[dispatch_teardown] blk=%d got ready=%d, write done\n",
                    AscendC::GetBlockIdx(), v);

    // ============ 收到就绪：写 done = {1,0,...}（先清 0 整块 UB 再置 slot0=1，免 done[1..7] 残留 status 旧值）============
    AscendC::Duplicate(ub, 0, STATUS_BLOCK_INT32);      // 清 0：否则上面 poll 留在 ub 的 status[1..7] 会随 32B DataCopy 一起写进 done
    AscendC::PipeBarrier<PIPE_ALL>();
    ub.SetValue(STATUS_READY_SLOT, 1);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopy(doneGt, ub, STATUS_BLOCK_INT32);  // UB -> GM
    AscendC::PipeBarrier<PIPE_ALL>();
}
