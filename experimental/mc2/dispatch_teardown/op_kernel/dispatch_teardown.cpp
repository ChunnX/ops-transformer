/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 *
 *  dispatch_teardown.cpp —— DispatchTeardown(mxfp8 多核) AIV 核函数。
 *  mxfp8：每 token = fp8(e4m3fn) payload(h 字节) + e8m0 scale(h/32 字节)，两者各独立 GM、随 token 一起搬。
 *  数据区与 expandX 同为 expert-major（块序 le*R+src，每块 bs 槽带空洞）→ 本算子无算术、无转置，
 *  纯【按 wishCnt 去空洞紧密化】的字节搬运（块 t 的源就是数据区块 t），payload/scale 都按 int8 整段搬。
 *
 *  多核切分（无 SyncAll、无跨核依赖）：
 *   - 所有核：载 wishCnt → 算每块 {dstStart, cnt} 与总有效行 N(=ΣwishCnt)；
 *     把 [0,N) 行均分给 aivNum 个核，每核搬自己那段（payload+scale 行各一对 DataCopy）。各核写不相交的 expandX 行。
 *   - 仅 core0：额外产 expand_idx(出现次数) / send_counts(expert-major 前缀和) / expert_token_nums(Σsrc wishCnt)。
 */
#include "kernel_operator.h"

namespace {
constexpr uint32_t B32 = 32U;
__aicore__ inline uint32_t RoundUp32B(uint32_t bytes) { return ((bytes + B32 - 1U) / B32) * B32; }

struct DispatchTeardownTilingPod {
    uint32_t bs;
    uint32_t topk;
    uint32_t h;
    uint32_t scaleH;
    uint32_t E;
    uint32_t R;
    uint32_t aivNum;
    uint32_t A;
};
}  // namespace

extern "C" __global__ __aicore__ void dispatch_teardown(
    GM_ADDR token_data, GM_ADDR token_scale, GM_ADDR wish_cnt, GM_ADDR expert_ids,
    GM_ADDR expand_x, GM_ADDR expand_scale, GM_ADDR send_counts, GM_ADDR expand_idx, GM_ADDR expert_token_nums,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    __gm__ DispatchTeardownTilingPod *tp = (__gm__ DispatchTeardownTilingPod *)tiling;
    const uint32_t bs = tp->bs;
    const uint32_t topk = tp->topk;
    const uint32_t h = tp->h;
    const uint32_t scaleH = tp->scaleH;
    const uint32_t E = tp->E;
    const uint32_t R = tp->R;
    const uint32_t aivNum = tp->aivNum;
    const uint32_t GE = R * E;
    const uint32_t nIds = bs * topk;
    const uint32_t blk = AscendC::GetBlockIdx();
    if (blk >= aivNum) { return; }

    // ---- GlobalTensors（payload/scale 按 int8 搬；int 数组按 int32）----
    AscendC::GlobalTensor<int8_t> tdG;  tdG.SetGlobalBuffer((__gm__ int8_t *)token_data, GE * bs * h);
    AscendC::GlobalTensor<int8_t> tsG;  tsG.SetGlobalBuffer((__gm__ int8_t *)token_scale, GE * bs * scaleH);
    AscendC::GlobalTensor<int8_t> exG;  exG.SetGlobalBuffer((__gm__ int8_t *)expand_x, tp->A * h);
    AscendC::GlobalTensor<int8_t> esG;  esG.SetGlobalBuffer((__gm__ int8_t *)expand_scale, tp->A * scaleH);
    AscendC::GlobalTensor<int32_t> wishG; wishG.SetGlobalBuffer((__gm__ int32_t *)wish_cnt, GE);

    AscendC::TPipe pipe;

    // ---- 所有核：载 wishCnt，算 expert-major 每块 {srcBlk, dstStart, cnt} ----
    AscendC::TBuf<AscendC::TPosition::VECCALC> wishBuf, srcBlkBuf, dstStartBuf, cntBuf;
    pipe.InitBuffer(wishBuf,     RoundUp32B(GE * sizeof(int32_t)));
    pipe.InitBuffer(srcBlkBuf,   RoundUp32B(GE * sizeof(int32_t)));
    pipe.InitBuffer(dstStartBuf, RoundUp32B(GE * sizeof(int32_t)));
    pipe.InitBuffer(cntBuf,      RoundUp32B(GE * sizeof(int32_t)));
    AscendC::LocalTensor<int32_t> wishUb     = wishBuf.Get<int32_t>();
    AscendC::LocalTensor<int32_t> srcBlkUb   = srcBlkBuf.Get<int32_t>();
    AscendC::LocalTensor<int32_t> dstStartUb = dstStartBuf.Get<int32_t>();
    AscendC::LocalTensor<int32_t> cntUb      = cntBuf.Get<int32_t>();

    AscendC::DataCopyPadExtParams<int32_t> padI{false, 0, 0, 0};
    {
        AscendC::DataCopyExtParams p{1, static_cast<uint32_t>(GE * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPad(wishUb, wishG, p, padI);
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    int32_t run = 0;
    for (uint32_t t = 0; t < GE; ++t) {            // 数据区已是 expert-major(le*R+src)，块 t 源即块 t
        int32_t c = wishUb.GetValue(t);
        srcBlkUb.SetValue(t, static_cast<int32_t>(t));
        dstStartUb.SetValue(t, run);
        cntUb.SetValue(t, c);
        run += c;
    }
    const uint32_t N = static_cast<uint32_t>(run);  // 总有效行

    // ---- 本核负责的行区间 [rowBegin, rowEnd) ----
    const uint32_t per = (N + aivNum - 1U) / aivNum;
    const uint32_t rowBegin = blk * per;
    uint32_t rowEnd = rowBegin + per;
    if (rowEnd > N) { rowEnd = N; }

    if (rowBegin < rowEnd) {
        // 单缓冲 + per-row PIPE_ALL（已实测稳定）。多核已把 N 行均分到各核，rows≤cores 时每核仅 ~1 行、
        // 核内无相邻行可重叠，per-core 双缓冲在本规模无收益（实测 706 vs 814us 边际、区间重叠）；
        // 且 TQue VECIN 直拷(GM→UB→GM)跨行 WAR 事件追踪不完整 → 间歇覆写。故保留稳定单缓冲。
        // 若未来 rows≫cores 需提吞吐：用显式 MTE2/MTE3 SetFlag/WaitFlag 软流水双缓冲，并多次跑复验。
        AscendC::TBuf<AscendC::TPosition::VECCALC> payBuf, scBuf;
        pipe.InitBuffer(payBuf, RoundUp32B(h * sizeof(int8_t)));
        pipe.InitBuffer(scBuf,  RoundUp32B(scaleH * sizeof(int8_t)));
        AscendC::LocalTensor<int8_t> payUb = payBuf.Get<int8_t>();
        AscendC::LocalTensor<int8_t> scUb  = scBuf.Get<int8_t>();

        AscendC::DataCopyPadExtParams<int8_t> pad8{false, 0, 0, 0};
        for (uint32_t g = rowBegin; g < rowEnd; ++g) {
            // 定位 g 属于哪个 expert-major 块
            uint32_t t = 0;
            for (; t < GE; ++t) {
                uint32_t ds = static_cast<uint32_t>(dstStartUb.GetValue(t));
                uint32_t c = static_cast<uint32_t>(cntUb.GetValue(t));
                if (g >= ds && g < ds + c) { break; }
            }
            if (t >= GE) { continue; }   // 防御：wishCnt 与 N 不自洽时不越界读块表
            uint32_t r = g - static_cast<uint32_t>(dstStartUb.GetValue(t));
            uint32_t sBlk = static_cast<uint32_t>(srcBlkUb.GetValue(t));
            uint32_t srcRow = sBlk * bs + r;

            // payload 行（h 字节，h%32==0 恒 32B 对齐 → 裸 DataCopy）
            AscendC::DataCopy(payUb, tdG[srcRow * h], h);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopy(exG[g * h], payUb, h);
            AscendC::PipeBarrier<PIPE_ALL>();
            // scale 行（scaleH=h/32 字节，可能非 32B 倍数 → DataCopyPad 防截断）
            { AscendC::DataCopyExtParams p{1, scaleH, 0, 0, 0}; AscendC::DataCopyPad(scUb, tsG[srcRow * scaleH], p, pad8); }
            AscendC::PipeBarrier<PIPE_ALL>();
            { AscendC::DataCopyExtParams p{1, scaleH, 0, 0, 0}; AscendC::DataCopyPad(esG[g * scaleH], scUb, p); }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    // ============ 仅 core0：三条 int 链路 ============
    if (blk != 0) { return; }

    AscendC::GlobalTensor<int32_t> eidG;  eidG.SetGlobalBuffer((__gm__ int32_t *)expert_ids, nIds);
    AscendC::GlobalTensor<int32_t> eidxG; eidxG.SetGlobalBuffer((__gm__ int32_t *)expand_idx, nIds);
    AscendC::GlobalTensor<int32_t> scsG;  scsG.SetGlobalBuffer((__gm__ int32_t *)send_counts, GE);
    AscendC::GlobalTensor<int32_t> etnG;  etnG.SetGlobalBuffer((__gm__ int32_t *)expert_token_nums, E);

    AscendC::TBuf<AscendC::TPosition::VECCALC> idsBuf, idxBuf, occBuf, scOutBuf, etnBuf;
    pipe.InitBuffer(idsBuf,   RoundUp32B(nIds * sizeof(int32_t)));
    pipe.InitBuffer(idxBuf,   RoundUp32B(nIds * sizeof(int32_t)));
    pipe.InitBuffer(occBuf,   RoundUp32B(GE * sizeof(int32_t)));
    pipe.InitBuffer(scOutBuf, RoundUp32B(GE * sizeof(int32_t)));
    pipe.InitBuffer(etnBuf,   RoundUp32B(E * sizeof(int32_t)));
    AscendC::LocalTensor<int32_t> idsUb   = idsBuf.Get<int32_t>();
    AscendC::LocalTensor<int32_t> idxUb   = idxBuf.Get<int32_t>();
    AscendC::LocalTensor<int32_t> occUb   = occBuf.Get<int32_t>();
    AscendC::LocalTensor<int32_t> scOutUb = scOutBuf.Get<int32_t>();
    AscendC::LocalTensor<int32_t> etnUb   = etnBuf.Get<int32_t>();

    {
        AscendC::DataCopyExtParams p{1, static_cast<uint32_t>(nIds * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPad(idsUb, eidG, p, padI);
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    // 链路 B：expand_idx = expert_ids 行主序出现次数
    AscendC::Duplicate(occUb, 0, static_cast<int32_t>(RoundUp32B(GE * sizeof(int32_t)) / sizeof(int32_t)));
    AscendC::PipeBarrier<PIPE_ALL>();
    for (uint32_t i = 0; i < nIds; ++i) {
        int32_t ge = idsUb.GetValue(i);
        int32_t occ = occUb.GetValue(ge);
        idxUb.SetValue(i, occ);
        occUb.SetValue(ge, occ + 1);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    {
        AscendC::DataCopyExtParams p{1, static_cast<uint32_t>(nIds * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPad(eidxG, idxUb, p);
    }

    // 链路 A1：expert_token_nums[le] = Σ_src wishCnt[le*R+src]（wishCnt 为 expert-major）
    for (uint32_t le = 0; le < E; ++le) {
        int32_t s = 0;
        for (uint32_t src = 0; src < R; ++src) { s += wishUb.GetValue(le * R + src); }
        etnUb.SetValue(le, s);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    {
        AscendC::DataCopyExtParams p{1, static_cast<uint32_t>(E * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPad(etnG, etnUb, p);
    }

    // 链路 A2：send_counts = expert-major inclusive 前缀和（= dstStart+cnt 累计）
    {
        int32_t acc = 0;
        for (uint32_t t = 0; t < GE; ++t) {
            acc += cntUb.GetValue(t);
            scOutUb.SetValue(t, acc);
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    {
        AscendC::DataCopyExtParams p{1, static_cast<uint32_t>(GE * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPad(scsG, scOutUb, p);
    }
}
