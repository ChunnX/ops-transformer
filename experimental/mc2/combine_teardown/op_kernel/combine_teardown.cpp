/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 *
 *  combine_teardown.cpp —— CombineTeardown(mxfp8 多核) AIV 核函数。
 *  对每 token i：x[i] = Σ_j expScale[i][j] · dequant( combine_data[ge*bs+pos], combine_scale[...] )
 *    ge=expert_ids[i][j]、pos=expand_idx[i][j]、行号=ge*bs+pos。
 *
 *  反量化（零额外依赖，仅 basic_api）：
 *   - fp8(e4m3fn)→fp32：Cast(CAST_NONE)。
 *   - e8m0→fp32 缩放因子：e8m0 字节 E 即偏置指数，2^(E-127) 的 fp32 位型 = (E<<23)（E∈[1,254]）。
 *     标量按组(group=32)解码 → 与 expert_scale 合并成一个标量 → 对该组 32 元素一次 Muls 直接反量化+加权。
 *   - fp32 累加，末 Cast 回 bf16。
 *  多核：按 (token, h-chunk=1024) 切，各核写 x 不相交列段，无 SyncAll。
 *  说明：避开 micro_api(VF_CALL/CastVf，本 toolkit 无 micro_api 头) 与 adv_api/BroadCast，纯 basic_api。
 */
#include "kernel_operator.h"

namespace {
constexpr uint32_t B32 = 32U;
constexpr uint32_t CHUNK = 1024U;       // 每 chunk 元素数 = 32 组 × 32
constexpr uint32_t GROUP = 32U;
__aicore__ inline uint32_t RoundUp32B(uint32_t bytes) { return ((bytes + B32 - 1U) / B32) * B32; }

// e8m0 字节 → fp32 缩放因子 2^(E-127)，用 fp32 位型构造（E∈[1,254] 正常数；0/255 为 0/NaN，测试数据回避）。
__aicore__ inline float E8m0ToFloat(uint8_t e)
{
    union { uint32_t u; float f; } cvt;
    cvt.u = static_cast<uint32_t>(e) << 23;           // 偏置指数置入 fp32 exponent 域，mantissa=0 → 2^(e-127)
    return cvt.f;
}

struct CombineTeardownTilingPod {
    uint32_t bs;
    uint32_t topk;
    uint32_t h;
    uint32_t scaleH;
    uint32_t globalExperts;
    uint32_t aivNum;
};
}  // namespace

extern "C" __global__ __aicore__ void combine_teardown(
    GM_ADDR combine_data, GM_ADDR combine_scale, GM_ADDR expert_scales, GM_ADDR expand_idx, GM_ADDR expert_ids,
    GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    __gm__ CombineTeardownTilingPod *tp = (__gm__ CombineTeardownTilingPod *)tiling;
    const uint32_t bs = tp->bs;
    const uint32_t topk = tp->topk;
    const uint32_t h = tp->h;
    const uint32_t scaleH = tp->scaleH;
    const uint32_t GE = tp->globalExperts;
    const uint32_t aivNum = tp->aivNum;
    const uint32_t nIds = bs * topk;
    const uint32_t numChunks = (h + CHUNK - 1U) / CHUNK;
    const uint32_t workItems = bs * numChunks;
    const uint32_t blk = AscendC::GetBlockIdx();
    if (blk >= aivNum) { return; }

    AscendC::GlobalTensor<fp8_e4m3fn_t> cdG; cdG.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)combine_data, GE * bs * h);
    AscendC::GlobalTensor<int8_t> csG;       csG.SetGlobalBuffer((__gm__ int8_t *)combine_scale, GE * bs * scaleH);
    AscendC::GlobalTensor<float> escG;       escG.SetGlobalBuffer((__gm__ float *)expert_scales, nIds);
    AscendC::GlobalTensor<int32_t> idxG;     idxG.SetGlobalBuffer((__gm__ int32_t *)expand_idx, nIds);
    AscendC::GlobalTensor<int32_t> idG;      idG.SetGlobalBuffer((__gm__ int32_t *)expert_ids, nIds);
    AscendC::GlobalTensor<bfloat16_t> xG;    xG.SetGlobalBuffer((__gm__ bfloat16_t *)x, bs * h);

    AscendC::TPipe pipe;

    // ---- 小数组：每核全量载入 ----
    AscendC::TBuf<AscendC::TPosition::VECCALC> escBuf, idxBuf, idBuf;
    pipe.InitBuffer(escBuf, RoundUp32B(nIds * sizeof(float)));
    pipe.InitBuffer(idxBuf, RoundUp32B(nIds * sizeof(int32_t)));
    pipe.InitBuffer(idBuf,  RoundUp32B(nIds * sizeof(int32_t)));
    AscendC::LocalTensor<float> escUb   = escBuf.Get<float>();
    AscendC::LocalTensor<int32_t> idxUb = idxBuf.Get<int32_t>();
    AscendC::LocalTensor<int32_t> idUb  = idBuf.Get<int32_t>();
    {
        AscendC::DataCopyPadExtParams<float> padF{false, 0, 0, 0.0f};
        AscendC::DataCopyPadExtParams<int32_t> padI{false, 0, 0, 0};
        AscendC::DataCopyExtParams pf{1, static_cast<uint32_t>(nIds * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyExtParams pi{1, static_cast<uint32_t>(nIds * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPad(escUb, escG, pf, padF);
        AscendC::DataCopyPad(idxUb, idxG, pi, padI);
        AscendC::DataCopyPad(idUb, idG, pi, padI);
    }
    AscendC::PipeBarrier<PIPE_ALL>();

    // ---- 计算缓冲 ----
    AscendC::TBuf<AscendC::TPosition::VECCALC> payBuf, scBuf, datBuf, accBuf, xBuf;
    pipe.InitBuffer(payBuf, RoundUp32B(CHUNK * sizeof(int8_t)));            // fp8 payload(1B/元素)
    pipe.InitBuffer(scBuf,  RoundUp32B((CHUNK / GROUP) * sizeof(int8_t)));  // e8m0(1B/元素)
    pipe.InitBuffer(datBuf, RoundUp32B(CHUNK * sizeof(float)));
    pipe.InitBuffer(accBuf, RoundUp32B(CHUNK * sizeof(float)));
    pipe.InitBuffer(xBuf,   RoundUp32B(CHUNK * sizeof(bfloat16_t)));
    AscendC::LocalTensor<fp8_e4m3fn_t> payUb = payBuf.Get<fp8_e4m3fn_t>();
    AscendC::LocalTensor<int8_t> scUb        = scBuf.Get<int8_t>();
    AscendC::LocalTensor<float> datUb        = datBuf.Get<float>();
    AscendC::LocalTensor<float> accUb        = accBuf.Get<float>();
    AscendC::LocalTensor<bfloat16_t> xUb     = xBuf.Get<bfloat16_t>();

    const uint32_t per = (workItems + aivNum - 1U) / aivNum;
    const uint32_t wBegin = blk * per;
    uint32_t wEnd = wBegin + per;
    if (wEnd > workItems) { wEnd = workItems; }

    AscendC::DataCopyPadExtParams<int8_t> pad8{false, 0, 0, 0};
    for (uint32_t w = wBegin; w < wEnd; ++w) {
        const uint32_t i = w / numChunks;             // token
        const uint32_t c = w % numChunks;             // chunk
        const uint32_t colOff = c * CHUNK;
        uint32_t len = h - colOff;
        if (len > CHUNK) { len = CHUNK; }
        const uint32_t nScale = len / GROUP;          // h 是 32 的倍数 → 整除
        const uint32_t scaleColOff = colOff / GROUP;

        AscendC::Duplicate(accUb, 0.0f, static_cast<int32_t>(len));
        AscendC::PipeBarrier<PIPE_ALL>();

        for (uint32_t j = 0; j < topk; ++j) {
            const uint32_t off = i * topk + j;
            const int32_t ge = idUb.GetValue(off);
            const int32_t pos = idxUb.GetValue(off);
            const float wscale = escUb.GetValue(off);
            const uint32_t srcRow = static_cast<uint32_t>(ge) * bs + static_cast<uint32_t>(pos);

            AscendC::DataCopy(payUb, cdG[srcRow * h + colOff], len);                 // fp8 payload chunk(len%32==0 恒对齐)
            { AscendC::DataCopyExtParams ps{1, nScale, 0, 0, 0}; AscendC::DataCopyPad(scUb, csG[srcRow * scaleH + scaleColOff], ps, pad8); }  // e8m0 scale chunk(尾块 nScale 可能<32B → Pad)
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::Cast(datUb, payUb, AscendC::RoundMode::CAST_NONE, len);         // fp8 → fp32
            AscendC::PipeBarrier<PIPE_V>();
            // 逐组反量化 + 加权（组标量 = 2^(e8m0-127) × expert_scale）
            for (uint32_t g = 0; g < nScale; ++g) {
                uint8_t e = static_cast<uint8_t>(scUb.GetValue(g));
                float combined = E8m0ToFloat(e) * wscale;
                AscendC::Muls(datUb[g * GROUP], datUb[g * GROUP], combined, static_cast<int32_t>(GROUP));
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(accUb, accUb, datUb, static_cast<int32_t>(len));            // 累加
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::Cast(xUb, accUb, AscendC::RoundMode::CAST_RINT, len);               // fp32 → bf16
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopy(xG[i * h + colOff], xUb, len);                             // 写 x 列段
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}
