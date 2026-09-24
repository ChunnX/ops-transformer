/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file kernel_kv_cache_turbo_quant_cube.h
 * \brief KvCacheTurboQuant encode kernel：AIC/AIV 混合 Cube 路径（bf16x3 拆分 GEMM）。
 *
 * 数据流（每 aiCore，tile M=32 向量，双缓冲 d=t%2）：
 *   AIV(subIdx0)：kv→norm→u（v3 bit 精确路径）→ 拆 u_hi/u_lo(bf16) → GM ubuf
 *                 → CrossCoreSetFlag(U_READY+d)
 *   AIC：WaitFlag(U_READY+d) → Nd2Nz u_hi/u_lo → L0A →
 *        y = u_hi@H_hi^T + u_hi@H_lo^T + u_lo@H_hi^T（3 次 Mmad 链式累加，L0C fp32）
 *        → Fixpipe float→float ND → GM ybuf → SetFlag(Y_READY+d)
 *   AIV：读 y → 量化 idx/yhat/r → gamma → 拆 r_hi/r_lo → GM rbuf → SetFlag(R_READY+d)
 *   AIC：同构 GEMM2：proj = r_hi@S_hi^T + r_hi@S_lo^T + r_lo@S_hi^T → SetFlag(P_READY+d)
 *   AIV：读 proj → qjl bit → 输出
 *
 * 精度（真机实测，对齐任务方 golden.py 的 host 等价实现）：norm 逐位一致、gamma ≤8/16384 行差 1 ULP、
 *       idx ≤25/786432 byte、qjl ≤194/262144 byte；y 与 fp32 参考的最大偏差 2.3e-6（y 量级 0.1）。
 *       （试过再补 lo@lo 的 bf16x4：idx 降到 ≤16、y 偏差 1.7e-6，但 prefill/mse2 的 10× 余量
 *        从 4.3%/2.4% 掉到 3.0%/1.1%，故保留 bf16x3。）
 * 权重 H/S 在 Init 由 AIV0 一次性 cast bf16 + hi/lo 拆分 → GM（hbuf/sbuf），AIC Nd2Nz 进 L1 常驻。
 *
 * 真机已证实（probe mode 50）：Nd2Nz 随路 / LoadData2D(compressor 式) / Mmad bf16+fp32 累加 /
 * Fixpipe NZ2ND float 直出 / MIX 内核三铁律（matmul_intf.h、workspace>0、ASCEND_IS_AIV 守卫）。
 */
#ifndef KERNEL_KV_CACHE_TURBO_QUANT_CUBE_H
#define KERNEL_KV_CACHE_TURBO_QUANT_CUBE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kv_cache_turbo_quant_common.h" // HEAD_DIM / CentTable（同一 namespace）
#include "kv_cache_turbo_quant_tiling_data.h"

using namespace AscendC;

namespace KvCacheTurboQuant {

// 调试打印开关：注意只要编译单元里出现 AscendC::printf，CANN 就会为每次下发
// 分配并初始化 dump-tensor workspace（每核 1MB 块，实测 ~14ms/launch），默认必须关掉。
#ifndef KV_TQ_ENABLE_PRINT
#define KVPRINT(...) ((void)0)
#else
#define KVPRINT(...) AscendC::printf(__VA_ARGS__)
#endif

constexpr uint64_t CUBE_SYNC_MODE2 = 2; // 跨核（0x2 = AIV 双 sub-block 联合 notify）
// flag ID 取 4~10：0~2 可能与 codegen wrapper 内部同步冲突，11~15 为框架保留（compressor 同款范围）
constexpr uint32_t CUBE_FLAG_HS = 4; // AIV0→AIC：H/S bf16 拆分完成
constexpr uint32_t CUBE_FLAG_U = 5;  // AIV→AIC：u_hi/u_lo 就绪（dbIdx 0/1）
constexpr uint32_t CUBE_FLAG_Y = 6;  // AIC→AIV：y 就绪
constexpr uint32_t CUBE_FLAG_R = 7;  // AIV→AIC：r_hi/r_lo 就绪
constexpr uint32_t CUBE_FLAG_P = 8;  // AIC→AIV：proj 就绪

constexpr int32_t CUBE_TILE_V = 64; // Cube tile 向量数（128 实测无收益且拖慢小 case；AIC 按 CUBE_MMAD_M=64 分轮）
constexpr int32_t CUBE_MMAD_M = 64;   // 单次 Mmad 的 m（A2 上限 64）
constexpr int32_t CUBE_SUBBATCH = 32; // AIV 子批（再腾 32KB UB 后重试 32：每子批屏障数减半）

template <uint8_t BITS>
class KernelKvCacheTurboQuantCube {
public:
    static constexpr int32_t CENT_CNT = 1 << BITS;
    static constexpr int32_t BOUND_CNT = CENT_CNT - 1;
    static constexpr int32_t IDX_BYTES = HEAD_DIM * BITS / 8;
    static constexpr int32_t SUB_ELEMS = CUBE_SUBBATCH * HEAD_DIM; // CUBE_SUBBATCH=32 时 4096
    static constexpr uint32_t WS_HEAD = 64 * 1024 * 1024; // 避开框架在 workspace 头部 ~60MB 的清零区（实测 effective）

    __aicore__ inline KernelKvCacheTurboQuantCube() {}

    __aicore__ inline void Init(GM_ADDR kvVectors, GM_ADDR rotationMatrix, GM_ADDR qjlMatrix, GM_ADDR quantIdx,
                                GM_ADDR quantQjl, GM_ADDR quantNorm, GM_ADDR quantGamma, GM_ADDR workspace,
                                const KvCacheTurboQuantTilingData *td, TPipe *pipe)
    {
        // MIX 核映射：AIC blockIdx ∈ [0,N)，AIV blockIdx ∈ [0,2N)
        if ASCEND_IS_AIV {
            aiCoreIdx_ = GetBlockIdx() / 2;
            aivSubIdx_ = GetBlockIdx() % 2;
        } else {
            aiCoreIdx_ = GetBlockIdx();
            aivSubIdx_ = 0;
        }
        numVectors_ = td->numVectors;
        uint32_t former = td->formerCoreNum;
        vecStart_ = static_cast<int64_t>(aiCoreIdx_) * td->vecsPerCore +
                    (aiCoreIdx_ < former ? static_cast<int64_t>(aiCoreIdx_) : static_cast<int64_t>(former));
        vecCount_ = td->vecsPerCore + (aiCoreIdx_ < former ? 1 : 0);
        tiles_ = static_cast<uint32_t>((vecCount_ + CUBE_TILE_V - 1) / CUBE_TILE_V);

        kvGm_.SetGlobalBuffer((__gm__ bfloat16_t *)kvVectors);
        idxGm_.SetGlobalBuffer((__gm__ uint8_t *)quantIdx);
        qjlOutGm_.SetGlobalBuffer((__gm__ uint8_t *)quantQjl);
        normGm_.SetGlobalBuffer((__gm__ bfloat16_t *)quantNorm);
        gamGm_.SetGlobalBuffer((__gm__ bfloat16_t *)quantGamma);
        normGmF16_.SetGlobalBuffer((__gm__ half *)quantNorm);
        gamGmF16_.SetGlobalBuffer((__gm__ half *)quantGamma);
        matDtype_ = td->matDtype;
        outDtype_ = td->outDtype;

        // workspace 布局（host 侧同公式定总量）
        // MIX 场景下 raw workspace 指针不等于用户区起点（框架在前后各留系统区），
        // AIC/AIV 视图还可能不同 —— 必须统一走 GetUserWorkspace 取用户区基址。
        GM_ADDR wsBase = workspace; // 框架给的 raw 指针即数据落点；GetUserWorkspace 会再外推（实测出界）
        if ASCEND_IS_AIC {
            if (aiCoreIdx_ == 0) {
                uint64_t r = (uint64_t)workspace;
                uint64_t u = (uint64_t)AscendC::GetUserWorkspace(workspace);
                KVPRINT("[WSPTR] AIC raw=%d %d user=%d %d\n", (int)(uint32_t)r, (int)(uint32_t)(r >> 32),
                        (int)(uint32_t)u, (int)(uint32_t)(u >> 32));
            }
        } else {
            if (aiCoreIdx_ == 0 && aivSubIdx_ == 0) {
                uint64_t r = (uint64_t)workspace;
                uint64_t u = (uint64_t)AscendC::GetUserWorkspace(workspace);
                KVPRINT("[WSPTR] AIV raw=%d %d user=%d %d\n", (int)(uint32_t)r, (int)(uint32_t)(r >> 32),
                        (int)(uint32_t)u, (int)(uint32_t)(u >> 32));
            }
        }
        uint64_t off = WS_HEAD;
        offHHi_ = off;
        off += HEAD_DIM * HEAD_DIM * 2;
        offHLo_ = off;
        off += HEAD_DIM * HEAD_DIM * 2;
        offSHi_ = off;
        off += HEAD_DIM * HEAD_DIM * 2;
        offSLo_ = off;
        off += HEAD_DIM * HEAD_DIM * 2;
        offUbHi_ = off;
        off += numVectors_ * HEAD_DIM * 2;
        offUbLo_ = off;
        off += numVectors_ * HEAD_DIM * 2;
        offRHi_ = off;
        off += numVectors_ * HEAD_DIM * 2;
        offRLo_ = off;
        off += numVectors_ * HEAD_DIM * 2;
        offY_ = off;
        off += numVectors_ * HEAD_DIM * 4;
        offP_ = off;
        off += numVectors_ * HEAD_DIM * 4;
        wsTotal_ = off;

        hHiGm_.SetGlobalBuffer((__gm__ bfloat16_t *)(wsBase + offHHi_));
        hLoGm_.SetGlobalBuffer((__gm__ bfloat16_t *)(wsBase + offHLo_));
        sHiGm_.SetGlobalBuffer((__gm__ bfloat16_t *)(wsBase + offSHi_));
        sLoGm_.SetGlobalBuffer((__gm__ bfloat16_t *)(wsBase + offSLo_));
        ubHiGm_.SetGlobalBuffer((__gm__ bfloat16_t *)(wsBase + offUbHi_));
        ubLoGm_.SetGlobalBuffer((__gm__ bfloat16_t *)(wsBase + offUbLo_));
        rHiGm_.SetGlobalBuffer((__gm__ bfloat16_t *)(wsBase + offRHi_));
        rLoGm_.SetGlobalBuffer((__gm__ bfloat16_t *)(wsBase + offRLo_));
        yGm_.SetGlobalBuffer((__gm__ float *)(wsBase + offY_));
        pGm_.SetGlobalBuffer((__gm__ float *)(wsBase + offP_));
        rotSrcGm_.SetGlobalBuffer((__gm__ float *)rotationMatrix);
        qjlSrcGm_.SetGlobalBuffer((__gm__ float *)qjlMatrix);
        rotSrcBf16Gm_.SetGlobalBuffer((__gm__ bfloat16_t *)rotationMatrix);
        qjlSrcBf16Gm_.SetGlobalBuffer((__gm__ bfloat16_t *)qjlMatrix);

        if ASCEND_IS_AIC {
            InitAic(pipe);
        } else {
            InitAiv(pipe);
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            ProcessAic();
        } else {
            ProcessAiv();
        }
    }

private:
    // 归约树（仅到 off=8，保持 32B 对齐）：t[0..8) = 8 个部分和（v3 同款，精度链 bit 精确）
    __aicore__ inline void TreeReduce8(LocalTensor<float> t)
    {
        Add(t, t, t[64], 64);
        Add(t, t, t[32], 32);
        Add(t, t, t[16], 16);
        Add(t, t, t[8], 8);
    }

    __aicore__ inline float SumPartials(LocalTensor<float> t)
    {
        float s = t.GetValue(0);
        for (int32_t i = 1; i < 8; i++) {
            s += t.GetValue(i);
        }
        return s;
    }

    // ---------------- 公共小工具 ----------------
    __aicore__ inline void SyncMte2V()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(e);
        WaitFlag<HardEvent::MTE2_V>(e);
    }
    __aicore__ inline void SyncVS()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(e);
        WaitFlag<HardEvent::V_S>(e);
    }
    __aicore__ inline void SyncVMte3()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(e);
        WaitFlag<HardEvent::V_MTE3>(e);
    }
    __aicore__ inline void SyncMte3V()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(e);
        WaitFlag<HardEvent::MTE3_V>(e);
    }
    __aicore__ inline void SyncSMte3()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(e);
        WaitFlag<HardEvent::S_MTE3>(e);
    }
    __aicore__ inline void SyncSV()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(e);
        WaitFlag<HardEvent::S_V>(e);
    }
    __aicore__ inline void SyncMte2Mte1()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
        SetFlag<HardEvent::MTE2_MTE1>(e);
        WaitFlag<HardEvent::MTE2_MTE1>(e);
    }
    __aicore__ inline void SyncMte1M()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_M));
        SetFlag<HardEvent::MTE1_M>(e);
        WaitFlag<HardEvent::MTE1_M>(e);
    }
    __aicore__ inline void SyncMMte1()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_MTE1));
        SetFlag<HardEvent::M_MTE1>(e);
        WaitFlag<HardEvent::M_MTE1>(e);
    }
    __aicore__ inline void SyncMFix()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_FIX));
        SetFlag<HardEvent::M_FIX>(e);
        WaitFlag<HardEvent::M_FIX>(e);
    }
    __aicore__ inline void SyncFixM()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::FIX_M));
        SetFlag<HardEvent::FIX_M>(e);
        WaitFlag<HardEvent::FIX_M>(e);
    }
    // 跨核 wait 必须挂在“真正消费该数据的流水”上：AIC/AIV 两侧消费 GM 数据都走 MTE2
    // （AIC: Nd2NzTile/Nd2NzWeights 搬 u/H；AIV: QuantSubbatch 搬 y、P 阶段搬 proj）。
    // 挂 PIPE_FIX 时 AIV/AIC 的 FIX 流水上并没有后续指令，等待形同虚设，
    // 会把 AIC 还没写完的 y/proj 读进来 —— 表现为部分 sub-batch 结果错、跨轮次不确定。
    __aicore__ inline void CrossWait(uint32_t flag, bool /*consumerPipeIsMte2*/)
    {
        CrossCoreWaitFlag<CUBE_SYNC_MODE2, PIPE_MTE2>(flag);
    }
    __aicore__ inline void CrossSet(uint32_t flag, bool fromFix)
    {
        if (fromFix) {
            CrossCoreSetFlag<CUBE_SYNC_MODE2, PIPE_FIX>(flag);
        } else {
            CrossCoreSetFlag<CUBE_SYNC_MODE2, PIPE_MTE3>(flag);
        }
    }

    // fp32 → bf16 hi/lo 拆分：hi=RNE(x)，lo=RNE(x-hi)（与 verify_cube_sim bf16x3 一致）
    __aicore__ inline void SplitHiLo(const LocalTensor<float> &srcFp, const LocalTensor<bfloat16_t> &dstHi,
                                     const LocalTensor<bfloat16_t> &dstLo, const LocalTensor<float> &tmpFp,
                                     uint32_t elems)
    {
        Cast(dstHi, srcFp, RoundMode::CAST_RINT, elems);
        Cast(tmpFp, dstHi, RoundMode::CAST_NONE, elems);
        Sub(tmpFp, srcFp, tmpFp, elems);
        Cast(dstLo, tmpFp, RoundMode::CAST_RINT, elems);
    }

    // ---------------- AIV 侧 ----------------
    __aicore__ inline void InitAiv(TPipe *pipe)
    {
        pipe_ = pipe;
        // 常驻小缓冲（子批 8 向量 = 4KB fp32）
        // xfBuf_/uBuf_ 只作桩：prep 的 xf 别名 yT_、u 别名 qjlBT_（prep 相位与 quant/P 相位不重叠）
        pipe->InitBuffer(xfBuf_, 32);
        pipe->InitBuffer(uBuf_, 32);
        pipe->InitBuffer(sqBuf_, SUB_ELEMS * sizeof(float));       // 4KB
        pipe->InitBuffer(uHiBuf_, SUB_ELEMS * sizeof(bfloat16_t)); // 2KB
        pipe->InitBuffer(uLoBuf_, SUB_ELEMS * sizeof(bfloat16_t));
        // 下面三个缓冲改为别名共用（腾出 UB 以允许更大的 SUB_ELEMS）：
        //   xinBuf_ → 输入别名 uHiT_（Cast 之后输入即死）；projBuf_ → 别名 yT_（proj 在 quant 之后才读）；
        //   tmpFpBuf_ → 别名 yhatT_（SplitHiLo 在 r=y-yhat 之后）。只留 32B 桩以便 Get() 合法。
        pipe->InitBuffer(tmpFpBuf_, 32);
        pipe->InitBuffer(xinBuf_, 32);
        // 量化链（v3 形态）
        pipe->InitBuffer(yBuf_, SUB_ELEMS * sizeof(float));
        pipe->InitBuffer(yhatBuf_, SUB_ELEMS * sizeof(float));
        pipe->InitBuffer(rBuf_, SUB_ELEMS * sizeof(float));
        pipe->InitBuffer(projBuf_, 32); // 别名 yT_（proj 读在 quant 之后）
        pipe->InitBuffer(idxFBuf_, SUB_ELEMS * sizeof(float));
        pipe->InitBuffer(qjlBBuf_, SUB_ELEMS * sizeof(float));
        pipe->InitBuffer(rHiBuf_, SUB_ELEMS * sizeof(bfloat16_t));
        pipe->InitBuffer(rLoBuf_, SUB_ELEMS * sizeof(bfloat16_t));
        pipe->InitBuffer(dupBuf_, SUB_ELEMS * sizeof(float)); // 子批宽度(1024元素)广播用
        pipe->InitBuffer(dupZeroBuf_, HEAD_DIM * sizeof(float));
        pipe->InitBuffer(maskBuf_, SUB_ELEMS);
        pipe->InitBuffer(normArrBuf_, CUBE_TILE_V * sizeof(float));
        pipe->InitBuffer(gamArrBuf_, CUBE_SUBBATCH * sizeof(float));
        pipe->InitBuffer(outIdxBuf_, CUBE_TILE_V * IDX_BYTES);
        pipe->InitBuffer(outQjlBuf_, CUBE_TILE_V * QJL_BYTES);
        pipe->InitBuffer(outNormBf16Buf_, CUBE_TILE_V * sizeof(bfloat16_t));
        pipe->InitBuffer(outGamBf16Buf_, CUBE_SUBBATCH * sizeof(bfloat16_t));
        // H/S staging 分块缓冲（16 行 = 8KB fp32）
        pipe->InitBuffer(stFpBuf_, 16 * HEAD_DIM * sizeof(float));
        pipe->InitBuffer(stBkBuf_, 32); // 别名 yhatT_（staging 的 hi/lo 拆分临时）
        pipe->InitBuffer(stHiBuf_, 16 * HEAD_DIM * sizeof(bfloat16_t)); // 16 行块（别名已腾出足够 UB）
        pipe->InitBuffer(stLoBuf_, 16 * HEAD_DIM * sizeof(bfloat16_t));

        xfT_ = xfBuf_.Get<float>();
        uT_ = uBuf_.Get<float>();
        sqT_ = sqBuf_.Get<float>();
        uHiT_ = uHiBuf_.Get<bfloat16_t>();
        uLoT_ = uLoBuf_.Get<bfloat16_t>();
        tmpFpT_ = tmpFpBuf_.Get<float>();
        xinT_ = xinBuf_.Get<bfloat16_t>();
        yT_ = yBuf_.Get<float>();
        yhatT_ = yhatBuf_.Get<float>();
        rT_ = rBuf_.Get<float>();
        projT_ = projBuf_.Get<float>();
        idxFT_ = idxFBuf_.Get<float>();
        qjlBT_ = qjlBBuf_.Get<float>();
        rHiT_ = rHiBuf_.Get<bfloat16_t>();
        rLoT_ = rLoBuf_.Get<bfloat16_t>();
        dupT_ = dupBuf_.Get<float>();
        dupZeroT_ = dupZeroBuf_.Get<float>();
        maskT_ = maskBuf_.Get<uint8_t>();
        normArrT_ = normArrBuf_.Get<float>();
        gamArrT_ = gamArrBuf_.Get<float>();
        outIdxT_ = outIdxBuf_.Get<uint8_t>();
        outQjlT_ = outQjlBuf_.Get<uint8_t>();
        outNormBf16T_ = outNormBf16Buf_.Get<bfloat16_t>();
        outGamBf16T_ = outGamBf16Buf_.Get<bfloat16_t>();
        stFpT_ = stFpBuf_.Get<float>();
        stBkT_ = stBkBuf_.Get<float>();
        stHiT_ = stHiBuf_.Get<bfloat16_t>();
        stLoT_ = stLoBuf_.Get<bfloat16_t>();

        // 别名：prep 的 xf 用 yT_、u 用 qjlBT_（prep 相位与 quant/P 相位不重叠）——省 32KB UB 以支持更大的 SUB_ELEMS
        xfT_ = yT_;
        uT_ = qjlBT_;
        Duplicate(dupZeroT_, 0.0f, HEAD_DIM);
    }

    // H/S fp32 → bf16 hi/lo 拆分，16 行一块搬 GM
    __aicore__ inline void StageMatrix(const GlobalTensor<float> &srcGm, const GlobalTensor<bfloat16_t> &hiGm,
                                       const GlobalTensor<bfloat16_t> &loGm)
    {
        if (aiCoreIdx_ == 0 && aivSubIdx_ == 0) {
            KVPRINT("[STG] begin hd=%d\n", (int)HEAD_DIM);
        }
        for (int32_t chunk = 0; chunk < HEAD_DIM / 16; chunk++) {
            if (aiCoreIdx_ == 0 && aivSubIdx_ == 0) {
                KVPRINT("[STG] c=%d off=%d\n", (int)chunk, (int)(chunk * 16 * HEAD_DIM));
            }
            // 关键：上一轮 V(SplitHiLo) 读完 stFp 之前，MTE2 不能覆写它
            {
                event_t e0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(e0);
                WaitFlag<HardEvent::V_MTE2>(e0);
            }
            DataCopy(stFpT_, srcGm[chunk * 16 * HEAD_DIM], 16 * HEAD_DIM);
            SyncMte2V();
            SplitHiLo(stFpT_, stHiT_, stLoT_, yhatT_, 16 * HEAD_DIM); // yhatT_ 在 staging 阶段空闲
            SyncVMte3();
            DataCopy(hiGm[chunk * 16 * HEAD_DIM], stHiT_, 16 * HEAD_DIM);
            DataCopy(loGm[chunk * 16 * HEAD_DIM], stLoT_, 16 * HEAD_DIM);
            SyncMte3V(); // 等 MTE3 读完 UB 再进入下一 chunk 覆写
        }
    }

    // H/S bf16 → hi 原样、lo 补零：输入本身就是 bf16 精度（hi=x 精确，x-hi=0），
    // 后续 GEMM 的 lo 项贡献为零，与"用户给 bf16 矩阵"的参考语义一致。
    __aicore__ inline void StageMatrixBf16(const GlobalTensor<bfloat16_t> &srcGm, const GlobalTensor<bfloat16_t> &hiGm,
                                           const GlobalTensor<bfloat16_t> &loGm)
    {
        for (int32_t chunk = 0; chunk < HEAD_DIM / 16; chunk++) {
            {
                event_t e0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(e0);
                WaitFlag<HardEvent::V_MTE2>(e0);
            }
            DataCopy(stHiT_, srcGm[chunk * 16 * HEAD_DIM], 16 * HEAD_DIM);
            SyncMte2V();
            Duplicate(stLoT_, bfloat16_t(0.0f), 16 * HEAD_DIM);
            SyncVMte3();
            DataCopy(hiGm[chunk * 16 * HEAD_DIM], stHiT_, 16 * HEAD_DIM);
            DataCopy(loGm[chunk * 16 * HEAD_DIM], stLoT_, 16 * HEAD_DIM);
            SyncMte3V(); // 等 MTE3 读完 UB 再进入下一 chunk 覆写
        }
    }

    // 子批 prep：kv → norm/u（v3 bit 精确路径）→ bf16 hi/lo → GM
    __aicore__ inline void PrepSubbatch(int64_t vecBase, int32_t n, int32_t s)
    {
        uint32_t elems = static_cast<uint32_t>(n) * HEAD_DIM;
        // 关键：上一次调用对本 UB 的 V 读必须先完成，MTE2 才能覆写
        {
            event_t e1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(e1);
            WaitFlag<HardEvent::V_MTE2>(e1);
        }
        DataCopy(uHiT_, kvGm_[vecBase * HEAD_DIM], elems); // 别名 uHiT_（Cast 之后输入即死）
        SyncMte2V();
        Cast(xfT_, uHiT_, RoundMode::CAST_NONE, elems);
        Mul(sqT_, xfT_, xfT_, elems);
        // 行内 128 元素 = 16 个 32B block：两级 BlockReduceSum（stage1 → 每行 16 个 block 和，
        // stage2 → 每行 2 个半行和），每行只需 2 次标量读 + 1 次加（原 SumPartials 是 8 读 + 7 加）。
        // 归约顺序变化 → norm 末位可能与 golden 不同（任务书精度判据为容差，不要求逐位）。
        // stage2 的 repeatTime 必须向上取整：子批行数 n 不是 4 的倍数时（尾子批 19 行等）
        // elems/8/64 会截断，尾部若干行的和不会被写出（残留上一轮的值 → norm/y 全错）。
        BlockReduceSum(dupT_, sqT_, static_cast<int32_t>(elems / 64), 64, 1, 1, 8);
        BlockReduceSum(dupT_[SUB_ELEMS / 2], dupT_, static_cast<int32_t>((elems / 8 + 63) / 64), 64, 1, 1, 8);
        SyncVS();
        for (int32_t v = 0; v < n; v++) {
            float sumsq = dupT_.GetValue(SUB_ELEMS / 2 + v * 2) + dupT_.GetValue(SUB_ELEMS / 2 + v * 2 + 1);
            float norm = sqrt(sumsq);
            float invNorm = 1.0f / (norm > 1e-30f ? norm : 1e-30f);
            normArrT_.SetValue(s * CUBE_SUBBATCH + v, norm);
            Muls(uT_[v * HEAD_DIM], xfT_[v * HEAD_DIM], invNorm, HEAD_DIM);
        }
        SyncSV();                                    // S 读 dupT_ 完成后，下一轮 prep 的 V 写才能覆盖它
        SplitHiLo(uT_, uHiT_, uLoT_, yhatT_, elems); // yhatT_ 在本阶段空闲，作 hi/lo 拆分临时
        SyncVMte3();
        DataCopy(ubHiGm_[vecBase * HEAD_DIM], uHiT_, elems);
        DataCopy(ubLoGm_[vecBase * HEAD_DIM], uLoT_, elems);
        SyncMte3V();
    }

    // 子批量化：读 y → idx/yhat/r/gamma → r hi/lo → GM；返回后由调用者处理 proj
    __aicore__ inline void QuantSubbatch(int64_t vecBase, int32_t n, int32_t s)
    {
        uint32_t elems = static_cast<uint32_t>(n) * HEAD_DIM;
        // 关键：上一次调用对本 UB 的 V 读完成前，MTE2 不能覆写
        {
            event_t e2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(e2);
            WaitFlag<HardEvent::V_MTE2>(e2);
        }
        DataCopy(yT_, yGm_[vecBase * HEAD_DIM], elems);
        SyncMte2V();
        // 子批宽度（elems = n×128 元素）一次算完：边界/质心只广播一次，指令数 ÷ n
        Duplicate(idxFT_, 0.0f, elems);
        Duplicate(yhatT_, CentTable<BITS>::CENT[0], elems);
        for (int32_t k = 0; k < BOUND_CNT; k++) {
            float b = (CentTable<BITS>::CENT[k] + CentTable<BITS>::CENT[k + 1]) * 0.5f;
            CompareScalar(maskT_, yT_, b, CMPMODE::GT, elems); // 边界用标量比较，省掉广播
            Duplicate(dupT_, static_cast<float>(k + 1), elems);
            Select(idxFT_, maskT_, dupT_, idxFT_, SELMODE::VSEL_TENSOR_TENSOR_MODE, elems);
            // yhat 复用同一个 mask：yhat = cent0 + Σ_k Δ_k·[y > b_k]（Δ_k = cent[k+1]-cent[k]）
            Duplicate(dupT_, CentTable<BITS>::CENT[k + 1], elems);
            Select(yhatT_, maskT_, dupT_, yhatT_, SELMODE::VSEL_TENSOR_TENSOR_MODE, elems);
        }
        Sub(rT_, yT_, yhatT_, elems);
        // gamma = norm * ||r||（与 norm 同款两级 BlockReduceSum，stage2 repeatTime 同样要向上取整）
        Mul(sqT_, rT_, rT_, elems);
        BlockReduceSum(dupT_, sqT_, static_cast<int32_t>(elems / 64), 64, 1, 1, 8);
        BlockReduceSum(dupT_[SUB_ELEMS / 2], dupT_, static_cast<int32_t>((elems / 8 + 63) / 64), 64, 1, 1, 8);
        SyncVS();
        for (int32_t v = 0; v < n; v++) {
            float rNorm = sqrt(dupT_.GetValue(SUB_ELEMS / 2 + v * 2) + dupT_.GetValue(SUB_ELEMS / 2 + v * 2 + 1));
            gamArrT_.SetValue(v, normArrT_.GetValue(s * CUBE_SUBBATCH + v) * rNorm);
        }
        SyncSV(); // S 读 dupT_ 完成后，后续 V 写（打包等）才能覆盖它
        SplitHiLo(rT_, rHiT_, rLoT_, yhatT_, elems); // yhat 在 Sub(r=y-yhat) 之后已死
        SyncVMte3();
        DataCopy(rHiGm_[vecBase * HEAD_DIM], rHiT_, elems);
        DataCopy(rLoGm_[vecBase * HEAD_DIM], rLoT_, elems);
        SyncMte3V();
    }

    // 子批 idx/norm/gamma 输出（本 AIV 自己的 outIdx 缓冲，v3 形态）
    __aicore__ inline void FlushIdxNormGamma(int64_t vecBase, int32_t n, int32_t s)
    {
        SyncSMte3();
        DataCopyExtParams idxParams;
        idxParams.blockCount = 1;
        idxParams.blockLen = static_cast<uint32_t>(n) * IDX_BYTES;
        idxParams.srcStride = 0;
        idxParams.dstStride = 0;
        DataCopyPad(idxGm_[vecBase * IDX_BYTES], outIdxT_[(s * CUBE_SUBBATCH) * IDX_BYTES], idxParams);
        SyncSV();
        DataCopyExtParams ngParams;
        ngParams.blockCount = 1;
        ngParams.blockLen = static_cast<uint32_t>(n) * 2;
        ngParams.srcStride = 0;
        ngParams.dstStride = 0;
        if (outDtype_ == 1) { // float16 输出：复用 bf16 暂存缓冲（同 2B/元素）
            auto outNormF16T_ = outNormBf16Buf_.Get<half>();
            auto outGamF16T_ = outGamBf16Buf_.Get<half>();
            Cast(outNormF16T_, normArrT_[s * CUBE_SUBBATCH], RoundMode::CAST_RINT, n);
            Cast(outGamF16T_, gamArrT_, RoundMode::CAST_RINT, n);
            SyncVMte3();
            DataCopyPad(normGmF16_[vecBase], outNormF16T_, ngParams);
            DataCopyPad(gamGmF16_[vecBase], outGamF16T_, ngParams);
        } else {
            Cast(outNormBf16T_, normArrT_[s * CUBE_SUBBATCH], RoundMode::CAST_RINT, n);
            Cast(outGamBf16T_, gamArrT_, RoundMode::CAST_RINT, n);
            SyncVMte3();
            DataCopyPad(normGm_[vecBase], outNormBf16T_, ngParams);
            DataCopyPad(gamGm_[vecBase], outGamBf16T_, ngParams);
        }
    }

    // 打包 idx：向量乘幂次表 + BlockReduceSum（每个 32B datablock 内 8 个 fp32 求和
    // 恰好等于"8 个元素各占 BITS 位字段合成一个字"；不能用 4/2/1 元素偏移的 Add 树——
    // 那些偏移是 16B/8B/4B，会触发 "UB address accessed by VEC instruction is not aligned"）
    __aicore__ inline void PackIdx(int32_t n, int32_t s)
    {
        uint32_t elems = static_cast<uint32_t>(n) * HEAD_DIM;
        if (BITS == 4) {
            // 4bit×8=32bit 超出 fp32 24bit 整数精度 → 用两个逐比特掩码分别归并低半字/高半字
            // （逐比特 mask：0x0F0F.. = 每个 8 元素组取前 4 个，0xF0F0.. = 取后 4 个）
            uint64_t mLo[1] = {0x0F0F0F0F0F0F0F0FULL};
            uint64_t mHi[1] = {0xF0F0F0F0F0F0F0F0ULL};
            for (uint32_t ch = 0; ch < elems; ch += 1024) { // 表固定 1024 宽 ⇒ 分块
                uint32_t cn = (elems - ch < 1024) ? (elems - ch) : 1024;
                Mul(sqT_[ch], idxFT_[ch], stFpT_, cn);
                BlockReduceSum(dupT_[ch / 8], sqT_[ch], static_cast<int32_t>(cn / 64), mLo, 1, 1, 8);
                BlockReduceSum(dupT_[SUB_ELEMS / 2 + ch / 8], sqT_[ch], static_cast<int32_t>(cn / 64), mHi, 1, 1, 8);
            }
            SyncVS();
            for (int32_t v = 0; v < n; v++) {
                int32_t iBase = (s * CUBE_SUBBATCH + v) * IDX_BYTES;
                for (int32_t g = 0; g < HEAD_DIM / 8; g++) {
                    uint32_t lo = static_cast<uint32_t>(static_cast<int32_t>(dupT_.GetValue(v * (HEAD_DIM / 8) + g)));
                    uint32_t hi = static_cast<uint32_t>(
                        static_cast<int32_t>(dupT_.GetValue(SUB_ELEMS / 2 + v * (HEAD_DIM / 8) + g)));
                    outIdxT_.SetValue(iBase + g * 4 + 0, static_cast<uint8_t>(lo & 0xFFu));
                    outIdxT_.SetValue(iBase + g * 4 + 1, static_cast<uint8_t>((lo >> 8) & 0xFFu));
                    outIdxT_.SetValue(iBase + g * 4 + 2, static_cast<uint8_t>(hi & 0xFFu));
                    outIdxT_.SetValue(iBase + g * 4 + 3, static_cast<uint8_t>((hi >> 8) & 0xFFu));
                }
            }
            return;
        }
        // 表固定 1024 宽 ⇒ 分块：repeatTime=cn/64，mask=64（连续模式），dstRepStride=1，srcBlkStride=1，srcRepStride=8
        for (uint32_t ch = 0; ch < elems; ch += 1024) {
            uint32_t cn = (elems - ch < 1024) ? (elems - ch) : 1024;
            Mul(sqT_[ch], idxFT_[ch], stFpT_, cn);
            BlockReduceSum(dupT_[ch / 8], sqT_[ch], static_cast<int32_t>(cn / 64), 64, 1, 1, 8);
        }
        SyncVS();
        for (int32_t v = 0; v < n; v++) {
            int32_t iBase = (s * CUBE_SUBBATCH + v) * IDX_BYTES;
            for (int32_t g = 0; g < HEAD_DIM / 8; g++) {
                uint32_t w0 = static_cast<uint32_t>(static_cast<int32_t>(dupT_.GetValue(v * (HEAD_DIM / 8) + g)));
                for (int32_t by = 0; by < BITS; by++) {
                    outIdxT_.SetValue(iBase + g * BITS + by, static_cast<uint8_t>((w0 >> (8 * by)) & 0xFFu));
                }
            }
        }
    }

    __aicore__ inline void ProcessAiv()
    {
        KVPRINT("[AIV%d] enter\n", (int)aivSubIdx_);
        // Init：H/S staging 数据由 AIV0 完成；MODE2 通知需要两个 sub-block 都参与
        if (aivSubIdx_ == 0) {
            if (matDtype_ == 1) {
                StageMatrixBf16(rotSrcBf16Gm_, hHiGm_, hLoGm_);
                StageMatrixBf16(qjlSrcBf16Gm_, sHiGm_, sLoGm_);
            } else {
                StageMatrix(rotSrcGm_, hHiGm_, hLoGm_);
                StageMatrix(qjlSrcGm_, sHiGm_, sLoGm_);
            }
            SyncSMte3();
        }
        // 复用 staging 缓冲 stFpT_（8KB=2048 float，staging 之后不再使用）前 SUB_ELEMS 个元素
        // 作为 idx 打包幂次表：mulTbl[i] = 2^(BITS*(i%8))（BITS=4 时按 4 位半字：2^(4*(i%4))）。
        // 打包即 Σ_l idx[l]*2^(BITS*l)：8 个字段共 ≤24 bit（BITS≤3）/16 bit（BITS=4 半字），fp32 精确无进位。
        // 表按固定 1024 宽（8 周期模式），不随 SUB_ELEMS 变；PackIdx/QJL 的 Mul 按 1024 分块调用。
        for (int32_t i = 0; i < 1024; i++) {
            int32_t l = i % 8;
            int32_t sh = (BITS == 4) ? (4 * (l % 4)) : (BITS * l);
            stFpT_.SetValue(i, static_cast<float>(1 << sh));
            stFpT_.SetValue(1024 + i, static_cast<float>(1 << l)); // QJL 打包用 2^(i%8)
        }
        SyncSV();
        KVPRINT("[AIV%d] staging done\n", (int)aivSubIdx_);
        KVPRINT("[AIV%d] HS set\n", (int)aivSubIdx_);
        CrossSet(CUBE_FLAG_HS, false);

        // 预准备 tile 0 的 u（prep 被提前到"等 Y"之前，使 tile t+1 的 prep 与 tile t 的 GEMM 重叠）
        {
            int32_t nTile0 = static_cast<int32_t>((vecCount_ < CUBE_TILE_V) ? vecCount_ : CUBE_TILE_V);
            int32_t subs0 = (nTile0 + CUBE_SUBBATCH - 1) / CUBE_SUBBATCH;
            for (int32_t s = aivSubIdx_; s < subs0; s += 2) {
                int32_t n = (s == subs0 - 1) ? (nTile0 - s * CUBE_SUBBATCH) : CUBE_SUBBATCH;
                PrepSubbatch(vecStart_ + s * CUBE_SUBBATCH, n, s);
            }
            CrossSet(CUBE_FLAG_U + 0, false);
        }

        for (uint32_t t = 0; t < tiles_; t++) {
            uint32_t d = t % 2;
            int64_t vecBase = vecStart_ + static_cast<int64_t>(t) * CUBE_TILE_V;
            int32_t nTile = static_cast<int32_t>((vecCount_ - static_cast<int64_t>(t) * CUBE_TILE_V) < CUBE_TILE_V ?
                                                     (vecCount_ - static_cast<int64_t>(t) * CUBE_TILE_V) :
                                                     CUBE_TILE_V);
            int32_t subs = (nTile + CUBE_SUBBATCH - 1) / CUBE_SUBBATCH;

            // 读 y → 量化 → r 拆分 → idx/norm/gamma
            CrossWait(CUBE_FLAG_Y + d, true);
            for (int32_t s = aivSubIdx_; s < subs; s += 2) {
                int32_t n = (s == subs - 1) ? (nTile - s * CUBE_SUBBATCH) : CUBE_SUBBATCH;
                int64_t vb = vecBase + s * CUBE_SUBBATCH;
                QuantSubbatch(vb, n, s);
                PackIdx(n, s);
                FlushIdxNormGamma(vb, n, s);
            }
            if (t < 2)
                KVPRINT("[AIV%d] tile %u R set\n", (int)aivSubIdx_, t);
            CrossSet(CUBE_FLAG_R + d, false);

            // 准备 tile t+1 的 u：与 AIC 的 GEMM2(t) 重叠。
            // 必须放在 quant/flush 之后 —— prep 会写 normArrT_，提前会覆盖本轮待 flush 的 norm。
            uint32_t tn = t + 1;
            if (tn < tiles_) {
                int64_t vecBaseN = vecStart_ + static_cast<int64_t>(tn) * CUBE_TILE_V;
                int32_t nTileN =
                    static_cast<int32_t>((vecCount_ - static_cast<int64_t>(tn) * CUBE_TILE_V) < CUBE_TILE_V ?
                                             (vecCount_ - static_cast<int64_t>(tn) * CUBE_TILE_V) :
                                             CUBE_TILE_V);
                int32_t subsN = (nTileN + CUBE_SUBBATCH - 1) / CUBE_SUBBATCH;
                for (int32_t s = aivSubIdx_; s < subsN; s += 2) {
                    int32_t n = (s == subsN - 1) ? (nTileN - s * CUBE_SUBBATCH) : CUBE_SUBBATCH;
                    PrepSubbatch(vecBaseN + s * CUBE_SUBBATCH, n, s);
                }
                CrossSet(CUBE_FLAG_U + (tn % 2), false);
            }

            // 读 proj → qjl bit → 打包输出（每 AIV 冲自己的 sub-batch）
            if (t < 2)
                KVPRINT("[AIV%d] tile %u P waited\n", (int)aivSubIdx_, t);
            CrossWait(CUBE_FLAG_P + d, true);
            SyncSMte3();
            for (int32_t s = aivSubIdx_; s < subs; s += 2) {
                int32_t n = (s == subs - 1) ? (nTile - s * CUBE_SUBBATCH) : CUBE_SUBBATCH;
                int64_t vb = vecBase + s * CUBE_SUBBATCH;
                uint32_t elems = static_cast<uint32_t>(n) * HEAD_DIM;
                {
                    event_t e3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                    SetFlag<HardEvent::V_MTE2>(e3);
                    WaitFlag<HardEvent::V_MTE2>(e3);
                }
                DataCopy(yT_, pGm_[vb * HEAD_DIM], elems); // 别名 yT_（y 已在 quant 用完）
                SyncMte2V();
                // 子批宽度：qjlBT_ 预清零当 false 值，dupT_ 置 1 当 true 值，再按 mask 写回
                Duplicate(qjlBT_, 0.0f, elems);
                Duplicate(dupT_, 1.0f, elems);
                Compare<float>(maskT_, yT_, qjlBT_, CMPMODE::GE, elems);
                Select(qjlBT_, maskT_, dupT_, qjlBT_, SELMODE::VSEL_TENSOR_TENSOR_MODE, elems);
                // QJL 打包：× 2^(i%8) 后按 32B datablock 归并即得每个字节（8 个 bit 合成 1 字节）
                // 表固定 1024 宽 ⇒ 按 1024 分块（子批最大 32 行 = 4096 元素）
                for (uint32_t ch = 0; ch < elems; ch += 1024) {
                    uint32_t cn = (elems - ch < 1024) ? (elems - ch) : 1024;
                    Mul(qjlBT_[ch], qjlBT_[ch], stFpT_[1024], cn);
                    BlockReduceSum(dupT_[ch / 8], qjlBT_[ch], static_cast<int32_t>(cn / 64), 64, 1, 1, 8);
                }
                SyncVS();
                // 注：fp32/uint8 之间的 Cast 在本 CANN 不支持（CastIntrinsicsImpl 无匹配重载），
                // 字节只能逐元素 SetValue 写出。
                for (int32_t v = 0; v < n; v++) {
                    int32_t qBase = v * QJL_BYTES;
                    for (int32_t g = 0; g < QJL_BYTES; g++) {
                        outQjlT_.SetValue(qBase + g, static_cast<uint8_t>(
                                                         static_cast<int32_t>(dupT_.GetValue(v * (HEAD_DIM / 8) + g))));
                    }
                }
                SyncSMte3();
                DataCopyExtParams qjlParams;
                qjlParams.blockCount = 1;
                qjlParams.blockLen = static_cast<uint32_t>(n) * QJL_BYTES;
                qjlParams.srcStride = 0;
                qjlParams.dstStride = 0;
                DataCopyPad(qjlOutGm_[vb * QJL_BYTES], outQjlT_, qjlParams);
            }
        }
    }

    // ---------------- AIC 侧 ----------------
    __aicore__ inline void InitAic(TPipe *pipe)
    {
        pipe_ = pipe;
        pipe->InitBuffer(hHiNzBuf_, HEAD_DIM * HEAD_DIM * sizeof(bfloat16_t));
        pipe->InitBuffer(hLoNzBuf_, HEAD_DIM * HEAD_DIM * sizeof(bfloat16_t));
        pipe->InitBuffer(sHiNzBuf_, HEAD_DIM * HEAD_DIM * sizeof(bfloat16_t));
        pipe->InitBuffer(sLoNzBuf_, HEAD_DIM * HEAD_DIM * sizeof(bfloat16_t));
        pipe->InitBuffer(a1Buf_, CUBE_TILE_V * HEAD_DIM * sizeof(bfloat16_t));
        pipe->InitBuffer(a2Buf_, CUBE_TILE_V * HEAD_DIM * sizeof(bfloat16_t));
        pipe->InitBuffer(b2Buf_, 128 * HEAD_DIM * sizeof(bfloat16_t));
        pipe->InitBuffer(c1Buf_, CUBE_TILE_V * HEAD_DIM * sizeof(float));
        hHiNzT_ = hHiNzBuf_.Get<bfloat16_t>();
        hLoNzT_ = hLoNzBuf_.Get<bfloat16_t>();
        sHiNzT_ = sHiNzBuf_.Get<bfloat16_t>();
        sLoNzT_ = sLoNzBuf_.Get<bfloat16_t>();
        a1T_ = a1Buf_.Get<bfloat16_t>();
        a2T_ = a2Buf_.Get<bfloat16_t>();
        b2T_ = b2Buf_.Get<bfloat16_t>();
        c1T_ = c1Buf_.Get<float>();
    }

    __aicore__ inline void Nd2NzWeights()
    {
        Nd2NzParams nz;
        nz.ndNum = 1;
        nz.nValue = HEAD_DIM;
        nz.dValue = HEAD_DIM;
        nz.srcDValue = HEAD_DIM;
        nz.dstNzC0Stride = HEAD_DIM;
        nz.dstNzNStride = 1;
        nz.srcNdMatrixStride = 0;
        nz.dstNzMatrixStride = 0;
        DataCopy(hHiNzT_, hHiGm_, nz);
        DataCopy(hLoNzT_, hLoGm_, nz);
        DataCopy(sHiNzT_, sHiGm_, nz);
        DataCopy(sLoNzT_, sLoGm_, nz);
        SyncMte2Mte1();
    }

    // 矩阵（CUBE_TILE_V×128 bf16 ND）GM→L1 Nz。
    // 统一按满 tile 装载/计算：尾部不足的行读到相邻有效数据（GM 内、不影响结果），
    // 只有 Fixpipe 出图用真实行数 —— 这样 Cube 侧永远走 probe 验证过的形状（m=CUBE_TILE_V）。
    __aicore__ inline void Nd2NzTile(const GlobalTensor<bfloat16_t> &srcGm, int64_t vecBase, int32_t /*nRows*/)
    {
        Nd2NzParams nz;
        nz.ndNum = 1;
        nz.nValue = static_cast<uint32_t>(CUBE_MMAD_M);
        nz.dValue = HEAD_DIM;
        nz.srcDValue = HEAD_DIM;
        nz.dstNzC0Stride = CUBE_MMAD_M;
        nz.dstNzNStride = 1;
        nz.srcNdMatrixStride = 0;
        nz.dstNzMatrixStride = 0;
        DataCopy(a1T_, srcGm[vecBase * HEAD_DIM], nz);
        SyncMte2Mte1();
    }

    __aicore__ inline void LoadA(int32_t /*nRows*/, int32_t kNow)
    {
        uint32_t mAlign = CUBE_MMAD_M;
        uint32_t mLoop = static_cast<uint32_t>(CUBE_MMAD_M) / 16;
        for (uint32_t i = 0; i < mLoop; i++) {
            LoadData2DParams ap{};
            ap.startIndex = i;
            ap.repeatTimes = kNow / 16;
            ap.srcStride = mAlign / 16;
            ap.dstGap = 0;
            ap.ifTranspose = false;
            LoadData(a2T_[i * 16 * kNow], a1T_, ap);
        }
    }

    __aicore__ inline void LoadB(const LocalTensor<bfloat16_t> &wNz)
    {
        LoadData2DParams bp{};
        bp.startIndex = 0;
        bp.repeatTimes = (128 / 16) * (HEAD_DIM / 16);
        bp.srcStride = 1;
        bp.dstGap = 0;
        bp.ifTranspose = false;
        LoadData(b2T_, wNz, bp);
    }

    __aicore__ inline MmadParams MakeMmadParams(int32_t nRows, bool init)
    {
        MmadParams mp{};
        mp.m = static_cast<uint16_t>(CUBE_MMAD_M); // 单轮 64 行（probe 已验证的形状）
        mp.n = HEAD_DIM;
        mp.k = HEAD_DIM;
        mp.cmatrixInitVal = init;
        mp.cmatrixSource = false;
        return mp;
    }

    __aicore__ inline void FixTile(const GlobalTensor<float> &dstGm, int64_t vecBase, int32_t nRows)
    {
        SyncMFix();
        FixpipeParamsV220 fp;
        fp.nSize = HEAD_DIM;
        fp.mSize = static_cast<uint16_t>(nRows);
        fp.srcStride = CUBE_MMAD_M;
        fp.dstStride = HEAD_DIM;
        fp.ndNum = 1;
        fp.reluEn = false;
        fp.quantPre = QuantMode_t::NoQuant;
        Fixpipe(dstGm[vecBase * HEAD_DIM], c1T_, fp);
        SyncFixM();
    }

    __aicore__ inline void ProcessAic()
    {
        CrossWait(CUBE_FLAG_HS, true);
        Nd2NzWeights();
        SyncFixM(); // c1 初始可用

        for (uint32_t t = 0; t < tiles_; t++) {
            uint32_t d = t % 2;
            int64_t vecBase = vecStart_ + static_cast<int64_t>(t) * CUBE_TILE_V;
            int32_t nTile = static_cast<int32_t>((vecCount_ - static_cast<int64_t>(t) * CUBE_TILE_V) < CUBE_TILE_V ?
                                                     (vecCount_ - static_cast<int64_t>(t) * CUBE_TILE_V) :
                                                     CUBE_TILE_V);

            // GEMM1: y = u @ H^T（tile 内按 CUBE_MMAD_M=64 行分轮，减少 tile 数/握手次数）
            CrossWait(CUBE_FLAG_U + d, true);
            for (int32_t h = 0; h < CUBE_TILE_V; h += CUBE_MMAD_M) {
                int32_t nt = nTile - h;
                if (nt <= 0)
                    break;
                if (nt > CUBE_MMAD_M)
                    nt = CUBE_MMAD_M;
                int64_t vb = vecBase + h;
                // A1 复用保护：上一次 LoadA(MTE1) 读完 a1 之前 MTE2 不能覆写
                {
                    event_t eA = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_MTE2));
                    SetFlag<HardEvent::MTE1_MTE2>(eA);
                    WaitFlag<HardEvent::MTE1_MTE2>(eA);
                }
                Nd2NzTile(ubHiGm_, vb, nt);
                LoadA(nt, HEAD_DIM);
                LoadB(hHiNzT_);
                SyncMte1M();
                Mmad(c1T_, a2T_, b2T_, MakeMmadParams(nt, true));
                SyncMMte1();
                // y 保留 bf16x3（实测 bf16 直用会把 qjl 翻转率从 3.4e-4 拉到 18%，有踩线风险）
                LoadB(hLoNzT_);
                SyncMte1M();
                Mmad(c1T_, a2T_, b2T_, MakeMmadParams(nt, false));
                SyncMMte1();
                {
                    event_t eB = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_MTE2));
                    SetFlag<HardEvent::MTE1_MTE2>(eB);
                    WaitFlag<HardEvent::MTE1_MTE2>(eB);
                }
                Nd2NzTile(ubLoGm_, vb, nt);
                LoadA(nt, HEAD_DIM);
                LoadB(hHiNzT_);
                SyncMte1M();
                Mmad(c1T_, a2T_, b2T_, MakeMmadParams(nt, false));
                SyncMMte1();
                // y 用 bf16x3（hi@hi + hi@lo + lo@hi）。补上 lo@lo 的 bf16x4 实测仅把 idx 差异
                // 从 10~25 降到 8~16 byte/786432、y 误差 2.3e-6→1.7e-6，却使 prefill/mse2 的
                // 10× 余量从 4.3%/2.4% 降到 3.0%/1.1%，得不偿失（剩余误差主要来自 Cube 内 fp32 累加顺序）。
                FixTile(yGm_, vb, nt);
            }
            CrossSet(CUBE_FLAG_Y + d, true);

            // GEMM2: proj = r @ S^T（同样分轮）
            CrossWait(CUBE_FLAG_R + d, true);
            for (int32_t h = 0; h < CUBE_TILE_V; h += CUBE_MMAD_M) {
                int32_t nt = nTile - h;
                if (nt <= 0)
                    break;
                if (nt > CUBE_MMAD_M)
                    nt = CUBE_MMAD_M;
                int64_t vb = vecBase + h;
                {
                    event_t eC = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_MTE2));
                    SetFlag<HardEvent::MTE1_MTE2>(eC);
                    WaitFlag<HardEvent::MTE1_MTE2>(eC);
                }
                Nd2NzTile(rHiGm_, vb, nt);
                LoadA(nt, HEAD_DIM);
                LoadB(sHiNzT_);
                SyncMte1M();
                Mmad(c1T_, a2T_, b2T_, MakeMmadParams(nt, true));
                SyncMMte1();
                // proj 保留 bf16x3（qjl 是符号位，对精度最敏感）
                LoadB(sLoNzT_);
                SyncMte1M();
                Mmad(c1T_, a2T_, b2T_, MakeMmadParams(nt, false));
                SyncMMte1();
                {
                    event_t eD = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_MTE2));
                    SetFlag<HardEvent::MTE1_MTE2>(eD);
                    WaitFlag<HardEvent::MTE1_MTE2>(eD);
                }
                Nd2NzTile(rLoGm_, vb, nt);
                LoadA(nt, HEAD_DIM);
                LoadB(sHiNzT_);
                SyncMte1M();
                Mmad(c1T_, a2T_, b2T_, MakeMmadParams(nt, false));
                SyncMMte1();
                FixTile(pGm_, vb, nt);
            }
            CrossSet(CUBE_FLAG_P + d, true);
        }
    }

private:
    GlobalTensor<bfloat16_t> kvGm_;
    GlobalTensor<uint8_t> idxGm_;
    GlobalTensor<uint8_t> qjlOutGm_;
    GlobalTensor<bfloat16_t> normGm_;
    GlobalTensor<bfloat16_t> gamGm_;
    GlobalTensor<half> normGmF16_;
    GlobalTensor<half> gamGmF16_;
    GlobalTensor<float> rotSrcGm_;
    GlobalTensor<float> qjlSrcGm_;
    GlobalTensor<bfloat16_t> rotSrcBf16Gm_;
    GlobalTensor<bfloat16_t> qjlSrcBf16Gm_;
    GlobalTensor<bfloat16_t> hHiGm_, hLoGm_, sHiGm_, sLoGm_;
    GlobalTensor<bfloat16_t> ubHiGm_, ubLoGm_, rHiGm_, rLoGm_;
    GlobalTensor<float> yGm_, pGm_;

    TPipe *pipe_ = nullptr;
    // AIV 缓冲
    TBuf<QuePosition::VECCALC> xfBuf_, uBuf_, sqBuf_, uHiBuf_, uLoBuf_, tmpFpBuf_, xinBuf_;
    TBuf<QuePosition::VECCALC> yBuf_, yhatBuf_, rBuf_, projBuf_, idxFBuf_, qjlBBuf_;
    TBuf<QuePosition::VECCALC> rHiBuf_, rLoBuf_, dupBuf_, dupZeroBuf_, maskBuf_;
    TBuf<QuePosition::VECCALC> normArrBuf_, gamArrBuf_, outIdxBuf_, outQjlBuf_;
    TBuf<QuePosition::VECCALC> outNormBf16Buf_, outGamBf16Buf_;
    TBuf<QuePosition::VECCALC> stFpBuf_, stBkBuf_, stHiBuf_, stLoBuf_;
    // AIC 缓冲
    TBuf<TPosition::A1> hHiNzBuf_, hLoNzBuf_, sHiNzBuf_, sLoNzBuf_, a1Buf_;
    TBuf<TPosition::A2> a2Buf_;
    TBuf<TPosition::B2> b2Buf_;
    TBuf<TPosition::CO1> c1Buf_;

    LocalTensor<float> xfT_, uT_, sqT_, tmpFpT_, yT_, yhatT_, rT_, projT_, idxFT_, qjlBT_;
    LocalTensor<float> dupT_, dupZeroT_, normArrT_, gamArrT_, stFpT_, stBkT_;
    LocalTensor<bfloat16_t> uHiT_, uLoT_, xinT_, rHiT_, rLoT_, outNormBf16T_, outGamBf16T_;
    LocalTensor<bfloat16_t> stHiT_, stLoT_;
    LocalTensor<uint8_t> maskT_, outIdxT_, outQjlT_;
    LocalTensor<bfloat16_t> hHiNzT_, hLoNzT_, sHiNzT_, sLoNzT_, a1T_, a2T_, b2T_;
    LocalTensor<float> c1T_;

    uint32_t aiCoreIdx_ = 0;
    uint32_t aivSubIdx_ = 0;
    int64_t numVectors_ = 0;
    int64_t vecStart_ = 0;
    int64_t vecCount_ = 0;
    uint32_t tiles_ = 0;
    uint64_t offHHi_ = 0, offHLo_ = 0, offSHi_ = 0, offSLo_ = 0;
    uint64_t offUbHi_ = 0, offUbLo_ = 0, offRHi_ = 0, offRLo_ = 0;
    uint64_t offY_ = 0, offP_ = 0;
    uint64_t wsTotal_ = 0;
    uint32_t matDtype_ = 0; // rotation/qjl_matrix 元素类型：0=float32，1=bfloat16
    uint32_t outDtype_ = 0; // quant_norm/quant_gamma 元素类型：0=bfloat16，1=float16
};

} // namespace KvCacheTurboQuant

#endif // KERNEL_KV_CACHE_TURBO_QUANT_CUBE_H
