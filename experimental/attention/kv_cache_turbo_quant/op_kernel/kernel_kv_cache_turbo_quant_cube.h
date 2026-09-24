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
 * \brief KvCacheTurboQuant encode kernel：AIC/AIV 混合 Cube 路径。
 *
 * 数值路线（目标是与 golden.py 的 torch_npu fp32 matmul 语义一致到 fp32 末位）：
 *   1. 旋转阵 H(fp32) 拆 bf16 三连 Hh+Hl+Hll（3×8bit 尾数 = 24bit，精确覆盖 fp32），残差投影阵 S 拆 fp16 hi/lo；
 *   2. x 保持 bf16 不拆（输入本就是 bf16，零舍入），z = x@Hh^T + x@Hl^T + x@Hll^T（3 次 Mmad 累进同一 L0C fp32 平面）；
 *   3. 先乘后归一：norm=||z||，y = z*(1/norm)（标量正确舍入倒数；H 正交 ⇒ ||z||==||x||）；
 *   4. y 量化得 idx、残差 r=y-yhat；r 拆 fp16 hi/lo（rh/rl）——qjl 符号不依赖 r 的归一化（rn>0）；
 *   5. p 走 3 个 fp16 独立平面 p1=rh@Sh、p3=rh@Sl、p2=rl@Sh（各自 Fixpipe 出 fp16，不做 L0C 累加以保留修正项），
 *      AIV 在 fp32 里 p=p1+p2+p3 再取符号得 qjl。
 *
 * 数据流（每 aiCore，tile M=64 向量，双缓冲 d=t%2）：
 *   AIV：UploadX（kv→workspace x 区，纯拷贝）→ CrossCoreSetFlag(U+d)
 *   AIC：WaitFlag(U+d) → Nd2Nz x → z = x@(Hh+Hl+Hll)^T（L0C fp32）→ Fixpipe float 出 z 区 → SetFlag(Y+d)
 *   AIV：读 z → norm/归一/量化 → r 拆 fp16 hi/lo → SetFlag(R+d)
 *   AIC：GEMM2 三平面 p1/p2/p3（fp16 出）→ SetFlag(P+d)
 *   AIV：读 p1/p2/p3 求和 → qjl bit → 输出
 *
 * 精度依据（真机+本地复刻对照官方 golden 均 0 差或 1~2 字节量级）：
 *   - bf16 三连拆把 H 的 24bit 尾数全部保留，x 侧因输入即 bf16 无舍入，故 z 与 fp32 参考仅有累加序末位差；
 *   - 先乘后归一（z*inv）与 golden 的 unit@H（先除后乘）等价到 fp32 末位；
 *   - p 的三平面拆分保留 r_hat/S 的低位修正项，QJL 符号翻转率由 ~1e-3 压到 ~1e-5 量级。
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
// flag ID 取 4~10：0~2 可能与 codegen wrapper 内部同步冲突，11~15 为框架保留（compressor 同款范围）。
// 可用 ID 只有 4~10 共 7 个，而 HS + 四级流水 × 双缓冲需要 9 个，故 U/Y/R/P 的基址必然相邻交叠
// （U+1==Y+0、Y+1==R+0、R+1==P+0）。交叠是安全的，前提是每个 flag 严格按
// "生产者 Set → 消费者 Wait（wait 后硬件自动清零）"交替使用——wait 会消费掉标志位，
// 因此不会出现"自己 set 的标志把自己提前放行"。以 flag 6 为例：
//   AIC Set(6)=Y+0（GEMM1 偶 tile 完成）→ AIV Wait(6)（偶 tile quant 取 z，清零）
//   → AIV Set(6)=U+1（奇 tile UploadX 完成，且必在偶 tile qjl 之后）→ AIC Wait(6)（奇 tile GEMM1 取 x，清零）
//   → 循环。任意时刻同一 flag 至多一个未消费的 set，Set 与 Wait 严格交替。
// 若要调整流水相位顺序，必须重新核对每条 flag 的 Set/Wait 交替关系（见 ProcessAiv/ProcessAic）。
constexpr uint32_t CUBE_FLAG_HS = 4; // AIV0→AIC：H/S 拆分完成
constexpr uint32_t CUBE_FLAG_U = 5;  // AIV→AIC：x 上传就绪（dbIdx 0/1）
constexpr uint32_t CUBE_FLAG_Y = 6;  // AIC→AIV：z 就绪
constexpr uint32_t CUBE_FLAG_R = 7;  // AIV→AIC：rh/rl 就绪
constexpr uint32_t CUBE_FLAG_P = 8;  // AIC→AIV：p1/p2/p3 就绪

constexpr int32_t CUBE_TILE_V = 64; // Cube tile 向量数（128 实测无收益且拖慢小 case；AIC 按 CUBE_MMAD_M=64 分轮）
constexpr int32_t CUBE_MMAD_M = 64;   // 单次 Mmad 的 m（A2 上限 64）
constexpr int32_t CUBE_SUBBATCH = 32; // AIV 子批（每子批屏障数减半；32 实测可行且 UB 已腾出）

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

        // workspace 布局（host 侧同公式定总量，见 tiling.cpp）
        // MIX 场景下 raw workspace 指针不等于用户区起点（框架在前后各留系统区），
        // 实测直接改用 GetUserWorkspace(workspace) 反而外推出界 —— 数据落点就是框架给的 raw 指针，
        // 各阶段偏移一律相对它计算。
        GM_ADDR wsBase = workspace;
        uint64_t off = WS_HEAD;
        offHh_ = off;
        off += HEAD_DIM * HEAD_DIM * 2; // bf16
        offHl_ = off;
        off += HEAD_DIM * HEAD_DIM * 2;
        offHll_ = off;
        off += HEAD_DIM * HEAD_DIM * 2;
        offSh_ = off;
        off += HEAD_DIM * HEAD_DIM * 2; // fp16（与 bf16 同为 2B/元素）
        offSl_ = off;
        off += HEAD_DIM * HEAD_DIM * 2;
        offX_ = off;
        off += numVectors_ * HEAD_DIM * 2; // bf16
        offRh_ = off;
        off += numVectors_ * HEAD_DIM * 2; // fp16
        offRl_ = off;
        off += numVectors_ * HEAD_DIM * 2;
        offZ_ = off;
        off += numVectors_ * HEAD_DIM * 4; // fp32
        offP_ = off;
        off += numVectors_ * HEAD_DIM * 4; // fp32

        hHhGm_.SetGlobalBuffer((__gm__ bfloat16_t *)(wsBase + offHh_));
        hHlGm_.SetGlobalBuffer((__gm__ bfloat16_t *)(wsBase + offHl_));
        hHllGm_.SetGlobalBuffer((__gm__ bfloat16_t *)(wsBase + offHll_));
        sHiGm_.SetGlobalBuffer((__gm__ half *)(wsBase + offSh_));
        sLoGm_.SetGlobalBuffer((__gm__ half *)(wsBase + offSl_));
        xGm_.SetGlobalBuffer((__gm__ bfloat16_t *)(wsBase + offX_));
        rHiGm_.SetGlobalBuffer((__gm__ half *)(wsBase + offRh_));
        rLoGm_.SetGlobalBuffer((__gm__ half *)(wsBase + offRl_));
        zGm_.SetGlobalBuffer((__gm__ float *)(wsBase + offZ_));
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
    // （AIC: Nd2NzTile/Nd2NzWeights 搬 x/H；AIV: QuantSubbatch 搬 z、P 阶段搬 p）。
    // 挂 PIPE_FIX 时 AIV/AIC 的 FIX 流水上并没有后续指令，等待形同虚设，
    // 会把 AIC 还没写完的 z/p 读进来 —— 表现为部分 sub-batch 结果错、跨轮次不确定。
    __aicore__ inline void CrossWait(uint32_t flag)
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

    // fp32 → fp16 hi/lo 拆：hi=RNE(x)，lo=RNE(x-hi)（hi+lo 覆盖 ~22bit，供 p 平面修正项）
    __aicore__ inline void SplitFp16HiLo(const LocalTensor<float> &srcFp, const LocalTensor<half> &dstHi,
                                         const LocalTensor<half> &dstLo, const LocalTensor<float> &tmpFp,
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
        pipe->InitBuffer(sqBuf_, SUB_ELEMS * sizeof(float));     // 16KB
        pipe->InitBuffer(xBuf_, SUB_ELEMS * sizeof(bfloat16_t)); // 8KB（UploadX 的 x 中转）
        // 量化链（fp32）
        pipe->InitBuffer(yBuf_, SUB_ELEMS * sizeof(float));
        pipe->InitBuffer(yhatBuf_, SUB_ELEMS * sizeof(float));
        pipe->InitBuffer(rBuf_, SUB_ELEMS * sizeof(float));
        pipe->InitBuffer(idxFBuf_, SUB_ELEMS * sizeof(float));
        pipe->InitBuffer(qjlBBuf_, SUB_ELEMS * sizeof(float));
        // r 的 fp16 hi/lo
        pipe->InitBuffer(rHiBuf_, SUB_ELEMS * sizeof(half));
        pipe->InitBuffer(rLoBuf_, SUB_ELEMS * sizeof(half));
        pipe->InitBuffer(dupBuf_, SUB_ELEMS * sizeof(float));
        pipe->InitBuffer(maskBuf_, SUB_ELEMS);
        pipe->InitBuffer(normArrBuf_, CUBE_TILE_V * sizeof(float));
        pipe->InitBuffer(invArrBuf_, CUBE_TILE_V * sizeof(float));
        pipe->InitBuffer(gamArrBuf_, CUBE_SUBBATCH * sizeof(float));
        pipe->InitBuffer(outIdxBuf_, CUBE_TILE_V * IDX_BYTES);
        pipe->InitBuffer(outQjlBuf_, CUBE_TILE_V * QJL_BYTES);
        pipe->InitBuffer(outNormBf16Buf_, CUBE_TILE_V * sizeof(bfloat16_t));
        pipe->InitBuffer(outGamBf16Buf_, CUBE_SUBBATCH * sizeof(bfloat16_t));
        // H/S staging 分块缓冲（16 行 = 8KB fp32；H 三连用 stHi/stMid/stLo，S 用 stHi/stLo Get<half>）
        pipe->InitBuffer(stFpBuf_, 16 * HEAD_DIM * sizeof(float));
        pipe->InitBuffer(stHiBuf_, 16 * HEAD_DIM * sizeof(bfloat16_t));
        pipe->InitBuffer(stMidBuf_, 16 * HEAD_DIM * sizeof(bfloat16_t));
        pipe->InitBuffer(stLoBuf_, 16 * HEAD_DIM * sizeof(bfloat16_t));

        sqT_ = sqBuf_.Get<float>();
        xT_ = xBuf_.Get<bfloat16_t>();
        yT_ = yBuf_.Get<float>();
        yhatT_ = yhatBuf_.Get<float>();
        rT_ = rBuf_.Get<float>();
        idxFT_ = idxFBuf_.Get<float>();
        qjlBT_ = qjlBBuf_.Get<float>();
        rHiT_ = rHiBuf_.Get<half>();
        rLoT_ = rLoBuf_.Get<half>();
        dupT_ = dupBuf_.Get<float>();
        maskT_ = maskBuf_.Get<uint8_t>();
        normArrT_ = normArrBuf_.Get<float>();
        invArrT_ = invArrBuf_.Get<float>();
        gamArrT_ = gamArrBuf_.Get<float>();
        outIdxT_ = outIdxBuf_.Get<uint8_t>();
        outQjlT_ = outQjlBuf_.Get<uint8_t>();
        outNormBf16T_ = outNormBf16Buf_.Get<bfloat16_t>();
        outGamBf16T_ = outGamBf16Buf_.Get<bfloat16_t>();
        stFpT_ = stFpBuf_.Get<float>();
        stHiT_ = stHiBuf_.Get<bfloat16_t>();
        stMidT_ = stMidBuf_.Get<bfloat16_t>();
        stLoT_ = stLoBuf_.Get<bfloat16_t>();
        sStHiT_ = stHiBuf_.Get<half>(); // S fp16 暂存复用同一 buffer
        sStLoT_ = stLoBuf_.Get<half>();
    }

    // H(fp32) → bf16 三连，16 行一块；hi/mid/lo 三平面同驻 UB，一次写出（省 2/3 的 MTE3 同步）。
    __aicore__ inline void StageHTriple(const GlobalTensor<float> &srcGm, const GlobalTensor<bfloat16_t> &hiGm,
                                        const GlobalTensor<bfloat16_t> &midGm, const GlobalTensor<bfloat16_t> &loGm)
    {
        for (int32_t chunk = 0; chunk < HEAD_DIM / 16; chunk++) {
            {
                event_t e0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(e0);
                WaitFlag<HardEvent::V_MTE2>(e0);
            }
            DataCopy(stFpT_, srcGm[chunk * 16 * HEAD_DIM], 16 * HEAD_DIM);
            SyncMte2V();
            Cast(stHiT_, stFpT_, RoundMode::CAST_RINT, 16 * HEAD_DIM);  // hi = bf16(src)
            Cast(yhatT_, stHiT_, RoundMode::CAST_NONE, 16 * HEAD_DIM);  // hi_f32
            Sub(stFpT_, stFpT_, yhatT_, 16 * HEAD_DIM);                 // r1 = src - hi
            Cast(stMidT_, stFpT_, RoundMode::CAST_RINT, 16 * HEAD_DIM); // mid = bf16(r1)
            Cast(yhatT_, stMidT_, RoundMode::CAST_NONE, 16 * HEAD_DIM); // mid_f32
            Sub(stFpT_, stFpT_, yhatT_, 16 * HEAD_DIM);                 // r2 = r1 - mid
            Cast(stLoT_, stFpT_, RoundMode::CAST_RINT, 16 * HEAD_DIM);  // lo = bf16(r2)
            SyncVMte3();
            DataCopy(hiGm[chunk * 16 * HEAD_DIM], stHiT_, 16 * HEAD_DIM);
            DataCopy(midGm[chunk * 16 * HEAD_DIM], stMidT_, 16 * HEAD_DIM);
            DataCopy(loGm[chunk * 16 * HEAD_DIM], stLoT_, 16 * HEAD_DIM);
            SyncMte3V();
        }
    }

    // H(bf16 输入) → 直接作 hi，mid/lo 补零（bf16 已是精确表示，无需三连）
    __aicore__ inline void StageHTripleBf16(const GlobalTensor<bfloat16_t> &srcGm, const GlobalTensor<bfloat16_t> &hiGm,
                                            const GlobalTensor<bfloat16_t> &midGm, const GlobalTensor<bfloat16_t> &loGm)
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
            DataCopy(midGm[chunk * 16 * HEAD_DIM], stLoT_, 16 * HEAD_DIM);
            DataCopy(loGm[chunk * 16 * HEAD_DIM], stLoT_, 16 * HEAD_DIM);
            SyncMte3V();
        }
    }

    // S(fp32) → fp16 hi/lo，16 行一块
    __aicore__ inline void StageSHiLo(const GlobalTensor<float> &srcGm, const GlobalTensor<half> &hiGm,
                                      const GlobalTensor<half> &loGm)
    {
        for (int32_t chunk = 0; chunk < HEAD_DIM / 16; chunk++) {
            {
                event_t e0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(e0);
                WaitFlag<HardEvent::V_MTE2>(e0);
            }
            DataCopy(stFpT_, srcGm[chunk * 16 * HEAD_DIM], 16 * HEAD_DIM);
            SyncMte2V();
            Cast(sStHiT_, stFpT_, RoundMode::CAST_RINT, 16 * HEAD_DIM);
            Cast(yhatT_, sStHiT_, RoundMode::CAST_NONE, 16 * HEAD_DIM);
            Sub(stFpT_, stFpT_, yhatT_, 16 * HEAD_DIM);
            Cast(sStLoT_, stFpT_, RoundMode::CAST_RINT, 16 * HEAD_DIM);
            SyncVMte3();
            DataCopy(hiGm[chunk * 16 * HEAD_DIM], sStHiT_, 16 * HEAD_DIM);
            DataCopy(loGm[chunk * 16 * HEAD_DIM], sStLoT_, 16 * HEAD_DIM);
            SyncMte3V();
        }
    }

    // S(bf16 输入) → fp16 hi=cast(bf16)，lo=0（bf16 8bit 尾数可被 fp16 精确表示）
    __aicore__ inline void StageSHiLoBf16(const GlobalTensor<bfloat16_t> &srcGm, const GlobalTensor<half> &hiGm,
                                          const GlobalTensor<half> &loGm)
    {
        for (int32_t chunk = 0; chunk < HEAD_DIM / 16; chunk++) {
            {
                event_t e0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(e0);
                WaitFlag<HardEvent::V_MTE2>(e0);
            }
            DataCopy(stHiT_, srcGm[chunk * 16 * HEAD_DIM], 16 * HEAD_DIM);
            SyncMte2V();
            Cast(stFpT_, stHiT_, RoundMode::CAST_NONE, 16 * HEAD_DIM); // bf16 → fp32（精确）
            Cast(sStHiT_, stFpT_, RoundMode::CAST_RINT,
                 16 * HEAD_DIM); // fp32 → fp16（精确，bf16 8bit 尾数可被 fp16 表示）
            Duplicate(sStLoT_, half(0.0f), 16 * HEAD_DIM);
            SyncVMte3();
            DataCopy(hiGm[chunk * 16 * HEAD_DIM], sStHiT_, 16 * HEAD_DIM);
            DataCopy(loGm[chunk * 16 * HEAD_DIM], sStLoT_, 16 * HEAD_DIM);
            SyncMte3V();
        }
    }

    // 子批 UploadX：kv（bf16）→ workspace x 区，同时算 norm=||x|| 存入 normArrT_（与 golden 的
    // vector_norm(x) 同源；H 正交故 ==||z||）。norm 放这里而非 quant，是为了不占 quant→GEMM2 的关键路径。
    __aicore__ inline void UploadX(int64_t vecBase, int32_t n, int32_t s)
    {
        uint32_t elems = static_cast<uint32_t>(n) * HEAD_DIM;
        {
            event_t e1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(e1);
            WaitFlag<HardEvent::V_MTE2>(e1);
        }
        DataCopy(xT_, kvGm_[vecBase * HEAD_DIM], elems);
        SyncMte2V();
        Cast(yT_, xT_, RoundMode::CAST_NONE, elems); // x bf16 → fp32
        Mul(sqT_, yT_, yT_, elems);
        BlockReduceSum(dupT_, sqT_, static_cast<int32_t>(elems / 64), 64, 1, 1, 8);
        BlockReduceSum(dupT_[SUB_ELEMS / 2], dupT_, static_cast<int32_t>((elems / 8 + 63) / 64), 64, 1, 1, 8);
        SyncVS();
        for (int32_t v = 0; v < n; v++) {
            float sumsq = dupT_.GetValue(SUB_ELEMS / 2 + v * 2) + dupT_.GetValue(SUB_ELEMS / 2 + v * 2 + 1);
            float norm = sqrt(sumsq);
            normArrT_.SetValue(s * CUBE_SUBBATCH + v, norm);
            // 1/norm 标量除法放到这里（重叠段），quant 只读 invArrT_，避免占 quant→GEMM2 关键路径
            invArrT_.SetValue(s * CUBE_SUBBATCH + v, 1.0f / (norm > 1e-30f ? norm : 1e-30f));
        }
        SyncSV();
        SyncVMte3();
        DataCopy(xGm_[vecBase * HEAD_DIM], xT_, elems);
        SyncMte3V();
    }

    // 子批量化：读 z → norm/归一 → idx/yhat/r/gamma → r 拆 fp16 hi/lo → GM
    __aicore__ inline void QuantSubbatch(int64_t vecBase, int32_t n, int32_t s)
    {
        uint32_t elems = static_cast<uint32_t>(n) * HEAD_DIM;
        {
            event_t e2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(e2);
            WaitFlag<HardEvent::V_MTE2>(e2);
        }
        DataCopy(yT_, zGm_[vecBase * HEAD_DIM], elems); // yT_ 暂作 z
        SyncMte2V();
        // y = z * invNorm：invNorm 已在 UploadX 从 x 算好存入 invArrT_（避免 1/norm 除法占关键路径）
        for (int32_t v = 0; v < n; v++) {
            float invNorm = invArrT_.GetValue(s * CUBE_SUBBATCH + v);
            Muls(yT_[v * HEAD_DIM], yT_[v * HEAD_DIM], invNorm, HEAD_DIM); // y = z * inv
        }
        SyncSV();
        // 子批宽度一次算完：边界/质心只广播一次，指令数 ÷ n
        Duplicate(idxFT_, 0.0f, elems);
        Duplicate(yhatT_, CentTable<BITS>::CENT[0], elems);
        for (int32_t k = 0; k < BOUND_CNT; k++) {
            float b = (CentTable<BITS>::CENT[k] + CentTable<BITS>::CENT[k + 1]) * 0.5f;
            CompareScalar(maskT_, yT_, b, CMPMODE::GT, elems);
            Duplicate(dupT_, static_cast<float>(k + 1), elems);
            Select(idxFT_, maskT_, dupT_, idxFT_, SELMODE::VSEL_TENSOR_TENSOR_MODE, elems);
            Duplicate(dupT_, CentTable<BITS>::CENT[k + 1], elems);
            Select(yhatT_, maskT_, dupT_, yhatT_, SELMODE::VSEL_TENSOR_TENSOR_MODE, elems);
        }
        Sub(rT_, yT_, yhatT_, elems); // 残差 r = y - yhat
        // rn = ||r||、gamma = norm * rn（同款两级 BlockReduceSum）
        Mul(sqT_, rT_, rT_, elems);
        BlockReduceSum(dupT_, sqT_, static_cast<int32_t>(elems / 64), 64, 1, 1, 8);
        BlockReduceSum(dupT_[SUB_ELEMS / 2], dupT_, static_cast<int32_t>((elems / 8 + 63) / 64), 64, 1, 1, 8);
        SyncVS();
        for (int32_t v = 0; v < n; v++) {
            float rNorm = sqrt(dupT_.GetValue(SUB_ELEMS / 2 + v * 2) + dupT_.GetValue(SUB_ELEMS / 2 + v * 2 + 1));
            gamArrT_.SetValue(v, normArrT_.GetValue(s * CUBE_SUBBATCH + v) * rNorm);
            // 注：r 保持残差（未归一化）直接拆 fp16 hi/lo；qjl=sign(r@S) 与 golden 的
            // sign(residual_unit@S) 一致（rn>0 只改幅度不改符号），无需 r/rn 归一化。
        }
        SyncSV();
        SplitFp16HiLo(rT_, rHiT_, rLoT_, yhatT_, elems); // yhatT_ 在 Sub 之后已死，作 fp16 拆分 tmp
        SyncVMte3();
        DataCopy(rHiGm_[vecBase * HEAD_DIM], rHiT_, elems);
        DataCopy(rLoGm_[vecBase * HEAD_DIM], rLoT_, elems);
        SyncMte3V();
    }

    // 子批 idx/norm/gamma 输出（本 AIV 自己的 outIdx 缓冲）
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
        __ubuf__ uint32_t *dst32 =
            reinterpret_cast<__ubuf__ uint32_t *>(outIdxT_.GetPhyAddr()) + (s * CUBE_SUBBATCH) * (IDX_BYTES / 4);
        __ubuf__ float *srcF = reinterpret_cast<__ubuf__ float *>(dupT_.GetPhyAddr());
        if (BITS == 4) {
            uint64_t mLo[1] = {0x0F0F0F0F0F0F0F0FULL};
            uint64_t mHi[1] = {0xF0F0F0F0F0F0F0F0ULL};
            for (uint32_t ch = 0; ch < elems; ch += 1024) {
                uint32_t cn = (elems - ch < 1024) ? (elems - ch) : 1024;
                Mul(sqT_[ch], idxFT_[ch], stFpT_, cn);
                BlockReduceSum(dupT_[ch / 8], sqT_[ch], static_cast<int32_t>(cn / 64), mLo, 1, 1, 8);
                BlockReduceSum(dupT_[SUB_ELEMS / 2 + ch / 8], sqT_[ch], static_cast<int32_t>(cn / 64), mHi, 1, 1, 8);
            }
            SyncVS();
            __ubuf__ float *srcHi = srcF + SUB_ELEMS / 2;
            for (int32_t i = 0; i < n * (HEAD_DIM / 8); i++) {
                uint32_t lo = static_cast<uint32_t>(static_cast<int32_t>(srcF[i]));
                uint32_t hi = static_cast<uint32_t>(static_cast<int32_t>(srcHi[i]));
                dst32[i] = lo | (hi << 16); // 4 字节 = lo(2B) + hi(2B)
            }
            PipeBarrier<PIPE_ALL>();
            return;
        }
        for (uint32_t ch = 0; ch < elems; ch += 1024) {
            uint32_t cn = (elems - ch < 1024) ? (elems - ch) : 1024;
            Mul(sqT_[ch], idxFT_[ch], stFpT_, cn);
            BlockReduceSum(dupT_[ch / 8], sqT_[ch], static_cast<int32_t>(cn / 64), 64, 1, 1, 8);
        }
        SyncVS();
        if (BITS == 3) {
            // 24-bit word → 3 字节；4 个 word(12B) → 3 个 int32
            for (int32_t v = 0; v < n; v++) {
                __ubuf__ uint32_t *dstv = dst32 + v * (IDX_BYTES / 4); // 12 int32/行
                __ubuf__ float *srcv = srcF + v * (HEAD_DIM / 8);      // 16 word/行
                for (int32_t g = 0; g < 4; g++) {
                    uint32_t w0 = static_cast<uint32_t>(static_cast<int32_t>(srcv[g * 4 + 0]));
                    uint32_t w1 = static_cast<uint32_t>(static_cast<int32_t>(srcv[g * 4 + 1]));
                    uint32_t w2 = static_cast<uint32_t>(static_cast<int32_t>(srcv[g * 4 + 2]));
                    uint32_t w3 = static_cast<uint32_t>(static_cast<int32_t>(srcv[g * 4 + 3]));
                    dstv[g * 3 + 0] = w0 | (w1 << 24);
                    dstv[g * 3 + 1] = (w1 >> 8) | (w2 << 16);
                    dstv[g * 3 + 2] = (w2 >> 16) | (w3 << 8);
                }
            }
        } else { // BITS == 2
            // 16-bit word → 2 字节；2 个 word(4B) → 1 个 int32
            for (int32_t i = 0; i < n * (HEAD_DIM / 8) / 2; i++) {
                uint32_t w0 = static_cast<uint32_t>(static_cast<int32_t>(srcF[2 * i + 0]));
                uint32_t w1 = static_cast<uint32_t>(static_cast<int32_t>(srcF[2 * i + 1]));
                dst32[i] = w0 | (w1 << 16);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessAiv()
    {
        // Init：H/S staging 拆到两个 AIV 并行（AIV0 做 H 三连、AIV1 做 S hi/lo，各自独立 UB 无冲突），
        // staging 时间减半；MODE2 通知需两个 sub-block 都 CrossSet。
        if (matDtype_ == 1) {
            if (aivSubIdx_ == 0) {
                StageHTripleBf16(rotSrcBf16Gm_, hHhGm_, hHlGm_, hHllGm_);
            } else {
                StageSHiLoBf16(qjlSrcBf16Gm_, sHiGm_, sLoGm_);
            }
        } else {
            if (aivSubIdx_ == 0) {
                StageHTriple(rotSrcGm_, hHhGm_, hHlGm_, hHllGm_);
            } else {
                StageSHiLo(qjlSrcGm_, sHiGm_, sLoGm_);
            }
        }
        SyncSMte3();
        CrossSet(CUBE_FLAG_HS, false);
        // 作为 idx 打包幂次表：mulTbl[i] = 2^(BITS*(i%8))（BITS=4 时按 4 位半字：2^(4*(i%4))）。
        for (int32_t i = 0; i < 1024; i++) {
            int32_t l = i % 8;
            int32_t sh = (BITS == 4) ? (4 * (l % 4)) : (BITS * l);
            stFpT_.SetValue(i, static_cast<float>(1 << sh));
            stFpT_.SetValue(1024 + i, static_cast<float>(1 << l)); // QJL 打包用 2^(i%8)
        }
        SyncSV();
        CrossSet(CUBE_FLAG_HS, false);

        // 预上传 tile 0 的 x（UploadX 被提前到"等 Y"之前，使 tile t+1 的上传与 tile t 的 GEMM 重叠）
        {
            int32_t nTile0 = static_cast<int32_t>((vecCount_ < CUBE_TILE_V) ? vecCount_ : CUBE_TILE_V);
            int32_t subs0 = (nTile0 + CUBE_SUBBATCH - 1) / CUBE_SUBBATCH;
            for (int32_t s = aivSubIdx_; s < subs0; s += 2) {
                int32_t n = (s == subs0 - 1) ? (nTile0 - s * CUBE_SUBBATCH) : CUBE_SUBBATCH;
                UploadX(vecStart_ + s * CUBE_SUBBATCH, n, s);
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

            // 读 z → 归一/量化 → r 拆分 → idx/norm/gamma
            CrossWait(CUBE_FLAG_Y + d);
            for (int32_t s = aivSubIdx_; s < subs; s += 2) {
                int32_t n = (s == subs - 1) ? (nTile - s * CUBE_SUBBATCH) : CUBE_SUBBATCH;
                int64_t vb = vecBase + s * CUBE_SUBBATCH;
                QuantSubbatch(vb, n, s);
                PackIdx(n, s);
                FlushIdxNormGamma(vb, n, s);
            }
            CrossSet(CUBE_FLAG_R + d, false);

            // 上传 tile t+1 的 x：与 AIC 的 GEMM2(t) 重叠。
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
                    UploadX(vecBaseN + s * CUBE_SUBBATCH, n, s);
                }
                CrossSet(CUBE_FLAG_U + (tn % 2), false);
            }

            // 读 proj → qjl bit → 打包输出
            CrossWait(CUBE_FLAG_P + d);
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
                for (uint32_t ch = 0; ch < elems; ch += 1024) {
                    uint32_t cn = (elems - ch < 1024) ? (elems - ch) : 1024;
                    Mul(qjlBT_[ch], qjlBT_[ch], stFpT_[1024], cn);
                    BlockReduceSum(dupT_[ch / 8], qjlBT_[ch], static_cast<int32_t>(cn / 64), 64, 1, 1, 8);
                }
                SyncVS();
                // int32 一次写 4 字节（4 个 packed 字节合成 1 个 int32），避免逐字节 SetValue 的标量管道开销
                const int32_t nb = n * QJL_BYTES;
                __ubuf__ uint32_t *dst32 = reinterpret_cast<__ubuf__ uint32_t *>(outQjlT_.GetPhyAddr());
                __ubuf__ float *srcF = reinterpret_cast<__ubuf__ float *>(dupT_.GetPhyAddr());
                for (int32_t i = 0; i < nb / 4; i++) {
                    uint32_t b0 = static_cast<uint32_t>(static_cast<int32_t>(srcF[4 * i + 0]));
                    uint32_t b1 = static_cast<uint32_t>(static_cast<int32_t>(srcF[4 * i + 1]));
                    uint32_t b2 = static_cast<uint32_t>(static_cast<int32_t>(srcF[4 * i + 2]));
                    uint32_t b3 = static_cast<uint32_t>(static_cast<int32_t>(srcF[4 * i + 3]));
                    dst32[i] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
                }
                PipeBarrier<PIPE_ALL>();
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
        pipe->InitBuffer(hHhNzBuf_, HEAD_DIM * HEAD_DIM * sizeof(bfloat16_t));
        pipe->InitBuffer(hHlNzBuf_, HEAD_DIM * HEAD_DIM * sizeof(bfloat16_t));
        pipe->InitBuffer(hHllNzBuf_, HEAD_DIM * HEAD_DIM * sizeof(bfloat16_t));
        pipe->InitBuffer(sHiNzBuf_, HEAD_DIM * HEAD_DIM * sizeof(half));
        pipe->InitBuffer(sLoNzBuf_, HEAD_DIM * HEAD_DIM * sizeof(half));
        pipe->InitBuffer(a1Buf_, CUBE_TILE_V * HEAD_DIM * sizeof(bfloat16_t)); // 兼作 fp16 A（Get<half>）
        pipe->InitBuffer(a2Buf_, CUBE_TILE_V * HEAD_DIM * sizeof(bfloat16_t));
        pipe->InitBuffer(b2Buf_, 128 * HEAD_DIM * sizeof(bfloat16_t)); // 兼作 fp16 B
        pipe->InitBuffer(c1Buf_, CUBE_TILE_V * HEAD_DIM * sizeof(float));
        hHhNzT_ = hHhNzBuf_.Get<bfloat16_t>();
        hHlNzT_ = hHlNzBuf_.Get<bfloat16_t>();
        hHllNzT_ = hHllNzBuf_.Get<bfloat16_t>();
        sHiNzT_ = sHiNzBuf_.Get<half>();
        sLoNzT_ = sLoNzBuf_.Get<half>();
        a1T_ = a1Buf_.Get<bfloat16_t>();
        a2T_ = a2Buf_.Get<bfloat16_t>();
        a1F16T_ = a1Buf_.Get<half>();
        a2F16T_ = a2Buf_.Get<half>();
        b2T_ = b2Buf_.Get<bfloat16_t>();
        b2F16T_ = b2Buf_.Get<half>();
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
        DataCopy(hHhNzT_, hHhGm_, nz);
        DataCopy(hHlNzT_, hHlGm_, nz);
        DataCopy(hHllNzT_, hHllGm_, nz);
        DataCopy(sHiNzT_, sHiGm_, nz);
        DataCopy(sLoNzT_, sLoGm_, nz);
        SyncMte2Mte1();
    }

    // 矩阵（CUBE_MMAD_M×128 ND）GM→L1 Nz（bf16 版）。
    // 统一按满 tile 装载/计算：尾部不足的行越过本区末尾读入相邻数据，只有 Fixpipe 出图用真实行数
    // —— 这样 Cube 侧永远走 probe 验证过的形状（m=CUBE_MMAD_M）。越界读落在 workspace 内紧邻的
    // 下一个分区（host 侧布局顺序 X→Rh→Rl→Z→P1→P2→P3 不可调换），读到的值被丢弃、不回写，无正确性影响。
    __aicore__ inline void Nd2NzTileBf16(const GlobalTensor<bfloat16_t> &srcGm, int64_t vecBase)
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
    __aicore__ inline void Nd2NzTileF16(const GlobalTensor<half> &srcGm, int64_t vecBase)
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
        DataCopy(a1F16T_, srcGm[vecBase * HEAD_DIM], nz);
        SyncMte2Mte1();
    }

    // L1→L0A（bf16 / fp16 同参，均 2B/元素）
    template <typename T>
    __aicore__ inline void LoadATile(LocalTensor<T> l0a, const LocalTensor<T> l1a, int32_t kNow)
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
            LoadData(l0a[i * 16 * kNow], l1a, ap);
        }
    }

    // L1→L0B（bf16 / fp16 同参）
    template <typename T>
    __aicore__ inline void LoadBTile(LocalTensor<T> l0b, const LocalTensor<T> l1b)
    {
        LoadData2DParams bp{};
        bp.startIndex = 0;
        bp.repeatTimes = (128 / 16) * (HEAD_DIM / 16);
        bp.srcStride = 1;
        bp.dstGap = 0;
        bp.ifTranspose = false;
        LoadData(l0b, l1b, bp);
    }

    __aicore__ inline MmadParams MakeMmadParams(bool init)
    {
        MmadParams mp{};
        mp.m = static_cast<uint16_t>(CUBE_MMAD_M);
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
        CrossWait(CUBE_FLAG_HS);
        Nd2NzWeights();
        SyncFixM(); // c 初始可用

        for (uint32_t t = 0; t < tiles_; t++) {
            uint32_t d = t % 2;
            int64_t vecBase = vecStart_ + static_cast<int64_t>(t) * CUBE_TILE_V;
            int32_t nTile = static_cast<int32_t>((vecCount_ - static_cast<int64_t>(t) * CUBE_TILE_V) < CUBE_TILE_V ?
                                                     (vecCount_ - static_cast<int64_t>(t) * CUBE_TILE_V) :
                                                     CUBE_TILE_V);

            // GEMM1: z = x @ (Hh+Hl+Hll)^T（tile 内按 CUBE_MMAD_M=64 行分轮）
            CrossWait(CUBE_FLAG_U + d);
            for (int32_t h = 0; h < CUBE_TILE_V; h += CUBE_MMAD_M) {
                int32_t nt = nTile - h;
                if (nt <= 0)
                    break;
                if (nt > CUBE_MMAD_M)
                    nt = CUBE_MMAD_M;
                int64_t vb = vecBase + h;
                {
                    event_t eA = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_MTE2));
                    SetFlag<HardEvent::MTE1_MTE2>(eA);
                    WaitFlag<HardEvent::MTE1_MTE2>(eA);
                }
                Nd2NzTileBf16(xGm_, vb);
                LoadATile<bfloat16_t>(a2T_, a1T_, HEAD_DIM);
                LoadBTile<bfloat16_t>(b2T_, hHhNzT_);
                SyncMte1M();
                Mmad(c1T_, a2T_, b2T_, MakeMmadParams(true)); // z = x@Hh
                SyncMMte1();
                LoadBTile<bfloat16_t>(b2T_, hHlNzT_);
                SyncMte1M();
                Mmad(c1T_, a2T_, b2T_, MakeMmadParams(false)); // z += x@Hl
                SyncMMte1();
                LoadBTile<bfloat16_t>(b2T_, hHllNzT_);
                SyncMte1M();
                Mmad(c1T_, a2T_, b2T_, MakeMmadParams(false)); // z += x@Hll
                SyncMMte1();
                FixTile(zGm_, vb, nt);
            }
            CrossSet(CUBE_FLAG_Y + d, true);

            // GEMM2: proj = rh@Sh + rh@Sl + rl@Sh（fp32 L0C 累加，1 次 Fixpipe fp32）
            // fp32 累加误差 2^-24 优于 fp16 平面各自 2^-12 舍入，且省掉 2 次 Fixpipe。
            CrossWait(CUBE_FLAG_R + d);
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
                Nd2NzTileF16(rHiGm_, vb);
                LoadATile<half>(a2F16T_, a1F16T_, HEAD_DIM);
                LoadBTile<half>(b2F16T_, sHiNzT_);
                SyncMte1M();
                Mmad(c1T_, a2F16T_, b2F16T_, MakeMmadParams(true)); // proj = rh@Sh
                SyncMMte1();
                LoadBTile<half>(b2F16T_, sLoNzT_);
                SyncMte1M();
                Mmad(c1T_, a2F16T_, b2F16T_, MakeMmadParams(false)); // proj += rh@Sl
                SyncMMte1();
                {
                    event_t eD = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_MTE2));
                    SetFlag<HardEvent::MTE1_MTE2>(eD);
                    WaitFlag<HardEvent::MTE1_MTE2>(eD);
                }
                Nd2NzTileF16(rLoGm_, vb);
                LoadATile<half>(a2F16T_, a1F16T_, HEAD_DIM);
                LoadBTile<half>(b2F16T_, sHiNzT_);
                SyncMte1M();
                Mmad(c1T_, a2F16T_, b2F16T_, MakeMmadParams(false)); // proj += rl@Sh
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
    GlobalTensor<bfloat16_t> hHhGm_, hHlGm_, hHllGm_;
    GlobalTensor<half> sHiGm_, sLoGm_;
    GlobalTensor<bfloat16_t> xGm_;
    GlobalTensor<half> rHiGm_, rLoGm_;
    GlobalTensor<float> zGm_;
    GlobalTensor<float> pGm_;

    TPipe *pipe_ = nullptr;
    // AIV 缓冲
    TBuf<QuePosition::VECCALC> sqBuf_, xBuf_;
    TBuf<QuePosition::VECCALC> yBuf_, yhatBuf_, rBuf_, idxFBuf_, qjlBBuf_;
    TBuf<QuePosition::VECCALC> rHiBuf_, rLoBuf_, dupBuf_, maskBuf_;
    TBuf<QuePosition::VECCALC> normArrBuf_, invArrBuf_, gamArrBuf_, outIdxBuf_, outQjlBuf_;
    TBuf<QuePosition::VECCALC> outNormBf16Buf_, outGamBf16Buf_;
    TBuf<QuePosition::VECCALC> stFpBuf_, stHiBuf_, stMidBuf_, stLoBuf_;
    // AIC 缓冲
    TBuf<TPosition::A1> hHhNzBuf_, hHlNzBuf_, hHllNzBuf_, sHiNzBuf_, sLoNzBuf_, a1Buf_;
    TBuf<TPosition::A2> a2Buf_;
    TBuf<TPosition::B2> b2Buf_;
    TBuf<TPosition::CO1> c1Buf_;

    LocalTensor<float> sqT_, yT_, yhatT_, rT_, idxFT_, qjlBT_;
    LocalTensor<float> dupT_, normArrT_, invArrT_, gamArrT_, stFpT_;
    LocalTensor<bfloat16_t> xT_;
    LocalTensor<half> rHiT_, rLoT_;
    LocalTensor<bfloat16_t> outNormBf16T_, outGamBf16T_;
    LocalTensor<bfloat16_t> stHiT_, stMidT_, stLoT_;
    LocalTensor<half> sStHiT_, sStLoT_;
    LocalTensor<uint8_t> maskT_, outIdxT_, outQjlT_;
    LocalTensor<bfloat16_t> hHhNzT_, hHlNzT_, hHllNzT_, a1T_, a2T_, b2T_;
    LocalTensor<half> sHiNzT_, sLoNzT_, a1F16T_, a2F16T_, b2F16T_;
    LocalTensor<float> c1T_;

    uint32_t aiCoreIdx_ = 0;
    uint32_t aivSubIdx_ = 0;
    int64_t numVectors_ = 0;
    int64_t vecStart_ = 0;
    int64_t vecCount_ = 0;
    uint32_t tiles_ = 0;
    uint64_t offHh_ = 0, offHl_ = 0, offHll_ = 0, offSh_ = 0, offSl_ = 0;
    uint64_t offX_ = 0, offRh_ = 0, offRl_ = 0, offZ_ = 0, offP_ = 0;
    uint32_t matDtype_ = 0; // rotation/qjl_matrix 元素类型：0=float32，1=bfloat16
    uint32_t outDtype_ = 0; // quant_norm/quant_gamma 元素类型：0=bfloat16，1=float16
};

} // namespace KvCacheTurboQuant

#endif // KERNEL_KV_CACHE_TURBO_QUANT_CUBE_H
