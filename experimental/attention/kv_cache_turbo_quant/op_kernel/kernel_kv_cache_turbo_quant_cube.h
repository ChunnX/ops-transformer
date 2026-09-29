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
 *   4. y 量化得 idx、残差 r=y-yhat；r 先归一化（r_hat=r/rn，rn>0 不改变 qjl 符号）再拆 fp16 hi/lo（rh/rl），
 *      归一化到 ~1 量级避免小残差的 fp16 denormal 丢精度；
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
        // fp32 矩阵 + 大 case：tie 重算走 H^T/S^T 预转置 + j 维 128 路并行 FMA 的向量化路径；
        // 小 case（tie 极少）保留标量重算，不付预转置开销。
        // STAGE B（2026-09-29）：级联一律走标量重算（自然布局 H/S，无预转置）——外科层已把
        // 级联压到 ~1-4 个/case，标量重算足够；向量化重算+转置（~130µs）不再划算。
        useVecRc_ = false;

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
        offHT_ = off;
        off += 2 * HEAD_DIM * HEAD_DIM * 4; // fp32 H^T，AIV0/AIV1 各一份（免跨 sub-block 同步）
        offST_ = off;
        off += 2 * HEAD_DIM * HEAD_DIM * 4; // fp32 S^T，同上
        offX_ = off;
        off += numVectors_ * HEAD_DIM * 2; // bf16
        offRh_ = off;
        off += numVectors_ * HEAD_DIM * 2; // fp16
        offRl_ = off;
        off += numVectors_ * HEAD_DIM * 2;
        offZ_ = off;
        off += numVectors_ * HEAD_DIM * 4; // fp32
        offP1_ = off;
        off += numVectors_ * HEAD_DIM * 4; // fp32（B11：p 平面改 fp32 输出，误差 ~1e-5→~5e-7）
        offP2_ = off;
        off += numVectors_ * HEAD_DIM * 4;
        offP3_ = off;
        off += numVectors_ * HEAD_DIM * 4;

        hHhGm_.SetGlobalBuffer((__gm__ bfloat16_t *)(wsBase + offHh_));
        hHlGm_.SetGlobalBuffer((__gm__ bfloat16_t *)(wsBase + offHl_));
        hHllGm_.SetGlobalBuffer((__gm__ bfloat16_t *)(wsBase + offHll_));
        sHiGm_.SetGlobalBuffer((__gm__ half *)(wsBase + offSh_));
        sLoGm_.SetGlobalBuffer((__gm__ half *)(wsBase + offSl_));
        htGm_.SetGlobalBuffer((__gm__ float *)(wsBase + offHT_ + static_cast<uint64_t>(aivSubIdx_) * HEAD_DIM * HEAD_DIM * 4));
        stGm_.SetGlobalBuffer((__gm__ float *)(wsBase + offST_ + static_cast<uint64_t>(aivSubIdx_) * HEAD_DIM * HEAD_DIM * 4));
        xGm_.SetGlobalBuffer((__gm__ bfloat16_t *)(wsBase + offX_));
        rHiGm_.SetGlobalBuffer((__gm__ half *)(wsBase + offRh_));
        rLoGm_.SetGlobalBuffer((__gm__ half *)(wsBase + offRl_));
        zGm_.SetGlobalBuffer((__gm__ float *)(wsBase + offZ_));
        p1Gm_.SetGlobalBuffer((__gm__ float *)(wsBase + offP1_));
        p2Gm_.SetGlobalBuffer((__gm__ float *)(wsBase + offP2_));
        p3Gm_.SetGlobalBuffer((__gm__ float *)(wsBase + offP3_));
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
    __aicore__ inline void SyncMte3S()
    {
        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(e);
        WaitFlag<HardEvent::MTE3_S>(e);
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
        pipe->InitBuffer(tieQjlBuf_, CUBE_TILE_V * QJL_BYTES);
        pipe->InitBuffer(outNormBf16Buf_, CUBE_TILE_V * sizeof(bfloat16_t));
        pipe->InitBuffer(outGamBf16Buf_, CUBE_SUBBATCH * sizeof(bfloat16_t));
        // H/S staging 分块缓冲（16 行 = 8KB fp32；H 三连用 stHi/stMid/stLo，S 用 stHi/stLo Get<half>）
        pipe->InitBuffer(stFpBuf_, 16 * HEAD_DIM * sizeof(float));
        // 顺延外科暂存（全 UB）：y 侧 gv/nh/fl + hits[8]+idxv[8]（19×128×4B=9.7KB）；
        // gamma gv + 整行 idx（1032×4B=4.1KB）；qjl gv/nh/fl+hits[8]+行16B（15×192×4B=11.5KB）
        pipe->InitBuffer(stashBuf_,
                         (19 * STASH_YCAP + (1 + HEAD_DIM) * STASH_GCAP + 15 * STASH_QCAP) * sizeof(int32_t));
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
        tieQjlT_ = tieQjlBuf_.Get<uint8_t>();
        outNormBf16T_ = outNormBf16Buf_.Get<bfloat16_t>();
        outGamBf16T_ = outGamBf16Buf_.Get<bfloat16_t>();
        stFpT_ = stFpBuf_.Get<float>();
        stHiT_ = stHiBuf_.Get<bfloat16_t>();
        stMidT_ = stMidBuf_.Get<bfloat16_t>();
        stLoT_ = stLoBuf_.Get<bfloat16_t>();
        {
            auto sa = stashBuf_.Get<int32_t>();
            stashYGv_ = sa;
            stashYNh_ = sa[STASH_YCAP];
            stashYFl_ = sa[2 * STASH_YCAP];
            stashYHit_ = sa[3 * STASH_YCAP];
            stashYIv_ = sa[11 * STASH_YCAP];
            auto ga = sa[19 * STASH_YCAP];
            gamGv_ = ga;
            gamIdx_ = ga[STASH_GCAP];
            auto qa = ga[(1 + HEAD_DIM) * STASH_GCAP];
            stashQGv_ = qa;
            stashQNh_ = qa[STASH_QCAP];
            stashQFl_ = qa[2 * STASH_QCAP];
            stashQHit_ = qa[3 * STASH_QCAP];
            stashQRow_ = qa[11 * STASH_QCAP];
            stashYC_ = 0;
            stashQC_ = 0;
            gamYC_ = 0;
        }
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

    // fp32 [128,128] 行主序 → 转置行主序写 GM：16 行一块读入 yhatT_，标量转成 [k][16] 布局
    // （rT_/dupT_ 双缓冲交替，隐藏 MTE3），再一次 DataCopyPad（128 个 64B 块、dst 跨行 448B）写出。
    // 供重算 y=unit@H / qjl=r_hat@S 改走 H^T/S^T 的 j 维 128 路并行 FMA（k 升序与 golden 逐位一致）。
    __aicore__ inline void StageTranspose(const GlobalTensor<float> &srcGm, const GlobalTensor<float> &dstGm)
    {
        const int32_t D = HEAD_DIM;
        for (int32_t rb = 0; rb < D; rb += 16) {
            {
                event_t e0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(e0);
                WaitFlag<HardEvent::V_MTE2>(e0);
            }
            DataCopy(yhatT_, srcGm[rb * D], 16 * D);
            SyncMte2V();
            SyncMte3S(); // 上一同构块的 DataCopyPad 可能还在读同一 rT_/dupT_
            if ((rb / 16) % 2 == 0) {
                for (int32_t k = 0; k < D; k++) {
                    for (int32_t jj = 0; jj < 16; jj++) {
                        rT_.SetValue(k * 16 + jj, yhatT_.GetValue(jj * D + k));
                    }
                }
                SyncSMte3();
                DataCopyExtParams p;
                p.blockCount = D;
                p.blockLen = 16 * sizeof(float); // 64B
                p.srcStride = 0;
                p.dstStride = (D - 16) * sizeof(float);
                DataCopyPad(dstGm[rb], rT_, p);
            } else {
                for (int32_t k = 0; k < D; k++) {
                    for (int32_t jj = 0; jj < 16; jj++) {
                        dupT_.SetValue(k * 16 + jj, yhatT_.GetValue(jj * D + k));
                    }
                }
                SyncSMte3();
                DataCopyExtParams p;
                p.blockCount = D;
                p.blockLen = 16 * sizeof(float);
                p.srcStride = 0;
                p.dstStride = (D - 16) * sizeof(float);
                DataCopyPad(dstGm[rb], dupT_, p);
            }
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
        Mul(sqT_, yT_, yT_, elems);                  // sq = x*x
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
        // rn = ||r||、gamma = norm * rn（两级 BlockReduceSum）
        Mul(sqT_, rT_, rT_, elems);
        BlockReduceSum(dupT_, sqT_, static_cast<int32_t>(elems / 64), 64, 1, 1, 8);
        BlockReduceSum(dupT_[SUB_ELEMS / 2], dupT_, static_cast<int32_t>((elems / 8 + 63) / 64), 64, 1, 1, 8);
        SyncVS();
        for (int32_t v = 0; v < n; v++) {
            float rNorm = sqrt(dupT_.GetValue(SUB_ELEMS / 2 + v * 2) + dupT_.GetValue(SUB_ELEMS / 2 + v * 2 + 1));
            gamArrT_.SetValue(v, normArrT_.GetValue(s * CUBE_SUBBATCH + v) * rNorm);
            // 复用 invArrT_ 存 invR（1/rn，clamp 1e-30），供下方 r_hat 归一化
            invArrT_.SetValue(s * CUBE_SUBBATCH + v, 1.0f / (rNorm > 1e-30f ? rNorm : 1e-30f));
        }
        SyncSV();
        // r_hat = r * invR：归一化后再拆 fp16 hi/lo（照 golden 的 residual_unit），
        // 避免小残差的 lo 分量落入 fp16 denormal 丢精度 —— 这是 fp16 三平面追平 golden 的关键。
        for (int32_t v = 0; v < n; v++) {
            float invR = invArrT_.GetValue(s * CUBE_SUBBATCH + v);
            Muls(rT_[v * HEAD_DIM], rT_[v * HEAD_DIM], invR, HEAD_DIM);
        }
        SyncSV();
        SplitFp16HiLo(rT_, rHiT_, rLoT_, yhatT_, elems); // yhatT_ 在 Sub 之后已死，作 fp16 拆分 tmp
        SyncVMte3();
        DataCopy(rHiGm_[vecBase * HEAD_DIM], rHiT_, elems);
        DataCopy(rLoGm_[vecBase * HEAD_DIM], rLoT_, elems);
        SyncMte3V();
    }

    // 从检测 Compare 的位图（maskT_，bit i = 元素 i）收集向量 v 的命中元素下标（≤cap），
    // 返回命中数（可超 cap，由调用方判级联）。比逐元素 GetValue(0/1 浮点) 快 ~8×。
    __aicore__ inline int32_t CollectHitsFromMask(int32_t v, int32_t *out, int32_t cap)
    {
        __ubuf__ uint8_t *mb = reinterpret_cast<__ubuf__ uint8_t *>(maskT_.GetPhyAddr()) + v * (HEAD_DIM / 8);
        int32_t nh = 0;
        for (int32_t b = 0; b < HEAD_DIM / 8; b++) {
            uint32_t byte = mb[b];
            while (byte != 0) {
                int32_t lane = 0;
                while ((byte & 1u) == 0u) {
                    byte >>= 1;
                    lane++;
                }
                if (nh < cap) {
                    out[nh] = b * 8 + lane;
                }
                nh++;
                byte >>= 1;
            }
        }
        return nh;
    }

    // tie 检测 + 精确重算：必须在 QuantSubbatch 之后、PackIdx 之前调用——读 yT_/idxFT_（Cube 的
    // y/idx，QuantSubbatch 留下），对 tie 向量重算后把 idx 写回 idxFT_ 局部槽位、norm/gamma 写回
    // normArrT_/gamArrT_、qjl 位写 tieQjlT_，全部经 PackIdx/FlushIdxNormGamma/qjl 覆盖统一写出，
    // 避免任何标量 SetValue 直接写 GM（会因 32B 粒度/延迟 flush 冲掉相邻向量的 norm）。
    __aicore__ inline void DetectAndRecomputeTies(int64_t vecBase, int32_t n, int32_t s)
    {
        uint32_t elems = static_cast<uint32_t>(n) * HEAD_DIM;
        // 向量化 tie 检测：dmin = min_k |y - b_k|（边界单调 ⇒ 全边界 min 与"相邻两边界 min"等价），
        // thr = max(9.54e-7*|y|, 4e-7)（16 ULP 相对 + 绝对下限；maxDy 实测 ≤2.68e-7（ZDUMP），
        // 4e-7 下限保留 1.5× 余量），hit = dmin < thr。
        // C7: 对称边界折叠——质心表位级对称（c[N-1-i]=-c[i]，IEEE 十进制转换/加法/
        // 舍入的符号对称性保证 b[N-2-k]=-b[k] 位级成立、中央边界 b_mid=0 精确）；
        // min(|y-b|,|y+b|) = ||y|-|b||（fp32 舍入符号对称 ⇒ 与原表达式逐位相同）。
        // dmin = min(|y|, min_i ||y|-bp_i||)：BOUND_CNT 轮×3 条向量指令 →
        // 1 条 Abs + (N/2-1) 轮×3 条，dmin 位级不变，检测语义不变。
        const int32_t ncent = BOUND_CNT + 1;
        Abs(dupT_, yT_, elems); // dupT_ = |y|，覆盖中央边界 b_mid = 0
        for (int32_t k = 0; k < ncent / 2 - 1; k++) {
            float b = (CentTable<BITS>::CENT[ncent / 2 + k] +
                       CentTable<BITS>::CENT[ncent / 2 + k + 1]) * 0.5f;
            Adds(yhatT_, dupT_, -b, elems);
            Abs(yhatT_, yhatT_, elems);
            Min(dupT_, dupT_, yhatT_, elems);
        }
        Abs(yhatT_, yT_, elems);
        Muls(yhatT_, yhatT_, 9.54e-7f, elems);
        Maxs(yhatT_, yhatT_, 4.0e-7f, elems);
        Compare<float>(maskT_, dupT_, yhatT_, CMPMODE::LT, elems);
        // mask → 0/1 → 两级 BlockReduceSum 聚合成每向量命中计数（rT_/qjlBT_ 此刻均已死，作暂存）
        Duplicate(rT_, 1.0f, elems);
        Duplicate(qjlBT_, 0.0f, elems);
        Select(qjlBT_, maskT_, rT_, qjlBT_, SELMODE::VSEL_TENSOR_TENSOR_MODE, elems);
        BlockReduceSum(sqT_, qjlBT_, static_cast<int32_t>(elems / 64), 64, 1, 1, 8);
        BlockReduceSum(sqT_[SUB_ELEMS / 2], sqT_, static_cast<int32_t>((elems / 8 + 63) / 64), 64, 1, 1, 8);
        auto yh32 = yhatT_.ReinterpretCast<int32_t>();
        auto gamI = gamArrT_.ReinterpretCast<int32_t>();
        And(yh32[32], gamI, gamI, n); // gamArrT_ 位模式向量拷贝至 int32 视图（消标量 float 类型双关）
        // 写入区 [32,32+n) 是 yhatT_ 死区（thr 向量被 Compare 消费后无人再读），且 yhatBuf_/dupBuf_/
        // gamArrBuf_ 三缓冲独立 InitBuffer 互不重叠——32 为任意安全暂存起点，无对齐/别名约束
        SyncVS();
        for (int32_t v = 0; v < n; v++) {
            bool tie = (sqT_.GetValue(SUB_ELEMS / 2 + v * 2) + sqT_.GetValue(SUB_ELEMS / 2 + v * 2 + 1)) > 0.5f;
            // gamma 落在 bf16/fp16 舍入边界附近（Cube 的 rnorm 归约序 vs golden tree_dot 差 1 ULP
            // 会跨边界翻转舍入）：也标记为 tie，重算用 tree_dot 精确复现 golden gamma。
            // C9e: 位模式经 And 向量拷贝至 yh32，标量端从 int32 视图读取解码——
            // 原直读 gamArrT_ 的 float→uint32 union 双关在标量端生成内存往返，
            // 2048 行 × 64 子批串行 ~57µs（stub17/17b 归因）
            if (!tie) {
                int32_t u = yh32.GetValue(32 + v);
                int32_t gl = (u & 0xFFFF) - 0x8000;
                if (gl < 8 && gl > -8) {
                    tie = true;
                }
            }
            tieVec_[s * CUBE_SUBBATCH + v] = tie;
        }
        SyncSV();
        // 顺延外科（deferred epilogue）：候选收集口径与旧外科一致（命中 ≤3e-7 收集、
        // (3e-7,4e-7) 由 maxDy≤2.68e-7 免检、gamma 边界标记、nraw>16/nh>8 直接全链），
        // 但只登记到本核 UB 暂存——主 tile 流水零停顿（实测内联外科每个含候选 tile 摧毁
        // 四级流水重叠，占总开销 124~219µs）；验证/重算集中到 ProcessAiv 尾声。
        // cube idx 值随命中点一并 stash（标量读 idxFT_），gamma 候选整行 128 值入 gamma 暂存。
        // 任何暂存溢出（病态输入）回退就地全链级联（语义同旧级联）。
        int32_t tmpHits[8];
        int32_t tmpIdx[8];
        int32_t ovY_[CUBE_SUBBATCH]; // 溢出兜底延后到收集循环后（RecomputeTieVec 会 clobber dupT_）
        int32_t nOvY_ = 0;
        for (int32_t v = 0; v < n; v++) {
            cascVec_[s * CUBE_SUBBATCH + v] = false;
            if (!tieVec_[s * CUBE_SUBBATCH + v]) {
                continue;
            }
            // gamma 边界判定（同检测口径）
            // C9f: 与检测循环同源——直接读 And 拷贝好的 yh32 int32 视图解码（位模式相同），
            // 消除 union float 双关在标量端的内存往返；此处为冷路径（仅 tie 候选），取一致性
            int32_t ugs = yh32.GetValue(32 + v);
            int32_t gl = (ugs & 0xFFFF) - 0x8000;
            bool gamFlag = (gl < 8 && gl > -8);
            // 命中收集（位图在 maskT_；dupT_ 此刻仍持 dmin）
            int32_t raw[16];
            int32_t nraw = CollectHitsFromMask(v, raw, 16);
            int32_t nh = 0;
            bool full = (nraw > 16);
            for (int32_t t = 0; t < nraw && t < 16; t++) {
                if (dupT_.GetValue(v * HEAD_DIM + raw[t]) > 3.0e-7f) {
                    continue;
                }
                if (nh < 8) {
                    tmpHits[nh] = raw[t];
                    tmpIdx[nh] = static_cast<int32_t>(idxFT_.GetValue(v * HEAD_DIM + raw[t]));
                }
                nh++;
            }
            if (nh > 8) {
                full = true;
                nh = 8;
            }
            if (nh == 0 && !gamFlag && !full) {
                continue; // 命中全免检且无 gamma 标记：旧外科对此类候选零动作
            }
            if (gamFlag && !full) { // gamma 候选：整行 cube idx 入 gamma 暂存（尾声成批 yExact）
                if (gamYC_ < STASH_GCAP) {
                    gamGv_.SetValue(gamYC_, static_cast<int32_t>(vecBase + v));
                    for (int32_t j = 0; j < HEAD_DIM; j++) {
                        gamIdx_.SetValue(gamYC_ * HEAD_DIM + j,
                                         static_cast<int32_t>(idxFT_.GetValue(v * HEAD_DIM + j)));
                    }
                    gamYC_++;
                } else if (stashYC_ < STASH_YCAP) { // gamma 暂存溢出 → 按 full 走 y 暂存
                    stashYGv_.SetValue(stashYC_, static_cast<int32_t>(vecBase + v));
                    stashYNh_.SetValue(stashYC_, 0);
                    stashYFl_.SetValue(stashYC_, 2);
                    stashYC_++;
                } else {
                    ovY_[nOvY_++] = v;
                }
                continue;
            }
            if (stashYC_ < STASH_YCAP) {
                stashYGv_.SetValue(stashYC_, static_cast<int32_t>(vecBase + v));
                stashYNh_.SetValue(stashYC_, nh);
                stashYFl_.SetValue(stashYC_, full ? 2 : 0);
                for (int32_t t = 0; t < nh; t++) {
                    stashYHit_.SetValue(stashYC_ * 8 + t, tmpHits[t]);
                    stashYIv_.SetValue(stashYC_ * 8 + t, tmpIdx[t]);
                }
                stashYC_++;
            } else {
                ovY_[nOvY_++] = v;
            }
        }
        for (int32_t i = 0; i < nOvY_; i++) { // 收集完成后再级联（dupT_ 的 dmin 已消费完）
            RecomputeTieVec(vecBase + ovY_[i], s * CUBE_SUBBATCH + ovY_[i]);
            cascVec_[s * CUBE_SUBBATCH + ovY_[i]] = true;
        }
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

    // 成批精确 y（≤4 个向量）：unit 行在 dupT_[uRow[gi]*D]，H 分 8 行块双缓冲读入 htF（rHiT_
    // fp32 视图，自然布局），prod 入 sqT_ 32 行（gi*8+e），16 路交织 k 升序求和（2 向量 ×
    // 8 j 行 × 16 累加器），结果写 yhatT_ 行 gi。每个点积仍严格 k 升序（fp32 乘/加舍入序列
    // 与 ComputeGolden 完全一致）；H 行块只读一次，nv 个向量共享，省去 (nv-1)×64KB GM 与同步。
    __aicore__ inline void ExactYBatch(const int32_t *uRow, int32_t nv)
    {
        const int32_t D = HEAD_DIM;
        // 16 行块（双缓冲 2×16）：staging 改 yT_（本函数仅在尾声被调用，yT_ 空闲），
        // 每 16 行一次 MTE2 同步（原 8 行块 ×2）；Mul/sum 仍按 8 行半块（16 路 acc 超寄存器预算）。
        auto htF = yBuf_.Get<float>(); // 4096 floats = 32 行（双缓冲 2×16）
        {
            event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(e);
            WaitFlag<HardEvent::V_MTE2>(e);
        }
        DataCopy(htF, rotSrcGm_[0], 16 * D);
        SyncMte2V();
        if (nv == 1) {
            // 单候选快路：每 16 行块 = 16 Mul + 1 次 SyncVS + 16 链求和（accA=前半 8 行、
            // accB=后半 8 行）+ 1 次 SyncMte2V——标量链交织深度翻倍（v5 8 链 ~2cyc/op 的
            // 主要修正），同步次数减半
            for (int32_t c16 = 0; c16 < D / 16; c16++) {
                int32_t cur = (c16 % 2) * 16 * D;
                if (c16 + 1 < D / 16) {
                    DataCopy(htF[((c16 + 1) % 2) * 16 * D], rotSrcGm_[(c16 + 1) * 16 * D], 16 * D);
                }
                for (int32_t e = 0; e < 16; e++) {
                    Mul(sqT_[e * D], dupT_[uRow[0] * D], htF[cur + e * D], D);
                }
                SyncVS();
                float accA[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                float accB[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                __ubuf__ float *pa = reinterpret_cast<__ubuf__ float *>(sqT_.GetPhyAddr());
                __ubuf__ float *pb2 = pa + 8 * D;
                for (int32_t k = 0; k < D; k++) {
                    for (int32_t t = 0; t < 8; t++) {
                        accA[t] = accA[t] + pa[t * D + k];
                        accB[t] = accB[t] + pb2[t * D + k];
                    }
                }
                for (int32_t t = 0; t < 8; t++) {
                    yhatT_.SetValue(c16 * 16 + t, accA[t]);
                    yhatT_.SetValue(c16 * 16 + 8 + t, accB[t]);
                }
                if (c16 + 1 < D / 16) {
                    SyncMte2V();
                }
            }
            SyncSV();
            return;
        }
        for (int32_t chunk = 0; chunk < D / 8; chunk++) {
            int32_t c16 = chunk / 2;   // 16 行块号
            int32_t half = chunk % 2;  // 块内 8 行半块
            int32_t cur = (c16 % 2) * 16 * D + half * 8 * D;
            if (half == 0 && c16 + 1 < D / 16) {
                DataCopy(htF[((c16 + 1) % 2) * 16 * D], rotSrcGm_[(c16 + 1) * 16 * D], 16 * D); // 预取下一块
            }
            for (int32_t gi = 0; gi < nv; gi++) {
                for (int32_t e = 0; e < 8; e++) {
                    Mul(sqT_[(gi * 8 + e) * D], dupT_[uRow[gi] * D], htF[cur + e * D], D);
                }
            }
            SyncVS();
            for (int32_t gi = 0; gi < nv; gi += 2) {
                int32_t nw = (nv - gi < 2) ? (nv - gi) : 2;
                if (nw == 2) { // 两组 acc8（16 路会超寄存器预算溢出到栈；8 路是实测最优）
                    float accA[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                    float accB[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                    __ubuf__ float *pa = reinterpret_cast<__ubuf__ float *>(sqT_.GetPhyAddr()) + gi * 8 * D;
                    __ubuf__ float *pb2 = pa + 8 * D;
                    for (int32_t k = 0; k < D; k++) {
                        for (int32_t t = 0; t < 8; t++) {
                            accA[t] = accA[t] + pa[t * D + k];
                            accB[t] = accB[t] + pb2[t * D + k];
                        }
                    }
                    for (int32_t t = 0; t < 8; t++) {
                        yhatT_.SetValue(gi * D + chunk * 8 + t, accA[t]);
                        yhatT_.SetValue((gi + 1) * D + chunk * 8 + t, accB[t]);
                    }
                } else {
                    float acc8[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                    __ubuf__ float *pa = reinterpret_cast<__ubuf__ float *>(sqT_.GetPhyAddr()) + gi * 8 * D;
                    for (int32_t k = 0; k < D; k++) {
                        for (int32_t t = 0; t < 8; t++) {
                            acc8[t] = acc8[t] + pa[t * D + k];
                        }
                    }
                    for (int32_t t = 0; t < 8; t++) {
                        yhatT_.SetValue((gi + t / 8) * D + chunk * 8 + (t % 8), acc8[t]);
                    }
                }
            }
            if (half == 1 && c16 + 1 < D / 16) {
                SyncMte2V();
            }
        }
        SyncSV();
    }

    // gamma 标记向量的 yExact 路径：全 128 元素精确 y（复用组内 unit 行 dupT_[ui*D]，自然布局
    // H 分 16 行块 + 8 行一组交织 k 升序求和）→ 精确 idx 比对（翻转返回 false 走全链级联）→
    // 精确 r → TreeDot rnorm → gamma = norm × rnorm（逐位 = golden）写入 gamArrT_。
    // 实测 r_cube 与 r_golden 每元素 δ 可达 ~1e-6 且符号相关，"cube y 近似 r"的 gamma 外科
    // 在该量级下不可靠（mse4 曾 2 ULP 错），故 gamma 向量必须全量精确 y。
    __aicore__ inline bool SurgicalGammaVec(int32_t v, int32_t yRow, float norm)
    {
        const int32_t D = HEAD_DIM;
        // 精确 idx（yhatT_[D..2D) 作 idx 行；qjlBT_[0..D) 作 yhat 行）与 cube idx 全量比对
        // （y 由 ExactYBatch 写入 yhatT_ 第 yRow 行）
        Duplicate(qjlBT_, CentTable<BITS>::CENT[0], D);
        Duplicate(yhatT_[D], 0.0f, D);
        for (int32_t k = 0; k < BOUND_CNT; k++) {
            float b = (CentTable<BITS>::CENT[k] + CentTable<BITS>::CENT[k + 1]) * 0.5f;
            CompareScalar(maskT_, yhatT_[yRow * D], b, CMPMODE::GT, D);
            Duplicate(rT_, CentTable<BITS>::CENT[k + 1], D);
            Select(qjlBT_, maskT_, rT_, qjlBT_, SELMODE::VSEL_TENSOR_TENSOR_MODE, D);
            Duplicate(rT_, static_cast<float>(k + 1), D);
            Select(yhatT_[D], maskT_, rT_, yhatT_[D], SELMODE::VSEL_TENSOR_TENSOR_MODE, D);
        }
        SyncVS();
        for (int32_t j = 0; j < D; j++) {
            if (static_cast<int32_t>(yhatT_.GetValue(D + j)) != static_cast<int32_t>(idxFT_.GetValue(v * D + j))) {
                return false; // idx 翻转 → 全链级联
            }
        }
        // r = y - yhat（yhatT_ 第 yRow 行原地）；TreeDot rnorm；gamma = norm × rnorm
        Sub(yhatT_[yRow * D], yhatT_[yRow * D], qjlBT_, D);
        Mul(sqT_, yhatT_[yRow * D], yhatT_[yRow * D], D);
        float rnorm = sqrt(TreeDotReduce8());
        gamArrT_.SetValue(v, norm * rnorm);
        return true;
    }

    // 一组（≤8）tie 候选向量的外科处理：x 行/H 行前置发射一次同步，unit 成批，逐候选精确
    // 一组（≤8）tie 候选向量的外科处理：x 行/H 行前置发射一次同步，unit 成批，逐候选精确
    // 点积（自然布局 H 行 + Mul + 8 路交织 k 升序求和，与 golden 逐位一致）。
    // hits/nh/gamRow/forceCasc 均在级联前由调用方收集（级联会覆盖 yT_/qjlBT_ 前 128 元素）；
    // gamma 标记向量走 yExact 全量精确 y。判定失败者函数内直接级联并置 cascVec_。
    __aicore__ inline void SurgicalGroup(int64_t vecBase, int32_t s, const int32_t *cand, int32_t gc,
                                         const int32_t (*hits)[8], int32_t *nh, const int32_t *gamRow,
                                         const bool *forceCasc)
    {
        const int32_t D = HEAD_DIM;
        // ---- 1. x 行直接用 UploadX 驻留 xT_ 的本子批数据（免 GM 重读；时序上 xT_ 在
        //    UploadX→QuantSubbatch 之间不被改写，级联对 xT_ 的写入发生在组末、下一子批
        //    会重新 UploadX，安全）；H 行前置发射（不等待），unit 计算与 H 行 GM 延迟重叠 ----
        int32_t totalHits = 0;
        for (int32_t i = 0; i < gc; i++) {
            totalHits += nh[i];
        }
        bool batchOk = (totalHits > 0 && totalHits <= 24);
        uint8_t rowCand[24];
        int32_t roCopied = 0;
        if (batchOk) {
            event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(e);
            WaitFlag<HardEvent::V_MTE2>(e);
            for (int32_t i = 0; i < gc; i++) {
                for (int32_t e = 0; e < nh[i]; e++) {
                    DataCopy(rT_[roCopied * D], rotSrcGm_[hits[i][e] * D], D);
                    rowCand[roCopied] = static_cast<uint8_t>(i);
                    roCopied++;
                }
            }
        }
        // ---- 2. unit 块（dupT_，从驻留 xT_ Cast）+ 成批 TreeDot → norms[]（与 H 行在途拷贝并行）----
        for (int32_t i = 0; i < gc; i++) {
            Cast(dupT_[i * D], xT_[cand[i] * D], RoundMode::CAST_NONE, D);
        }
        for (int32_t i = 0; i < gc; i++) {
            Mul(sqT_[i * D], dupT_[i * D], dupT_[i * D], D);
        }
        for (int32_t off = 64; off >= 8; off >>= 1) {
            Add(sqT_[0], sqT_[0], sqT_[off], off, gc, BinaryRepeatParams(1, 1, 1, 16, 16, 16));
        }
        SyncVS();
        float norms[8];
        for (int32_t i = 0; i < gc; i++) {
            float s0 = sqT_.GetValue(i * D);
            for (int32_t t = 1; t < 8; t++) {
                s0 = s0 + sqT_.GetValue(i * D + t);
            }
            norms[i] = sqrt(s0);
        }
        for (int32_t i = 0; i < gc; i++) {
            float invN = 1.0f / (norms[i] > 1e-30f ? norms[i] : 1e-30f);
            Muls(dupT_[i * D], dupT_[i * D], invN, D);
        }
        if (batchOk) {
            SyncMte2V(); // 只需等 H 行（x 已是 UB 驻留）
        }
        // ---- 3. 命中点积 + idx 判定（先于 yExact：后者用 rT_ 做 H 块暂存会冲掉点积行）----
        bool okG[8];
        for (int32_t i = 0; i < gc; i++) {
            okG[i] = !forceCasc[i];
        }
        if (batchOk) {
            for (int32_t r = 0; r < roCopied; r++) {
                Mul(sqT_[r * D], dupT_[rowCand[r] * D], rT_[r * D], D);
            }
            SyncVS();
            int32_t rb = 0;
            for (int32_t i = 0; i < gc; i++) {
                if (!okG[i] || gamRow[i] >= 0) { // gamma 候选由 yExact 全量验证，免点积
                    rb += nh[i];
                    continue;
                }
                float yex[8];
                SumRowsConst(sqT_, rb, nh[i], yex);
                for (int32_t e = 0; e < nh[i]; e++) {
                    int32_t c = 0;
                    for (int32_t k = 0; k < BOUND_CNT; k++) {
                        float b = (CentTable<BITS>::CENT[k] + CentTable<BITS>::CENT[k + 1]) * 0.5f;
                        if (yex[e] > b) {
                            c++;
                        }
                    }
                    if (c != static_cast<int32_t>(idxFT_.GetValue(cand[i] * D + hits[i][e]))) {
                        okG[i] = false;
                    }
                }
                rb += nh[i];
            }
        } else if (totalHits > 24) {
            for (int32_t i = 0; i < gc; i++) {
                if (nh[i] > 0) {
                    okG[i] = false; // 罕见：行缓冲装不下，保守级联
                }
            }
        }
        // ---- 4. gamma 批量 yExact（≤4 个一批共享 H 行块装载；norm 用成批 TreeDot 值）----
        {
            int32_t gis[8];
            int32_t ng = 0;
            for (int32_t i = 0; i < gc; i++) {
                if (okG[i] && gamRow[i] >= 0) {
                    gis[ng++] = i;
                }
            }
            for (int32_t g0 = 0; g0 < ng; g0 += 4) {
                int32_t nv = (ng - g0 < 4) ? (ng - g0) : 4;
                int32_t uRow[4];
                for (int32_t u = 0; u < nv; u++) {
                    uRow[u] = gis[g0 + u]; // unit 行 = 组内候选下标
                }
                ExactYBatch(uRow, nv);
                for (int32_t u = 0; u < nv; u++) {
                    int32_t i = gis[g0 + u];
                    if (!SurgicalGammaVec(cand[i], u, norms[i])) {
                        okG[i] = false;
                    }
                }
            }
        }
        // ---- 5. 级联 ----
        for (int32_t i = 0; i < gc; i++) {
            if (!okG[i]) {
                RecomputeTieVec(vecBase + cand[i], s * CUBE_SUBBATCH + cand[i]);
                cascVec_[s * CUBE_SUBBATCH + cand[i]] = true;
            }
        }
    }

    // qjlExact 批量（≤4 个 |p'|≤guard 候选）：x 行批量重载（P 相 xT_ 无驻留）→ 成批 unit →
    // ExactYBatch 共享 H 行块 → 逐候选 r/r_hat（yhatT_ 行原地）→ 命中元素 S 行点积 → 翻 bit。
    // p'' 逐位 = golden p（unit/y/r 全链 golden 语序）。
    __aicore__ inline void SurgicalQjlBatch(int64_t vecBase, const int32_t *cand, const int32_t (*hits)[8],
                                            const int32_t *nh, int32_t nv)
    {
        const int32_t D = HEAD_DIM;
        // ---- 1. x 行批量 + 成批 TreeDot unit ----
        {
            event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(e);
            WaitFlag<HardEvent::V_MTE2>(e);
        }
        for (int32_t u = 0; u < nv; u++) {
            DataCopy(xT_[u * D], kvGm_[(vecBase + cand[u]) * D], D);
        }
        SyncMte2V();
        for (int32_t u = 0; u < nv; u++) {
            Cast(dupT_[u * D], xT_[u * D], RoundMode::CAST_NONE, D);
        }
        for (int32_t u = 0; u < nv; u++) {
            Mul(sqT_[u * D], dupT_[u * D], dupT_[u * D], D);
        }
        for (int32_t off = 64; off >= 8; off >>= 1) {
            Add(sqT_[0], sqT_[0], sqT_[off], off, nv, BinaryRepeatParams(1, 1, 1, 16, 16, 16));
        }
        SyncVS();
        for (int32_t u = 0; u < nv; u++) {
            float s0 = sqT_.GetValue(u * D);
            for (int32_t t = 1; t < 8; t++) {
                s0 = s0 + sqT_.GetValue(u * D + t);
            }
            float norm = sqrt(s0);
            float invN = 1.0f / (norm > 1e-30f ? norm : 1e-30f);
            Muls(dupT_[u * D], dupT_[u * D], invN, D);
        }
        // ---- 2. 成批精确 y ----
        int32_t uRow[4];
        for (int32_t u = 0; u < nv; u++) {
            uRow[u] = u;
        }
        ExactYBatch(uRow, nv);
        // ---- 3. 逐候选 r/r_hat + 命中 S 点积 + 翻 bit ----
        auto stF = rLoBuf_.Get<float>();
        for (int32_t u = 0; u < nv; u++) {
            Duplicate(qjlBT_, CentTable<BITS>::CENT[0], D);
            for (int32_t k = 0; k < BOUND_CNT; k++) {
                float b = (CentTable<BITS>::CENT[k] + CentTable<BITS>::CENT[k + 1]) * 0.5f;
                CompareScalar(maskT_, yhatT_[u * D], b, CMPMODE::GT, D);
                Duplicate(rT_, CentTable<BITS>::CENT[k + 1], D);
                Select(qjlBT_, maskT_, rT_, qjlBT_, SELMODE::VSEL_TENSOR_TENSOR_MODE, D);
            }
            Sub(yhatT_[u * D], yhatT_[u * D], qjlBT_, D); // r（原地）
            Mul(sqT_, yhatT_[u * D], yhatT_[u * D], D);
            float rnorm = sqrt(TreeDotReduce8());
            float invR = 1.0f / (rnorm > 1e-30f ? rnorm : 1e-30f);
            Muls(yhatT_[u * D], yhatT_[u * D], invR, D); // r_hat（原地）
            SyncVS();
            {
                event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(e);
                WaitFlag<HardEvent::V_MTE2>(e);
            }
            for (int32_t e = 0; e < nh[u]; e++) {
                DataCopy(stF[e * D], qjlSrcGm_[hits[u][e] * D], D);
            }
            SyncMte2V();
            for (int32_t e = 0; e < nh[u]; e++) {
                Mul(dupT_[e * D], yhatT_[u * D], stF[e * D], D);
            }
            SyncVS();
            float pex[8];
            SumRowsConst(dupT_, 0, nh[u], pex);
            __ubuf__ uint8_t *qdst = reinterpret_cast<__ubuf__ uint8_t *>(outQjlT_.GetPhyAddr()) +
                                     cand[u] * QJL_BYTES;
            for (int32_t e = 0; e < nh[u]; e++) {
                int32_t j = hits[u][e];
                int32_t bit = (pex[e] >= 0.0f) ? 1 : 0;
                int32_t cur = (qdst[j / 8] >> (j % 8)) & 1;
                if (cur != bit) {
                    qdst[j / 8] = qdst[j / 8] ^ static_cast<uint8_t>(1u << (j % 8));
                }
            }
        }
    }

    // 一组（≤8）qjl 候选向量的外科处理：rh/rl 行批量装载无损重建 cube r_hat，S 行成批，
    // 对 |p|<thr 命中元素精确点积（8 路交织 k 升序）。|p'|>3e-6 时 p' 符号即 golden 符号：
    // 与 cube bit 不一致直接翻 outQjlT_ 对应 bit（免级联）；|p'|≤3e-6 才级联。
    // hits/nh/forceCasc 由调用方在级联前收集（级联会覆盖 qjlBT_ 前 128 元素）。
    __aicore__ inline void SurgicalGroupQjl(int64_t vecBase, int32_t s, const int32_t *cand, int32_t gc,
                                            const int32_t (*hits)[8], const int32_t *nh, const bool *forceCasc)
    {
        const int32_t D = HEAD_DIM;
        // ---- 1. rh/rl 行 + S 行前置发射（命中下标已知），只等一次同步 ----
        {
            event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(e);
            WaitFlag<HardEvent::V_MTE2>(e);
        }
        for (int32_t i = 0; i < gc; i++) {
            DataCopy(rHiT_[i * D], rHiGm_[(vecBase + cand[i]) * D], D);
            DataCopy(rLoT_[i * D], rLoGm_[(vecBase + cand[i]) * D], D);
        }
        int32_t totalHits = 0;
        for (int32_t i = 0; i < gc; i++) {
            totalHits += nh[i];
        }
        bool batchOk = (totalHits > 0 && totalHits <= 30);
        uint8_t rowCand[30];
        int32_t roCopied = 0;
        if (batchOk) {
            for (int32_t i = 0; i < gc; i++) {
                for (int32_t e = 0; e < nh[i]; e++) {
                    DataCopy(rT_[roCopied * D], qjlSrcGm_[hits[i][e] * D], D);
                    rowCand[roCopied] = static_cast<uint8_t>(i);
                    roCopied++;
                }
            }
        }
        SyncMte2V();
        // rl cast 放 yT_ 后段（避免覆盖 rT_ 里的 S 行）；r_hat 行入 dupT_
        for (int32_t i = 0; i < gc; i++) {
            Cast(yT_[i * D], rHiT_[i * D], RoundMode::CAST_NONE, D);
            Cast(yT_[(8 + i) * D], rLoT_[i * D], RoundMode::CAST_NONE, D);
        }
        for (int32_t i = 0; i < gc; i++) {
            Add(dupT_[i * D], yT_[i * D], yT_[(8 + i) * D], D); // dupT_ = r_hat 行
        }
        // ---- 2. 点积 + bit 补丁（S 行已随 rh/rl 同步到位）----
        bool okG[8];
        bool needExact[8];
        for (int32_t i = 0; i < gc; i++) {
            okG[i] = !forceCasc[i];
            needExact[i] = false;
        }
        if (batchOk) {
            for (int32_t r = 0; r < roCopied; r++) {
                Mul(sqT_[r * D], dupT_[rowCand[r] * D], rT_[r * D], D);
            }
            SyncVS();
            int32_t rb = 0;
            for (int32_t i = 0; i < gc; i++) {
                if (!okG[i]) {
                    rb += nh[i];
                    continue;
                }
                float pex[8];
                SumRowsConst(sqT_, rb, nh[i], pex);
                __ubuf__ uint8_t *qdst = reinterpret_cast<__ubuf__ uint8_t *>(outQjlT_.GetPhyAddr()) +
                                         cand[i] * QJL_BYTES;
                for (int32_t e = 0; e < nh[i]; e++) {
                    float p2 = pex[e];
                    float ap = (p2 < 0.0f) ? -p2 : p2;
                    // guard 按 BITS 分档：|Δp'| ~ Δy/rnorm 随机游走，rnorm 随边界间距增大
                    // （bits=2 rnorm~0.28 / bits=3 ~0.16 / bits=4 ~0.1），取 ~1.5-2× 余量
                    const float qg = (BITS == 2) ? 2.0e-6f : ((BITS == 3) ? 2.5e-6f : 3.0e-6f);
                    if (ap <= qg) {
                        needExact[i] = true;
                        continue;
                    }
                    int32_t j = hits[i][e];
                    int32_t bit = (p2 >= 0.0f) ? 1 : 0;
                    int32_t cur = (qdst[j / 8] >> (j % 8)) & 1;
                    if (cur != bit) {
                        qdst[j / 8] = qdst[j / 8] ^ static_cast<uint8_t>(1u << (j % 8));
                    }
                }
                rb += nh[i];
            }
        } else if (totalHits > 30) {
            for (int32_t i = 0; i < gc; i++) {
                if (nh[i] > 0) {
                    okG[i] = false;
                }
            }
        }
        // ---- 3. guard 失败元素走 golden 全链精确点积（yExact + 精确 rhat + 命中元素点积翻
        //    bit，~6µs，替代 15µs 全链级联）；forceCasc/溢出仍走全链级联 ----
        {
            int32_t qis[8];
            int32_t nq = 0;
            for (int32_t i = 0; i < gc; i++) {
                if (okG[i] && needExact[i]) {
                    qis[nq++] = i;
                }
            }
            for (int32_t g0 = 0; g0 < nq; g0 += 4) {
                int32_t nv = (nq - g0 < 4) ? (nq - g0) : 4;
                int32_t qc2[4];
                int32_t hloc[4][8];
                int32_t qnh2[4];
                for (int32_t u = 0; u < nv; u++) {
                    qc2[u] = cand[qis[g0 + u]];
                    qnh2[u] = nh[qis[g0 + u]];
                    for (int32_t e = 0; e < qnh2[u]; e++) {
                        hloc[u][e] = hits[qis[g0 + u]][e];
                    }
                }
                SurgicalQjlBatch(vecBase, qc2, hloc, qnh2, nv);
            }
        }
        for (int32_t i = 0; i < gc; i++) {
            if (!okG[i]) {
                RecomputeTieVec(vecBase + cand[i], s * CUBE_SUBBATCH + cand[i], true);
                cascVec_[s * CUBE_SUBBATCH + cand[i]] = true;
            }
        }
    }

    // 命中行求和：sq 的 [rb, rb+nh) 行各自做 128 元素 k 升序线性求和（与 golden 逐位一致）。
    // 按 nh 分档字面量展开——运行时上界的数组累加循环会把累加器溢出到栈（实测慢 3×）。
    __aicore__ inline void SumRowsConst(const LocalTensor<float> &sq, int32_t rb, int32_t nh, float *out)
    {
        const int32_t D = HEAD_DIM;
        // int64 双字装载版：一次 UB 读取相邻两 float（小端：低 32=偶 k，高 32=奇 k），
        // 链内加法顺序 lo->hi 与标量 k 升序逐位等价；nh 链交错保持（填流水）。
        // UB 读次数减半 —— 标量 UB 读端口是求和链瓶颈（~1.1cyc/op）。
        __ubuf__ uint64_t *p64[8];
        float acc8[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        __ubuf__ float *baseF = reinterpret_cast<__ubuf__ float *>(sq.GetPhyAddr()) + rb * D;
        for (int32_t t = 0; t < nh; t++) {
            p64[t] = reinterpret_cast<__ubuf__ uint64_t *>(baseF + t * D);
        }
        for (int32_t k2 = 0; k2 < D / 2; k2++) {
            for (int32_t t = 0; t < nh; t++) {
                uint64_t w = p64[t][k2];
                uint32_t wlo = static_cast<uint32_t>(w);
                uint32_t whi = static_cast<uint32_t>(w >> 32);
                acc8[t] = acc8[t] + *reinterpret_cast<const float *>(&wlo);
                acc8[t] = acc8[t] + *reinterpret_cast<const float *>(&whi);
            }
        }
        for (int32_t t = 0; t < nh; t++) {
            out[t] = acc8[t];
        }
    }

    // pairwise 128→64→32→16→8 归约 sqT_ 前 8 项（向量 Add，与标量逐项 pairwise 逐位一致），
    // 再 8 个标量顺序累加 —— 对齐 golden TreeDot。调用前 sqT_ 须已为平方值。
    __aicore__ inline float TreeDotReduce8()
    {
        Add(sqT_[0], sqT_[0], sqT_[64], 64);
        Add(sqT_[0], sqT_[0], sqT_[32], 32);
        Add(sqT_[0], sqT_[0], sqT_[16], 16);
        Add(sqT_[0], sqT_[0], sqT_[8], 8);
        SyncVS();
        float s0 = sqT_.GetValue(0);
        for (int32_t i = 1; i < 8; i++) {
            s0 = s0 + sqT_.GetValue(i);
        }
        return s0;
    }

    // tie 向量精确重算的向量化版（fp32 矩阵 + 预转置就绪）：matmul 改走 H^T/S^T 的
    // j 维 128 路并行 × k 维升序串行 FMA（acc[j] += u_k * HT[k][j]，逐 k 广播标量），
    // 每个 fp32 乘/加的舍入序列与 ComputeGolden 的标量链完全一致；idx/r/gamma/qjl 链
    // 用与 QuantSubbatch 相同的 Compare/Select 向量模式（无算术、只选取，逐位等价）。
    __aicore__ inline void RecomputeTieVecVec(int64_t gvec, int32_t lv, bool qjlOnly)
    {
        const int32_t D = HEAD_DIM;
        const int32_t lvLocal = lv % CUBE_SUBBATCH;
        const int32_t idxBase = qjlOnly ? 0 : lvLocal * D;
        {
            event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(e);
            WaitFlag<HardEvent::V_MTE2>(e);
        }
        DataCopy(xT_, kvGm_[gvec * D], D);
        SyncMte2V();
        Cast(yT_, xT_, RoundMode::CAST_NONE, D); // yT_ = x（fp32）
        Mul(sqT_, yT_, yT_, D);
        float norm = sqrt(TreeDotReduce8());
        float invNorm = 1.0f / (norm > 1e-30f ? norm : 1e-30f);
        Muls(yT_, yT_, invNorm, D); // yT_ = unit
        SyncVS();
        // y = unit@H^T（j 维 128 路并行 × k 升序 128 步；H^T 分 16 行块读入 yhatT_）
        Duplicate(rT_, 0.0f, D);
        for (int32_t kc = 0; kc < D / 16; kc++) {
            {
                event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(e);
                WaitFlag<HardEvent::V_MTE2>(e);
            }
            DataCopy(yhatT_, htGm_[kc * 16 * D], 16 * D);
            SyncMte2V();
            for (int32_t jj = 0; jj < 16; jj++) {
                float uk = yT_.GetValue(kc * 16 + jj);
                Muls(sqT_, yhatT_[jj * D], uk, D);
                Add(rT_, rT_, sqT_, D);
            }
        }
        // idx / yhat（Compare/Select 链，与 QuantSubbatch 同模式，只选取无算术）
        Duplicate(idxFT_[idxBase], 0.0f, D);
        Duplicate(yhatT_, CentTable<BITS>::CENT[0], D);
        for (int32_t k = 0; k < BOUND_CNT; k++) {
            float b = (CentTable<BITS>::CENT[k] + CentTable<BITS>::CENT[k + 1]) * 0.5f;
            CompareScalar(maskT_, rT_, b, CMPMODE::GT, D);
            Duplicate(dupT_, static_cast<float>(k + 1), D);
            Select(idxFT_[idxBase], maskT_, dupT_, idxFT_[idxBase], SELMODE::VSEL_TENSOR_TENSOR_MODE, D);
            Duplicate(dupT_, CentTable<BITS>::CENT[k + 1], D);
            Select(yhatT_, maskT_, dupT_, yhatT_, SELMODE::VSEL_TENSOR_TENSOR_MODE, D);
        }
        Sub(rT_, rT_, yhatT_, D); // r = y - yhat
        Mul(sqT_, rT_, rT_, D);
        float rnorm = sqrt(TreeDotReduce8());
        float gamma = norm * rnorm;
        float invR = 1.0f / (rnorm > 1e-30f ? rnorm : 1e-30f);
        Muls(rT_, rT_, invR, D); // rT_ = r_hat
        SyncVS();
        // qjl = sign(r_hat@S^T)（同构 FMA；S^T 分 16 行块读入 yhatT_）
        Duplicate(qjlBT_, 0.0f, D);
        for (int32_t kc = 0; kc < D / 16; kc++) {
            {
                event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(e);
                WaitFlag<HardEvent::V_MTE2>(e);
            }
            DataCopy(yhatT_, stGm_[kc * 16 * D], 16 * D);
            SyncMte2V();
            for (int32_t jj = 0; jj < 16; jj++) {
                float rk = rT_.GetValue(kc * 16 + jj);
                Muls(sqT_, yhatT_[jj * D], rk, D);
                Add(qjlBT_, qjlBT_, sqT_, D);
            }
        }
        // 符号 → 0/1（acc >= 0 ⇒ 1，与标量三元一致）
        Duplicate(dupT_, 1.0f, D);
        Duplicate(yhatT_, 0.0f, D);
        Compare<float>(maskT_, qjlBT_, yhatT_, CMPMODE::GE, D);
        Select(qjlBT_, maskT_, dupT_, yhatT_, SELMODE::VSEL_TENSOR_TENSOR_MODE, D);
        SyncSV();
        // ---- 写回（同标量版）：qjl → tieQjlT_，idx 已在 idxFT_ 局部槽位，norm/gamma → normArrT_/gamArrT_ ----
        __ubuf__ uint8_t *tieQjl = reinterpret_cast<__ubuf__ uint8_t *>(tieQjlT_.GetPhyAddr()) + lv * QJL_BYTES;
        for (int32_t g = 0; g < D / 8; g++) {
            uint32_t qword = 0;
            for (int32_t lane = 0; lane < 8; lane++) {
                int32_t e = g * 8 + lane;
                qword |= static_cast<uint32_t>(static_cast<int32_t>(qjlBT_.GetValue(e))) << lane;
            }
            tieQjl[g] = static_cast<uint8_t>(qword & 0xFFU);
        }
        if (!qjlOnly) {
            normArrT_.SetValue(lv, norm);
            gamArrT_.SetValue(lvLocal, gamma);
        }
    }

    // tie 向量精确重算（对齐 ComputeGolden 逐行）：norm=sqrt(TreeDot(x,x))→inv→unit=x*inv→
    // y[j]=Σ_k unit[k]*H[j,k]（k 升序）→idx→r→rnorm→gamma→invR→r_hat→qjl=sign(Σ_k r_hat[k]*S[j,k])。
    // 结果：idx/norm/gamma 直接写 GM，qjl 位写 tieQjlT_[lv*QJL_BYTES]（lv=tile 内局部向量下标）。
    __aicore__ inline void RecomputeTieVec(int64_t gvec, int32_t lv, bool qjlOnly = false)
    {
        if (useVecRc_) {
            RecomputeTieVecVec(gvec, lv, qjlOnly);
            return;
        }
        const int32_t D = HEAD_DIM;
        const int32_t lvLocal = lv % CUBE_SUBBATCH; // 子批内局部向量下标
        const int32_t idxBase = qjlOnly ? 0 : lvLocal * D; // qjlOnly 时用 scratch 0..D-1，不污染子批 idxFT_
        {
            event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(e);
            WaitFlag<HardEvent::V_MTE2>(e);
        }
        DataCopy(xT_, kvGm_[gvec * D], D);
        SyncMte2V();
        Cast(yT_, xT_, RoundMode::CAST_NONE, D); // x bf16 → fp32（yT_ 暂作 unit）
        // norm = sqrt(TreeDot(x,x))（向量 pairwise，与标量逐项 pairwise 逐位一致）
        Mul(sqT_, yT_, yT_, D);
        float norm = sqrt(TreeDotReduce8());
        float invNorm = 1.0f / (norm > 1e-30f ? norm : 1e-30f);
        Muls(yT_, yT_, invNorm, D); // unit = x * invNorm
        SyncVS();
        // y = unit@H（每点积严格 k 升序；H 分 8 行块读入 rHiT_ 的 fp32 视图（不冲幂次表），
        // 双缓冲预取隐藏 GM 延迟；8 行一组 Mul + 交织线性求和）
        auto htF = qjlBBuf_.Get<float>(); // 4096 floats = 32 行（双缓冲 2×16；y 段时 qjlBT_ 未启用）
        {
            event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(e);
            WaitFlag<HardEvent::V_MTE2>(e);
        }
        DataCopy(htF, rotSrcGm_[0], 16 * D);
        SyncMte2V();
        for (int32_t c16 = 0; c16 < D / 16; c16++) { // 16 行块：16 Mul + 1 SyncVS + 16 链求和
            int32_t cur = (c16 % 2) * 16 * D;
            if (c16 + 1 < D / 16) {
                DataCopy(htF[((c16 + 1) % 2) * 16 * D], rotSrcGm_[(c16 + 1) * 16 * D], 16 * D); // 预取下一块
            }
            for (int32_t e = 0; e < 16; e++) {
                Mul(dupT_[e * D], yT_, htF[cur + e * D], D); // prod_e = unit * H_row
            }
            SyncVS();
            float accA[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            float accB[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            __ubuf__ float *pa = reinterpret_cast<__ubuf__ float *>(dupT_.GetPhyAddr());
            __ubuf__ float *pb2 = pa + 8 * D;
            for (int32_t k = 0; k < D; k++) {
                for (int32_t e = 0; e < 8; e++) {
                    accA[e] = accA[e] + pa[e * D + k];
                    accB[e] = accB[e] + pb2[e * D + k];
                }
            }
            for (int32_t e = 0; e < 8; e++) {
                rT_.SetValue(c16 * 16 + e, accA[e]); // rT_ 暂作 y
                rT_.SetValue(c16 * 16 + 8 + e, accB[e]);
            }
            if (c16 + 1 < D / 16) {
                SyncMte2V(); // 等预取完成（与上面的求和/写回并行）
            }
        }
        // idx / yhat（Compare/Select 链，与 QuantSubbatch 同模式，只选取无算术；写到 tie 向量
        // 在子批 idxFT_ 里的局部槽位，PackIdx 统一打包）
        Duplicate(idxFT_[idxBase], 0.0f, D);
        Duplicate(yhatT_, CentTable<BITS>::CENT[0], D);
        for (int32_t k = 0; k < BOUND_CNT; k++) {
            float b = (CentTable<BITS>::CENT[k] + CentTable<BITS>::CENT[k + 1]) * 0.5f;
            CompareScalar(maskT_, rT_, b, CMPMODE::GT, D);
            Duplicate(dupT_, static_cast<float>(k + 1), D);
            Select(idxFT_[idxBase], maskT_, dupT_, idxFT_[idxBase], SELMODE::VSEL_TENSOR_TENSOR_MODE, D);
            Duplicate(dupT_, CentTable<BITS>::CENT[k + 1], D);
            Select(yhatT_, maskT_, dupT_, yhatT_, SELMODE::VSEL_TENSOR_TENSOR_MODE, D);
        }
        Sub(rT_, rT_, yhatT_, D); // r = y - yhat
        // rnorm = sqrt(TreeDot(r,r))
        Mul(sqT_, rT_, rT_, D);
        float rnorm = sqrt(TreeDotReduce8());
        float gamma = norm * rnorm;
        float invR = 1.0f / (rnorm > 1e-30f ? rnorm : 1e-30f);
        Muls(rT_, rT_, invR, D); // r_hat = r * invR
        SyncVS();
        // qjl = sign(r_hat @ S)（每点积严格 k 升序；staging 用 yT_（qjl 段时 unit 已死），
        // 双缓冲 2×16；stFpT_ 幂次表不再被用作 staging，无需恢复）
        auto htS = yBuf_.Get<float>(); // 4096 floats = 32 行
        {
            event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(e);
            WaitFlag<HardEvent::V_MTE2>(e);
        }
        DataCopy(htS, qjlSrcGm_[0], 16 * D);
        SyncMte2V();
        for (int32_t c16 = 0; c16 < D / 16; c16++) { // 16 行块：16 Mul + 1 SyncVS + 16 链求和
            int32_t cur = (c16 % 2) * 16 * D;
            if (c16 + 1 < D / 16) {
                DataCopy(htS[((c16 + 1) % 2) * 16 * D], qjlSrcGm_[(c16 + 1) * 16 * D], 16 * D); // 预取下一块
            }
            for (int32_t e = 0; e < 16; e++) {
                Mul(dupT_[e * D], rT_, htS[cur + e * D], D); // prod_e = r_hat * S_row
            }
            SyncVS();
            float accA[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            float accB[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            __ubuf__ float *pa = reinterpret_cast<__ubuf__ float *>(dupT_.GetPhyAddr());
            __ubuf__ float *pb2 = pa + 8 * D;
            for (int32_t k = 0; k < D; k++) {
                for (int32_t e = 0; e < 8; e++) {
                    accA[e] = accA[e] + pa[e * D + k];
                    accB[e] = accB[e] + pb2[e * D + k];
                }
            }
            for (int32_t e = 0; e < 8; e++) {
                float a = accA[e];
                float b2 = accB[e];
                qjlBT_.SetValue(c16 * 16 + e, (a >= 0.0f) ? 1.0f : 0.0f);
                qjlBT_.SetValue(c16 * 16 + 8 + e, (b2 >= 0.0f) ? 1.0f : 0.0f);
            }
            if (c16 + 1 < D / 16) {
                SyncMte2V();
            }
        }
        SyncSV();
        // ---- 写回：qjl → tieQjlT_（idx 已写 idxFT_ 局部槽位、norm/gamma 写 normArrT_/gamArrT_，
        // 统一由 PackIdx/FlushIdxNormGamma 打包写出，避免标量 SetValue 直接写 GM 冲相邻向量）----
        __ubuf__ uint8_t *tieQjl = reinterpret_cast<__ubuf__ uint8_t *>(tieQjlT_.GetPhyAddr()) + lv * QJL_BYTES;
        for (int32_t g = 0; g < D / 8; g++) {
            uint32_t qword = 0;
            for (int32_t lane = 0; lane < 8; lane++) {
                int32_t e = g * 8 + lane;
                qword |= static_cast<uint32_t>(static_cast<int32_t>(qjlBT_.GetValue(e))) << lane;
            }
            tieQjl[g] = static_cast<uint8_t>(qword & 0xFFU);
        }
        if (!qjlOnly) {
            normArrT_.SetValue(lv, norm);
            gamArrT_.SetValue(lvLocal, gamma);
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

    // ---- 顺延外科（deferred epilogue）支撑 ----
    // 尾声全链重算（真翻转 / hits 溢出 / gamma idx 不符 / qjl guard 失败时调用）：
    // RecomputeTieVec 全链 golden 语序 → 0 号 scratch 槽位 → 直接写 GM 输出。
    __aicore__ inline void EpilogueFullRecompute(int64_t gv, bool qjlOnly)
    {
        SyncMte3S(); // 等此前尾声写出的 DataCopyPad（读 outIdxT_/outNormBf16T_/tieQjlT_）完成
        RecomputeTieVec(gv, 0, qjlOnly);
        if (!qjlOnly) {
            PackIdx(1, 0); // idxFT_[0..D) → outIdxT_（结尾自带 PipeBarrier<PIPE_ALL>）
            SyncSMte3();
            DataCopyExtParams ep;
            ep.blockCount = 1;
            ep.blockLen = IDX_BYTES;
            ep.srcStride = 0;
            ep.dstStride = 0;
            DataCopyPad(idxGm_[gv * IDX_BYTES], outIdxT_, ep);
            DataCopyExtParams np;
            np.blockCount = 1;
            np.blockLen = 2;
            np.srcStride = 0;
            np.dstStride = 0;
            if (outDtype_ == 1) {
                auto epNormF16 = outNormBf16Buf_.Get<half>();
                auto epGamF16 = outGamBf16Buf_.Get<half>();
                Cast(epNormF16, normArrT_[0], RoundMode::CAST_RINT, 1);
                Cast(epGamF16, gamArrT_[0], RoundMode::CAST_RINT, 1);
                SyncVMte3();
                DataCopyPad(normGmF16_[gv], epNormF16, np);
                DataCopyPad(gamGmF16_[gv], epGamF16, np);
            } else {
                Cast(outNormBf16T_, normArrT_[0], RoundMode::CAST_RINT, 1);
                Cast(outGamBf16T_, gamArrT_[0], RoundMode::CAST_RINT, 1);
                SyncVMte3();
                DataCopyPad(normGm_[gv], outNormBf16T_, np);
                DataCopyPad(gamGm_[gv], outGamBf16T_, np);
            }
        }
        SyncSMte3();
        DataCopyExtParams qp;
        qp.blockCount = 1;
        qp.blockLen = QJL_BYTES;
        qp.srcStride = 0;
        qp.dstStride = 0;
        DataCopyPad(qjlOutGm_[gv * QJL_BYTES], tieQjlT_, qp);
    }

    // 尾声集中验证：y 侧命中点积 + stash 的 cube idx 值比对（翻转才全链）；gamma 候选成批
    // yExact（≤4 共享 H 行块流）；qjl 侧 rh/rl 无损重建 + 命中 S 点积，guard 内翻 GM bit。
    // 与"一律全链重算"产出逐位一致：非翻转候选的 cube 输出本就等于 golden（norm 同源
    // TreeDot、idx/qjl 经检测带证明不翻转、gamma 非边界 1ULP 内同舍入），故只验证不写回。
    __aicore__ inline void EpilogueSurgery()
    {
        const int32_t D = HEAD_DIM;
        nFull_ = 0;
        nFlip_ = 0;
        // ---- 1. y 侧简单候选（非 full）：成批（≤4/批）共享同步与装载，命中点积 + cube idx 比对 ----
        {
            int32_t i = 0;
            while (i < stashYC_) {
                int32_t slot[4];
                int64_t gvs[4];
                int32_t nv = 0;
                while (i < stashYC_ && nv < 4) { // 跳过 full（登记统一全站点）与 gamma（pass 2）
                    int32_t fl = stashYFl_.GetValue(i);
                    int64_t gv = stashYGv_.GetValue(i);
                    if ((fl & 2) != 0) {
                        if (nFull_ < FULL_CAP) {
                            fullGv_[nFull_] = gv;
                            fullQjl_[nFull_] = 0;
                            nFull_++;
                        } else {
                            EpilogueFullRecompute(gv, false); // 病态溢出：此刻无批状态，安全
                        }
                    } else if ((fl & 1) == 0) {
                        slot[nv] = i;
                        gvs[nv] = gv;
                        nv++;
                    }
                    i++;
                }
                if (nv == 0) {
                    continue;
                }
                SyncMte3S();
                {
                    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                    SetFlag<HardEvent::V_MTE2>(e);
                    WaitFlag<HardEvent::V_MTE2>(e);
                }
                int32_t nhs[4];
                for (int32_t u = 0; u < nv; u++) {
                    nhs[u] = stashYNh_.GetValue(slot[u]);
                    DataCopy(xT_[u * D], kvGm_[gvs[u] * D], D);
                    for (int32_t e = 0; e < nhs[u]; e++) { // rT_ 32 行：u*8+e
                        DataCopy(rT_[(u * 8 + e) * D], rotSrcGm_[stashYHit_.GetValue(slot[u] * 8 + e) * D], D);
                    }
                }
                SyncMte2V();
                for (int32_t u = 0; u < nv; u++) {
                    Cast(dupT_[u * D], xT_[u * D], RoundMode::CAST_NONE, D);
                }
                for (int32_t u = 0; u < nv; u++) {
                    Mul(sqT_[u * D], dupT_[u * D], dupT_[u * D], D);
                }
                for (int32_t off = 64; off >= 8; off >>= 1) { // 成批 TreeDot（同旧外科语序）
                    Add(sqT_[0], sqT_[0], sqT_[off], off, nv, BinaryRepeatParams(1, 1, 1, 16, 16, 16));
                }
                SyncVS();
                for (int32_t u = 0; u < nv; u++) {
                    float s0 = sqT_.GetValue(u * D);
                    for (int32_t t = 1; t < 8; t++) {
                        s0 = s0 + sqT_.GetValue(u * D + t);
                    }
                    float norm = sqrt(s0);
                    float invN = 1.0f / (norm > 1e-30f ? norm : 1e-30f);
                    Muls(dupT_[u * D], dupT_[u * D], invN, D);
                }
                for (int32_t u = 0; u < nv; u++) { // unit 就绪后 sqT_ 改作 prod（32 行 = 4 候选 × 8 命中）
                    for (int32_t e = 0; e < nhs[u]; e++) {
                        Mul(sqT_[(u * 8 + e) * D], dupT_[u * D], rT_[(u * 8 + e) * D], D);
                    }
                }
                SyncVS();
                for (int32_t u = 0; u < nv; u++) {
                    float yex[8];
                    SumRowsConst(sqT_, u * 8, nhs[u], yex); // 严格 k 升序（= golden 点积）
                    bool flip = false;
                    for (int32_t e = 0; e < nhs[u]; e++) {
                        int32_t c = 0;
                        for (int32_t k = 0; k < BOUND_CNT; k++) {
                            float b = (CentTable<BITS>::CENT[k] + CentTable<BITS>::CENT[k + 1]) * 0.5f;
                            if (yex[e] > b) {
                                c++;
                            }
                        }
                        if (c != stashYIv_.GetValue(slot[u] * 8 + e)) {
                            flip = true;
                        }
                    }
                    if (flip) { // flip 降级：登记 flip 批（pass4b 精确链，省全链 ~34µs）
                        if (nFlip_ < FLIP_CAP) {
                            flipGv_[nFlip_++] = gvs[u];
                        } else if (nFull_ < FULL_CAP) {
                            fullGv_[nFull_] = gvs[u];
                            fullQjl_[nFull_] = 0;
                            nFull_++;
                        } else { // 病态兜底标记，尾声最后全链
                            stashYFl_.SetValue(slot[u], stashYFl_.GetValue(slot[u]) | 4);
                        }
                    }
                }
            }
        }
        // ---- 2. gamma 候选：成批（≤4）全量精确 y（共享 H 行块流）→ 精确 idx 比对 → gamma ----
        for (int32_t g0 = 0; g0 < gamYC_; g0 += 4) {
            int32_t nv = (gamYC_ - g0 < 4) ? (gamYC_ - g0) : 4;
            int64_t gvs[4];
            for (int32_t u = 0; u < nv; u++) {
                gvs[u] = gamGv_.GetValue(g0 + u);
            }
            SyncMte3S();
            {
                event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(e);
                WaitFlag<HardEvent::V_MTE2>(e);
            }
            for (int32_t u = 0; u < nv; u++) {
                DataCopy(xT_[u * D], kvGm_[gvs[u] * D], D);
            }
            SyncMte2V();
            for (int32_t u = 0; u < nv; u++) {
                Cast(dupT_[u * D], xT_[u * D], RoundMode::CAST_NONE, D);
            }
            for (int32_t u = 0; u < nv; u++) {
                Mul(sqT_[u * D], dupT_[u * D], dupT_[u * D], D);
            }
            for (int32_t off = 64; off >= 8; off >>= 1) { // 成批 TreeDot（同旧外科语序）
                Add(sqT_[0], sqT_[0], sqT_[off], off, nv, BinaryRepeatParams(1, 1, 1, 16, 16, 16));
            }
            SyncVS();
            float norms[4];
            for (int32_t u = 0; u < nv; u++) {
                float s0 = sqT_.GetValue(u * D);
                for (int32_t t = 1; t < 8; t++) {
                    s0 = s0 + sqT_.GetValue(u * D + t);
                }
                norms[u] = sqrt(s0);
            }
            for (int32_t u = 0; u < nv; u++) {
                float invN = 1.0f / (norms[u] > 1e-30f ? norms[u] : 1e-30f);
                Muls(dupT_[u * D], dupT_[u * D], invN, D);
            }
            int32_t uRow[4] = {0, 1, 2, 3};
            ExactYBatch(uRow, nv); // yhatT_ 行 u = 精确 y（严格 k 升序分离舍入）
            for (int32_t u = 0; u < nv; u++) {
                // 精确 idx（Compare/Select，无算术）与 stash 的 cube idx 行全量比对
                Duplicate(idxFT_, CentTable<BITS>::CENT[0], D);
                Duplicate(idxFT_[D], 0.0f, D);
                for (int32_t k = 0; k < BOUND_CNT; k++) {
                    float b = (CentTable<BITS>::CENT[k] + CentTable<BITS>::CENT[k + 1]) * 0.5f;
                    CompareScalar(maskT_, yhatT_[u * D], b, CMPMODE::GT, D);
                    Duplicate(sqT_, CentTable<BITS>::CENT[k + 1], D);
                    Select(idxFT_, maskT_, sqT_, idxFT_, SELMODE::VSEL_TENSOR_TENSOR_MODE, D);
                    Duplicate(sqT_, static_cast<float>(k + 1), D);
                    Select(idxFT_[D], maskT_, sqT_, idxFT_[D], SELMODE::VSEL_TENSOR_TENSOR_MODE, D);
                }
                SyncVS();
                bool flip = false;
                for (int32_t j = 0; j < D; j++) {
                    if (static_cast<int32_t>(idxFT_.GetValue(D + j)) != gamIdx_.GetValue((g0 + u) * D + j)) {
                        flip = true;
                        break;
                    }
                }
                if (flip) {
                    // flip 降级：登记 flip 批（pass4b 精确链）
                    if (nFlip_ < FLIP_CAP) {
                        flipGv_[nFlip_++] = gvs[u];
                    } else if (nFull_ < FULL_CAP) {
                        fullGv_[nFull_] = gvs[u];
                        fullQjl_[nFull_] = 0;
                        nFull_++;
                    }
                    continue;
                }
                Sub(yhatT_[u * D], yhatT_[u * D], idxFT_, D); // r = y - yhat（原地）
                Mul(sqT_, yhatT_[u * D], yhatT_[u * D], D);
                float rnorm = sqrt(TreeDotReduce8());
                float gamma = norms[u] * rnorm; // = golden gamma
                gamArrT_.SetValue(0, gamma);
                SyncSV();
                DataCopyExtParams np;
                np.blockCount = 1;
                np.blockLen = 2;
                np.srcStride = 0;
                np.dstStride = 0;
                if (outDtype_ == 1) {
                    auto g16 = outGamBf16Buf_.Get<half>();
                    Cast(g16, gamArrT_[0], RoundMode::CAST_RINT, 1);
                    SyncVMte3();
                    DataCopyPad(gamGmF16_[gvs[u]], g16, np);
                } else {
                    Cast(outGamBf16T_, gamArrT_[0], RoundMode::CAST_RINT, 1);
                    SyncVMte3();
                    DataCopyPad(gamGm_[gvs[u]], outGamBf16T_, np);
                }
            }
        }
        // ---- 3. qjl 候选：rh/rl 无损重建 cube r_hat，命中 S 精确点积，guard 内翻 GM bit ----
        nQex_ = 0;
        {
            int32_t i = 0;
            while (i < stashQC_) {
                int32_t slot[4];
                int64_t gvs[4];
                int32_t nv = 0;
                while (i < stashQC_ && nv < 4) {
                    int32_t fl = stashQFl_.GetValue(i);
                    int64_t gv = stashQGv_.GetValue(i);
                    // 与 fullList 去重：flip/gamma 不符/full 的向量由全链重算写出完整精确 qjl，
                    // 无需再过 qjl 验证/精确链（省一次全 y，~30µs/个）
                    bool inFull = false;
                    for (int32_t f = 0; f < nFull_; f++) {
                        if (fullGv_[f] == gv) {
                            inFull = true;
                            break;
                        }
                    }
                    for (int32_t f = 0; f < nFlip_; f++) {
                        if (flipGv_[f] == gv) {
                            inFull = true;
                            break;
                        }
                    }
                    if (inFull) {
                        i++;
                        continue;
                    }
                    if ((fl & 2) != 0) {
                        if (nFull_ < FULL_CAP) {
                            fullGv_[nFull_] = gv;
                            fullQjl_[nFull_] = 1;
                            nFull_++;
                        } else {
                            EpilogueFullRecompute(gv, true); // 病态溢出：此刻无批状态，安全
                        }
                    } else {
                        slot[nv] = i;
                        gvs[nv] = gv;
                        nv++;
                    }
                    i++;
                }
                if (nv == 0) {
                    continue;
                }
                SyncMte3S();
                {
                    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                    SetFlag<HardEvent::V_MTE2>(e);
                    WaitFlag<HardEvent::V_MTE2>(e);
                }
                int32_t nhs[4];
                for (int32_t u = 0; u < nv; u++) {
                    nhs[u] = stashQNh_.GetValue(slot[u]);
                    DataCopy(rHiT_[u * D], rHiGm_[gvs[u] * D], D);
                    DataCopy(rLoT_[u * D], rLoGm_[gvs[u] * D], D);
                    for (int32_t e = 0; e < nhs[u]; e++) {
                        DataCopy(rT_[(u * 8 + e) * D], qjlSrcGm_[stashQHit_.GetValue(slot[u] * 8 + e) * D], D);
                    }
                }
                SyncMte2V();
                for (int32_t u = 0; u < nv; u++) {
                    Cast(yhatT_[u * D], rHiT_[u * D], RoundMode::CAST_NONE, D);
                    Cast(idxFT_[u * D], rLoT_[u * D], RoundMode::CAST_NONE, D);
                    Add(dupT_[u * D], yhatT_[u * D], idxFT_[u * D], D); // dupT_ = cube r_hat（无损）
                }
                for (int32_t u = 0; u < nv; u++) {
                    for (int32_t e = 0; e < nhs[u]; e++) {
                        Mul(sqT_[(u * 8 + e) * D], dupT_[u * D], rT_[(u * 8 + e) * D], D);
                    }
                }
                SyncVS();
                for (int32_t u = 0; u < nv; u++) {
                    float pex[8];
                    SumRowsConst(sqT_, u * 8, nhs[u], pex);
                    const float qg = (BITS == 2) ? 2.0e-6f : ((BITS == 3) ? 2.5e-6f : 3.0e-6f);
                    bool needExact = false;
                    for (int32_t e = 0; e < nhs[u]; e++) {
                        float ap = (pex[e] < 0.0f) ? -pex[e] : pex[e];
                        if (ap <= qg) {
                            needExact = true;
                            break;
                        }
                    }
                    if (needExact) {
                        qexSlots_[nQex_++] = slot[u]; // Lever B：pass 4 成批精确链
                        continue;
                    }
                    uint32_t qrow[QJL_BYTES / 4];
                    for (int32_t t = 0; t < QJL_BYTES / 4; t++) {
                        qrow[t] = static_cast<uint32_t>(stashQRow_.GetValue(slot[u] * (QJL_BYTES / 4) + t));
                    }
                    bool dirty = false;
                    for (int32_t e = 0; e < nhs[u]; e++) {
                        int32_t j = stashQHit_.GetValue(slot[u] * 8 + e);
                        int32_t bit = (pex[e] >= 0.0f) ? 1 : 0;
                        int32_t cur = static_cast<int32_t>((qrow[j / 32] >> (j % 32)) & 1u);
                        if (cur != bit) {
                            qrow[j / 32] = qrow[j / 32] ^ (1u << (j % 32));
                            dirty = true;
                        }
                    }
                    if (dirty) { // 写 GM 只读 outQjlT_，不碰批内共享状态 → 批内直接写安全
                        __ubuf__ uint32_t *qdst = reinterpret_cast<__ubuf__ uint32_t *>(outQjlT_.GetPhyAddr());
                        for (int32_t t = 0; t < QJL_BYTES / 4; t++) {
                            qdst[t] = qrow[t];
                        }
                        SyncSMte3();
                        DataCopyExtParams qp;
                        qp.blockCount = 1;
                        qp.blockLen = QJL_BYTES;
                        qp.srcStride = 0;
                        qp.dstStride = 0;
                        DataCopyPad(qjlOutGm_[gvs[u] * QJL_BYTES], outQjlT_, qp);
                    }
                }
            }
        }
        // ---- 4. qjlExact 批（guard 失败候选，≤4/批共享 H 行块流）：精确全链到命中符号——
        //    unit → ExactYBatch 精确 y → idx/yhat（Compare/Select 无算术）→ r → rnorm →
        //    r_hat → 命中元素 S 行严格 k 升序点积 → 只翻 stash cube qjl 行的对应 bit。
        //    与 golden 逐位一致（每环节语序同 RecomputeTieVec），但只做 ≤8 个点积而非全 128。
        auto stF = rLoBuf_.Get<float>(); // 2048 floats = 16 行（尾声 rLoT_ 已死）
        for (int32_t g0 = 0; g0 < nQex_; g0 += 4) {
            int32_t nv = (nQex_ - g0 < 4) ? (nQex_ - g0) : 4;
            int64_t gvs[4];
            for (int32_t u = 0; u < nv; u++) {
                gvs[u] = stashQGv_.GetValue(qexSlots_[g0 + u]);
            }
            SyncMte3S();
            {
                event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(e);
                WaitFlag<HardEvent::V_MTE2>(e);
            }
            for (int32_t u = 0; u < nv; u++) {
                DataCopy(xT_[u * D], kvGm_[gvs[u] * D], D);
            }
            SyncMte2V();
            for (int32_t u = 0; u < nv; u++) {
                Cast(dupT_[u * D], xT_[u * D], RoundMode::CAST_NONE, D);
            }
            for (int32_t u = 0; u < nv; u++) {
                Mul(sqT_[u * D], dupT_[u * D], dupT_[u * D], D);
            }
            for (int32_t off = 64; off >= 8; off >>= 1) { // 成批 TreeDot（同旧外科语序）
                Add(sqT_[0], sqT_[0], sqT_[off], off, nv, BinaryRepeatParams(1, 1, 1, 16, 16, 16));
            }
            SyncVS();
            for (int32_t u = 0; u < nv; u++) {
                float s0 = sqT_.GetValue(u * D);
                for (int32_t t = 1; t < 8; t++) {
                    s0 = s0 + sqT_.GetValue(u * D + t);
                }
                float norm = sqrt(s0);
                float invN = 1.0f / (norm > 1e-30f ? norm : 1e-30f);
                Muls(dupT_[u * D], dupT_[u * D], invN, D);
            }
            int32_t uRow[4] = {0, 1, 2, 3};
            ExactYBatch(uRow, nv); // yhatT_ 行 u = 精确 y（严格 k 升序分离舍入）
            for (int32_t u = 0; u < nv; u++) {
                int32_t slot = qexSlots_[g0 + u];
                // yhat（Compare/Select，无算术）
                Duplicate(qjlBT_, CentTable<BITS>::CENT[0], D);
                for (int32_t k = 0; k < BOUND_CNT; k++) {
                    float b = (CentTable<BITS>::CENT[k] + CentTable<BITS>::CENT[k + 1]) * 0.5f;
                    CompareScalar(maskT_, yhatT_[u * D], b, CMPMODE::GT, D);
                    Duplicate(rT_, CentTable<BITS>::CENT[k + 1], D);
                    Select(qjlBT_, maskT_, rT_, qjlBT_, SELMODE::VSEL_TENSOR_TENSOR_MODE, D);
                }
                Sub(yhatT_[u * D], yhatT_[u * D], qjlBT_, D); // r（原地）
                Mul(sqT_, yhatT_[u * D], yhatT_[u * D], D);
                float rnorm = sqrt(TreeDotReduce8());
                float invR = 1.0f / (rnorm > 1e-30f ? rnorm : 1e-30f);
                Muls(yhatT_[u * D], yhatT_[u * D], invR, D); // r_hat（原地）
                SyncVS();
                {
                    event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                    SetFlag<HardEvent::V_MTE2>(e);
                    WaitFlag<HardEvent::V_MTE2>(e);
                }
                int32_t nh = stashQNh_.GetValue(slot);
                for (int32_t e = 0; e < nh; e++) {
                    DataCopy(stF[e * D], qjlSrcGm_[stashQHit_.GetValue(slot * 8 + e) * D], D);
                }
                SyncMte2V();
                for (int32_t e = 0; e < nh; e++) {
                    Mul(dupT_[e * D], yhatT_[u * D], stF[e * D], D); // unit 行已被 ExactYBatch 消费
                }
                SyncVS();
                float pex[8];
                SumRowsConst(dupT_, 0, nh, pex); // 严格 k 升序（= golden p）
                uint32_t qrow[QJL_BYTES / 4];
                for (int32_t t = 0; t < QJL_BYTES / 4; t++) {
                    qrow[t] = static_cast<uint32_t>(stashQRow_.GetValue(slot * (QJL_BYTES / 4) + t));
                }
                bool dirty = false;
                for (int32_t e = 0; e < nh; e++) {
                    int32_t j = stashQHit_.GetValue(slot * 8 + e);
                    int32_t bit = (pex[e] >= 0.0f) ? 1 : 0;
                    int32_t cur = static_cast<int32_t>((qrow[j / 32] >> (j % 32)) & 1u);
                    if (cur != bit) {
                        qrow[j / 32] = qrow[j / 32] ^ (1u << (j % 32));
                        dirty = true;
                    }
                }
                if (dirty) {
                    __ubuf__ uint32_t *qdst = reinterpret_cast<__ubuf__ uint32_t *>(outQjlT_.GetPhyAddr());
                    for (int32_t t = 0; t < QJL_BYTES / 4; t++) {
                        qdst[t] = qrow[t];
                    }
                    SyncSMte3();
                    DataCopyExtParams qp;
                    qp.blockCount = 1;
                    qp.blockLen = QJL_BYTES;
                    qp.srcStride = 0;
                    qp.dstStride = 0;
                    DataCopyPad(qjlOutGm_[gvs[u] * QJL_BYTES], outQjlT_, qp);
                }
            }
        }
        // ---- 4b. flip 批（真翻转降级链）：成批 ExactYBatch 精确 y → 精确 idx/yhat
        //    （Compare/Select 无算术）→ r/rnorm/gamma → rhat → qjl 全行向量链
        //    （质心 rank-1 变化 ~±0.25 破坏非命中位 cube 符号 → 必须全行重算；
        //    S^T staging + 广播 Muls + 向量 Add，语序同 RecomputeTieVecVec 逐位等价）
        //    → idx/norm/gamma/qjl 全写出（替代 EpilogueFullRecompute 44µs 全链）
        for (int32_t g0 = 0; g0 < nFlip_; g0 += 4) {
            int32_t nv = (nFlip_ - g0 < 4) ? (nFlip_ - g0) : 4;
            int64_t fvs[4];
            for (int32_t u = 0; u < nv; u++) {
                fvs[u] = flipGv_[g0 + u];
            }
            SyncMte3S();
            {
                event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(e);
                WaitFlag<HardEvent::V_MTE2>(e);
            }
            for (int32_t u = 0; u < nv; u++) {
                DataCopy(xT_[u * D], kvGm_[fvs[u] * D], D);
            }
            SyncMte2V();
            for (int32_t u = 0; u < nv; u++) {
                Cast(dupT_[u * D], xT_[u * D], RoundMode::CAST_NONE, D);
            }
            for (int32_t u = 0; u < nv; u++) {
                Mul(sqT_[u * D], dupT_[u * D], dupT_[u * D], D);
            }
            for (int32_t off = 64; off >= 8; off >>= 1) { // 成批 TreeDot（同 pass2/4 语序）
                Add(sqT_[0], sqT_[0], sqT_[off], off, nv, BinaryRepeatParams(1, 1, 1, 16, 16, 16));
            }
            SyncVS();
            float fnorms[4];
            for (int32_t u = 0; u < nv; u++) {
                float s0 = sqT_.GetValue(u * D);
                for (int32_t t = 1; t < 8; t++) {
                    s0 = s0 + sqT_.GetValue(u * D + t);
                }
                fnorms[u] = sqrt(s0);
            }
            for (int32_t u = 0; u < nv; u++) {
                float invN = 1.0f / (fnorms[u] > 1e-30f ? fnorms[u] : 1e-30f);
                Muls(dupT_[u * D], dupT_[u * D], invN, D);
            }
            int32_t uRow[4] = {0, 1, 2, 3};
            ExactYBatch(uRow, nv); // yhatT_ 行 u = 精确 y（严格 k 升序分离舍入）
            for (int32_t u = 0; u < nv; u++) {
                // 精确 idx 序号（idxFT_ 槽 0，PackIdx(1,0) 槽位约定）+ yhat（qjlBT_）
                Duplicate(idxFT_, 0.0f, D);
                Duplicate(qjlBT_, CentTable<BITS>::CENT[0], D);
                for (int32_t k = 0; k < BOUND_CNT; k++) {
                    float b = (CentTable<BITS>::CENT[k] + CentTable<BITS>::CENT[k + 1]) * 0.5f;
                    CompareScalar(maskT_, yhatT_[u * D], b, CMPMODE::GT, D);
                    Duplicate(sqT_, CentTable<BITS>::CENT[k + 1], D);
                    Select(qjlBT_, maskT_, sqT_, qjlBT_, SELMODE::VSEL_TENSOR_TENSOR_MODE, D);
                    Duplicate(sqT_, static_cast<float>(k + 1), D);
                    Select(idxFT_, maskT_, sqT_, idxFT_, SELMODE::VSEL_TENSOR_TENSOR_MODE, D);
                }
                SyncVS();
                Sub(yhatT_[u * D], yhatT_[u * D], qjlBT_, D); // r = y - yhat（原地）
                Mul(sqT_, yhatT_[u * D], yhatT_[u * D], D);
                float rnorm = sqrt(TreeDotReduce8());
                float gamma = fnorms[u] * rnorm; // = golden gamma
                float invR = 1.0f / (rnorm > 1e-30f ? rnorm : 1e-30f);
                Muls(yhatT_[u * D], yhatT_[u * D], invR, D); // rhat（原地）
                SyncVS();
                // qjl 全行向量链（语序同 RecomputeTieVecVec）
                Duplicate(rT_, 0.0f, D);
                auto stFp = rLoBuf_.Get<float>(); // 16 行 staging（pass4 已消费，空闲）
                for (int32_t kc = 0; kc < D / 16; kc++) {
                    {
                        event_t e = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                        SetFlag<HardEvent::V_MTE2>(e);
                        WaitFlag<HardEvent::V_MTE2>(e);
                    }
                    DataCopy(stFp, stGm_[kc * 16 * D], 16 * D);
                    SyncMte2V();
                    for (int32_t jj = 0; jj < 16; jj++) {
                        float rk = yhatT_.GetValue(u * D + kc * 16 + jj);
                        Muls(sqT_, stFp[jj * D], rk, D);
                        Add(rT_, rT_, sqT_, D);
                    }
                }
                // 符号 → 0/1（acc >= 0 ⇒ 1，与全链三元一致）
                Duplicate(dupT_, 1.0f, D);
                Duplicate(qjlBT_, 0.0f, D);
                Compare<float>(maskT_, rT_, qjlBT_, CMPMODE::GE, D);
                Select(rT_, maskT_, dupT_, qjlBT_, SELMODE::VSEL_TENSOR_TENSOR_MODE, D);
                SyncSV();
                __ubuf__ uint8_t *tieQjl = reinterpret_cast<__ubuf__ uint8_t *>(tieQjlT_.GetPhyAddr());
                for (int32_t g = 0; g < D / 8; g++) {
                    uint32_t qword = 0;
                    for (int32_t lane = 0; lane < 8; lane++) {
                        int32_t e = g * 8 + lane;
                        qword |= static_cast<uint32_t>(static_cast<int32_t>(rT_.GetValue(e))) << lane;
                    }
                    tieQjl[g] = static_cast<uint8_t>(qword & 0xFFU);
                }
                gamArrT_.SetValue(0, gamma);
                normArrT_.SetValue(0, fnorms[u]);
                SyncSMte3();
                PackIdx(1, 0); // idxFT_ → outIdxT_（结尾自带 PipeBarrier<PIPE_ALL>）
                SyncSMte3();
                DataCopyExtParams ep;
                ep.blockCount = 1;
                ep.blockLen = IDX_BYTES;
                ep.srcStride = 0;
                ep.dstStride = 0;
                DataCopyPad(idxGm_[fvs[u] * IDX_BYTES], outIdxT_, ep);
                DataCopyExtParams np;
                np.blockCount = 1;
                np.blockLen = 2;
                np.srcStride = 0;
                np.dstStride = 0;
                if (outDtype_ == 1) {
                    auto epNormF16 = outNormBf16Buf_.Get<half>();
                    auto epGamF16 = outGamBf16Buf_.Get<half>();
                    Cast(epNormF16, normArrT_[0], RoundMode::CAST_RINT, 1);
                    Cast(epGamF16, gamArrT_[0], RoundMode::CAST_RINT, 1);
                    SyncVMte3();
                    DataCopyPad(normGmF16_[fvs[u]], epNormF16, np);
                    DataCopyPad(gamGmF16_[fvs[u]], epGamF16, np);
                } else {
                    Cast(outNormBf16T_, normArrT_[0], RoundMode::CAST_RINT, 1);
                    Cast(outGamBf16T_, gamArrT_[0], RoundMode::CAST_RINT, 1);
                    SyncVMte3();
                    DataCopyPad(normGm_[fvs[u]], outNormBf16T_, np);
                    DataCopyPad(gamGm_[fvs[u]], outGamBf16T_, np);
                }
                SyncSMte3();
                DataCopyExtParams qp;
                qp.blockCount = 1;
                qp.blockLen = QJL_BYTES;
                qp.srcStride = 0;
                qp.dstStride = 0;
                DataCopyPad(qjlOutGm_[fvs[u] * QJL_BYTES], tieQjlT_, qp);
            }
        }
        // ---- 5. 统一全站点：所有全链重算（flip/full/gamma 不符/qjl full）仅此一处内联副本 ----
        // （此前 4 个调用点各内联一份 RecomputeTieVec+PackIdx，尾声代码体积 ×4；空尾声消融
        // 实测代码存在本身拖慢热循环 65~85µs——寄存器分配/I-cache 税）
        for (int32_t i = 0; i < nFull_; i++) {
            EpilogueFullRecompute(fullGv_[i], fullQjl_[i] != 0);
        }
        // 病态兜底：fullList 溢出标记（fl|4）的候选
        for (int32_t j = 0; j < stashYC_; j++) {
            int32_t fl = stashYFl_.GetValue(j);
            if ((fl & 4) != 0 && (fl & 2) == 0) {
                EpilogueFullRecompute(static_cast<int64_t>(stashYGv_.GetValue(j)), false);
            }
        }
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

        // H^T/S^T 预转置（fp32 大 case）：放在 tile0 上传之后，与 AIC 的 GEMM1(tile0) 重叠；
        // 每个 AIV 各转一份到自己的 workspace 区，无需跨 sub-block 同步。
        if (useVecRc_) {
            StageTranspose(rotSrcGm_, htGm_);
            StageTranspose(qjlSrcGm_, stGm_);
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
                DetectAndRecomputeTies(vb, n, s);
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
                // p = p1 + p2 + p3：三平面 fp32 直读求和（B11：Fixpipe 改 fp32 出，免 Cast，
                // p 误差由 fp16 舍入 ~1e-5 降到 ~5e-7，qjl 检测阈值随之收紧到 1.5e-6）
                DataCopy(yT_, p1Gm_[vb * HEAD_DIM], elems);
                DataCopy(rT_, p2Gm_[vb * HEAD_DIM], elems);
                SyncMte2V();
                Add(yT_, yT_, rT_, elems); // yT_ = p1+p2
                {
                    event_t e3b = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                    SetFlag<HardEvent::V_MTE2>(e3b);
                    WaitFlag<HardEvent::V_MTE2>(e3b);
                }
                DataCopy(rT_, p3Gm_[vb * HEAD_DIM], elems);
                SyncMte2V();
                Add(yT_, yT_, rT_, elems); // yT_ = p1+p2+p3
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
                // qjl 符号边界：p=r_hat@S 近 0（fp16 三平面累加误差实测 ~1e-5 量级，阈值取 3e-5≈3×）
                // 时 Cube 符号可能翻转，且这些向量的 y/gamma 可能都不在边界上（不触发 quant 阶段的
                // tie），需在此单独重算 qjl。
                // 向量化检测：|p| < thr 逐元素比较 → 0/1 → 两级 BlockReduceSum 聚合每向量命中数
                // （rT_ 的 p3/qjlBT_ 的 qjl 位/dupT_ 的打包值均已消费完，sqT_ 作归约暂存）
                Abs(rT_, yT_, elems);
                CompareScalar(maskT_, rT_, 2.0e-5f, CMPMODE::LT, elems);
                Duplicate(rT_, 1.0f, elems);
                Duplicate(qjlBT_, 0.0f, elems);
                Select(qjlBT_, maskT_, rT_, qjlBT_, SELMODE::VSEL_TENSOR_TENSOR_MODE, elems);
                BlockReduceSum(sqT_, qjlBT_, static_cast<int32_t>(elems / 64), 64, 1, 1, 8);
                BlockReduceSum(sqT_[SUB_ELEMS / 2], sqT_, static_cast<int32_t>((elems / 8 + 63) / 64), 64, 1, 1, 8);
                SyncVS();
                bool rcFlag[CUBE_SUBBATCH];
                for (int32_t v = 0; v < n; v++) {
                    rcFlag[v] = false;
                    if (cascVec_[s * CUBE_SUBBATCH + v]) {
                        continue; // 已级联的向量 qjl 已由全链重算精确写出
                    }
                    rcFlag[v] = (sqT_.GetValue(SUB_ELEMS / 2 + v * 2) + sqT_.GetValue(SUB_ELEMS / 2 + v * 2 + 1)) > 0.5f;
                }
                SyncSV();
                { // 顺延登记 qjl 候选（命中过滤口径同旧外科；cube qjl 行 16B 进 UB 暂存；
                  // 溢出回退就地 qjlOnly 级联——延后到收集循环后，因为 RecomputeTieVec 会
                  // clobber yT_（后续候选的 |p| 过滤还要读））
                    int32_t tmpHits[8];
                    int32_t ovQ_[CUBE_SUBBATCH];
                    int32_t nOvQ_ = 0;
                    __ubuf__ uint32_t *qsrc32 =
                        reinterpret_cast<__ubuf__ uint32_t *>(outQjlT_.GetPhyAddr());
                    for (int32_t v = 0; v < n; v++) {
                        if (!rcFlag[v]) {
                            continue;
                        }
                        int32_t raw[16];
                        int32_t nraw = CollectHitsFromMask(v, raw, 16);
                        int32_t nh = 0;
                        bool full = (nraw > 16);
                        for (int32_t t = 0; t < nraw && t < 16; t++) {
                            float pj = yT_.GetValue(v * HEAD_DIM + raw[t]);
                            float ap = (pj < 0.0f) ? -pj : pj;
                            if (ap > 1.5e-5f) {
                                continue; // maxDp≤8.2e-6 翻不了符号，免检
                            }
                            if (nh < 8) {
                                tmpHits[nh] = raw[t];
                            }
                            nh++;
                        }
                        if (nh > 8) {
                            full = true;
                            nh = 8;
                        }
                        if (nh == 0 && !full) {
                            continue; // 全部免检通过
                        }
                        if (stashQC_ < STASH_QCAP) {
                            stashQGv_.SetValue(stashQC_, static_cast<int32_t>(vb + v));
                            stashQNh_.SetValue(stashQC_, nh);
                            stashQFl_.SetValue(stashQC_, full ? 2 : 0);
                            for (int32_t t = 0; t < nh; t++) {
                                stashQHit_.SetValue(stashQC_ * 8 + t, tmpHits[t]);
                            }
                            for (int32_t t = 0; t < QJL_BYTES / 4; t++) { // cube qjl 行（同 S 管道有序）
                                stashQRow_.SetValue(stashQC_ * (QJL_BYTES / 4) + t,
                                                    static_cast<int32_t>(qsrc32[v * (QJL_BYTES / 4) + t]));
                            }
                            stashQC_++;
                        } else {
                            ovQ_[nOvQ_++] = v;
                        }
                    }
                    for (int32_t i = 0; i < nOvQ_; i++) { // 收集完成后再级联（yT_ 的 p 已消费完）
                        RecomputeTieVec(vb + ovQ_[i], s * CUBE_SUBBATCH + ovQ_[i], true);
                        cascVec_[s * CUBE_SUBBATCH + ovQ_[i]] = true;
                    }
                }
                // 级联向量 qjl 覆盖为精确重算值（外科翻 bit 的已直接写在 outQjlT_）
                for (int32_t v = 0; v < n; v++) {
                    if (cascVec_[s * CUBE_SUBBATCH + v]) {
                        __ubuf__ uint8_t *qsrc = reinterpret_cast<__ubuf__ uint8_t *>(tieQjlT_.GetPhyAddr()) +
                                                 (s * CUBE_SUBBATCH + v) * QJL_BYTES;
                        __ubuf__ uint8_t *qdst = reinterpret_cast<__ubuf__ uint8_t *>(outQjlT_.GetPhyAddr()) +
                                                 v * QJL_BYTES;
                        for (int32_t b = 0; b < QJL_BYTES; b++) {
                            qdst[b] = qsrc[b];
                        }
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

        // 主流水全速跑完后，集中处理顺延候选（各 AIV 并行，退出关键路径）
        EpilogueSurgery();
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

    // fp32 L0C → fp16 GM（F322F16 就地量化出 fp16）。用 CFG_ROW_MAJOR 的模板 Fixpipe
    // 让输出按行主序落 GM（默认 CFG_NZ 的 fp16 是 fractal 排布，AIV 按行主序读会错位）。
    __aicore__ inline void FixTileF16(const GlobalTensor<half> &dstGm, int64_t vecBase, int32_t nRows)
    {
        SyncMFix();
        FixpipeParamsV220 fp;
        fp.nSize = HEAD_DIM;
        fp.mSize = static_cast<uint16_t>(nRows);
        fp.srcStride = CUBE_MMAD_M;
        fp.dstStride = HEAD_DIM;
        fp.ndNum = 1;
        fp.reluEn = false;
        fp.quantPre = QuantMode_t::F322F16;
        fp.unitFlag = 0;
        Fixpipe<half, float, CFG_ROW_MAJOR>(dstGm[vecBase * HEAD_DIM], c1T_, fp);
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

            // GEMM2: 三个独立 fp16 平面 p1=rh@Sh / p3=rh@Sl / p2=rl@Sh，各自 Mmad(init)
            // + Fixpipe 出 fp16（F322F16，CFG_ROW_MAJOR 行主序落 GM），不做 L0C 累加 —— 累加会
            // 抹掉修正项导致 qjl 符号整体翻转。AIV 侧 fp32 求和 p=p1+p2+p3 再取符号。
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
                Mmad(c1T_, a2F16T_, b2F16T_, MakeMmadParams(true)); // p1 = rh@Sh
                SyncMMte1();
                FixTile(p1Gm_, vb, nt); // fp32 出（B11）
                LoadBTile<half>(b2F16T_, sLoNzT_);
                SyncMte1M();
                Mmad(c1T_, a2F16T_, b2F16T_, MakeMmadParams(true)); // p3 = rh@Sl（复用同一 L0A）
                SyncMMte1();
                FixTile(p3Gm_, vb, nt);
                {
                    event_t eD = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_MTE2));
                    SetFlag<HardEvent::MTE1_MTE2>(eD);
                    WaitFlag<HardEvent::MTE1_MTE2>(eD);
                }
                Nd2NzTileF16(rLoGm_, vb);
                LoadATile<half>(a2F16T_, a1F16T_, HEAD_DIM);
                LoadBTile<half>(b2F16T_, sHiNzT_);
                SyncMte1M();
                Mmad(c1T_, a2F16T_, b2F16T_, MakeMmadParams(true)); // p2 = rl@Sh
                SyncMMte1();
                FixTile(p2Gm_, vb, nt);
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
    GlobalTensor<float> p1Gm_, p2Gm_, p3Gm_;
    GlobalTensor<float> htGm_, stGm_; // H^T/S^T（fp32，向量化重算用；仅 useVecRc_ 时有效）

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
    bool tieVec_[CUBE_TILE_V];
    bool cascVec_[CUBE_TILE_V]; // 真正走了全链级联重算的向量（qjl 覆盖源 = tieQjlT_）
    LocalTensor<uint8_t> tieQjlT_;
    TBuf<QuePosition::VECCALC> tieQjlBuf_;
    uint64_t offHh_ = 0, offHl_ = 0, offHll_ = 0, offSh_ = 0, offSl_ = 0;
    uint64_t offHT_ = 0, offST_ = 0;
    uint64_t offX_ = 0, offRh_ = 0, offRl_ = 0, offZ_ = 0, offP1_ = 0, offP2_ = 0, offP3_ = 0;
    // 顺延外科暂存（全 UB，无跨核共享区）：溢出（病态输入）回退就地全链级联
    static constexpr int32_t STASH_YCAP = 128;
    static constexpr int32_t STASH_QCAP = 192;
    static constexpr int32_t STASH_GCAP = 8;
    int32_t stashYC_ = 0, stashQC_ = 0, gamYC_ = 0;
    static constexpr int32_t FULL_CAP = 24; // 统一全站点容量（溢出回退即时/标记全链）
    int32_t qexSlots_[STASH_QCAP]; // pass4 qjlExact 批的暂存槽位清单
    static constexpr int32_t FLIP_CAP = 8; // flip 降级批容量（实测 ~1-2/case；满则回退全链）
    int64_t flipGv_[FLIP_CAP];
    int32_t nFlip_ = 0;
    int32_t nQex_ = 0;
    int64_t fullGv_[FULL_CAP];
    uint8_t fullQjl_[FULL_CAP];
    int32_t nFull_ = 0;
    TBuf<QuePosition::VECCALC> stashBuf_;
    LocalTensor<int32_t> stashYGv_, stashYNh_, stashYFl_, stashYHit_, stashYIv_;
    LocalTensor<int32_t> gamGv_, gamIdx_;
    LocalTensor<int32_t> stashQGv_, stashQNh_, stashQFl_, stashQHit_, stashQRow_;
    bool useVecRc_ = false; // fp32 矩阵 + 大 case：向量化重算（H^T/S^T 预转置 + j 维并行 FMA）
    uint32_t matDtype_ = 0; // rotation/qjl_matrix 元素类型：0=float32，1=bfloat16
    uint32_t outDtype_ = 0; // quant_norm/quant_gamma 元素类型：0=bfloat16，1=float16
};

} // namespace KvCacheTurboQuant

#endif // KERNEL_KV_CACHE_TURBO_QUANT_CUBE_H
