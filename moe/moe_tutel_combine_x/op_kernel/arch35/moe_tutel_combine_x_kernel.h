/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * =============================================================================
 * op_kernel/arch35/moe_tutel_combine_x_kernel.h
 * =============================================================================
 * Role: Main Ascend C device kernel class for MoeTutelCombineX on arch35
 *        (Ascend 950): `template <typename T, bool isEmptyTensor, bool
 *        hasBatch> class MoeTutelCombineXKernel` — the shared normal-path
 *        class covering tilingKey 0 (normal-no-batch, 2D [EC, H]) and
 *        tilingKey 2 (normal-batch, 3D [B, EC, H]); only the non-empty
 *        combinations are instantiated by the kernel entry
 *        (op_kernel/moe_tutel_combine_x_apt.cpp).
 *
 * Dataflow (serial 3-stage + double loop):
 *   GM metadata scalar reads (ReadGmByPassDCache, not via UB)
 *     -> predicate dispatch
 *     -> valid row: y_grad row segment GM->UB (MTE2, DataCopyPad)
 *                  -> VF gated multiply (V: xBuf = yBuf x gate)
 *                  -> x_grad row segment UB->GM (MTE3, DataCopyPad)
 *     -> invalid row: zeroBuf zero template UB->GM (MTE3) — no y_grad read.
 *
 * UB plan: three line buffers yBuf/xBuf/zeroBuf, each
 *   alignUp(hChunk, 64)*sizeof(T) bytes (within the per-buffer UB budget),
 *   plus two 32-byte gate-scalar staging buffers (16b paths). No TQue, no
 *   double-buffering (queue depth 1), P = 3.
 *
 * Sync (woven into Process): MTE2_V / V_MTE3 tight pairs per valid chunk +
 *   MTE3_V cross-chunk WAR guard on xBuf (xBufInFlight state, first chunk
 *   skips the wait) + V_MTE2 cross-chunk WAR guard on yBuf (yBufVfInFlight
 *   state, first chunk skips the wait; closes the yBuf read-before-overwrite
 *   race). No cross-core SyncAll (token ranges are disjoint; single-writer
 *   per output row).
 *
 * Operator name variants:
 *   - PascalCase:   MoeTutelCombineX
 *   - snake_case:   moe_tutel_combine_x
 *   - UPPER_SNAKE:  MOE_TUTEL_COMBINE_X
 * =============================================================================
 */

#ifndef MOE_TUTEL_COMBINE_X_KERNEL_H_
#define MOE_TUTEL_COMBINE_X_KERNEL_H_

#include "kernel_operator.h"
#include "moe_tutel_combine_x_tiling_struct.h"

// =============================================================================
// CastTrait definitions (namespace scope; widening and narrowing must not
// share one trait). Round-trip semantics:
//   b16 -> fp32 widening: sat=UNKNOWN, round=CAST_NONE
//   fp32 -> b16 narrowing: sat=NO_SAT (fp16 overflow reverts to +/-Inf, never
//   saturates to MAX), round=CAST_RINT (round-to-nearest-even).
// =============================================================================
// 扩位 b16 -> fp32（sat=UNKNOWN, round=CAST_NONE）
constexpr AscendC::Reg::CastTrait kCastTraitB16ToF32 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                        AscendC::Reg::MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::CAST_NONE};

// 缩位 fp32 -> b16（fp16 / bf16 统一：sat=NO_SAT, round=CAST_RINT 即 round-to-nearest-even）
constexpr AscendC::Reg::CastTrait kCastTraitF32ToB16 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
                                                        AscendC::Reg::MaskMergeMode::ZEROING,
                                                        AscendC::RoundMode::CAST_RINT};

// =============================================================================
// Shared row-segment gated-multiply VF chain:
//   xUb[0:segLen) = yUb[0:segLen) x gateScalar
// fp32:  LoadAlign -> Muls -> StoreAlign                    (3-segment chain)
// 16b:   LoadAlign(DIST_UNPACK_B16) -> Cast fp32 -> Muls
//        -> Cast(CAST_RINT, NO_SAT) -> StoreAlign(DIST_PACK_B32)  (5 segments)
// The fp32 intermediates of the 16b path live in the register domain
// (RegTensor), not in UB. Single-writer chain: x_grad is never read here.
// =============================================================================
template <typename T>
__simd_vf__ inline void MoeTutelCombineXRowVf(__ubuf__ T *yUb, __ubuf__ T *xUb, float gateScalar, uint32_t totalElems,
                                              uint16_t repeatTime)
{
    constexpr bool kIsB16 = !AscendC::IsSameType<T, float>::value;
    constexpr uint32_t kStep = 256 / sizeof(float); // = 64：fp32 lane 数 = b16 每 repeat 解包元素数
    AscendC::Reg::RegTensor<float> fReg;
    AscendC::Reg::RegTensor<T> bReg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kStep);
        mask = AscendC::Reg::UpdateMask<float>(remaining); // 引用传递，内部自动递减 VL_T=64，勿手动再减
        if constexpr (kIsB16) {
            AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bReg, yUb + off);
            AscendC::Reg::Cast<float, T, kCastTraitB16ToF32>(fReg, bReg, mask);
            AscendC::Reg::Muls(fReg, fReg, gateScalar, mask);
            AscendC::Reg::Cast<T, float, kCastTraitF32ToB16>(bReg, fReg, mask);
            AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(xUb + off, bReg, mask);
        } else {
            AscendC::Reg::LoadAlign(fReg, reinterpret_cast<__ubuf__ float *>(yUb) + off);
            AscendC::Reg::Muls(fReg, fReg, gateScalar, mask);
            AscendC::Reg::StoreAlign(reinterpret_cast<__ubuf__ float *>(xUb) + off, fReg, mask);
        }
    }
}

// =============================================================================
// MoeTutelCombineXKernel — normal path (key=0 / key=2)
// =============================================================================
template <typename T, bool isEmptyTensor, bool hasBatch>
class MoeTutelCombineXKernel {
public:
    __aicore__ inline MoeTutelCombineXKernel() {}

    // -- Init：GM 绑定 / TBuf 分配 / zeroBuf 零模板一次置零 / capacity 哨兵校正 /
    //    本核 token 区间现算 --
    __aicore__ inline void Init(GM_ADDR ins[kMaxInputSlots], GM_ADDR outs[kMaxOutputSlots],
                                const MoetutelcombinexTilingData *td)
    {
        td_ = td;
        // -- GM 绑定（y_grad [EC,H] / [B,EC,H]；gates.indices.locations 元数据；x_grad）--
        gmY_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(ins[0]));
        gmIndices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(ins[2]));
        gmLocations_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(ins[3]));
        gmX_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(outs[0]));
        // 标量读专用裸指针（ReadGmByPassDCache 以 __gm__ 指针为实参）
        rawGates_ = reinterpret_cast<__gm__ T *>(ins[1]);
        rawIndices_ = reinterpret_cast<__gm__ int32_t *>(ins[2]);
        rawLocations_ = reinterpret_cast<__gm__ int32_t *>(ins[3]);
        // -- TBuf 分配（三路同规格，容量 = alignUp(hChunk,64)*sizeof(T)，
        //    64 元素粒度同时满足 zeroBuf Duplicate 分配粒度与 32B 对齐）--
        const int64_t bufElems = ((td->hChunk + 63) / 64) * 64; // zeroBuf 置零元素数
        const int64_t bufBytes = bufElems * static_cast<int64_t>(sizeof(T));
        pipe_.InitBuffer(yBuf_, static_cast<uint32_t>(bufBytes));    // CopyIn y 行段
        pipe_.InitBuffer(xBuf_, static_cast<uint32_t>(bufBytes));    // VF 结果 / CopyOut x 行段
        pipe_.InitBuffer(zeroBuf_, static_cast<uint32_t>(bufBytes)); // 零模板，全 kernel 常驻
        pipe_.InitBuffer(gateTmp_, 32); // 1 元素 T，32B 对齐（16b gate Cast 中转）
        pipe_.InitBuffer(gateS_, 32);   // 1 元素 float（16b gate Cast 结果）
        // -- 参数预取（TilingData 字段 -> 寄存器副本）--
        batchSize_ = td->batchSize;
        samples_ = td->samples;
        hidden_ = td->hidden;
        eC_ = td->eC;
        tokenTotal_ = td->tokenTotal;
        hChunk_ = td->hChunk;
        hChunkCnt_ = td->hChunkCnt;
        hChunkTail_ = td->hChunkTail;
        // -- zeroBuf 零模板：Duplicate(0) 一次置零 + PipeBarrier --
        // 执行后: 持有=[zeroBuf]（常驻；置零结果对后续零行 CopyOut 的 MTE3 读可见）
        AscendC::LocalTensor<T> zero = zeroBuf_.Get<T>();
        AscendC::Duplicate(zero, static_cast<T>(0), static_cast<int32_t>(bufElems));
        AscendC::PipeBarrier<PIPE_ALL>();
        // -- capacity 哨兵校正（host 不可读 indices 时 capacity=EC 哨兵）--
        // capacity == eC && eC > 0 => E=1 哨兵或真实 E=1，两种来源重扫结论一致；
        // 每核独立扫描同一只读数据，无跨核同步。扫描界 B*S（key=0 时 B=1 即 S）。
        capacity_ = td->capacity;
        if (capacity_ == eC_ && eC_ > 0) {
            int32_t maxIdx = -1;
            for (int64_t t = 0; t < tokenTotal_; ++t) {
                const int32_t v = AscendC::ReadGmByPassDCache<int32_t>(rawIndices_ + t);
                if (v > maxIdx) {
                    maxIdx = v;
                }
            }
            const int64_t e = (maxIdx >= 0) ? (static_cast<int64_t>(maxIdx) + 1) : 1; // 全 -1 记 E=1
            capacity_ = eC_ / e; // E > EC 时自然得 0（谓词 idx1 < C 全假，输出全零）
        }
        // -- 本核 token 区间（现算，不占 TilingData；核间切分公式，连续无交叠）--
        const int64_t blk = static_cast<int64_t>(AscendC::GetBlockIdx());
        if (blk < static_cast<int64_t>(td->bigCoreCnt)) {
            tokenStart_ = blk * (td->coreTokenBase + 1);
            tokenEnd_ = tokenStart_ + td->coreTokenBase + 1;
        } else {
            tokenStart_ = static_cast<int64_t>(td->bigCoreCnt) * (td->coreTokenBase + 1) +
                          (blk - static_cast<int64_t>(td->bigCoreCnt)) * td->coreTokenBase;
            tokenEnd_ = tokenStart_ + td->coreTokenBase;
        }
    }

    // -- Process：token 主循环 x H 维 chunk 循环，谓词分派 + 流水同步织入 --
    __aicore__ inline void Process()
    {
        // 事件 ID 经 FetchEventID 获取（禁硬编码）
        const event_t evMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        const event_t evVtoMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        const event_t evMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
        // yBuf WAR 守卫事件（V 读 yBuf -> 下一次 MTE2 覆写 yBuf）：
        // VF 后 SetFlag<V_MTE2>，下一次 CopyInYRow 前 WaitFlag<V_MTE2>（首块免 Wait）。
        const event_t evVtoMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        bool xBufInFlight = false;   // 是否有在飞的 xBuf CopyOut（MTE3_V WAR 守卫状态）
        bool yBufVfInFlight = false; // 是否有在飞的 yBuf VF 读（V_MTE2 WAR 守卫状态）

        // 执行前: 持有=[zeroBuf]（Init 置零常驻）
        for (int64_t token = tokenStart_; token < tokenEnd_; ++token) {
            // -- (b, t) 解码与段基址（key=0：b = 0 静态折叠、无除法；key=2：b = token / S）--
            int64_t ySegBase = 0; // y_grad 段基址（元素）= b*EC*H
            if constexpr (hasBatch) {
                const int64_t b = token / samples_; // b = token / S（int64 除法）
                ySegBase = b * eC_ * hidden_;
            }
            const int64_t metaIdx = token; // 元数据线性下标：key=0 t=token / key=2 b*S+t=token
            // -- 元数据标量读（不占 UB；indices / locations 只读不改写）--
            const int32_t idx0 = AscendC::ReadGmByPassDCache<int32_t>(rawIndices_ + metaIdx);
            const int32_t idx1 = AscendC::ReadGmByPassDCache<int32_t>(rawLocations_ + metaIdx);
            const float gate = LoadGateScalar(metaIdx);
            // -- S1: srcRow = idx0*C + idx1（int64 寻址）--
            const int64_t srcRow = static_cast<int64_t>(idx0) * capacity_ + static_cast<int64_t>(idx1);
            // -- S2: 五条件有效性谓词 --
            //    核心守卫 idx0>=0 且 idx1<C；防御收敛 idx1>=0 且 0<=srcRow<EC（契约外元数据折零贡献）
            const bool valid = (idx0 >= 0) && (idx1 >= 0) && (static_cast<int64_t>(idx1) < capacity_) &&
                               (srcRow >= 0) && (srcRow < eC_);
            const int64_t xRowBase = token * hidden_; // x_grad 行基址（b*S*H + t*H = token*H）
            // -- H 维 chunk 循环（同一行所有 chunk 共享谓词与 gate）--
            for (int64_t k = 0; k < hChunkCnt_; ++k) {
                const int64_t segLen = (k == hChunkCnt_ - 1) ? hChunkTail_ : hChunk_;
                const int64_t segOff = k * hChunk_;
                if (valid) {
                    // 执行前: 持有=[zeroBuf]
                    if (xBufInFlight) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(evMte3V); // WAR: 上轮 MTE3 读完 xBuf，V 才可再写
                    }
                    if (yBufVfInFlight) {
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                            evVtoMte2); // WAR: 上轮 VF 读完 yBuf，MTE2 才可覆写
                    }
                    CopyInYRow(ySegBase + srcRow * hidden_ + segOff, segLen); // MTE2 -> yBuf（gather 取行）
                    // 执行中: 持有=[zeroBuf, yBuf]
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMte2V);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMte2V); // V 等 yBuf 就绪
                    GatedRowCompute(gate, segLen);                          // 门控乘（VF 链）
                    // 执行中: 持有=[zeroBuf, yBuf, xBuf]   <- P_T=3 峰值（16b 的 fp32 中间驻寄存器域不计入）
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoMte2); // V 读完 yBuf，下一次 MTE2 覆写解禁
                    yBufVfInFlight = true;
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoMte3);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVtoMte3); // MTE3 等 xBuf 就绪；兼作 V 排空
                    CopyOutXRow(xRowBase + segOff, xBuf_, segLen);            // MTE3: xBuf -> GM
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(evMte3V);    // 标记 MTE3 读完 xBuf
                    xBufInFlight = true;
                    // 执行后: 持有=[zeroBuf]（yBuf / xBuf 下一 chunk / 下一 token 复用）
                } else {
                    // -- 无效行：零模板直写（容量丢弃 / 溢出容量 / 契约外防御收敛）--
                    // 执行中: 持有=[zeroBuf]   <- 无 yBuf / xBuf 参与，无越界读
                    CopyOutXRow(xRowBase + segOff, zeroBuf_, segLen);
                }
            }
        }
    }

private:
    // -- LoadGateScalar：逐 token 取 fp32 gate 标量。
    //    fp32：标量位型直读；16b：1 元素 UB 中转 Cast(CAST_NONE) 抬 fp32，
    //    前后 PipeBarrier<PIPE_ALL> 串行化标量域。--
    __aicore__ inline float LoadGateScalar(int64_t metaIdx)
    {
        if constexpr (AscendC::IsSameType<T, float>::value) {
            // fp32：标量直读（int32 位型读出后按 float 位型重解释）
            int32_t gb = AscendC::ReadGmByPassDCache<int32_t>(reinterpret_cast<__gm__ int32_t *>(rawGates_ + metaIdx));
            return *reinterpret_cast<float *>(&gb);
        } else {
            // 16b：1 元素 UB 中转 Cast fp32（CAST_NONE 扩位）
            int16_t gb = AscendC::ReadGmByPassDCache<int16_t>(reinterpret_cast<__gm__ int16_t *>(rawGates_ + metaIdx));
            T gt;
            *reinterpret_cast<int16_t *>(&gt) = gb;
            gateTmp_.Get<T>().SetValue(0, gt);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Cast(gateS_.Get<float>(), gateTmp_.Get<T>(), AscendC::RoundMode::CAST_NONE, 1);
            AscendC::PipeBarrier<PIPE_ALL>();
            return gateS_.Get<float>().GetValue(0);
        }
    }

    // -- CopyInYRow：y_grad 源行段 GM -> UB（有效条目专用；无效条目不搬入）--
    __aicore__ inline void CopyInYRow(int64_t yRowGmOff, int64_t segLen)
    {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen =
            static_cast<uint32_t>(segLen * static_cast<int64_t>(sizeof(T))); // 字节精度，不要求 32B 对齐
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0}; // 不 pad（越界保护由谓词保证不触发）
        AscendC::DataCopyPad(yBuf_.Get<T>(), gmY_[yRowGmOff], copyParams, padParams);
        // 同步: MTE2 写 yBuf -> V 读，事件对 MTE2_V
    }

    // -- GatedRowCompute：xBuf[0:segLen) = yBuf[0:segLen) x gateScalar，
    //    一条 asc_vf_call 吃掉整条共享 VF 链（MoeTutelCombineXRowVf），不拆分。--
    __aicore__ inline void GatedRowCompute(float gateScalar, int64_t segLen)
    {
        constexpr uint32_t kStep = 256 / sizeof(float); // = 64：fp32 lane 数 = b16 每 repeat 解包元素数
        const uint16_t repeatTime = static_cast<uint16_t>((segLen + kStep - 1) / kStep);
        asc_vf_call<MoeTutelCombineXRowVf<T>>(reinterpret_cast<__ubuf__ T *>(yBuf_.Get<T>().GetPhyAddr()),
                                              reinterpret_cast<__ubuf__ T *>(xBuf_.Get<T>().GetPhyAddr()), gateScalar,
                                              static_cast<uint32_t>(segLen), repeatTime);
    }

    // -- CopyOutXRow：x_grad 行段 UB -> GM（有效行 xBuf / 无效行 zeroBuf 共用）。
    //    每行单写覆盖。--
    __aicore__ inline void CopyOutXRow(int64_t xRowGmOff, AscendC::TBuf<AscendC::TPosition::VECCALC> &srcBuf,
                                       int64_t segLen)
    {
        AscendC::DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(segLen * static_cast<int64_t>(sizeof(T))); // 有效字节，不要求 32B 对齐
        params.srcStride = 0;
        params.dstStride = 0;
        AscendC::DataCopyPad(gmX_[xRowGmOff], srcBuf.Get<T>(), params); // MTE3: UB -> GM
        // 同步: V 写 xBuf -> MTE3 读（V_MTE3）与 MTE3 读 xBuf -> 下轮 V 写（MTE3_V）
    }

    // -- state --
    AscendC::TPipe pipe_;                            // 类成员 TPipe
    const MoetutelcombinexTilingData *td_ = nullptr; // Init 缓存（哨兵校正等直读）

    AscendC::GlobalTensor<T> gmY_;               // y_grad
    AscendC::GlobalTensor<int32_t> gmIndices_;   // indices
    AscendC::GlobalTensor<int32_t> gmLocations_; // locations
    AscendC::GlobalTensor<T> gmX_;               // x_grad
    __gm__ T *rawGates_ = nullptr;               // 标量读专用（LoadGateScalar）
    __gm__ int32_t *rawIndices_ = nullptr;       // 标量读专用
    __gm__ int32_t *rawLocations_ = nullptr;     // 标量读专用

    AscendC::TBuf<AscendC::TPosition::VECCALC> yBuf_;    // CopyIn y 行段
    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuf_;    // VF 结果 / CopyOut x 行段
    AscendC::TBuf<AscendC::TPosition::VECCALC> zeroBuf_; // 零模板（Init 一次置零，全 kernel 常驻）
    AscendC::TBuf<AscendC::TPosition::VECCALC> gateTmp_; // 1 元素 T（16b gate Cast 中转）
    AscendC::TBuf<AscendC::TPosition::VECCALC> gateS_;   // 1 元素 fp32（16b gate Cast 结果）

    // TilingData 预取副本
    int64_t batchSize_ = 0;
    int64_t samples_ = 0;
    int64_t hidden_ = 0;
    int64_t eC_ = 0;
    int64_t capacity_ = 0;
    int64_t tokenTotal_ = 0;
    int64_t hChunk_ = 0;
    int64_t hChunkCnt_ = 0;
    int64_t hChunkTail_ = 0;
    int64_t tokenStart_ = 0;
    int64_t tokenEnd_ = 0;
};

#endif // MOE_TUTEL_COMBINE_X_KERNEL_H_
