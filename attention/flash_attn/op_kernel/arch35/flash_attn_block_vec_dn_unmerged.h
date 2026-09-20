/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attn_block_vec_dn.h
 * \brief FANoQuantGqaBlockVecDnUnmerged —— Dn 路径专用 Vec Block 模板（独立类，无 base 基类）。
 */
#ifndef FLASH_ATTN_BLOCK_VEC_DN_UNMERGED_H_
#define FLASH_ATTN_BLOCK_VEC_DN_UNMERGED_H_

#include <limits>
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn.h"
#include "../../../common/op_kernel/arch35/vf/vf_flashupdate_new.h"
#include "vf_flashupdate_dn.h"
#include "../../../common/op_kernel/arch35/vf/vf_div_cast_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_flash_decode_arch35.h"
#include "../../../common/op_kernel/const_def.h"
#include "../../../common/op_kernel/vector_common.h"
#include "../../../common/op_kernel/init_output.h"
#include "memory_copy_arch35.h"
#include "../utils/attn_sink_gs1.h"
#include "../utils/attenmask_gs1.h"
#include "../../../common/op_kernel/arch_info.h"

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;

#include "flash_attn_dn_unmerged_scheduler.h"
#include "../utils/cv_task_scheduler.h"
#include "flash_attn_block_vec_flashdecode.h"

namespace FlashAttnKernel {
using ArchInfo::CV_RATIO;

struct DnUnmergedVecConstInfo {
    /* 轴长度 */
    uint32_t bSize;
    uint64_t t1Size;
    uint64_t t2Size;
    uint32_t dSize;
    uint32_t dSizeV;
    uint32_t dBasicBlock;
    uint32_t gSize; /* g轴的大小 */
    uint32_t n2Size;
    uint32_t n1Size = 0;      /* N1(query头)轴的大小, 不合轴 DN 预埋使用, 原 ND/合轴 DN 未使用 */
    uint64_t s1Size;          /* s1总大小 */
    uint64_t s2Size;          /* s2总大小 */
    uint64_t cuSeqLensQSize;  /* 用户输入的actualseq的长度 */
    uint64_t cuSeqLensKVSize; /* 用户输入的actualseq_kv的长度 */
    uint64_t seqUsedQSize;    /* 用户输入的seq used q的长度 */
    uint64_t seqUsedKvSize;   /* 用户输入的seq used kv的长度 */

    /* FA kernel meta */

    uint32_t sparseMode; // sparse
    uint32_t attenMaskBatch;
    uint32_t attenMaskS1Size;
    uint32_t attenMaskS2Size;
    int64_t preTokens;
    int64_t nextTokens;
    bool isExistRowInvalid;
    float scaleValue;

    /* 核信息 */
    uint32_t aicIdx;
    uint32_t aivIdx;
    uint8_t subBlockIdx;
    uint32_t coreNum;

    /* FA中间结果写出workspace信息 */
    uint32_t accumOutSize;
    uint32_t logSumExpSize;

    /* 输出shape */
    FA_LAYOUT outputLayout;
    bool needInitOutput;
    bool enableFlashDecode;
    uint32_t blockSize;
    uint32_t maxBlockNumPerBatch;
    uint32_t paLayoutType;
    /* kvcache非连续场景的strides, 0表示连续 */
    uint64_t keyBnStride = 0;
    uint64_t keyN2Stride = 0;
    uint64_t valueBnStride = 0;
    uint64_t valueN2Stride = 0;
    bool isSoftmaxLseEnable;
    bool learnableSinkFlag = false;
};
struct DnUnmergedVecRunInfo {
    uint32_t loop = 0;
    uint32_t mloop = 0;
    bool isValid = false;
    bool isFirstS2Loop = false;
    bool isLastS2Loop = false;

    uint32_t bIdx = 0;
    uint32_t n2Idx = 0;
    uint32_t n1Idx = 0; // 不合轴 DN 预埋: 当前 query 头(b*n1 空间), 原 ND/合轴 DN 未使用
    uint32_t gIdx = 0;
    uint32_t s1Idx = 0;
    uint32_t s2Idx = 0;
    uint64_t actS1Size = 1;   // 当前处理head的S1轴实际大小
    uint64_t actS2Size = 1;   // 当前处理head的S2轴实际大小
    uint32_t actMSize = 0;    // GS1方向上的长度
    uint32_t actMSizeAlign32; // GS1 方向上长度对齐
    uint32_t actVecMSize;     // VEC 视角, 基本块GS1方向长度
    uint32_t vecMbaseIdx;     // VEC 对应的M 轴起始位置,V0 为0， V1 为 V0的actVecMSize

    uint32_t actSingleLoopS2Size = 0; // S2方向长度
    uint32_t actSingleLoopS2SizeAlign;
    bool isS2SplitCore = false;
    uint32_t faTmpOutWsPos = 0; // FA阶段，S2外切，需要写到workspace时，写出到第几块M*D的GM块
    bool isFirstFdBlock = false;

    int64_t preTokensLeftUp = 0;
    int64_t nextTokensLeftUp = 0;
};

// Scores are [KV, Q]; mask bytes are [Q, KV]. Both mask modes use the
// same scale-before-padding order, including zero and negative scales.
template <bool HasMask, uint32_t MaskPitch>
__simd_vf__ inline void PrepareDnScoresVF(__ubuf__ float *scores, __ubuf__ uint32_t *mask, uint32_t rows,
                                          uint32_t pitch, uint32_t cols, uint32_t alignedCols, float scale,
                                          float paddingValue, bool needMask)
{
    Reg::RegTensor<int32_t> indices;
    Reg::RegTensor<uint32_t> words, bytes, byteMask;
    Reg::RegTensor<float> values, minimum, padding;
    Reg::MaskReg active = Reg::UpdateMask<float>(rows);
    Reg::MaskReg keep;
    if constexpr (HasMask) {
        Reg::Arange(indices, 0);
        Reg::Muls(indices, indices, static_cast<int32_t>(MaskPitch / 4U), active);
        Reg::Duplicate(minimum, -1000000000000.0F);
        Reg::Duplicate(byteMask, 255U);
    }
    for (uint16_t col = 0; col < cols; ++col) {
        Reg::LoadAlign(values, scores + col * pitch);
        Reg::Muls(values, values, scale, active);
        if constexpr (HasMask) {
            if (needMask) {
                Reg::Gather(words, mask + col / 4U, reinterpret_cast<Reg::RegTensor<uint32_t> &>(indices), active);
                Reg::ShiftRights(bytes, words, static_cast<int16_t>((col % 4U) * 8U), active);
                Reg::And(bytes, bytes, byteMask, active);
                Reg::Compares<uint32_t, CMPMODE::EQ>(keep, bytes, 0U, active);
                Reg::Select(values, values, minimum, keep);
            }
        }
        Reg::StoreAlign(scores + col * pitch, values, active);
    }
    Reg::Duplicate(padding, paddingValue);
    for (uint16_t col = cols; col < alignedCols; ++col) {
        Reg::StoreAlign(scores + col * pitch, padding, active);
    }
}

template <typename FA_T>
class FANoQuantGqaBlockVecDnUnmerged {
public:
    using RunInfo = DnUnmergedVecRunInfo;
    using ConstInfo = DnUnmergedVecConstInfo;
    static constexpr uint32_t PRELOAD_N = 2;
    using INPUT_T = typename FA_T::inputType;
    using OUTPUT_T = typename FA_T::outputType;
    static constexpr uint32_t mBaseSize = (uint32_t)FA_T::mBaseSize;
    static constexpr uint32_t s2BaseSize = (uint32_t)FA_T::s2BaseSize;
    static constexpr uint32_t dBaseSize = (uint32_t)FA_T::dBaseSize;
    static constexpr uint32_t dVBaseSize = (uint32_t)FA_T::dVBaseSize;
    static constexpr FA_LAYOUT LAYOUT_T = FA_T::qLayout;
    static constexpr FA_LAYOUT LAYOUT_KV = FA_T::kvLayout;
    static constexpr FA_LAYOUT LAYOUT_OUT = FA_T::attnOutLayout;
    static constexpr bool PAGE_ATTENTION = FA_T::pageAttention;
    static constexpr bool HAS_MASK = FA_T::hasMask;

    using T = float;
    static constexpr uint32_t dTemplateAlign64 = BaseApi::Align64Func((uint16_t)FA_T::dVBaseSize);

    static constexpr uint32_t DB = 2;
    // 索引使用 loop & (DB - 1) 代替 loop % DB，要求 DB 必须是2的幂，否则位掩码结果错误
    static_assert(DB > 0 && (DB & (DB - 1)) == 0, "DB must be a power of two for bitmask indexing");

    // 核间同步ID
    static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4;
    static constexpr uint32_t CC_MM1_0 = 0U;
    static constexpr uint32_t CC_MM2_0 = 2U;
    static constexpr uint32_t CC_L1P_0 = 5U;
    static constexpr uint32_t CC_L1P_1 = 6U;
    static constexpr uint32_t CC_L1P_2 = 7U;

    // 核内同步ID
    // MTE3<->V, 输出buffer
    static constexpr uint32_t UB_OUT_VEC2_RES_EVENT0 = 0;
    static constexpr uint32_t UB_OUT_VEC1_RES_EVENT0 = 2;
    static constexpr uint32_t UB_OUT_VEC1_RES_EVENT1 = 3;
    static constexpr uint32_t UB_OUT_LSE_OUT_EVENT0 = 4;
    static constexpr uint32_t UB_OUT_LSE_OUT_EVENT1 = 5;
    // MTE2<->V, mask 拷入buffer(FD block 占用 2/4/8/9/10/11, 6/7 空闲)
    static constexpr uint32_t UB_IN_MASK_EVENT0 = 6;

    // L1
    static constexpr uint32_t L1_P_BUFCNT = 3U;
    static constexpr uint32_t L1_P_BUF_BYTES = mBaseSize * s2BaseSize * sizeof(INPUT_T);
    LocalTensor<uint8_t> l1PBuffers_;

    // UB
    static constexpr uint32_t UB_MM1_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_MM1_RES_BUF_BYTES = mBaseSize / CV_RATIO * s2BaseSize * sizeof(T);
    static constexpr uint32_t UB_MM2_RES_BUFCNT = 1U;
    static constexpr uint32_t UB_MM2_RES_BUF_BYTES = mBaseSize / CV_RATIO * dVBaseSize * sizeof(T);
    LocalTensor<uint8_t> ubMm1ResBuffers_;
    LocalTensor<uint8_t> ubMm2ResBuffers_;
    uint32_t mm1ResUbBufId_ = 0U;
    uint32_t mm2ResUbBufId_ = 0U;
    static constexpr uint32_t UB_MM_RES_TOTAL_BYTES =
        UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES + UB_MM2_RES_BUFCNT * UB_MM2_RES_BUF_BYTES;
    static __aicore__ inline constexpr uint32_t GetFdBaseOffset()
    {
        return UB_MM_RES_TOTAL_BYTES;
    }
    static constexpr uint32_t UB_VEC2_RES_BUF_BYTES = mBaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
    LocalTensor<T> ubVec2Res_; // 存放vec2阶段VEC的中间处理结果, 并且作为attn_out的输出buffer, 需配对的MTE3和V的同步ID

    static constexpr uint32_t UB_VEC1_RES_BUFCNT = dVBaseSize <= 64U ? 2U : 1U;
    // DN VF 按 64 lanes 打包 P, 逻辑 Vector M 为 32(mBase=64) 时也按实际打包宽度定容
    static constexpr uint32_t UB_VEC1_RES_BUF_BYTES = (64U + 1U) * s2BaseSize * sizeof(INPUT_T);
    LocalTensor<uint8_t> ubVec1ResBuffers_;
    uint32_t vec1ResUbBufId_ = 0;

    static constexpr uint32_t UB_SOFTMAX_MAX_BUFCNT = 3U;
    static constexpr uint32_t UB_SOFTMAX_MAX_BUF_BYTES =
        AttentionCommon::Align(mBaseSize / CV_RATIO * static_cast<uint32_t>(sizeof(T)), 256U);
    LocalTensor<T> softmaxSumBuf_;
    static constexpr uint32_t UB_SOFTMAX_SUM_BUFCNT = 3U;
    static constexpr uint32_t UB_SOFTMAX_SUM_BUF_BYTES =
        AttentionCommon::Align(mBaseSize / CV_RATIO * static_cast<uint32_t>(sizeof(T)), 256U);
    LocalTensor<T> softmaxMaxBuf_;
    static constexpr uint32_t UB_SOFTMAX_EXP_BUFCNT = 3U;
    static constexpr uint32_t UB_SOFTMAX_EXP_BUF_BYTES =
        AttentionCommon::Align(mBaseSize / CV_RATIO * static_cast<uint32_t>(sizeof(T)), 256U);
    LocalTensor<T> softmaxExpBuf_;

    static constexpr uint32_t UB_LSE_OUT_BUFCNT = 2U;
    static constexpr uint32_t UB_LSE_OUT_BUF_BYTES =
        AttentionCommon::Align(mBaseSize / CV_RATIO * 8 * static_cast<uint32_t>(sizeof(T)), 256U);
    LocalTensor<uint8_t> ubLseOutBuffers_;
    uint32_t lseOutUbBufId_ = 0;

    static constexpr uint32_t UB_MASK_BUFCNT = 2U;
    // 32 字节错峰, 避免整列 mask 从同一 UB bank 连续 gather 造成 bank 冲突
    static constexpr uint32_t MASK_PITCH = AttentionCommon::Align(s2BaseSize, 32U) + 32U;
    static constexpr uint32_t UB_MASK_BUF_BYTES = mBaseSize / CV_RATIO * MASK_PITCH;
    LocalTensor<uint8_t> ubMaskBuffers_;

    ConstInfo constInfo_{};

    using SEQLEN_T = uint32_t;
    SeqLensTool<LAYOUT_T, SEQLEN_T> qSeqLensTool_;
    SeqLensTool<LAYOUT_KV, SEQLEN_T> kvSeqLensTool_;

    // GM
    static constexpr GmFormat OUT_FORMAT = GetAttentionOutGmFormat<LAYOUT_OUT>();
    using FaGmTensorOut = FaGmTensor<OUTPUT_T, OUT_FORMAT, SEQLEN_T, IS_TND<LAYOUT_OUT>()>;
    FaGmTensorOut outGmTensor_;
    // 不合轴 DN: 输出按 (b, n1, s1) 寻址(quant_flash_attn USE_DN 先例用 UbFormat::S1_ONLY)
    CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, UbFormat::S1_ONLY> copyAttenOutUbToGm_;
    GlobalTensor<OUTPUT_T> attentionOutGm_;
    GlobalTensor<float> softmaxLseGm_;
    GlobalTensor<uint8_t> attenMaskGmInt_;
    GlobalTensor<float> accumOutGm_;
    GlobalTensor<float> softmaxFDSumGm_;
    GlobalTensor<float> softmaxFDMaxGm_;
    GlobalTensor<float> sinkGm_;

    T negativeFloatScalar_;
    // DN 每个任务只处理一个 G 头, M 轴为该头的连续 S1; mask 按单 G 的 GS 形态搬入,
    // BSND/TND 不能沿用 ND 的 S1G 展开, 否则会把相邻 G 的 mask 行搬进当前 M 块。
    static constexpr MaskFormat MASK_LAYOUT = MaskFormat::GS;
    static constexpr T BOOL_ATTEN_MASK_SCALAR_VALUE = -1000000000000.0;
    uint32_t negativeIntScalar_ = *((uint32_t *)&BOOL_ATTEN_MASK_SCALAR_VALUE);

    // ==================== Functions ======================
    using Hooks = DnUnmergedSchedulerHooks<FA_T, ConstInfo, RunInfo>;
    Hooks scheduler_{constInfo_, qSeqLensTool_, kvSeqLensTool_};
    using VecFdBlock = FiaBlockVecFlashDecode<FA_T, true, ConstInfo, true>;
    VecFdBlock vecFdBlock_{constInfo_, qSeqLensTool_, kvSeqLensTool_};
    GlobalTensor<uint32_t> fdMetaDataGm_;
    FDparams fdParams_;
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv,
                                __gm__ uint8_t *seqUsedQ, __gm__ uint8_t *seqUsedKv, __gm__ uint8_t *learnableSink,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t *fiaMetaData, __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                                __tiling_data_ptr__ FlashAttnNoQuantTilingArch35 *tiling)
    {
        scheduler_.Init(cuSeqLensQ, cuSeqLensKv, seqUsedQ, seqUsedKv, learnableSink, fiaMetaData, tiling);
        InitBlock(attenMask, learnableSink, softmaxLse, attentionOut, workspace);
        ClearOutput();
        if (constInfo_.enableFlashDecode) {
            fdMetaDataGm_.SetGlobalBuffer((__gm__ uint32_t *)(fiaMetaData + METADATA_HEADER_OFFSET +
                                                              scheduler_.sectionNum_ * scheduler_.metadata_aic_num_ *
                                                                  METADATA_STRIDE * sizeof(uint32_t)),
                                          scheduler_.sectionNum_ * scheduler_.metadata_aiv_num_ * METADATA_STRIDE);
            vecFdBlock_.InitBlock(learnableSink, softmaxLse, attentionOut);
            vecFdBlock_.InitGlobalTensor(softmaxFDMaxGm_, softmaxFDSumGm_, accumOutGm_);
        }
    }
    __aicore__ inline void Process()
    {
        InitBuffers();
        InitCrossCoreSync();
        AllocEventID();
        for (uint32_t section = 0; section < scheduler_.sectionNum_; ++section) {
            if (constInfo_.aicIdx < constInfo_.coreNum) {
                scheduler_.BeginSection(section);
                AttentionTask::CvTaskScheduler<PRELOAD_N, 4, RunInfo>::Run(scheduler_, *this);
            }
            FlashDecode(section);
        }
        FreeEventID();
        UnInitCrossCoreSync();
    }
    __aicore__ inline void ComputeFirst(RunInfo &info)
    {
        ProcessVec1(info);
    }
    __aicore__ inline void ComputeSecond(RunInfo &info)
    {
        ProcessVec2(info);
    }
    __aicore__ inline uint32_t GetFDMetaDataIndex(uint32_t coreIdx, uint32_t metaIdx, uint32_t sectionIdx)
    {
        return METADATA_STRIDE * scheduler_.metadata_aiv_num_ * sectionIdx + METADATA_STRIDE * coreIdx + metaIdx;
    }
    __aicore__ inline void GetFDSectionInfo(uint32_t sectionIdx)
    {
        fdParams_.mLen = fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, FA_FD_M_NUM_INDEX, sectionIdx));
        fdParams_.fdCoreEnable = fdParams_.mLen > 0 ? 1U : 0U;
        if (!fdParams_.fdCoreEnable) {
            return;
        }
        fdParams_.fdBN1Idx =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, FA_FD_BN1_IDX_INDEX, sectionIdx));
        fdParams_.fdMIdx = fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, FA_FD_M_IDX_INDEX, sectionIdx));
        fdParams_.fdWorkspaceIdx =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, FA_FD_WORKSPACE_IDX_INDEX, sectionIdx));
        fdParams_.fdS2SplitNum =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, FA_FD_WORKSPACE_NUM_INDEX, sectionIdx));
        fdParams_.mStart =
            fdMetaDataGm_.GetValue(GetFDMetaDataIndex(constInfo_.aivIdx, FA_FD_M_START_INDEX, sectionIdx));
    }
    __aicore__ inline void FlashDecode(uint32_t sectionIdx)
    {
        if (!constInfo_.enableFlashDecode) {
            return;
        }
        GetFDSectionInfo(sectionIdx);
        vecFdBlock_.template InitBuffers<GetFdBaseOffset()>();
        AscendC::ICachePreLoad(2);
        AscendC::SyncAll();
        vecFdBlock_.FlashDecode(fdParams_);
        AscendC::SyncAll();
    }
    __aicore__ inline void InitBlock(__gm__ uint8_t *attenMask, __gm__ uint8_t *learnableSink,
                                     __gm__ uint8_t *softmaxLse, __gm__ uint8_t *attentionOut,
                                     __gm__ uint8_t *workspace)
    {
        uint32_t tmp1 = NEGATIVE_MIN_VALUE_FP32;
        this->negativeFloatScalar_ = *((T *)&tmp1);

        this->attentionOutGm_.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        InitAttenOutBuffer(constInfo_.bSize, constInfo_.n2Size, constInfo_.gSize, constInfo_.s1Size, constInfo_.dSizeV,
                           outGmTensor_, attentionOut);

        if (constInfo_.isSoftmaxLseEnable) {
            softmaxLseGm_.SetGlobalBuffer((__gm__ float *)softmaxLse);
        }

        if constexpr (HAS_MASK) {
            attenMaskGmInt_.SetGlobalBuffer((__gm__ uint8_t *)attenMask);
        }

        if (constInfo_.enableFlashDecode) {
            accumOutGm_.SetGlobalBuffer((__gm__ float *)workspace);
            softmaxFDSumGm_.SetGlobalBuffer((__gm__ float *)workspace + constInfo_.accumOutSize);
            softmaxFDMaxGm_.SetGlobalBuffer((__gm__ float *)workspace + constInfo_.accumOutSize +
                                            constInfo_.logSumExpSize);
        }
        if (constInfo_.learnableSinkFlag) {
            sinkGm_.SetGlobalBuffer((__gm__ float *)learnableSink);
        }
    }

    __aicore__ inline void InitBuffers()
    {
        /*--------------------------------------------L1--------------------------------------------*/
        // l1P 三缓冲
        uint32_t addrL1 = 0;
        l1PBuffers_ = LocalTensor<uint8_t>(TPosition::A1, addrL1, L1_P_BUFCNT * L1_P_BUF_BYTES);

        static_assert(UB_MM_RES_TOTAL_BYTES + UB_VEC2_RES_BUF_BYTES + UB_VEC1_RES_BUFCNT * UB_VEC1_RES_BUF_BYTES +
                              UB_SOFTMAX_MAX_BUFCNT * UB_SOFTMAX_MAX_BUF_BYTES +
                              UB_SOFTMAX_SUM_BUFCNT * UB_SOFTMAX_SUM_BUF_BYTES +
                              UB_SOFTMAX_EXP_BUFCNT * UB_SOFTMAX_EXP_BUF_BYTES +
                              UB_LSE_OUT_BUFCNT * UB_LSE_OUT_BUF_BYTES + (HAS_MASK ? DB * UB_MASK_BUF_BYTES : 0U) <=
                          248U * 1024U,
                      "DN UB capacity exceeded");
        /*--------------------------------------------UB--------------------------------------------*/
        uint32_t addrUb = 0;
        ubMm1ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES);
        addrUb += UB_MM1_RES_BUFCNT * UB_MM1_RES_BUF_BYTES;
        ubMm2ResBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MM2_RES_BUFCNT * UB_MM2_RES_BUF_BYTES);
        addrUb += UB_MM2_RES_BUFCNT * UB_MM2_RES_BUF_BYTES;
        ubVec2Res_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_VEC2_RES_BUF_BYTES)
                         .template ReinterpretCast<T>(); // 输出BUF: attn_out拷出
        addrUb += UB_VEC2_RES_BUF_BYTES;
        ubVec1ResBuffers_ = LocalTensor<uint8_t>(
            TPosition::VECIN, addrUb,
            UB_VEC1_RES_BUFCNT * UB_VEC1_RES_BUF_BYTES); // 2 * 32.25K = 64.5K, 输出BUF: softmax结果拷贝至L1
        addrUb += UB_VEC1_RES_BUFCNT * UB_VEC1_RES_BUF_BYTES;

        // softmaxSum×3 + softmaxMax×3 + softmaxExp×3，各 256 bytes
        softmaxSumBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb,
                                              UB_SOFTMAX_SUM_BUFCNT * UB_SOFTMAX_SUM_BUF_BYTES)
                             .template ReinterpretCast<T>(); // 3 * 0.25K = 0.75K, 常驻BUF
        addrUb += UB_SOFTMAX_SUM_BUFCNT * UB_SOFTMAX_SUM_BUF_BYTES;
        softmaxMaxBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb,
                                              UB_SOFTMAX_MAX_BUFCNT * UB_SOFTMAX_MAX_BUF_BYTES)
                             .template ReinterpretCast<T>(); // 3 * 0.25K = 0.75K, 常驻BUF
        addrUb += UB_SOFTMAX_MAX_BUFCNT * UB_SOFTMAX_MAX_BUF_BYTES;
        softmaxExpBuf_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb,
                                              UB_SOFTMAX_EXP_BUFCNT * UB_SOFTMAX_EXP_BUF_BYTES)
                             .template ReinterpretCast<T>(); // 3 * 0.25K = 0.75K, 常驻BUF
        addrUb += UB_SOFTMAX_EXP_BUFCNT * UB_SOFTMAX_EXP_BUF_BYTES;

        ubLseOutBuffers_ = LocalTensor<uint8_t>(
            TPosition::VECIN, addrUb,
            UB_LSE_OUT_BUFCNT *
                UB_LSE_OUT_BUF_BYTES); // 输出BUF:
                                       // FD中间结果SUM和MAX拷出至GM，或者LSE结果拷出；按actVecMSize*8(fp32块对齐)定容
        addrUb += UB_LSE_OUT_BUFCNT * UB_LSE_OUT_BUF_BYTES;
        ubMaskBuffers_ = LocalTensor<uint8_t>(TPosition::VECIN, addrUb, UB_MASK_BUFCNT * UB_MASK_BUF_BYTES);
    }

    __aicore__ inline void ResetSoftmaxBuffer(uint32_t slotIdx, const RunInfo &runInfo)
    {
        if (runInfo.actVecMSize == 0) {
            return;
        }
        constexpr uint32_t softmaxBufElementCount = UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T);
        LocalTensor<T> sumUb = softmaxSumBuf_[slotIdx * softmaxBufElementCount];
        LocalTensor<T> maxUb = softmaxMaxBuf_[slotIdx * softmaxBufElementCount];
        if (constInfo_.learnableSinkFlag && runInfo.isFirstFdBlock) {
            Duplicate<T>(sumUb, static_cast<T>(1), softmaxBufElementCount);

            // sink 按 GS 线性坐标定位当前 G 头: g*S1 + s1
            uint32_t gs1Start = runInfo.gIdx * runInfo.actS1Size + runInfo.s1Idx + runInfo.vecMbaseIdx;

            Mutex::Lock<PIPE_MTE2>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
            {
                LocalTensor<float> sinkTmpUb =
                    ubVec1ResBuffers_[vec1ResUbBufId_ * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<float>();
                AttentionCommon::SinkCopyInGS1(sinkTmpUb, sinkGm_, gs1Start, runInfo.actVecMSize, runInfo.actS1Size,
                                               runInfo.n2Idx, constInfo_.gSize);
            }
            Mutex::Unlock<PIPE_MTE2>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);

            Mutex::Lock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
            {
                LocalTensor<float> sinkTmpUb =
                    ubVec1ResBuffers_[vec1ResUbBufId_ * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<float>();

                AttentionCommon::SinkExpandMaxVf<T, float, true>(maxUb, sinkTmpUb, gs1Start, runInfo.actVecMSize,
                                                                 runInfo.actS1Size, constInfo_.gSize);
            }
            Mutex::Unlock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
        } else {
            Duplicate<T>(sumUb, static_cast<T>(0), softmaxBufElementCount);
            Duplicate<T>(maxUb, static_cast<T>(-std::numeric_limits<float>::infinity()), softmaxBufElementCount);
        }
    }

    __aicore__ inline void InitCrossCoreSync()
    {
        for (uint32_t slot = 0; slot < UB_MM1_RES_BUFCNT; ++slot) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM1_0 + slot);
        }
        for (uint32_t slot = 0; slot < UB_MM2_RES_BUFCNT; ++slot) {
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(CC_MM2_0 + slot);
        }
    }

    __aicore__ inline void UnInitCrossCoreSync() {}

    __aicore__ inline void AllocEventID() {}

    __aicore__ inline void FreeEventID() {}

    __aicore__ inline void ProcessVec1(RunInfo runInfo)
    {
        uint32_t mm1ResUbBufId = mm1ResUbBufId_;
        mm1ResUbBufId_ = (mm1ResUbBufId_ + 1) % UB_MM1_RES_BUFCNT;
        uint32_t pL1BufId = runInfo.loop % L1_P_BUFCNT;
        uint32_t mmSyncIdx = CC_MM1_0 + mm1ResUbBufId;
        uint32_t v1c2CrossCoreSyncIdx = CC_L1P_0 + pL1BufId;
        LocalTensor<INPUT_T> pL1Tensor = l1PBuffers_[pL1BufId * L1_P_BUF_BYTES].template ReinterpretCast<INPUT_T>();
        auto mm1ResUbTensor = ubMm1ResBuffers_[mm1ResUbBufId * UB_MM1_RES_BUF_BYTES].template ReinterpretCast<T>();

        if (unlikely(runInfo.isFirstS2Loop)) {
            AscendC::PipeBarrier<PIPE_V>();
            ResetSoftmaxBuffer(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT, runInfo);
            AscendC::PipeBarrier<PIPE_V>();
        }

        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx);
        ProcessVec1Dn(pL1Tensor, mm1ResUbTensor, runInfo);
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mmSyncIdx); // 通知BMM1: Vec1已读完MM1结果, 可覆写
        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(v1c2CrossCoreSyncIdx);
    }

    __aicore__ inline void ClearOutput()
    {
        if (constInfo_.needInitOutput) {
            uint32_t vecCoreNum = CV_RATIO * constInfo_.coreNum;
            uint64_t tSize = constInfo_.bSize * constInfo_.s1Size;
            if constexpr (LAYOUT_T == FA_LAYOUT::TND) {
                tSize = qSeqLensTool_.cuSeqLensParser.GetTSize();
            }
            uint64_t attenOutTotalSize = tSize * constInfo_.n2Size * constInfo_.gSize * constInfo_.dSizeV;

            static constexpr OUTPUT_T ATTEN_OUT_INIT_VAL = 0;
            static constexpr uint32_t ATTEN_OUT_POP_BUF_START_ADDR = 0;
            static constexpr uint32_t ATTEN_OUT_POP_BUF_ELE_SIZE = BUFFER_SIZE_BYTE_32K / sizeof(OUTPUT_T);
            AttentionCommon::InitOutput<OUTPUT_T, EVENT_ID0, ATTEN_OUT_POP_BUF_START_ADDR, ATTEN_OUT_POP_BUF_ELE_SIZE,
                                        true>(attentionOutGm_, attenOutTotalSize, vecCoreNum, ATTEN_OUT_INIT_VAL);

            if (constInfo_.isSoftmaxLseEnable) {
                uint64_t lseTotalSize = tSize * constInfo_.n2Size * constInfo_.gSize;

                static constexpr float LSE_INIT_VAL = 3e+99;
                static constexpr uint32_t LSE_POP_BUF_START_ADDR = BUFFER_SIZE_BYTE_32K;
                static constexpr uint32_t LSE_POP_BUF_ELE_SIZE = BUFFER_SIZE_BYTE_32K / sizeof(float);
                AttentionCommon::InitOutput<float, EVENT_ID1, LSE_POP_BUF_START_ADDR, LSE_POP_BUF_ELE_SIZE, true>(
                    softmaxLseGm_, lseTotalSize, vecCoreNum, LSE_INIT_VAL);
            }

            SyncAll();
        }
    }

    __aicore__ inline void InitAttenOutBuffer(uint32_t batchSize, uint32_t n2Size, uint32_t gSize, uint32_t qSeqSize,
                                              uint32_t headDim, FaGmTensorOut &outGmTensor, __gm__ uint8_t *gm)
    {
        outGmTensor.gmTensor.SetGlobalBuffer((__gm__ OUTPUT_T *)gm);
        if constexpr (GmLayoutParams<OUT_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
            outGmTensor.offsetCalculator.Init(batchSize, n2Size, gSize, qSeqSize, headDim, qSeqLensTool_.seqUsedParser);
        } else {
            outGmTensor.offsetCalculator.Init(n2Size, gSize, headDim, qSeqLensTool_.cuSeqLensParser);
        }
    }

    __aicore__ inline void SoftmaxDataCopyOut(RunInfo runInfo, LocalTensor<float> &sumUb, LocalTensor<float> &maxUb)
    {
        if (constInfo_.enableFlashDecode) {
            if (runInfo.isS2SplitCore) {
                ComputeLogSumExpAndCopyToGm(runInfo, sumUb, maxUb);
            }
        }

        if (constInfo_.enableFlashDecode) {
            if (!runInfo.isS2SplitCore && constInfo_.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
            }
        } else {
            if (constInfo_.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(sumUb, maxUb, runInfo);
            }
        }
    }

    __aicore__ inline void SoftmaxLseCopyOut(LocalTensor<float> &softmaxSumTmp, LocalTensor<float> &softmaxMaxTmp,
                                             RunInfo &runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        Mutex::Lock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        LocalTensor<float> lseUb =
            ubLseOutBuffers_[lseOutUbBufId_ * UB_LSE_OUT_BUF_BYTES].template ReinterpretCast<float>();
        ComputeLseOutputVF(lseUb, softmaxSumTmp, softmaxMaxTmp, runInfo.actVecMSize,
                           HAS_MASK ? negativeIntScalar_ : NEGATIVE_MIN_VALUE_FP32);
        Mutex::Unlock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        DataCopySoftmaxLseS1Only<LAYOUT_T>(softmaxLseGm_, lseUb, runInfo.bIdx, runInfo.n2Idx, runInfo.gIdx,
                                           runInfo.s1Idx + runInfo.vecMbaseIdx, runInfo.actVecMSize, constInfo_,
                                           qSeqLensTool_);
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        lseOutUbBufId_ = (lseOutUbBufId_ + 1) % UB_LSE_OUT_BUFCNT;
    }

    __aicore__ inline void ProcessVec1Dn(LocalTensor<INPUT_T> &pL1Tensor, LocalTensor<T> &mm1ResUbTensor,
                                         RunInfo runInfo)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }

        static constexpr uint32_t vec1S2CopyLenDn = s2BaseSize / CV_RATIO;
        static constexpr uint32_t vec1HalfS1BaseSize = mBaseSize >> 1;
        static constexpr uint32_t vec1S2CopyCountDn = (mBaseSize >> 4) / CV_RATIO;
        static constexpr uint32_t vec1S2strideDn = s2BaseSize * 8;
        static constexpr uint32_t vec1ResOffsetDn = s2BaseSize * 32 + 64;

        LocalTensor<uint8_t> attenMaskUb;
        LocalTensor<T> sumUb =
            softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) * (UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T))];
        LocalTensor<T> maxUb =
            softmaxMaxBuf_[(runInfo.mloop % UB_SOFTMAX_MAX_BUFCNT) * (UB_SOFTMAX_MAX_BUF_BYTES / sizeof(T))];
        LocalTensor<T> expUb =
            softmaxExpBuf_[(runInfo.loop % UB_SOFTMAX_EXP_BUFCNT) * (UB_SOFTMAX_EXP_BUF_BYTES / sizeof(T))];

        // 块级粗筛: 本 S2 块可能存在被掩蔽的列才搬 mask; 全可见块直接跳过
        const uint32_t maskBufId = runInfo.loop & (DB - 1);
        bool needMask = false;
        if constexpr (HAS_MASK) {
            const int64_t firstQ = runInfo.s1Idx + runInfo.vecMbaseIdx;
            const int64_t lastQ = firstQ + runInfo.actVecMSize - 1;
            const int64_t lastKv = runInfo.s2Idx + runInfo.actSingleLoopS2Size - 1;
            if (constInfo_.scaleValue <= 0 || firstQ + runInfo.nextTokensLeftUp < lastKv ||
                lastQ - runInfo.preTokensLeftUp > runInfo.s2Idx) {
                attenMaskUb = ubMaskBuffers_[maskBufId * UB_MASK_BUF_BYTES];
                needMask = AttenMaskCopyIn(attenMaskUb, runInfo);
                if (needMask) {
                    Mutex::Lock<PIPE_V>(UB_IN_MASK_EVENT0 + maskBufId);
                }
            }
        }

        Mutex::Lock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);

        float descaleQK = 1.0;

        LocalTensor<INPUT_T> stage1CastTensor =
            ubVec1ResBuffers_[vec1ResUbBufId_ * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<INPUT_T>();
        PrepareDnScoresVF<HAS_MASK, MASK_PITCH>(
            (__ubuf__ float *)mm1ResUbTensor.GetPhyAddr(), (__ubuf__ uint32_t *)attenMaskUb.GetPhyAddr(),
            runInfo.actVecMSize, runInfo.actMSizeAlign32 / CV_RATIO, runInfo.actSingleLoopS2Size,
            runInfo.actSingleLoopS2SizeAlign, static_cast<float>(constInfo_.scaleValue), negativeFloatScalar_,
            needMask);
        AscendC::PipeBarrier<PIPE_V>();
        FaVectorApi::ProcessVec1VfDn<T, INPUT_T, true, false, s2BaseSize>(
            stage1CastTensor, sumUb, maxUb, mm1ResUbTensor, expUb, nullptr, attenMaskUb, runInfo.actMSizeAlign32 >> 1,
            runInfo.actSingleLoopS2SizeAlign, runInfo.actSingleLoopS2Size, 1.0F, descaleQK, negativeFloatScalar_, 0.0F,
            false);

        if (needMask) {
            Mutex::Unlock<PIPE_V>(UB_IN_MASK_EVENT0 + maskBufId);
        }

        Mutex::Unlock<PIPE_V>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
        LocalTensor<INPUT_T> mm2AL1Tensor = pL1Tensor;

        if (runInfo.actSingleLoopS2Size > vec1S2CopyLenDn) {
            DataCopy(mm2AL1Tensor[constInfo_.subBlockIdx * vec1HalfS1BaseSize * runInfo.actSingleLoopS2SizeAlign],
                     stage1CastTensor,
                     {vec1S2CopyCountDn, vec1S2CopyLenDn, 1,
                      static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign - vec1S2CopyLenDn)});
            DataCopy(mm2AL1Tensor[constInfo_.subBlockIdx * vec1HalfS1BaseSize * runInfo.actSingleLoopS2SizeAlign +
                                  vec1S2strideDn],
                     stage1CastTensor[vec1ResOffsetDn],
                     {vec1S2CopyCountDn, static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign - vec1S2CopyLenDn),
                      static_cast<uint16_t>(s2BaseSize - runInfo.actSingleLoopS2SizeAlign + 1), vec1S2CopyLenDn});
        } else {
            DataCopy(mm2AL1Tensor[constInfo_.subBlockIdx * vec1HalfS1BaseSize * runInfo.actSingleLoopS2SizeAlign],
                     stage1CastTensor,
                     {vec1S2CopyCountDn, static_cast<uint16_t>(runInfo.actSingleLoopS2SizeAlign),
                      static_cast<uint16_t>(vec1S2CopyLenDn - runInfo.actSingleLoopS2SizeAlign + 1), 0});
        }

        Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_);
        vec1ResUbBufId_ = (vec1ResUbBufId_ + 1U) % UB_VEC1_RES_BUFCNT;
    }

    __aicore__ inline bool AttenMaskCopyIn(LocalTensor<uint8_t> attenMaskUb, RunInfo &runInfo)
    {
        MaskInfo maskInfo;
        // mask 工具按单 G 的 GS 线性坐标搬入, G 已在 BN1 轴展开; gs1StartIdx 即本头纯 s1 起始
        const uint32_t maskS1Start = runInfo.s1Idx + runInfo.vecMbaseIdx;
        maskInfo.gs1StartIdx = maskS1Start;
        maskInfo.gs1dealNum = runInfo.actVecMSize;
        maskInfo.s1Size = runInfo.actS1Size;
        maskInfo.gSize = constInfo_.gSize;
        maskInfo.s2StartIdx = runInfo.s2Idx;
        maskInfo.s2dealNum = runInfo.actSingleLoopS2Size;
        maskInfo.s2Size = runInfo.actS2Size;
        maskInfo.nBaseSize = s2BaseSize;
        maskInfo.preToken = constInfo_.preTokens;
        maskInfo.nextToken = constInfo_.nextTokens;
        maskInfo.sparseMode = static_cast<SparseMode>(constInfo_.sparseMode);
        maskInfo.batchIdx = (constInfo_.attenMaskBatch == 1) ? 0 : runInfo.bIdx;
        maskInfo.attenMaskBatchStride = constInfo_.attenMaskS1Size * constInfo_.attenMaskS2Size;
        maskInfo.attenMaskS1Stride = constInfo_.attenMaskS2Size;
        maskInfo.attenMaskDstStride = (MASK_PITCH - AttentionCommon::Align(maskInfo.s2dealNum, 32U)) / 32;
        maskInfo.maskValue = negativeIntScalar_;
        maskInfo.s1LeftPaddingSize = 0;
        maskInfo.s2LeftPaddingSize = 0;
        maskInfo.maskFormat = MASK_LAYOUT;
        maskInfo.attenMaskType = MASK_BOOL;

        const uint32_t bufIdx = runInfo.loop & (DB - 1);
        const bool skipMask = IsSkipAttentionmask(maskInfo);
        const bool skipMaskForPre = IsSkipAttentionmaskForPre(maskInfo);
        if (skipMask && skipMaskForPre) {
            if (constInfo_.scaleValue > 0) {
                return false;
            }
            // 零/负 scale 必须先于 softmax VF 的负 padding 哨兵写入施加, 否则 padding 被乘为 0/正数污染分母
            Mutex::Lock<PIPE_V>(UB_IN_MASK_EVENT0 + bufIdx);
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * MASK_PITCH);
            Mutex::Unlock<PIPE_V>(UB_IN_MASK_EVENT0 + bufIdx);
            return true;
        }
        if (!skipMask) {
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSize>(attenMaskUb, attenMaskGmInt_, maskInfo, false,
                                                                        UB_IN_MASK_EVENT0 + bufIdx);
        } else {
            Mutex::Lock<PIPE_V>(UB_IN_MASK_EVENT0 + bufIdx);
            Duplicate(attenMaskUb, static_cast<uint8_t>(0U), maskInfo.gs1dealNum * MASK_PITCH);
            Mutex::Unlock<PIPE_V>(UB_IN_MASK_EVENT0 + bufIdx);
        }
        if (!skipMaskForPre) {
            // BAND 窗口: pre 半带与 next 半带分别拷入后按位合并, 可见 = next | !pre
            const uint32_t preBufId = bufIdx ^ 1U;
            LocalTensor<uint8_t> attenMaskUbPre = ubMaskBuffers_[preBufId * UB_MASK_BUF_BYTES];
            AttentionmaskCopyIn<uint8_t, MASK_LAYOUT, true, s2BaseSize>(attenMaskUbPre, attenMaskGmInt_, maskInfo, true,
                                                                        UB_IN_MASK_EVENT0 + preBufId);
            Mutex::Lock<PIPE_V>(UB_IN_MASK_EVENT0 + preBufId);
            Mutex::Lock<PIPE_V>(UB_IN_MASK_EVENT0 + bufIdx);
            MergeBand(attenMaskUb.GetPhyAddr(), attenMaskUbPre.GetPhyAddr(),
                      (maskInfo.gs1dealNum * MASK_PITCH + 255U) / 256U);
            Mutex::Unlock<PIPE_V>(UB_IN_MASK_EVENT0 + bufIdx);
            Mutex::Unlock<PIPE_V>(UB_IN_MASK_EVENT0 + preBufId);
        }
        return true;
    }

    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUTPUT_T> &attenOutUb,
                                                uint32_t vecMIdx, uint32_t dealRowCount)
    {
        FaUbTensor<OUTPUT_T> ubTensor{.tensor = attenOutUb, .rowCount = dealRowCount, .colCount = dTemplateAlign64};
        GmCoordS1Only gmCoord{.bIdx = info.bIdx,
                              .n2Idx = info.n2Idx,
                              .gIdx = info.gIdx,
                              .s1Idx = info.s1Idx + info.vecMbaseIdx + vecMIdx,
                              .dIdx = 0,
                              .s1DealSize = dealRowCount,
                              .dDealSize = (uint32_t)constInfo_.dSizeV};
        copyAttenOutUbToGm_(outGmTensor_, ubTensor, gmCoord);
    }

    __aicore__ inline void BroadCastAndCopyOut(const RunInfo &runInfo, LocalTensor<float> &sumUb,
                                               LocalTensor<float> &maxUb, int64_t gmOffset, int64_t calculateSize)
    {
        LocalTensor<float> sumBrdcstBuf =
            ubLseOutBuffers_[lseOutUbBufId_ * UB_LSE_OUT_BUF_BYTES].template ReinterpretCast<float>();
        Mutex::Lock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        FaVectorApi::BroadcastMaxSum(sumBrdcstBuf, sumUb, runInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        DataCopy(softmaxFDSumGm_[gmOffset], sumBrdcstBuf, calculateSize);
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        lseOutUbBufId_ = (lseOutUbBufId_ + 1U) % UB_LSE_OUT_BUFCNT;

        LocalTensor<float> maxBrdcstBuf =
            ubLseOutBuffers_[lseOutUbBufId_ * UB_LSE_OUT_BUF_BYTES].template ReinterpretCast<float>();
        Mutex::Lock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        FaVectorApi::BroadcastMaxSum(maxBrdcstBuf, maxUb, runInfo.actVecMSize);
        Mutex::Unlock<PIPE_V>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        Mutex::Lock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        DataCopy(softmaxFDMaxGm_[gmOffset], maxBrdcstBuf, calculateSize);
        Mutex::Unlock<PIPE_MTE3>(UB_OUT_LSE_OUT_EVENT0 + lseOutUbBufId_);
        lseOutUbBufId_ = (lseOutUbBufId_ + 1U) % UB_LSE_OUT_BUFCNT;
    }

    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfo &runInfo, LocalTensor<float> &sumUb,
                                                       LocalTensor<float> &maxUb)
    {
        if (unlikely(runInfo.actVecMSize == 0)) {
            return;
        }
        int64_t calculateSize = runInfo.actVecMSize * fp32BaseSize;
        int64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * fp32BaseSize + runInfo.vecMbaseIdx * fp32BaseSize;
        // Copy sum to gm
        BroadCastAndCopyOut(runInfo, sumUb, maxUb, gmOffset, calculateSize);
    }

    __aicore__ inline void Bmm2ResForFDCopyOut(const RunInfo &runInfo, LocalTensor<T> &ubVec2Res, uint32_t mStartVec,
                                               uint32_t mDealSize)
    {
        int64_t dSizeAligned64 = (int64_t)dVBaseSize;
        uint64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * constInfo_.dSizeV +
                            (runInfo.vecMbaseIdx + mStartVec) * constInfo_.dSizeV;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = mDealSize;
        dataCopyParams.blockLen = constInfo_.dSizeV * sizeof(T);
        dataCopyParams.srcStride = (dSizeAligned64 - constInfo_.dSizeV) / (AttentionCommon::BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;

        DataCopyPad(accumOutGm_[gmOffset], ubVec2Res, dataCopyParams);
    }

    __aicore__ inline void ProcessVec2(RunInfo runInfo)
    {
        uint32_t mm2ResUbBufId = mm2ResUbBufId_;
        mm2ResUbBufId_ = (mm2ResUbBufId_ + 1) % UB_MM2_RES_BUFCNT;
        uint32_t mm2SyncIdx = CC_MM2_0 + mm2ResUbBufId;
        if (unlikely(runInfo.actVecMSize == 0)) {
            CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mm2SyncIdx);
            CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mm2SyncIdx);
            return;
        }

        const uint32_t dSizeV = constInfo_.dSizeV;
        const uint32_t dSizeAligned16 = (dSizeV + 15U) / 16U * 16U;
        CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mm2SyncIdx);
        {
            LocalTensor<T> mm2ResUbTensor =
                ubMm2ResBuffers_[mm2ResUbBufId * UB_MM2_RES_BUF_BYTES].template ReinterpretCast<T>();
            Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
            if (runInfo.isFirstS2Loop) {
                DataCopy(ubVec2Res_, mm2ResUbTensor, dSizeV * 64U);
            } else {
                LocalTensor<T> expUb =
                    softmaxExpBuf_[(runInfo.loop % UB_SOFTMAX_EXP_BUFCNT) * (UB_SOFTMAX_EXP_BUF_BYTES / sizeof(T))];
                if (runInfo.isLastS2Loop) {
                    FaVectorApi::FlashUpdateDnRows64VF<true>(ubVec2Res_, mm2ResUbTensor, expUb,
                                                             static_cast<uint16_t>(dSizeV));
                } else {
                    FaVectorApi::FlashUpdateDnRows64VF<false>(ubVec2Res_, mm2ResUbTensor, expUb,
                                                              static_cast<uint16_t>(dSizeV));
                }
            }
            Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
        }

        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(mm2SyncIdx);

        if (runInfo.isLastS2Loop) {
            // Borrow the next Vec1 UB slot without advancing its ring index.
            static_assert(UB_VEC1_RES_BUF_BYTES >= 64U * dTemplateAlign64 * sizeof(OUTPUT_T),
                          "Vec1 slot must hold the packed Vec2 output");
            uint32_t vec1ResSyncIdx = UB_OUT_VEC1_RES_EVENT0 + vec1ResUbBufId_;
            LocalTensor<T> vec2TmpUb =
                ubVec1ResBuffers_[vec1ResUbBufId_ * UB_VEC1_RES_BUF_BYTES].template ReinterpretCast<T>();
            AscendC::PipeBarrier<PIPE_V>();
            LocalTensor<float> sumUb =
                softmaxSumBuf_[(runInfo.mloop % UB_SOFTMAX_SUM_BUFCNT) * (UB_SOFTMAX_SUM_BUF_BYTES / sizeof(T))];
            LocalTensor<float> maxUb =
                softmaxMaxBuf_[(runInfo.mloop % UB_SOFTMAX_MAX_BUFCNT) * (UB_SOFTMAX_MAX_BUF_BYTES / sizeof(T))];
            if constexpr (HAS_MASK) {
                Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
                FaVectorApi::FinalizeDnMaskRowsVF(ubVec2Res_, sumUb, maxUb, runInfo.actVecMSize,
                                                  static_cast<uint16_t>(dSizeV));
                AscendC::PipeBarrier<PIPE_V>();
                Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
            }
            SoftmaxDataCopyOut(runInfo, sumUb, maxUb);
            if (constInfo_.enableFlashDecode && runInfo.isS2SplitCore) {
                constexpr uint32_t rowsPerCopy = UB_VEC1_RES_BUF_BYTES / (dTemplateAlign64 * sizeof(T));
                static_assert(rowsPerCopy > 0, "Vec1 slot must hold one workspace row");
                Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
                for (uint32_t mStartVec = 0; mStartVec < runInfo.actVecMSize; mStartVec += rowsPerCopy) {
                    uint32_t mDealSize = AttentionCommon::Min(rowsPerCopy, runInfo.actVecMSize - mStartVec);
                    Mutex::Lock<PIPE_V>(vec1ResSyncIdx);
                    FaVectorApi::FlashOutputDnWorkspaceVF(
                        vec2TmpUb, ubVec2Res_[mStartVec], sumUb[mStartVec], static_cast<uint16_t>(mDealSize),
                        static_cast<uint16_t>(dSizeV), static_cast<uint16_t>(dTemplateAlign64));
                    Mutex::Unlock<PIPE_V>(vec1ResSyncIdx);
                    Mutex::Lock<PIPE_MTE3>(vec1ResSyncIdx);
                    Bmm2ResForFDCopyOut(runInfo, vec2TmpUb, mStartVec, mDealSize);
                    Mutex::Unlock<PIPE_MTE3>(vec1ResSyncIdx);
                }
                Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
            } else {
                Mutex::Lock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
                Mutex::Lock<PIPE_V>(vec1ResSyncIdx);
                LocalTensor<OUTPUT_T> vec2ResUbTensor = vec2TmpUb.template ReinterpretCast<OUTPUT_T>();
                FaVectorApi::FlashPackOutputDnRows64VF(vec2ResUbTensor, ubVec2Res_, sumUb,
                                                       static_cast<uint16_t>(dSizeV));
                AscendC::PipeBarrier<PIPE_V>();
                LocalTensor<OUTPUT_T> attenOut = ubVec2Res_.template ReinterpretCast<OUTPUT_T>();
                uint64_t srcList[16], dstList[16];
                for (uint32_t dOffset = 0; dOffset < dSizeAligned16; dOffset += 16) {
                    for (uint32_t i = 0; i < 16; ++i) {
                        srcList[i] = reinterpret_cast<uint64_t>(vec2ResUbTensor.GetPhyAddr()) +
                                     ((dOffset + i) * 64U) * sizeof(OUTPUT_T);
                        dstList[i] = reinterpret_cast<uint64_t>(attenOut.GetPhyAddr()) +
                                     (i * dTemplateAlign64 + dOffset) * sizeof(OUTPUT_T);
                    }
                    TransDataTo5HD<OUTPUT_T>(dstList, srcList, {false, false, 4, dTemplateAlign64, 1});
                }
                // The next Vec1 can reuse this slot on PIPE_V without an intervening MTE3.
                AscendC::PipeBarrier<PIPE_V>();
                Mutex::Unlock<PIPE_V>(vec1ResSyncIdx);
                Mutex::Unlock<PIPE_V>(UB_OUT_VEC2_RES_EVENT0);
                Mutex::Lock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
                Bmm2DataCopyOutTrans(runInfo, attenOut, 0, runInfo.actVecMSize);
                Mutex::Unlock<PIPE_MTE3>(UB_OUT_VEC2_RES_EVENT0);
            }
        }
    }
};

} // namespace FlashAttnKernel
#endif
