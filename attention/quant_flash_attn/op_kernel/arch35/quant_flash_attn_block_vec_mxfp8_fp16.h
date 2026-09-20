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
 * \file quant_flash_attn_block_vec_mxfp8_fp16.h
 * \brief quant_mode=3 (QFA_MXFP8_FP16): MXFP8量化 + softmax FP16 的 VECTOR 侧实现
 *
 * 相比 quant_flash_attn_block_vec_mxfp8.h 的 MXFP8 实现, 本文件按以下设计修改:
 * 1. CV 1:1: 基本块128x512, 单个V核处理完整128行S1(m=128, 一行FP16恰好占满一个寄存器),
 *    不再切分S1到两个V核, subBlock1不参与VEC计算(actVecMSize=0);
 * 2. V1接入 ProcessVec1VfDnMxfp8V2: softmax数据类型FP16(half), 状态tensor(max/exp)改为half;
 * 3. 不计算row_sum(C2阶段V矩阵d维多一行, p*V随路得出) -> 删除softmaxSum/preLoopSum/firstLoopSum;
 * 4. 不乘softmaxscale(C1阶段FIXPIPE随路完成) -> MM1结果(bmm1ResBuf)为half, VF内无dScale乘法;
 * 5. P的FP8打包从4-way(4列x64B)改为2-way(2列x128B), 单V核数据量翻倍:
 *    P区域UB大小从16640B变为33280B(行距仍2080B), pScale(E8M0, 每行一个共m=128个)紧随其后;
 * 6. V2(row_sum除法/bf16 cast/最后一次性转置搬出)与LSE输出已在V2阶段实现(见ProcessVec2)。
 */

#ifndef QUANT_FLASH_ATTN_BLOCK_VEC_MXFP8_FP16_H_
#define QUANT_FLASH_ATTN_BLOCK_VEC_MXFP8_FP16_H_

#include "kernel_operator.h"
#include "memory_copy_arch35_quant_flash_attn.h"
#include "adv_api/activation/softmax.h"
#include "quant_flash_attn_common_def.h"
#include "vf_flashupdate_trans_fp16.h"
#if __has_include("../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h")
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn_mxfp8_v2.h"
#include "../../../common/op_kernel/vector_common.h"
#include "../../../common/op_kernel/arch35/util_regbase.h"
#else
#include "../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "../../common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_dn_mxfp8_v2.h"
#include "../../common/op_kernel/vector_common.h"
#include "../../common/op_kernel/arch35/util_regbase.h"
#endif

using namespace AscendC;
using namespace FaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace AttentionCommon;

namespace BaseApi {

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, uint8_t KvLayoutType = 0,
          bool isFd = false, bool useDn = false, bool isDAligned = true>
class QuantFlashAttnBlockVecMxfp8Fp16 {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t mBaseSize = (uint32_t)s1TemplateType;
    static constexpr uint32_t s2BaseSize = (uint32_t)s2TemplateType;
    static constexpr uint32_t dTemplateAlign64 = Align64Func((uint16_t)dVTemplateType);
    static constexpr bool isFp8 = IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                                  IsSameType<INPUT_T, hifloat8_t>::value;
    static constexpr uint32_t DB = 2;
    static constexpr uint32_t PRELOAD_N = 2;      // C1 C1 C2
    static constexpr uint32_t s2SplitSize = 256U; // 仅支持d=128, 无需按d=256缩小split
    static constexpr uint32_t MXFP_GROUP_SIZE = 32U;

    static constexpr bool FLASH_DECODE = isFd;

    static constexpr uint32_t initOutputEventId = 0U; // attenOut和lse刷无效行会用到剩余ub，需加同步

    static constexpr ActualSeqLensMode Q_MODE = GetQActSeqMode<layout>();

    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    using attenMaskGmType = int8_t; // 不支持mask, 仅占位保持类型接口
    using quantGmType = typename std::conditional<(isFp8), GlobalTensor<float>, int8_t>::type;

    using ConstInfoX = ConstInfo_t;

    using MM1_OUT_T = half; // C1 FIXPIPE随路cast FP16
    using MM2_OUT_T = T;    // MM2输出仍为float(O累加与row_sum除法需要FP32精度)
    using OUT_T = OUTPUT_T;

    // gm
    TPipe *tPipe_ = nullptr;
    GlobalTensor<OUTPUT_T> attentionOutGm_;
    GlobalTensor<float> softmaxLseGm_;
    GlobalTensor<int32_t> actualSeqLengthsGmQ_;
    using QSeqParserType = ActualSeqLensParser<Q_MODE, int32_t>;
    QSeqParserType *qActSeqLensParser_ = nullptr;

    __aicore__ inline void SetCuSeqLensParser(QSeqParserType &qParser)
    {
        this->qActSeqLensParser_ = &qParser;
    }

    quantGmType pScaleGm_;

    // ub
    TBuf<> commonTBuf_; // common的复用空间
    TQue<QuePosition::VECOUT, 1> stage1OutQue_[2];
    TBuf<> stage2OutBuf_;
    TEventID mte3ToVId_[2]; // 存放MTE3_V的eventId, 2份表示可能存在pingpong
    TEventID vToMte3Id_[2]; // 存放V_MTE3的eventId, 2份表示可能存在pingpong
    /* FP16 softmax状态tensor: 每行m=128个half(256B), 无sum(C2阶段V多一行随路得出row_sum) */
    TBuf<> softmaxMaxBuf_[PRELOAD_N + 1];
    TBuf<> softmaxExpBuf_[PRELOAD_N + 1];
    TBuf<> preLoopMaxBuf_;
    TBuf<> vselrIndexesBuf_[4];
    /* ============V2 (FP16 softmax) 阶段专用buffer============= */
    // 跨S2 loop的全局row_sum累加器(128 float): rs_acc = rs_acc*exp + 本loop的row_sum(MM2尾行)
    TBuf<> rowSumAccBuf_;
    // half exp -> float(128): 供转置布局O乘加的行内per-列j寄存器load
    TBuf<> expF32Buf_;
    // 最后输出转置中转: (mBaseSize, dAlign) OUTPUT_T
    TBuf<> mm2TransBuf_;
    // 32x32 tile两级搬运临时: float tile 4KB + OUTPUT_T tile 2KB
    TBuf<> transTmpBuf_;
    // LSE计算: [0,128) = rowSumAcc/pScale, [128,256) = max(half->float)
    TBuf<> lseTmpBuf_;

    const ConstInfoX &constInfo_;
    half negativeHalfScalar_{static_cast<half>(-65504.0f)}; // HF16_MIN: FP16域的padding/初始max值
    float pScaleValue_{1.0f};
    uint32_t minValue_{0}; // LSE无效行检测的min check value, 由UpdateMinCheckValue计算

    // ==================== Functions ======================
    __aicore__ inline QuantFlashAttnBlockVecMxfp8Fp16(ConstInfoX &constInfo)
        : constInfo_(constInfo){};

    __aicore__ inline void InitVecBlock(TPipe *pipe, __gm__ uint8_t *actualSeqQlenAddr,
                                        __gm__ uint8_t *actualSeqKvlenAddr, __gm__ uint8_t *pScale,
                                        __gm__ uint8_t *attenMask, __gm__ uint8_t *softmaxLse,
                                        __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace)
    {
        tPipe_ = pipe;
        // 初始minValue: FP16域无效行max初值(-65504)转float后的位表示
        float minInit = static_cast<float>(negativeHalfScalar_);
        minValue_ = *reinterpret_cast<uint32_t *>(&minInit);
        UpdateMinCheckValue();
        InitVecInput(actualSeqQlenAddr, actualSeqKvlenAddr, pScale, softmaxLse, attentionOut);
    }

    __aicore__ inline void InitVecInput(__gm__ uint8_t *actualSeqQlenAddr, __gm__ uint8_t *actualSeqKvlenAddr,
                                        __gm__ uint8_t *pScale, __gm__ uint8_t *softmaxLse,
                                        __gm__ uint8_t *attentionOut)
    {
        this->attentionOutGm_.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        if (constInfo_.isSoftmaxLseEnable) {
            softmaxLseGm_.SetGlobalBuffer((__gm__ float *)softmaxLse);
        }
        actualSeqLengthsGmQ_.SetGlobalBuffer((__gm__ int32_t *)actualSeqQlenAddr, constInfo_.seqUsedQSize);
        if constexpr (isFp8) {
            if (pScale != nullptr) {
                pScaleGm_.SetGlobalBuffer((__gm__ float *)pScale);
                pScaleValue_ = this->pScaleGm_.GetValue(0);
            }
        }
    }

    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                       RunInfoX runInfo, uint32_t subLoop)
    {
        bmm1ResBuf.WaitCrossCore();
        if (unlikely(runInfo.actVecMSize == 0)) {
            // CV 1:1: subBlock1不参与VEC计算, 仅推进buffer同步状态
            bmm1ResBuf.SetCrossCore();
            if (((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) ||
                (runInfo.actSingleLoopS2Size <= s2SplitSize)) {
                outputBuf.SetCrossCore();
            }
            return;
        }

        ProcessVec1Dn(outputBuf, bmm1ResBuf, runInfo, subLoop);
    }

    __aicore__ inline void ClearOutput()
    {
        if (IsInitAttentionOutGm()) {
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId); // 释放剩余ub
            InitOutputSingleCore();
            if (constInfo_.isSoftmaxLseEnable) {
                InitLseOutputSingleCore();
            }
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            SyncAll();
        }
    }

    __aicore__ inline bool IsInitAttentionOutGm()
    {
        return constInfo_.needInitOutput;
    }

    __aicore__ inline void InitOutputSingleCore()
    {
        // BNSD: tSize = B * S1
        int64_t tSize = constInfo_.bSize * constInfo_.s1Size;
        int64_t totalOutputSize = tSize * constInfo_.realN2Size * constInfo_.realGSize * constInfo_.dSizeV;
        // CV 1:1: AIV数 = coreNum(AIC数), aivIdx ∈ [0, coreNum), 每核均分total
        int64_t singleCoreSize = (totalOutputSize + constInfo_.coreNum - 1) / constInfo_.coreNum;
        int64_t tailSize = totalOutputSize - constInfo_.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            if constexpr (IsSameType<OUT_T, int8_t>::value) {
                GlobalTensor<half> attentionOutTmpGm;
                attentionOutTmpGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(attentionOutGm_.GetPhyAddr(0)));
                matmul::InitOutput<half>(attentionOutTmpGm[constInfo_.aivIdx * singleCoreSize], singleInitOutputSize,
                                         0);
            } else {
                matmul::InitOutput<OUT_T>(attentionOutGm_[constInfo_.aivIdx * singleCoreSize], singleInitOutputSize, 0);
            }
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    __aicore__ inline void InitLseOutputSingleCore()
    {
        int64_t tSize = constInfo_.bSize * constInfo_.s1Size;
        int64_t totalOutputSize = tSize * constInfo_.realN2Size * constInfo_.realGSize;
        // CV 1:1: AIV数 = coreNum(AIC数), aivIdx ∈ [0, coreNum), 每核均分total
        int64_t singleCoreSize = (totalOutputSize + constInfo_.coreNum - 1) / constInfo_.coreNum;
        int64_t tailSize = totalOutputSize - constInfo_.aivIdx * singleCoreSize;
        int64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;

        if (singleInitOutputSize > 0) {
            WaitFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
            matmul::InitOutput<float>(softmaxLseGm_[constInfo_.aivIdx * singleCoreSize], singleInitOutputSize,
                                      3e+99); // 3e+99: set the value of invalid batch to inf
            SetFlag<AscendC::HardEvent::MTE3_V>(initOutputEventId);
        }
    }

    // =================================Private Functions=================================

    __aicore__ inline void ProcessVec1Dn(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                         Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                         RunInfoX runInfo, uint32_t subLoop)
    {
        uint32_t softmaxBufIdx = runInfo.mloop % (PRELOAD_N + 1);
        uint32_t expBufIdx = runInfo.loop % (PRELOAD_N + 1);
        int64_t stage1Offset = subLoop % DB;

        // FP16 softmax状态tensor: half, 每行m=128个元素; 无sum tensor(row_sum由C2阶段随路得出)
        LocalTensor<half> maxUb = this->softmaxMaxBuf_[softmaxBufIdx].template Get<half>();
        LocalTensor<half> preLoopMaxUb = this->preLoopMaxBuf_.template Get<half>();
        LocalTensor<half> expUb = this->softmaxExpBuf_[expBufIdx].template Get<half>();

        // C1 FIXPIPE随路cast FP16: bmm1结果直接为half, softmaxscale已随路乘完
        LocalTensor<half> mmRes = bmm1ResBuf.template GetTensor<half>();
        auto stage1CastTensor = this->stage1OutQue_[stage1Offset].template AllocTensor<INPUT_T>();
        constexpr int32_t pScaleByteOffset = 2 * 16640; // 2*16640: P区域(2-way打包)的softmax res UB size
        LocalTensor<fp8_e8m0_t> pScaleSubLoop0Tensor =
            stage1CastTensor.template ReinterpretCast<fp8_e8m0_t>()[pScaleByteOffset];

        static constexpr int32_t s2BaseSizeCur = s2BaseSize >> 1;
        uint32_t s2CalcSize = runInfo.actSingleLoopS2Size;
        if (runInfo.actSingleLoopS2Size > s2SplitSize) {
            s2CalcSize = subLoop == 0 ? s2BaseSizeCur : runInfo.actSingleLoopS2Size - s2BaseSizeCur;
        }

        // CV 1:1单V核: m恒为128(一行S1的FP16恰好占满一个寄存器), 尾块不足128行时按整块处理, 无效行不搬出
        constexpr uint32_t vfM = mBaseSize;
        if (unlikely(runInfo.isFirstS2Loop)) {
            // NoUpdate: 首个S2 loop无历史max; softmaxscale不乘(C1随路), row_sum不算(C2随路)
            FaVectorApi::ProcessVec1VfDnMxfp8V2<half, INPUT_T, false, s2BaseSizeCur>(
                stage1CastTensor, maxUb, mmRes, expUb, this->vselrIndexesBuf_, pScaleSubLoop0Tensor, vfM, s2CalcSize,
                pScaleValue_, negativeHalfScalar_, preLoopMaxUb, subLoop);
        } else {
            // Update: 与历史max合并, expUb输出O的重缩放因子exp(旧max-新max)(half)
            FaVectorApi::ProcessVec1VfDnMxfp8V2<half, INPUT_T, true, s2BaseSizeCur>(
                stage1CastTensor, maxUb, mmRes, expUb, this->vselrIndexesBuf_, pScaleSubLoop0Tensor, vfM, s2CalcSize,
                pScaleValue_, negativeHalfScalar_, preLoopMaxUb, subLoop);
        }
        bmm1ResBuf.SetCrossCore();

        this->stage1OutQue_[stage1Offset].template EnQue(stage1CastTensor);
        this->stage1OutQue_[stage1Offset].template DeQue<INPUT_T>();
        //-------------------------Data copy to l1-------------------------
        uint64_t pScaleL1Offset = mBaseSize * s2BaseSize; // PScale在L1的偏移量（单位：元素）
        LocalTensor<fp8_e8m0_t> mm2AScaleL1Tensor = outputBuf.GetTensor<fp8_e8m0_t>(pScaleL1Offset);
        constexpr uint16_t pScaleDstStride = s2BaseSizeCur / MXFP_GROUP_SIZE / 2 - 1;
        constexpr uint32_t copyTimes = s2BaseSizeCur / MXFP_GROUP_SIZE / 2;
        if ((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) {
            // subLoop==1时VF写入了P0的scale修正(每行一个共m=128个E8M0, DIST_PACK_B16存储每值2B共256B),
            // 按s2分组复制到L1: 每次拷贝mBaseSize/16个32B块(源256B全量), dstStride=3使块间隔4个32B单元,
            // L1按16行x2分形排布(与原mxfp8 DN单核64行4块同构), copyTimes次迭代覆盖P0全k段1KB区域
            constexpr uint16_t pScaleBlockCount = mBaseSize / 16;
            for (uint16_t i = 0; i < copyTimes; i++) {
                DataCopy(mm2AScaleL1Tensor[i * 32], pScaleSubLoop0Tensor, {pScaleBlockCount, 1, 0, pScaleDstStride});
            }
        }

        LocalTensor<INPUT_T> mm2AL1Tensor = outputBuf.GetTensor<INPUT_T>();
        int64_t subLoopOffset = s2BaseSizeCur * mBaseSize * subLoop;
        constexpr uint16_t elementSize = 32;
        // CV 1:1: 单V核处理全部mBaseSize行(旧DN为mBaseSize>>1, 另一半由subBlock1处理)
        constexpr uint32_t singleProcessSOuterSize = mBaseSize;
        constexpr uint32_t actCopyCount = (singleProcessSOuterSize + elementSize - 1) / elementSize;
        // 2-way新打包: P区域16个2080B行(旧DN单核8个); srcStride由旧的66(=2*65-64)变为196(=4*65-64),
        // 使block(i,k)命中行 2080*(i+4k), 4次迭代x4个block恰好覆盖全部16行;
        // dst侧公式与原mxfp8 DN一致(8192*i+2048*b), L1按B位NZ分形(kg外层/nb内层)排布,
        // 与MM2的LoadDataToL0BMx(srcStride=8个512B)寻址匹配
        constexpr uint16_t pSrcStride = 4 * (((s2BaseSizeCur >> 2) + 1)) - s2BaseSizeCur / 4;
        for (uint32_t i = 0; i < actCopyCount; i++) {
            uint32_t dstOffset = 32 * s2BaseSizeCur * i + subLoopOffset;
            uint32_t srcOffset = i * (((s2BaseSizeCur >> 2) + 1) << 5);
            DataCopy(mm2AL1Tensor[dstOffset], stage1CastTensor[srcOffset], {4, s2BaseSizeCur / 4, pSrcStride, 0});
        }
        //-----------------------------------------------------------------
        this->stage1OutQue_[stage1Offset].template FreeTensor(stage1CastTensor);

        if (((runInfo.actSingleLoopS2Size > s2SplitSize) && (subLoop % 2 == 1)) ||
            (runInfo.actSingleLoopS2Size <= s2SplitSize)) {
            outputBuf.SetCrossCore();
        }
        // LSE已改由V2阶段输出(见SoftmaxLseCopyOut): V1阶段无sum(row_sum在C2随路得出),
        // lse = alignedMax + ln(rowSumAcc/pScale), rowSumAcc由V2跨loop累加(MM2输出尾带首行)
        return;
    }

    // 与V1 VF对无效行max的half域LN2对齐保持一致: min * INV_LN2 -> ceil -> * LN2
    // fp16路径V1不乘softmaxscale(C1阶段FIXPIPE随路完成), 此处也不乘
    __aicore__ inline void UpdateMinCheckValue()
    {
        float min = *((float *)&minValue_);
        float tmp = min * INV_LN2;
        // int32_t 范围 [-2147483648, 2147483647], 下限约 -2.1e9
        // 若 tmp 小于 int32 最小值, float 精度丢失, 直接原样计算
        if (tmp > -2147483648.0f) {
            min = static_cast<int32_t>(tmp) * LN2;
        } else {
            min = tmp * LN2;
        }
        minValue_ = static_cast<uint32_t>(*reinterpret_cast<int32_t *>(&min));
    }

    // 函数1: 首loop O初始化: vec2ResUb(d x 128) <- mmRes前d行(连续直拷, 含无效列无妨)
    __aicore__ inline void CopyMm2ResToUb(LocalTensor<T> &vec2ResUb, LocalTensor<T> &mmRes, RunInfoX &runInfo)
    {
        // 首loop无历史O, M直接作为O初值(免exp乘加); 转置布局连续直拷, 无效列数据不搬出
        DataCopy(vec2ResUb, mmRes, dTemplateAlign64 * mBaseSize);
    }

    // 函数2: rowSumAcc初始化: 首loop直接取mmRes第d行(128 float连续)
    __aicore__ inline void InitRowSumAcc(LocalTensor<T> &mmRes, RunInfoX &runInfo)
    {
        LocalTensor<T> rowSumAccUb = this->rowSumAccBuf_.template Get<T>();
        LocalTensor<T> rsUb = mmRes[dTemplateAlign64 * mBaseSize]; // 第d行起32行(全等)为row_sum, 取首行(每列j一个值)
        DataCopy(rowSumAccUb, rsUb, mBaseSize);                    // 128 float = 512B, 32B对齐
    }

    // 函数3: half exp(half) -> float: softmaxExpBuf[loop%(PRELOAD_N+1)] -> expF32Buf
    __aicore__ inline void CastExpToF32(LocalTensor<float> &expF32Ub, RunInfoX &runInfo)
    {
        LocalTensor<half> expUb = this->softmaxExpBuf_[runInfo.loop % (PRELOAD_N + 1)].template Get<half>();
        Cast(expF32Ub, expUb, RoundMode::CAST_NONE, mBaseSize); // half->float无损
    }

    // 函数4: rowSumAcc = rowSumAcc * exp + 本loop row_sum(一维128 element-wise)
    __aicore__ inline void UpdateRowSumAcc(LocalTensor<T> &mmRes, LocalTensor<float> &expF32Ub)
    {
        LocalTensor<T> rowSumAccUb = this->rowSumAccBuf_.template Get<T>();
        LocalTensor<T> rsUb = mmRes[dTemplateAlign64 * mBaseSize];
        Mul(rowSumAccUb, rowSumAccUb, expF32Ub, mBaseSize);
        Add(rowSumAccUb, rowSumAccUb, rsUb, mBaseSize);
    }

    // 函数5: 转置+cast bf16+搬出:
    // 1. in-place cast: O(d,128)float -> 同地址前半OUTPUT_T(原方案Bmm2ResCastAndCopyOut同款);
    // 2. 转置: (d,128)OUTPUT_T -> mm2TransBuf(128,d), 32x32 tile两级DataCopy,
    //    gather(srcStride=(行距256B-块长64B)/32=6, dstStride=0) -> scatter(srcStride=0, dstStride=行距-块长);
    //    两步均为MTE3, 队列内保序免逐tile同步;
    // 3. 搬出: (128,d)行主序 -> GM BNSD, 复用CopyAttenOutUbToGm<S1_ONLY>(转置后布局恰好匹配)
    __aicore__ inline void TransposeAndCastOut(RunInfoX &runInfo, LocalTensor<T> &vec2ResUb, uint32_t vecMSize)
    {
        constexpr uint32_t dAlign = dTemplateAlign64;
        if (unlikely(vecMSize == 0)) {
            return;
        }
        // 1. in-place cast到前半(OUTPUT_T尺寸为T的1/2以下, 不破坏后序待读数据)
        LocalTensor<OUTPUT_T> attenOutUb;
        attenOutUb.SetAddr(vec2ResUb.address_);
        Cast(attenOutUb, vec2ResUb, RoundMode::CAST_ROUND, dAlign * mBaseSize);
        SetFlag<HardEvent::V_MTE3>(vToMte3Id_[0]); // V写前半 -> MTE3读
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id_[0]);

        // 2. 32x32 tile两级DataCopy转置(纯MTE3, 队列内保序)
        LocalTensor<OUTPUT_T> outUb = this->mm2TransBuf_.template Get<OUTPUT_T>();
        LocalTensor<OUTPUT_T> tmpUb = this->transTmpBuf_.template Get<OUTPUT_T>();
        constexpr uint32_t tileLen = 32;
        DataCopyExtParams gatherParams;
        gatherParams.blockCount = tileLen;                  // tile内源行数(d方向32行)
        gatherParams.blockLen = tileLen * sizeof(OUTPUT_T); // 每行取32列
        gatherParams.srcStride = (mBaseSize - tileLen) * sizeof(OUTPUT_T) / BYTE_BLOCK; // 源行距128列
        gatherParams.dstStride = 0;                                                     // tmp连续(32x32行主序)
        DataCopyExtParams scatterParams;
        scatterParams.blockCount = tileLen; // tmp行 -> 目标S1行
        scatterParams.blockLen = tileLen * sizeof(OUTPUT_T);
        scatterParams.srcStride = 0;                                                      // tmp行距=块长, 连续
        scatterParams.dstStride = dAlign * sizeof(OUTPUT_T) - tileLen * sizeof(OUTPUT_T); // 目标行距dAlign
        uint32_t colBlocks = (vecMSize + tileLen - 1) / tileLen; // S1方向块数(尾块越界列不搬出)
        for (uint32_t r = 0; r < dAlign / tileLen; r++) {        // d方向块
            for (uint32_t c = 0; c < colBlocks; c++) {           // S1方向块
                // gather: tile(32行x32列) -> tmp(32x32行主序)
                DataCopyPad(tmpUb, attenOutUb[(r * tileLen) * mBaseSize + c * tileLen], gatherParams);
                // scatter: tmp行k -> out行(32c+k)列32r起32个(转置位置)
                DataCopyPad(outUb[(c * tileLen) * dAlign + r * tileLen], tmpUb, scatterParams);
            }
        }

        // 3. 搬出: (mBaseSize, dAlign)行主序OUTPUT_T -> GM BNSD
        FaUbTensor<OUTPUT_T, !isDAligned> ubTensor{.tensor = outUb, .rowCount = vecMSize, .colCount = dAlign};
        GmCoordS1Only gmCoord{.bIdx = runInfo.bIdx,
                              .n2Idx = runInfo.realN2Idx,
                              .gIdx = 0, // DN分支realGSize=1
                              .s1Idx = runInfo.gS1Idx + runInfo.vecMbaseIdx,
                              .dIdx = 0,
                              .s1DealSize = vecMSize,
                              .dDealSize = (uint32_t)constInfo_.dSizeV};
        constexpr GmFormat OUT_FORMAT = GmFormat::BNGSD;
        FaGmTensor<OUTPUT_T, OUT_FORMAT, int32_t> outGmTensor;
        outGmTensor.gmTensor = attentionOutGm_;
        outGmTensor.offsetCalculator.Init(constInfo_.bSize, constInfo_.realN2Size, constInfo_.realGSize,
                                          constInfo_.s1Size, constInfo_.dSizeV, *qActSeqLensParser_);
        CopyAttenOutUbToGm<OUTPUT_T, OUT_FORMAT, UbFormat::S1_ONLY> copyAttenOutUbToGm;
        copyAttenOutUbToGm(outGmTensor, ubTensor, gmCoord);
    }

    // 函数6: LSE输出: lse = alignedMax + ln(rowSumAcc/pScale), invalid行(max==HF16_MIN)输出3e+99;
    // BNSD下lse GM为[B, N2*G, S1], 连续s1为连续内存, 紧凑计算后一次DataCopyPad搬出;
    // 按actualS1裁剪, 越界行保留ClearOutput阶段初始化的3e+99
    __aicore__ inline void SoftmaxLseCopyOut(RunInfoX &runInfo, uint32_t vecMSize)
    {
        if (unlikely(vecMSize == 0)) {
            return;
        }
        // lseTmp布局: [0, mBaseSize) = maxF32(half->float), [mBaseSize, 2*mBaseSize) = lse结果
        LocalTensor<float> lseTmp = this->lseTmpBuf_.template Get<float>();
        LocalTensor<float> maxF32Ub = lseTmp[0];
        LocalTensor<float> lseUb = lseTmp[mBaseSize];
        LocalTensor<half> maxUb = this->softmaxMaxBuf_[runInfo.mloop % (PRELOAD_N + 1)].template Get<half>();
        Cast(maxF32Ub, maxUb, RoundMode::CAST_NONE, mBaseSize); // half->float无损

        LocalTensor<T> rowSumAccUb = this->rowSumAccBuf_.template Get<T>();
        float pScaleRec = 1.0f / pScaleValue_;    // pScale默认1.0, lse = max + ln(rs/pScale)
        float minValueF = *((float *)&minValue_); // half域LN2对齐后的无效行max(float域)
        FaVectorApi::ComputeLseTrans<float>(lseUb, rowSumAccUb, maxF32Ub, vecMSize, pScaleRec, minValueF);
        SetFlag<HardEvent::V_MTE3>(vToMte3Id_[0]); // V写lse -> MTE3读
        WaitFlag<HardEvent::V_MTE3>(vToMte3Id_[0]);

        uint32_t s1Start = runInfo.gS1Idx + runInfo.vecMbaseIdx;
        uint64_t qActSeqLens = qActSeqLensParser_->GetActualSeqLength(runInfo.bIdx);
        uint32_t dealCount = vecMSize;
        if (qActSeqLens < (uint64_t)s1Start + vecMSize) {
            dealCount = qActSeqLens > s1Start ? (uint32_t)(qActSeqLens - s1Start) : 0;
        }
        if (dealCount == 0) {
            return;
        }
        uint64_t gmOffset = (uint64_t)runInfo.bIdx * constInfo_.n2Size * constInfo_.gSize * constInfo_.s1Size +
                            (uint64_t)runInfo.n2Idx * constInfo_.gSize * constInfo_.s1Size + s1Start;
        DataCopyExtParams lseCopyParams;
        lseCopyParams.blockCount = 1;
        lseCopyParams.blockLen = dealCount * sizeof(float);
        lseCopyParams.srcStride = 0;
        lseCopyParams.dstStride = 0;
        DataCopyPad(softmaxLseGm_[gmOffset], lseUb, lseCopyParams);
    }

    /* ============V2 (FP16 softmax) 主流程=============
     * MM2输出mmRes为转置布局(d+32)x128 float行主序:
     *   前 d 行 = (P*V)^T, 第 d 行起32行 = row_sum(全1带随路得出, 全等, 每列j一个值);
     * O累加器(stage2OutBuf)保持该转置布局(d, 128), softmax因子(exp/row_sum)是
     * per-S1行(即转置布局的列j), 沿行内连续 -> 乘加/除法免转置, 行主序直接算;
     * 转置只在最后输出做一次(32x32 tile两级DataCopy + 随路cast bf16)。
     * 全局sum语义: MM2尾行row_sum只是本loop的ΣP(对齐当前max),
     * V2维护rowSumAcc = rowSumAcc*exp + 本loop rs(一维128元素, 开销极小)。
     */
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfoX runInfo)
    {
        bmm2ResBuf.WaitCrossCore();
        if (unlikely(runInfo.actVecMSize == 0)) {
            bmm2ResBuf.SetCrossCore();
            return;
        }
        uint32_t vecMSize = runInfo.actVecMSize;                          // DN: 单V核处理全部S1行
        LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();        // (d+32)x128 float
        LocalTensor<T> vec2ResUb = this->stage2OutBuf_.template Get<T>(); // (d, 128) O累加器
        WaitFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);

        if (unlikely(runInfo.isFirstS2Loop)) {
            // 首loop: O直接初始化为M(免exp乘加); rowSumAcc初始化为本loop row_sum
            CopyMm2ResToUb(vec2ResUb, mmRes, runInfo);
            InitRowSumAcc(mmRes, runInfo);
            // MTE3写(vec2ResUb/rowSumAcc) -> 后续VF/LastDivTrans读(V), 跨pipe同步
            SetFlag<HardEvent::MTE3_V>(mte3ToVId_[1]);
            WaitFlag<HardEvent::MTE3_V>(mte3ToVId_[1]);
        } else {
            LocalTensor<float> expF32Ub = this->expF32Buf_.template Get<float>();
            CastExpToF32(expF32Ub, runInfo);
            UpdateRowSumAcc(mmRes, expF32Ub);
            FaVectorApi::FlashUpdateTrans<T, dTemplateAlign64, mBaseSize>(vec2ResUb, mmRes, expF32Ub);
        }
        bmm2ResBuf.SetCrossCore();

        if (unlikely(runInfo.isLastS2Loop)) {
            // 单loop时O=M已就绪, 多loop时O已完成乘加, 统一走除法
            LocalTensor<T> rowSumAccUb = this->rowSumAccBuf_.template Get<T>();
            FaVectorApi::LastDivTrans<T, dTemplateAlign64, mBaseSize>(vec2ResUb, rowSumAccUb);
            TransposeAndCastOut(runInfo, vec2ResUb, vecMSize);
            if (constInfo_.isSoftmaxLseEnable) {
                SoftmaxLseCopyOut(runInfo, vecMSize);
            }
        }
        SetFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
    }

    __aicore__ inline void SoftmaxInitBuffer()
    {
        // FP16 softmax状态tensor: 每行m=128个half(256B), 无sum相关buffer
        tPipe_->InitBuffer(softmaxMaxBuf_[0], 256); // [128, 1]
        tPipe_->InitBuffer(softmaxMaxBuf_[1], 256); // [128, 1]
        tPipe_->InitBuffer(softmaxMaxBuf_[2], 256); // [128, 1]

        tPipe_->InitBuffer(softmaxExpBuf_[0], 256); // [128, 1]
        tPipe_->InitBuffer(softmaxExpBuf_[1], 256); // [128, 1]
        tPipe_->InitBuffer(softmaxExpBuf_[2], 256); // [128, 1]

        tPipe_->InitBuffer(preLoopMaxBuf_, 256); // [128, 1]
    }

    __aicore__ inline void InitBuffers()
    {
        SoftmaxInitBuffer();
        // V2的O暂存: MM2输出为float, 单V核处理完整mBaseSize行(CV 1:1, 旧DN为mBaseSize/2行)
        tPipe_->InitBuffer(stage2OutBuf_, mBaseSize * dTemplateAlign64 * sizeof(T));
        // P(FP8 2-way打包)区域33280B + pScale(E8M0 x128)256B对齐
        tPipe_->InitBuffer(stage1OutQue_[0], 1, 2 * 16640 + 256);
        tPipe_->InitBuffer(stage1OutQue_[1], 1, 2 * 16640 + 256);
        tPipe_->InitBuffer(commonTBuf_, 512);
        // V2阶段: O累加器保持MM2输出的转置布局(d, 128), 与stage2OutBuf同尺寸(字节数不变)
        tPipe_->InitBuffer(rowSumAccBuf_, mBaseSize * sizeof(float));
        tPipe_->InitBuffer(expF32Buf_, mBaseSize * sizeof(float));
        tPipe_->InitBuffer(mm2TransBuf_, mBaseSize * dTemplateAlign64 * sizeof(OUTPUT_T));
        tPipe_->InitBuffer(transTmpBuf_, 32 * 32 * sizeof(OUTPUT_T));
        tPipe_->InitBuffer(lseTmpBuf_, 2 * mBaseSize * sizeof(float));

        // DN->NZ Gather索引表: 按VF V2的2-way打包重排(旧4-way为4x64转置: SetValue(i*64+j, i+4j),
        // 每pack寄存器[列A|列B]各128B经Gather重排为NZ); 新2-way为2x128等价形式:
        // 位置128i+j取值i+2j, 即输出位置p取源(p>>7)+2*(p&127), 与VF V2的Gather语义一致
        tPipe_->InitBuffer(vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::DN_INDEX)], 256);
        LocalTensor<uint8_t> vselrIndexesTensor =
            vselrIndexesBuf_[static_cast<int>(VselrIndexEnum::DN_INDEX)].template Get<uint8_t>();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 128; j++) {
                vselrIndexesTensor.SetValue(i * 128 + j, i + (j << 1));
            }
        }
    }

    __aicore__ inline void AllocEventID()
    {
        mte3ToVId_[0] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        mte3ToVId_[1] = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        vToMte3Id_[0] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        vToMte3Id_[1] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::MTE3_V>(mte3ToVId_[0]);
        SetFlag<HardEvent::MTE3_V>(mte3ToVId_[1]);
    }

    __aicore__ inline void FreeEventID()
    {
        WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId_[0]);
        WaitFlag<AscendC::HardEvent::MTE3_V>(mte3ToVId_[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId_[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(mte3ToVId_[1]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id_[0]);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(vToMte3Id_[1]);
    }
};

template <typename INPUT_T, typename T, typename OUTPUT_T, LayOutTypeEnum layout = LayOutTypeEnum::None,
          LayOutTypeEnum outLayout = LayOutTypeEnum::None, S1TemplateType s1TemplateType = S1TemplateType::Aligned128,
          S2TemplateType s2TemplateType = S2TemplateType::Aligned128,
          DTemplateType dTemplateType = DTemplateType::Aligned128,
          DTemplateType dVTemplateType = DTemplateType::Aligned128, bool hasAtten = false, uint8_t KvLayoutType = 0,
          bool isFd = false, bool useDn = false>
class QuantFlashAttnBlockVecMxfp8Fp16Dummy {
public:
    static constexpr bool FLASH_DECODE = isFd;
    using OUT_T = OUTPUT_T;
    using ConstInfoX = ConstInfo_t;
    __aicore__ inline QuantFlashAttnBlockVecMxfp8Fp16Dummy(ConstInfoX &constInfo){};
};
} // namespace BaseApi
#endif // QUANT_FLASH_ATTN_BLOCK_VEC_MXFP8_FP16_H_
