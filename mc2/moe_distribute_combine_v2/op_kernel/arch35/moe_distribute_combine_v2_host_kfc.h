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
 * \file moe_distribute_combine_v2_host_kfc.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_COMBINE_V2_A5_LAYERED_HOSTKFC_H
#define MOE_DISTRIBUTE_COMBINE_V2_A5_LAYERED_HOSTKFC_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/reduce/sum.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../moe_distribute_combine_v2_tiling.h"
#if __has_include("../../moe_distribute_dispatch_v2/check_winsize.h")
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../moe_distribute_dispatch_v2/check_winsize.h"
#include "../../moe_distribute_dispatch_v2/moe_distribute_v2_base.h"
#else 
#include "../../../common/op_kernel/moe_distribute_base.h"
#include "../../../moe_distribute_dispatch_v2/op_kernel/check_winsize.h" 
#include "../../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_base.h" 
#endif

namespace MoeDistributeCombineV2A5Impl {
constexpr uint8_t BUFFER_NUM = 2;                      // 多buf
constexpr uint32_t STATE_OFFSET = 32U;                 // 状态空间偏移地址
constexpr uint32_t STATE_SIZE = 1024UL * 1024UL;       // 1M
constexpr uint32_t COMBINE_STATE_OFFSET = 64U * 1024U; // 本卡状态空间偏移地址，前面的地址给dispatch用
constexpr uint8_t EP_DOMAIN = 0;
constexpr uint8_t TP_DOMAIN = 1;
constexpr uint32_t UB_ALIGN = 32U; // UB按32字节对齐
constexpr uint32_t FLOAT_PER_UB_ALIGN = 8U;
constexpr uint64_t WIN_STATE_OFFSET = 500UL * 1024UL;
constexpr uint64_t STATE_WIN_OFFSET = 975UL * 1024UL; // 预留48*512内存
constexpr uint64_t TIMEOUT_OFFSET = 1000UL * 1024UL;
constexpr uint64_t TIMEOUT_DETECTION_THRESHOLD = 50000UL;
constexpr uint64_t CYCLES_PER_US = 50UL;
constexpr uint64_t TIMEOUT_DETECTION_TX_UNITS = 8UL;
constexpr uint64_t STATE_HCCL_OFFSET = 32UL;
constexpr uint32_t EXPAND_IDX_INFO = 3U; // expand_idx是按3元组保存信息，分别为rank_id token_id topk_id
constexpr uint32_t ALIGNED_LEN = 256U;   // blockReduceMax中，最多支持连续256字节数据参与计算
constexpr float SCALE_PARAM = 127.0;     // 计算量化参数所需的缩放倍数
constexpr uint64_t ALIGNED_LEN_256 = 256UL;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint32_t REDUCE_NUM = 8U;
constexpr uint32_t ELASTIC_INFO_OFFSET = 4U;
constexpr uint32_t RANK_LIST_NUM = 2;
constexpr uint8_t EP_WORLD_SIZE_IDX = 1U;
constexpr uint8_t SHARE_RANK_NUM_IDX = 2U;
constexpr uint8_t MOE_NUM_IDX = 3U;
constexpr uint32_t DIM_NUM = 2;
constexpr size_t MASK_CALC_NEED_WORKSPACE = 10UL * 1024UL;
constexpr uint32_t BLOCK_NUM = ALIGNED_LEN / UB_ALIGN; // blockReduceMax中，最多支持连续256字节数据参与计算
// =============新增常量=================
constexpr size_t MAX_REDUCE_TILE_SIZE = 32;
constexpr uint32_t BATCH_SIZE = 16;
constexpr uint64_t READY_FLAG = 12345UL;
constexpr uint64_t PACKED_TOKEN_READY_FLAG = 1UL;
constexpr uint32_t BLOCK_SIZE = 32U;
constexpr uint32_t SPLIT_BLOCK_SIZE = 512U;
constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U;
constexpr uint32_t SPLIT_BLOCK_FLAG_SIZE = 32U;
constexpr uint8_t MAX_SERVER_RANK_SIZE = 16;
constexpr uint32_t SHARE_FLAG_ENTRY_STRIDE_BYTES = 512U;
constexpr uint32_t TOKEN_META_BYTES = 12U;    // scale(float)+originRankId(u32)+originTokenId(u32)
constexpr uint32_t TOKEN_ONE_META_BYTES = 4U; // originTokenId(u32)
constexpr uint32_t FLAG_CNT_U64 = 2;          // flag + cnt 数量为2
constexpr size_t TARGET_CNT_CALC_NEED_WORKSPACE = 10UL * 1024UL;
constexpr uint32_t BLOCK_COPY_BYTES_FIRST = 256U; // 第一个copy chunk的字节数 8*32B
constexpr uint32_t BLOCK_COPY_BYTES_SECOND = SPLIT_BLOCK_SIZE - BLOCK_COPY_BYTES_FIRST; // 第二个copy chunk的字节数
//================================

#define TemplateMoeDistributeCombineV2A5KFCTypeClass                                                                                           \
    typename ExpandXType, typename XType, typename ExpandIdxType, bool IsNeedReduceScatter, bool IsInt8Quant
#define TemplateMoeDistributeCombineV2A5KFCTypeFunc ExpandXType, XType, ExpandIdxType, IsNeedReduceScatter, IsInt8Quant

using namespace MoeDistributeV2Base;
using namespace AscendC;
template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
class MoeDistributeCombineV2A5LayeredHostcpu {
public:
    __aicore__ inline MoeDistributeCombineV2A5LayeredHostcpu(){};
    __aicore__ inline void Init(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount,
                                GM_ADDR tpSendCount, GM_ADDR expertScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX,
                                GM_ADDR elasticInfo, GM_ADDR oriX, GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2,
                                GM_ADDR constExpertV, GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeCombineV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitInputAndOutput(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx,
                                              GM_ADDR epSendCount, GM_ADDR expertScales, GM_ADDR xActiveMask,
                                              GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX,
                                              GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2,
                                              GM_ADDR constExpertV, GM_ADDR XOut);
    __aicore__ inline void InitAttrs(const MoeDistributeCombineV2TilingData *tilingData);
    __aicore__ inline void InitTilingAttrs(const MoeDistributeCombineV2TilingData *tilingData);
    __aicore__ inline void InitElasticInfo(uint32_t &sharedExpertRankNum);
    __aicore__ inline void InitElasticInfoTensor();
    __aicore__ inline void InitInt8Quant();
    __aicore__ inline void AlltoAllBuffInitAndMaskCal();
    __aicore__ inline void InitAlltoAllBuffers();
    __aicore__ inline void ReduceScatterTrans();
    __aicore__ inline void TokenMaskCalCnt();
    __aicore__ inline void ExpertMaskCalCnt();
    __aicore__ inline void GenerateActiveMask(half val);
    __aicore__ inline void MaskSpecialExpert();
    __aicore__ inline void MaskAlign();
    __aicore__ inline void SetWaitTpStatus(); // SetWaitTpStatusAndDisPatch()
    __aicore__ inline void CustomAdd(LocalTensor<XType> &dst, LocalTensor<XType> &src0, LocalTensor<XType> &src1);
    __aicore__ inline void Int8QuantProcess();
    __aicore__ inline void Int8DequantProcess(LocalTensor<XType> &src);
    __aicore__ inline void ProcessConstantExpert(uint32_t tokenIndex, uint32_t const_expert_idx, float scaleVal);
    __aicore__ inline void ProcessCopyExpert(uint32_t tokenIndex, float scaleVal);
    __aicore__ inline void ProcessMoeExpert(uint32_t tokenIndexOffset, uint32_t topkId, float scaleVal);
    __aicore__ inline void ExpertScaleCopy(const uint32_t beginIndex, const uint32_t endIndex,
                                           const uint32_t tokenPerAivNum);
    __aicore__ inline void CalConstExpertAlpha(GlobalTensor<ExpandXType> constExpertAlphaGM, uint32_t const_expert_idx,
                                               float &alphaFloat);
    __aicore__ inline void BuffInit();
    __aicore__ inline void SplitCoreCal();
    __aicore__ inline void WaitDispatch(uint32_t tokenIndex);

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
    {
        return (x < y) ? x : y;
    }

    template <typename T>
    inline __aicore__ T RoundUp(const T val, const T align)
    {
        static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
        if (align == 0 || val + align - 1 < val) {
            return val;
        }
        return (val + align - 1) / align * align;
    }

    template <typename T>
    inline __aicore__ T DivCeil(const T val, const T div)
    {
        static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
        if (div == 0 || val + div - 1 < val) {
            return val;
        }
        return (val + div - 1) / div;
    }

    __aicore__ GM_ADDR GetWindowsInAddr(const int32_t rankId)
    {
        return windowInGM_;
    }
    TPipe *tpipe_{nullptr};
    GlobalTensor<ExpandXType> expandXGM_;
    GlobalTensor<bool> xActiveMaskGM_;
    GlobalTensor<int32_t> expertIdsGM_;
    GlobalTensor<ExpandIdxType> expandIdxGM_;
    GlobalTensor<ExpandIdxType> epSendCountGM_;
    GlobalTensor<ExpandIdxType> tpSendCountGM_;
    GlobalTensor<ExpandIdxType> elasticInfoGM_;
    GlobalTensor<float> expertScalesGM_;
    GlobalTensor<XType> sharedExpertXGM_;
    GlobalTensor<XType> expandOutGlobal_;
    GlobalTensor<XType> rankWindow_; // 用于存对端window的变量
    GlobalTensor<XType> tpRankWindow_;
    GlobalTensor<XType> rowTmpGlobal_;
    GlobalTensor<ExpandXType> oriXGM_;
    GlobalTensor<ExpandXType> constExpertAlpha1GM_;
    GlobalTensor<ExpandXType> constExpertAlpha2GM_;
    GlobalTensor<ExpandXType> constExpertVGM_;
    GlobalTensor<uint32_t> selfDataStatusGMTensor_;

    GM_ADDR epWindowGM_;
    GM_ADDR tpWindowGM_;
    GM_ADDR stateGM_;
    GM_ADDR maskCalcWorkspaceGM_;
    GM_ADDR statusDataSpaceGm_;

    LocalTensor<XType> winTpSendCountTensor_;
    LocalTensor<ExpandXType> gmTpSendCountTensor_;
    LocalTensor<float> winTpSendCountFloatTensor_;
    LocalTensor<float> gmTpSendCountFloatTensor_;
    LocalTensor<int32_t> elasticInfoTensor_;
    LocalTensor<bool> maskStrideTensor_;
    LocalTensor<bool> maskGenerateTensor_;
    LocalTensor<uint32_t> dataStateLocalTensor_;
    LocalTensor<float> stateResetTensor_;

    // tiling侧已确保数据上限， 相乘不会越界，因此统一采用uin32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t epWorldSizeOriginal_{0};
    uint32_t tpWorldSize_{0};
    uint32_t epRankId_{0};
    uint32_t epRankIdOriginal_{0};
    uint32_t tpRankId_{0};
    uint32_t coreIdx_{0}; // aiv id
    uint32_t sharedExpertNum_{0};
    uint32_t moeExpertPerRankNum_{0}; // 每张卡部署的moe专家数
    uint32_t moeSendNum_{0};          // moeExpertPerRankNum_ * epWorldSize_
    uint32_t bufferNum_{0};
    uint32_t zeroExpertNum_{0};
    uint32_t copyExpertNum_{0};
    uint32_t constExpertNum_{0};
    uint32_t moeExpertNum_{0};
    uint32_t globalBs_{0};
    __gm__ HcclOpResParam *epWinContext_{nullptr};
    __gm__ HcclOpResParam *tpWinContext_{nullptr};
    uint32_t tpStateOffsetOnWin_{0};
    uint32_t bsKNum_{0};
    uint32_t startTokenId_{0};
    uint32_t endTokenId_{0};
    uint32_t sendCntNum_{0};
    uint32_t ubSize_{0};
    uint32_t dataState_{0};
    uint32_t stateOffset_{0};
    uint64_t activeMaskBsCnt_{0};
    uint64_t winStatusOffset_{0};
    uint64_t totalWinSizeEp_{0};
    uint64_t totalWinSizeTp_{0};
    uint64_t winDataSizeOffsetEp_{0};
    uint64_t winDataSizeOffsetTp_{0};
    uint32_t selfSendCnt_{0};
    uint32_t tpRemoteSendCnt_{0};
    uint32_t activeMaskAlignSize_{0};
    uint32_t hExpandXTypeSize_{0};
    uint32_t hAlign32Size_{0};
    uint32_t hFloatAlign32Size_{0};
    uint32_t hFloatAlign256Size_{0};
    uint32_t hExpandXAlign32Size_{0};
    uint32_t hAlignWinSize_{0};
    uint32_t hAlignWinCnt_{0};
    uint32_t tokenScaleCnt_{0};
    uint32_t scaleNumAlignSize_{0};
    uint32_t flagRcvCount_{0};
    uint32_t axisBsAlignSize_{0};
    uint32_t expertScaleBeginIdx_{0};

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> moeQueue_;
    TQue<QuePosition::VECIN, 1> moeSumQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> gmTpSendCountQueue_;
    TQue<QuePosition::VECIN, 1> gmTpSendCountInQueue_;
    TQue<QuePosition::VECIN, 1> winTpSendCountInQueue_;
    TQue<QuePosition::VECOUT, 1> xOutQueue_;
    TBuf<> readStateBuf_;
    TBuf<> expertScalesBuf_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<> sumFloatBuf_;
    TBuf<> mulBuf_;
    TBuf<> indexCountsBuf_;
    TBuf<> winTpSendCountFloatBuf_;
    TBuf<> gmTpSendCountFloatBuf_;
    TBuf<> tokenBuf_;
    TBuf<> xActMaskTBuf_;
    TBuf<> xActMaskCastTBuf_;
    TBuf<> tokenTargetTBuf_;
    TBuf<> validBsIndexTBuf_;
    TBuf<> xActMaskSumTBuf_;
    TBuf<> stateBuf_;
    TBuf<> stateResetBuf_;
    TBuf<> expertMaskBuf_;
    TBuf<> elasticInfoBuf_;
    bool isInputTokenMaskFlag_ = false;
    bool isInputExpertMaskFlag_ = false;
    bool hasSharedExpertX_ = false;
    bool hasElasticInfoFlag_ = false;
    bool isScalingDownFlag_ = false;
    bool isShareExpertRankFlag_ = false;
    bool enableSpecialExpert_ = false;

    // int8量化
    TBuf<> xAbsBuf_;
    TBuf<> xMaxBuf_;
    TBuf<> xScaleMulBuf_;

    LocalTensor<int8_t> castLocalTensor_;
    LocalTensor<half> fp16CastTensor_;
    LocalTensor<float> absFloatTensor_;
    LocalTensor<float> reduceMaxFloatTensor_;
    LocalTensor<XType> scaleDivTensor_;
    LocalTensor<float> scaleDivFloatTensor_;
    LocalTensor<float> scaleDupLocalTensor_;
    LocalTensor<XType> sendLocalTensor_;
    LocalTensor<half> tokenTargetTensor_;
    LocalTensor<int32_t> validBsIndexTensor_;
    LocalTensor<bool> expertMaskTensor_;
    LocalTensor<float> expertScalesLocal_;
    LocalTensor<float> rowTmpFloatLocal_;
    LocalTensor<float> mulBufLocal_;
    LocalTensor<float> sumFloatBufLocal_;

    uint32_t mask_{0};
    uint32_t repeatNum_{0};
    uint32_t scaleNum_{0};
    float scaleValFloat_;

private:
    /*====================新增函数==========================*/
    __aicore__ inline void CommunInit(const MoeDistributeCombineV2TilingData *tilingData, GM_ADDR workspaceGM);
    __aicore__ inline void SplitCoreByServer();
    // 偏移量计算相关函数
    __aicore__ inline void PrepareServerShareLayout(LocalTensor<ExpandIdxType> expandIdxLocal);
    __aicore__ inline void CalcLocalTargetCnt(LocalTensor<ExpandIdxType> expandIdxLocal);
    __aicore__ inline void BuildPrefixBaseOnCore0();
    __aicore__ inline void LoadLocalBaseFromGm();
    // 1、Server内通信
    __aicore__ inline void AlltoAllDispatch();
    __aicore__ inline void SumToWindow();

    __aicore__ inline void DispatchTokensToShareMem(LocalTensor<ExpandIdxType> expandIdxLocal);
    __aicore__ inline void DispatchTokenInner(uint32_t tkIndex, uint32_t originRankId, uint32_t originTokenId,
                                              uint64_t shareDataAddr);
    __aicore__ inline void UpdateShareFlag();
    __aicore__ inline void ProcessOneServer(uint32_t toServerId, LocalTensor<float> sumTileUb,
                                            LocalTensor<uint32_t> existFlagUb);
    __aicore__ inline void AccumulateRankDataToUb(uint32_t fromLocalRank, uint32_t targetServerId, uint32_t baseId,
                                                  uint32_t endId);
    __aicore__ inline void ReadRankTokenCnt(uint32_t fromLocalRank, uint32_t &tokenCnt, GM_ADDR shareBase);
    __aicore__ inline void TokenToWinOut(GlobalTensor<ExpandXType> dstWinGMTensor, uint32_t originTokenId,
                                         uint32_t toServerId, LocalTensor<float> srcSumTensor);
    __aicore__ inline void WriteWinOutHeader(GlobalTensor<uint64_t> headerGm, uint32_t winTokenCnt);

    // 2、Server间通信
    __aicore__ inline void AlltoAllServerDispatch();
    __aicore__ inline void WaitWinInCount();
    __aicore__ inline void WaitWinInTokenAndCombine();
    __aicore__ inline void tokenAtomicAdd(GlobalTensor<ExpandXType> globalSet, LocalTensor<ExpandXType> localSet);

private:
    /*====================新增server内通信变量==========================*/
    // ================= 分核（按 toServerId） =================
    uint32_t startServerId_{0};
    uint32_t endServerId_{0};
    uint32_t sendServerNum_{0}; // 本核负责的 toServer 数
    // ================= Server 拓扑 =================
    uint32_t serverRankSize_{0}; // 一个 Server 内 Rank 数（host 传入）
    uint32_t serverNum_{0};      // epWorldSize_ / serverRankSize_
    uint32_t serverId_{0};       // 本卡所在Server Id
    uint32_t localRankId_{0};    // Server 内本卡的局部 Rank Id
    // ================= Token  =================
    uint32_t maxLocalBs_{0};
    uint32_t tokenElemNum_{0};        // axisH， 每个 token 包含的元素个数
    uint32_t tokenDataBytes_{0};      // axisH * sizeof(ExpandXType)
    uint32_t tokenDataBytesAlign_{0}; // tokenDataBytes_ 按 UB_ALIGN 对齐后的大小
    uint32_t metaBytesAlign_{0};      // 12Bytes的三元组UB对其后的字节数，32B
    uint32_t tokenMetaBytes_{0};      // tokenData + MetaInfo 按 UB_ALIGN 对齐后的大小
    uint32_t tokenDataBlockNum_{0};   // WinOut中每个 token拆分成多少个480Block
    uint32_t packedTokenBytes_{0};    // 处理后的token在WinOut中占用的字节数
    // ================= Win区 布局 =================
    // Server之间通信
    uint32_t winHeaderBytes_{SPLIT_BLOCK_SIZE};
    uint32_t winOutSliceBytes_{0};
    uint32_t winOutTotalBytes_{0};
    uint32_t winInSliceBytes_{0};
    uint32_t winInTotalBytes_{0};
    // Server内通信
    uint32_t shareFlagSliceBytes_{0};
    uint32_t shareFlagTotalBytes_{0};
    uint32_t shareDataSliceBytes_{0};
    uint32_t shareDataTotalBytes_{0};
    uint64_t flagPadOffset_{0};
    // ================= ShareData 布局 =================
    uint32_t flagU64CopyCntAlign_{0}; // 每个 fromLocalRank 需要拷贝的 flag + cnt 数量（uint64_t）

    // 对worldSize按卡分核，得到每个核上处理的卡的数量
    uint32_t localMoeExpertNum_{0}; // 每张卡的moe专家数
    GM_ADDR windowInGM_;
    GM_ADDR windowOutGM_;
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_; // Server之间通信使用
    // Server 内共享区基址
    uint64_t serverShareAddr_[MAX_SERVER_RANK_SIZE]; // [targetLocalRank]
    // 共享数据token写偏移，shape: [aivNum_][serverRankSize_]
    GlobalTensor<uint32_t> gmCoreTargetCnt_;  // 每核->每 targetLocalRank 的 token 数
    GlobalTensor<uint32_t> gmCoreTargetBase_; // prefix base：每核->每 targetLocalRank 的起始 offset
    GM_ADDR cntGM;
    GM_ADDR baseGM;
    // UB侧，每核私有，用于运行时统计/递增。三个都是长度 serverRankSize_ 的小数组
    TBuf<> localTargetCntBuf_;
    TBuf<> localTargetBaseBuf_;
    TBuf<> localTargetRunBuf_;
    TBuf<> rankTotalCntBuf_;
    LocalTensor<uint32_t> localTargetCnt_;            // 本核 token → targetLocalRank 的计数
    LocalTensor<uint32_t> localTargetBase_;           // 本核从 gmCoreTargetBase_ 读回的 base
    LocalTensor<uint32_t> localTargetRun_;            // 本核运行时递增 offset
    LocalTensor<uint32_t> rankTotalCntToTargetLocal_; // 本rank总的 token → targetLocalRank 的计数
    LocalTensor<float> sumFloatLocal_;
    /*====================新增server间通信变量==========================*/
    LocalTensor<uint64_t> batchWriteItemLocalB64;
    LocalTensor<uint32_t> batchWriteItemLocalB32;
    GlobalTensor<ExpandXType> localOutWindow_;
    GlobalTensor<ExpandXType> localInWindow_;
    LocalTensor<uint32_t> countTensor;
    TBuf<> countBuf_;
    TBuf<> sumBuf_;
    TBuf<> tokenIdBuf_;
    TBuf<> localCntBuf_;
    TBuf<> localTokenIdBuf_;
    TBuf<> localOutTensorBuf_;
    TBuf<> localFlagBuf_;
    TBuf<> tempBuf_;
    TBuf<> outBuf_;
    TBuf<> outBuf1_;
    TBuf<> localOutTempBuf_;
    LocalTensor<uint64_t> localCntTensor_;
};

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::Init(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR tpSendCount,
    GM_ADDR expertScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR XOut, GM_ADDR workspaceGM,
    TPipe *pipe, const MoeDistributeCombineV2TilingData *tilingData)
{
    tpipe_ = pipe;
    coreIdx_ = GetBlockIdx();
    maskCalcWorkspaceGM_ = workspaceGM + coreIdx_ * MASK_CALC_NEED_WORKSPACE;
    InitInputAndOutput(expandX, expertIds, expandIdx, epSendCount, expertScales, xActiveMask, sharedExpertX,
                       elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, XOut);
    InitAttrs(tilingData);
    // 检查hcclwinsize是否越界
    CheckWindowSize(totalWinSizeEp_, epWinContext_->winSize, tpipe_, XOut);
    if constexpr (IsInt8Quant) {
        InitInt8Quant();
    }
    PipeBarrier<PIPE_ALL>();
    // 当前win区划分为前后两半区，连续两次dispatch，切换半区
    winDataSizeOffsetEp_ =
        static_cast<uint64_t>(dataState_) * (tilingData->moeDistributeCombineV2Info.totalWinSizeEp / 2UL);
    winStatusOffset_ = COMBINE_STATE_OFFSET + dataState_ * WIN_STATE_OFFSET; // 前面的预留给dispatch使用
    epWindowGM_ = GetWindowsInAddr(epRankIdOriginal_);
    CommunInit(tilingData, workspaceGM);
    if (isShareExpertRankFlag_) {
        DataCacheCleanAndInvalid<ExpandIdxType, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            epSendCountGM_[epWorldSize_ - 1]);
        selfSendCnt_ = epSendCountGM_(epWorldSize_ - 1);
    } else {
        DataCacheCleanAndInvalid<ExpandIdxType, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            epSendCountGM_[moeSendNum_ - 1]);
        selfSendCnt_ = epSendCountGM_(moeSendNum_ - 1); // moeSendNum_ = epWorldSize_ * moeExpertPerRankNum_;
    }
    SplitCoreCal();
    SplitCoreByServer();
    if constexpr (IsNeedReduceScatter) {
        auto contextGM1 = AscendC::GetHcclContext<1>();
        tpWinContext_ = (__gm__ HcclOpResParam *)contextGM1;
        tpSendCountGM_.SetGlobalBuffer((__gm__ int32_t *)tpSendCount);
        tpWorldSize_ = tilingData->moeDistributeCombineV2Info.tpWorldSize;
        tpRankId_ = tilingData->moeDistributeCombineV2Info.tpRankId;
        tpWindowGM_ = GetWindowsInAddr(tpRankId_);
        CheckWindowSize(totalWinSizeTp_, tpWinContext_->winSize, tpipe_, XOut);
        winDataSizeOffsetTp_ =
            static_cast<uint64_t>(dataState_) * (tilingData->moeDistributeCombineV2Info.totalWinSizeTp / 2UL);
        tpStateOffsetOnWin_ = tpRankId_ * WIN_ADDR_ALIGN;
        tpRankWindow_.SetGlobalBuffer((__gm__ XType *)tpWindowGM_);
        tpRemoteSendCnt_ = tpSendCountGM_(1 - tpRankId_);
    }
    tpipe_->InitBuffer(moeQueue_, BUFFER_NUM, tokenMetaBytes_);
    flagRcvCount_ = axisK_ + sharedExpertNum_;
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::CommunInit(
    const MoeDistributeCombineV2TilingData *tilingData, GM_ADDR workspaceGM)
{
    // hccl初始化接口，hccl_.Init(tilingData);
    // 1. Server 拓扑
    serverRankSize_ = tilingData->moeDistributeCombineV2Info.serverRankSize;
    serverNum_ = epWorldSize_ / serverRankSize_;
    localRankId_ = epRankId_ % serverRankSize_;
    serverId_ = epRankId_ / serverRankSize_;
    localMoeExpertNum_ = moeExpertNum_ / epWorldSize_;
    // 2. Token
    tokenElemNum_ = axisH_;
    tokenDataBytes_ = axisH_ * sizeof(ExpandXType);
    tokenDataBytesAlign_ = RoundUp<uint32_t>(tokenDataBytes_, UB_ALIGN); // UB_ALIGN=32B
    metaBytesAlign_ = RoundUp<uint32_t>(TOKEN_META_BYTES, UB_ALIGN);
    tokenMetaBytes_ = tokenDataBytesAlign_ + metaBytesAlign_;
    flagU64CopyCntAlign_ = RoundUp<uint32_t>(FLAG_CNT_U64 * sizeof(uint64_t), UB_ALIGN) / sizeof(uint64_t); // 4
    maxLocalBs_ = globalBs_ / serverRankSize_; // maxLocalBs_ = tilingData->maxLocalBs;
    // token 数据需要的 480B block 数（向上取整）
    tokenDataBlockNum_ = DivCeil<uint32_t>(tokenDataBytes_, SPLIT_BLOCK_DATA_SIZE);
    // WinOut Data 区：一个 token 的 bytes
    packedTokenBytes_ = static_cast<uint64_t>(tokenDataBlockNum_ + 1) * WIN_ADDR_ALIGN; // WIN_ADDR_ALIGN=512B

    // win addr
    // 每个 toServer slice 总大小（bytes）：header(512B) + maxLocalBs_ * tokenStride
    winOutSliceBytes_ = WIN_ADDR_ALIGN + maxLocalBs_ * packedTokenBytes_;
    winOutTotalBytes_ = winOutSliceBytes_ * serverNum_; // 发送区
    winInTotalBytes_ = winOutTotalBytes_;               // 接收区
    windowOutGM_ = GetWindowsInAddr(epRankId_) + winDataSizeOffsetEp_;
    windowInGM_ = windowOutGM_ + winOutTotalBytes_;
    // 3. Share 区
    shareFlagSliceBytes_ = 512U;
    shareFlagTotalBytes_ = shareFlagSliceBytes_ * serverRankSize_; // share区的flag区
    shareDataSliceBytes_ = tokenMetaBytes_ * maxLocalBs_;
    shareDataTotalBytes_ = shareDataSliceBytes_ * serverRankSize_; // share区的data区

    for (int i = 0; i < MIN(serverRankSize_, MAX_SERVER_RANK_SIZE); i++) {
        // 一个Server内的全部RankId号，epRankId_为本Rank的Id号
        uint32_t rankIdServerInner = epRankId_ / serverRankSize_ * serverRankSize_ + i;
        serverShareAddr_[i] =
            reinterpret_cast<uint64_t>(winOutTotalBytes_ + winInTotalBytes_ + GetWindowsInAddr(rankIdServerInner));
    }
    // 4. token offset 表
    cntGM = workspaceGM + (aivNum_ * MASK_CALC_NEED_WORKSPACE);
    baseGM = cntGM + (aivNum_ * serverRankSize_ * sizeof(uint32_t));
    gmCoreTargetCnt_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(cntGM), aivNum_ * serverRankSize_);
    gmCoreTargetBase_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(baseGM), aivNum_ * serverRankSize_);
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::TokenMaskCalCnt()
{
    // 一维mask, 计算得到有效bs数量
    LocalTensor<bool> xActiveMaskTensor = xActMaskTBuf_.Get<bool>();
    LocalTensor<half> tempTensor = xActMaskCastTBuf_.Get<half>();
    LocalTensor<half> sumOutTensor = xActMaskSumTBuf_.Get<half>();
    DataCopyExtParams xActiveMaskParams{1U, static_cast<uint32_t>(axisBS_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> xActiveMaskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(xActiveMaskTensor, xActiveMaskGM_, xActiveMaskParams, xActiveMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> xActiveMaskInt8Tensor = xActiveMaskTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, xActiveMaskInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams params{1, axisBsAlignSize_, axisBS_};
    Sum(sumOutTensor, tempTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    activeMaskBsCnt_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::ExpertMaskCalCnt()
{
    // 二维mask, 挑选有效token
    uint64_t rsvdCnt = 0;
    uint32_t mask = axisBS_;
    LocalTensor<bool> maskStrideTensor = tokenBuf_.Get<bool>();
    LocalTensor<half> tempTensor = rowTmpFloatBuf_.Get<half>();
    LocalTensor<half> maskTempTensor = sumFloatBuf_.Get<half>();
    LocalTensor<uint8_t> maskTensor = tokenBuf_.Get<uint8_t>();
    LocalTensor<int32_t> bsIndexTensor = mulBuf_.Get<int32_t>();
    LocalTensor<uint32_t> maskTensorInt32 = tokenBuf_.Get<uint32_t>();
    DataCopyExtParams xActiveMaskParams{static_cast<uint16_t>(axisBS_), static_cast<uint32_t>(axisK_ * sizeof(bool)),
                                        0U, 0U, 0U};
    DataCopyPadExtParams<bool> xActiveMaskCopyPadParams{false, 0U, 0U, 0U};
    SumParams axisBsSumParams{
        1, static_cast<uint32_t>(Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half)), axisBS_};
    uint32_t calCnt = Ceil(axisBS_ * sizeof(half), ALIGNED_LEN) * ALIGNED_LEN / sizeof(half);

    Duplicate<half>(maskTempTensor, (half)0, calCnt);
    DataCopyPad(maskStrideTensor, xActiveMaskGM_, xActiveMaskParams, xActiveMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskStrideInt8Tensor = maskStrideTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, maskStrideInt8Tensor, RoundMode::CAST_NONE, activeMaskAlignSize_);
    PipeBarrier<PIPE_V>();
    uint32_t innerAlign = Ceil(axisK_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half) * BUFFER_NUM;
    SumParams axisKSumParams{axisBS_, innerAlign, axisK_};
    Sum(tokenTargetTensor_, tempTensor, axisKSumParams);
    PipeBarrier<PIPE_V>();
    Mins(maskTempTensor, tokenTargetTensor_, static_cast<half>(1), axisBS_);
    PipeBarrier<PIPE_V>();
    CompareScalar(maskTensor, maskTempTensor, static_cast<half>(1), AscendC::CMPMODE::EQ, calCnt);
    CreateVecIndex(bsIndexTensor, 0, axisBS_);
    PipeBarrier<PIPE_V>();
    GatherMask(validBsIndexTensor_, bsIndexTensor, maskTensorInt32, true, mask, {1, 1, 0, 0}, activeMaskBsCnt_);
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::InitInputAndOutput(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR expertScales,
    GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX, GM_ADDR constExpertAlpha1,
    GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR XOut)
{
    expandXGM_.SetGlobalBuffer((__gm__ ExpandXType *)expandX);
    expertIdsGM_.SetGlobalBuffer((__gm__ ExpandIdxType *)expertIds);
    expandIdxGM_.SetGlobalBuffer((__gm__ ExpandIdxType *)expandIdx);
    epSendCountGM_.SetGlobalBuffer((__gm__ int32_t *)epSendCount);
    expertScalesGM_.SetGlobalBuffer((__gm__ float *)expertScales);
    xActiveMaskGM_.SetGlobalBuffer((__gm__ bool *)xActiveMask);
    sharedExpertXGM_.SetGlobalBuffer((__gm__ XType *)sharedExpertX);
    elasticInfoGM_.SetGlobalBuffer((__gm__ int32_t *)elasticInfo);
    oriXGM_.SetGlobalBuffer((__gm__ ExpandXType *)oriX);
    constExpertAlpha1GM_.SetGlobalBuffer((__gm__ ExpandXType *)constExpertAlpha1);
    constExpertAlpha2GM_.SetGlobalBuffer((__gm__ ExpandXType *)constExpertAlpha2);
    constExpertVGM_.SetGlobalBuffer((__gm__ ExpandXType *)constExpertV);

    expandOutGlobal_.SetGlobalBuffer((__gm__ XType *)XOut);
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void
MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::InitElasticInfo(uint32_t &sharedExpertRankNum)
{
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(elasticInfoGM_);
    isScalingDownFlag_ = elasticInfoGM_.GetValue(0);
    if (isScalingDownFlag_) {
        epWorldSize_ = elasticInfoGM_.GetValue(EP_WORLD_SIZE_IDX);
        sharedExpertRankNum = elasticInfoGM_.GetValue(SHARE_RANK_NUM_IDX);
        uint32_t moeExpertNum = elasticInfoGM_.GetValue(MOE_NUM_IDX);
        epRankId_ = elasticInfoGM_.GetValue(ELASTIC_INFO_OFFSET + epRankId_);
        moeExpertPerRankNum_ = moeExpertNum / (epWorldSize_ - sharedExpertRankNum);
    }
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::InitTilingAttrs(
    const MoeDistributeCombineV2TilingData *tilingData)
{
    axisBS_ = tilingData->moeDistributeCombineV2Info.bs;
    axisH_ = tilingData->moeDistributeCombineV2Info.h;
    axisK_ = tilingData->moeDistributeCombineV2Info.k;
    aivNum_ = tilingData->moeDistributeCombineV2Info.aivNum;
    ubSize_ = tilingData->moeDistributeCombineV2Info.totalUbSize;
    globalBs_ = tilingData->moeDistributeCombineV2Info.globalBs;
    hasElasticInfoFlag_ = tilingData->moeDistributeCombineV2Info.hasElasticInfo;
    epWorldSizeOriginal_ = tilingData->moeDistributeCombineV2Info.epWorldSize;
    epRankId_ = tilingData->moeDistributeCombineV2Info.epRankId;
    epRankIdOriginal_ = tilingData->moeDistributeCombineV2Info.epRankId;
    epWorldSize_ = tilingData->moeDistributeCombineV2Info.epWorldSize;
    moeExpertPerRankNum_ = tilingData->moeDistributeCombineV2Info.moeExpertPerRankNum;
    tpWorldSize_ = tilingData->moeDistributeCombineV2Info.tpWorldSize;
    tpRankId_ = tilingData->moeDistributeCombineV2Info.tpRankId;
    totalWinSizeEp_ = tilingData->moeDistributeCombineV2Info.totalWinSizeEp;
    totalWinSizeTp_ = tilingData->moeDistributeCombineV2Info.totalWinSizeTp;
    isInputTokenMaskFlag_ = tilingData->moeDistributeCombineV2Info.isTokenMask;
    isInputExpertMaskFlag_ = tilingData->moeDistributeCombineV2Info.isExpertMask;
    hasSharedExpertX_ = tilingData->moeDistributeCombineV2Info.hasSharedExpertX;
    bufferNum_ = tilingData->moeDistributeCombineV2Info.bufferNum;
    zeroExpertNum_ = tilingData->moeDistributeCombineV2Info.zeroExpertNum;
    copyExpertNum_ = tilingData->moeDistributeCombineV2Info.copyExpertNum;
    constExpertNum_ = tilingData->moeDistributeCombineV2Info.constExpertNum;
    moeExpertNum_ = tilingData->moeDistributeCombineV2Info.moeExpertNum;
    enableSpecialExpert_ = (constExpertNum_ + zeroExpertNum_ + copyExpertNum_ > 0U);
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::InitAttrs(
    const MoeDistributeCombineV2TilingData *tilingData)
{
    InitTilingAttrs(tilingData);
    uint32_t sharedExpertRankNum = tilingData->moeDistributeCombineV2Info.sharedExpertRankNum;
    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    epWinContext_ = (__gm__ HcclOpResParam *)contextGM0;
    statusDataSpaceGm_ = (GM_ADDR)(epWinContext_->localWindowsExp);
    selfDataStatusGMTensor_.SetGlobalBuffer(
        (__gm__ uint32_t *)(statusDataSpaceGm_ + STATE_WIN_OFFSET + coreIdx_ * WIN_ADDR_ALIGN));
    TBuf<> dataStateBuf;
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);
    dataState_ = 0; // 标志位，标志0区还是1区
    if (hasElasticInfoFlag_) {
        InitElasticInfo(sharedExpertRankNum);
    }
    sharedExpertNum_ = tilingData->moeDistributeCombineV2Info.sharedExpertNum;
    moeSendNum_ = epWorldSize_ * moeExpertPerRankNum_; // 部署在所有Rank上的总专家数
    if (epRankId_ < sharedExpertRankNum) {
        isShareExpertRankFlag_ = true;
    }

    stateOffset_ = STATE_OFFSET;
    uint32_t hFloatSize = axisH_ * static_cast<uint32_t>(sizeof(float));
    hAlign32Size_ = Ceil(axisH_, UB_ALIGN) * UB_ALIGN;
    hFloatAlign32Size_ = Ceil(hFloatSize, UB_ALIGN) * UB_ALIGN;
    hFloatAlign256Size_ = Ceil(hFloatSize, ALIGNED_LEN) * ALIGNED_LEN;

    hExpandXTypeSize_ = axisH_ * sizeof(ExpandXType);
    hExpandXAlign32Size_ = Ceil(hExpandXTypeSize_, UB_ALIGN) * UB_ALIGN;

    hAlignWinSize_ = Ceil(hExpandXTypeSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    hAlignWinCnt_ = hAlignWinSize_ / sizeof(ExpandXType);
    bsKNum_ = axisBS_ * axisK_;
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::InitInt8Quant()
{
    scaleValFloat_ = static_cast<float>(1.0f / SCALE_PARAM);
    uint32_t scaleGranu = static_cast<uint32_t>(UB_ALIGN / sizeof(float)); // 计算每个block得到的reducemax结果数量
    scaleNum_ = (hExpandXAlign32Size_ / sizeof(ExpandXType)) / scaleGranu; // 得到有效scale的个数
    repeatNum_ = static_cast<uint32_t>(hFloatAlign256Size_ /
                                       ALIGNED_LEN); // BlockReduceMax 与 Brcb的重复迭代次数，每次256b参与计算
    mask_ = static_cast<uint32_t>(ALIGNED_LEN / sizeof(float));
    tokenScaleCnt_ = hAlign32Size_ / sizeof(ExpandXType) + scaleNum_; // int8_align + scale有效个数
}


template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::InitElasticInfoTensor()
{
    uint32_t elasticInfoSize =
        (ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSizeOriginal_) * static_cast<uint32_t>(sizeof(uint32_t));
    uint32_t elasticInfoSizeAlign = Ceil(elasticInfoSize, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(elasticInfoBuf_, elasticInfoSizeAlign);
    elasticInfoTensor_ = elasticInfoBuf_.Get<int32_t>();
    DataCopyExtParams elasticInfoParams = {
        1U, static_cast<uint32_t>((ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSizeOriginal_) * sizeof(int32_t)), 0U,
        0U, 0U};
    DataCopyPadExtParams<int32_t> elasticInfoCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(elasticInfoTensor_, elasticInfoGM_, elasticInfoParams, elasticInfoCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::BuffInit()
{
    tpipe_->Reset();
    tpipe_->InitBuffer(readStateBuf_, UB_ALIGN); // 32
    if constexpr (IsNeedReduceScatter) {
        tpipe_->InitBuffer(gmTpSendCountInQueue_, BUFFER_NUM, hExpandXAlign32Size_);  // 28K 存储输入拷过来的token
        tpipe_->InitBuffer(xOutQueue_, BUFFER_NUM, hExpandXAlign32Size_);             // 14K 存储输出token
        tpipe_->InitBuffer(winTpSendCountInQueue_, BUFFER_NUM, hExpandXAlign32Size_); // 14K * 2 存储对端win区token
        if constexpr (AscendC::IsSameType<XType, bfloat16_t>::value) {
            tpipe_->InitBuffer(winTpSendCountFloatBuf_, hFloatAlign32Size_); // 28K 参与量化及customAdd中token的v核运算
            tpipe_->InitBuffer(gmTpSendCountFloatBuf_, hFloatAlign32Size_);  // 28K 参与量化及customAdd中token的v核运算
            winTpSendCountFloatTensor_ = winTpSendCountFloatBuf_.Get<float>();
            gmTpSendCountFloatTensor_ = gmTpSendCountFloatBuf_.Get<float>();
        }
    } else {
        tpipe_->InitBuffer(gmTpSendCountQueue_, BUFFER_NUM, hExpandXAlign32Size_); // 28K 存储搬入token
        if constexpr (IsInt8Quant) {
            uint32_t tokenScaleAlign32Size = Ceil(tokenScaleCnt_ * sizeof(ExpandXType), UB_ALIGN) * UB_ALIGN;
            tpipe_->InitBuffer(xOutQueue_, BUFFER_NUM, tokenScaleAlign32Size); // 28K 输出token搬运
            tpipe_->InitBuffer(xAbsBuf_, hFloatAlign256Size_); // 28K blockReduceMax计算及后续Cast计算，256对齐
            uint32_t hFloatAlign256Cnt = hFloatAlign256Size_ / sizeof(float);
            tpipe_->InitBuffer(xMaxBuf_, (hFloatAlign256Cnt / REDUCE_NUM) * sizeof(float)); // 3.5K 存储ReduceMax结果
            tpipe_->InitBuffer(xScaleMulBuf_, hFloatAlign256Size_);                         // 28K 参与Brcb计算，256对齐
            tpipe_->InitBuffer(winTpSendCountFloatBuf_, hFloatAlign32Size_); // 28K 参与Div等token v核运算
            winTpSendCountFloatTensor_ = winTpSendCountFloatBuf_.Get<float>();
            absFloatTensor_ = xAbsBuf_.Get<float>();
            reduceMaxFloatTensor_ = xMaxBuf_.Get<float>();
            scaleDupLocalTensor_ = xScaleMulBuf_.Get<float>();
            fp16CastTensor_ = xAbsBuf_.Get<half>();
            Duplicate(absFloatTensor_, float(0), hFloatAlign256Cnt); // 统一写0
        }
        if (isScalingDownFlag_) {
            InitElasticInfoTensor();
        }
    }
    tpipe_->InitBuffer(indexCountsBuf_, sendCntNum_ * EXPAND_IDX_INFO * sizeof(int32_t));
    // A5 Server内/间通信 Buffer初始化
    uint32_t floatTokenBytes = axisH_ * sizeof(float);
    uint32_t floatTokenBytesAlign = RoundUp<uint32_t>(floatTokenBytes, UB_ALIGN);
    tpipe_->InitBuffer(sumBuf_, floatTokenBytesAlign * MAX_REDUCE_TILE_SIZE);
    tpipe_->InitBuffer(countBuf_, MAX_REDUCE_TILE_SIZE);
    tpipe_->InitBuffer(tempBuf_, BATCH_SIZE * tokenMetaBytes_);
    tpipe_->InitBuffer(outBuf_, tokenDataBytesAlign_);
    tpipe_->InitBuffer(localCntBuf_, UB_ALIGN);
    tpipe_->InitBuffer(localTokenIdBuf_, UB_ALIGN);
    tpipe_->InitBuffer(localOutTensorBuf_, tokenDataBytesAlign_);
    tpipe_->InitBuffer(localOutTempBuf_, packedTokenBytes_);
    tpipe_->InitBuffer(localFlagBuf_, UB_ALIGN);
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::MaskAlign()
{
    // 扩展后的二维mask通过GM对齐内轴元素个数
    uint32_t calcCnt = Ceil(axisBS_ * axisK_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    GlobalTensor<bool> MaskGMTensor;
    MaskGMTensor.SetGlobalBuffer((__gm__ bool *)maskCalcWorkspaceGM_);
    DataCopyExtParams maskCalcParams = {1U, static_cast<uint32_t>(calcCnt * sizeof(bool)), 0U, 0U, 0U};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(MaskGMTensor, maskGenerateTensor_, maskCalcParams);
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
    DataCopyExtParams xActiveMaskParams{static_cast<uint16_t>(axisBS_), static_cast<uint32_t>(axisK_ * sizeof(bool)),
                                        0U, 0U, 0U};
    DataCopyPadExtParams<bool> xActiveMaskCopyPadParams{true, 0U, static_cast<uint8_t>(UB_ALIGN - axisK_), 0U};
    DataCopyPad(maskStrideTensor_, MaskGMTensor, xActiveMaskParams, xActiveMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::GenerateActiveMask(half val)
{
    maskStrideTensor_ = tokenBuf_.Get<bool>();
    LocalTensor<half> maskCalcTensor = tokenBuf_.Get<half>();

    if (isInputTokenMaskFlag_) {
        // 根据一维场景下的activeMaskBsCnt_，构造出二维mask
        uint32_t calcCnt = Ceil(axisBS_ * axisK_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
        Duplicate<half>(maskCalcTensor, static_cast<half>(0), calcCnt);
        PipeBarrier<PIPE_V>();
        uint32_t activeCalcCnt =
            Ceil(activeMaskBsCnt_ * axisK_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
        Duplicate<half>(maskCalcTensor, static_cast<half>(1), activeCalcCnt);
        PipeBarrier<PIPE_V>();
        Cast(maskGenerateTensor_.ReinterpretCast<uint8_t>(), maskCalcTensor, RoundMode::CAST_NONE, calcCnt);
    } else {
        // 构造二维全true的mask
        uint32_t calcCnt = Ceil(axisBS_ * axisK_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
        Duplicate<half>(maskCalcTensor, val, calcCnt);
        PipeBarrier<PIPE_V>();
        Cast(maskGenerateTensor_.ReinterpretCast<uint8_t>(), maskCalcTensor, RoundMode::CAST_NONE, calcCnt);
    }
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::MaskSpecialExpert()
{
    LocalTensor<int32_t> expertIdsTensor_ = mulBuf_.Get<int32_t>();
    LocalTensor<float> expertIdsFloat = rowTmpFloatBuf_.Get<float>();
    LocalTensor<uint8_t> maskTensor = mulBuf_.Get<uint8_t>();
    LocalTensor<half> maskCalcTensor = tokenBuf_.Get<half>();
    LocalTensor<half> maskCalcSelectedTensor = rowTmpFloatBuf_.Get<half>();
    maskStrideTensor_ = tokenBuf_.Get<bool>();
    LocalTensor<half> tempTensor = rowTmpFloatBuf_.Get<half>();

    // 拷入expertIds
    uint32_t mask = axisBS_ * axisK_;
    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(mask * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor_, expertIdsGM_, expertIdsCntParams, expertIdsCntCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    // 根据expertId小于moeExpertNum，得到考虑特殊专家后的mask
    uint32_t calcCnt = Ceil(mask * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int32_t);
    Cast(expertIdsFloat, expertIdsTensor_, RoundMode::CAST_NONE, calcCnt);
    PipeBarrier<PIPE_V>();
    int32_t tmpMoeExpertNum = static_cast<int32_t>(moeExpertNum_);
    CompareScalar(maskTensor, expertIdsFloat, static_cast<float>(tmpMoeExpertNum), AscendC::CMPMODE::LT, calcCnt);
    PipeBarrier<PIPE_V>();
    if (isInputExpertMaskFlag_) {
        Cast(maskCalcTensor, expertMaskTensor_.ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE, calcCnt);
    } else {
        Cast(maskCalcTensor, maskGenerateTensor_.ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE, calcCnt);
    }
    PipeBarrier<PIPE_V>();
    Select(maskCalcSelectedTensor, maskTensor, maskCalcTensor, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
           calcCnt);
    PipeBarrier<PIPE_V>();
    Cast(maskGenerateTensor_.ReinterpretCast<uint8_t>(), maskCalcSelectedTensor, RoundMode::CAST_NONE, calcCnt);

    // 通过GM对齐内轴元素个数
    MaskAlign();

    // 更新考虑特殊专家后的
    uint32_t calCnt = Ceil(axisBS_ * sizeof(half), ALIGNED_LEN) * ALIGNED_LEN / sizeof(half);
    LocalTensor<int8_t> maskStrideInt8Tensor = maskStrideTensor_.ReinterpretCast<int8_t>();
    activeMaskAlignSize_ = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN);
    Cast(tempTensor, maskStrideInt8Tensor, RoundMode::CAST_NONE, activeMaskAlignSize_);
    PipeBarrier<PIPE_V>();
    uint32_t innerAlign = Ceil(axisK_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half) * BUFFER_NUM;
    SumParams axisKSumParams{axisBS_, innerAlign, axisK_};
    Sum(tokenTargetTensor_, tempTensor, axisKSumParams);
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::AlltoAllBuffInitAndMaskCal()
{
    tpipe_->Reset();
    activeMaskBsCnt_ = axisBS_;
    activeMaskAlignSize_ = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN);
    InitAlltoAllBuffers();
    if constexpr (IsInt8Quant) {
        scaleNumAlignSize_ = Ceil(scaleNum_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(xAbsBuf_, scaleNumAlignSize_);
        fp16CastTensor_ = mulBuf_.Get<half>();
        absFloatTensor_ = rowTmpFloatBuf_.Get<float>();
        scaleDupLocalTensor_ = mulBuf_.Get<float>();
        scaleDivFloatTensor_ = xAbsBuf_.Get<float>();
    }
    if (isInputTokenMaskFlag_) {
        axisBsAlignSize_ = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(xActMaskTBuf_, axisBsAlignSize_);
        tpipe_->InitBuffer(xActMaskCastTBuf_, axisBsAlignSize_ * sizeof(half));
        tpipe_->InitBuffer(xActMaskSumTBuf_, axisBsAlignSize_ * sizeof(half));
        TokenMaskCalCnt(); // 计算一维mask
    }
    if (isInputExpertMaskFlag_) {
        tpipe_->InitBuffer(tokenTargetTBuf_, Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN);
        tpipe_->InitBuffer(validBsIndexTBuf_, Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
        tpipe_->InitBuffer(expertMaskBuf_, Ceil(axisBS_ * axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN);
        tokenTargetTensor_ = tokenTargetTBuf_.Get<half>();
        validBsIndexTensor_ = validBsIndexTBuf_.Get<int32_t>();
        ExpertMaskCalCnt(); // 计算二维mask
        expertMaskTensor_ = expertMaskBuf_.Get<bool>();
        DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
        DataCopyExtParams maskParams{1U, static_cast<uint32_t>(axisBS_ * axisK_ * sizeof(bool)), 0U, 0U, 0U};
        DataCopyPad(expertMaskTensor_, xActiveMaskGM_, maskParams, maskCopyPadParams);
        SyncFunc<AscendC::HardEvent::V_S>();
    }
    if (enableSpecialExpert_) {
        maskGenerateTensor_ = sumFloatBuf_.Get<bool>();
        if (!isInputExpertMaskFlag_) {
            tpipe_->InitBuffer(tokenTargetTBuf_, Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN);
            tokenTargetTensor_ = tokenTargetTBuf_.Get<half>();
            GenerateActiveMask(static_cast<half>(1));
        }
        MaskSpecialExpert();
    }
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::InitAlltoAllBuffers()
{
    uint32_t maxSizeTokenBuf = hExpandXAlign32Size_;
    uint32_t maxSizeRowTmpFloatBuf = hFloatAlign32Size_;
    uint32_t bsKFloatAlign = Ceil(bsKNum_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint32_t mulBufSize = hFloatAlign256Size_ > bsKFloatAlign ? hFloatAlign256Size_ : bsKFloatAlign;
    if (isInputExpertMaskFlag_ || enableSpecialExpert_) {
        uint32_t activeMaskAlignHalfSize = activeMaskAlignSize_ * sizeof(half);
        maxSizeTokenBuf = (activeMaskAlignSize_ > hExpandXAlign32Size_ ? activeMaskAlignSize_ : hExpandXAlign32Size_);
        maxSizeRowTmpFloatBuf =
            (activeMaskAlignHalfSize > hFloatAlign32Size_ ? activeMaskAlignHalfSize : hFloatAlign32Size_);
    }
    tpipe_->InitBuffer(tokenBuf_, maxSizeTokenBuf);             // 16K 用于搬入输入token
    tpipe_->InitBuffer(rowTmpFloatBuf_, maxSizeRowTmpFloatBuf); // 32K 用于存储cast之后的fp32 token数据
    tpipe_->InitBuffer(mulBuf_, mulBufSize);              // 32K buffer复用， 最大用于存储Brcb之后的token，需要256对齐
    tpipe_->InitBuffer(sumFloatBuf_, hFloatAlign32Size_); // 32K add
    tpipe_->InitBuffer(moeSumQueue_, bufferNum_, hExpandXAlign32Size_); // 32K 搬入
    tpipe_->InitBuffer(stateBuf_, (flagRcvCount_)*STATE_OFFSET);
    tpipe_->InitBuffer(stateResetBuf_, (flagRcvCount_)*STATE_OFFSET); // 清理状态区
    tpipe_->InitBuffer(outBuf1_, (tokenDataBlockNum_ * SPLIT_BLOCK_SIZE)*BUFFER_NUM);
    stateResetTensor_ = stateResetBuf_.Get<float>();
    Duplicate<float>(stateResetTensor_, (float)0.0, static_cast<uint32_t>(flagRcvCount_ * FLOAT_PER_UB_ALIGN));
    SyncFunc<AscendC::HardEvent::V_MTE3>();
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::SplitCoreCal()
{
    // 对需要发送的token数平均分核，得到每个核上处理的卡的数量
    sendCntNum_ = selfSendCnt_ / aivNum_;
    uint32_t remainderRankNum = selfSendCnt_ % aivNum_;

    startTokenId_ = sendCntNum_ * coreIdx_;

    if (coreIdx_ < remainderRankNum) {
        sendCntNum_++;
        startTokenId_ += coreIdx_;
    } else {
        startTokenId_ += remainderRankNum;
    }
    endTokenId_ = startTokenId_ + sendCntNum_;
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::SplitCoreByServer()
{
    // 要处理的总量是 serverNum_（toServerId 的数量）
    if (serverNum_ == 0 || aivNum_ == 0) {
        sendServerNum_ = 0;
        startServerId_ = 0;
        endServerId_ = 0;
        return;
    }
    // 均匀分核
    sendServerNum_ = serverNum_ / aivNum_;
    uint32_t remainder = serverNum_ % aivNum_;
    startServerId_ = sendServerNum_ * coreIdx_;
    if (coreIdx_ < remainder) {
        sendServerNum_++;
        startServerId_ += coreIdx_;
    } else {
        startServerId_ += remainder;
    }
    endServerId_ = startServerId_ + sendServerNum_;
}


// 当前逻辑为tp=2场景，泛化待重新适配，本卡token在最前面
// 当tp为2时，直接把对端tp的数据分核处理发送
template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::ReduceScatterTrans()
{
    uint32_t tokenTpOffset = selfSendCnt_;
    uint32_t offset = selfSendCnt_ * axisH_;
    GlobalTensor<ExpandXType> dataCopyInGM = expandXGM_[offset];
    GM_ADDR rankGM = GetWindowsInAddr(1 - static_cast<int32_t>(tpRankId_));
    rankWindow_.SetGlobalBuffer((__gm__ XType *)rankGM);
    uint32_t tpSendCntNum = tpRemoteSendCnt_ / aivNum_;
    uint32_t remainderRankNum = tpRemoteSendCnt_ % aivNum_;
    uint32_t copyStartIdx = tpSendCntNum * coreIdx_;
    if (coreIdx_ < remainderRankNum) {
        tpSendCntNum++;
        copyStartIdx += coreIdx_;
    } else {
        copyStartIdx += remainderRankNum;
    }
    if (tpSendCntNum == 0U) {
        return;
    }
    uint32_t copyEndIdx = copyStartIdx + tpSendCntNum;

    LocalTensor<ExpandXType> tmpUb;

    // 确定rankid
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    for (uint32_t tokenNumIdx = copyStartIdx; tokenNumIdx < copyEndIdx; tokenNumIdx++) {
        tmpUb = moeQueue_.AllocTensor<ExpandXType>();
        DataCopyPad(tmpUb, dataCopyInGM[tokenNumIdx * axisH_], expandXCopyParams, copyPadExtParams);
        moeQueue_.EnQue(tmpUb);
        tmpUb = moeQueue_.DeQue<ExpandXType>();
        DataCopyPad(rankWindow_[tokenNumIdx * hAlignWinCnt_], tmpUb, expandXCopyParams);
        moeQueue_.FreeTensor<ExpandXType>(tmpUb);
    }
}

// 流水流程
// 46 -> gm -> ub syncall win->gm add -> alltoall
// 2 -> win wait syncall gm -> ub win ->gm add -> alltoall
template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::SetWaitTpStatus()
{
    PipeBarrier<PIPE_ALL>();
    if ((coreIdx_ >= tpRemoteSendCnt_) && (coreIdx_ >= selfSendCnt_)) {
        return;
    }

    uint32_t tpToRankId = 1U - tpRankId_; // 当前适配按tpWorldSize_==2来写
    PipeBarrier<PIPE_ALL>();
    LocalTensor<int32_t> statusFlagUb = readStateBuf_.Get<int32_t>();
    statusFlagUb(0) = 1;
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    GlobalTensor<int32_t> tpStatusWinTensor;
    stateGM_ =
        GetWindowsInAddr(tpToRankId) + coreIdx_ * WIN_ADDR_ALIGN; // GetWinStateAddrByRankId(tpToRankId, TP_DOMAIN)
    tpStatusWinTensor.SetGlobalBuffer((__gm__ int32_t *)stateGM_);
    DataCopy<int32_t>(tpStatusWinTensor, statusFlagUb, 8UL); // 8是数据大小，按32对齐拷贝
    SyncFunc<AscendC::HardEvent::MTE3_S>();

    GM_ADDR tpStatusWin =
        GetWindowsInAddr(tpRankId_) + coreIdx_ * WIN_ADDR_ALIGN; // GetWinStateAddrByRankId(tpRankId_, TP_DOMAIN)
    GlobalTensor<int32_t> selfStatusWinTensor;
    selfStatusWinTensor.SetGlobalBuffer((__gm__ int32_t *)tpStatusWin);
    int32_t sumOfFlag = 0;
    while (sumOfFlag != 1) {
        DataCopy<int32_t>(statusFlagUb, selfStatusWinTensor, 8);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        sumOfFlag = statusFlagUb.GetValue(0);
        SyncFunc<AscendC::HardEvent::S_MTE2>();
    }
    selfStatusWinTensor(0) = 0;
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfStatusWinTensor);
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::PrepareServerShareLayout(
    LocalTensor<ExpandIdxType> expandIdxLocal)
{
    // Step1：统计本核 -> 每个 targetLocalRank 的 token 数。写 gmCoreTargetCnt_[coreIdx_][t]
    CalcLocalTargetCnt(expandIdxLocal);
    AscendC::SyncAll<true>(); // 等所有核写完 cnt
    // Step2core0 计算 prefix base
    BuildPrefixBaseOnCore0();
    AscendC::SyncAll<true>(); // 等 base 计算完成
    // Step3：每核读回自己的 base，并清空 run
    LoadLocalBaseFromGm();
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::CalcLocalTargetCnt(
    LocalTensor<ExpandIdxType> expandIdxLocal)
{
    // localTargetCnt_ : uint32_t[serverRankSize_]
    tpipe_->InitBuffer(localTargetCntBuf_, serverRankSize_ * sizeof(uint32_t));
    localTargetCnt_ = localTargetCntBuf_.Get<uint32_t>();
    // 清零
    for (uint32_t t = 0; t < serverRankSize_; ++t) {
        localTargetCnt_.SetValue(t, 0);
    }
    //  必须和 AlltoAllDispatch 的 token 遍历顺序一致
    for (uint32_t loop = 0; loop < sendCntNum_; ++loop) {
        uint32_t tkIndex = startTokenId_ + ((loop + epRankId_) % sendCntNum_);
        uint32_t baseOffset = (tkIndex - startTokenId_) * EXPAND_IDX_INFO;
        uint32_t originRankId = static_cast<uint32_t>(expandIdxLocal(baseOffset));
        if (isScalingDownFlag_) {
            originRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + originRankId);
        }
        uint32_t targetLocalRank = originRankId % serverRankSize_;
        uint32_t cnt = localTargetCnt_.GetValue(targetLocalRank) + 1;
        localTargetCnt_.SetValue(targetLocalRank, cnt);
    }
    // 写 gmCoreTargetCnt_[coreIdx_][t]。gmCoreTargetCnt_ 逻辑形状：[aivNum_][serverRankSize_]
    uint32_t base = coreIdx_ * serverRankSize_;
    for (uint32_t t = 0; t < serverRankSize_; ++t) {
        uint32_t localCnt = localTargetCnt_.GetValue(t);
        gmCoreTargetCnt_.SetValue(base + t, localCnt);
    }
    // 确保写出到 GM
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        gmCoreTargetCnt_);
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::BuildPrefixBaseOnCore0()
{
    if (coreIdx_ != 0) {
        return;
    }
    tpipe_->InitBuffer(rankTotalCntBuf_, serverRankSize_ * sizeof(uint32_t));
    rankTotalCntToTargetLocal_ = rankTotalCntBuf_.Get<uint32_t>();
    // 1. rankTotalCntToTargetLocal_ = 0
    Duplicate(rankTotalCntToTargetLocal_, 0U, serverRankSize_);

    // 2. reduce：Σ gmCoreTargetCnt_[c][t]
    for (uint32_t c = 0; c < aivNum_; ++c) {
        uint32_t base = c * serverRankSize_;
        for (uint32_t t = 0; t < serverRankSize_; ++t) {
            uint32_t v = gmCoreTargetCnt_.GetValue(base + t);
            rankTotalCntToTargetLocal_.SetValue(t, rankTotalCntToTargetLocal_.GetValue(t) + v);
        }
    }
    // 3. 对每个 targetLocalRank 单独做 prefix-sum
    for (uint32_t t = 0; t < serverRankSize_; ++t) {
        uint32_t prefix = 0;
        for (uint32_t c = 0; c < aivNum_; ++c) {
            uint32_t idx = c * serverRankSize_ + t;
            uint32_t cnt = gmCoreTargetCnt_.GetValue(idx);
            gmCoreTargetBase_.SetValue(idx, prefix);
            prefix += cnt;
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        gmCoreTargetBase_);
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::LoadLocalBaseFromGm()
{
    tpipe_->InitBuffer(localTargetBaseBuf_, serverRankSize_ * sizeof(uint32_t));
    tpipe_->InitBuffer(localTargetRunBuf_, serverRankSize_ * sizeof(uint32_t));
    localTargetBase_ = localTargetBaseBuf_.Get<uint32_t>();
    localTargetRun_ = localTargetRunBuf_.Get<uint32_t>();
    uint32_t base = coreIdx_ * serverRankSize_;
    for (uint32_t t = 0; t < serverRankSize_; ++t) {
        localTargetBase_.SetValue(t, gmCoreTargetBase_.GetValue(base + t));
        localTargetRun_.SetValue(t, 0); // 运行时 offset，从 0 开始
    }
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::AlltoAllDispatch()
{
    //===========================Server内通信==========================//
    if (sendCntNum_ == 0U || serverRankSize_ == 0) { // 空闲核，直接返回
        return;
    }
    LocalTensor<ExpandIdxType> expandIdxLocal = indexCountsBuf_.Get<ExpandIdxType>();
    const DataCopyExtParams bskParams{1U, static_cast<uint32_t>(sendCntNum_ * EXPAND_IDX_INFO * sizeof(uint32_t)), 0U,
                                      0U, 0U};
    const DataCopyPadExtParams<ExpandIdxType> copyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expandIdxLocal, expandIdxGM_[startTokenId_ * EXPAND_IDX_INFO], bskParams, copyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    // 1、计算要发送的token的数量与偏移 gmCoreTargetCnt_ / gmCoreTargetBase_
    PrepareServerShareLayout(expandIdxLocal);

    // 2、Server内AlltoAll发送 token：发往ShareData[targetLocalRank][fromLocalRank][tokenOffset]
    DispatchTokensToShareMem(expandIdxLocal);

    AscendC::SyncAll<true>(); // 等所有核把 ShareData 写完

    // 3、一个核负责写 ShareFlag
    UpdateShareFlag();

    SyncFunc<AscendC::HardEvent::MTE3_S>();
    AlltoAllBuffInitAndMaskCal(); // 暂时不修改
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::DispatchTokensToShareMem(
    LocalTensor<ExpandIdxType> expandIdxLocal)
{
    for (uint32_t loop = 0; loop < sendCntNum_; loop++) {
        const uint32_t tkIndex = startTokenId_ + ((loop + epRankId_) % sendCntNum_); // 错位发送
        const uint32_t baseOffset = (tkIndex - startTokenId_) * EXPAND_IDX_INFO;
        uint32_t originRankId = static_cast<uint32_t>(expandIdxLocal(baseOffset));      // 位置0是rank_id
        uint32_t originTokenId = static_cast<uint32_t>(expandIdxLocal(baseOffset + 1)); // 位置1是token_id

        if (isScalingDownFlag_) {
            originRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + originRankId);
        }

        uint32_t targetLocalRank = originRankId % serverRankSize_;
        const uint32_t runVal = localTargetRun_.GetValue(targetLocalRank);
        const uint32_t tokenOffset = localTargetBase_.GetValue(targetLocalRank) + runVal;
        localTargetRun_.SetValue(targetLocalRank, runVal + 1U);

        const uint64_t targetShareBase = serverShareAddr_[targetLocalRank];
        const uint64_t shareDataAddr = targetShareBase + static_cast<uint64_t>(shareFlagTotalBytes_) +
                                       static_cast<uint64_t>(localRankId_) * shareDataSliceBytes_ +
                                       static_cast<uint64_t>(tokenOffset) * tokenMetaBytes_;
        DispatchTokenInner(tkIndex, originRankId, originTokenId, shareDataAddr);
    }
}


template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::DispatchTokenInner(
    uint32_t tkIndex, uint32_t originRankId, uint32_t originTokenId, uint64_t shareDataAddr)
{
    LocalTensor<uint8_t> payloadUb = outBuf_.Get<uint8_t>();
    LocalTensor<ExpandXType> tmpUb = tempBuf_.Get<ExpandXType>();

    DataCopyExtParams inputCopyParams{1U, static_cast<uint32_t>(tokenDataBytes_), 0U, 0U, 0U};
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};

    if constexpr (IsNeedReduceScatter) {
        DataCopyPad(tmpUb, expandXGM_[tkIndex * axisH_], inputCopyParams, copyPadExtParams);
        LocalTensor<ExpandXType> winInputUb = localOutTempBuf_.Get<ExpandXType>();
        uint32_t tokenWinOffset = tkIndex * hAlignWinCnt_;
        DataCopyPad(winInputUb, tpRankWindow_[tokenWinOffset], inputCopyParams, copyPadExtParams);
        PipeBarrier<PIPE_MTE2>();
        LocalTensor<ExpandXType> outTokenUb = payloadUb.template ReinterpretCast<ExpandXType>();
        CustomAdd(outTokenUb, tmpUb, winInputUb);
        PipeBarrier<PIPE_V>();
    } else if constexpr (IsInt8Quant) {
        DataCopyPad(tmpUb, expandXGM_[tkIndex * axisH_], inputCopyParams, copyPadExtParams);
        gmTpSendCountTensor_ = tmpUb;
        sendLocalTensor_ = payloadUb.template ReinterpretCast<ExpandXType>();
        Int8QuantProcess();
    } else {
        LocalTensor<ExpandXType> tokenUb = payloadUb.template ReinterpretCast<ExpandXType>();
        DataCopyPad(tokenUb, expandXGM_[tkIndex * axisH_], inputCopyParams, copyPadExtParams);
        PipeBarrier<PIPE_MTE2>();
    }

    float scaleVal = expertScalesLocal_.GetValue(tkIndex);
    LocalTensor<float> metaFP32 = (payloadUb[tokenDataBytesAlign_]).template ReinterpretCast<float>();
    metaFP32.SetValue(0, scaleVal);
    LocalTensor<uint32_t> metaU32 = (payloadUb[tokenDataBytesAlign_ + 4U]).template ReinterpretCast<uint32_t>();
    metaU32.SetValue(0, originRankId);
    metaU32.SetValue(1, originTokenId);
    // 写入 ShareData
    GlobalTensor<uint8_t> shareDataByteGm;
    shareDataByteGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(shareDataAddr));
    DataCopy(shareDataByteGm, payloadUb,
             static_cast<uint32_t>(tokenMetaBytes_)); // tokenMetaBytes_为uint8，字节数==元素数
    PipeBarrier<PIPE_MTE3>();
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::UpdateShareFlag()
{
    if (coreIdx_ == 0) {
        LocalTensor<uint64_t> flagUb = stateBuf_.Get<uint64_t>();
        for (uint32_t targetLocalRank = 0; targetLocalRank < serverRankSize_; ++targetLocalRank) {
            uint64_t cnt = static_cast<uint64_t>(
                rankTotalCntToTargetLocal_.GetValue(targetLocalRank)); // 这是汇总后的 cnt，不是 localTargetCnt_
            flagUb.SetValue(0, READY_FLAG);
            flagUb.SetValue(1, cnt);
            uint64_t flagAddr =
                serverShareAddr_[targetLocalRank] + static_cast<uint64_t>(localRankId_ * shareFlagSliceBytes_);
            GlobalTensor<uint64_t> flagGm;
            flagGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(flagAddr));
            // 写 2 × uint64（flag + cnt）
            DataCopy(flagGm, flagUb, flagU64CopyCntAlign_);
        }
    }
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::SumToWindow()
{
    //===========================Server内加权求和==========================//
    // Step1. 等待本卡 ShareFlag[fromLocalRank][*] 全部 READY
    if (coreIdx_ == 0) {
        uint64_t shareFlagAddr = serverShareAddr_[localRankId_];
        LocalTensor<uint64_t> flagUb = stateBuf_.Get<uint64_t>();
        for (uint32_t RankId = 0; RankId < serverRankSize_; ++RankId) {
            uint64_t flagOffset = static_cast<uint64_t>(RankId) * 512UL;
            GlobalTensor<uint64_t> flagGm;
            flagGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(shareFlagAddr + flagOffset));
            while (true) {
                DataCopy(flagUb, flagGm, flagU64CopyCntAlign_);
                SyncFunc<AscendC::HardEvent::MTE2_S>();
                if (flagUb.GetValue(0) == READY_FLAG) {
                    break;
                }
            }
        }
    }
    AscendC::SyncAll<true>();
    // Step2. ShareData → WinOut（按 toServerId 分核）
    if (sendServerNum_ == 0) {
        return;
    }
    LocalTensor<float> sumTileUb = sumBuf_.Get<float>();
    LocalTensor<uint32_t> existFlagUb = countBuf_.Get<uint32_t>();
    for (uint32_t toServerId = startServerId_; toServerId < endServerId_; ++toServerId) {
        ProcessOneServer(toServerId, sumTileUb, existFlagUb);
    }
    AscendC::SyncAll<true>();
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::ProcessOneServer(
    uint32_t toServerId, LocalTensor<float> sumTileUb, LocalTensor<uint32_t> existFlagUb)
{
    GM_ADDR winOutSliceBase = windowOutGM_ + (toServerId * winOutSliceBytes_);
    GlobalTensor<uint64_t> winOutHeaderGm;
    winOutHeaderGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(winOutSliceBase));
    GM_ADDR winOutDataBase = winOutSliceBase + WIN_ADDR_ALIGN;
    uint64_t currentWinDataOffset = 0;
    uint32_t currentWinTokenCnt = 0;
    for (uint32_t baseId = 0; baseId < maxLocalBs_; baseId += MAX_REDUCE_TILE_SIZE) {
        uint32_t endId = baseId + MAX_REDUCE_TILE_SIZE;
        if (endId > maxLocalBs_)
            endId = maxLocalBs_;
        uint32_t currentTileLen = endId - baseId;
        // 1. 清零
        Duplicate(sumTileUb, 0.0f, currentTileLen * axisH_);
        Duplicate(existFlagUb, static_cast<uint32_t>(0), currentTileLen);
        // 2. 累加所有的Rank
        for (uint32_t fromRank = 0; fromRank < serverRankSize_; ++fromRank) {
            AccumulateRankDataToUb(fromRank, toServerId, baseId, endId);
        }
        // 3. 写入 WinOut
        for (uint32_t i = 0; i < currentTileLen; ++i) {
            if (existFlagUb.GetValue(i) == 0)
                continue;
            uint32_t realTokenId = baseId + i;
            GlobalTensor<ExpandXType> dstWinTokenGm;
            dstWinTokenGm.SetGlobalBuffer(
                reinterpret_cast<__gm__ ExpandXType *>(winOutDataBase + currentWinDataOffset));
            LocalTensor<float> tokenSumSlice = sumTileUb[i * axisH_];
            TokenToWinOut(dstWinTokenGm, realTokenId, toServerId, tokenSumSlice);
            currentWinTokenCnt++;
            currentWinDataOffset += packedTokenBytes_;
        }
    }
    WriteWinOutHeader(winOutHeaderGm, currentWinTokenCnt);
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::AccumulateRankDataToUb(
    uint32_t fromLocalRank, uint32_t targetServerId, uint32_t baseId, uint32_t endId)
{
    uint32_t cnt = 0;
    GM_ADDR shareBase = reinterpret_cast<__gm__ uint8_t *>(serverShareAddr_[localRankId_]);
    ReadRankTokenCnt(fromLocalRank, cnt, shareBase);
    if (cnt == 0U) return;
    GM_ADDR shareDataBase = shareBase + static_cast<uint64_t>(shareFlagTotalBytes_ + (fromLocalRank * shareDataSliceBytes_));
    LocalTensor<float> sumTileUb = sumBuf_.Get<float>();
    LocalTensor<uint8_t> existFlagUb = countBuf_.Get<uint8_t>();
    LocalTensor<float> tmpUb = localOutTempBuf_.Get<float>();
    LocalTensor<ExpandXType> inputBatchUb = tempBuf_.Get<ExpandXType>();
    uint32_t elemsPerPack = tokenMetaBytes_ / sizeof(ExpandXType);
    uint32_t metaOffsetElems = tokenDataBytesAlign_ / sizeof(ExpandXType);
    uint32_t processed = 0;
    while (processed < cnt) {
        uint32_t curBatch = (cnt - processed) > BATCH_SIZE ? BATCH_SIZE : (cnt - processed);
        GlobalTensor<ExpandXType> batchGm;
        batchGm.SetGlobalBuffer(reinterpret_cast<__gm__ ExpandXType *>(shareDataBase + static_cast<uint64_t>(processed * tokenMetaBytes_)));
        uint32_t copyLen = (curBatch * tokenMetaBytes_) / sizeof(ExpandXType); // tokenMetaBytes_是32B对齐的
        DataCopy(inputBatchUb, batchGm, copyLen);
        PipeBarrier<PIPE_MTE2>();
        for (uint32_t k = 0; k < curBatch; k++) {
            uint32_t packBase = k * elemsPerPack;
            LocalTensor<float> metaF = inputBatchUb[packBase + metaOffsetElems].template ReinterpretCast<float>();
            float scaleVal = metaF.GetValue(0);
            LocalTensor<uint32_t> metaU32 = inputBatchUb[packBase + metaOffsetElems].template ReinterpretCast<uint32_t>();
            uint32_t originRankId = metaU32.GetValue(1);
            uint32_t originTokenId = metaU32.GetValue(2);
            if ((originRankId / serverRankSize_) != targetServerId || originTokenId < baseId ||
                originTokenId >= endId) {
                continue;
            }
            uint32_t offsetInTile = originTokenId - baseId;
            existFlagUb.SetValue(offsetInTile, 1);
            LocalTensor<ExpandXType> srcToken = inputBatchUb[packBase];
            LocalTensor<float> dstSum = sumTileUb[offsetInTile * axisH_];
            Cast(tmpUb, srcToken, AscendC::RoundMode::CAST_NONE, axisH_);
            PipeBarrier<PIPE_V>();
            Muls(tmpUb, tmpUb, scaleVal, axisH_);
            PipeBarrier<PIPE_V>();
            Add(dstSum, dstSum, tmpUb, axisH_);
            PipeBarrier<PIPE_V>();
        }
        processed += curBatch;
    }
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void
MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::ReadRankTokenCnt(uint32_t fromLocalRank,
                                                                              uint32_t &tokenCnt, GM_ADDR shareBase)
{
    GM_ADDR flagAddr = shareBase + static_cast<uint64_t>(fromLocalRank) * shareFlagSliceBytes_;
    GlobalTensor<uint64_t> flagGm;
    flagGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(flagAddr));
    LocalTensor<uint64_t> flagUb = stateBuf_.Get<uint64_t>(flagU64CopyCntAlign_);
    DataCopy(flagUb, flagGm, flagU64CopyCntAlign_);
    PipeBarrier<PIPE_ALL>();
    tokenCnt = static_cast<uint32_t>(flagUb.GetValue(1));
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void
MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::TokenToWinOut(GlobalTensor<ExpandXType> dstWinGMTensor,
                                                                           uint32_t originTokenId, uint32_t toServerId,
                                                                           LocalTensor<float> srcSumTensor)
{
    //  从 sumFloatLocal_ 生成 token 的 UB 表示（ExpandXType[axisH_]）
    LocalTensor<ExpandXType> outUb = outBuf_.Get<ExpandXType>();
    LocalTensor<ExpandXType> outTensor = outBuf1_.Get<ExpandXType>();
    Cast(outUb, srcSumTensor, AscendC::RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> outUbBytes = outUb.template ReinterpretCast<uint8_t>();
    LocalTensor<uint32_t> metaU32 = outUbBytes[tokenDataBytesAlign_].template ReinterpretCast<uint32_t>();
    metaU32.SetValue(0, originTokenId);
    SyncFunc<AscendC::HardEvent::S_V>();
    const uint32_t dstRepeatSize = SPLIT_BLOCK_SIZE / BLOCK_SIZE;      // 16
    const uint32_t srcRepeatSize = SPLIT_BLOCK_DATA_SIZE / BLOCK_SIZE; // 15
    uint32_t flagPadOffsetElems = flagPadOffset_ / sizeof(ExpandXType);
    uint32_t copyElemsFirst = BLOCK_COPY_BYTES_FIRST / sizeof(ExpandXType);
    uint32_t copyElemsSecond = BLOCK_COPY_BYTES_SECOND / sizeof(ExpandXType);
    Copy(outTensor[flagPadOffsetElems], outUb, uint64_t(copyElemsFirst), uint8_t(tokenDataBlockNum_),
         {1, 1, dstRepeatSize, srcRepeatSize});
    Copy(outTensor[flagPadOffsetElems + 128], outUb[copyElemsFirst], uint64_t(copyElemsSecond), uint8_t(tokenDataBlockNum_),
         {1, 1, dstRepeatSize, srcRepeatSize});
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(dstWinGMTensor, outTensor[flagPadOffset_ / sizeof(ExpandXType)], packedTokenBytes_ / sizeof(ExpandXType));
    flagPadOffset_ = (tokenDataBlockNum_ * SPLIT_BLOCK_SIZE) - flagPadOffset_;
}


template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void
MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::WriteWinOutHeader(GlobalTensor<uint64_t> headerGm,
                                                                               uint32_t winTokenCnt)
{
    LocalTensor<uint64_t> headerUb = stateBuf_.Get<uint64_t>(flagU64CopyCntAlign_);
    headerUb.SetValue(0, READY_FLAG);
    headerUb.SetValue(1, winTokenCnt);
    DataCopy(headerGm, headerUb, flagU64CopyCntAlign_);
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::Int8QuantProcess()
{
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    castLocalTensor_ = sendLocalTensor_.template ReinterpretCast<int8_t>();              // 长度为int8H_Align + scaleNum
    scaleDivTensor_ = castLocalTensor_[hAlign32Size_].template ReinterpretCast<XType>(); // 偏移前面的int8

    Cast(winTpSendCountFloatTensor_, gmTpSendCountTensor_, RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    Abs(absFloatTensor_, winTpSendCountFloatTensor_,
        axisH_); // absFloatTensor_ align到256并写0，支持ReduceMax与Brcb
    PipeBarrier<PIPE_V>();
    BlockReduceMax(reduceMaxFloatTensor_, absFloatTensor_, repeatNum_, mask_, 1, 1, BLOCK_NUM); // 32->1 256->8
    PipeBarrier<PIPE_V>();
    Muls(reduceMaxFloatTensor_, reduceMaxFloatTensor_, scaleValFloat_, scaleNum_); // 有效个数
    PipeBarrier<PIPE_V>();
    Cast(scaleDivTensor_, reduceMaxFloatTensor_, RoundMode::CAST_RINT, scaleNum_); // 有效个数
    PipeBarrier<PIPE_V>();
    Brcb(scaleDupLocalTensor_, reduceMaxFloatTensor_, repeatNum_, {1, BLOCK_NUM}); // 一次256
    PipeBarrier<PIPE_V>();
    Div(winTpSendCountFloatTensor_, winTpSendCountFloatTensor_, scaleDupLocalTensor_, axisH_); // 有效个数
    PipeBarrier<PIPE_V>();
    Cast(fp16CastTensor_, winTpSendCountFloatTensor_, RoundMode::CAST_RINT, axisH_);
    PipeBarrier<PIPE_V>();
    Cast(castLocalTensor_, fp16CastTensor_, RoundMode::CAST_RINT, axisH_);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::CustomAdd(LocalTensor<XType> &dst,
                                                                                              LocalTensor<XType> &src0,
                                                                                              LocalTensor<XType> &src1)
{
    if constexpr (AscendC::IsSameType<XType, bfloat16_t>::value) {
        Cast(winTpSendCountFloatTensor_, src0, RoundMode::CAST_NONE, axisH_);
        Cast(gmTpSendCountFloatTensor_, src1, RoundMode::CAST_NONE, axisH_);
        PipeBarrier<PIPE_V>();
        Add(winTpSendCountFloatTensor_, winTpSendCountFloatTensor_, gmTpSendCountFloatTensor_, axisH_);
        PipeBarrier<PIPE_V>();
        Cast(dst, winTpSendCountFloatTensor_, RoundMode::CAST_RINT, axisH_);
    } else {
        Add(dst, src0, src1, axisH_);
    }
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void
MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::Int8DequantProcess(LocalTensor<XType> &src)
{
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    castLocalTensor_ = src.template ReinterpretCast<int8_t>();
    scaleDivTensor_ = src[hAlign32Size_ / 2];
    SyncFunc<AscendC::HardEvent::S_V>();
    Cast(scaleDivFloatTensor_, scaleDivTensor_, RoundMode::CAST_NONE, scaleNum_);
    Cast(fp16CastTensor_, castLocalTensor_, RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    Cast(absFloatTensor_, fp16CastTensor_, RoundMode::CAST_NONE, axisH_);
    Brcb(scaleDupLocalTensor_, scaleDivFloatTensor_, repeatNum_, {1, BLOCK_NUM});
    PipeBarrier<PIPE_V>();
    Mul(absFloatTensor_, absFloatTensor_, scaleDupLocalTensor_, axisH_);
    PipeBarrier<PIPE_V>();
    Cast(src, absFloatTensor_, RoundMode::CAST_RINT, axisH_);
    PipeBarrier<PIPE_V>();
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::CalConstExpertAlpha(
    GlobalTensor<ExpandXType> constExpertAlphaGM, uint32_t const_expert_idx, float &alphaFloat)
{
    LocalTensor<ExpandXType> weightLocal = moeSumQueue_.AllocTensor<ExpandXType>();
    LocalTensor<float> weightFloatLocal = mulBuf_.Get<float>();
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};

    // 使用moeSumQueue_分配缓冲区来存储alpha1对应的权重矩阵Wc
    DataCopyPad(weightLocal, constExpertAlphaGM[const_expert_idx * axisH_], expandXCopyParams, copyPadExtParams);
    moeSumQueue_.EnQue(weightLocal);
    weightLocal = moeSumQueue_.DeQue<ExpandXType>();
    Cast(weightFloatLocal, weightLocal, AscendC::RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();

    // 计算Wc * x
    Mul(weightFloatLocal, weightFloatLocal, rowTmpFloatLocal_, axisH_);
    PipeBarrier<PIPE_V>();
    uint32_t innerAlign = Ceil(axisH_ * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
    SumParams params{1, innerAlign, axisH_};
    Sum(weightFloatLocal, weightFloatLocal, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    alphaFloat = weightFloatLocal.GetValue(0);
    moeSumQueue_.FreeTensor<ExpandXType>(weightLocal);
}

// 处理常量专家
template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::ProcessConstantExpert(
    uint32_t tokenIndex, uint32_t const_expert_idx, float scaleVal)
{
    PipeBarrier<PIPE_ALL>();
    LocalTensor<ExpandXType> rowTmpLocal = tokenBuf_.Get<ExpandXType>();
    LocalTensor<float> alphaFloatLocal = tokenBuf_.Get<float>();
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    float alpha1Float = static_cast<float>(0.0);
    float alpha2Float = static_cast<float>(0.0);

    // 读取输入token
    DataCopyPad(rowTmpLocal, oriXGM_[tokenIndex * axisH_], expandXCopyParams, copyPadExtParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    Cast(rowTmpFloatLocal_, rowTmpLocal, AscendC::RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();

    // 计算Wc * x
    CalConstExpertAlpha(constExpertAlpha1GM_, const_expert_idx, alpha1Float);
    CalConstExpertAlpha(constExpertAlpha2GM_, const_expert_idx, alpha2Float);

    // 计算softmax(Wc * x)
    float maxAlphaFloat = (alpha1Float > alpha2Float) ? alpha1Float : alpha2Float;
    alphaFloatLocal.SetValue(0, alpha1Float - maxAlphaFloat);
    alphaFloatLocal.SetValue(1, alpha2Float - maxAlphaFloat);
    SyncFunc<AscendC::HardEvent::S_V>();
    Exp(alphaFloatLocal, alphaFloatLocal, 2);
    SyncFunc<AscendC::HardEvent::V_S>();
    float alphaSumFloat = alphaFloatLocal.GetValue(0) + alphaFloatLocal.GetValue(1);
    alpha1Float = alphaFloatLocal.GetValue(0) / alphaSumFloat;
    alpha2Float = alphaFloatLocal.GetValue(1) / alphaSumFloat;

    // 使用moeSumQueue_分配缓冲区来存储常量专家向量v
    LocalTensor<float> constVFloatLocal = mulBuf_.Get<float>();
    LocalTensor<ExpandXType> constUb = moeSumQueue_.AllocTensor<ExpandXType>();
    DataCopyPad(constUb, constExpertVGM_[const_expert_idx * axisH_], expandXCopyParams, copyPadExtParams);
    moeSumQueue_.EnQue(constUb);
    constUb = moeSumQueue_.DeQue<ExpandXType>();

    Cast(constVFloatLocal, constUb, AscendC::RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    moeSumQueue_.FreeTensor<ExpandXType>(constUb);

    // 计算 alpha1 * x + alpha2 * v
    SyncFunc<AscendC::HardEvent::S_V>();
    Muls(rowTmpFloatLocal_, rowTmpFloatLocal_, alpha1Float, axisH_);
    Muls(constVFloatLocal, constVFloatLocal, alpha2Float, axisH_);
    PipeBarrier<PIPE_V>();
    Add(rowTmpFloatLocal_, rowTmpFloatLocal_, constVFloatLocal, axisH_);
    PipeBarrier<PIPE_V>();

    // 乘以专家权重
    Muls(mulBufLocal_, rowTmpFloatLocal_, scaleVal, axisH_);
    PipeBarrier<PIPE_V>();
    Add(sumFloatBufLocal_, sumFloatBufLocal_, mulBufLocal_, axisH_);
    PipeBarrier<PIPE_V>();
}

// 处理拷贝专家
template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void
MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::ProcessCopyExpert(uint32_t tokenIndex, float scaleVal)
{
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};
    DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    LocalTensor<ExpandXType> tmpUb = moeSumQueue_.AllocTensor<ExpandXType>();
    DataCopyPad(tmpUb, oriXGM_[tokenIndex * axisH_], expandXCopyParams, copyPadExtParams);
    moeSumQueue_.EnQue(tmpUb);
    tmpUb = moeSumQueue_.DeQue<ExpandXType>();

    Cast(rowTmpFloatLocal_, tmpUb, AscendC::RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    moeSumQueue_.FreeTensor<ExpandXType>(tmpUb);
    Muls(mulBufLocal_, rowTmpFloatLocal_, scaleVal, axisH_);
    PipeBarrier<PIPE_V>();
    Add(sumFloatBufLocal_, sumFloatBufLocal_, mulBufLocal_, axisH_);
    PipeBarrier<PIPE_V>();
}

// 处理Moe专家
template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void
MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::ProcessMoeExpert(uint32_t tokenIndexOffset,
                                                                              uint32_t topkId, float scaleVal)
{
    uint32_t processLen = axisH_;
    const DataCopyExtParams xScaleCopyParams{1U, static_cast<uint32_t>(tokenScaleCnt_ * sizeof(ExpandXType)), 0U, 0U,
                                             0U};
    const DataCopyExtParams expandXCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    const DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};

    GM_ADDR wAddr = (__gm__ uint8_t *)(epWindowGM_) + (tokenIndexOffset + topkId) * hAlignWinSize_;
    rowTmpGlobal_.SetGlobalBuffer((__gm__ XType *)wAddr);
    LocalTensor<XType> tmpUb = moeSumQueue_.AllocTensor<XType>();
    if constexpr (IsInt8Quant) {
        DataCopyPad(tmpUb, rowTmpGlobal_, xScaleCopyParams, copyPadExtParams);
    } else {
        DataCopyPad(tmpUb, rowTmpGlobal_, expandXCopyParams, copyPadExtParams);
    }
    moeSumQueue_.EnQue(tmpUb);
    tmpUb = moeSumQueue_.DeQue<XType>();
    if constexpr (IsInt8Quant) {
        Int8DequantProcess(tmpUb);
    }
    Cast(rowTmpFloatLocal_, tmpUb, AscendC::RoundMode::CAST_NONE, processLen);
    PipeBarrier<PIPE_V>();
    AscendC::Muls(mulBufLocal_, rowTmpFloatLocal_, scaleVal, processLen);
    PipeBarrier<PIPE_V>();
    AscendC::Add(sumFloatBufLocal_, sumFloatBufLocal_, mulBufLocal_, processLen);
    moeSumQueue_.FreeTensor<XType>(tmpUb);
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::ExpertScaleCopy(
    const uint32_t beginIndex, const uint32_t endIndex, const uint32_t tokenPerAivNum)
{
    expertScaleBeginIdx_ = beginIndex;
    uint32_t expertScaleEndIdx = endIndex;
    uint32_t expertScaleCntPerCore = tokenPerAivNum * axisK_;
    if (isInputExpertMaskFlag_) {
        expertScaleBeginIdx_ = validBsIndexTensor_.GetValue(beginIndex);
        expertScaleEndIdx = validBsIndexTensor_.GetValue(endIndex - 1);
        expertScaleCntPerCore = (expertScaleEndIdx - expertScaleBeginIdx_ + 1) * axisK_;
    }
    tpipe_->InitBuffer(expertScalesBuf_, Ceil(expertScaleCntPerCore * sizeof(float), UB_ALIGN) * UB_ALIGN);
    expertScalesLocal_ = expertScalesBuf_.Get<float>();
    const DataCopyExtParams tokenScaleParams{1U, static_cast<uint32_t>(expertScaleCntPerCore * sizeof(float)), 0U, 0U,
                                             0U};
    const DataCopyPadExtParams<float> copyPadFloatParams{false, 0U, 0U, 0U};
    DataCopyPad(expertScalesLocal_, expertScalesGM_[expertScaleBeginIdx_ * axisK_], tokenScaleParams,
                copyPadFloatParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
}

// 传递token
template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::AlltoAllServerDispatch()
{
    if (sendServerNum_ > 0) {
        // 获取winOut和winIn，发送count+token
        uint32_t targRankId = localRankId_ % serverRankSize_ + coreIdx_ * serverRankSize_;
        uint64_t srcrdmaAddr = (uint64_t)(GetWindowsInAddr(localRankId_));
        uint64_t dstrdmaAddr = (uint64_t)(GetWindowsInAddr(targRankId));
        // 填充batchwrite结构体
        batchWriteItemLocalB64(0) = srcrdmaAddr;
        batchWriteItemLocalB64(0 + 1) = dstrdmaAddr;
        if (coreIdx_ == (startServerId_ / sendServerNum_)) {
            batchWriteItemLocalB64(0 + 2) = 0;
        } else {
            batchWriteItemLocalB64(0 + 2) =
                winHeaderBytes_ / sizeof(uint64_t) + (maxLocalBs_ * packedTokenBytes_) / sizeof(ExpandXType);
        }
        batchWriteItemLocalB32(0 + 6) = HcclDataType::HCCL_DATA_TYPE_FP16;
        batchWriteItemLocalB32(0 + 7) = targRankId;
        // 结构体填充完后，写入GM中
        SyncFunc<AscendC::HardEvent::S_MTE3>();
    }
    SyncAll<true>();
    // 0核处理 hccl_.BatchWrite<true>()
    if (coreIdx_ == (startServerId_ / sendServerNum_)) {
        uint64_t srcAddr = (uint64_t)(GetWindowsInAddr(localRankId_));
        uint64_t dstAddr = (uint64_t)(GetWindowsInAddr(localRankId_));
        localInWindow_.SetGlobalBuffer((__gm__ ExpandXType *)(dstAddr));
        localOutWindow_.SetGlobalBuffer((__gm__ ExpandXType *)(srcAddr));
        uint32_t tokenAlignCnt = packedTokenBytes_ / sizeof(ExpandXType);
        for (uint32_t tokenId = 0U; tokenId < maxLocalBs_; tokenId++) {
            LocalTensor<ExpandXType> inUb = moeQueue_.AllocTensor<ExpandXType>();
            DataCopy(inUb, localOutWindow_[tokenId * tokenAlignCnt], tokenAlignCnt);
            moeQueue_.EnQue(inUb);
            LocalTensor<ExpandXType> outUb = moeQueue_.DeQue<ExpandXType>();
            DataCopy(localInWindow_[tokenId * tokenAlignCnt], outUb, tokenAlignCnt);
            moeQueue_.FreeTensor<ExpandXType>(outUb);
        }
    }
}

// 读取count的flag位，确保所有server组都执行结束
template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::WaitWinInCount()
{
    if (coreIdx_ < serverNum_) {
        tpipe_->InitBuffer(localCntBuf_, UB_ALIGN);
        localCntTensor_ = localCntBuf_.Get<uint64_t>();
        GlobalTensor<uint64_t> flagGm;
        flagGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(windowInGM_));
        uint64_t sumofCount = 0;
        // 对winIn中的count的flag位进行等待求和判断
        while (sumofCount != serverNum_) {
            for (uint32_t serverIndex = 0; serverIndex < serverNum_; serverIndex++) {
                uint32_t flagOffset = serverIndex * winInSliceBytes_;
                DataCopy(localCntTensor_, flagGm[flagOffset], flagU64CopyCntAlign_);
                PipeBarrier<PIPE_ALL>();
                uint64_t cnt = localCntTensor_.GetValue(0);
                sumofCount += cnt;
            }
        }
    }
    SyncAll<true>();
}

// token到齐等待和combine求和
template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::WaitWinInTokenAndCombine()
{
    for (uint32_t serverIndex = startServerId_; serverIndex < endServerId_; serverIndex++) {
        localCntTensor_ = localCntBuf_.Get<uint64_t>();
        GlobalTensor<uint64_t> flagGm;
        flagGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(windowInGM_));
        uint32_t offsetOfNum = serverIndex * winInSliceBytes_; // 获取token数目值偏移
        DataCopy(localCntTensor_, flagGm[offsetOfNum], flagU64CopyCntAlign_);
        uint64_t count = localCntTensor_.GetValue(1); // 获取token数目
        for (uint32_t countIndex = 0; countIndex < count; countIndex++) {
            LocalTensor<uint32_t> localTokenIdTensor = localTokenIdBuf_.Get<uint32_t>();
            uint32_t offsetOfToken = countIndex * packedTokenBytes_; // 获取token的偏移
            GM_ADDR winInTKAddr = windowInGM_ + winHeaderBytes_ + serverIndex * winInSliceBytes_;
            GlobalTensor<uint32_t> IdGm;
            IdGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(winInTKAddr));
            DataCopy(localTokenIdTensor, IdGm[offsetOfToken], flagU64CopyCntAlign_ * 2); // uint32 32B对齐后的数量为：8
            uint32_t tokenId = localTokenIdTensor.GetValue(0);
            uint32_t sumOfBlockFlag = 0;
            while (sumOfBlockFlag != tokenDataBlockNum_) {
                for (uint32_t blockIndex = 0; blockIndex < tokenDataBlockNum_; blockIndex++) {
                    LocalTensor<uint32_t> localFlagTensor = localFlagBuf_.Get<uint32_t>();
                    GlobalTensor<uint32_t> flagGm;
                    GM_ADDR winInFlagAddr = winInTKAddr + SPLIT_BLOCK_SIZE + (blockIndex + 1) * SPLIT_BLOCK_DATA_SIZE +
                                            blockIndex * SPLIT_BLOCK_FLAG_SIZE; // 获取block中flag偏移
                    flagGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(winInFlagAddr));
                    DataCopy(localFlagTensor, flagGm, flagU64CopyCntAlign_ * 2); // uint32 32B对齐后的数量为：8
                    uint32_t cnt = localFlagTensor.GetValue(0);
                    sumOfBlockFlag += cnt;
                }
            }
            // 完整到达，拼凑block
            LocalTensor<ExpandXType> localTempTensor = localOutTempBuf_.Get<ExpandXType>();
            LocalTensor<ExpandXType> localOutTensor = localOutTensorBuf_.Get<ExpandXType>();
            GlobalTensor<ExpandXType> blockGm;
            blockGm.SetGlobalBuffer(reinterpret_cast<__gm__ ExpandXType *>(winInTKAddr) + SPLIT_BLOCK_SIZE);
            DataCopy(localTempTensor, blockGm, (SPLIT_BLOCK_SIZE * tokenDataBlockNum_) / sizeof(ExpandXType));
            const uint32_t dstRepeatSize = SPLIT_BLOCK_DATA_SIZE / BLOCK_SIZE; // 15
            const uint32_t srcRepeatSize = SPLIT_BLOCK_SIZE / BLOCK_SIZE;      // 16
            Copy(localOutTensor, localTempTensor, uint64_t(SPLIT_BLOCK_DATA_SIZE / sizeof(ExpandXType)),
                 uint8_t(tokenDataBlockNum_), {1, 1, dstRepeatSize, srcRepeatSize});
            // 将拼凑好的token, combine后填入到对应expand out id位置中
            PipeBarrier<PIPE_V>();
            tokenAtomicAdd(expandOutGlobal_[tokenId * axisH_], localOutTensor);
        }
    }
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void
MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::tokenAtomicAdd(GlobalTensor<ExpandXType> globalSet,
                                                                            LocalTensor<ExpandXType> localSet)
{
    AscendC::SetAtomicAdd<ExpandXType>();
    DataCopy(globalSet, localSet, tokenDataBytesAlign_ / sizeof(ExpandXType));
    AscendC::SetAtomicNone();
}

template <TemplateMoeDistributeCombineV2A5KFCTypeClass>
__aicore__ inline void MoeDistributeCombineV2A5LayeredHostcpu<TemplateMoeDistributeCombineV2A5KFCTypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        if constexpr (IsNeedReduceScatter) {
            ReduceScatterTrans();
        }
        BuffInit();
        if constexpr (IsNeedReduceScatter) {
            SetWaitTpStatus();
        }
        // 1、Server内
        AlltoAllDispatch();
        SumToWindow();
        // 2、Server间
        AlltoAllServerDispatch();
        WaitWinInCount();
        WaitWinInTokenAndCombine();
    }
}

} // namespace MoeDistributeCombineV2A5Impl
#endif // MOE_DISTRIBUTE_COMBINE_IMPL_H