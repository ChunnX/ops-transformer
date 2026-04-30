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
 * \file attention_to_ffn.h
 * \brief AttentionToFFN算子核心实现——将Attention侧的token数据发送到FFN侧
 *
 * 本文件实现了MoE（Mixture of Experts）模型中Attention到FFN的token分发流程：
 *   1. 从GM读取输入token数据（X）和专家路由表（expertRankTable）
 *   2. 根据expertIds查找每个token对应的目标FFN rank和本地专家ID
 *   3. 可选地对token数据进行量化（smoothScales + 动态量化→int8）
 *   4. 通过HCCL通信窗口（peermem）将token数据写入目标FFN卡的内存
 *   5. 设置tokenInfoTable中的flag，通知FFN侧数据已就绪
 *
 * 模板参数：
 *   - XType: 输入数据类型（half/bfloat16_t）
 *   - isQuant: 是否启用量化模式（int8量化以减少通信量）
 *   - isSync: 是否为同步模式（异步模式需记录发送状态到workspace）
 *   - isActiveMask: 是否启用activeMask（仅发送有效token，跳过padding token）
 *
 * 核心数据结构：
 *   - tokenInfoTable: 通信窗口中的token状态表，flag位标记FFN是否可写、localExpId标记专家归属
 *   - tokenData: 通信窗口中的token数据区，存放实际的token向量数据
 *   - expertRankTable: 专家→FFN卡的映射表，支持一卡多专家和负载均衡
 *
 * 执行流水线（Process）：
 *   ActiveMaskCalCnt(可选) → SendTokenToFFN → SyncAll → SetFlagToFFN → SetFlagInAttn(可选)
 */

#ifndef ATTENTION_TO_FFN_H
#define ATTENTION_TO_FFN_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/reduce/sum.h"
#include "adv_api/reduce/reduce.h"
#include "kernel_tiling/kernel_tiling.h"
#include "attention_to_ffn_tiling.h"
#if __has_include("../common/op_kernel/moe_distribute_base.h")
#include "../common/op_kernel/moe_distribute_base.h"
#include "../common/op_kernel/mc2_kernel_utils.h"
#else
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#endif

namespace AttentionToFFNImpl {
constexpr uint8_t BUFFER_NUM = 2;              // 双缓冲机制，提高数据搬运和计算的重叠度
constexpr uint32_t UB_ALIGN = 32;              // UB（Unified Buffer）按32字节对齐，硬件约束
constexpr uint8_t WIN_OFFSET_CNT = 2;          // 通信窗口偏移数量：[0]=tokenInfoTable偏移，[1]=tokenData偏移
constexpr uint32_t SCALE_PARAM_PAD_SIZE = 128; // 量化参数预留128字节，用于存放动态量化的反量化scale
constexpr uint32_t WIN_ALIGN = 512;            // 通信窗口偏移按512字节对齐
constexpr uint32_t REP_STRIDE = 8;             // ReduceMax相邻迭代间的地址步长（256bit/sizeof(float)=8）
constexpr uint32_t STATUS_REP_STRIDE = 8;      // 状态tensor相邻元素间的步长（32B/sizeof(int32)=8）
constexpr uint32_t EXPERT_TABLE_REP_STRIDE = 16; // 专家RankTable缓存刷新步长（64B/sizeof(int32)=16）
constexpr uint32_t WORKSPACE_ELEMENT_OFFSET = 128; // Workspace元素偏移对齐粒度（128字节）
constexpr uint32_t DYNAMIC_QUANT = 2;          // 动态量化模式标识
constexpr uint32_t RANK_OFFSET_STRIDE = 2;     // RankTable中每条记录的步长：第0位为rankCnt，之后每2个一组(rankId, localExpId)
constexpr uint32_t TOKEN_INFO_TABLE_RS = 2;    // tokenInfoTable前2位为flag和layer_id，后续为各token的状态
constexpr uint32_t TOKEN_INFO_TABLE_COPY_BLOCK_CNT = 2; // tokenInfoTable在DataCopy时每次搬运2个int32（flag+layerId）
constexpr float INT8_MAX_VALUE = 127.0f;       // int8最大值，用于动态量化scale计算


#define TemplateAttentionToFFNTypeClass typename XType, bool isQuant, bool isSync, bool isActiveMask
#define TemplateAttentionToFFNTypeFunc XType, isQuant, isSync, isActiveMask

using namespace AscendC;
template <TemplateAttentionToFFNTypeClass>
class AttentionToFFN {
public:
    __aicore__ inline AttentionToFFN() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR sessionId, GM_ADDR microBatchId, GM_ADDR layerId, GM_ADDR expertIds,
                                GM_ADDR expertRankTable, GM_ADDR scales, GM_ADDR active_mask, GM_ADDR workspaceGM, TPipe *pipe,
                                const AttentionToFFNTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ReduceMaxInplace(const LocalTensor<float>& srcLocal, uint32_t count);
    __aicore__ inline void QuantInit(GM_ADDR scales);
    __aicore__ inline void InitByTinglingData(const AttentionToFFNTilingData *tilingData);
    __aicore__ inline void QuantProcess(uint32_t expertIndex);
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId, uint32_t &endTokenId, uint32_t &sendTokenNum);
    __aicore__ inline void SendTokenToFFN();
    __aicore__ inline void SendTokenToFFNByTokenIdx(uint32_t tokenIdx);
    __aicore__ inline void SetFlagToFFN();
    __aicore__ inline void ActiveMaskCalCnt();
    __aicore__ inline void SetFlagInAttn();
    __aicore__ inline void FindExpertRank(int32_t expertId);
    __aicore__ inline void SetExpertAndRank(uint32_t tokenIdx, uint32_t tokenId, uint32_t topkId);
    __aicore__ inline void CheckFlagAndSetTableGM(int32_t toRankId, GM_ADDR &toRankAddr, GlobalTensor<int32_t> &tokenInfoTableGMTensor);
    TPipe *tpipe_{nullptr};                          // 管道管理器，管理AscendC的Queue和Buffer
    GlobalTensor<XType> xGMTensor_;                  // GM上输入token数据（FP16/BF16）
    GlobalTensor<int32_t> sessionIdGMTensor_;        // GM上sessionId，标识当前Attention Worker
    GlobalTensor<int32_t> microBatchIdGMTensor_;     // GM上microBatchId，标识当前microBatch
    GlobalTensor<int32_t> layerIdGMTensor_;          // GM上layerId，标识当前模型层
    GlobalTensor<int32_t> expertIdsGMTensor_;        // GM上expertIds，每个token的topK专家索引
    GlobalTensor<int32_t> expertRankTableGMTensor_;  // GM上expertRankTable，专家到FFN卡的映射表
    GlobalTensor<int32_t> syncStatusGMTensor_;       // GM上同步状态，异步场景记录发送状态到workspace
    GlobalTensor<float> scalesGMTensor_;             // GM上smoothScales量化参数
    GlobalTensor<bool> activeMaskGMTensor_;          // GM上activeMask激活掩码
    LocalTensor<XType> xInTensor_;                   // UB上输入token数据（从GM搬入）
    LocalTensor<int32_t> statusTensor_;              // UB上状态tensor，存放localExpId用于写入tokenInfoTable
    LocalTensor<float> scalesFp32Tensor_;            // UB上量化scale（FP32）
    LocalTensor<int8_t> xOutTensor_;                 // UB上量化后的int8输出数据
    LocalTensor<int32_t> ffnStatusTensor_;           // UB上FFN发送状态（同步模式使用）
    LocalTensor<int32_t> syncStatusWorkspaceTensor_; // UB上同步状态workspace（同步模式使用）
    LocalTensor<int32_t> ffnFlagTensor_;             // UB上FFN flag（用于检查FFN端是否可写入）
    LocalTensor<float> smoothScalesTensor_;          // UB上smoothScales（量化模式使用）
    LocalTensor<int32_t> expertIdsTensor_;           // UB上expertIds（从GM搬入）

    TBuf<> smoothScalesBuf_;                         // 量化使用，搬入smoothScales
    TBuf<> expertIdsBuf_;                            // 存放expertIds数据
    TBuf<> statusBuf_;                               // 存放status状态数据
    TBuf<> receiveDataCastFloatBuf_;                  // 量化使用，存放Cast到FP32的中间结果
    TBuf<> sumOutBuf_;                               // 存放Sum归约结果（activeMask计算有效token数）
    TBuf<> activeMaskBuf_;                           // 存放activeMask数据
    TBuf<> castTempBuf_;                             // Cast临时缓冲区
    TBuf<> ffnStatusBuf_;                            // 存放FFN发送状态
    TBuf<> syncStatusWorkspaceBuf_;                  // 存放同步状态workspace数据
    TBuf<> ffnFlagBuf_;                              // 存放FFN flag数据
    TBuf<> attnStatusBuf_;                           // 存放Attention侧状态（activeMask场景复用expertIdsBuf_）
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_;  // 非量化使用，双缓冲Queue
    TQue<QuePosition::VECIN, 1> xInQueue_;           // 量化使用，量化前的输入Queue
    TQue<QuePosition::VECOUT, 1> xOutQueue_;         // 量化使用，量化后的输出Queue
    GM_ADDR syncStatusWorkspaceGM_;                  // 异步场景使用workSpace记录发送状态信息

    int32_t dstExpertId_{0};           // 当前token要发送到的目标专家ID
    int32_t toRankId_{0};              // 当前token要发送到的目标FFN rank ID
    int32_t localExpId_{0};            // 目标专家在目标FFN卡上的本地专家ID

    uint32_t aivId_{0};                // 当前AIV核心ID（用于多核并行分配）
    uint32_t rankId_{0};               // 当前卡在本通信域中的rank ID
    uint32_t axisX_{0};                // token组数X
    uint32_t axisBS_{0};               // batch sequence size
    uint32_t axisH_{0};                // hidden size
    uint32_t axisL_{0};                // 模型层数L
    uint32_t axisK_{0};                // topK专家数
    bool isScales_{false};             // 是否有smoothScales输入
    uint32_t expertNum_{0};            // 所有专家总数（moeExpert + sharedExpert）
    uint32_t expRankTableM_{0};        // expertRankTable最后一维长度M
    uint32_t attentionWorkerNum_{0};   // Attention Worker节点数量
    uint32_t infoTableLastDimNum_{0};  // tokenInfoTable最后一维大小
    uint32_t aivNum_{0};               // AIV核心总数
    uint32_t worldSize_{0};            // 通信域大小（总卡数）
    uint64_t hSize_{0};                // 单个token的隐藏层字节数 = H * sizeof(XType)
    uint32_t sessionId_{0};            // 当前Attention Worker的session ID
    uint32_t microBatchId_{0};         // 当前microBatch的ID
    uint32_t expertIdsCnt_{0};         // expertIds元素总数 = X * BS * K
    uint32_t layerId_{0};              // 当前模型层ID
    uint32_t totalSendNum_{0};         // 总发送token数 = X * curBsCnt * (K + sharedExpertNum)
    uint32_t sendNum_{0};              // 当前AIV核心需要发送的token数
    uint32_t quantMode_{0};            // 量化模式
    uint32_t scaleParamPad_{0};        // 量化参数填充大小
    uint32_t ffnNum_{0};               // FFN节点数量 = worldSize - attentionWorkerNum
    uint32_t ffnStartRankId_{0};       // FFN节点起始rank ID
    uint64_t microBatchNum_{0};        // microBatch切分数量
    uint64_t axisHS_{0};               // ffnTokenDataShape第5维（单个token发送数据长度）
    uint64_t hCommuSize_{0};           // 量化模式下通信数据大小 = H * sizeof(int8) + scaleParamPad
    uint64_t ffnNumAlignSize_{0};      // FFN数量对齐后的大小（32B对齐）
    uint64_t layIdsExpRankTableOffset_{0}; // expertRankTable中当前层的偏移 = layerId * expertRankTableCnt
    uint64_t preExpRankTableOffset_{0};    // expertRankTable前一层偏移（预留）
    uint64_t winTokenDataOffset_{0};       // 通信窗口中tokenData区域的偏移（基于sessionId和microBatchId）
    uint64_t winInfoTableOffset_{0};       // 通信窗口中tokenInfoTable区域的偏移（基于sessionId和microBatchId）
    uint64_t winOffset_[WIN_OFFSET_CNT]{0, 0}; // 通信窗口偏移：[0]=tokenInfoTable起始偏移，[1]=tokenData起始偏移
    uint32_t curBsCnt_{0};             // 当前有效token数（activeMask场景可能小于axisBS_）
    uint32_t axisBsAlignSize_{0};      // BS按32B对齐后的大小
    uint32_t moeExpertNum_{0};         // MoE路由专家数量
    uint32_t expertRankTableCnt_{0};   // expertRankTable元素总数 = expertNum * expRankTableM
    uint32_t ffnNumAlignCnt_{0};       // ffnNumAlignSize对应的int32元素个数
    uint32_t aivWorkspaceOffset_{0};   // 每个AIV核心在workspace中的偏移（128B对齐）
    uint8_t windowType_{0};            // 通信窗口类型
    uint8_t sharedExpertNum_{1};       // 共享专家数量（默认1个）
    __gm__ HcclOpResParam *winContext_{nullptr}; // HCCL通信窗口上下文，包含通信域和窗口信息
};

// QuantInit: 量化模式初始化
// 分配量化流水线所需的UB缓冲区：
//   - xInQueue_: 输入Queue，双缓冲存放FP16/BF16的token数据（每块H*sizeof(XType)）
//   - xOutQueue_: 输出Queue，双缓冲存放int8量化结果+scale参数（每块H*sizeof(int8)+128B）
//   - receiveDataCastFloatBuf_: Cast中间缓冲区，存放FP32中间结果（H*4字节）
//   - smoothScalesBuf_: smoothScales缓冲区，存放per-channel量化scale（H*4字节）
// 量化通信数据大小 = H*sizeof(int8) + 128B(scale参数预留)
template <TemplateAttentionToFFNTypeClass>
__aicore__ inline void AttentionToFFN<TemplateAttentionToFFNTypeFunc>::QuantInit(GM_ADDR scales)
{
    scaleParamPad_ = SCALE_PARAM_PAD_SIZE; // 预留128B给量化参数
    hCommuSize_ = axisH_ * sizeof(int8_t) + scaleParamPad_;
    tpipe_->InitBuffer(xInQueue_, BUFFER_NUM, hSize_); // 14K *2
    tpipe_->InitBuffer(xOutQueue_, BUFFER_NUM, hCommuSize_); // 7K *2 + 256B
    if (isScales_) { // 输入smoothScale参数
        scalesGMTensor_.SetGlobalBuffer((__gm__ float*)(scales));
    }
    uint32_t hFp32Size = axisH_ * sizeof(float);
    tpipe_->InitBuffer(receiveDataCastFloatBuf_, hFp32Size);  // H * 4
    tpipe_->InitBuffer(smoothScalesBuf_, hFp32Size);
}

// InitByTinglingData: 从TilingData解析所有运行时参数
// 包括：AIV核心ID、HCCL通信上下文、矩阵维度(X/BS/H/L/K)、专家信息、量化参数、通信域信息等
// 同时计算派生参数：hSize_(单token字节数)、expertIdsCnt_(expertIds元素数)、ffnNumAlignSize_(FFN数对齐)等
template <TemplateAttentionToFFNTypeClass>
__aicore__ inline void AttentionToFFN<TemplateAttentionToFFNTypeFunc>::InitByTinglingData(const AttentionToFFNTilingData *tilingData)
{
    aivId_ = GetBlockIdx();
    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    winContext_ = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    rankId_ = winContext_->localUsrRankId;
    axisX_ = tilingData->attentionToFFNInfo.X;
    axisBS_ = tilingData->attentionToFFNInfo.BS;
    axisH_ = tilingData->attentionToFFNInfo.H;
    axisL_ = tilingData->attentionToFFNInfo.L;
    axisK_ = tilingData->attentionToFFNInfo.K;
    expertNum_ = tilingData->attentionToFFNInfo.expertNum; // 所有专家数：共享专家+所有的moe专家
    moeExpertNum_ = tilingData->attentionToFFNInfo.moeExpertNum;
    expRankTableM_ = tilingData->attentionToFFNInfo.expRankTableM;
    microBatchNum_ = tilingData->attentionToFFNInfo.microBatchNum;
    attentionWorkerNum_ = tilingData->attentionToFFNInfo.attentionWorkerNum;
    infoTableLastDimNum_ = tilingData->attentionToFFNInfo.infoTableLastDimNum;
    axisHS_ = tilingData->attentionToFFNInfo.HS;
    aivNum_ = tilingData->attentionToFFNInfo.aivNum;
    worldSize_ = tilingData->attentionToFFNInfo.worldSize;
    quantMode_ = tilingData->attentionToFFNInfo.quantMode;
    isScales_ = tilingData->attentionToFFNInfo.isScales;
    ffnStartRankId_ = tilingData->attentionToFFNInfo.ffnStartRankId;
    windowType_ = tilingData->attentionToFFNInfo.windowType;
    sharedExpertNum_ = tilingData->attentionToFFNInfo.sharedExpertNum;
    ffnNum_ = worldSize_ - attentionWorkerNum_;
    curBsCnt_ = axisBS_;
    hSize_ = axisH_ * sizeof(XType);
    expertIdsCnt_ = axisX_ * axisBS_ * axisK_; 
    expertRankTableCnt_ = expertNum_ * expRankTableM_;
    ffnNumAlignSize_ = Ceil(ffnNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    axisBsAlignSize_ = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
    aivWorkspaceOffset_ = Ceil(ffnNum_ * sizeof(int32_t), WORKSPACE_ELEMENT_OFFSET) * WORKSPACE_ELEMENT_OFFSET;
}

template <TemplateAttentionToFFNTypeClass>
__aicore__ inline void AttentionToFFN<TemplateAttentionToFFNTypeFunc>::Init(GM_ADDR x, GM_ADDR sessionId, GM_ADDR microBatchId,
    GM_ADDR layerId, GM_ADDR expertIds, GM_ADDR expertRankTable, GM_ADDR scales, GM_ADDR activeMask, GM_ADDR workspaceGM, TPipe *pipe, const AttentionToFFNTilingData *tilingData)
{
    // 保存管道管理器指针，后续通过tpipe_分配UB缓冲区
    tpipe_ = pipe;
    // 从tilingData解析所有运行时参数（矩阵维度、专家数、通信域信息等）
    InitByTinglingData(tilingData);

    // 设置各输入tensor的GM全局地址
    xGMTensor_.SetGlobalBuffer((__gm__ XType*)x);
    sessionIdGMTensor_.SetGlobalBuffer((__gm__ int32_t*)sessionId);
    microBatchIdGMTensor_.SetGlobalBuffer((__gm__ int32_t*)microBatchId);
    layerIdGMTensor_.SetGlobalBuffer((__gm__ int32_t*)layerId);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t*)expertIds);
    expertRankTableGMTensor_.SetGlobalBuffer((__gm__ int32_t*)expertRankTable);
    activeMaskGMTensor_.SetGlobalBuffer((__gm__ bool*)activeMask);
    // 读取sessionId/microBatchId/layerId：先刷新Cache Line再读取，确保数据一致性
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sessionIdGMTensor_);
    sessionId_ = sessionIdGMTensor_.GetValue(0); // 当前x=1，session_ids直接从gm上读取第一个值
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(microBatchIdGMTensor_);
    microBatchId_ = microBatchIdGMTensor_.GetValue(0); // 当前x=1，microBatchId_直接从gm上读取第一个值
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(layerIdGMTensor_);
    layerId_ = layerIdGMTensor_.GetValue(0); // 当前x=1，layerId_直接从gm上读取第一个值
    // 计算当前层在expertRankTable中的偏移：layerId * (expertNum * expRankTableM)
    layIdsExpRankTableOffset_ = layerId_ * expertRankTableCnt_;

    // 分配expertIds和ffnFlag的UB缓冲区（32B对齐）
    uint32_t expertIdsAlign = Ceil(expertIdsCnt_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN; // 约束32对齐
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsAlign); // 对齐32B
    tpipe_->InitBuffer(ffnFlagBuf_, UB_ALIGN); // 对齐32B
    expertIdsTensor_ = expertIdsBuf_.Get<int32_t>();
    ffnFlagTensor_ = ffnFlagBuf_.Get<int32_t>();
    // 根据量化模式分配不同的缓冲区
    if constexpr (isQuant) {
        // 量化模式：分配输入Queue(FP16/BF16)和输出Queue(int8+scale参数)
        QuantInit(scales);
        // 复用量化缓冲区作为Cast临时缓冲区和Sum输出缓冲区
        castTempBuf_ = receiveDataCastFloatBuf_;
        sumOutBuf_ = smoothScalesBuf_;
    } else {
        // 非量化模式：分配双缓冲Queue，直接搬运FP16/BF16数据
        tpipe_->InitBuffer(xQueue_, BUFFER_NUM, hSize_); // H * 2
    }
    // 同步模式：分配FFN发送状态缓冲区和workspace
    if constexpr (isSync) {
 	  	tpipe_->InitBuffer(ffnStatusBuf_, ffnNumAlignSize_); // ffnNum_
 	    ffnStatusTensor_ = ffnStatusBuf_.Get<int32_t>();
 	    syncStatusWorkspaceGM_ = workspaceGM;
    }
    // activeMask模式：分配activeMask缓冲区，并复用expertIdsBuf_作为attnStatusBuf_
    if constexpr (isActiveMask) {
        tpipe_->InitBuffer(activeMaskBuf_, axisBsAlignSize_); // BS
        if constexpr (!isQuant) {
            // 非量化+activeMask：需要额外的Cast和Sum缓冲区用于计算有效token数
            uint32_t bsAlignHalf = Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN; // 约束32对齐
            tpipe_->InitBuffer(castTempBuf_, bsAlignHalf); // BS * 2
            tpipe_->InitBuffer(sumOutBuf_, bsAlignHalf); // BS * 2
        }
        attnStatusBuf_ = expertIdsBuf_; // activeMask场景复用expertIdsBuf_（此时expertIds已用完）
    }

    // 计算通信窗口偏移：
    //   winOffset_[0] = 0：tokenInfoTable从窗口起始位置开始
    //   winOffset_[1]：tokenData紧跟tokenInfoTable之后，按512B对齐
    winOffset_[0] = 0;
    winOffset_[1] = Ceil(attentionWorkerNum_ * microBatchNum_ * infoTableLastDimNum_ * sizeof(int32_t), WIN_ALIGN) * WIN_ALIGN; // token_info_table大小 偏移向上取整
    // 计算当前sessionId和microBatchId在tokenInfoTable中的偏移
    winInfoTableOffset_ = (sessionId_ * microBatchNum_ * infoTableLastDimNum_ + microBatchId_ * infoTableLastDimNum_) * sizeof(int32_t); // tokenInfoTable上当前attnWorkId以及microBatchId偏移
    // 计算当前sessionId和microBatchId在tokenData中的偏移
    winTokenDataOffset_ = (sessionId_ * microBatchNum_ * axisBS_ * (axisK_ + sharedExpertNum_) * axisHS_) + (microBatchId_ * axisBS_ * (axisK_ + sharedExpertNum_) * axisHS_);// tokenData上attnWorkId以及microBatchId偏移
}

// ActiveMaskCalCnt: 计算activeMask中有效token的数量
// 流程：GM搬运activeMask(bool) → Cast为half → Sum归约得到有效token总数
// 结果存入curBsCnt_，后续SendTokenToFFN仅发送curBsCnt_个有效token
// 注意：activeMask为true表示token有效，Sum(half(bool→1/0))即统计有效token数
template <TemplateAttentionToFFNTypeClass>
__aicore__ inline void AttentionToFFN<TemplateAttentionToFFNTypeFunc>::ActiveMaskCalCnt()
{   // 搬运x_active_mask, 当前仅用于计算有效token总数
    LocalTensor<bool> activeMaskTensor = activeMaskBuf_.Get<bool>();
    LocalTensor<half> tempTensor = castTempBuf_.Get<half>();
    LocalTensor<half> sumOutTensor = sumOutBuf_.Get<half>();
    // 从GM搬运activeMask到UB（bool类型，长度为BS）
    DataCopyExtParams activeMaskParams = {1U, static_cast<uint32_t>(axisBS_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> activeMaskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(activeMaskTensor, activeMaskGMTensor_, activeMaskParams, activeMaskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    // bool → int8 → half：两步Cast，bool不能直接Cast到half
    LocalTensor<int8_t> activeMaskInt8Tensor = activeMaskTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, activeMaskInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    // Sum归约：true=1, false=0，求和即得到有效token数
    SumParams params{1, axisBsAlignSize_, axisBS_};
    Sum(sumOutTensor, tempTensor, params);
    // V→S：等待向量计算完成，读取标量结果
    SyncFunc<AscendC::HardEvent::V_S>();
    // 将有效token数写入curBsCnt_，后续SendTokenToFFN仅发送前curBsCnt_个token
    curBsCnt_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
}

// ReduceMaxInplace: 原地FP32向量求最大值（用于动态量化scale计算）
// 算法分三步执行，利用硬件Max指令的repeat模式：
//   1. 将count个元素按64个一组（256bit/sizeof(float)），依次与下一组做Max归约
//      即 src[0:63] = Max(src[0:63], src[64:127])，repsFp32-1次迭代
//   2. 处理尾部不足64个的剩余元素（remsFp32），与前面归约结果做Max
//   3. 最终对前64个（或更少）元素做WholeReduceMax，得到全局最大值
// REP_STRIDE=8：相邻repeat间src的步长=64元素（256bit/sizeof(float)），dst步长=0（原地写回）
template <TemplateAttentionToFFNTypeClass>
__aicore__ inline void AttentionToFFN<TemplateAttentionToFFNTypeFunc>::ReduceMaxInplace(const LocalTensor<float>& srcLocal,
    uint32_t count)
{
    // 将count个FP32元素按64个一组（256bit/sizeof(float)）划分
    uint64_t repsFp32 = count >> 6;        // 完整分组数：count / 64
    uint64_t offsetsFp32 = repsFp32 << 6;  // 完整分组的元素总数：repsFp32 * 64
    uint64_t remsFp32 = count & 0x3f;      // 尾部剩余元素数：count % 64
    const uint64_t elemPerRefFp32 = 64UL; // 每组元素数：256bit / sizeof(float)
    // 第一步：逐组Max归约，srcLocal[0:63] = Max(srcLocal[0:63], srcLocal[64:127])，迭代repsFp32-1次
    // REP_STRIDE=8：相邻repeat间src1的步长=64元素，dst/src0步长=0（原地写回前64个位置）
    if (likely(repsFp32 > 1)) { // 8 is rep stride
        Max(srcLocal, srcLocal[elemPerRefFp32], srcLocal, elemPerRefFp32, repsFp32 - 1, {1, 1, 1, 0, REP_STRIDE, 0});
        PipeBarrier<PIPE_V>();
    }
    // 第二步：处理尾部不足64个的剩余元素，与前64个归约结果做Max
    if (unlikely(remsFp32 > 0) && unlikely(offsetsFp32 > 0)) {
        Max(srcLocal, srcLocal[offsetsFp32], srcLocal, remsFp32, 1, {1, 1, 1, 0, REP_STRIDE, 0});
        PipeBarrier<PIPE_V>();
    }
    // 第三步：对前64个（或更少）元素做WholeReduceMax，得到全局最大值
    uint32_t mask = (repsFp32 > 0) ? elemPerRefFp32 : count;
    // 8 is rep stride
    WholeReduceMax(srcLocal, srcLocal, mask, 1, REP_STRIDE, 1, REP_STRIDE);
}

// QuantProcess: 对单个token执行量化处理（FP16/BF16 → int8）
// 量化流水线：
//   1. Cast: FP16/BF16 → FP32（提高量化精度）
//   2. SmoothScales乘法（可选）: FP32 × smoothScales[expertIdx]（per-channel平滑量化）
//   3. 动态量化（可选，quantMode_==DYNAMIC_QUANT）:
//      a. Abs + ReduceMaxInplace: 求FP32向量绝对值最大值
//      b. dynamicScale = 127.0 / maxAbs: 计算动态缩放因子
//      c. Muls: FP32 × dynamicScale: 缩放到int8范围
//   4. Cast: FP32 → int32 → half → int8（三步Cast，硬件约束）
//   5. 在int8数据末尾写入反量化scale = 1.0/dynamicScale（供FFN侧反量化使用）
// expertIndex: 当前token对应的专家索引，用于索引smoothScales
template <TemplateAttentionToFFNTypeClass>
__aicore__ inline void AttentionToFFN<TemplateAttentionToFFNTypeFunc>::QuantProcess(uint32_t expertIndex)
{
    float dynamicScale = 0.0;
    // 计算int8输出的对齐大小（32B对齐），用于在末尾写入反量化scale
    uint32_t hOutSizeAlign = Ceil(axisH_ * sizeof(int8_t), UB_ALIGN) * UB_ALIGN;
    LocalTensor<float> floatLocalTemp;
    floatLocalTemp = receiveDataCastFloatBuf_.Get<float>();
    // 步骤1: FP16/BF16 → FP32，提高后续量化计算精度
    Cast(floatLocalTemp, xInTensor_, RoundMode::CAST_NONE, axisH_);
    PipeBarrier<PIPE_V>();
    xInQueue_.FreeTensor<XType>(xInTensor_);
            
    // 步骤2（可选）: 应用smoothScales，per-channel平滑量化
    if (isScales_) {
        smoothScalesTensor_ = smoothScalesBuf_.Get<float>();
        DataCopyExtParams scalesCopyInParams{1U, static_cast<uint32_t>(axisH_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> copyPadExtParams{false, 0U, 0U, 0U};
        // 搬入当前专家对应的smoothScales（长度为H的FP32向量）
        DataCopyPad(smoothScalesTensor_, scalesGMTensor_[expertIndex * axisH_], scalesCopyInParams, copyPadExtParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        // FP32 × smoothScales: 逐元素乘法
        Mul(floatLocalTemp, floatLocalTemp, smoothScalesTensor_, axisH_);
        PipeBarrier<PIPE_V>();
    }

    // 步骤3（可选）: 动态量化，计算动态缩放因子并缩放到int8范围
    if (quantMode_ == DYNAMIC_QUANT) {
        // 复用smoothScalesBuf_作为绝对值临时缓冲区
        LocalTensor<float> floatLocalAbsTemp = smoothScalesBuf_.Get<float>();
        // 求绝对值
        Abs(floatLocalAbsTemp, floatLocalTemp, axisH_);
        PipeBarrier<PIPE_V>();
        // 原地ReduceMax：求绝对值最大值
        ReduceMaxInplace(floatLocalAbsTemp, axisH_);
        
        // V→S: 等待向量计算完成，读取标量结果
        SyncFunc<AscendC::HardEvent::V_S>();
        // 动态缩放因子 = 127.0 / maxAbs，使量化后数据充分利用int8范围[-127,127]
        dynamicScale = float(INT8_MAX_VALUE) / floatLocalAbsTemp.GetValue(0);
        // S→V: 标量读取完成，恢复向量计算
        SyncFunc<AscendC::HardEvent::S_V>();
        // FP32 × dynamicScale: 缩放到int8范围
        Muls(floatLocalTemp, floatLocalTemp, dynamicScale, axisH_);
        PipeBarrier<PIPE_V>();
    }
    // 步骤4: FP32 → int32 → half → int8 三步Cast（硬件约束，不能直接FP32→int8）
    LocalTensor<half> halfLocalTemp = floatLocalTemp.ReinterpretCast<half>();
    LocalTensor<int32_t> int32LocalTemp = floatLocalTemp.ReinterpretCast<int32_t>();
    // FP32 → int32，四舍五入
    Cast(int32LocalTemp, floatLocalTemp, RoundMode::CAST_RINT, axisH_);
    PipeBarrier<PIPE_V>();
    // 设置DeqScale=1.0，确保int32→half转换不做额外缩放
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();

    // int32 → half，四舍五入
    Cast(halfLocalTemp, int32LocalTemp, RoundMode::CAST_ROUND, axisH_);

    PipeBarrier<PIPE_V>();
    // half → int8，截断量化
    Cast(xOutTensor_, halfLocalTemp, RoundMode::CAST_TRUNC, axisH_);

    // 步骤5: 在int8数据末尾写入反量化scale = 1.0/dynamicScale
    // FFN侧读取int8数据后，乘以此scale即可恢复FP32精度
    floatLocalTemp = xOutTensor_.template ReinterpretCast<float>();
    floatLocalTemp.SetValue(hOutSizeAlign / sizeof(float), float(1.0) / dynamicScale); // int8->float32
}

// FindExpertRank: 根据expertId在expertRankTable中查找目标FFN rank和本地专家ID
// expertRankTable布局（每行对应一个专家）：
//   [rankCnt, (rankId0, localExpId0), (rankId1, localExpId1), ...]
//   - rankCnt: 该专家部署在多少张FFN卡上
//   - 每对(rankId, localExpId): FFN卡的rank ID和该专家在该卡上的本地ID
// 负载均衡策略：sessionId % rankCnt 决定当前Attention Worker发送到哪张FFN卡
// 结果写入 toRankId_（目标FFN rank）和 localExpId_（本地专家ID）
template <TemplateAttentionToFFNTypeClass>
__aicore__ inline void AttentionToFFN<TemplateAttentionToFFNTypeFunc>::FindExpertRank(int32_t expertId)
{
    uint64_t expRankTableOffset = expertId * expRankTableM_ + layIdsExpRankTableOffset_;
    // 此前已刷新expertRankTableGM的整个Cache，可直接读取
    uint32_t rankCnt = expertRankTableGMTensor_.GetValue(expRankTableOffset); // 该专家部署在多少卡上
    uint32_t rankOffset = (sessionId_ % rankCnt) * RANK_OFFSET_STRIDE + 1;       // 第1位为rankCnt, 后面两两组合
    toRankId_ = expertRankTableGMTensor_.GetValue(expRankTableOffset + rankOffset); // 拿到当前要发送的卡号
    localExpId_ = expertRankTableGMTensor_.GetValue(expRankTableOffset + rankOffset + 1); // 拿到当前要发送的local专家号
}

// SplitToCore: 将发送任务均匀分配给各AIV核心
// 负载均衡策略：每个AIV核心处理 curSendCnt/curUseAivNum 个token
//   - 余数部分：前remainderTokenNum个AIV核心各多处理1个token
//   - 例如：10个token分给3个AIV，分配为[4,3,3]或[3,4,3]取决于aivId_
// 输出参数：
//   - startTokenId: 当前AIV核心的起始token索引
//   - endTokenId: 当前AIV核心的结束token索引（不含）
//   - sendTokenNum: 当前AIV核心需要发送的token数量
template <TemplateAttentionToFFNTypeClass>
__aicore__ inline void AttentionToFFN<TemplateAttentionToFFNTypeFunc>::SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId, uint32_t &endTokenId, uint32_t &sendTokenNum)
{
    sendTokenNum = curSendCnt / curUseAivNum; // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum; // 余数
    startTokenId = sendTokenNum * aivId_; // 每个aiv发送时的起始rankid
    if (aivId_ < remainderTokenNum) { // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += aivId_;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

// SetExpertAndRank: 设置当前token的目标专家和目标FFN rank
// tokenIdx: 当前AIV核心内的token序号（用于写入statusTensor_）
// tokenId: token在BS维度中的全局ID（用于索引expertIds和tokenData）
// topkId: 当前token的第几个topK专家（0~K-1为MoE路由专家，K为共享专家）
// 流程：
//   1. 从expertIdsTensor_读取目标专家ID（MoE专家直接读取，共享专家=moeExpertNum_+偏移）
//   2. 调用FindExpertRank查表获取toRankId_和localExpId_
//   3. 将localExpId_写入statusTensor_（后续写入tokenInfoTable通知FFN侧）
//   4. 同步模式下，标记ffnStatusTensor_[toRankId_] = 1（记录已向该FFN发送过token）
template <TemplateAttentionToFFNTypeClass>
__aicore__ inline void AttentionToFFN<TemplateAttentionToFFNTypeFunc>::SetExpertAndRank(uint32_t tokenIdx, uint32_t tokenId, uint32_t topkId)
{
    if (tokenIdx == 0) {
        SyncFunc<AscendC::HardEvent::MTE2_S>(); // 等待前面的expertIdsTensor_的搬入
    }
    if (topkId < axisK_) { // moe专家
        dstExpertId_ = expertIdsTensor_.GetValue(tokenId * axisK_ + topkId);
    } else {
        dstExpertId_ = moeExpertNum_ + (topkId - axisK_);  // 前moeExpertNum_个为路由专家，共享专家排在后面
    }
    FindExpertRank(dstExpertId_); // 查表获取当前要发送的rankId以及localExpId
    statusTensor_.SetValue(tokenIdx * STATUS_REP_STRIDE, localExpId_);
    if constexpr (isSync) { // 异步场景记录FFN节点发送状态
        if (tokenIdx == 0) {
            SyncFunc<AscendC::HardEvent::V_S>(); // 等待前面ffnStatusTensor_初始化
        }
 	    ffnStatusTensor_.SetValue(toRankId_ - ffnStartRankId_, 1);
 	}
}

// CheckFlagAndSetTableGM: 检查FFN侧flag并设置tokenInfoTable的GM地址
// 流程：
//   1. 根据userMemType获取目标FFN rank的通信窗口地址（peermem或userMem）
//   2. 计算tokenInfoTable在通信窗口中的偏移，设置tokenInfoTableGMTensor
//   3. 轮询检查FFN侧的flag位：读取tokenInfoTable前2个int32（flag+layerId）
//      flag==1表示FFN尚未消费上一轮数据，需等待；flag==0表示FFN已就绪可写入
//   4. flag清零后退出循环，等待S_MTE3同步后返回
// 注意：此函数实现了Attention→FFN的流控机制，防止Attention写入过快覆盖FFN未消费的数据
template <TemplateAttentionToFFNTypeClass>
__aicore__ inline void AttentionToFFN<TemplateAttentionToFFNTypeFunc>::CheckFlagAndSetTableGM(int32_t toRankId, GM_ADDR &toRankAddr,
    GlobalTensor<int32_t> &tokenInfoTableGMTensor)
{
    if (winContext_->userMemType == 0) {
        toRankAddr = (GM_ADDR)(((HcclRankRelationResV2 *)(winContext_->remoteRes[toRankId].nextDevicePtr))->windowsIn);
    } else {
        toRankAddr = (GM_ADDR)(__gm__ uint8_t*)(winContext_->userMemRes[toRankId].addr);
    }
    GM_ADDR tokenInfoTable = (__gm__ uint8_t*)(toRankAddr + winInfoTableOffset_);
    tokenInfoTableGMTensor.SetGlobalBuffer((__gm__ int32_t*)tokenInfoTable);

    int32_t ffnFlage = 1;
    while (ffnFlage == 1) {
        DataCopy(ffnFlagTensor_, tokenInfoTableGMTensor, STATUS_REP_STRIDE);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        ffnFlage = ffnFlagTensor_.GetValue(0);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>(); // 等待flag
}

// SendTokenToFFNByTokenIdx: 发送单个token到目标FFN卡
// tokenIdx: 当前AIV核心内的token序号
// 完整流程：
//   1. 计算token在全局BS*(K+1)中的偏移：tokenOffset = aivId_ + tokenIdx * aivNum_
//   2. 分解出tokenId（BS中的位置）和topkId（第几个topK专家）
//   3. 量化模式：从GM搬运token → SetExpertAndRank → QuantProcess → CheckFlag → 写入peermem
//      非量化模式：从GM搬运token → SetExpertAndRank → CheckFlag → 写入peermem
//   4. 将localExpId写入tokenInfoTable对应位置（通知FFN侧该token属于哪个本地专家）
// tokenData在peermem中的地址计算：
//   toRankAddr + winOffset_[1] + (winTokenDataOffset_ + tokenId*(K+1)*HS + topkId*HS) * sizeof
template <TemplateAttentionToFFNTypeClass>
__aicore__ inline void AttentionToFFN<TemplateAttentionToFFNTypeFunc>::SendTokenToFFNByTokenIdx(uint32_t tokenIdx)
{
    GM_ADDR toRankAddr;
    GlobalTensor<int32_t> tokenInfoTableGMTensor;
    // DataCopy参数：statusCopyParams用于写入1个int32的localExpId到tokenInfoTable
    DataCopyParams statusCopyParams = {1U, static_cast<uint16_t>(1 * sizeof(int32_t)), 0U, 0U};
    // DataCopy参数：xCopyParams用于搬运1行H个XType元素的token数据
    DataCopyExtParams xCopyParams = {1U, static_cast<uint32_t>(axisH_ * sizeof(XType)), 0U, 0U, 0U};
    // DataCopy参数：hCommuCopyOutParams用于量化模式下搬运int8数据+scale参数
    DataCopyExtParams hCommuCopyOutParams = {1U, static_cast<uint32_t>(hCommuSize_), 0U, 0U, 0U};
    DataCopyPadExtParams<XType> copyPadExtParams{false, 0U, 0U, 0U};

    // 计算当前token在全局BS*(K+1)中的线性偏移（交错分配：aivId_为起始偏移，aivNum_为步长）
    uint32_t tokenOffset = aivId_ + (tokenIdx * aivNum_); // 当前token在BS*(K+1)的偏移
    uint32_t tokenId = tokenOffset / (axisK_ + sharedExpertNum_); // 当前token在BS中的Id
    uint32_t topkId = tokenOffset % (axisK_ + sharedExpertNum_); // 当前token的第几个topk
    if constexpr (isQuant) {
        // === 量化模式发送流程 ===
        GlobalTensor<int8_t> tokenDataGMTensor;
        // 从GM搬运token数据到UB（FP16/BF16）
        xInTensor_ = xInQueue_.AllocTensor<XType>();
        DataCopyPad(xInTensor_, xGMTensor_[tokenId * axisH_], xCopyParams, copyPadExtParams);
        xInQueue_.EnQue(xInTensor_);
        xInTensor_ = xInQueue_.DeQue<XType>();
        xOutTensor_ = xOutQueue_.AllocTensor<int8_t>();
        // 查表获取目标FFN rank和本地专家ID
        SetExpertAndRank(tokenIdx, tokenId, topkId);
        // 执行量化：FP16/BF16 → int8，末尾附带反量化scale
        QuantProcess(dstExpertId_);
        // 检查FFN侧flag（流控），并设置tokenInfoTable的GM地址
        CheckFlagAndSetTableGM(toRankId_, toRankAddr, tokenInfoTableGMTensor);
        // 计算tokenData在目标FFN卡peermem中的地址
        GM_ADDR tokenData = (__gm__ uint8_t*)(toRankAddr + winOffset_[1] + (winTokenDataOffset_ + 
            (tokenId * (axisK_ + sharedExpertNum_) * axisHS_) + (topkId * axisHS_)) * sizeof(int8_t));
        tokenDataGMTensor.SetGlobalBuffer((__gm__ int8_t*)tokenData);
        // 将量化后的int8数据+scale写入目标FFN卡的peermem
        xOutQueue_.EnQue(xOutTensor_);
        xOutTensor_ = xOutQueue_.DeQue<int8_t>();
        DataCopyPad(tokenDataGMTensor, xOutTensor_, hCommuCopyOutParams);
        xOutQueue_.FreeTensor<int8_t>(xOutTensor_);
    } else {
        // === 非量化模式发送流程 ===
        GlobalTensor<XType> tokenDataGMTensor;
        // 从GM搬运token数据到UB（FP16/BF16）
        xInTensor_ = xQueue_.AllocTensor<XType>();
        DataCopyPad(xInTensor_, xGMTensor_[tokenId * axisH_], xCopyParams, copyPadExtParams);
        // 查表获取目标FFN rank和本地专家ID
        SetExpertAndRank(tokenIdx, tokenId, topkId);
        // 检查FFN侧flag（流控），并设置tokenInfoTable的GM地址
        CheckFlagAndSetTableGM(toRankId_, toRankAddr, tokenInfoTableGMTensor);
        // 计算tokenData在目标FFN卡peermem中的地址
        GM_ADDR tokenData = (__gm__ uint8_t*)(toRankAddr + winOffset_[1] + (winTokenDataOffset_ +
            (tokenId * (axisK_ + sharedExpertNum_) * axisHS_) + (topkId * axisHS_)) * sizeof(XType));
        tokenDataGMTensor.SetGlobalBuffer((__gm__ XType*)tokenData);
        // 将FP16/BF16 token数据写入目标FFN卡的peermem
        xQueue_.EnQue(xInTensor_);
        xInTensor_ = xQueue_.DeQue<XType>();
        DataCopyPad(tokenDataGMTensor, xInTensor_, xCopyParams);
        xQueue_.FreeTensor<XType>(xInTensor_);
    }

    // 将localExpId写入tokenInfoTable对应位置（偏移TOKEN_INFO_TABLE_RS=2，跳过flag和layerId）
    DataCopyPad(tokenInfoTableGMTensor[TOKEN_INFO_TABLE_RS + tokenOffset], statusTensor_[tokenIdx * STATUS_REP_STRIDE], statusCopyParams);
}

// SendTokenToFFN: 主发送函数，协调所有AIV核心将token发送到FFN侧
// 完整流程：
//   1. 搬运expertIds到UB（所有AIV核心共享同一份expertIds）
//   2. 同步模式下：初始化ffnStatusTensor_为全0（记录各FFN卡的发送状态）
//   3. SplitToCore: 将totalSendNum_个token均匀分配给各AIV核心
//   4. 分配statusTensor_的UB缓冲区（存放localExpId，每token占32B对齐）
//   5. 刷新expertRankTable的Cache Line（确保查表时数据一致性）
//   6. 循环调用SendTokenToFFNByTokenIdx发送每个token
//   7. 同步模式下：将ffnStatusTensor_写入workspace（供SetFlagToFFN聚合使用）
template <TemplateAttentionToFFNTypeClass>
__aicore__ inline void AttentionToFFN<TemplateAttentionToFFNTypeFunc>::SendTokenToFFN()
{
    uint32_t startId = 0;
    uint32_t endId = 0;
    // 计算总发送token数：X * 有效BS * (topK专家数 + 共享专家数)
    totalSendNum_ = axisX_ * curBsCnt_ * (axisK_ + sharedExpertNum_); // 总发送数：axisX_ * curBsCnt_ * (axisK_ + 1)
    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
    // 一次性将expertIds从GM搬入UB，后续SetExpertAndRank直接从UB读取
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, copyPadParams);
    // 计算每个AIV核心在workspace中的步长（128B对齐）
    uint32_t aivWorkspaceStride = aivWorkspaceOffset_ / sizeof(int32_t);
    if constexpr (isSync) {
        ffnNumAlignCnt_ = ffnNumAlignSize_ / sizeof(int32_t);
        syncStatusGMTensor_.SetGlobalBuffer((__gm__ int32_t*)syncStatusWorkspaceGM_);
        // 初始化FFN发送状态为全0（0=未发送，1=已发送）
        Duplicate<int32_t>(ffnStatusTensor_, static_cast<int32_t>(0), ffnNumAlignCnt_); // 初始化workSpace
    }

    // 将totalSendNum_个token均匀分配给各AIV核心
    SplitToCore(totalSendNum_, aivNum_, startId, endId, sendNum_);
    // 分配statusTensor_缓冲区：每token占32B（STATUS_REP_STRIDE * sizeof(int32_t)），最少1个token的空间
    uint32_t statusCnt = sendNum_ > 0 ? sendNum_ : 1;   // buf复用
    tpipe_->InitBuffer(statusBuf_, statusCnt * UB_ALIGN); // 对齐32B
 	statusTensor_ = statusBuf_.Get<int32_t>();

    // 边界检查：当前AIV核心无任务时，同步模式下仍需写入workspace（全0状态）
    if (startId >= totalSendNum_) {
        if constexpr (isSync) {
            SyncFunc<AscendC::HardEvent::V_MTE3>();
            DataCopy(syncStatusGMTensor_[aivId_ * aivWorkspaceStride], ffnStatusTensor_, ffnNumAlignCnt_);
        }
        return;
    }

    // 一次性刷新expertRankTable当前层的Cache Line
    // 每16个int32（64B）刷新一次Cache Line，确保后续FindExpertRank查表时数据一致性
    for (uint32_t idx = 0; idx < Ceil(expertRankTableCnt_, EXPERT_TABLE_REP_STRIDE); ++idx) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            expertRankTableGMTensor_[layIdsExpRankTableOffset_ + idx * EXPERT_TABLE_REP_STRIDE]);
    }

    // 逐token发送：每个token独立完成 搬运→查表→量化(可选)→写入peermem→设置flag
    for (uint32_t tokenIdx = 0; tokenIdx < sendNum_; ++tokenIdx) {
        SendTokenToFFNByTokenIdx(tokenIdx);
    }
    // 同步模式下：将本AIV核心的FFN发送状态写入workspace，供SetFlagToFFN聚合
    if constexpr (isSync) { //异步场景存储token发送状态到workspace
        DataCopy(syncStatusGMTensor_[aivId_ * aivWorkspaceStride], ffnStatusTensor_, ffnNumAlignCnt_);
 	}
}

// SetFlagToFFN: 向所有已发送token的FFN节点设置flag，通知FFN侧数据已就绪
// 流程：
//   1. SplitToCore: 将FFN节点分配给各AIV核心
//   2. 同步模式下：
//      a. 从workspace读取所有AIV核心的发送状态（syncStatusWorkspaceTensor_）
//      b. ReduceSum聚合：任一AIV核心向某FFN发送过token，则该FFN需要被通知
//      c. 根据聚合结果跳过未发送token的FFN节点
//   3. 遍历分配到的FFN节点，获取其通信窗口地址
//   4. 向tokenInfoTable写入[flag=1, layerId]（2个int32），通知FFN侧可以开始消费数据
// 注意：flag=1表示Attention侧本轮数据已全部写入，FFN侧可以开始读取
template <TemplateAttentionToFFNTypeClass>
__aicore__ inline void AttentionToFFN<TemplateAttentionToFFNTypeFunc>::SetFlagToFFN()
{
    // 同步所有FFN节点全发 异步只有选中的节点发送
    uint32_t startFFNId = 0;
    uint32_t endFFNId = 0;
    uint32_t sentFFNNum = 0;
    // 将FFN节点分配给各AIV核心
    SplitToCore(ffnNum_, aivNum_, startFFNId, endFFNId, sentFFNNum);
    // 准备要写入tokenInfoTable的数据：[flag=1, layerId]
    statusTensor_.SetValue(0, 1);
    statusTensor_.SetValue(1, layerId_);
    GlobalTensor<int32_t> tableFlagGMTensor;
    // 每次写入2个int32（flag + layerId）
    DataCopyExtParams statusParams = {1U, static_cast<uint32_t>(sizeof(int32_t) * TOKEN_INFO_TABLE_COPY_BLOCK_CNT), 0U, 0U, 0U};
    if (startFFNId >= ffnNum_) {
        return;
    }

    // 将FFN ID偏移到实际的rank ID（FFN节点从ffnStartRankId_开始编号）
    startFFNId += ffnStartRankId_;
    endFFNId += ffnStartRankId_;
    GM_ADDR toRankAddr;
    if constexpr (isSync) {
        // === 同步模式：聚合所有AIV核心的发送状态 ===
        // 从workspace读取所有AIV核心的FFN发送状态矩阵（aivNum_ × sentFFNNum）
        uint32_t sentFFNNumAlignSize = Ceil(sentFFNNum * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
        DataCopyExtParams syncStatusParams = {static_cast<uint16_t>(aivNum_), static_cast<uint32_t>(sentFFNNum * sizeof(int32_t)), 
                                              static_cast<uint32_t>(aivWorkspaceOffset_ - sentFFNNum * sizeof(int32_t)), 0U, 0U};
        DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
        tpipe_->InitBuffer(syncStatusWorkspaceBuf_, aivNum_ * sentFFNNumAlignSize);
        syncStatusWorkspaceTensor_ = syncStatusWorkspaceBuf_.Get<int32_t>();
        DataCopyPad(syncStatusWorkspaceTensor_, syncStatusGMTensor_[startFFNId - ffnStartRankId_], syncStatusParams, copyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        // ReduceSum沿AIV维度聚合：任一AIV核心向某FFN发送过token，则该FFN状态为1
        LocalTensor<float> syncStatusWorkspaceTensorFloat = syncStatusWorkspaceTensor_.ReinterpretCast<float>();
 	    LocalTensor<float> ffnStatusTensorFloat = ffnStatusTensor_.ReinterpretCast<float>();
        const uint32_t shape[] = {aivNum_, static_cast<uint32_t>(sentFFNNumAlignSize / sizeof(float))};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(ffnStatusTensorFloat, syncStatusWorkspaceTensorFloat, shape, true);
        SyncFunc<AscendC::HardEvent::V_S>();
    }

    // 遍历分配到的FFN节点，向每个已发送token的FFN设置flag
    for (uint32_t ffnIdx = startFFNId; ffnIdx < endFFNId; ++ffnIdx) {
        if constexpr (isSync) {
            // 同步模式：跳过未收到任何token的FFN节点
            if (ffnStatusTensor_.GetValue(ffnIdx - startFFNId) == 0) {
                continue;
            }
        }
        // 获取目标FFN rank的通信窗口地址
        if (winContext_->userMemType == 0) {
            toRankAddr = (GM_ADDR)(((HcclRankRelationResV2 *)(winContext_->remoteRes[ffnIdx].nextDevicePtr))->windowsIn);
        } else {
            toRankAddr = (GM_ADDR)(winContext_->userMemRes[ffnIdx].addr);
        }
        // 计算tokenInfoTable的GM地址
        GM_ADDR tableFlagGM = (__gm__ uint8_t*)(toRankAddr + winInfoTableOffset_);
        tableFlagGMTensor.SetGlobalBuffer((__gm__ int32_t*)tableFlagGM);
        if (ffnIdx == startFFNId) {
            // 第一个FFN节点前等待S_MTE3同步（确保statusTensor_数据就绪）
            SyncFunc<AscendC::HardEvent::S_MTE3>();
        }
        // 写入[flag=1, layerId]到tokenInfoTable头部，通知FFN侧数据已就绪
        DataCopyPad(tableFlagGMTensor, statusTensor_, statusParams);
    }
}

// SetFlagInAttn: 在本rank的tokenInfoTable中为被activeMask过滤掉的token设置flag=1
// activeMask场景下，BS中只有前curBsCnt_个token是有效的，后面的(BS-curBsCnt_)个是padding
// 但FFN侧期望收到完整的BS*(K+1)个flag位，因此需要将padding位置的flag也置为1
// 这样FFN侧不会因为缺少flag而无限等待
// 流程：
//   1. 计算被mask掉的token数：totalNum = X * (BS - curBsCnt) * (K + sharedExpertNum)
//   2. SplitToCore: 将这些padding token的flag设置任务分配给各AIV核心
//   3. 计算本rank的tokenInfoTable中padding区域的偏移（紧跟有效token之后）
//   4. 将padding位置的flag批量写为1（Duplicate + DataCopyPad）
template <TemplateAttentionToFFNTypeClass>
__aicore__ inline void AttentionToFFN<TemplateAttentionToFFNTypeFunc>::SetFlagInAttn()
{
    // 未mask数据总量：axisX_ * (axisBS_ - curBsCnt_) * (axisK_ + sharedExpertNum_)
    uint32_t startId = 0;
    uint32_t endId = 0;
    uint32_t sendNum = 0;
    // 计算被activeMask过滤掉的padding token总数
    uint32_t totalNum = axisX_ * (axisBS_ - curBsCnt_) * (axisK_ + sharedExpertNum_);
    if (totalNum == 0) {
        return;
    }
    // 将padding token的flag设置任务分配给各AIV核心
    SplitToCore(totalNum, aivNum_, startId, endId, sendNum);
    if (startId >= totalNum) {
        return;
    }

    // 计算本rank的tokenInfoTable中padding区域的起始偏移
    // sendMaskTokenCnt: 有效token数 × (K+1)，即有效token在tokenInfoTable中占用的位置数
    uint64_t sendMaskTokenCnt = curBsCnt_ * (axisK_ + sharedExpertNum_);
    // attnTokenInfoTableOffset: padding区域在tokenInfoTable中的字节偏移
    uint64_t attnTokenInfoTableOffset = (microBatchId_ * axisBS_ * (axisK_ + sharedExpertNum_) + sendMaskTokenCnt) * sizeof(int32_t);
    // 获取本rank的通信窗口地址（写入自己的tokenInfoTable）
    GM_ADDR selfRankAddr = (GM_ADDR)(winContext_->localWindowsIn);
    GM_ADDR attnTokenInfoTableGM = (__gm__ uint8_t*)(selfRankAddr + attnTokenInfoTableOffset);
    GlobalTensor<int32_t> attnTableGMTensor;
    attnTableGMTensor.SetGlobalBuffer((__gm__ int32_t*)attnTokenInfoTableGM);
    DataCopyExtParams dataCopyParams = {1U, static_cast<uint32_t>(sendNum * sizeof(int32_t)), 0U, 0U, 0U};
    // 将padding位置的flag全部设为1（Duplicate生成全1向量，DataCopyPad写入GM）
    LocalTensor<int32_t> tempTensor = attnStatusBuf_.Get<int32_t>();
    Duplicate<int32_t>(tempTensor, (uint32_t)1, sendNum);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(attnTableGMTensor[startId], tempTensor, dataCopyParams);
}

// Process: 算子主执行入口，仅在AIV核心上执行
// 执行流水线：
//   1. ActiveMaskCalCnt(可选): 计算activeMask中有效token数，更新curBsCnt_
//   2. SendTokenToFFN: 将token数据写入各FFN卡的peermem，设置tokenInfoTable中的localExpId
//   3. PipeBarrier + SyncAll: 全局同步，确保所有AIV核心完成token发送
//   4. SetFlagToFFN: 向所有已发送token的FFN节点设置flag=1，通知数据就绪
//   5. SetFlagInAttn(可选): 为activeMask过滤掉的padding token补写flag=1
template <TemplateAttentionToFFNTypeClass>
__aicore__ inline void AttentionToFFN<TemplateAttentionToFFNTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        if constexpr (isActiveMask) {
            ActiveMaskCalCnt();
        }
        SendTokenToFFN();
        PipeBarrier<PIPE_ALL>();
        SyncAll<true>();
        SetFlagToFFN();
        if constexpr (isActiveMask) { // activeMask attn将die状态区上的未mask的token flag位先写上
            SetFlagInAttn();
        }
    }
}
}
#endif