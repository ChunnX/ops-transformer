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
 * \file attention_to_ffn_tiling.h
 * \brief
 */
#ifndef ATTENTION_TO_FFN_TILING_H
#define ATTENTION_TO_FFN_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct AttentionToFFNInfo {
    uint32_t X;                       // micro batch sequence size（token组数），当前版本只支持X=1
    uint32_t BS;                      // batch sequence size（本卡最终输出的token数量），范围(0, 512]
    uint32_t H;                       // hidden size隐藏层大小，范围[1024, 8192]
    uint32_t K;                       // 每个token选取的topK专家数量，范围(0, min(16, moeExpertNum)]
    uint32_t L;                       // 模型层数，当前版本只支持L=1
    uint32_t expertNum;               // 所有专家数 = moeExpertNum + sharedExpertNum
    uint32_t moeExpertNum;            // 路由专家（MoE专家）数量，范围(0, 1024]
    uint32_t sharedExpertNum;         // 共享专家数量，范围[0, 4]
    uint32_t HS;                      // ffnTokenDataShape的第5维，即单个token发送到FFN的数据长度
    uint32_t expRankTableM;           // expert_rank_table最后一维的长度，体现为部署在FFN节点上数量最多的专家部署信息列表的长度
    uint32_t microBatchNum;           // microBatch切分的数量，来自ffnTokenDataShape[1]
    uint32_t attentionWorkerNum;      // Attention节点数量，来自ffnTokenDataShape[0]
    uint32_t infoTableLastDimNum;     // tokenInfoTable最后一维的大小，来自ffnTokenInfoTableShape[2]
    uint32_t aivNum;                  // AI Vector核心总数，用于多核并行
    uint32_t worldSize;               // 通信域大小（总卡数），范围[2, 768]
    uint32_t syncFlag;                // 同步标志：0=同步模式，1=异步模式
    uint32_t quantMode;               // 量化模式：0=非量化，1=静态量化（暂不支持），2=动态量化
    uint32_t ffnStartRankId;          // FFN节点的起始rankId
    bool isScales;                    // 是否输入了smoothScales量化参数
    bool isActiveMask;                // 是否输入了activeMask激活掩码
    uint8_t windowType;               // 通信窗口类型，由环境变量ASCEND_ATTN_TO_FFN_WIN_TYPE控制
    uint64_t totalUbSize;             // 每个AIV核心的UB（Unified Buffer）总大小
    uint64_t totalWinSize;            // 通信窗口总大小（字节），由HCCL_BUFFSIZE环境变量决定
};

struct AttentionToFFNTilingData {
    Mc2InitTiling mc2InitTiling;      // MC2通信初始化Tiling参数，包含通信域基础配置
    Mc2CcTiling mc2CcTiling1;         // MC2集合通信Tiling参数，包含AlltoAll通信算法配置
    AttentionToFFNInfo attentionToFFNInfo; // 算子核心Tiling信息
};

#endif // ATTENTION_TO_FFN_TILING_H