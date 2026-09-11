/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_allv_quant_grouped_matmul_tiling_data.h
 * \brief AllToAllv + MX 量化 GroupedMatMul 融合算子 tiling 结构与 host 侧构建。
 *
 * 语义对齐 aclnnAlltoAllvQuantGroupedMatMul（MX 模式）：
 *   - gmm_x        [BSK, H1] 量化 token，按 (dstRank, expert) 有序；
 *   - gmm_x_scale  [BSK, ceildiv(H1,64), 2] E8M0 MX scale；
 *   - gmm_weight   本 rank 专家权重，transB=false: [E, H1, N1]，transB=true: [E, N1, H1]；
 *   - gmm_weight_scale transB=false: [E, ceildiv(H1,64), N1, 2]，transB=true: [E, N1, ceildiv(H1,64), 2]；
 *   - gmm_y        [sumRecv, N1]，行序为 expert-major（组内按 src rank 排列）。
 *
 * 通信采用“直接按 permute 后布局落窗”：发送端将 (dst, expert) 段直接写入
 * 接收窗的 expert-major 位置，免去通信后的独立 permute 步骤。
 */

#pragma once

#include <cstdint>
#include <algorithm>
#include "kernel_tiling/kernel_tiling.h" // Mc2InitTiling / Mc2CcTiling

// 表容量约束（与结构体固定数组配套）
constexpr uint32_t A2AV_GMM_MAX_RANK_NUM = 16;   // 最大 rank 数
constexpr uint32_t A2AV_GMM_MAX_EXPERT_NUM = 32; // 单 rank 最大专家数
// 远程发送单次 WriteNbi 的行分片上限对应的字节预算（1MB）
constexpr uint64_t A2AV_GMM_COMM_TILE_BYTES = 1024UL * 1024UL;

struct AllToAllvQuantGmmTilingData {
    // 问题维度
    uint32_t rankSize{0};
    uint32_t rankId{0};
    uint32_t expertNum{0}; // 本 rank 专家数 E
    uint32_t k{0};         // H1
    uint32_t n{0};         // N1
    uint32_t transB{0};    // 0: weight [E,K,N]；1: weight [E,N,K]
    uint32_t scaleKLen{0}; // ceildiv(K,64) * 2，A scale 每行字节数
    uint64_t winScaleOffset{0}; // 接收窗中 scaleA 区字节偏移 = totalRecvRows * K
    uint64_t dstWinScaleOffset[A2AV_GMM_MAX_RANK_NUM]; // 各目的 rank 窗口的 scaleA 区偏移
    uint32_t totalRecvRows{0};

    // 接收计划：本 rank 窗口内 (src, expert) 段的行数与落窗行号（expert-major）
    uint32_t recvCnt[A2AV_GMM_MAX_RANK_NUM][A2AV_GMM_MAX_EXPERT_NUM];
    uint32_t recvDstRow[A2AV_GMM_MAX_RANK_NUM][A2AV_GMM_MAX_EXPERT_NUM];

    // 发送计划：本 rank 发往 (dst, expert) 的行数 / gmm_x 内源行号 / 对端窗口目标行号
    uint32_t sendCnt[A2AV_GMM_MAX_RANK_NUM][A2AV_GMM_MAX_EXPERT_NUM];
    uint32_t sendSrcRow[A2AV_GMM_MAX_RANK_NUM][A2AV_GMM_MAX_EXPERT_NUM];
    uint32_t sendDstRow[A2AV_GMM_MAX_RANK_NUM][A2AV_GMM_MAX_EXPERT_NUM];

    // GMM 分组列表（expert 累计行数）与计算 tiling
    uint32_t groupListCum[A2AV_GMM_MAX_EXPERT_NUM];
    uint32_t usedCoreNum{0};
    uint32_t baseM{0};
    uint32_t baseN{0};
    uint32_t baseK{0};
    uint32_t kAL1{0};
    uint32_t kBL1{0};
    uint32_t scaleKAL1{0};
};

// host 侧 tiling 构建（ST launcher 在 -xasc 模式下同样以 host 代码编译执行）
/*!
 * \brief host 侧 tiling 构建。
 *
 * \param[in]  rankId       本 rank id
 * \param[in]  rankSize     world size
 * \param[in]  expertNum    单 rank 专家数 E
 * \param[in]  k            H1
 * \param[in]  n            N1
 * \param[in]  transB       权重是否转置（[E,N,K] 存放）
 * \param[in]  expTokenNums 全量计数矩阵，布局 [rankSize][rankSize*E]，
 *                          expTokenNums[src * rankSize * E + dst * E + e] 为 src 发往 (dst, e) 的 token 数
 * \param[in]  usedCoreNum  计划使用的 AIC 核数（须 >= rankSize）
 * \param[out] tiling       输出 tiling
 */
inline void BuildAllToAllvQuantGmmTiling(uint32_t rankId, uint32_t rankSize, uint32_t expertNum, uint32_t k,
                                         uint32_t n, bool transB, const uint32_t *expTokenNums, uint32_t usedCoreNum,
                                         AllToAllvQuantGmmTilingData &tiling)
{
    tiling = {};
    tiling.rankId = rankId;
    tiling.rankSize = rankSize;
    tiling.expertNum = expertNum;
    tiling.k = k;
    tiling.n = n;
    tiling.transB = transB ? 1U : 0U;
    tiling.scaleKLen = static_cast<uint32_t>(((k + 63U) / 64U) * 2U);
    tiling.usedCoreNum = std::max(usedCoreNum, rankSize);

    const uint32_t stride = rankSize * expertNum; // expTokenNums 行宽
    auto expAt = [expTokenNums, stride, expertNum](uint32_t s, uint32_t d, uint32_t e) -> uint32_t {
        return expTokenNums[s * stride + d * expertNum + e];
    };

    // ---- 接收侧：本 rank 窗口 expert-major 布局 ----
    // groupStart(e) = sum_{e'<e} sum_s exp(s, rankId, e')
    // recvDstRow(s, e) = groupStart(e) + sum_{s'<s} exp(s', rankId, e)
    uint32_t groupStart[A2AV_GMM_MAX_EXPERT_NUM] = {0};
    uint32_t cum = 0;
    for (uint32_t e = 0; e < expertNum; ++e) {
        groupStart[e] = cum;
        for (uint32_t s = 0; s < rankSize; ++s) {
            cum += expAt(s, rankId, e);
        }
    }
    for (uint32_t e = 0; e < expertNum; ++e) {
        uint32_t prefix = 0;
        for (uint32_t s = 0; s < rankSize; ++s) {
            tiling.recvCnt[s][e] = expAt(s, rankId, e);
            tiling.recvDstRow[s][e] = groupStart[e] + prefix;
            prefix += expAt(s, rankId, e);
        }
        tiling.groupListCum[e] = groupStart[e];
        for (uint32_t s = 0; s < rankSize; ++s) {
            tiling.groupListCum[e] += expAt(s, rankId, e);
        }
    }
    tiling.totalRecvRows = cum;
    tiling.winScaleOffset = static_cast<uint64_t>(cum) * k;

    // 各目的 rank 窗口的 scaleA 区偏移（发送端写入对端 scale 区时使用对端偏移）
    for (uint32_t d = 0; d < rankSize; ++d) {
        uint32_t totalRecvD = 0;
        for (uint32_t s = 0; s < rankSize; ++s) {
            for (uint32_t ee = 0; ee < expertNum; ++ee) {
                totalRecvD += expAt(s, d, ee);
            }
        }
        tiling.dstWinScaleOffset[d] = static_cast<uint64_t>(totalRecvD) * k;
    }

    // ---- 发送侧：本 rank 发往 (dst, e) ----
    // gmm_x 内行序为 (dst-major, expert 内)：sendSrcRow(d, e) = sendRowBase(d) + sum_{e'<e} exp(rankId, d, e')
    // 对端窗口落点：sendDstRow(d, e) = groupStart_d(e) + sum_{s'<rankId} exp(s', d, e)
    for (uint32_t d = 0; d < rankSize; ++d) {
        uint32_t rowBase = 0;
        for (uint32_t dd = 0; dd < d; ++dd) {
            for (uint32_t e = 0; e < expertNum; ++e) {
                rowBase += expAt(rankId, dd, e);
            }
        }
        uint32_t prefixE = 0;
        for (uint32_t e = 0; e < expertNum; ++e) {
            tiling.sendCnt[d][e] = expAt(rankId, d, e);
            tiling.sendSrcRow[d][e] = rowBase + prefixE;
            prefixE += expAt(rankId, d, e);
        }

        // 对端 d 的 expert-major 组起点与 src 前缀
        uint32_t running = 0;
        for (uint32_t ee = 0; ee < expertNum; ++ee) {
            uint32_t gStart = running;
            for (uint32_t s = 0; s < rankSize; ++s) {
                running += expAt(s, d, ee);
            }
            uint32_t srcPrefix = 0;
            for (uint32_t s = 0; s < rankId; ++s) {
                srcPrefix += expAt(s, d, ee);
            }
            tiling.sendDstRow[d][ee] = gStart + srcPrefix;
        }
    }

    // ---- GMM 计算 tiling（dav-3510 保守取值，L1 512KB / L0A,L0B 64KB / L0C 256KB）----
    auto ceilAlign = [](uint32_t v, uint32_t align) -> uint32_t {
        return (v + align - 1U) / align * align;
    };
    uint32_t maxGroupRows = 1;
    for (uint32_t e = 0; e < expertNum; ++e) {
        maxGroupRows = std::max(maxGroupRows, tiling.groupListCum[e]);
    }
    tiling.baseK = std::min(k, 128U);
    tiling.baseK = ceilAlign(tiling.baseK, 64U); // MX mmad 要求 baseK 对齐 64
    // L0A/L0B pingpong 半区约束：baseM * baseK <= 32KB 且 baseN * baseK <= 32KB
    const uint32_t maxByL0 = std::max(16U, 32768U / tiling.baseK);
    tiling.baseM = std::min(std::min(maxGroupRows, 256U), maxByL0);
    tiling.baseM = ceilAlign(tiling.baseM, 16U); // CUBE_BLOCK
    if (tiling.baseM > maxByL0) {
        tiling.baseM = ceilAlign(maxByL0, 16U);
    }
    tiling.baseN = std::min(std::min(n, 256U), maxByL0);
    // L0C 双缓冲约束：baseM * baseN * 4B <= 128KB（先按对齐粒度向下取整，防止 ceilAlign 回推超限）
    const uint32_t nAlign = transB ? 16U : 32U;
    uint32_t maxNByL0c = (std::max<uint32_t>(nAlign, 32768U / tiling.baseM) / nAlign) * nAlign;
    if (tiling.baseN > maxNByL0c) {
        tiling.baseN = maxNByL0c;
    }
    tiling.baseN = ceilAlign(tiling.baseN, nAlign);
    if (tiling.baseN > maxNByL0c) {
        tiling.baseN = maxNByL0c;
    }
    // L1 半区预算（A + B + scale）：约 220KB
    const uint64_t l1Budget = 220UL * 1024UL;
    uint64_t kAL1 = std::min<uint64_t>(k, 512U);
    const uint64_t perK = static_cast<uint64_t>(tiling.baseM) + tiling.baseN +
                          (static_cast<uint64_t>(tiling.baseM) + tiling.baseN) / 32U;
    while (kAL1 > 64U && kAL1 * perK > l1Budget) {
        kAL1 -= 64U;
    }
    tiling.kAL1 = static_cast<uint32_t>(ceilAlign(static_cast<uint32_t>(kAL1), 64U));
    tiling.kBL1 = tiling.kAL1;
    tiling.scaleKAL1 = tiling.kAL1; // MX 要求 scaleKAL1 == scaleKBL1
}

// =====================================================================================
// hcomm（HCCL 集合通信）版本：供算子 op_host/op_kernel 集成使用
//
// 与 UDMA 直写版本不同，本版本通过 Hccl<>::AlltoAllV 完成变长通信（AIV0 下发），
// 通信通过 recv offset 直接输出为 expert-major 布局；GMM 按“段式分组”消费：
// 每个 (expert, srcRank) 段是一个独立计算分组（段内 A/scaleA/C 连续），
// 免去通信后的独立 permute 步骤。
// =====================================================================================

namespace Apace {

// 段数上限：W*E（与算子侧 MAX_EXPERT_SIZE 对齐）
constexpr uint32_t A2AV_GMM_MAX_SEG_NUM = 256;
// HCCL device API rank-parameter ABI capacity.  This is intentionally larger
// than A2AV_GMM_MAX_RANK_NUM, which is the current fixed tiling-table capacity.
constexpr uint32_t A2AV_GMM_HCCL_PARAM_RANK_NUM = 128;
static_assert(A2AV_GMM_MAX_RANK_NUM <= A2AV_GMM_HCCL_PARAM_RANK_NUM,
              "APACE rank capacity must fit the HCCL parameter ABI");
// On-device binary-search stage: run the first expert's scale and data
// collectives sequentially, without Blaze compute.
constexpr bool A2AV_GMM_DIAG_FIRST_EXPERT_SERIAL_COMM = true;
/*!
 * \brief hcomm 版 tiling 数据（host 构建后随 kernel tiling buffer 下发）
 */
struct hcommA2avQuantGmmTilingData {
    Mc2InitTiling mc2InitTiling; // 复用算子侧 SetHcclTiling 结果
    Mc2CcTiling mc2CcTiling;
    // 问题维度
    uint32_t rankSize{0};
    uint32_t expertNum{0}; // 本 rank 专家数 E
    uint32_t k{0};         // H1（元素计）
    uint32_t n{0};         // N1
    uint32_t transB{0};
    uint32_t scaleKLen{0}; // ceildiv(K,64)*2
    uint32_t isMxFp4{0};
    uint32_t segNum{0};      // 段数 = W*E
    uint32_t usedCoreNum{0}; // AIC 核数
    uint32_t hasPermuteOut{0};
    // workspace 内通信缓冲偏移（字节）
    uint64_t dataRecvOffset{0};  // 数据接收区
    uint64_t scaleRecvOffset{0}; // scale 接收区
    // AlltoAllV 通信参数（元素计，axis 已乘入；fp4 场景数据 axis 为打包字节数）
    uint64_t dataSendCnt[A2AV_GMM_MAX_RANK_NUM]{0};
    uint64_t dataSendOff[A2AV_GMM_MAX_RANK_NUM]{0};
    uint64_t dataRecvCnt[A2AV_GMM_MAX_RANK_NUM]{0};
    uint64_t dataRecvOff[A2AV_GMM_MAX_RANK_NUM]{0};
    uint64_t scaleSendCnt[A2AV_GMM_MAX_RANK_NUM]{0};
    uint64_t scaleSendOff[A2AV_GMM_MAX_RANK_NUM]{0};
    uint64_t scaleRecvCnt[A2AV_GMM_MAX_RANK_NUM]{0};
    uint64_t scaleRecvOff[A2AV_GMM_MAX_RANK_NUM]{0};
    // 段表（expert-major：segIdx = e*W + s）
    uint32_t segList[A2AV_GMM_MAX_SEG_NUM + 1]{0}; // 段累计行数（kernel 分组列表）
    uint32_t segAOff[A2AV_GMM_MAX_SEG_NUM]{0};     // 段在数据/scale 接收区的行偏移
    uint32_t segCOff[A2AV_GMM_MAX_SEG_NUM]{0};     // 段在 gmm_y 输出的行偏移
    uint32_t segSendRows[A2AV_GMM_MAX_SEG_NUM]{0}; // expert-major: [expert][dst rank]
    // GMM 计算 tiling
    uint32_t baseM{0};
    uint32_t baseN{0};
    uint32_t baseK{0};
    uint32_t kAL1{0};
    uint32_t kBL1{0};
    uint32_t scaleKAL1{0};
    // optional shared-expert local MX matmul
    uint32_t localMatmul{0};
    uint32_t mmM{0};
    uint32_t mmK{0};
    uint32_t mmN{0};
    uint32_t mmBaseM{0};
    uint32_t mmBaseN{0};
    uint32_t mmBaseK{0};
    uint32_t mmKAL1{0};
    uint32_t mmKBL1{0};
    uint32_t mmScaleKAL1{0};
    uint32_t mmGroupList[1]{0};
};

/*!
 * \brief host 侧 hcomm 版 tiling 构建。
 *
 * \param[in]  rankSize       world size W
 * \param[in]  expertNum      本 rank 专家数 E
 * \param[in]  k              H1（元素计）
 * \param[in]  n              N1
 * \param[in]  transB         权重是否转置（[E,N,K] 存放）
 * \param[in]  isMxFp4        量化数据是否 fp4（打包存储）
 * \param[in]  sendCounts     本 rank 发送计数，布局 [dst][e]：sendCounts[dst*E + e]
 * \param[in]  recvCounts     本 rank 接收计数，布局 [src][e]：recvCounts[src*E + e]
 * \param[in]  usedCoreNum    AIC 核数
 * \param[in]  dataRecvOffset workspace 内数据接收区偏移（字节）
 * \param[in]  scaleRecvOffset workspace 内 scale 接收区偏移（字节）
 * \param[out] tiling         输出（mc2InitTiling/mc2CcTiling 由调用方另行填充）
 */
inline void BuildHcommA2avQuantGmmTiling(uint32_t rankSize, uint32_t expertNum, uint32_t k, uint32_t n, bool transB,
                                         bool isMxFp4, const int64_t *sendCounts, const int64_t *recvCounts,
                                         uint32_t usedCoreNum, uint64_t dataRecvOffset, uint64_t scaleRecvOffset,
                                         bool hasPermuteOut, bool localMatmul, uint32_t mmM, uint32_t mmK, uint32_t mmN,
                                         bool transMmB,
                                         hcommA2avQuantGmmTilingData &tiling)
{
    tiling = {};
    tiling.rankSize = rankSize;
    tiling.expertNum = expertNum;
    tiling.k = k;
    tiling.n = n;
    tiling.transB = transB ? 1U : 0U;
    tiling.isMxFp4 = isMxFp4 ? 1U : 0U;
    tiling.usedCoreNum = usedCoreNum;
    tiling.dataRecvOffset = dataRecvOffset;
    tiling.scaleRecvOffset = scaleRecvOffset;
    tiling.hasPermuteOut = hasPermuteOut ? 1U : 0U;
    tiling.segNum = rankSize * expertNum;
    tiling.scaleKLen = static_cast<uint32_t>(((k + 63U) / 64U) * 2U);
    tiling.localMatmul = localMatmul ? 1U : 0U;
    tiling.mmM = mmM;
    tiling.mmK = mmK;
    tiling.mmN = mmN;
    tiling.mmGroupList[0] = mmM;

    auto sendAt = [sendCounts, expertNum](uint32_t d, uint32_t e) -> int64_t {
        return sendCounts[d * expertNum + e];
    };
    auto recvAt = [recvCounts, expertNum](uint32_t s, uint32_t e) -> int64_t {
        return recvCounts[s * expertNum + e];
    };

    // ---- AlltoAllV 通信参数（axis 乘入，元素计）----
    const uint64_t dataAxis = isMxFp4 ? static_cast<uint64_t>((k + 1U) / 2U) : static_cast<uint64_t>(k);
    const uint64_t scaleAxis = static_cast<uint64_t>(tiling.scaleKLen);
    for (uint32_t i = 0; i < rankSize; ++i) {
        uint64_t sendRows = 0;
        uint64_t recvRows = 0;
        for (uint32_t e = 0; e < expertNum; ++e) {
            sendRows += static_cast<uint64_t>(sendAt(i, e));
            recvRows += static_cast<uint64_t>(recvAt(i, e));
        }
        tiling.dataSendCnt[i] = sendRows * dataAxis;
        tiling.dataRecvCnt[i] = recvRows * dataAxis;
        tiling.scaleSendCnt[i] = sendRows * scaleAxis;
        tiling.scaleRecvCnt[i] = recvRows * scaleAxis;
    }
    tiling.dataSendOff[0] = 0;
    tiling.dataRecvOff[0] = 0;
    tiling.scaleSendOff[0] = 0;
    tiling.scaleRecvOff[0] = 0;
    for (uint32_t i = 1; i < rankSize; ++i) {
        tiling.dataSendOff[i] = tiling.dataSendOff[i - 1] + tiling.dataSendCnt[i - 1];
        tiling.dataRecvOff[i] = tiling.dataRecvOff[i - 1] + tiling.dataRecvCnt[i - 1];
        tiling.scaleSendOff[i] = tiling.scaleSendOff[i - 1] + tiling.scaleSendCnt[i - 1];
        tiling.scaleRecvOff[i] = tiling.scaleRecvOff[i - 1] + tiling.scaleRecvCnt[i - 1];
    }

    // ---- 段表（expert-major：segIdx = e*W + s）----
    // 接收区 rank-major：src s 区起始行 = sum_{s'<s} sum_e recvCnt(s', e)
    // 组 e 的段 (s) 数据行偏移 = src 区起始行 + sum_{e'<e} recvCnt(s, e')
    // 输出 expert-major：expert e 输出区起始行 = sum_{e'<e} sum_s recvCnt(s, e')
    //                   段 (e, s) 输出行偏移 = expert 起始 + sum_{s'<s} recvCnt(s', e)
    uint32_t srcStart[A2AV_GMM_MAX_RANK_NUM] = {0};
    uint32_t expertStart[A2AV_GMM_MAX_SEG_NUM] = {0};
    {
        uint32_t running = 0;
        for (uint32_t s = 0; s < rankSize; ++s) {
            srcStart[s] = running;
            for (uint32_t e = 0; e < expertNum; ++e) {
                running += static_cast<uint32_t>(recvAt(s, e));
            }
        }
    }
    {
        uint32_t running = 0;
        for (uint32_t e = 0; e < expertNum; ++e) {
            expertStart[e] = running;
            for (uint32_t s = 0; s < rankSize; ++s) {
                running += static_cast<uint32_t>(recvAt(s, e));
            }
        }
    }
    uint32_t segCum = 0;
    for (uint32_t e = 0; e < expertNum; ++e) {
        uint32_t srcPrefix = 0; // sum_{s'<s} recvCnt(s', e)，输出侧行偏移
        for (uint32_t s = 0; s < rankSize; ++s) {
            const uint32_t segIdx = e * rankSize + s;
            const uint32_t rows = static_cast<uint32_t>(recvAt(s, e));
            tiling.segSendRows[segIdx] = static_cast<uint32_t>(sendAt(s, e));
            // 数据/scale 段行偏移（行数相同，scale 每行 scaleKLen 字节）
            uint32_t ePrefix = 0; // sum_{e'<e} recvCnt(s, e')
            for (uint32_t e2 = 0; e2 < e; ++e2) {
                ePrefix += static_cast<uint32_t>(recvAt(s, e2));
            }
            tiling.segAOff[segIdx] = srcStart[s] + ePrefix;
            tiling.segCOff[segIdx] = expertStart[e] + srcPrefix;
            srcPrefix += rows;
            segCum += rows;
            // segList 为包含式累计（list[i] = 前 i+1 段总行数），与 kernel GetGroupRows 语义一致
            tiling.segList[segIdx] = segCum;
        }
    }

    // ---- GMM 计算 tiling（dav-3510 保守取值，L1 512KB / L0A,L0B 64KB / L0C 256KB）----
    auto ceilAlign = [](uint32_t v, uint32_t align) -> uint32_t {
        return (v + align - 1U) / align * align;
    };
    uint32_t maxSegRows = 1;
    for (uint32_t seg = 0; seg < tiling.segNum; ++seg) {
        // segList is inclusive: entry seg is the cumulative row count through
        // that segment.  Do not read segList[segNum], which is not populated.
        const uint32_t prevRows = (seg == 0U) ? 0U : tiling.segList[seg - 1U];
        const uint32_t rows = tiling.segList[seg] - prevRows;
        maxSegRows = std::max(maxSegRows, rows);
    }
    tiling.baseK = std::min(k, 128U);
    tiling.baseK = ceilAlign(tiling.baseK, 64U); // MX mmad 要求 baseK 对齐 64
    // L0A/L0B pingpong 半区约束：baseM * baseK <= 32KB 且 baseN * baseK <= 32KB
    const uint32_t maxByL0 = std::max(16U, 32768U / tiling.baseK);
    tiling.baseM = std::min(std::min(maxSegRows, 256U), maxByL0);
    tiling.baseM = ceilAlign(tiling.baseM, 16U); // CUBE_BLOCK
    if (tiling.baseM > maxByL0) {
        tiling.baseM = ceilAlign(maxByL0, 16U);
    }
    tiling.baseN = std::min(std::min(n, 256U), maxByL0);
    // L0C 双缓冲约束：baseM * baseN * 4B <= 128KB（先按对齐粒度向下取整，防止 ceilAlign 回推超限）
    const uint32_t nAlign = transB ? 16U : 32U;
    uint32_t maxNByL0c = (std::max<uint32_t>(nAlign, 32768U / tiling.baseM) / nAlign) * nAlign;
    if (tiling.baseN > maxNByL0c) {
        tiling.baseN = maxNByL0c;
    }
    tiling.baseN = ceilAlign(tiling.baseN, nAlign);
    if (tiling.baseN > maxNByL0c) {
        tiling.baseN = maxNByL0c;
    }
    // L1 半区预算（A + B + scale）：约 220KB
    const uint64_t l1Budget = 220UL * 1024UL;
    uint64_t kAL1 = std::min<uint64_t>(k, 512U);
    const uint64_t perK = static_cast<uint64_t>(tiling.baseM) + tiling.baseN +
                          (static_cast<uint64_t>(tiling.baseM) + tiling.baseN) / 32U;
    while (kAL1 > 64U && kAL1 * perK > l1Budget) {
        kAL1 -= 64U;
    }
    tiling.kAL1 = static_cast<uint32_t>(ceilAlign(static_cast<uint32_t>(kAL1), 64U));
    tiling.kBL1 = tiling.kAL1;
    tiling.scaleKAL1 = tiling.kAL1; // MX 要求 scaleKAL1 == scaleKBL1

    if (localMatmul) {
        tiling.mmBaseK = ceilAlign(std::min(mmK, 128U), 64U);
        const uint32_t mmMaxByL0 = std::max(16U, 32768U / tiling.mmBaseK);
        tiling.mmBaseM = ceilAlign(std::min(std::min(mmM, 256U), mmMaxByL0), 16U);
        const uint32_t mmNAlign = transMmB ? 16U : 32U;
        tiling.mmBaseN = ceilAlign(std::min(std::min(mmN, 256U), mmMaxByL0), mmNAlign);
        const uint32_t mmMaxNByL0c =
            (std::max<uint32_t>(mmNAlign, 32768U / tiling.mmBaseM) / mmNAlign) * mmNAlign;
        tiling.mmBaseN = std::min(tiling.mmBaseN, mmMaxNByL0c);
        uint64_t mmKAL1 = std::min<uint64_t>(mmK, 512U);
        const uint64_t mmPerK = static_cast<uint64_t>(tiling.mmBaseM) + tiling.mmBaseN +
                                (static_cast<uint64_t>(tiling.mmBaseM) + tiling.mmBaseN) / 32U;
        while (mmKAL1 > 64U && mmKAL1 * mmPerK > l1Budget) {
            mmKAL1 -= 64U;
        }
        tiling.mmKAL1 = ceilAlign(static_cast<uint32_t>(mmKAL1), 64U);
        tiling.mmKBL1 = tiling.mmKAL1;
        tiling.mmScaleKAL1 = tiling.mmKAL1;
    }
}

} // namespace Apace
