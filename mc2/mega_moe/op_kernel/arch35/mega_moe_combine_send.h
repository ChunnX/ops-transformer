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
 * \file mega_moe_combine_send.h
 * \brief
 */

#ifndef MEGA_MOE_COMBINE_SEND_H
#define MEGA_MOE_COMBINE_SEND_H

#include "kernel_operator.h"
#include "mega_moe_base.h"
using namespace AscendC;

// ============================================================================
// MegaMoeCombineImpl - MoE Combine发送实现
// ============================================================================
// CombineTokens函数负责将GMM2矩阵乘法的输出结果按triple路由信息
// 逐token发送(Combine)到目标卡的peermem窗口, 实现MoE的跨卡结果合并。
//
// MoE Combine阶段的核心逻辑:
//   每个token经过GMM2计算后得到一行输出向量, 需要根据该token的
//   路由信息(toRank, tokenIdx, topkIdx)将其发送到对应目标卡的
//   peermem窗口中。目标卡最终会按(tokenIdx, topkIdx)位置聚合
//   所有源卡发来的结果, 完成MoE的加权合并。
//
// 数据流向:
//   UB(l0cOutUbGMM2) -> peermem(GM) 跨卡窗口写入
//
// 关键概念:
//   - peermem: 多卡间共享内存窗口, 每张卡可通过GetRankWinAddrWithOffset
//     访问任意目标卡的peermem区域, 实现跨卡直接写入
//   - triple信息: 每个token的路由元数据, 包含目标卡号/源token索引/top-K索引
//   - topK: MoE中每个token选择的专家数, 影响目标缓冲的布局
// ============================================================================
namespace MegaMoeCombineImpl {

// ============================================================================
// CombineTokens - 逐token将GMM2输出Combine发送到目标卡的peermem窗口
// ============================================================================
//
// 参数说明:
//   - ElementMMadOut2: GMM2矩阵乘输出的数据类型(如bfloat16_t)
//   - BlockShape: 当前tile的实际形状类型(来自BlockScheduler)
//   - mLoc: 当前tile在M方向(行/token)的起始偏移, 用于定位triple信息
//   - nLoc: 当前tile在N方向(列/隐藏维)的起始偏移, 用于定位输出列起始
//   - n: 输出矩阵的完整N维度(隐藏维度大小)
//   - tripleTensor: VECCALC上的triple路由信息局部缓冲, 每token8个int32:
//       [0] toRankId  - 目标卡号(Combine发送的目标rank)
//       [1] tokenIdx  - token在源卡中的原始索引
//       [2] topkIdx   - top-K选择的专家索引(0~topK-1)
//       [3-7] 预留/其他辅助信息
//   - l0cOutUbGMM2: UB上GMM2的矩阵乘输出LocalTensor, 每行对应一个token
//   - actualBlockShape: 当前tile的实际形状(尾部tile可能不满块)
//   - params: 全局参数, 包含peermem信息/tilingData等
//
// 执行流程:
//   1. 计算peermem中Combine发送区域的偏移量(gmRemoteBaseOffset)
//   2. 配置DataCopy参数: 每次搬运1行(1个token的完整N维输出)
//   3. 遍历当前tile的每个token:
//      a. 从tripleTensor读取(toRankId, tokenIdx, topkIdx)
//      b. 通过GetRankWinAddrWithOffset获取目标卡的peermem窗口地址
//      c. 计算目标偏移: (tokenIdx * topK + topkIdx) * n + nLoc
//         布局含义: 目标缓冲按[token][topK专家]排列, 每个位置存n维向量
//      d. DataCopyPad将当前token的输出行写入目标卡的peermem对应位置
// ============================================================================
template <typename ElementMMadOut2, typename BlockShape>
__aicore__ inline void CombineTokens(
    uint32_t mLoc, uint32_t nLoc, uint32_t n, LocalTensor<int32_t>& tripleTensor,
    LocalTensor<ElementMMadOut2>& l0cOutUbGMM2, BlockShape& actualBlockShape, const Params& params)
{
    // 当前tile的行数(token数), 即需要Combine发送的token数量
    // 尾部tile的lenTile可能小于标准tile大小(L1_TILE_M_128)
    int32_t lenTile = Get<M_VALUE>(actualBlockShape);

    // gmRemoteD: 目标卡的peermem全局张量, 用于跨卡写入
    // 每个token的目标卡不同, 需在循环中动态SetGlobalBuffer
    AscendC::GlobalTensor<ElementMMadOut2> gmRemoteD;

    // gmRemoteBaseOffset: Combine发送区域在peermem窗口中的偏移量
    // peermem窗口起始 = rankSyncInWorldPtr(各卡同步基址)
    // Combine发送区起始 = combineSendPtr(在PeermemInfo构造时计算)
    // 偏移量 = combineSendPtr - rankSyncInWorldPtr
    // GetRankWinAddrWithOffset(toRankId, offset) 返回:
    //   winRankAddr_[toRankId] + offset, 即目标卡的peermem窗口 + Combine区偏移
    uint64_t gmRemoteBaseOffset = params.peermemInfo.combineSendPtr - params.peermemInfo.rankSyncInWorldPtr;

    // DataCopyExtParams: UB->GM的数据搬运参数配置
    // blockCount=1: 每次搬运1个连续块(1行, 即1个token的完整N维输出)
    // blockLen: 单块长度 = actualN * sizeof(ElementMMadOut2) 字节
    //   即当前tile的N维实际长度(尾部tile的N可能不满L1_TILE_N)
    // 其他参数(stride等)为0, 表示数据连续无间隔
    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
    ub2GmParams.blockCount = 1;
    ub2GmParams.blockLen = Get<N_VALUE>(actualBlockShape) * sizeof(ElementMMadOut2);

    // MTE2_S事件同步: 确保triple信息从GM到VECCALC的搬运已完成
    // (在GroupMatmul2Combine中DataCopy(tripleTensor, tripleGm)之后设置)
    // 此处等待该事件, 保证tripleTensor中的路由数据可读
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);

    // ---- 逐token Combine发送主循环 ----
    // 遍历当前tile中的每个token, 根据其triple路由信息
    // 将GMM2输出行写入目标卡的peermem Combine区域
    for (int32_t tileIdx = 0; tileIdx < lenTile; ++tileIdx) {
        // 从tripleTensor读取当前token的路由元数据:
        // triple布局: 每个token占8个int32槽位
        //   [0] toRankId:  目标卡号, 决定数据发送到哪张卡
        //   [1] tokenIdx:  token在源卡中的原始索引, 用于定位目标缓冲的token行
        //   [2] topkIdx:   top-K专家索引(0~topK-1), 用于定位目标缓冲的专家列
        uint32_t toRankId = tripleTensor.GetValue(tileIdx * 8);
        uint32_t tokenIdx = tripleTensor.GetValue(tileIdx * 8 + 1);
        uint32_t topkIdx = tripleTensor.GetValue(tileIdx * 8 + 2);

        // 设置目标卡的GM全局缓冲地址:
        // GetRankWinAddrWithOffset(toRankId, gmRemoteBaseOffset) 返回:
        //   winRankAddr_[toRankId] + gmRemoteBaseOffset
        // 即: 目标卡toRankId的peermem窗口中Combine发送区域的起始地址
        // winRankAddr_[] 在HCCL初始化时填充, 存储各卡的peermem窗口基址
        gmRemoteD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(
            GetRankWinAddrWithOffset(toRankId, gmRemoteBaseOffset)));

        // 计算目标偏移量(gmDstOffset):
        // 目标缓冲布局: [tokenIdx][topkIdx][n维向量]
        //   - tokenIdx * topK: 该token在目标缓冲中的行级偏移(topK个专家位置)
        //   - topkIdx: 该token的第几个top-K专家(列级偏移)
        //   - * n: 每个位置存储完整的n维输出向量
        //   - + nLoc: 当前tile在N方向的起始列偏移(处理N维分块)
        //
        // 示例(topK=2, n=4096, nLoc=0):
        //   tokenIdx=3, topkIdx=0 -> offset = (3*2+0)*4096 = 第3个token的第1个专家位置
        //   tokenIdx=3, topkIdx=1 -> offset = (3*2+1)*4096 = 第3个token的第2个专家位置
        uint64_t gmDstOffset = (tokenIdx * params.tilingData->topK + topkIdx) * n + nLoc;

        // DataCopyPad: 将当前token的GMM2输出行从UB搬运到目标卡的peermem
        // 源: l0cOutUbGMM2[tileIdx * actualN], 即UB中第tileIdx个token的输出行起始
        // 目: gmRemoteD[gmDstOffset], 即目标卡peermem中该token的Combine位置
        // ub2GmParams: 搬运1行, 长度为actualN * sizeof(ElementMMadOut2)
        // Pad模式: 源数据不足blockLen时自动填充(Pad), 处理尾部tile不满块的情况
        AscendC::DataCopyPad(gmRemoteD[gmDstOffset],
            l0cOutUbGMM2[tileIdx * Get<N_VALUE>(actualBlockShape)], ub2GmParams);
    }
}
}  // namespace MegaMoeCombineImpl

#endif  // MEGA_MOE_COMBINE_SEND_H