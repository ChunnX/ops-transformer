/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 */
#ifndef DISPATCH_FFN_COMBINE_CATLASS_COMBINE_H
#define DISPATCH_FFN_COMBINE_CATLASS_COMBINE_H

#include "kernel_operator.h"
#include "gmmsq_base.h"
#include "dispatch_ffn_combine_ops.h"

using namespace AscendC;

namespace DispatchFFNCombineImpl {

template <typename ElementC>
__aicore__ inline void CombineDispatchImpl(
    LocalTensor<ElementC>& l0cOutUb,
    GemmCoord& actualShape,
    uint32_t mLoc,
    uint32_t nLoc,
    const CombineOpsParams& params,
    GlobalTensor<int32_t>& tokenPerExpert,
    GlobalTensor<int32_t>& preSumBeforeRank,
    GlobalTensor<int32_t>& cumsumMM,
    uint32_t groupIdx)
{
    if constexpr(g_coreType != AscendC::AIV) {
        return;
    }
    
    // 参考kernel_gmm_swiglu_mxquant.h的GroupMatmul2的AIV部分
    uint32_t n = params.n / 2;  // GMM2的n是GMM1的一半
    
    // 等待cube完成
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    
    int32_t lenTile = actualShape.m();
    int32_t stTile = mLoc;
    int32_t edTile = stTile + lenTile;
    int32_t preSumRankInExpert = 0;
    int32_t tileOffset = 0;
    
    // 遍历所有目标EP
    for (int32_t dstEpIdx = 0; dstEpIdx < params.epWorldSize; dstEpIdx++) {
        // 获取当前rank在这个expert上的token数量
        // tokenPerExpert的布局: [EP, expertPerRank]
        uint32_t lenRankInExpert = tokenPerExpert.GetValue(dstEpIdx * params.moeExpertNum + groupIdx);
        
        // 获取目标expert的前缀和
        uint32_t dstExpertOffset = preSumBeforeRank.GetValue(dstEpIdx * params.moeExpertNum + groupIdx);
        
        uint32_t stRankInExpert = preSumRankInExpert;
        uint32_t edRankInExpert = stRankInExpert + lenRankInExpert;
        preSumRankInExpert += lenRankInExpert;
        
        // 检查当前tile是否与这个rank的数据有交集
        if (stRankInExpert >= edTile) {
            break;
        } else if (edRankInExpert <= stTile) {
            continue;
        }
        
        // 计算交集范围
        uint32_t stData = (stTile > stRankInExpert) ? stTile : stRankInExpert;
        uint32_t edData = (edTile < edRankInExpert) ? edTile : edRankInExpert;
        uint32_t lenData = edData - stData;
        
        if (lenData <= 0) {
            continue;
        }
        
        // 计算目标偏移
        uint32_t dstOffsetInExpert = 0;
        if (stTile > stRankInExpert) {
            dstOffsetInExpert = stTile - stRankInExpert;
        }
        
        // 获取远端peermem地址
        GM_ADDR dstPeermemPtr = params.peermemBuffer[dstEpIdx];
        
        // 计算远端偏移地址
        uint64_t gmDstOffset = (dstOffsetInExpert + dstExpertOffset) * n + nLoc;
        
        // 设置远端tensor
        GlobalTensor<ElementC> gmRemoteD;
        gmRemoteD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC*>(dstPeermemPtr));
        auto gmTileD = gmRemoteD[gmDstOffset];
        
        // 设置DataCopy参数
        AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
        ub2GmParams.blockCount = lenData;
        ub2GmParams.blockLen = actualShape.n() * sizeof(ElementC);
        ub2GmParams.dstStride = (n - actualShape.n()) * sizeof(ElementC);
        
        // 执行DataCopy到远端peermem
        AscendC::DataCopyPad(gmTileD, l0cOutUb[tileOffset * actualShape.n()], ub2GmParams);
        
        tileOffset += lenData;
    }
    
    // 设置完成标志
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
}

}  // namespace DispatchFFNCombineImpl

#endif  // DISPATCH_FFN_COMBINE_CATLASS_COMBINE_H
