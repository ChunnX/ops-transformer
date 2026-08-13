#ifndef FLASH_ATTN_DROPMASK_H_
#define FLASH_ATTN_DROPMASK_H_

#include "flash_attn_common_def.h"

#if __has_include("../../../common/op_kernel/arch35/dropmask.h")
#include "../../../common/op_kernel/arch35/dropmask.h"
#else
#include "../../common/arch35/dropmask.h"
#endif

namespace BaseApi {

template <FA_LAYOUT LAYOUT_T, uint32_t M_BASE_SIZE, uint32_t S2_BASE_SIZE>
__aicore__ inline uint64_t ComputeDropOffsetFa(const RunInfoX &runInfo,
                                               const ConstInfo_t<FiaKernelType::NO_QUANT> &constInfo,
                                               uint64_t baseOffset)
{
    constexpr uint32_t philoxAlign = 16;
    uint64_t s2SizeAligned = CeilDiv(runInfo.actS2Size, philoxAlign) * philoxAlign;

    uint64_t bOffset = runInfo.bIdx * constInfo.n2Size * constInfo.gSize * runInfo.actS1Size * s2SizeAligned;
    uint64_t n2Offset = runInfo.n2Idx * constInfo.gSize * runInfo.actS1Size * s2SizeAligned;
    uint64_t gs1Offset = (runInfo.gS1Idx + runInfo.vecMbaseIdx) * s2SizeAligned;
    uint64_t s2Offset = runInfo.s2Idx;

    return bOffset + n2Offset + gs1Offset + s2Offset + baseOffset;
}

template <FA_LAYOUT LAYOUT_T, uint32_t M_BASE_SIZE, uint32_t S2_BASE_SIZE>
__aicore__ inline void GenDropMaskFa(LocalTensor<uint8_t> &dropMaskUb,
                                     LocalTensor<int32_t> &dropmaskIndexVec,
                                     const RunInfoX &runInfo,
                                     const ConstInfo_t<FiaKernelType::NO_QUANT> &constInfo)
{
    uint64_t maskOffset = ComputeDropOffsetFa<LAYOUT_T, M_BASE_SIZE, S2_BASE_SIZE>(runInfo, constInfo, constInfo.offset);
    uint64_t dropMaskOffset = CeilDiv(maskOffset, 16);
    uint32_t eachRowCount = CeilDiv(CeilDiv(S2_BASE_SIZE, 4), 4) * 4;
    uint32_t eachRowIndexNum = CeilDiv(eachRowCount, 4);
    uint32_t eachRowOffset = CeilDiv(runInfo.actS2Size, 16);

    bool noTailBlock = (LAYOUT_T != FA_LAYOUT::TND &&
                        runInfo.actS1Size % M_BASE_SIZE == 0 && runInfo.actS2Size % S2_BASE_SIZE == 0);
    if (noTailBlock) {
        if (runInfo.isFirstS2Loop) {
            GenIndexVec(dropmaskIndexVec, runInfo.actVecMSize, eachRowIndexNum, eachRowOffset);
        }
    } else if (runInfo.isFirstS2Loop || runInfo.isLastS2Loop) {
        GenIndexVec(dropmaskIndexVec, runInfo.actVecMSize, eachRowIndexNum, eachRowOffset);
    }

    uint64_t seed = constInfo.seed;
    uint32_t seedHigh = static_cast<uint32_t>(seed >> 32);
    uint32_t seedLow = static_cast<uint32_t>(seed & 0xffffffff);
    uint32_t offsetHigh = static_cast<uint32_t>(dropMaskOffset >> 32);
    uint32_t offsetLow = static_cast<uint32_t>(dropMaskOffset & 0xffffffff);
    GenMaskByIndexVec(dropMaskUb, dropmaskIndexVec, {seedLow, seedHigh},
                      {offsetLow, offsetHigh, 0, 0},
                      runInfo.actVecMSize * eachRowCount, constInfo.keepProbUint8);
}

} // namespace BaseApi

#endif // FLASH_ATTN_DROPMASK_H_
