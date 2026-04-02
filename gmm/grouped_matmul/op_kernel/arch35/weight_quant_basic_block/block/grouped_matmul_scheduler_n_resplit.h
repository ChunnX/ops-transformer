#ifndef GROUPED_MATMUL_SCHEDULER_N_RESPLIT_H
#define GROUPED_MATMUL_SCHEDULER_N_RESPLIT_H

#include "../prologue/tool.h"

namespace Block {

class GroupedMatmulSchedulerNResplit {
public:
    struct Params {
        uint64_t mainBlockCount;
        uint64_t mainBlockSize;
        uint64_t firstTailBlockCount;
        uint64_t firstTailBlockSize;
        uint64_t secondTailBlockCount;
        uint64_t secondTailBlockSize;
        uint64_t coreNum;
        uint64_t cubeNumBlocksN;
        int32_t baseM;
        uint64_t nSize;
    };

    __aicore__ inline explicit GroupedMatmulSchedulerNResplit(const Params &params) : params_(params)
    {
    }

    template <typename Handler>
    __aicore__ inline void operator()(uint64_t &startBasicBlockId, uint64_t mSize, Handler &handler) const
    {
        uint64_t mBlkNum =
            WeightQuantBatchMatmulV2::Arch35::CeilDivide(mSize, static_cast<uint64_t>(params_.baseM));
        uint64_t mL1Step = WeightQuantBatchMatmulV2::Arch35::CeilDivide(mSize, mBlkNum);
        uint64_t cubeBlockIdx = AscendC::GetBlockIdx();
        if ASCEND_IS_AIV {
            cubeBlockIdx = cubeBlockIdx >> 1;
        }
        uint64_t curBasicBlockId =
            cubeBlockIdx >= startBasicBlockId ? cubeBlockIdx : cubeBlockIdx + params_.coreNum;
        uint64_t basicBlockLimit = startBasicBlockId;
        for (uint64_t mOffset = 0; mOffset < mSize; mOffset += mL1Step) {
            uint64_t mL1Size = mOffset + mL1Step > mSize ? mSize - mOffset : mL1Step;
            handler.OnM(mOffset, mL1Size);

            uint64_t nBaseOffset = 0;
            RunSegment(curBasicBlockId, basicBlockLimit, nBaseOffset, mOffset, mL1Size, handler,
                       params_.mainBlockCount, params_.mainBlockSize);
            RunSegment(curBasicBlockId, basicBlockLimit, nBaseOffset, mOffset, mL1Size, handler,
                       params_.firstTailBlockCount, params_.firstTailBlockSize);
            RunSegment(curBasicBlockId, basicBlockLimit, nBaseOffset, mOffset, mL1Size, handler,
                       params_.secondTailBlockCount, params_.secondTailBlockSize);
        }
        startBasicBlockId = basicBlockLimit % params_.coreNum;
    }

private:
    template <typename Handler>
    __aicore__ inline void RunSegment(uint64_t &curBasicBlockId, uint64_t &basicBlockLimit, uint64_t &nBaseOffset,
                                      uint64_t mOffset, uint64_t mL1Size, Handler &handler,
                                      uint64_t blockCount, uint64_t blockSize) const
    {
        if (blockCount == 0) {
            return;
        }
        for (; curBasicBlockId < basicBlockLimit + blockCount; curBasicBlockId += params_.coreNum) {
            uint64_t nOffset = nBaseOffset + ((curBasicBlockId - basicBlockLimit) % blockCount) * blockSize;
            uint64_t nL1Size = nOffset + blockSize > params_.nSize ? params_.nSize - nOffset : blockSize;
            handler.OnMN(mOffset, mL1Size, nOffset, nL1Size);
        }
        nBaseOffset += blockSize * blockCount;
        basicBlockLimit += blockCount;
    }

    Params params_;
};

} // namespace Block

#endif
