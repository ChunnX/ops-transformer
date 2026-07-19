/**
 * @file gen_position_ids_from_mask.h
 *
 * 直接 GM 逐元素读写版:
 *   - S 极小, 逐元素访问 GM 足够, 且规避 UB DataCopy 对齐 + int64 UB SetValue 两个坑
 *   - 内部全程 int32 累加, 仅写回时转 int64
 *   - 按 B 切核, 一行的 cumsum 留在同一核
 */
#ifndef GEN_POSITION_IDS_FROM_MASK_H_
#define GEN_POSITION_IDS_FROM_MASK_H_

#include "kernel_operator.h"

using namespace AscendC;

template <typename TIn>
class KernelGenPositionIdsFromMask {
public:
    __aicore__ inline KernelGenPositionIdsFromMask() {}

    __aicore__ inline void Init(GM_ADDR attentionMask, GM_ADDR positionIds,
                                const GenPositionIdsFromMaskTilingData *tiling)
    {
        b_ = tiling->b;
        s_ = tiling->s;
        fillValue_ = tiling->paddingFillValue;
        coreNum_ = tiling->coreNum;

        int64_t blockIdx = GetBlockIdx();
        int64_t base = b_ / coreNum_;
        int64_t rem  = b_ % coreNum_;
        if (blockIdx < rem) {
            rowNum_ = base + 1;
            rowBegin_ = blockIdx * (base + 1);
        } else {
            rowNum_ = base;
            rowBegin_ = rem * (base + 1) + (blockIdx - rem) * base;
        }
        if (rowNum_ <= 0) {
            return;
        }

        maskGm_.SetGlobalBuffer(reinterpret_cast<__gm__ TIn *>(attentionMask), b_ * s_);
        posGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(positionIds), b_ * s_);
    }

    __aicore__ inline void Process()
    {
        for (int64_t r = 0; r < rowNum_; ++r) {
            int64_t rowOff = (rowBegin_ + r) * s_;
            int32_t running = 0;
            for (int64_t i = 0; i < s_; ++i) {
                int32_t m = static_cast<int32_t>(maskGm_.GetValue(rowOff + i));
                running += m;
                int64_t p = (m == 0) ? fillValue_ : static_cast<int64_t>(running - 1);
                posGm_.SetValue(rowOff + i, p);
            }
        }
    }

private:
    GlobalTensor<TIn> maskGm_;
    GlobalTensor<int64_t> posGm_;

    int64_t b_ = 0;
    int64_t s_ = 0;
    int64_t fillValue_ = 1;
    int64_t coreNum_ = 1;
    int64_t rowNum_ = 0;
    int64_t rowBegin_ = 0;
};

#endif  // GEN_POSITION_IDS_FROM_MASK_H_
