#ifndef SPLIT_POLICY_H_
#define SPLIT_POLICY_H_

#include "flash_attn_def.h"

// Extracted from: op_host/fa_adjust_sinner_souter.h
// Function: AdjustSinnerAndSouter
// Purpose: Determine sOuter/sInner tiling block sizes based on operator parameters
inline void AdjustSinnerAndSouter(uint32_t vHeadDim, uint32_t s1Size, int64_t s2Size, int32_t maskMode, int64_t winLeft,
                                  int64_t winRight, uint32_t qLayout, uint32_t &sOuterFactor, uint32_t &sInnerFactor)
{
    sOuterFactor = SOUTER_64;
    sInnerFactor = SINNER_128;

    bool checkQueryAndValueS = s1Size <= SOUTER_64 && s2Size > SINNER_128;

    if (vHeadDim <= DSIZE_128) {
        int64_t winLeftTmp = winLeft;
        int64_t winRightTmp = winRight;
        if (maskMode == 0) {
            winLeftTmp = (winLeftTmp > 0) ? 0 : winLeftTmp;
        } else if (maskMode == 4) {
            winRightTmp = (winRightTmp > 0) ? 0 : winRightTmp;
        }
        bool checkSparseMode = (maskMode != 2 && winLeftTmp + winRightTmp > 128);
        if (checkQueryAndValueS && checkSparseMode) {
            sOuterFactor = SOUTER_32;
            sInnerFactor = SINNER_256;
        }
    } else if (vHeadDim > DSIZE_128 && s1Size != 1) {
        if (checkQueryAndValueS && vHeadDim <= DSIZE_256) {
            sOuterFactor = SOUTER_32;
            sInnerFactor = SINNER_256;
        } else {
            sOuterFactor = SOUTER_64;
            sInnerFactor = SINNER_128;
        }
    } else if (s1Size == 1 && vHeadDim > DSIZE_128) {
        sOuterFactor = SOUTER_64;
        sInnerFactor = SINNER_128;
    }
}

#endif // SPLIT_POLICY_H_
