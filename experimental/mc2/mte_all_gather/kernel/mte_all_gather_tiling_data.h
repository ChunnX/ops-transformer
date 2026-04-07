/*!
 * \file mte_all_gather_tiling_data.h
 * \brief 定义TilingData
 */

#ifndef MTE_ALL_GATHER_TILING_DATA_H
#define MTE_ALL_GATHER_TILING_DATA_H

#include <kernel_tiling/kernel_tiling.h>

struct MteAllGatherTilingInfo {
    uint64_t bs;
    uint64_t hiddenSize;
    uint64_t aivNum;
};

struct MteAllGatherTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    MteAllGatherTilingInfo mteAllGatherTilingInfo;
};

#endif // MTE_ALL_GATHER_TILING_DATA_H
