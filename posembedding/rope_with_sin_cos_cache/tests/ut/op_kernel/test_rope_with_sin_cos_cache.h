#ifndef _ROPE_WITH_SIN_COS_CACHE_H
#define _ROPE_WITH_SIN_COS_CACHE_H

#include "kernel_tiling/kernel_tiling.h"

#include <cstdint>
#include <cstring>

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__
#define ORIG_DTYPE_QUERYIN DT_FLOAT

#define __aicore__

inline void InitRopeWithSinCosCacheTilingData(uint8_t* tiling, RopeWithSinCosCacheTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(RopeWithSinCosCacheTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    RopeWithSinCosCacheTilingData tilingData;      \
    InitRopeWithSinCosCacheTilingData(tilingPointer, &tilingData)
#endif
