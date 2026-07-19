/**
 * @file gen_position_ids_from_mask.cpp
 */
#include "gen_position_ids_from_mask.h"

extern "C" __global__ __aicore__ void gen_position_ids_from_mask(
    GM_ADDR attention_mask, GM_ADDR position_ids, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    // tiling key: 1=int32, 2=int64, 3=bool(int8). 输入 dtype 分三条入口, 内部统一转 int32.
    if (TILING_KEY_IS(1)) {
        KernelGenPositionIdsFromMask<int32_t> op;
        op.Init(attention_mask, position_ids, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelGenPositionIdsFromMask<int64_t> op;
        op.Init(attention_mask, position_ids, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelGenPositionIdsFromMask<int8_t> op;  // bool 以 int8 搬运
        op.Init(attention_mask, position_ids, &tilingData);
        op.Process();
    }
}