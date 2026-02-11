#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_REPEAT_INTERLEAVE_GRAD_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_REPEAT_INTERLEAVE_GRAD_H_

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RepeatInterleaveGradTilingData)
TILING_DATA_FIELD_DEF(int64_t, batch_dim_num);
TILING_DATA_FIELD_DEF(int64_t, repeats_i_grad_dim_num);
TILING_DATA_FIELD_DEF(int64_t, repeats_o_grad_dim_num);
TILING_DATA_FIELD_DEF(int64_t, data_dim_num);

TILING_DATA_FIELD_DEF(int64_t, core_num);
TILING_DATA_FIELD_DEF(int64_t, batch_dim_num_each_core);
TILING_DATA_FIELD_DEF(int64_t, batch_dim_num_last_core);
TILING_DATA_FIELD_DEF(int64_t, core_num_each_batch);
TILING_DATA_FIELD_DEF(int64_t, element_num_each_core);
TILING_DATA_FIELD_DEF(int64_t, element_num_last_core);

TILING_DATA_FIELD_DEF(int64_t, element_num_each_loop);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RepeatInterleaveGrad, RepeatInterleaveGradTilingData)
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_REPEAT_INTERLEAVE_GRAD_H_