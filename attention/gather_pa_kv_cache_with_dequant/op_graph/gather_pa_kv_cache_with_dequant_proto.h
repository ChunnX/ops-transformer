/** 
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef OPS_OP_PROTO_INC_GatherPaKvCacheWithDequant_H_
#define OPS_OP_PROTO_INC_GatherPaKvCacheWithDequant_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

REG_OP(GatherPaKvCacheWithDequant)
    .INPUT(key_cache, TensorType({DT_INT8}))
    .INPUT(key_scale_cache, TensorType({DT_FLOAT}))
    .INPUT(value_cache, TensorType({DT_FLOAT8_E4M3FN}))
    .INPUT(value_scale, TensorType({DT_FLOAT}))
    .INPUT(block_tables, TensorType::IndexNumberType())
    .INPUT(seq_lens, TensorType::IndexNumberType())
    .INPUT(key, TensorType({DT_BF16}))
    .INPUT(value, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(seq_offset, TensorType::IndexNumberType())
    .OUTPUT(key, TensorType({DT_BF16}))
    .OUTPUT(value, TensorType({DT_BF16}))
    .ATTR(cache_mode, String, "Norm")
    .ATTR(is_seq_lens_cumsum, Bool, true)
    .ATTR(key_dst_type, Int, DT_BF16)
    .ATTR(value_dst_type, Int, DT_BF16)
.OP_END_FACTORY_REG(GatherPaKvCacheWithDequant)

} // namespace ge

#endif
