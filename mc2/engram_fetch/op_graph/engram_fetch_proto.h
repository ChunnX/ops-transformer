#ifndef ENGRAM_FETCH_PROTO_H_
#define ENGRAM_FETCH_PROTO_H_

#include "graph/operator_reg.h"
namespace ge {

REG_OP(EngramFetch)
    .INPUT(bufferhandle, TensorType({DT_INT32}))
    .INPUT(indices, TensorType({DT_INT32}))
    .OUTPUT(fetched, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(num_ranks, Int)
    .REQUIRED_ATTR(num_entries_per_rank, Int)
    .ATTR(hccl_buffer_size, Int, 0)
    .ATTR(output_dtype, Int, 1)
    .OP_END_FACTORY_REG(EngramFetch)

} // namespace ge
#endif // ENGRAM_FETCH_PROTO_H_