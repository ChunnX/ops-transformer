/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * EngramFetch proto header (minimal skeleton)
 */

#ifndef MC2_ENGRAM_FETCH_OP_GRAPH_ENGRAM_FETCH_PROTO_H
#define MC2_ENGRAM_FETCH_OP_GRAPH_ENGRAM_FETCH_PROTO_H

#include "graph/operator.h"

namespace ops {
namespace engram_fetch {

ge::Operator CreateEngramFetchOperator();

} // namespace engram_fetch
} // namespace ops

#endif // MC2_ENGRAM_FETCH_OP_GRAPH_ENGRAM_FETCH_PROTO_H