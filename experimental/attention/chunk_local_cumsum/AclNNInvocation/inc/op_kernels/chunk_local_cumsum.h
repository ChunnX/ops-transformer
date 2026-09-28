/**
 * @file mi_layernorm.h
 *
 * Copyright (C) 2024. Tsingmao Intelligence Technologies Co., Ltd. All rights
 * reserved.
 *
 */
#ifndef MI_QKV_UNIFY_H
#define MI_QKV_UNIFY_H

#include <vector>
#include "acl/acl.h"
#include "tensor.h"
// #include "op_runner.h"

void ChunkLocalCumsum(aclrtStream& stream, void* workspace, Tensor& input, Tensor& cu_seqlens, Tensor& output,int64_t head_num,int64_t chunk_size,
              const std::vector<int64_t>& cu_seqlen);

#endif
