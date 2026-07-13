/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file collective_comm_api.h
 * \brief Hcomm 集合通信统一 API
 */

#pragma once

#include "collective_comm_base.h"
#include "all_to_all/all_to_all_udma_get.h"
#include "all_to_all/all_to_all_udma_put.h"

namespace Atcoss {
namespace AivComm {

using namespace AscendC;

enum class CommCollectiveOp {
    AllToAll,
    AllGather,
    ReduceScatter
};

enum class CommMode {
    GET,
    PUT
};

template<CommCollectiveOp Op, CommMode Mode, typename T> struct CollectiveCommHelper;
template<typename T> struct CollectiveCommHelper<CommCollectiveOp::AllToAll, CommMode::GET, T> {
    using type = AllToAllCommGetImpl<T>;
};
template<typename T> struct CollectiveCommHelper<CommCollectiveOp::AllToAll, CommMode::PUT, T> {
    using type = AllToAllCommPutImpl<T>;
};

template<CommCollectiveOp Op, CommMode Mode, typename T>
using CollectiveComm = typename CollectiveCommHelper<Op, Mode, T>::type;

} // namespace AivComm
} // namespace Atcoss
