/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file collective_comm_udma_api.h
 * \brief AivUdma 集合通信统一API - 四段式编程范式
 */

#pragma once

#include "kernel_operator.h"
#include "shmem.h"
#include "collective_comm_base.h"
#include "all_to_all/all_to_all_udma_put.h"
#include "all_to_all/all_to_all_udma_get.h"
#include "all_gather/all_gather_udma_put.h"
#include "all_gather/all_gather_udma_get.h"

namespace Dista {
namespace AivComm {

using namespace AscendC;

template<CollectiveOp Op, CommMode Mode, typename T>
using CollectiveComm = typename std::conditional<
    Op == CollectiveOp::AllToAll,
    typename std::conditional<
        Mode == CommMode::PUT,
        AllToAllPutImpl<T>,
        AllToAllGetImpl<T>
    >::type,
    typename std::conditional<
        Op == CollectiveOp::AllGather,
        typename std::conditional<
            Mode == CommMode::PUT,
            AllGatherPutImpl<T>,
            AllGatherGetImpl<T>
        >::type,
        typename std::conditional<
            Mode == CommMode::PUT,
            ReduceScatterPutImpl<T>,
            ReduceScatterGetImpl<T>
        >::type
    >::type
>::type;

} // namespace AivComm
} // namespace Dista