/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.cann.huawei.com/CANN_OPEN_SOURCE_LICENSE_ACK
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is
 * distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitations under the License.
 */

#ifndef OP_API_INC_LEVEL0_PDIST_H_
#define OP_API_INC_LEVEL0_PDIST_H_

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor *Pdist(const aclTensor *x, float p, aclOpExecutor *executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_PDIST_H_
