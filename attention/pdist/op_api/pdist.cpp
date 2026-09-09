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

#include <cstddef>
#include <cstdint>

#include "pdist.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Pdist);

namespace {

static bool CheckNotNull(const aclTensor *x, const aclOpExecutor *executor)
{
    OP_CHECK(x != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Pdist x is nullptr."), return false);
    OP_CHECK(executor != nullptr, OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Pdist executor is nullptr."), return false);
    return true;
}
} // namespace

const aclTensor *Pdist(const aclTensor *x, float p, aclOpExecutor *executor)
{
    L0_DFX(Pdist, x, p);

    if (!CheckNotNull(x, executor)) {
        return nullptr;
    }

    const int64_t nSize = x->GetViewShape().GetDim(0);
    gert::Shape yShape({nSize * (nSize - 1) / 2});
    auto y = executor->AllocTensor(yShape, x->GetDataType(), Format::FORMAT_ND);
    OP_CHECK(y != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Pdist AllocTensor failed."), return nullptr);

    auto ret = INFER_SHAPE(Pdist, OP_INPUT(x), OP_OUTPUT(y), OP_ATTR(p));
    OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr, "Pdist InferShape failed.");

    ret = ADD_TO_LAUNCHER_LIST_AICORE(Pdist, OP_INPUT(x), OP_OUTPUT(y), OP_ATTR(p));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return nullptr,
                                         "Pdist ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return y;
}

} // namespace l0op
