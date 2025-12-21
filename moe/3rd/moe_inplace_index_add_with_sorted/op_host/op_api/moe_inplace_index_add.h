/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file inplace_index_add.h
 * \brief
 */
#ifndef OP_API_INC_LEVEL0_OP_MOE_INDEX_ADD_H_
#define OP_API_INC_LEVEL0_OP_MOE_INDEX_ADD_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *MoeInplaceIndexAddAiCore(const aclTensor *self, const int64_t dim, const aclTensor *index,
                                       const aclTensor *source, const aclTensor *alphaTensor,
                                       aclOpExecutor *executor);

const aclTensor *MoeInplaceIndexAddAiCpu(const aclTensor *self, const int64_t dim, const aclTensor *index,
                                      const aclTensor *source, const aclTensor *alphaTensor,
                                      aclOpExecutor *executor);

const aclTensor *MoeInplaceIndexAddWithSorted(const aclTensor *self, const int64_t dim, const aclTensor *sortedIndices,
                                           const aclTensor *pos, const aclTensor *value, const aclTensor *alphaTensor,
                                           aclOpExecutor *executor);
}
#endif
