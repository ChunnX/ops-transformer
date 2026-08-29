/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mla_prolog_v5.h
 * \brief MlaPrologV5 L0 接口声明（独立工程手写实现，替代源仓框架生成的 aclnnInner 层）
 */

#ifndef MLA_PROLOG_V5_H
#define MLA_PROLOG_V5_H

#include "aclnn/aclnn_base.h"

namespace l0op {

aclnnStatus MlaPrologV5(const aclTensor *tokenX, const aclTensor *weightDq, const aclTensor *weightUqQr,
                        const aclTensor *weightUk, const aclTensor *weightDkvKr, const aclTensor *rmsnormGammaCq,
                        const aclTensor *rmsnormGammaCkv, const aclTensor *ropeSin, const aclTensor *ropeCos,
                        aclTensor *kvCacheRef, aclTensor *krCacheRef, const aclTensor *cacheIndexOptional,
                        const aclTensor *dequantScaleXOptional, const aclTensor *dequantScaleWDqOptional,
                        const aclTensor *dequantScaleWUqQrOptional, const aclTensor *dequantScaleWDkvKrOptional,
                        const aclTensor *quantScaleCkvOptional, const aclTensor *quantScaleCkrOptional,
                        const aclTensor *smoothScalesCqOptional, const aclTensor *actualSeqLenOptional,
                        const aclTensor *kNopeClipAlphaOptional, double rmsnormEpsilonCq, double rmsnormEpsilonCkv,
                        char *cacheModeOptional, bool queryNormFlag, int64_t weightQuantMode, int64_t kvCacheQuantMode,
                        int64_t queryQuantMode, int64_t ckvkrRepoMode, int64_t quantScaleRepoMode, int64_t tileSize,
                        double qcQrScale, double kcScale, bool doRope, const aclTensor *queryOut,
                        const aclTensor *queryRopeOut, const aclTensor *dequantScaleQNopeOut,
                        const aclTensor *queryNormOut, const aclTensor *dequantScaleQNormOut, aclOpExecutor *executor);

} // namespace l0op

#endif // MLA_PROLOG_V5_H
