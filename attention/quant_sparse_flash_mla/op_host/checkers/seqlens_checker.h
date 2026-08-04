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
 * \file seqlens_checker.h
 * \brief SeqLensChecker validates the SeqLens参数组: cu_seqlens_q,
 *        cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q, seqused_ori_kv,
 *        seqused_cmp_kv.
 */

#ifndef QSMLA_SEQLENS_CHECKER_H
#define QSMLA_SEQLENS_CHECKER_H

#include "base_checker.h"

namespace optiling {

class SeqLensChecker : public QSMLABaseChecker {
public:
    SeqLensChecker() = default;
    ~SeqLensChecker() override = default;

    ge::graphStatus CheckSinglePara(const QSMLATilingInfo &info) const override;
    ge::graphStatus CheckParaExistence(const QSMLATilingInfo &info) const override;
    ge::graphStatus CheckCrossFeature(const QSMLATilingInfo &info) const override;
    ge::graphStatus CheckMultiParaConsistency(const QSMLATilingInfo &info) const override;

private:
    ge::graphStatus CheckCuSeqLensQ(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckCuSeqLensOriKv(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckCuSeqLensCmpKv(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckSequsedQ(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckSequsedOriKv(const QSMLATilingInfo &info) const;
    ge::graphStatus CheckSequsedCmpKv(const QSMLATilingInfo &info) const;
};

} // namespace optiling
#endif // QSMLA_SEQLENS_CHECKER_H
