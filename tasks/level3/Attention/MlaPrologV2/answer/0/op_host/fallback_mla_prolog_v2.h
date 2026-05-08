/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: MulanPSL-2.0
 */
#ifndef FALLBACK_MLA_PROLOG_V2_H
#define FALLBACK_MLA_PROLOG_V2_H

#include <vector>
#include "log/log.h"
#include "fallback/fallback_comm.h"
#include "fallback/fallback.h"
#include "fallback_mla_prolog.h"

#ifdef __cplusplus
extern "C" {
#endif
namespace fallback {

constexpr size_t DEQUANT_SCALE_Q_NOPE_INDEX = 4;

struct MlaPrologV2FallBackParam : MlaPrologFallBackParam {
    const gert::Tensor *dequantScaleQNope = nullptr;
};

graphStatus GetMlaPrologV2OutputTensor(const OpExecuteContext *ctx, MlaPrologV2FallBackParam &param);
graphStatus MlaV2HostExecuteFunc(OpExecuteContext *host_api_ctx);

} // namespace fallback

#ifdef __cplusplus
}
#endif

#endif
