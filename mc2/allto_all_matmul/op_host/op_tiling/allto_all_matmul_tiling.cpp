/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_all_matmul_tiling.cpp
 * \brief hosttiling
 */
#include <register/op_def_registry.h>
#include <register/op_impl_registry.h>

#include "allto_all_matmul_tiling_base.h"
#include "mc2_exception_dump.h"

using namespace ge;
using Ops::Transformer::OpTiling::TilingRegistryArch;
using Ops::Transformer::OpTiling::TilingRegistryNew;

namespace MC2Tiling {

static ge::graphStatus AlltoAllMatmulTilingFunc(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
    NpuArch npuArch = ascendcPlatform.GetCurNpuArch();
    if (npuArch == NpuArch::DAV_3510) {
        return TilingRegistryArch::GetInstance().DoTilingImpl(context);
    }
    return TilingRegistryNew::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingParseForAlltoAllMatmul(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

struct AlltoAllMatmulCompileInfo {};

IMPL_OP_OPTILING(AlltoAllMatmul)
    .Tiling(AlltoAllMatmulTilingFunc)
    .TilingParse<AlltoAllMatmulCompileInfo>(TilingParseForAlltoAllMatmul);

// Register exception dump func
inline void AlltoAllMatmulDumpTilingData(aclrtExceptionInfo *args, void *userdata, const char *op)
{
    OP_LOGE(op, "Start AlltoAllMatmulDumpTilingData");
    auto getArgsFunc = Mc2Exception::GetAclrtGetArgsFromExceptionInfoFunc();
    if (getArgsFunc == nullptr) {
        return;
    }

    void *devArgsPtr = nullptr;
    uint32_t devArgsLen = 0;
    if (getArgsFunc(args, &devArgsPtr, &devArgsLen) != ACL_SUCCESS) {
        return;
    }
    OP_LOGE(op, "end getArgsFunc");

    uint32_t deviceId = aclrtGetDeviceIdFromExceptionInfo(args);
    OP_LOGE(op, "Tiling dump start: devArgsPtr=%p devArgsLen=%u", devArgsPtr, devArgsLen);

    constexpr uint32_t argsDumpSize = 256U;
    std::vector<uint8_t> argsBuf(argsDumpSize, 0);
    if (aclrtMemcpy(argsBuf.data(), argsDumpSize, devArgsPtr, argsDumpSize, ACL_MEMCPY_DEVICE_TO_HOST) !=
        ACL_SUCCESS) {
        OP_LOGE(op, "aclrtMemcpy args buffer failed.");
        return;
    }
    OP_LOGE(op, "end aclrtMemcpy");

    auto dumpPathFunc = Mc2Exception::GetAcldumpGetPathFunc();
    if (dumpPathFunc == nullptr) {
        return;
    }
    OP_LOGE(op, "end GetAcldumpGetPathFunc");
    const char *dumpPath = dumpPathFunc(acldumpType::AIC_ERR_BRIEF_DUMP);
    if (dumpPath == nullptr) {
        OP_LOGE(op, "acldumpGetPath returned NULL");
        return;
    }
    OP_LOGE(op, "end dumpPathFunc");

    Mc2Exception::DumpToFileWithSize(std::string(dumpPath),
        Mc2Exception::GenDumpFileNameWithPrefix(args, op, "args_info_"), deviceId, argsBuf.data(), argsDumpSize);

    constexpr uint32_t tilingGmOffset = 12U * sizeof(uint64_t);
    uint64_t tilingGM = 0;
    if (aclrtMemcpy(&tilingGM, sizeof(uint64_t), reinterpret_cast<uint8_t *>(devArgsPtr) + tilingGmOffset,
                    sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        OP_LOGE(op, "aclrtMemcpy tilingGM pointer failed, offset=%u.", tilingGmOffset);
        return;
    }
    OP_LOGE(op, "tilingGM=0x%lx (offset=%u)", tilingGM, tilingGmOffset);
    if (tilingGM == 0) {
        OP_LOGE(op, "tilingGM is 0, skip tiling data dump.");
        return;
    }

    constexpr uint32_t tilingDumpSize = 2048U;
    std::vector<uint8_t> tilingBuf(tilingDumpSize, 0);
    if (aclrtMemcpy(tilingBuf.data(), tilingDumpSize, reinterpret_cast<void *>(tilingGM), tilingDumpSize,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        OP_LOGE(op, "aclrtMemcpy tiling data failed.");
        return;
    }
    Mc2Exception::DumpToFileWithSize(std::string(dumpPath),
        Mc2Exception::GenDumpFileNameWithPrefix(args, op, "tiling_data_"), deviceId, tilingBuf.data(),
        tilingDumpSize);
    OP_LOGE(op, "end AlltoAllMatmulDumpTilingData");
}

inline void AlltoAllMatmulExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    const char* socName = aclrtGetSocName();
    if (std::strstr(socName, "Ascend950") == nullptr) {
        return;
    }
    Mc2Exception::Mc2ExceptionImplTmp(args, userdata, "AlltoAllMatmul");
    AlltoAllMatmulDumpTilingData(args, userdata, "AlltoAllMatmul");
}

__attribute__((constructor)) void RegisterAlltoAllMatmulExceptionFunc()
{
    IMPL_OP(AlltoAllMatmul)
        .ExceptionDumpParseFunc(AlltoAllMatmulExceptionImplWrapper);
}
} // namespace MC2Tiling
