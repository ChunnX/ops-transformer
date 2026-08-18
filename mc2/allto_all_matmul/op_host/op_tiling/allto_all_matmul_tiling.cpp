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
#include "../../op_kernel/arch35/allto_all_matmul_tiling_data.h"
#include "../../../common/op_kernel/moe_distribute_comm_ctx.h"
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

// 从异常上下文中获取 dump 基地址：读取 HcclCombinOpParam，从 workSpace+workSpaceSize 处扫描 Header magic
static uint64_t GetDumpBaseAddrFromException(aclrtExceptionInfo *args)
{
    auto aclrtGetArgsFromExceptionInfoFunc = Mc2Exception::GetAclrtGetArgsFromExceptionInfoFunc();
    if (aclrtGetArgsFromExceptionInfoFunc == nullptr) {
        return 0;
    }

    void *devArgsPtr = nullptr;
    uint32_t devArgsLen = 0;
    auto ret = aclrtGetArgsFromExceptionInfoFunc(args, &devArgsPtr, &devArgsLen);
    if (ret != ACL_SUCCESS) {
        return 0;
    }

    uint64_t argsAddr = 0;
    ret = aclrtMemcpy(&argsAddr, sizeof(uint64_t), devArgsPtr, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        return 0;
    }

    // 读取 HcclCombinOpParam 获取 workSpace 和 workSpaceSize
    HcclCombinOpParam hcclParam;
    ret = aclrtMemcpy(&hcclParam, sizeof(HcclCombinOpParam), reinterpret_cast<void *>(argsAddr),
                      sizeof(HcclCombinOpParam), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        return 0;
    }

    // dump 区域位于 workspace 末尾，从 workSpace+workSpaceSize 开始扫描 Header magic
    uint64_t scanBase = hcclParam.workSpace + hcclParam.workSpaceSize;
    MC2ExceptionDump::Header hdr;
    ret = aclrtMemcpy(&hdr, sizeof(MC2ExceptionDump::Header), reinterpret_cast<void *>(scanBase),
                      sizeof(MC2ExceptionDump::Header), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret == ACL_SUCCESS && hdr.headerMagic == MC2ExceptionDump::HEADER_MAGIC) {
        return scanBase;
    }

    // 如果 workSpace+workSpaceSize 处未找到，尝试直接从 argsAddr 扫描（兼容 dump 区域在 argsAddr 的场景）
    ret = aclrtMemcpy(&hdr, sizeof(MC2ExceptionDump::Header), reinterpret_cast<void *>(argsAddr),
                      sizeof(MC2ExceptionDump::Header), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret == ACL_SUCCESS && hdr.headerMagic == MC2ExceptionDump::HEADER_MAGIC) {
        return argsAddr;
    }

    return 0;
}

// 异常时从 device 读取 tilingData 并打印
static void PrintTilingDataOnException(aclrtExceptionInfo *args)
{
    uint64_t dumpBaseAddr = GetDumpBaseAddrFromException(args);
    if (dumpBaseAddr == 0) {
        dumpBaseAddr = Mc2Exception::MC2DumpBaseAddrManager::GetInstance().GetDumpBaseAddr();
    }
    std::vector<uint8_t> tilingBuf;
    if (Mc2Exception::GetTilingDataFromException(args, tilingBuf, dumpBaseAddr) != 0) {
        OP_LOGE("AlltoAllMatmul", "Failed to get tilingData from exception, dumpBaseAddr=0x%lx.", dumpBaseAddr);
        return;
    }

    if (tilingBuf.size() >= sizeof(AlltoAllQuantMatmulTilingData)) {
        auto *tilingData = reinterpret_cast<AlltoAllQuantMatmulTilingData *>(tilingBuf.data());
        auto &info = tilingData->alltoAllQuantMatmulTilingInfo;
        OP_LOGE("AlltoAllMatmul", "ExceptionTilingInfo: rankDim=%u tileM=%u tileCnt=%u tailM=%u tailCnt=%u "
                "rankM=%u rankN=%u rankK=%u commLen=%lu permuteLen=%lu hcclDataType=%lu",
                info.rankDim, info.tileM, info.tileCnt, info.tailM, info.tailCnt,
                info.rankM, info.rankN, info.rankK, info.commLen, info.permuteLen, info.hcclDataType);
        auto &wsPlan = tilingData->workspacePlanInfo;
        OP_LOGE("AlltoAllMatmul", "ExceptionWorkspacePlan: segCnt=%u", wsPlan.segCnt);
        for (uint8_t i = 0; i < wsPlan.segCnt; i++) {
            OP_LOGE("AlltoAllMatmul", "ExceptionWorkspaceSeg[%u]: addr=%lu size=%lu segType=%u order=%u", i,
                    wsPlan.segInfos[i].addr, wsPlan.segInfos[i].size, wsPlan.segInfos[i].segType,
                    wsPlan.segInfos[i].order);
        }
    } else if (tilingBuf.size() >= sizeof(AlltoAllMatmulTilingData)) {
        auto *tilingData = reinterpret_cast<AlltoAllMatmulTilingData *>(tilingBuf.data());
        auto &info = tilingData->alltoAllMatmulTilingInfo;
        OP_LOGE("AlltoAllMatmul", "ExceptionTilingInfo: rankDim=%u tileM=%u tileCnt=%u tailM=%u tailCnt=%u "
                "rankM=%u rankN=%u rankK=%u commLen=%lu permuteLen=%lu hcclDataType=%lu",
                info.rankDim, info.tileM, info.tileCnt, info.tailM, info.tailCnt,
                info.rankM, info.rankN, info.rankK, info.commLen, info.permuteLen, info.hcclDataType);
    } else {
        OP_LOGE("AlltoAllMatmul", "TilingData size=%zu too small, skip parse.", tilingBuf.size());
    }
}

// 异常时将完整 dump 区域（Header+Tiling+BlockStage+DumpParams+dump data）落盘为 bin 文件
static void DumpTilingDataToFile(aclrtExceptionInfo *args)
{
    uint64_t dumpBaseAddr = GetDumpBaseAddrFromException(args);
    if (dumpBaseAddr == 0) {
        OP_LOGE("AlltoAllMatmul", "DumpTilingDataToFile: failed to get dumpBaseAddr, skip file dump.");
        return;
    }

    std::vector<uint8_t> winContent;
    if (Mc2Exception::ParseStructuredDump(dumpBaseAddr, winContent) != 0) {
        OP_LOGE("AlltoAllMatmul", "DumpTilingDataToFile: ParseStructuredDump failed, skip file dump.");
        return;
    }

    auto acldumpGetPathFunc = Mc2Exception::GetAcldumpGetPathFunc();
    if (acldumpGetPathFunc == nullptr) {
        OP_LOGE("AlltoAllMatmul", "DumpTilingDataToFile: failed to load acldumpGetPath, skip file dump.");
        return;
    }

    const char *dumpPath = acldumpGetPathFunc(acldumpType::AIC_ERR_BRIEF_DUMP);
    if (dumpPath == nullptr) {
        OP_LOGE("AlltoAllMatmul", "DumpTilingDataToFile: acldumpGetPath returned NULL, skip file dump.");
        return;
    }

    uint32_t deviceId = aclrtGetDeviceIdFromExceptionInfo(args);
    std::string fileName = Mc2Exception::GenDumpFileNameWithPrefix(args, "AlltoAllMatmul", "tiling_data_dump_");
    OP_LOGE("AlltoAllMatmul", "DumpTilingDataToFile: deviceId=%u fileName=%s bufSize=%zu", deviceId, fileName.c_str(),
            winContent.size());

    if (Mc2Exception::DumpToFileWithSize(std::string(dumpPath), fileName, deviceId, winContent.data(),
                                         static_cast<uint32_t>(winContent.size())) != 0) {
        OP_LOGE("AlltoAllMatmul", "DumpTilingDataToFile: failed to dump tiling data to file.");
    }
}

// Register exception dump func
inline void AlltoAllMatmulExceptionImplWrapper(aclrtExceptionInfo *args, void *userdata)
{
    const char* socName = aclrtGetSocName();
    if (std::strstr(socName, "Ascend950") == nullptr) {
        return;
    }
    Mc2Exception::Mc2ExceptionImplTmp(args, userdata, "AlltoAllMatmul");
    PrintTilingDataOnException(args);
    DumpTilingDataToFile(args);
}

__attribute__((constructor)) void RegisterAlltoAllMatmulExceptionFunc()
{
    IMPL_OP(AlltoAllMatmul)
        .ExceptionDumpParseFunc(AlltoAllMatmulExceptionImplWrapper);
}
} // namespace MC2Tiling
