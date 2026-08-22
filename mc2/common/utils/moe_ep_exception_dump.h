#ifndef MOE_EP_EXCEPTION_DUMP_H
#define MOE_EP_EXCEPTION_DUMP_H

#include "mc2_exception_dump_common.h"
#include "../op_kernel/mc2_moe_context.h"
#include "../op_kernel/moe_ep_exception_dump_defs.h"
#include "mc2_log.h"
#include <vector>
#include <string>

#if RUNTIME_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION && METADEF_VERSION_NUM >= EXCEPTION_DUMP_SUPPORT_VERSION
namespace Mc2Exception {

static const char *MOE_EP_REGION_NAMES[] = {
    "per_core_diag",
    "count_notify",
    "expert_count",
    "dispatch_slot_state",
    "combine_token_state",
    "combine_state_tail",
    "hybrid_scaleout_status",
};

static const uint64_t MOE_EP_METADATA_DUMP_SIZE = MOE_EP_METADATA_SIZE;

inline void MoeEpExceptionImpl(aclrtExceptionInfo *args, void *userdata, const char *opName)
{
    (void)userdata;
    const char *socName = aclrtGetSocName();
    if (std::strstr(socName, "Ascend950") == nullptr) {
        OP_LOGE(COMMON_NAME, "MoeEpExceptionImpl: soc %s not supported, skip", socName);
        return;
    }
    OP_LOGD(COMMON_NAME, "MoeEpExceptionImpl start: op=%s soc=%s", opName, socName);

    auto getArgsFunc = GetAclrtGetArgsFromExceptionInfoFunc();
    if (getArgsFunc == nullptr) {
        OP_LOGE(COMMON_NAME, "Failed to load aclrtGetArgsFromExceptionInfo");
        return;
    }
    void *devArgsPtr = nullptr;
    uint32_t devArgsLen = 0;
    auto ret = getArgsFunc(args, &devArgsPtr, &devArgsLen);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(COMMON_NAME, "aclrtGetArgsFromExceptionInfo failed. ret=%d", ret);
        return;
    }
    uint32_t deviceId = aclrtGetDeviceIdFromExceptionInfo(args);
    OP_LOGD(COMMON_NAME, "deviceId=%u devArgsPtr=%p devArgsLen=%u", deviceId, devArgsPtr, devArgsLen);

    uint64_t contextAddr = 0;
    ret = aclrtMemcpy(&contextAddr, sizeof(uint64_t), devArgsPtr, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(COMMON_NAME, "D2H context addr failed. ret=%d", ret);
        return;
    }
    OP_LOGD(COMMON_NAME, "contextAddr=0x%lx", contextAddr);

    std::vector<uint8_t> ctxBuf(sizeof(Mc2Aclnn::MoeCommContext), 0);
    ret = aclrtMemcpy(ctxBuf.data(), sizeof(Mc2Aclnn::MoeCommContext),
                      reinterpret_cast<void *>(contextAddr), sizeof(Mc2Aclnn::MoeCommContext),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(COMMON_NAME, "D2H MoeCommContext failed. ret=%d", ret);
        return;
    }
    auto *ctx = reinterpret_cast<Mc2Aclnn::MoeCommContext *>(ctxBuf.data());
    uint32_t epRankId = ctx->epRankId;
    if (epRankId >= Mc2Aclnn::HCCL_MAX_RANK_SIZE) {
        OP_LOGE(COMMON_NAME, "epRankId=%u out of range", epRankId);
        return;
    }
    uint64_t winBase = ctx->epHcclBuffer[epRankId];
    OP_LOGD(COMMON_NAME, "epRankId=%u winBase=0x%lx", epRankId, winBase);
    if (winBase == 0) {
        OP_LOGE(COMMON_NAME, "winBase is null");
        return;
    }

    auto dumpPathFunc = GetAcldumpGetPathFunc();
    if (dumpPathFunc == nullptr) {
        OP_LOGE(COMMON_NAME, "Failed to load acldumpGetPath");
        return;
    }
    const char *dumpPath = dumpPathFunc(acldumpType::AIC_ERR_BRIEF_DUMP);
    if (dumpPath == nullptr) {
        OP_LOGE(COMMON_NAME, "acldumpGetPath returned null");
        return;
    }
    OP_LOGD(COMMON_NAME, "dumpPath=%s", dumpPath);

    std::vector<uint8_t> metaBuf(MOE_EP_METADATA_DUMP_SIZE, 0);
    ret = aclrtMemcpy(metaBuf.data(), MOE_EP_METADATA_DUMP_SIZE, reinterpret_cast<void *>(winBase),
                      MOE_EP_METADATA_DUMP_SIZE, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        OP_LOGE(COMMON_NAME, "D2H Metadata region failed. ret=%d", ret);
        return;
    }
    std::string metaFileName = GenMoeEpDumpFileName(args, opName, epRankId, "metadata");
    if (DumpRegionToFile(std::string(dumpPath), metaFileName, deviceId, metaBuf.data(),
                         MOE_EP_METADATA_DUMP_SIZE) != 0) {
        OP_LOGE(COMMON_NAME, "Dump metadata.bin failed");
        return;
    }

    auto *metadata = reinterpret_cast<MoeEpDumpMetadata *>(metaBuf.data());
    OP_LOGD(COMMON_NAME, "layoutVersion=%u nmt=%u topK=%u hidden=%u epWorldSize=%u aivNum=%u networkMode=%u",
            metadata->layoutVersion, metadata->nmt, metadata->topK, metadata->hidden,
            metadata->epWorldSize, metadata->aivNum, metadata->networkMode);

    for (uint32_t i = 0; i < MOE_EP_DUMP_REGION_COUNT; ++i) {
        uint64_t regionOffset = metadata->regions[i].offset;
        uint64_t regionSize = metadata->regions[i].size;
        if (regionSize == 0) {
            OP_LOGD(COMMON_NAME, "Region %u (%s): size=0, skip", i, MOE_EP_REGION_NAMES[i]);
            continue;
        }
        OP_LOGD(COMMON_NAME, "Region %u (%s): offset=%lu size=%lu", i, MOE_EP_REGION_NAMES[i],
                regionOffset, regionSize);
        std::vector<uint8_t> regionBuf(regionSize, 0);
        ret = aclrtMemcpy(regionBuf.data(), regionSize,
                          reinterpret_cast<void *>(winBase + regionOffset), regionSize,
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            OP_LOGE(COMMON_NAME, "D2H region %u (%s) failed. ret=%d", i, MOE_EP_REGION_NAMES[i], ret);
            return;
        }
        std::string regionFileName = GenMoeEpDumpFileName(args, opName, epRankId, MOE_EP_REGION_NAMES[i]);
        if (DumpRegionToFile(std::string(dumpPath), regionFileName, deviceId, regionBuf.data(),
                              regionSize) != 0) {
            OP_LOGE(COMMON_NAME, "Dump region %u (%s) failed", i, MOE_EP_REGION_NAMES[i]);
            return;
        }
    }
    OP_LOGE(COMMON_NAME, "MoeEpExceptionImpl done: op=%s rank=%u", opName, epRankId);
}

} // namespace Mc2Exception
#endif
#endif
