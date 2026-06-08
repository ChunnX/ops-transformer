#include "../torch/update_context_torch.h"

namespace ascend_ops {
namespace Update_Context {

constexpr uint32_t OP_TYPE_ALL_TO_ALLV = 8;
constexpr uint32_t EP_OP_TYPE = OP_TYPE_ALL_TO_ALLV;
constexpr uint32_t COMM_ENGINE_AIV = 4;
const std::string EP_ALG_CONFIG = "AlltoAll=level0:fullmesh;level1:pairwise";
static bool isInit = false;
static std::string last_group_ep_str = "";
static at::Tensor output;
int64_t bufferSize = 0;

TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("updateContext(Tensor x, str group_ep, int ep_world_size) -> (Tensor, int)");
}

static int32_t CreatMc2Context(HcclComm &comm, int64_t worldSize, int64_t &cclBufferSize, Mc2ContextStru *mc2Context)
{
    uint32_t rankId;
    (void)HcclGetRankId(comm, &rankId);
    mc2Context->epRankId = rankId;
    mc2Context->rankSizePerServer = (uint32_t)worldSize;

    for (uint64_t remoteRankId = 0; remoteRankId < (uint64_t)worldSize; remoteRankId++) {
        void *remoteAddr = nullptr;
        uint64_t commSize = 0;
        HcclResult ret;
        if (rankId == remoteRankId) {
            ret = static_cast<HcclResult>(HcclGetHcclBuffer(comm, &remoteAddr, &commSize));
            cclBufferSize = commSize;
        } else {
            ret = static_cast<HcclResult>(HcclGetRemoteIpcHcclBuf(comm, remoteRankId, &remoteAddr, &commSize));
        }
        TORCH_CHECK((ret == HCCL_SUCCESS), "Get HcclBuffer failed, ret=", ret);
        mc2Context->epHcclBuffer[remoteRankId] = (uint64_t)remoteAddr;
    }
    return 0;
}

static int32_t CreateHcclContext(HcclComm &commHandle, void *opArgs, int64_t worldSize,
                                  const char* groupName, std::string algConfig, uint32_t opType)
{
    HcclResult ret = static_cast<HcclResult>(HcclKfcOpArgsSetAlgConfig(opArgs, const_cast<char*>(algConfig.c_str())));
    TORCH_CHECK(ret == 0, "HcclKfcOpArgsSetAlgConfig failed, ret:", ret);

    ret = static_cast<HcclResult>(HcclCommGetHandleWithName(groupName, &commHandle));
    TORCH_CHECK(ret == 0, "HcclCommGetHandleWithName failed, ret:", ret);

    void *opsResCtx;
    ret = static_cast<HcclResult>(HcclCreateOpResCtx(commHandle, opType, opArgs, &opsResCtx));
    TORCH_CHECK(ret == 0, "HcclCreateOpResCtx failed, ret:", ret);

    uint32_t rankId = 0;
    uint32_t worldSizeHccl = 0;
    ret = static_cast<HcclResult>(HcclGetRankSize(commHandle, &worldSizeHccl));
    TORCH_CHECK(ret == HCCL_SUCCESS, "HcclGetRankSize failed, ret:", ret);

    ret = static_cast<HcclResult>(HcclGetRankId(commHandle, &rankId));
    TORCH_CHECK(ret == HCCL_SUCCESS, "HcclGetRankId failed, ret:", ret);

    if (rankId >= worldSizeHccl || (uint64_t)worldSize != worldSizeHccl) {
        TORCH_CHECK(false, "rankId/worldSize mismatch");
        return -1;
    }
    return 0;
}

static int32_t GetMc2Context(Mc2ContextStru* mc2ContextHost, int64_t epWorldSize, int64_t &cclBufferSize, const char* groupEpStr)
{
    void* opArgs = nullptr;
    HcclResult ret = static_cast<HcclResult>(HcclKfcAllocOpArgs(&opArgs));
    TORCH_CHECK(ret == 0, "HcclKfcAllocOpArgs failed, ret:", ret);

    ret = static_cast<HcclResult>(HcclKfcOpArgsSetCommEngine(opArgs, (uint8_t)COMM_ENGINE_AIV));
    TORCH_CHECK(ret == 0, "HcclKfcOpArgsSetCommEngine failed, ret:", ret);

    HcclComm epCommHandle;
    int32_t contextRet = CreateHcclContext(epCommHandle, opArgs, epWorldSize, groupEpStr, EP_ALG_CONFIG, EP_OP_TYPE);
    TORCH_CHECK(contextRet == 0, "CreateHcclContext failed, ret:", contextRet);

    contextRet = CreatMc2Context(epCommHandle, epWorldSize, cclBufferSize, mc2ContextHost);
    TORCH_CHECK(contextRet == 0, "CreatMc2Context failed, ret:", contextRet);

    ret = static_cast<HcclResult>(HcclKfcFreeOpArgs(opArgs));
    TORCH_CHECK(ret == 0, "HcclKfcFreeOpArgs failed, ret:", ret);

    return 0;
}

std::tuple<at::Tensor, int64_t> update_context(const at::Tensor &x, c10::string_view group_ep, int64_t ep_world_size)
{
    std::string new_group = std::string(group_ep);
    if (isInit && new_group == last_group_ep_str) {
        return std::make_tuple(output, bufferSize);
    }

    Mc2ContextStru mc2ContextHost[2];
    memset(mc2ContextHost, 0, sizeof(mc2ContextHost));
    int32_t ret = GetMc2Context(&mc2ContextHost[0], ep_world_size, bufferSize, new_group.c_str());
    TORCH_CHECK(ret == 0, "GetMc2Context failed, ret=", ret);

    at::Tensor hostCtx = at::from_blob(&mc2ContextHost, {sizeof(Mc2ContextStru) * 2 / (int64_t)sizeof(int32_t)}, at::kInt);
    output = at::empty({sizeof(Mc2ContextStru) * 2 / (int64_t)sizeof(int32_t)},
        at::TensorOptions().dtype(at::kInt).device(c10::DeviceType::PrivateUse1));
    output.copy_(hostCtx);

    last_group_ep_str = new_group;
    isInit = true;
    return std::make_tuple(output, bufferSize);
}

TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("updateContext", update_context);
}

} // namespace Update_Context
} // namespace ascend_ops
