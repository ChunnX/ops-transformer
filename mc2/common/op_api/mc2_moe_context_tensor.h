/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mc2_moe_context_tensor.h
 * \brief
 */

#include "mc2_moe_context.h"
#include "opdev/platform.h"
#include "hccl/hccl_rank_graph.h"
#include "mmpa/mmpa_api.h"

namespace Mc2Aclnn {
constexpr uint64_t KopyDefaultCtxOffset = 0; // 默认从最开始拷贝
constexpr uint64_t MaxContextTagSize = 255; // 最大上下文标签大小
constexpr uint32_t HCCL_COMM_LAYERS_MTE_CCU = 1; // 当走MTE或者CCU通信时， hccl获取的组网层数应该为1

// 定义 HCCL 接口函数指针类型
typedef HcclResult (*HcomGetCommHandleByGroup_t)(const char* group, HcclComm* comm);
typedef HcclResult (*HcclRankGraphGetLinks_t)(HcclComm comm, uint32_t layers, uint32_t srcRank,
    uint32_t dstRank, CommLink** links, uint32_t* linkNum);
typedef HcclResult (*HcclRankGraphGetLayers_t)(HcclComm comm, uint32_t** layerList, uint32_t* layerNum);
typedef HcclResult (*HcclChannelAcquire_t)(HcclComm comm, CommEngine engine, HcclChannelDesc* desc,
    uint32_t channelNum, ChannelHandle* handle);
typedef HcclResult (*HcclGetHcclBuffer_t)(HcclComm comm, void** buffer, uint64_t* size);
typedef HcclResult (*HcclChannelGetHcclBuffer_t)(HcclComm comm, ChannelHandle handle,
    void** buffer, uint64_t* size);
typedef HcclResult (*HcclEngineCtxCreate_t)(HcclComm comm, const char* tag, CommEngine engine,
    uint64_t size, void** ctx);
typedef HcclResult (*HcclEngineCtxGet_t)(HcclComm comm, const char* tag, CommEngine engine,
    void** ctx, uint64_t* size);
typedef HcclResult (*HcclEngineCtxCopy_t)(HcclComm comm, CommEngine engine, const char* tag,
    void* hostPtr, uint64_t size, uint64_t offset);
typedef HcclResult (*HcclGetRankId_t)(HcclComm comm, uint32_t* rankId);
typedef HcclResult (*HcclGetRankSize_t)(HcclComm comm, uint32_t* rankSize);

// 定义 Hccl 函数指针变量
static HcomGetCommHandleByGroup_t g_HcomGetCommHandleByGroup = nullptr;
static HcclRankGraphGetLinks_t g_HcclRankGraphGetLinks = nullptr;
static HcclRankGraphGetLayers_t g_HcclRankGraphGetLayers = nullptr;
static HcclChannelAcquire_t g_HcclChannelAcquire = nullptr;
static HcclGetHcclBuffer_t g_HcclGetHcclBuffer = nullptr;
static HcclChannelGetHcclBuffer_t g_HcclChannelGetHcclBuffer = nullptr;
static HcclEngineCtxCreate_t g_HcclEngineCtxCreate = nullptr;
static HcclEngineCtxGet_t g_HcclEngineCtxGet = nullptr;
static HcclEngineCtxCopy_t g_HcclEngineCtxCopy = nullptr;
static HcclGetRankId_t g_HcclGetRankId = nullptr;
static HcclGetRankSize_t g_HcclGetRankSize = nullptr;

// 加载 Hccl 接口符号
template<typename T>
static bool LoadSymbol(void* handle, T& func_ptr, const char* symbol_name)
{
    dlerror();
    func_ptr = reinterpret_cast<T>(dlsym(handle, symbol_name));
    const char* error = dlerror();
    if (error) {
        OP_LOGE(ACLNN_ERR_INNER, "Failed to load symbol '%s' : %s", symbol_name, error);
        return false;
    }
    OP_LOGD("Loaded symbol: %s", symbol_name);

    return true;
}

#define LOAD_HCCL_SYMBOL(hcclHandle, symbol, name) \
    do { \
        if (!LoadSymbol(hcclHandle, symbol, name)) { \
            OP_LOGD("load %s symbol error", name); \
            dlclose(hcclHandle); \
            hcclHandle = nullptr; \
            return ACLNN_ERR_INNER; \
        } \
    } while (0)

static aclnnStatus GetCommHandle(const char* groupEp, HcclComm& hcclHandle)
{
    OP_LOGD("Start to get HCCL communication handle");
    auto ret = g_HcomGetCommHandleByGroup(groupEp, &hcclHandle); // 获取HCCL通信句柄
    if (ret != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get HCCL Communication handle failed groupEp is:%s", groupEp);
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL Communication Handle Success");
    return ACLNN_SUCCESS;
}

static aclnnStatus GetHcclCommLink(const HcclComm& hcclHandle, const uint32_t netLayers, const uint32_t srcRankId,
                                   const uint32_t dstRankId, const CommProtocol& protocol, CommLink*& links)
{
    OP_LOGD("Start to get HCCL communication link");
    CommLink* linksList{nullptr};
    uint32_t netLinkNum = 0;
    auto hcclRet = g_HcclRankGraphGetLinks(hcclHandle, netLayers, srcRankId, dstRankId, &linksList, &netLinkNum);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get HCCL Communication link failed");
        return ACLNN_ERR_INNER;
    }
    if (netLinkNum == 0) {
        OP_LOGE(ACLNN_ERR_INNER, "The Net Link Is nullptr.");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL Rank Links  Success  Links Num is: %d", netLinkNum);
    uint32_t linksIndex = 0;
    while (linksIndex < netLinkNum) { // 遍历组网支持的协议
        if (linksList[linksIndex].linkAttr.linkProtocol == protocol) { // 如果与目标协议相同返回对应的link
            links = &linksList[linksIndex];
            break;
        }
        linksIndex++;
    }
    if (linksIndex == netLinkNum) { // 遍历完没有找到匹配的协议
        OP_LOGE(ACLNN_ERR_INNER, "Failed to obtain communication handle: No matching protocol \
                found in the connection configuration");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL Communication Link Success");
    return ACLNN_SUCCESS;
}

static aclnnStatus GetHcclCommChannel(const HcclComm hcclHandle, const uint32_t rankDim, const uint32_t srcRankId,
                                     const CommProtocol& protocol, const CommEngine& engine,
                                     std::vector<ChannelHandle>& channeles)
{
    OP_LOGD("Start to get HCCL communication channel");
    uint32_t channelNum = rankDim - 1;
    std::vector<HcclChannelDesc> channelDesc;
    channelDesc.resize(channelNum);
    channeles.resize(channelNum);
    CommLink* links{nullptr};
    HcclResult hcclRet;
    aclnnStatus aclnnRet;
    uint32_t netLayerNum{0};
    uint32_t netLayers{0};
    uint32_t* netLayerList{nullptr};

    hcclRet = g_HcclRankGraphGetLayers(hcclHandle, &netLayerList, &netLayerNum);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get HCCL Rank Graph Layers failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL Rank Graph Layers Success, netLayerNum:%d", netLayerNum);
    netLayers = netLayerNum - 1; // 组网拓扑层级，从0开始，则最大为num - 1

    hcclRet = HcclChannelDescInit(channelDesc.data(), channelNum);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Hccl Channel Init failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Hccl Channel Init Success");

    for (uint32_t channelIndex = 0; channelIndex < rankDim; channelIndex++) {
        if (channelIndex == srcRankId) { // 本卡不需要初始化channel,本卡hccl buffer地址调用接口获取
            continue;
        }
        uint32_t dstRankId = channelIndex; // 目标rank id
        uint32_t channelId = channelIndex > srcRankId ? channelIndex - 1 : channelIndex; // 通道id,比本卡Id大的卡Id全部左移一位
        aclnnRet = GetHcclCommLink(hcclHandle, netLayers, srcRankId, dstRankId, protocol, links); // 遍历组网支持的所有通信协议
        CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
        channelDesc[channelId].channelProtocol = protocol; // 通信协议
        channelDesc[channelId].remoteRank = dstRankId; // 目标rank id
        channelDesc[channelId].notifyNum = channelNum; // 通信的notify数量
        channelDesc[channelId].localEndpoint = links->srcEndpointDesc;
        channelDesc[channelId].remoteEndpoint = links->dstEndpointDesc;
    }
    hcclRet = g_HcclChannelAcquire(hcclHandle, engine, channelDesc.data(), channelNum, channeles.data());
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get HCCL Channel Resource failed, engine is:%d", engine);
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL Channel Resource success");
    return ACLNN_SUCCESS;
}

static aclnnStatus GetHcclCommResource(const HcclComm hcclHandle, const CommEngine& engine,
                                       const CommProtocol& protocol, Mc2MoeContext* mc2ContextStruct)
{
    OP_LOGD("Start to get HCCL communication resource");
    HcclResult hcclRet;
    void* tempBuffer = nullptr;
    uint64_t buffersize = 0;
    uint32_t rankId = mc2ContextStruct->epRankId;
    uint32_t epRankSize = mc2ContextStruct->epRankSize;
    std::vector<ChannelHandle> channels;
    // 获取HCCL通信通道，后续获取远端通信地址时需要使用
    auto aclnnRet = GetHcclCommChannel(hcclHandle, epRankSize, rankId, protocol, engine, channels);
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
    OP_LOGD("Create HCCL Communication Channel success");
    for (uint32_t rankIdIndex = 0; rankIdIndex < epRankSize; rankIdIndex++) { // 获取对应的每张卡的通信地址
        if (rankIdIndex == rankId) { // 获取本卡的通信地址
            hcclRet = g_HcclGetHcclBuffer(hcclHandle, &tempBuffer, &mc2ContextStruct->winSize);
        } else { // 获取其他卡的通信地址
            uint32_t channelIndex = rankIdIndex < rankId ? rankIdIndex : rankIdIndex - 1;
            hcclRet = g_HcclChannelGetHcclBuffer(hcclHandle, channels[channelIndex], &tempBuffer, &buffersize);
        }
        if (hcclRet != HCCL_SUCCESS) {
            OP_LOGE(ACLNN_ERR_INNER, "Get Hccl Communicate Buffer Failed,\
                    srcRankId:%d, dstRankId:%d", rankId, rankIdIndex);
            return ACLNN_ERR_INNER;
        }
        mc2ContextStruct->epHcclBuffer_[rankIdIndex] =  reinterpret_cast<uint64_t>(tempBuffer);
    }
    OP_LOGD("Get Hccl Communicate Buffer Success");
    return ACLNN_SUCCESS;
}

static aclnnStatus CreatMc2Context(const HcclComm hcclHandle, const std::string& mc2ContextTag,
                                   const CommEngine& engine, const CommProtocol& protocol,
                                   void* &ctx, Mc2MoeContext*  mc2ContextStruct)
{
    OP_LOGD("Start to create MC2 Context");
    HcclResult hcclRet;
    aclnnStatus aclnnRet;
    uint64_t dstCtxOffset = KopyDefaultCtxOffset; // 拷贝地址的偏移量，即将数据从host侧拷贝到device侧时从哪个偏移地址开始拷贝，默认从0开始，全部拷贝
    uint64_t ctxSize = sizeof(Mc2MoeContext); // 需要分配的context结构体大小

    hcclRet = g_HcclEngineCtxCreate(hcclHandle, mc2ContextTag.c_str(), engine, ctxSize, &ctx);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get HCCL Context Memory failed, mc2ContextTag is:%s, engine is:%d",\
                mc2ContextTag.c_str(), engine);
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL Context Memory success");
    hcclRet = g_HcclGetRankId(hcclHandle, &mc2ContextStruct->epRankId); // 获取本卡ID
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get HCCL Rank Id failed");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL Rank Id success");
    hcclRet = g_HcclGetRankSize(hcclHandle, &mc2ContextStruct->epRankSize); // 获取通讯域存在多少卡
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Hccl Get Rank Size failed.");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("Get HCCL Rank Size success");
    aclnnRet = GetHcclCommResource(hcclHandle, engine, protocol, mc2ContextStruct); // 获取所需的HCCL通信资源，即通信buffer地址
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
    // 把host对应的数据拷贝到device侧
    hcclRet = g_HcclEngineCtxCopy(hcclHandle, engine, mc2ContextTag.c_str(), mc2ContextStruct, ctxSize, dstCtxOffset);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Failed to copy Mc2MoeContext from host to device");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("CreatMc2Context Success");
    return ACLNN_SUCCESS;
}

static aclnnStatus CreatMc2ContextTensor(void* ctx, aclTensor* &mc2Context)
{
    OP_LOGD("Start to create Mc2Context Tensor");
    if (ctx == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "Create Mc2Context Tensor failed ctx is nullptr.");
        return ACLNN_ERR_INNER;
    }
    uint64_t mc2ContextLength = sizeof(Mc2MoeContext);
    int64_t shap[1] = {mc2ContextLength / sizeof(uint32_t)}; // 默认1维
    int64_t strides[1] = {1};
    mc2Context = aclCreateTensor(
        shap, 1, aclDataType::ACL_INT32, strides, 0,
        aclFormat::ACL_FORMAT_ND, shap, 1, ctx); // 创建mc2Context Tensor
    if (mc2Context == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "Create Mc2Context Tensor failed.");
        return ACLNN_ERR_INNER;
    }
    OP_LOGD("CreatMc2ContextTensor Success");
    return ACLNN_SUCCESS;
}

static aclnnStatus GetHcclBufferSize(const HcclComm& hcclHandle, uint64_t& hcclBuffSize)
{
    // 多轮次调用的时候，host侧没有保存mc2contxet结构体的数据，需要重新获取hccl buffer 大小
    void* tempBuffer = nullptr;
    auto hcclRet = g_HcclGetHcclBuffer(hcclHandle, &tempBuffer, &hcclBuffSize);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get HCCL Buffer Size failed");
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}


static aclnnStatus GetCommEngine(const HcclComm& hcclHandle, CommEngine& engine, CommProtocol& protocol)
{
    OP_LOGD("Start to get CommEngine");
    aclnnStatus aclnnRet;
    uint32_t netLayerNum{0};
    uint32_t* netLayerList{nullptr};

    auto hcclRet = g_HcclRankGraphGetLayers(hcclHandle, &netLayerList, &netLayerNum);
    if (hcclRet != HCCL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER, "Get HCCL Rank Graph Layers failed");
        return ACLNN_ERR_INNER;
    }
    if (netLayerNum == HCCL_COMM_LAYERS_MTE_CCU) { // 目前CCU走原有方案,通过别的方式判断是否走CCU,目前只支持MTE
        engine = CommEngine::COMM_ENGINE_AIV;
        protocol = CommProtocol::COMM_PROTOCOL_UB_MEM;
        OP_LOGD("Get CommEngine Success engine is:%d, protocol is:%d", engine, protocol);
        return ACLNN_SUCCESS;
    }
    OP_LOGE(ACLNN_ERR_INNER, "Current version only supports MTE communication mode, other modes are not supported.");
    return ACLNN_ERR_INNER;
}

extern inline aclnnStatus GetMc2ContextTensor(const char* groupEp, const char* opName, uint64_t& hcclBuffSize, aclTensor* &mc2Context)
{
    OP_LOGD("Start to get Mc2MoeContext Tensor");
    void* ctx{nullptr};
    uint64_t ctxSize{0};
    CommEngine engine;
    CommProtocol protocol;
    Mc2MoeContext mc2ContextStruct;
    std::string mc2ContextTag = std::string(groupEp) + std::string(opName);
    if (mc2ContextTag.size() > MaxContextTagSize) { // 检查上下文标签大小最大允许255个字符串长度
        OP_LOGE(ACLNN_ERR_INNER, "Mc2ContextTag is too long, max size is %d, but current size is %d",\
                MaxContextTagSize, mc2ContextTag.size());
        return ACLNN_ERR_INNER;
    }
    HcclComm hcclHandle;
    aclnnStatus aclnnRet = GetCommHandle(groupEp, hcclHandle); // 根据EP域名称获取通信上下文句柄
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
    aclnnRet = GetCommEngine(hcclHandle, engine, protocol); // 获取对应的通信引擎和通信协议
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
    auto hcclRet = g_HcclEngineCtxGet(hcclHandle, mc2ContextTag.c_str(), engine, &ctx, &ctxSize); // 检查是否已经存在context结构体，存在则直接跳过，否则进行创建
    if(hcclRet != HCCL_SUCCESS) {
        aclnnRet = CreatMc2Context(hcclHandle, mc2ContextTag, engine, protocol, ctx, &mc2ContextStruct); // 不存在context结构体，进行创建
        CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
        hcclBuffSize = mc2ContextStruct.winSize; // 从mc2结构体中获取对应的buffer大小
    } else {
        aclnnRet = GetHcclBufferSize(hcclHandle, hcclBuffSize); // 已经存在context结构体，则hccl buffer的大小单独进行获取
        CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
    }
    aclnnRet = CreatMc2ContextTensor(ctx, mc2Context); // 创建对应的输入aclnn Tensor
    CHECK_RET(aclnnRet == ACLNN_SUCCESS, aclnnRet);
    OP_LOGD("Get Mc2MoeContext Tensor Success");
    return ACLNN_SUCCESS;
}

static string RealPath(const string& path)
{
    string res;
    char resolved_path[PATH_MAX] = {};
    const char* p = realpath(path.c_str(), resolved_path);
    if (p == nullptr) {
        res.assign("");
    } else {
        res.assign(resolved_path);
    }
    return res;
}

// 获取.so路径
static bool GetBuiltinLibPath(string &libPath)
{
    string libPathEnv;
    const char *currHomePath = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, currHomePath);
    if (currHomePath) {
        const std::string kernelPath = std::string(currHomePath) + "/lib64";
        libPathEnv = RealPath(kernelPath);
        OP_CHECK(libPathEnv.empty(),
                 OP_LOGI("lib kernel path %s", libPathEnv.c_str()),
                 return true);
    }

    OP_LOGD("print GetBuiltinLibPath func test\n");
    const char *currLibPath = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, currLibPath);
    OP_CHECK(currLibPath != nullptr, OP_LOGW("ASCEND_LIB_PATH not config."), return false);
    libPathEnv = currLibPath;

    OP_CHECK(!libPathEnv.empty(), OP_LOGW("ASCEND_LIB_PATH is empty"), return false);

    libPath = libPathEnv;
    string filePath = RealPath(libPath);
    OP_CHECK(!filePath.empty(),
        OP_LOGW("ASCEND_OPP_PATH is invalid, path is: %s", libPathEnv.c_str()),
        return false);
    OP_LOGI("ASCEND_OPP_PATH is: %s", libPathEnv.c_str());
    if (libPath.back() != '/') {
        libPath += '/';
    }
    return true;
}

// 加载 libhcomm.so 并链接 Hccl 接口函数指针
extern inline aclnnStatus LoadBuiltinOpApi(const std::string &libPath)
{
    OP_LOGI("Load op api in path: %s", libPath.c_str());
    string soRealPath = libPath + "libhcomm.so";
    void *hcclHandle = dlopen(soRealPath.c_str(), RTLD_NOW | RTLD_GLOBAL);

    if (hcclHandle == nullptr) {
        OP_LOGW("Load op hcomm so failed, so path: %s", soRealPath.c_str());
        return ACLNN_ERR_INNER;
    }
    
    OP_LOGI("Load op hcomm so successfully, so path: %s", soRealPath.c_str());
    LOAD_HCCL_SYMBOL(hcclHandle, g_HcomGetCommHandleByGroup, "HcomGetCommHandleByGroup");
    LOAD_HCCL_SYMBOL(hcclHandle, g_HcclRankGraphGetLinks, "HcclRankGraphGetLinks");
    LOAD_HCCL_SYMBOL(hcclHandle, g_HcclRankGraphGetLayers, "HcclRankGraphGetLayers");
    LOAD_HCCL_SYMBOL(hcclHandle, g_HcclChannelAcquire, "HcclChannelAcquire");
    LOAD_HCCL_SYMBOL(hcclHandle, g_HcclGetHcclBuffer, "HcclGetHcclBuffer");
    LOAD_HCCL_SYMBOL(hcclHandle, g_HcclChannelGetHcclBuffer, "HcclChannelGetHcclBuffer");
    LOAD_HCCL_SYMBOL(hcclHandle, g_HcclEngineCtxCreate, "HcclEngineCtxCreate");
    LOAD_HCCL_SYMBOL(hcclHandle, g_HcclEngineCtxGet, "HcclEngineCtxGet");
    LOAD_HCCL_SYMBOL(hcclHandle, g_HcclEngineCtxCopy, "HcclEngineCtxCopy");
    LOAD_HCCL_SYMBOL(hcclHandle, g_HcclGetRankId, "HcclGetRankId");
    LOAD_HCCL_SYMBOL(hcclHandle, g_HcclGetRankSize, "HcclGetRankSize");
    OP_LOGD("All Hccl symbols loaded successfully");

    return ACLNN_SUCCESS;
}

}