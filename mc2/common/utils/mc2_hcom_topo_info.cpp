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
 * \file mc2_hcom_topo_info.cc
 * \brief
 */

#include <cstdlib>
#include <string>
#include <dlfcn.h>
#include "mc2_log_compat.h"
#include "mc2_hcom_topo_info.h"
#include "ops_legacy/op_tiling/hcom_topo_info.h"
#include "mc2_tiling_utils.h"
#ifndef BUILD_OPEN_PROJECT
#include "hcom/hcom_topo_info.h"
#endif

namespace {
constexpr size_t CCL_BUFFER_RATIO = 2;
} // namespace

namespace Mc2Hcom {
#if defined(ASC_DEVKIT_VERSION_NUM) && (ASC_DEVKIT_VERSION_NUM >= 90000000)
const std::string HCOM_GET_COMM_FUNC_NAME = "HcclCommGetHandleWithName";
#else
const std::string HCOM_GET_COMM_FUNC_NAME = "HcomGetCommHandleByGroup";
#endif
const std::string HCCL_GET_RANK_SIZE_NAME = "HcclGetRankSize";
const std::string HCCL_GET_RANK_ID_NAME = "HcclGetRankId";
const std::string HCCL_GET_NET_LAYERS_NAME = "HcclRankGraphGetLayers";
const std::string HCCL_GET_TOPO_TYPE_NAME = "HcclRankGraphGetTopoTypeByLayer";
const std::string HCCL_GET_RANKS_BY_LAYER_NAME = "HcclRankGraphGetRanksByLayer";
const std::string HCCL_GET_LINKS_NAME = "HcclRankGraphGetLinks";
const std::string HCCL_GET_TOPO_INSTS_BY_LAYER_NAME = "HcclRankGraphGetTopoInstsByLayer";
const std::string HCCL_GET_TOPO_TYPE_BY_INST_NAME = "HcclRankGraphGetTopoType";
const std::string HCCL_GET_RANKS_BY_TOPO_INST_NAME = "HcclRankGraphGetRanksByTopoInst";
const std::string HCCL_GET_INST_SIZE_LIST_NAME = "HcclRankGraphGetInstSizeListByLayer";
#ifndef BUILD_OPEN_PROJECT
const std::string HCCL_GET_SIZE_NAME = "HcclRankGraphGetRankSizeByLayer";
#endif
const std::string COMM_GET_HCCL_BUFFER_NAME = "HcclGetHcclBuffer";

static const string GetLibPath()
{
    const char* ascendPath = std::getenv("ASCEND_HOME_PATH");
    if (ascendPath == nullptr) {
        OP_LOGE("", "Ascend home path doesn't exist.");
        return "";
    }
#if defined(__x86_64__)
    std::string hcclPathPostfix = "/x86_64-linux/lib64/libhccl_fwk.so";
#elif defined(__aarch64__)
    std::string hcclPathPostfix = "/aarch64-linux/lib64/libhccl_fwk.so";
#else
    return "";
#endif
    std::string fullPath = ascendPath + hcclPathPostfix;
    OP_LOGI("", "Loading lib in path %s.", fullPath.c_str());
    return fullPath;
}

template <typename T>
T GetHcclLibFunc(void* handle, const std::string& funcName)
{
    T func = reinterpret_cast<T>(dlsym(handle, funcName.c_str()));
    if (func == nullptr) {
        OP_LOGE("", "Load func=%s error=%s in lib hccl failed.", funcName.c_str(), dlerror());
    }
    return func;
}

namespace {
// L0组网形态的降级推导(按本rank实例类型+PCIE混联近似，用于拓扑实例聚合接口缺失的旧库)：
// 1DMESH与CUSTOM均按mesh族类型处理(部分mesh组网环境的L0实例会上报CUSTOM)；
// mesh族叠加PCIE混联判为MESH_1D_CLOS；CLOS及其余未识别形态按默认CLOS处理
Mc2Level0Topo DeriveLevel0TopoImpl(const CommTopo topoType, const bool pcieMix)
{
    switch (topoType) {
        case COMM_TOPO_1DMESH:
        case COMM_TOPO_CUSTOM:
            return pcieMix ? Mc2Level0Topo::MESH_1D_CLOS : Mc2Level0Topo::MESH_1D;
        case COMM_TOPO_CLOS:
            return Mc2Level0Topo::CLOS;
        default:
            return Mc2Level0Topo::CLOS;
    }
}

// 由L0实例结构事实推导组网形态汇总值(仅用于日志展示与comm_sets映射，不参与算法表匹配)：
// 1DMESH实例覆盖L0→MESH_1D；CLOS实例覆盖L0且有1DMESH实例→MESH_1D_CLOS；
// CLOS实例覆盖L0→CLOS；其余结构默认CLOS
Mc2Level0Topo SummarizeLevel0Topo(const Mc2TopoInfo& topoInfo)
{
    constexpr uint32_t BIT_CLOS = 1U << COMM_TOPO_CLOS;
    constexpr uint32_t BIT_1DMESH = 1U << COMM_TOPO_1DMESH;
    if ((topoInfo.l0CoverTypeSet & BIT_1DMESH) != 0) {
        return Mc2Level0Topo::MESH_1D;
    }
    if ((topoInfo.l0CoverTypeSet & BIT_CLOS) != 0) {
        return (topoInfo.l0InstTypeSet & BIT_1DMESH) != 0 ? Mc2Level0Topo::MESH_1D_CLOS : Mc2Level0Topo::CLOS;
    }
    return Mc2Level0Topo::CLOS;
}
} // namespace

// 扫描L0实例内本rank到各对端的链路，按CommProtocol枚举值拼协议位图(任意协议类型均进入位图，
// 消除"同为1DMESH拓扑但介质不同"的碰撞)；查询能力缺失或查询失败时保守返回0
uint32_t MC2HcomTopology::ScanL0LinkProtocols(HcclComm comm, uint32_t layer0Id,
                                              const std::vector<uint32_t>& level0Ranks) const
{
    if (level0Ranks.empty()) {
        return 0;
    }
    uint32_t myRank = 0;
    if (CallCommGetRankId(comm, &myRank) != HCCL_SUCCESS) {
        OP_LOGW("", "[ScanL0LinkProtocols] Failed to get rank id, protocol set is empty.");
        return 0;
    }
    uint32_t protoSet = 0;
    for (const uint32_t peerRank : level0Ranks) {
        if (peerRank == myRank) {
            continue;
        }
        CommLink* links = nullptr;
        uint32_t linkNum = 0;
        if (CallCommGetLinks(comm, layer0Id, myRank, peerRank, &links, &linkNum) != HCCL_SUCCESS || links == nullptr) {
            continue;
        }
        for (uint32_t idx = 0; idx < linkNum; ++idx) {
            const uint32_t protocol = static_cast<uint32_t>(links[idx].linkAttr.linkProtocol);
            if (protocol >= MC2_PROTO_BITMAP_WIDTH) {
                OP_LOGW("", "[ScanL0LinkProtocols] Link protocol %u exceeds bitmap width %u, ignored.", protocol,
                        MC2_PROTO_BITMAP_WIDTH);
                continue;
            }
            protoSet |= (1U << protocol);
        }
    }
    return protoSet;
}

Mc2Level0Topo MC2HcomTopology::DeriveLevel0Topo(const CommTopo topoType, const bool pcieMix)
{
    return DeriveLevel0TopoImpl(topoType, pcieMix);
}

// 按拓扑实例聚合拼接L0实例结构事实：
// 遍历L0本rank所属的拓扑实例，按CommTopo枚举值拼实例类型位图与覆盖位图(任意类型实例均进入位图)，
// 再经GetInstSizeListByLayer补全层视角的实例数与大小分布；
// 实例查询失败或L0实例rank数未知时返回false，由调用方退化为按本rank实例类型推导
bool MC2HcomTopology::CalcL0InstStruct(HcclComm comm, uint32_t layer0Id, uint32_t level0RankSize,
                                       Mc2TopoInfo& topoInfo) const
{
    uint32_t* topoInsts = nullptr;
    uint32_t topoInstNum = 0;
    if (CallCommGetTopoInstsByLayer(comm, layer0Id, &topoInsts, &topoInstNum) != HCCL_SUCCESS || topoInsts == nullptr ||
        topoInstNum == 0) {
        return false;
    }
    const std::vector<uint32_t> instIds(topoInsts, topoInsts + topoInstNum); // 及时拷贝，避免重复调用失效

    if (level0RankSize == 0) {
        return false; // L0实例rank数未知，无法做覆盖判定
    }
    for (const uint32_t instId : instIds) {
        uint32_t instTopoType = 0;
        uint32_t* ranks = nullptr;
        uint32_t rankNum = 0;
        if (CallCommGetTopoTypeByInst(comm, layer0Id, instId, &instTopoType) != HCCL_SUCCESS ||
            CallCommGetRanksByTopoInst(comm, layer0Id, instId, &ranks, &rankNum) != HCCL_SUCCESS) {
            return false;
        }
        if (instTopoType >= MC2_TOPO_TYPE_BITMAP_WIDTH) {
            OP_LOGW("", "[CalcL0InstStruct] Inst topo type %u exceeds bitmap width %u, ignored.", instTopoType,
                    MC2_TOPO_TYPE_BITMAP_WIDTH);
            continue;
        }
        topoInfo.l0InstTypeSet |= (1U << instTopoType);
        if (rankNum == level0RankSize) { // 该实例覆盖本rank所在L0网络实例的全部rank
            topoInfo.l0CoverTypeSet |= (1U << instTopoType);
        }
    }

    // 全层视角：实例数与各实例rank数分布(接口缺失时保持0/空，仅影响MULTI_INST与观测，不影响类型位图)
    uint32_t* instSizeList = nullptr;
    uint32_t listSize = 0;
    if (CallCommGetInstSizeListByLayer(comm, layer0Id, &instSizeList, &listSize) == HCCL_SUCCESS &&
        instSizeList != nullptr && listSize > 0) {
        topoInfo.l0InstNum = listSize;
        topoInfo.l0InstSizeList.assign(instSizeList, instSizeList + listSize); // 及时拷贝
    } else {
        OP_LOGW("", "[CalcL0InstStruct] Failed to get inst size list, l0InstNum unknown.");
    }
    return true;
}

// 进程级单例：符号解析(库句柄dlopen+dlsym)与失败告警每进程仅执行一次；
// 构造由GetInstance()的静态局部变量触发(线程安全)，析构不dlclose。
// 不dlclose的原因：本模块可能是libhccl_fwk.so的唯一持有者，dlclose会真实卸载该库及其驱动相关依赖，
// 反复卸载/重载会触发驱动侧会话反复初始化而报错，故库生命周期与进程一致
MC2HcomTopology& MC2HcomTopology::GetInstance()
{
    static MC2HcomTopology instance;
    return instance;
}

MC2HcomTopology::MC2HcomTopology()
{
    const std::string libPath = GetLibPath();
    if (libPath.empty()) {
        OP_LOGE("", "Ascend home path doesn't exist.");
        return;
    }
    handle_ = dlopen(libPath.c_str(), RTLD_NOW);
    if (handle_ == nullptr) {
        OP_LOGE("", "Load lib hccl failed, path=%s.", libPath.c_str());
        return;
    }

    getCommHandle_ = GetHcclLibFunc<FuncGetHandle>(handle_, HCOM_GET_COMM_FUNC_NAME);
    getNetLayers_ = GetHcclLibFunc<FuncGetNetLayers>(handle_, HCCL_GET_NET_LAYERS_NAME);
    getTopoTypeByLayer_ = GetHcclLibFunc<FuncGetTopoTypeByLayer>(handle_, HCCL_GET_TOPO_TYPE_NAME);
    getRanksByLayer_ = GetHcclLibFunc<FuncGetRanksByLayer>(handle_, HCCL_GET_RANKS_BY_LAYER_NAME);
    getLinks_ = GetHcclLibFunc<FuncGetLinks>(handle_, HCCL_GET_LINKS_NAME);
    getRankId_ = GetHcclLibFunc<FuncGetRankId>(handle_, HCCL_GET_RANK_ID_NAME);
    getRankSize_ = GetHcclLibFunc<FuncGetRankSize>(handle_, HCCL_GET_RANK_SIZE_NAME);
    getTopoInstsByLayer_ = GetHcclLibFunc<FuncGetTopoInstsByLayer>(handle_, HCCL_GET_TOPO_INSTS_BY_LAYER_NAME);
    getTopoTypeByInst_ = GetHcclLibFunc<FuncGetTopoTypeByInst>(handle_, HCCL_GET_TOPO_TYPE_BY_INST_NAME);
    getRanksByTopoInst_ = GetHcclLibFunc<FuncGetRanksByTopoInst>(handle_, HCCL_GET_RANKS_BY_TOPO_INST_NAME);
    getInstSizeList_ = GetHcclLibFunc<FuncGetInstSizeList>(handle_, HCCL_GET_INST_SIZE_LIST_NAME);
#ifndef BUILD_OPEN_PROJECT
    getInstSize_ = GetHcclLibFunc<FuncGetInstSize>(handle_, HCCL_GET_SIZE_NAME);
#endif
    getHcclBuffer_ = GetHcclLibFunc<FuncGetHcclBuffer>(handle_, COMM_GET_HCCL_BUFFER_NAME);

    if (getCommHandle_ == nullptr || getTopoTypeByLayer_ == nullptr || getHcclBuffer_ == nullptr) {
        OP_LOGE("", "Lib load new topo functions failed.");
        getCommHandle_ = nullptr;
        getNetLayers_ = nullptr;
        getTopoTypeByLayer_ = nullptr;
        getRanksByLayer_ = nullptr;
        getLinks_ = nullptr;
        getRankId_ = nullptr;
        getRankSize_ = nullptr;
        getTopoInstsByLayer_ = nullptr;
        getTopoTypeByInst_ = nullptr;
        getRanksByTopoInst_ = nullptr;
        getInstSizeList_ = nullptr;
#ifndef BUILD_OPEN_PROJECT
        getInstSize_ = nullptr;
#endif
        getHcclBuffer_ = nullptr;
        return;
    }
    if (getNetLayers_ == nullptr) {
        OP_LOGW("", "Lib load getNetLayers_ failed, layer query in CommGetTopoInfoByGroup may not work.");
    }
    if (getRanksByLayer_ == nullptr || getLinks_ == nullptr || getRankId_ == nullptr || getRankSize_ == nullptr) {
        OP_LOGW("", "Lib load topo assemble functions failed, CommGetTopoInfoByGroup may be degraded.");
    }
    if (getTopoInstsByLayer_ == nullptr || getTopoTypeByInst_ == nullptr || getRanksByTopoInst_ == nullptr ||
        getInstSizeList_ == nullptr) {
        OP_LOGW("", "Lib load topo inst functions failed, L0 inst aggregation may be degraded.");
    }
#ifndef BUILD_OPEN_PROJECT
    if (getInstSize_ == nullptr) {
        OP_LOGW("", "Lib load getInstSize_ failed, CommGetInstSizeByGroup may not work.");
    }
#endif

    OP_LOGI("", "Init MC2HcomTopology Success.");
}

MC2HcomTopology::~MC2HcomTopology()
{
    // 库句柄进程级常驻，析构不dlclose(原因见GetInstance注释)
    handle_ = nullptr;
}

HcclResult MC2HcomTopology::CallHcomGetCommHandleByGroup(const char* group, HcclComm* commHandle) const
{
    if (getCommHandle_ == nullptr) {
        OP_LOGE("", "Failed to get comm handle, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getCommHandle_(group, commHandle));
}

HcclResult MC2HcomTopology::CallCommGetNetLayers(HcclComm comm, uint32_t** netLayer, uint32_t* netLayerNum) const
{
    if (getNetLayers_ == nullptr) {
        OP_LOGE("", "Failed to get net layers, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getNetLayers_(comm, netLayer, netLayerNum));
}

HcclResult MC2HcomTopology::CallCommGetInstTopoTypeByNetLayer(HcclComm comm, uint32_t netLayer,
                                                              uint32_t* topoType) const
{
    if (getTopoTypeByLayer_ == nullptr) {
        OP_LOGE("", "Failed to get topo type, func load failed.");
        return HCCL_E_PTR;
    }
    CommTopo topoRet;
    HcclResult ret = static_cast<HcclResult>(getTopoTypeByLayer_(comm, netLayer, &topoRet));
    if (ret == HCCL_SUCCESS) {
        *topoType = static_cast<uint32_t>(topoRet);
    }
    return ret;
}

HcclResult MC2HcomTopology::CallCommGetRanksByLayer(HcclComm comm, uint32_t netLayer, uint32_t** ranks,
                                                    uint32_t* rankNum) const
{
    if (getRanksByLayer_ == nullptr) {
        OP_LOGE("", "Failed to get ranks by layer, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getRanksByLayer_(comm, netLayer, ranks, rankNum));
}

HcclResult MC2HcomTopology::CallCommGetLinks(HcclComm comm, uint32_t netLayer, uint32_t srcRankId, uint32_t dstRankId,
                                             CommLink** links, uint32_t* linkNum) const
{
    if (getLinks_ == nullptr) {
        OP_LOGE("", "Failed to get links, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getLinks_(comm, netLayer, srcRankId, dstRankId, links, linkNum));
}

HcclResult MC2HcomTopology::CallCommGetRankId(HcclComm comm, uint32_t* rankId) const
{
    if (getRankId_ == nullptr) {
        OP_LOGE("", "Failed to get rank id, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getRankId_(comm, rankId));
}

HcclResult MC2HcomTopology::CallCommGetRankSize(HcclComm comm, uint32_t* rankSize) const
{
    if (getRankSize_ == nullptr) {
        OP_LOGE("", "Failed to get rank size, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getRankSize_(comm, rankSize));
}

HcclResult MC2HcomTopology::CallCommGetTopoInstsByLayer(HcclComm comm, uint32_t netLayer, uint32_t** topoInsts,
                                                        uint32_t* topoInstNum) const
{
    if (getTopoInstsByLayer_ == nullptr) {
        OP_LOGE("", "Failed to get topo insts, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getTopoInstsByLayer_(comm, netLayer, topoInsts, topoInstNum));
}

HcclResult MC2HcomTopology::CallCommGetTopoTypeByInst(HcclComm comm, uint32_t netLayer, uint32_t topoInstId,
                                                      uint32_t* topoType) const
{
    if (getTopoTypeByInst_ == nullptr) {
        OP_LOGE("", "Failed to get topo type by inst, func load failed.");
        return HCCL_E_PTR;
    }
    CommTopo topoRet;
    HcclResult ret = static_cast<HcclResult>(getTopoTypeByInst_(comm, netLayer, topoInstId, &topoRet));
    if (ret == HCCL_SUCCESS) {
        *topoType = static_cast<uint32_t>(topoRet);
    }
    return ret;
}

HcclResult MC2HcomTopology::CallCommGetRanksByTopoInst(HcclComm comm, uint32_t netLayer, uint32_t topoInstId,
                                                       uint32_t** ranks, uint32_t* rankNum) const
{
    if (getRanksByTopoInst_ == nullptr) {
        OP_LOGE("", "Failed to get ranks by topo inst, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getRanksByTopoInst_(comm, netLayer, topoInstId, ranks, rankNum));
}

HcclResult MC2HcomTopology::CallCommGetInstSizeListByLayer(HcclComm comm, uint32_t netLayer, uint32_t** instSizeList,
                                                           uint32_t* listSize) const
{
    if (getInstSizeList_ == nullptr) {
        OP_LOGE("", "Failed to get inst size list, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getInstSizeList_(comm, netLayer, instSizeList, listSize));
}

#ifdef BUILD_OPEN_PROJECT
HcclResult MC2HcomTopology::CallHcomGetRankSizeEx(const char* group, uint32_t* ranksize) const
{
    if (getRankSize_ == nullptr) {
        OP_LOGE("", "Failed to get rank size, func load failed.");
        return HCCL_E_PTR;
    }
    HcclComm comm;
    HcclResult ret = CallHcomGetCommHandleByGroup(group, &comm);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Failed to get comm handle for rank size.");
        return ret;
    }
    return static_cast<HcclResult>(getRankSize_(comm, ranksize));
}
#else
HcclResult MC2HcomTopology::CallCommGetInstSizeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t* rankNum) const
{
    if (getInstSize_ == nullptr) {
        OP_LOGE("", "Failed to get inst size, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getInstSize_(comm, netLayer, rankNum));
}
#endif

HcclResult MC2HcomTopology::CallCommGetCCLBufSizeCfg(HcclComm comm, uint64_t* cclBufferSize) const
{
    if (getHcclBuffer_ == nullptr) {
        OP_LOGE("", "Failed to get hccl buffer, func load failed.");
        return HCCL_E_PTR;
    }
    void* buffer = nullptr;
    uint64_t size = 0;
    HcclResult ret = static_cast<HcclResult>(getHcclBuffer_(comm, &buffer, &size));
    if (ret == HCCL_SUCCESS) {
        *cclBufferSize = size / CCL_BUFFER_RATIO;
    }
    return ret;
}

HcclResult MC2HcomTopology::CallCommGetHcclBuffer(HcclComm comm, void** buffer, uint64_t* size) const
{
    if (getHcclBuffer_ == nullptr) {
        OP_LOGE("", "Failed to get inst size, func load failed.");
        return HCCL_E_PTR;
    }
    return static_cast<HcclResult>(getHcclBuffer_(comm, buffer, size));
}

HcclResult MC2HcomTopology::CommGetCclBufferSizeByGroup(const char* group, uint64_t* cclBufferSize, HcclComm* hcclComm)
{
    if (group == nullptr || cclBufferSize == nullptr || hcclComm == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("", "group, cclBufferSize, hcclComm");
        return HCCL_E_PTR;
    }
    MC2HcomTopology& topo = GetInstance();
    HcclResult ret = topo.CallHcomGetCommHandleByGroup(group, hcclComm);
    if (ret != HCCL_SUCCESS) {
        OP_LOGI("", "get nullptr comm handle.");
        *hcclComm = nullptr;
        return HCCL_SUCCESS;
    }
    ret = topo.CallCommGetCCLBufSizeCfg(*hcclComm, cclBufferSize);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Failed to get buffer size.");
        return ret;
    }
    OP_LOGI("", "cclBufferSize is %lu", *cclBufferSize);
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CommGetHcclBufferByGroup(const char* group, void** buffer, uint64_t* size)
{
    if (group == nullptr || buffer == nullptr || size == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("", "group, buffer, size");
        return HCCL_E_PTR;
    }
    HcclComm hcclComm;
    MC2HcomTopology& topo = GetInstance();
    HcclResult ret = topo.CallHcomGetCommHandleByGroup(group, &hcclComm);
    if (ret != HCCL_SUCCESS) {
        OP_LOGI("", "Failed to get comm handle.");
        hcclComm = nullptr;
        return ret;
    }
    ret = topo.CallCommGetHcclBuffer(hcclComm, buffer, size);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Failed to get buffer size.");
        return ret;
    }
    OP_LOGI("", "localBufferSize is %lu", *size);
    return HCCL_SUCCESS;
}

#ifdef BUILD_OPEN_PROJECT
HcclResult MC2HcomTopology::CommGetGroupLocalWindowSize(const char* group, uint64_t* cclBufferSize)
{
    if (ge::HcomTopoInfo::Instance().GetGroupLocalWindowSize(group, *cclBufferSize) != ge::GRAPH_SUCCESS) {
        OP_LOGD("", "Get winSize from GetGroupLocalWindowSize=%lu", *cclBufferSize);
        *cclBufferSize = mc2tiling::Mc2TilingUtils::GetMaxWindowSize();
        return HCCL_SUCCESS;
    }
    OP_LOGD("", "Get winSize from GetMaxWindowSize=%lu", *cclBufferSize);
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CommGetInstSizeByGroup(const char* group, uint32_t* rankNum)
{
    if (group == nullptr || rankNum == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("", "group, rankNum");
        return HCCL_E_PTR;
    }
    MC2HcomTopology& topo = GetInstance();
    HcclResult ret = topo.CallHcomGetRankSizeEx(group, rankNum);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Failed to get rank size.");
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::TryGetGroupTopoType(const char* group, uint32_t* topoType)
{
    if (group == nullptr || topoType == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("", "group, topoType");
        return HCCL_E_PTR;
    }
    // 基于一次性装配的组网整体信息推导，L0层号取自HcclRankGraphGetLayers返回的实际层号
    Mc2TopoInfo topoInfo;
    HcclResult ret = CommGetTopoInfoByGroup(group, topoInfo);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "Failed to get topo info.");
        return ret;
    }
    if (topoInfo.level0Topo == Mc2Level0Topo::UNKNOWN) {
        OP_LOGW("", "GroupTopoInfo not set.");
        return HCCL_E_NOT_FOUND;
    }
    // 映射到comm_sets取值空间(调用方按COMM_MESH/COMM_SWITCH语义比较)：
    // 纯1D Mesh判定为mesh；混联/CLOS等交换拓扑判定为非mesh，走分层/交换算法分支
    *topoType = (topoInfo.level0Topo == Mc2Level0Topo::MESH_1D) ? COMM_MESH : COMM_SWITCH;
    OP_LOGD("", "comm_sets from topo info is %u, COMM_MESH is %u", *topoType, COMM_MESH);
    return HCCL_SUCCESS;
}
#else
HcclResult MC2HcomTopology::CommGetGroupLocalWindowSize(const char* group, uint64_t* cclBufferSize)
{
    uint32_t ret = ge::HcomTopoInfo::Instance().GetGroupLocalWindowSize(group, *cclBufferSize);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE("", "cclBufferSize not set.");
        return HCCL_E_NOT_FOUND;
    }
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::CommGetInstSizeByGroup(const char* group, uint32_t* rankNum)
{
    int64_t rankSize = *rankNum;
    uint32_t ret = ge::HcomTopoInfo::Instance().GetGroupRankSize(group, rankSize);
    if (ret != ge::GRAPH_SUCCESS) {
        return HCCL_E_NOT_FOUND;
    }
    *rankNum = static_cast<uint32_t>(rankSize);
    return HCCL_SUCCESS;
}

HcclResult MC2HcomTopology::TryGetGroupTopoType(const char* group, uint32_t* topoType)
{
    ge::HcomTopoInfo::TopoInfo topoInfo;
    if (!ge::HcomTopoInfo::Instance().TryGetGroupTopoInfo(group, topoInfo)) {
        OP_LOGW("", "GroupTopoInfo not set.");
        *topoType = COMM_ALG_DEFAULT;
        return HCCL_E_NOT_FOUND;
    }
    *topoType = topoInfo.topo_level_descs[static_cast<int32_t>(ge::HcomTopoInfo::TopoLevel::L0)].comm_sets;
    OP_LOGD("", "comm_sets from TopoInfo is %u, COMM_MESH is %u", *topoType, COMM_MESH);
    return HCCL_SUCCESS;
}
#endif

// 一次性装配组网整体信息，接口用法要点：
// 1. HcclRankGraphGetLayers先取实际网络层号列表，后续所有层内查询均使用列表中的层号(L0层号不能假设为字面0，
//    ranktable可配置非连续层号如{0,3})；
// 2. 最小覆盖层数topoLevelNums：按层号顺序遍历，首个本rank所在实例覆盖通信域全部rank的层即停止；
// 3. L0实例结构事实按CommTopo/CommProtocol枚举值直接拼位图(实例类型/覆盖/实例数/链路协议)，作为算法匹配的
//    原始依据；组网形态汇总值level0Topo由结构事实推导(实验接口缺失时按本rank实例类型降级)，仅用于日志与comm_sets映射
HcclResult MC2HcomTopology::CommGetTopoInfoByGroup(const char* group, Mc2TopoInfo& topoInfo)
{
    if (group == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT("", "group");
        return HCCL_E_PTR;
    }
    topoInfo = Mc2TopoInfo{};
    MC2HcomTopology& topo = GetInstance();

    HcclComm comm;
    HcclResult ret = topo.CallHcomGetCommHandleByGroup(group, &comm);
    if (ret != HCCL_SUCCESS) {
        OP_LOGE("", "[CommGetTopoInfoByGroup] Failed to get comm handle for group=%s, ret=%d.", group,
                static_cast<int>(ret));
        return ret;
    }

    // 1. 实际网络层号列表(及时拷贝，同一通信域重复调用可能使前次结果失效)
    uint32_t* layerList = nullptr;
    uint32_t layerNum = 0;
    ret = topo.CallCommGetNetLayers(comm, &layerList, &layerNum);
    if (ret != HCCL_SUCCESS || layerList == nullptr || layerNum == 0) {
        OP_LOGE("", "[CommGetTopoInfoByGroup] Failed to get net layers for group=%s, ret=%d, layerNum=%u.", group,
                static_cast<int>(ret), layerNum);
        return (ret == HCCL_SUCCESS) ? HCCL_E_PTR : ret;
    }
    topoInfo.netLayers.assign(layerList, layerList + layerNum);
    const uint32_t layer0Id = topoInfo.netLayers[0]; // L0实际层号

    // 2. 通信域总rank数：优先HcclGetRankSize，能力缺失时退化为最高层实例rank数(最高层实例覆盖整个通信域)
    uint32_t userRankSize = 0;
    if (topo.CallCommGetRankSize(comm, &userRankSize) != HCCL_SUCCESS) {
        userRankSize = 0;
    }
    if (userRankSize == 0) {
        uint32_t* ranks = nullptr;
        uint32_t rankNum = 0;
        if (topo.CallCommGetRanksByLayer(comm, topoInfo.netLayers.back(), &ranks, &rankNum) == HCCL_SUCCESS &&
            ranks != nullptr) {
            userRankSize = rankNum;
        }
    }
    topoInfo.userRankSize = userRankSize;

    // 3. 按层装配：L0实例rank数(本server卡数) + 最小覆盖层数
    std::vector<uint32_t> level0Ranks;
    for (uint32_t idx = 0; idx < layerNum; ++idx) {
        const uint32_t layerId = topoInfo.netLayers[idx];
        uint32_t* ranks = nullptr;
        uint32_t rankNum = 0;
        if (topo.CallCommGetRanksByLayer(comm, layerId, &ranks, &rankNum) == HCCL_SUCCESS && ranks != nullptr) {
            if (idx == 0) { // 及时拷贝L0实例rank列表，供链路协议扫描使用
                level0Ranks.assign(ranks, ranks + rankNum);
                topoInfo.level0RankSize = rankNum;
            }
            topoInfo.topoLevelNums = idx + 1;
            if (userRankSize != 0 && rankNum == userRankSize) {
                break; // 本层实例已覆盖通信域全部rank，后续层不再计入
            }
        } else {
            OP_LOGW("", "[CommGetTopoInfoByGroup] Failed to get ranks by layer=%u for group=%s, layer count degraded.",
                    layerId, group);
            topoInfo.topoLevelNums = idx + 1;
        }
    }

    // 4. L0实例结构事实：优先按拓扑实例聚合拼接(算法匹配的原始依据)，
    //    实验接口缺失时结构事实保持0(具体算法表项不命中，落默认算法兜底)；
    //    组网形态汇总值level0Topo由结构事实或降级路径推导(仅日志/comm_sets映射用)
    topoInfo.l0LinkProtocols = topo.ScanL0LinkProtocols(comm, layer0Id, level0Ranks);
    topoInfo.level0PcieMix = (topoInfo.l0LinkProtocols & (1U << COMM_PROTOCOL_PCIE)) != 0;
    if (topo.CalcL0InstStruct(comm, layer0Id, topoInfo.level0RankSize, topoInfo)) {
        topoInfo.level0Topo = SummarizeLevel0Topo(topoInfo);
    } else {
        uint32_t rawTopoType = 0;
        ret = topo.CallCommGetInstTopoTypeByNetLayer(comm, layer0Id, &rawTopoType);
        if (ret != HCCL_SUCCESS) {
            OP_LOGW("", "[CommGetTopoInfoByGroup] Failed to get L0 topo type for group=%s, layer0Id=%u, ret=%d.", group,
                    layer0Id, static_cast<int>(ret));
        } else {
            topoInfo.level0Topo = DeriveLevel0Topo(static_cast<CommTopo>(rawTopoType), topoInfo.level0PcieMix);
        }
    }

    OP_LOGI("",
            "[CommGetTopoInfoByGroup] group=%s, userRankSize=%u, layerNum=%u, layer0Id=%u, topoLevelNums=%u, "
            "level0Topo=%u, l0InstTypeSet=0x%x, l0CoverTypeSet=0x%x, l0InstNum=%u, l0LinkProtocols=0x%x, "
            "level0RankSize=%u.",
            group, topoInfo.userRankSize, layerNum, layer0Id, topoInfo.topoLevelNums,
            static_cast<uint32_t>(topoInfo.level0Topo), topoInfo.l0InstTypeSet, topoInfo.l0CoverTypeSet,
            topoInfo.l0InstNum, topoInfo.l0LinkProtocols, topoInfo.level0RankSize);
    return HCCL_SUCCESS;
}
} // namespace Mc2Hcom
