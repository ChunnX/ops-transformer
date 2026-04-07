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
 * \file mc2_hcom_topo_info.h
 * \brief
 */

#ifndef MC2_HCOM_TOPOLOGY_H
#define MC2_HCOM_TOPOLOGY_H

#include <memory>
#include <string>
#include "hccl/hcom.h"
#include "mc2_moe_context.h"
#include "opdev/platform.h"
#include "hccl/hccl_rank_graph.h"
#include "mmpa/mmpa_api.h"
#include "opdev/op_errno.h"
#include "acl/acl.h"

static constexpr uint32_t COMM_ALG_DEFAULT = 0U;
static constexpr uint32_t COMM_MESH = 0b1U;
static constexpr uint32_t COMM_SWITCH = (COMM_MESH << 1U);
static constexpr uint32_t COMM_RING = (COMM_MESH << 2U);
static constexpr uint32_t COMM_PAIRWISE = (COMM_MESH << 3U);
static constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
namespace Mc2Hcom {
using namespace Mc2Aclnn;
class MC2HcomTopology {
public:
    static HcclResult CommGetInstSizeByGroup(const char *group, uint32_t *rankNum);
    static HcclResult TryGetGroupTopoType(const char *group, uint32_t *topoType);
    static HcclResult CommGetCclBufferSizeByGroup(const char *group, uint64_t *cclBufferSize, HcclComm *hcclComm);
    static HcclResult CommGetGroupLocalWindowSize(const char *group, uint64_t* cclBufferSize);
    
    // 新增的静态接口，通过实例内部的hcclHandle_获取数据
    static aclnnStatus GetMc2ContextTensor(const char* groupEp, const char* opName,
                                         uint64_t& hcclBuffSize, aclTensor*& mc2Context);

private:
    static MC2HcomTopology &GetInstance();
    explicit MC2HcomTopology(const char *libPath);
    ~MC2HcomTopology();
    
    // 内部辅助方法
    aclnnStatus LoadHcclSymbols();
    aclnnStatus GetCommHandle(const char* groupEp);
    aclnnStatus GetHcclCommLink(uint32_t netLayers, uint32_t srcRankId,
                              uint32_t dstRankId, const CommProtocol& protocol, CommLink*& links);
    aclnnStatus GetHcclCommChannel(uint32_t rankDim, uint32_t srcRankId,
                                 const CommProtocol& protocol, const CommEngine& engine, std::vector<ChannelHandle>& channels);
    aclnnStatus GetHcclCommResource(const CommEngine& engine,
                                  const CommProtocol& protocol, Mc2MoeContext* mc2ContextStruct);
    aclnnStatus CreatMc2Context(const std::string& mc2ContextTag,
                              const CommEngine& engine, const CommProtocol& protocol, void*& ctx, Mc2MoeContext* mc2ContextStruct);
    aclnnStatus CreatMc2ContextTensor(void* ctx, aclTensor*& mc2Context);
    aclnnStatus GetHcclBufferSize(uint64_t& hcclBuffSize);
    aclnnStatus GetCommEngine(CommEngine& engine, CommProtocol& protocol);
    aclnnStatus GetaclCreateTensorSymbol();

    HcclResult CallHcomGetCommHandleByGroup(const char *group, HcclComm *commHandle) const;
#ifdef BUILD_OPEN_PROJECT
    HcclResult CallHcomGetRankSizeEx(const char *group, uint32_t *ranksize, uint32_t flag) const;
    HcclResult CallHcomGetL0TopoTypeEx(const char *group, CommTopo *topoType, uint32_t flag) const;
#else
    HcclResult CallCommGetNetLayers(HcclComm comm, uint32_t **netLayer, uint32_t *netLayerNum) const;
    HcclResult CallCommGetInstTopoTypeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t *topoType) const;
    HcclResult CallCommGetInstSizeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t *rankNum) const;
#endif
    HcclResult CallCommGetCCLBufSizeCfg(HcclComm comm, uint64_t *cclBufferSize) const;

    void *handle_ = nullptr;
    void* hcclLibHandle_ = nullptr;
    HcclComm hcclHandle_ = nullptr;  // 实例变量存储HCCL句柄
    bool isNewHcclLib = true;
    std::string CreateTensorLibPath_;
    
    using FuncGetHandle = HcclResult (*)(const char *, HcclComm *);
    using FuncGetCclBufferSize = HcclResult (*)(HcclComm, uint64_t *);
    
    // HCCL新API函数指针
    HcclResult (*HcomGetCommHandleByGroup)(const char*, HcclComm*) = nullptr;
    HcclResult (*HcclRankGraphGetLinks)(HcclComm, uint32_t, uint32_t, uint32_t, CommLink**, uint32_t*) = nullptr;
    HcclResult (*HcclRankGraphGetLayers)(HcclComm, uint32_t**, uint32_t*) = nullptr;
    HcclResult (*HcclChannelAcquire)(HcclComm, const CommEngine&, const HcclChannelDesc*, uint32_t, ChannelHandle*) = nullptr;
    HcclResult (*HcclGetHcclBuffer)(HcclComm, void**, uint64_t*) = nullptr;
    HcclResult (*HcclChannelGetHcclBuffer)(HcclComm, const ChannelHandle&, void**, uint64_t*) = nullptr;
    HcclResult (*HcclEngineCtxCreate)(HcclComm, const char*, const CommEngine&, uint64_t, void**) = nullptr;
    HcclResult (*HcclEngineCtxGet)(HcclComm, const char*, const CommEngine&, void**, uint64_t*) = nullptr;
    HcclResult (*HcclEngineCtxCopy)(HcclComm, const CommEngine&, const char*, const void*, uint64_t, uint64_t) = nullptr;
    HcclResult (*HcclGetRankId)(HcclComm, uint32_t*) = nullptr;
    HcclResult (*HcclGetRankSize)(HcclComm, uint32_t*) = nullptr;
    aclTensor* (*aclCreateTensor)(const int64_t *viewDims, uint64_t viewDimsNum, aclDataType dataType, const int64_t *stride,
                               int64_t offset, aclFormat format, const int64_t *storageDims, uint64_t storageDimsNum, void *tensorData) = nullptr;

#ifdef BUILD_OPEN_PROJECT
    using FuncGetRankSizeEx = HcclResult (*)(const char *group, uint32_t *rankSize, uint32_t flag);
    using FuncGetL0TopoTypeEx = HcclResult (*)(const char *group, CommTopo *topoType, uint32_t flag);
#else
    using FuncGetNetLayers = HcclResult (*)(HcclComm, uint32_t **, uint32_t *);
    using FuncGetTopoType = HcclResult (*)(HcclComm, uint32_t, uint32_t *);
    using FuncGetInstSize = HcclResult (*)(HcclComm, uint32_t, uint32_t *);
#endif
    
    FuncGetHandle getCommHandle_ = nullptr;
#ifdef BUILD_OPEN_PROJECT
    FuncGetRankSizeEx getRankSizeEx_ = nullptr;
    FuncGetL0TopoTypeEx getL0TopoTypeEx_ = nullptr;
#else
    FuncGetNetLayers getNetLayers_ = nullptr;
    FuncGetTopoType getTopoType_ = nullptr;
    FuncGetInstSize getInstSize_ = nullptr;
#endif
    FuncGetCclBufferSize getCclBufferSize_ = nullptr;
};
}  // namespace Mc2Hcom
#endif