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
#include "hccl/hcom.h"
#include "hccl/hccl_rank_graph.h"

static constexpr uint32_t COMM_ALG_DEFAULT = 0U;
static constexpr uint32_t COMM_MESH = 0b1U;
static constexpr uint32_t COMM_SWITCH = (COMM_MESH << 1U);
static constexpr uint32_t COMM_RING = (COMM_MESH << 2U);
static constexpr uint32_t COMM_PAIRWISE = (COMM_MESH << 3U);
static constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
namespace Mc2Hcom {
class MC2HcomTopology {
public:
    static HcclResult CommGetInstSizeByGroup(const char *group, uint32_t *rankNum);
    static HcclResult TryGetGroupTopoType(const char *group, uint32_t *topoType);
    static HcclResult CommGetCclBufferSizeByGroup(const char *group, uint64_t *cclBufferSize, HcclComm *hcclComm);
    static HcclResult CommGetGroupLocalWindowSize(const char *group, uint64_t* cclBufferSize);

private:
    static MC2HcomTopology &GetInstance();
    explicit MC2HcomTopology(const char *libPath);
    HcclResult CallHcomGetCommHandleByGroup(const char *group, HcclComm *commHandle) const;
    HcclResult CallCommGetNetLayers(HcclComm comm, uint32_t **netLayer, uint32_t *netLayerNum) const;
    HcclResult CallCommGetInstTopoTypeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t *topoType) const;
    HcclResult CallCommGetInstSizeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t *rankNum) const;
    HcclResult CallCommGetCCLBufSizeCfg(HcclComm comm, uint64_t *cclBufferSize) const;
    HcclResult CallHcomGetRankSizeEx(const char *group, uint32_t *ranksize, uint32_t flag) const;
    HcclResult CallHcomGetL0TopoTypeEx(const char *group, CommTopo *topoType, uint32_t flag) const;

    void *handle_ = nullptr;
    bool isNewHcclLib = true;

    using FuncGetHandle = HcclResult (*)(const char *, HcclComm *);
    using FuncGetNetLayers = HcclResult (*)(HcclComm, uint32_t **, uint32_t *);
    using FuncGetTopoType = HcclResult (*)(HcclComm, uint32_t, uint32_t *);
    using FuncGetInstSize = HcclResult (*)(HcclComm, uint32_t, uint32_t *);
    using FuncGetCclBufferSize = HcclResult (*)(HcclComm, uint64_t *);
    using FuncGetRankSizeEx = HcclResult (*)(const char *group, uint32_t *rankSize, uint32_t flag);
    using FuncGetL0TopoTypeEx = HcclResult (*)(const char *group, CommTopo *topoType, uint32_t flag);
    void *hcclHandle_ = nullptr;
    FuncGetHandle getCommHandle_ = nullptr;
    FuncGetNetLayers getNetLayers_ = nullptr;
    FuncGetTopoType getTopoType_ = nullptr;
    FuncGetInstSize getInstSize_ = nullptr;
    FuncGetCclBufferSize getCclBufferSize_ = nullptr;
    FuncGetRankSizeEx getRankSizeEx_ = nullptr;
    FuncGetL0TopoTypeEx getL0TopoTypeEx_ = nullptr;
};
}  // namespace Mc2Hcom
#endif