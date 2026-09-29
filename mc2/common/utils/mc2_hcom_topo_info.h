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
#include <vector>
#include "hccl/hcom.h"

#ifdef BUILD_OPEN_PROJECT
#include "hccl/hccl_rank_graph.h"
#endif

static constexpr uint32_t COMM_ALG_DEFAULT = 0U;
static constexpr uint32_t COMM_MESH = 0b1U;
static constexpr uint32_t COMM_SWITCH = (COMM_MESH << 1U);
static constexpr uint32_t COMM_RING = (COMM_MESH << 2U);
static constexpr uint32_t COMM_PAIRWISE = (COMM_MESH << 3U);
static constexpr uint32_t COMM_IS_NOT_SET_DEVICE = 0;
namespace Mc2Hcom {
// L0组网形态，取值对齐HCCL参考实现的Level0Shape语义，由CommGetTopoInfoByGroup装配推导
enum class Mc2Level0Topo : uint32_t {
    CLOS = 0,              ///< CLOS/交换互联拓扑
    MESH_1D = 1,           ///< 1D Mesh互联拓扑
    MESH_1D_CLOS = 2,      ///< Mesh与CLOS混联拓扑(如PCIE混联机型)
    UNKNOWN = 0xFFFFFFFFU, ///< 查询失败或未装配
};

// 组网整体信息，由CommGetTopoInfoByGroup一次性装配(对齐参考实现TopoInfoWithNetLayerDetails中算法选择所需字段)
struct Mc2TopoInfo {
    uint32_t userRankSize = 0;       ///< 通信域内总rank数，查询失败为0
    std::vector<uint32_t> netLayers; ///< 实际网络层号列表(HcclRankGraphGetLayers返回值)
    uint32_t topoLevelNums = 0; ///< 最小覆盖层数：首个覆盖通信域全部rank的层计数，0表示未知
    Mc2Level0Topo level0Topo = Mc2Level0Topo::UNKNOWN; ///< L0组网形态
    bool level0PcieMix = false;                        ///< L0实例内是否存在PCIE协议链路(混联标记)
    uint32_t level0RankSize = 0;                       ///< L0实例内rank数(本server卡数)，0表示未知

    bool IsMeshTopo() const
    {
        return level0Topo == Mc2Level0Topo::MESH_1D || level0Topo == Mc2Level0Topo::MESH_1D_CLOS;
    }
};

class MC2HcomTopology {
public:
    static HcclResult CommGetInstSizeByGroup(const char* group, uint32_t* rankNum);
    static HcclResult TryGetGroupTopoType(const char* group, uint32_t* topoType);
    static HcclResult CommGetCclBufferSizeByGroup(const char* group, uint64_t* cclBufferSize, HcclComm* hcclComm);
    static HcclResult CommGetGroupLocalWindowSize(const char* group, uint64_t* cclBufferSize);
    static HcclResult CommGetHcclBufferByGroup(const char* group, void** buffer, uint64_t* size);
    static HcclResult CommGetNetLayersByGroup(const char* group, std::vector<uint32_t>& layers);
    static HcclResult CommGetTopoTypeByLayer(const char* group, uint32_t layer, uint32_t* topoType);
    static HcclResult CommGetTopoInfoByGroup(const char* group, Mc2TopoInfo& topoInfo);

private:
    // 构造时从进程级常驻库句柄解析HCCL符号并执行查询；库dlopen一次后进程内常驻不dlclose：
    // tiling进程内本模块可能是libhccl_fwk.so唯一持有者，dlclose会真实卸载该库及其驱动依赖链，
    // 下次dlopen重新执行库初始化(含dsmi设备枚举)，驱动侧会话未随卸载清理会导致dsmi报错；
    // mc2_context.cpp在op_api进程中的每次dlopen/dlclose因其余运行时组件持有该库而不会触发卸载，等效于常驻
    MC2HcomTopology();
    ~MC2HcomTopology();
    // 扫描L0实例内本rank到各对端的链路，存在PCIE协议链路则判定为混联机型；查询失败保守置false
    bool ScanLevel0PcieMix(HcclComm comm, uint32_t layer0Id, const std::vector<uint32_t>& level0Ranks) const;
    // 按拓扑实例聚合推导L0组网形态(对齐参考实现CalcLevel0TopoShape的实例结构判定)：
    // 单个1DMESH实例覆盖L0全部rank→MESH_1D；单个CLOS实例覆盖L0全部rank→CLOS；
    // 两个实例(1个CLOS覆盖L0全部rank+1个1DMESH)→MESH_1D_CLOS(UBX机型)；依赖按实例聚合的实验接口，
    // 能力缺失或L0实例rank数未知时返回false，由调用方退化为按本rank实例类型近似推导
    bool CalcLevel0TopoShapeByInst(HcclComm comm, uint32_t layer0Id, uint32_t level0RankSize,
                                   Mc2Level0Topo& level0Topo) const;
    // 由L0实例拓扑类型与PCIE混联标记推导L0组网形态的降级路径(对齐参考实现Level0Shape语义)
    static Mc2Level0Topo DeriveLevel0Topo(const CommTopo topoType, const bool pcieMix);
    HcclResult CallHcomGetCommHandleByGroup(const char* group, HcclComm* commHandle) const;
    HcclResult CallCommGetNetLayers(HcclComm comm, uint32_t** netLayer, uint32_t* netLayerNum) const;
    HcclResult CallCommGetInstTopoTypeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t* topoType) const;
    HcclResult CallCommGetRanksByLayer(HcclComm comm, uint32_t netLayer, uint32_t** ranks, uint32_t* rankNum) const;
    HcclResult CallCommGetLinks(HcclComm comm, uint32_t netLayer, uint32_t srcRankId, uint32_t dstRankId,
                                CommLink** links, uint32_t* linkNum) const;
    HcclResult CallCommGetRankId(HcclComm comm, uint32_t* rankId) const;
    HcclResult CallCommGetRankSize(HcclComm comm, uint32_t* rankSize) const;
    HcclResult CallCommGetTopoInstsByLayer(HcclComm comm, uint32_t netLayer, uint32_t** topoInsts,
                                           uint32_t* topoInstNum) const;
    HcclResult CallCommGetTopoTypeByInst(HcclComm comm, uint32_t netLayer, uint32_t topoInstId,
                                         uint32_t* topoType) const;
    HcclResult CallCommGetRanksByTopoInst(HcclComm comm, uint32_t netLayer, uint32_t topoInstId, uint32_t** ranks,
                                          uint32_t* rankNum) const;
#ifdef BUILD_OPEN_PROJECT
    HcclResult CallHcomGetRankSizeEx(const char* group, uint32_t* ranksize, uint32_t flag) const;
#else
    HcclResult CallCommGetInstSizeByNetLayer(HcclComm comm, uint32_t netLayer, uint32_t* rankNum) const;
#endif
    HcclResult CallCommGetCCLBufSizeCfg(HcclComm comm, uint64_t* cclBufferSize) const;
    HcclResult CallCommGetHcclBuffer(HcclComm comm, void** buffer, uint64_t* size) const;

    void* handle_ = nullptr;

    using FuncGetHandle = HcclResult (*)(const char*, HcclComm*);
    using FuncGetNetLayers = HcclResult (*)(HcclComm, uint32_t**, uint32_t*);
    using FuncGetTopoTypeByLayer = HcclResult (*)(HcclComm, uint32_t, CommTopo*);
    using FuncGetRanksByLayer = HcclResult (*)(HcclComm, uint32_t, uint32_t**, uint32_t*);
    using FuncGetLinks = HcclResult (*)(HcclComm, uint32_t, uint32_t, uint32_t, CommLink**, uint32_t*);
    using FuncGetRankId = HcclResult (*)(HcclComm, uint32_t*);
    using FuncGetRankSize = HcclResult (*)(HcclComm, uint32_t*);
    using FuncGetTopoInstsByLayer = HcclResult (*)(HcclComm, uint32_t, uint32_t**, uint32_t*);
    using FuncGetTopoTypeByInst = HcclResult (*)(HcclComm, uint32_t, uint32_t, CommTopo*);
    using FuncGetRanksByTopoInst = HcclResult (*)(HcclComm, uint32_t, uint32_t, uint32_t**, uint32_t*);
#ifndef BUILD_OPEN_PROJECT
    using FuncGetInstSize = HcclResult (*)(HcclComm, uint32_t, uint32_t*);
#endif
    using FuncGetHcclBuffer = HcclResult (*)(HcclComm, void**, uint64_t*);
    void* hcclHandle_ = nullptr;
    FuncGetHandle getCommHandle_ = nullptr;
    FuncGetNetLayers getNetLayers_ = nullptr;
    FuncGetTopoTypeByLayer getTopoTypeByLayer_ = nullptr;
    FuncGetRanksByLayer getRanksByLayer_ = nullptr;
    FuncGetLinks getLinks_ = nullptr;
    FuncGetRankId getRankId_ = nullptr;
    FuncGetRankSize getRankSize_ = nullptr;
    FuncGetTopoInstsByLayer getTopoInstsByLayer_ = nullptr;
    FuncGetTopoTypeByInst getTopoTypeByInst_ = nullptr;
    FuncGetRanksByTopoInst getRanksByTopoInst_ = nullptr;
#ifndef BUILD_OPEN_PROJECT
    FuncGetInstSize getInstSize_ = nullptr;
#endif
    FuncGetHcclBuffer getHcclBuffer_ = nullptr;
};
} // namespace Mc2Hcom
#endif
