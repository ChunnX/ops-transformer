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
namespace Mc2Hcom {
// L0事实位图宽度：按CommTopo/CommProtocol枚举值置位的位图宽度，超出宽度的枚举值会被忽略并告警
// (该宽度同时决定算法表标志字中类型/协议区的掩码宽度，新枚举值超出时需同步扩宽)
constexpr uint32_t MC2_TOPO_TYPE_BITMAP_WIDTH = 8;
constexpr uint32_t MC2_PROTO_BITMAP_WIDTH = 16;
// L0组网形态汇总值：由L0实例结构事实推导(实验接口缺失时按本rank实例类型降级)，
// 仅用于日志展示与comm_sets映射，不参与算法表匹配(匹配走TOPO_FLAG_*实例结构事实位)
enum class Mc2Level0Topo : uint32_t {
    CLOS = 0,              ///< CLOS/交换互联拓扑
    MESH_1D = 1,           ///< 1D Mesh互联拓扑
    MESH_1D_CLOS = 2,      ///< Mesh与CLOS混联拓扑(如PCIE混联组网)
    UNKNOWN = 0xFFFFFFFFU, ///< 查询失败或未装配
};

// 组网整体信息，由CommGetTopoInfoByGroup一次性装配，供算法选择比较使用。
// L0实例结构事实按CommTopo/CommProtocol枚举值直接拼位图(实例级查询接口缺失时相应事实自动降级为0)，
// 任意已枚举类型/协议的实例均进入位图，新形态无需扩枚举即可被算法表匹配
struct Mc2TopoInfo {
    uint32_t userRankSize = 0;       ///< 通信域内总rank数，查询失败为0
    std::vector<uint32_t> netLayers; ///< 实际网络层号列表(HcclRankGraphGetLayers返回值)
    uint32_t topoLevelNums = 0; ///< 最小覆盖层数：首个覆盖通信域全部rank的层计数，0表示未知
    // L0实例结构事实(按拓扑实例聚合直接拼接，算法匹配的原始依据；实例聚合接口缺失时保持0)
    uint32_t l0InstTypeSet = 0; ///< L0实例类型存在位图(按CommTopo枚举值置位，本rank所属实例视角)
    uint32_t l0CoverTypeSet = 0; ///< L0覆盖类型位图(该类型实例的rank数等于L0网络实例rank数即覆盖，按CommTopo置位)
    uint32_t l0InstNum = 0;       ///< L0实例总数(全层视角，来自GetInstSizeListByLayer，0表示未知)
    uint32_t l0LinkProtocols = 0; ///< L0链路协议位图(本rank到L0实例内各对端，按CommProtocol枚举值置位)
    std::vector<uint32_t> l0InstSizeList;              ///< L0各实例rank数列表(全层视角)
    Mc2Level0Topo level0Topo = Mc2Level0Topo::UNKNOWN; ///< L0组网形态汇总值(日志/comm_sets映射用)
    bool level0PcieMix = false;  ///< L0是否存在PCIE协议链路(l0LinkProtocols的PCIE位便捷视图)
    uint32_t level0RankSize = 0; ///< L0实例内rank数(本server卡数)，0表示未知
};

class MC2HcomTopology {
public:
    static HcclResult CommGetInstSizeByGroup(const char* group, uint32_t* rankNum);
    static HcclResult TryGetGroupTopoType(const char* group, uint32_t* topoType);
    static HcclResult CommGetCclBufferSizeByGroup(const char* group, uint64_t* cclBufferSize, HcclComm* hcclComm);
    static HcclResult CommGetGroupLocalWindowSize(const char* group, uint64_t* cclBufferSize);
    static HcclResult CommGetHcclBufferByGroup(const char* group, void** buffer, uint64_t* size);
    static HcclResult CommGetTopoInfoByGroup(const char* group, Mc2TopoInfo& topoInfo);

private:
    // 构造时从进程级常驻库句柄解析HCCL符号并执行查询；库dlopen一次后进程内常驻不dlclose
    // (dlclose可能真实卸载libhccl_fwk.so及其驱动相关依赖，反复卸载/重载会触发驱动侧会话
    // 反复初始化而报错，故库生命周期与进程一致，见cpp中AcquireLibHandle)
    MC2HcomTopology();
    ~MC2HcomTopology();
    // 扫描L0实例内本rank到各对端的链路，按CommProtocol枚举值拼协议位图；查询失败返回0(保守)
    uint32_t ScanL0LinkProtocols(HcclComm comm, uint32_t layer0Id, const std::vector<uint32_t>& level0Ranks) const;
    // 按拓扑实例聚合拼接L0实例结构事实(l0InstTypeSet/l0CoverTypeSet/l0InstNum/l0InstSizeList)，
    // 依赖按实例聚合的实验接口，
    // 能力缺失或L0实例rank数未知时返回false，由调用方退化为按本rank实例类型推导
    bool CalcL0InstStruct(HcclComm comm, uint32_t layer0Id, uint32_t level0RankSize, Mc2TopoInfo& topoInfo) const;
    // 由L0实例拓扑类型与PCIE混联标记推导L0组网形态的降级路径
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
    HcclResult CallCommGetInstSizeListByLayer(HcclComm comm, uint32_t netLayer, uint32_t** instSizeList,
                                              uint32_t* listSize) const;
#ifdef BUILD_OPEN_PROJECT
    HcclResult CallHcomGetRankSizeEx(const char* group, uint32_t* ranksize) const;
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
    using FuncGetInstSizeList = HcclResult (*)(HcclComm, uint32_t, uint32_t**, uint32_t*);
#ifndef BUILD_OPEN_PROJECT
    using FuncGetInstSize = HcclResult (*)(HcclComm, uint32_t, uint32_t*);
#endif
    using FuncGetHcclBuffer = HcclResult (*)(HcclComm, void**, uint64_t*);
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
    FuncGetInstSizeList getInstSizeList_ = nullptr;
#ifndef BUILD_OPEN_PROJECT
    FuncGetInstSize getInstSize_ = nullptr;
#endif
    FuncGetHcclBuffer getHcclBuffer_ = nullptr;
};
} // namespace Mc2Hcom
#endif
