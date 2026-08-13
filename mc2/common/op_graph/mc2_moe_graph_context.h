/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mc2_moe_graph_context.h
 * \brief MC2 MoeContext builder for graph fusion pass (A5), shares the same context
 *        semantics with the aclnn single-op path (mc2/common/op_api/mc2_context.cpp).
 */

#ifndef MC2_MOE_GRAPH_CONTEXT_H
#define MC2_MOE_GRAPH_CONTEXT_H

#define HCCL_CHANNEL_SUPPORT_VERSION 89999700
#if __has_include("version/hcomm_version.h")
#include "version/hcomm_version.h"
#else
#define HCOMM_VERSION_NUM HCCL_CHANNEL_SUPPORT_VERSION
#endif
#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <mutex>
#include "hccl/hccl_rank_graph.h"
#include "graph/graph.h"

namespace ops {
class Mc2MoeGraphContext {
public:
    // 按 graphName + groupEp 获取（首次构建并缓存）Mc2MoeContext 序列化数据与 HCCL buffer 大小。
    // 图名参与缓存 key：多图（不同 graphName）各自独立缓存，不做跨图复用，避免多图交替执行时
    // 共享同一份 context 快照。同一通信域查询出的 window buffer 地址相同，但每张图独立持有。
    static bool GetContextData(const std::string &graphName, const std::string &groupEp,
                               std::vector<int32_t> &contextData, int64_t &hcclBuffSize);

    // 在图中创建承载 context 数据的 Const 节点（1D int32，与 aclnn 侧 context tensor 同构）。
    // 节点名按 graphName + groupEp 唯一（mc2_moe_context_<graphName>_<groupEp>），
    // 避免多 pass 创建同名节点导致拓扑遍历冲突，也避免跨图复用
    static ge::graphStatus CreateContextConstNode(ge::Graph &graph, const std::string &graphName,
                                                  const std::string &groupEp,
                                                  const std::vector<int32_t> &contextData, ge::GNode &constNode);

    // 按 graphName + groupEp 在图中查找已存在的 context Const 节点（dispatch/combine 两个 pass 共享同一个）。
    // 找到返回 true 并回填 constNode；未找到返回 false，避免调用方对空 GNode 判空（GetType 行为无保证）。
    static bool FindContextConstNode(ge::Graph &graph, const std::string &graphName, const std::string &groupEp,
                                     ge::GNode &constNode);

    // 调试：回读图中 context Const 的 value 并打印关键字段，用于分辨内容在 pass 结束时是否仍正确
    static void VerifyContextConstNode(ge::Graph &graph, const std::string &graphName, const std::string &groupEp);

private:
    static constexpr const char *kContextConstNamePrefix = "mc2_moe_context_";
    Mc2MoeGraphContext() = default;
    ~Mc2MoeGraphContext();
    static Mc2MoeGraphContext &GetInstance();

    bool BuildContextData(const std::string &groupEp, std::vector<int32_t> &contextData, int64_t &hcclBuffSize);
    // 校验缓存 context 中本卡 window buffer 地址是否仍是 HCCL 当前值（多图/长跑场景 HCCL 可能重建 buffer）
    bool IsContextCacheValid(const std::string &groupEp, const std::vector<int32_t> &contextData);
    bool LoadHcclSymbols();
    bool GetCommHandle(const char *groupEp, HcclComm &hcclHandle);
    bool GetHcclCommLink(const HcclComm &hcclHandle, uint32_t netLayerId, uint32_t srcRankId, uint32_t dstRankId,
                         const CommProtocol &protocol, CommLink *&links);
    bool GetNetLayers(const HcclComm &hcclHandle, uint32_t *&netLayerList, uint32_t &netLayerNum);
    bool GetRankSizePerServer(const HcclComm &hcclHandle, uint32_t netLayers);
    bool InitHcclChannel(const HcclComm &hcclHandle, uint32_t rankDim, uint32_t srcRankId, const CommProtocol &protocol,
                         std::vector<HcclChannelDesc> &channelDesc);
    bool GetHcclCommChannel(const HcclComm &hcclHandle, uint32_t rankDim, uint32_t srcRankId,
                            const CommProtocol &protocol, const CommEngine &engine,
                            std::vector<ChannelHandle> &channels);
    bool CheckLinks(uint32_t &netLinkNum, CommLink *linksList);
    bool CheckProtocolSupport(const HcclComm &hcclHandle, uint32_t *&layerList, uint32_t &layerNum);
    bool GetCommProtocol(const HcclComm &hcclHandle, CommProtocol &protocol);
    const std::string GetLibPath();
    template <typename T>
    T GetHcclLibFunc(void *handle, const std::string &funcName);

    void *hcclLibHandle_ = nullptr;
    uint64_t hcclBuffSize_ = 0;
    uint32_t epRankSize_ = 0;
    uint32_t rankSizePerServer_ = 0;
    std::unordered_map<uint32_t, uint32_t> layerMap; // 记录本卡与其他卡的通信层数，key为其他卡的rankId，value为通信层数
    std::mutex contextMutex_;                        // 多线程编译（多图并发）下保护 contextCache_

    HcclResult (*HcomGetCommHandleByGroup)(const char *, HcclComm *) = nullptr;
    HcclResult (*HcclRankGraphGetLinks)(HcclComm, uint32_t, uint32_t, uint32_t, CommLink **, uint32_t *) = nullptr;
    HcclResult (*HcclRankGraphGetLayers)(HcclComm, uint32_t **, uint32_t *) = nullptr;
    HcclResult (*HcclRankGraphGetRankSizeByLayer)(HcclComm, uint32_t, uint32_t *) = nullptr;
    HcclResult (*HcclChannelAcquire)(HcclComm, CommEngine, HcclChannelDesc *, uint32_t, ChannelHandle *) = nullptr;
    HcclResult (*HcclGetHcclBuffer)(HcclComm, void **, uint64_t *) = nullptr;
    HcclResult (*HcclChannelGetHcclBuffer)(HcclComm, ChannelHandle, void **, uint64_t *) = nullptr;
    HcclResult (*HcclGetRankId)(HcclComm, uint32_t *) = nullptr;
    HcclResult (*HcclGetRankSize)(HcclComm, uint32_t *) = nullptr;
    HcclResult (*HcclRankGraphGetRanksByLayer)(HcclComm, uint32_t, uint32_t **, uint32_t *) = nullptr;

    std::unordered_map<std::string, std::pair<std::vector<int32_t>, int64_t>> contextCache_;
};
} // namespace ops
#endif // HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
#endif // MC2_MOE_GRAPH_CONTEXT_H
