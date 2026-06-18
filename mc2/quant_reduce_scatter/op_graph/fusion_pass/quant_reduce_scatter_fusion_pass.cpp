/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
**/

#include "quant_reduce_scatter_fusion_pass.h"
#if CANN_VERSION_NUM >= GRAPH_FUSION_SUPPORT_VERSION
#include "graph/graph.h"
#include "graph/gnode.h"
#include "graph/operator.h"
#include "graph/operator_factory.h"
#include "graph/ge_error_codes.h"
#include "ge/fusion/pass/pattern_fusion_pass.h"
#include "mc2_common_log.h"
#include "mc2_platform_info.h"
#include "op_graph/mc2_moe_graph_context.h"

namespace ops {

namespace {
const std::string PASS_NAME = "QuantReduceScatterFusionPass";
const std::string FUSED_OP_TYPE = "QuantReduceScatter";
const std::string FUSED_OP_TYPE2 = "QuantReduceScatterV2";
const std::string QRS_CONTEXT_TAG = "QUANT_REDUCE_SCATTER_V2_CONTEXT";
const std::string QRS_GROUP_NAME = "group";

// QuantReduceScatterV2总共3个input：context, x, scale
constexpr size_t kV2TotalInputs = 3;
constexpr size_t kV2TotalOutputs = 1;

// hccl_buffer_size attr单独添加，其他必选attr写在下面
const std::vector<std::string> REQUIRED_INT_ATTRS = {
    "world_size"
};

// 可选int型attr
const std::vector<std::string> OPTIONAL_INT_ATTRS = {
    "output_dtype"
};

// 可选str型attr
const std::vector<std::string> OPTIONAL_STR_ATTRS = {
    "reduce_op"
};

}  // namespace

/**
 * @brief 获取quant_reduce_scatter的属性，并给quant_reduce_scatter_v2的属性赋值
 * @param qrsNode quant_reduce_scatter图根节点
 * @param fusionNode 融合后的图根节点
 * @param hcclBuffSize 创建context tensor对象时指定的hcclBuffSize
 */
ge::graphStatus QuantReduceScatterFusionPass::AddAttr(ge::GNode &qrsNode, ge::GNode &fusionNode,
                                                      int64_t &hcclBuffSize)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Enter QuantReduceScatterFusionPass::AddAttr");
    ge::AscendString nameStr;
    qrsNode.GetName(nameStr);

    // hccl_buffer_size attr
    fusionNode.SetAttr(ge::AscendString("hccl_buffer_size"), hcclBuffSize);
    // 必选int型attr
    int64_t attrInt = 0;
    for (const auto &attrName : REQUIRED_INT_ATTRS) {
        if (qrsNode.GetAttr(ge::AscendString(attrName.c_str()), attrInt) != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(PASS_NAME.c_str(), "node %s get required attr %s failed",
                      nameStr.GetString(), attrName.c_str());
            return ge::GRAPH_FAILED;
        }
        fusionNode.SetAttr(ge::AscendString(attrName.c_str()), attrInt);
    }
    // 可选int型attr
    for (const auto &attrName : OPTIONAL_INT_ATTRS) {
        if (qrsNode.GetAttr(ge::AscendString(attrName.c_str()), attrInt) == ge::GRAPH_SUCCESS) {
            fusionNode.SetAttr(ge::AscendString(attrName.c_str()), attrInt);
        }
    }
    // 可选str型attr
    ge::AscendString strVal;
    for (const auto &attrName : OPTIONAL_STR_ATTRS) {
        if (qrsNode.GetAttr(ge::AscendString(attrName.c_str()), strVal) == ge::GRAPH_SUCCESS) {
            fusionNode.SetAttr(ge::AscendString(attrName.c_str()), strVal);
        }
    }

    OPS_LOG_I(PASS_NAME.c_str(), "Finish QuantReduceScatterFusionPass::AddAttr");
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief 创建fusion图的根节点
 * @param graph 计算图
 * @param qrsNode quant_reduce_scatter图根节点
 * @param hcclBuffSize 创建context tensor对象时指定的hcclBuffSize
 * @param tensorDesc 创建的context tensor
 * @param fusionNode 融合后的图根节点（quant_reduce_scatter_v2）
 */
ge::graphStatus QuantReduceScatterFusionPass::CreateFusionNode(ge::Graph &graph,
                                                               ge::GNode &qrsNode,
                                                               int64_t &hcclBuffSize,
                                                               const ge::TensorDesc &tensorDesc,
                                                               ge::GNode &fusionNode)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Enter QuantReduceScatterFusionPass::CreateFusionNode");
    ge::AscendString nameStr;
    qrsNode.GetName(nameStr);
    // qrsNode的name默认是QuantReduceScatter，后面加"V2"，即QuantReduceScatterV2
    auto fusionOp = ge::OperatorFactory::CreateOperator(std::string(nameStr.GetString()) + "V2", FUSED_OP_TYPE2);
    if (fusionOp.IsEmpty()) {
        OPS_LOG_E(PASS_NAME.c_str(), "node %s: %s op does not found in factory",
                  nameStr.GetString(), FUSED_OP_TYPE2.c_str());
        return ge::GRAPH_FAILED;
    }

    // 先对第0个input，即context执行UpdateInputDesc
    fusionOp.UpdateInputDesc(static_cast<uint32_t>(0), tensorDesc);
    // 然后从1一个input开始循环执行
    for (size_t idx = 1; idx < kV2TotalInputs; idx++) {
        // v2第idx的input tensor对应v1第(idx-1)个input tensor
        auto srcPair = qrsNode.GetInDataNodesAndPortIndexs(static_cast<int32_t>(idx - 1));
        if (srcPair.first == nullptr) {
            continue;
        }
        ge::TensorDesc srcDesc;
        if (srcPair.first->GetOutputDesc(srcPair.second, srcDesc) != ge::GRAPH_SUCCESS) {
            continue;
        }
        fusionOp.UpdateInputDesc(static_cast<uint32_t>(idx), srcDesc);
    }

    size_t outputNum = qrsNode.GetOutputsSize();
    for (size_t idx = 0; idx < outputNum; idx++) {
        ge::TensorDesc outDesc;
        if (qrsNode.GetOutputDesc(static_cast<int32_t>(idx), outDesc) == ge::GRAPH_SUCCESS) {
            fusionOp.UpdateOutputDesc(static_cast<uint32_t>(idx), outDesc);
        }
    }
    fusionNode = graph.AddNodeByOp(fusionOp);

    OPS_LOG_I(PASS_NAME.c_str(), "Finish QuantReduceScatterFusionPass::CreateFusionNode");
    return AddAttr(qrsNode, fusionNode, hcclBuffSize);
}

/**
 * @brief 增加context节点后，在graph上给context节点添加edge
 * @param graph 计算图
 * @param qrsNodePtr quant_reduce_scatter图根节点
 * @param fusionNode 融合后的图根节点
 * @param contextNodePtr 新增的context节点
 */
ge::graphStatus QuantReduceScatterFusionPass::AddEdge(ge::Graph &graph, ge::GNode &qrsNodePtr,
                                                      ge::GNode &fusionNode, ge::GNode &contextNodePtr)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Enter QuantReduceScatterFusionPass::AddEdge");

    ge::graphStatus edgeRet;
    for (size_t idx = 0; idx < kV2TotalInputs; idx++) {
        auto srcPair = qrsNodePtr.GetInDataNodesAndPortIndexs(static_cast<int32_t>(idx));
        if (srcPair.first == nullptr) {
            continue;
        }
        edgeRet = graph.AddDataEdge(*srcPair.first, srcPair.second, fusionNode, static_cast<int32_t>(idx) + 1);
        if (edgeRet != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    // 创建一个size为kV2TotalOutputs的数组
    std::vector<std::vector<std::pair<ge::GNodePtr, int32_t>>> outputConsumers(kV2TotalOutputs);
    for (size_t i = 0; i < kV2TotalOutputs; i++) {
        outputConsumers[i] = qrsNodePtr.GetOutDataNodesAndPortIndexs(static_cast<int32_t>(i));
    }
    // 从图中把QuantReduceScatter根节点删除
    graph.RemoveNode(qrsNodePtr);
    for (size_t i = 0; i < kV2TotalOutputs; i++) {
        for (auto &outPair : outputConsumers[i]) {
            // 融合后的图根节点的idx=i的位置和output节点outPair.secend的位置之间添加一条边
            edgeRet = graph.AddDataEdge(fusionNode, static_cast<int32_t>(i), *outPair.first, outPair.second);
            if (edgeRet != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        }
    }
    // 在context节点idx=0的位置和融合后的图根节点idx=0的位置添加一条边
    return graph.AddDataEdge(contextNodePtr, static_cast<int32_t>(0), fusionNode, static_cast<int32_t>(0));
}

/**
 * @brief 先创建context节点，然后将context节点加到graph图中，为context节点添加edge
 * @param graph 计算图
 * @param qrsNodePtr quant_reduce_scatter图根节点
 * @param contextNode 新增的context节点
 * @param hcclBuffSize 创建context tensor对象时指定的hcclBuffSize
 * @param tensorDesc 创建的context tensor
 */
ge::graphStatus QuantReduceScatterFusionPass::FusionNode(ge::Graph &graph, ge::GNode &qrsNodePtr,
                                                         ge::GNode &contextNode, int64_t &hcclBuffSize,
                                                         const ge::TensorDesc &tensorDesc)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Enter QuantReduceScatterFusionPass::FusionNode");

    ge::GNode fusionNode;
    if (CreateFusionNode(graph, qrsNodePtr, hcclBuffSize, tensorDesc, fusionNode) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return AddEdge(graph, qrsNodePtr, fusionNode, contextNode);
}

/**
 * @brief 实现fusion主函数逻辑
 * @param graph 计算图
 * @param qrsNodePtr quant_reduce_scatter图根节点
 * @param sharedContextNode
 */
ge::graphStatus QuantReduceScatterFusionPass::Fusion(ge::Graph &graph, ge::GNode &qrsNodePtr,
                                                     ge::GNode &sharedContextNode)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Enter QuantReduceScatterFusionPass::Fusion");

#if HCOMM_VERSION_NUM >= HCCL_CHANNEL_SUPPORT_VERSION
    int64_t hcclBuffSize = 0;
    if (Mc2MoeGraphContext::GetMc2ContextA5ForQrs(graph, QRS_CONTEXT_TAG, QRS_GROUP_NAME, qrsNodePtr,
                                                  sharedContextNode, hcclBuffSize) != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(PASS_NAME.c_str(), "Get mc2 context failed, V2 node kept as-is");
        return ge::GRAPH_SUCCESS;
    }

    // 创建context tensor
    ge::Shape shape({static_cast<int64_t>(sizeof(Mc2MoeContext) / sizeof(int32_t))});
    ge::TensorDesc tensorDesc(shape, ge::FORMAT_ND, ge::DT_INT32);
    return FusionNode(graph, qrsNodePtr, sharedContextNode, hcclBuffSize, tensorDesc);
#else
    return ge::GRAPH_SUCCESS;
#endif
}

/**
 * @brief 执行
 * @param graph 计算图
 * @param passContext
 */
ge::graphStatus QuantReduceScatterFusionPass::Run(ge::GraphPtr &graph, ge::CustomPassContext &passContext)
{
    OPS_LOG_I(PASS_NAME.c_str(), "Enter QuantReduceScatterFusionPass::Run");

    (void)passContext;
    if (!IsTargetPlatformNpuArch(PASS_NAME.c_str(), NPUARCH_A5)) {
        return ge::GRAPH_SUCCESS;
    }

    ge::GNode sharedContextNode;
    // 遍历图中每个节点
    for (auto &gNode : graph->GetDirectNode()) {
        ge::AscendString nodeType;
        if ((gNode.GetType(nodeType) != ge::GRAPH_SUCCESS) || (nodeType != FUSED_OP_TYPE.c_str())) {
            continue;
        }

        ge::AscendString nameStr;
        gNode.GetName(nameStr);
        auto ret = Fusion(*graph, gNode, sharedContextNode);
        if (ret != ge::GRAPH_SUCCESS) {
            OPS_LOG_E(PASS_NAME.c_str(), "node: %s V1->V2 fusion failed", nameStr.GetString());
        }
    }
    return ge::GRAPH_SUCCESS;
}

REG_FUSION_PASS(QuantReduceScatterFusionPass).Stage(ge::CustomPassStage::kCompatibleInherited);

}  // namespace ops

#endif  // GRAPH_FUSION_SUPPORT_VERSION
