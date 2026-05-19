/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file moe_ffn.cpp
 */

#include <cmath>
#include <vector>
#include "moe_ffn_tiling.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace optiling {

#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }

#define OPS_CHECK_NULL_WITH_CONTEXT_RET(context, ptr, ret)                                       \
  if ((ptr) == nullptr) {                                                                        \
    const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
    printf("EZ9999 op[%s], %s is nullptr!", name, #ptr);                           \
    return ret;                                                                                  \
  }

#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) printf(err_msg, ##__VA_ARGS__)

#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      fflush(stdout);                         \
      expr;                                   \
    }                                         \
  } while (0)

bool AddWorkspaceFFN(gert::TilingContext* context, const size_t workspace) {
  size_t* workspace_size = context->GetWorkspaceSizes(1);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, workspace_size, false);
  *workspace_size = workspace;
  return true;
}

bool MoeFFNTiling::GetPlatformInfo(gert::TilingContext* context) {
  fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, platformInfo, false);

  _Params.CoreNum = platformInfo->GetCoreNum() / 2;
  // _Params.CoreNum = 24;
  platformInfo->GetLocalMemSize(fe::LocalMemType::UB, _Params.UBSize);
  platformInfo->GetLocalMemSize(fe::LocalMemType::L1, _Params.L1Size);
  platformInfo->GetLocalMemSize(fe::LocalMemType::L0_A, _Params.L0ASize);
  platformInfo->GetLocalMemSize(fe::LocalMemType::L0_B, _Params.L0BSize);
  platformInfo->GetLocalMemSize(fe::LocalMemType::L0_C, _Params.L0CSize);
  return true;
}

bool MoeFFNTiling::CheckTensorShape(gert::TilingContext* context, gert::Shape& shape, uint64_t ndim, std::vector<uint64_t> dims) {
  OP_TILING_CHECK(shape.GetDimNum() != ndim,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "MoeFFN tensor shape ndim:%lu wrong expected %lu, please check.",
                  static_cast<uint64_t>(shape.GetDimNum()), ndim), return false);
  for(uint32_t i = 0; i < ndim; i++) {
    OP_TILING_CHECK(static_cast<uint64_t>(shape.GetDim(i)) != dims[i],
                VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                "MoeFFN tensor shape dim[%d]:%lu wrong expected %lu, please check.",
                i, static_cast<uint64_t>(shape.GetDim(i)), dims[i]), return false);
  }
  return true;
}

bool MoeFFNTiling::CheckInOutShapes(gert::TilingContext* context) {
  uint32_t idx = 0;
  auto x = context->GetInputShape(idx++);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, x, false);
  auto shape = x->GetStorageShape();
  OP_TILING_CHECK(shape.GetDimNum() != 2,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "MoeFFN get x shape ndim is not 2, please check."), return false);
  OP_TILING_CHECK(shape.GetDim(1) % _Params.scaleGroupSize != 0,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "MoeFFN get x shape k-dim not aligned to %lu, please check.", _Params.scaleGroupSize), return false);
  _Params.originM = (uint64_t)shape.GetDim(0);
  _Params.originK_13 = (uint64_t)shape.GetDim(1);
  _Params.fracK_13 = _Params.originK_13 / FRACTAL_FLOAT16;
  _Params.scaleK_13 = _Params.originK_13 / _Params.scaleGroupSize;
  _Params.fracN_2 = _Params.fracK_13;
  _Params.originN_2 = _Params.originK_13;

  auto quantized_weight_13 = context->GetInputShape(idx++);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, quantized_weight_13, false);
  shape = quantized_weight_13->GetStorageShape();
  OP_TILING_CHECK(shape.GetDimNum() != 5,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "MoeFFN get quantized_weight_13 shape ndim is not 5, please check."), return false);
  OP_TILING_CHECK((uint64_t)shape.GetDim(1) != _Params.fracK_13 || shape.GetDim(3) != FRACTAL_FLOAT16 || shape.GetDim(4) != (FRACTAL_FLOAT16 / 8),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "MoeFFN get quantized_weight_13 shape not (G, K//16, N//16, 16, 2), please check."), return false);
  _Params.originE = (uint64_t)shape.GetDim(0);
  // printf("_Params.originE:%lu\n",_Params.originE);
  // fflush(stdout);
  OP_TILING_CHECK(_Params.originE > 384,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "MoeFFN get experts_num > 384 but only support experts_num <= 384, please check."), return false);
  // printf("########################\n");
  // fflush(stdout);
  _Params.fracN_13 = (uint64_t)shape.GetDim(2);
  _Params.originN_13 = _Params.fracN_13 * FRACTAL_FLOAT16;
  _Params.fracK_2 = _Params.fracN_13 / 2;
  _Params.originK_2 = _Params.originN_13 / 2;
  _Params.scaleK_2 = _Params.originK_2 / _Params.scaleGroupSize;

  auto weight_scale_13 = context->GetInputShape(idx++);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, weight_scale_13, false);
  shape = weight_scale_13->GetStorageShape();
  OP_TILING_CHECK(!CheckTensorShape(context, shape, 3, {_Params.originE, _Params.scaleK_13, _Params.originN_13}),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "MoeFFN get weight_scale_13 shape wrong, please check."), return false);

  auto quantized_weight_2 = context->GetInputShape(idx++);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, quantized_weight_2, false);
  shape = quantized_weight_2->GetStorageShape();
  OP_TILING_CHECK(shape.GetDimNum() != 5,
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "MoeFFN get quantized_weight_2 shape ndim is not 5, please check."), return false);
  OP_TILING_CHECK((uint64_t)shape.GetDim(0) != _Params.originE|| (uint64_t)shape.GetDim(1) != _Params.fracK_2 || (uint64_t)shape.GetDim(2) != _Params.fracN_2 || shape.GetDim(3) != FRACTAL_FLOAT16 || shape.GetDim(4) != (FRACTAL_FLOAT16 / 8),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "MoeFFN get quantized_weight_2 shape not (G, K//16, N//16, 16, 2), please check."), return false);

  auto weight_scale_2 = context->GetInputShape(idx++);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, weight_scale_2, false);
  shape = weight_scale_2->GetStorageShape();
  OP_TILING_CHECK(!CheckTensorShape(context, shape, 3, {_Params.originE, _Params.scaleK_2, _Params.originN_2}),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "MoeFFN get weight_scale_2 shape wrong, please check."), return false);

  auto expanded_expert_idx = context->GetInputShape(idx++);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, expanded_expert_idx, false);
  shape = expanded_expert_idx->GetStorageShape();
  OP_TILING_CHECK(!CheckTensorShape(context, shape, 1, {_Params.originM * _Params.topK}),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "MoeFFN get expanded_expert_idx shape wrong, please check."), return false);

  auto sorted_row_idx = context->GetInputShape(idx++);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, sorted_row_idx, false);
  shape = sorted_row_idx->GetStorageShape();
  OP_TILING_CHECK(!CheckTensorShape(context, shape, 1, {_Params.originM * _Params.topK}),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "MoeFFN get sorted_row_idx shape wrong, please check."), return false);
  auto expanded_row_idx = context->GetInputShape(idx++);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, expanded_row_idx, false);
  shape = expanded_row_idx->GetStorageShape();
  OP_TILING_CHECK(!CheckTensorShape(context, shape, 2, {_Params.originM, _Params.topK}),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "MoeFFN get expanded_row_idx shape wrong, please check."), return false);
  auto topk_weights = context->GetInputShape(idx++);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, topk_weights, false);
  shape = topk_weights->GetStorageShape();
  OP_TILING_CHECK(!CheckTensorShape(context, shape, 2, {_Params.originM, _Params.topK}),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "MoeFFN get topk_weights shape wrong, please check."), return false);

  auto weight_offset_13 = context->GetInputShape(idx++);
  if(weight_offset_13 == nullptr){
    _Params.needBias_13 = 0;
  } else {
    _Params.needBias_13 = 1;
    shape = weight_offset_13->GetStorageShape();
    OP_TILING_CHECK(!CheckTensorShape(context, shape, 3, {_Params.originE, _Params.scaleK_13, _Params.originN_13}),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                    "MoeFFN get weight_offset_13 shape wrong, please check."), return false);
  }
  auto weight_offset_2 = context->GetInputShape(idx++);
  if(weight_offset_2 == nullptr){
    _Params.needBias_2 = 0;
  } else {
    _Params.needBias_2 = 1;
    shape = weight_offset_2->GetStorageShape();
    OP_TILING_CHECK(!CheckTensorShape(context, shape, 3, {_Params.originE, _Params.scaleK_2, _Params.originN_2}),
                    VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                    "MoeFFN get weight_offset_2 shape wrong, please check."), return false);
  }

  auto y = context->GetOutputShape(0);
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, y, false);
  shape = y->GetStorageShape();
  OP_TILING_CHECK(!CheckTensorShape(context, shape, 2, {_Params.originM, _Params.originN_2}),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
                  "MoeFFN get y shape wrong, please check."), return false);
  return true;
}

bool MoeFFNTiling::GetTilingData(gert::TilingContext* context) {
  // _Params.clearBaseN = _Params.UBSize / 4;
  // uint64_t total_count = _Params.originM * _Params.originN;
  // uint64_t step_count = 2 * _Params.CoreNum * _Params.clearBaseN;

  // _Params.clearOutLoop = total_count / step_count;
  // uint64_t total_count_tail = total_count % step_count;
  // uint64_t total_repeat_tail = (total_count_tail + 128 - 1UL) / 128;
  // _Params.clearOutTailN = total_repeat_tail / (_Params.CoreNum * 2) * 128;
  // _Params.clearOutTailCoreNum = total_repeat_tail % (_Params.CoreNum * 2);

  // _Params.castBaseN = _Params.UBSize / 6 / 2;
  // total_count = _Params.originM * _Params.originN;
  // step_count = 2 * _Params.CoreNum * _Params.castBaseN;

  // _Params.castOutLoop = total_count / step_count;
  // total_count_tail = total_count % step_count;
  // total_repeat_tail = (total_count_tail + 256 - 1UL) / 256;
  // _Params.castOutTailN = total_repeat_tail / (_Params.CoreNum * 2) * 256;
  // _Params.castOutTailCoreNum = total_repeat_tail % (_Params.CoreNum * 2);
  return true;
}

bool MoeFFNTiling::SetTilingData(gert::TilingContext* context) {
  tilingData.set_CoreNum(_Params.CoreNum);
  tilingData.set_dataType(static_cast<uint32_t>(_Params.dataType));
  tilingData.set_UBSize(_Params.UBSize);
  tilingData.set_L1Size(_Params.L1Size);
  tilingData.set_L0ASize(_Params.L0ASize);
  tilingData.set_L0BSize(_Params.L0BSize);
  tilingData.set_L0CSize(_Params.L0CSize);

  tilingData.set_scaleGroupSize(_Params.scaleGroupSize);
  tilingData.set_originE(_Params.originE);
  tilingData.set_originM(_Params.originM);
  tilingData.set_topK(_Params.topK);
  tilingData.set_originN_13(_Params.originN_13);
  tilingData.set_originK_13(_Params.originK_13);
  tilingData.set_scaleK_13(_Params.scaleK_13);
  tilingData.set_fracN_13(_Params.fracN_13);
  tilingData.set_fracK_13(_Params.fracK_13);
  tilingData.set_originN_2(_Params.originN_2);
  tilingData.set_originK_2(_Params.originK_2);
  tilingData.set_scaleK_2(_Params.scaleK_2);
  tilingData.set_fracN_2(_Params.fracN_2);
  tilingData.set_fracK_2(_Params.fracK_2);

  tilingData.set_needBias_13(_Params.needBias_13);
  tilingData.set_needBias_2(_Params.needBias_2);
  // printf("CoreNum:%d\n",_Params.CoreNum);
  // printf("UBSize:%d\n",_Params.UBSize/1024);
  // printf("L1Size:%d\n",_Params.L1Size/1024);
  // printf("L0ASize:%d\n",_Params.L0ASize/1024);
  // printf("L0BSize:%d\n",_Params.L0BSize/1024);
  // printf("L0CSize:%d\n",_Params.L0CSize/1024);
  // printf("scaleGroupSize:%d\n",_Params.scaleGroupSize);
  // printf("originE:%d\n",_Params.originE);
  // printf("originM:%d\n",_Params.originM);
  // printf("topK:%d\n",_Params.topK);
  // printf("originN_13:%d\n",_Params.originN_13);
  // printf("originK_13:%d\n",_Params.originK_13);
  // printf("scaleK_13:%d\n",_Params.scaleK_13);
  // printf("fracN_13:%d\n",_Params.fracN_13);
  // printf("fracK_13:%d\n",_Params.fracK_13);
  // printf("originN_2:%d\n",_Params.originN_2);
  // printf("originK_2:%d\n",_Params.originK_2);
  // printf("scaleK_2:%d\n",_Params.scaleK_2);
  // printf("fracN_2:%d\n",_Params.fracN_2);
  // printf("fracK_2:%d\n",_Params.fracK_2);
  tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  return true;
}

bool MoeFFNTiling::SetLaunchInfo(gert::TilingContext* context) {
  context->SetBlockDim(_Params.CoreNum);
  context->SetScheduleMode(1);
  // context->SetBlockDim(1);
  if(_Params.dataType == ge::DataType::DT_FLOAT16) {
    if(_Params.originM == 1){
      tilingKey = MoeFFNTilingKey::FP16Single;
    } else {
      tilingKey = MoeFFNTilingKey::FP16NORMAL;
    }
  } else if(_Params.dataType == ge::DataType::DT_BF16) {
    if(_Params.originM == 1){
      tilingKey = MoeFFNTilingKey::BF16Single;
    } else {
      tilingKey = MoeFFNTilingKey::BF16NORMAL;
    }
  }
  context->SetTilingKey(static_cast<uint64_t>(tilingKey));
  int64_t workspaceSize = SYS_WORKSPACE_910B + (_Params.originE * 4 + 512 - 1) / 512 * 512
                                             + _Params.L0ASize * 8  * _Params.CoreNum
                                             + (_Params.topK * _Params.originM * _Params.originK_13 * 2 + 512 - 1) / 512 * 512
                                             + (_Params.topK * _Params.originM * _Params.originN_13 * 4 + 512 - 1) / 512 * 512
                                             + (_Params.topK * _Params.originM * _Params.originK_2 * 2 + 512 - 1) / 512 * 512;
  AddWorkspaceFFN(context, workspaceSize);
  return true;
}

bool MoeFFNTiling::GetCheckAttr(gert::TilingContext* context) {
  auto attrs = context->GetAttrs();
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, attrs, false);
  //  optional
  const int64_t *scale_group_size = attrs->GetAttrPointer<int64_t>(0);
  OP_TILING_CHECK(*scale_group_size % 32 != 0,
              VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
              "MoeFFN get scale_group_size not aligned to 32, please check."), return false);
  _Params.scaleGroupSize = *scale_group_size;
  const int64_t *topk = attrs->GetAttrPointer<int64_t>(1);
  OP_TILING_CHECK(*topk <= 0,
              VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(),
              "MoeFFN get topk <= 0, please check."), return false);
  _Params.topK = *topk;
  return true;
}

ge::graphStatus MoeFFNTiling::runTiling(gert::TilingContext* context) {
  // printf("hello in MoeFFNTiling::runTiling\n");
  // fflush(stdout);
  // 910b AscendC platformINFO
  // printf("step 0 in MoeFFNTiling::runTiling\n");
  OP_TILING_CHECK(!GetPlatformInfo(context),
                VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get platforminfo fail."),
                return ge::GRAPH_FAILED);
  // attr
  // printf("step 1 in MoeFFNTiling::runTiling\n");
  OP_TILING_CHECK(!GetCheckAttr(context),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "check attr fail."),
                  return ge::GRAPH_FAILED);
  // shape
  // printf("step 2 in MoeFFNTiling::runTiling\n");
  OP_TILING_CHECK(!CheckInOutShapes(context),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "check shape fail."),
                  return ge::GRAPH_FAILED);
  auto tempGetInputDesc = context->GetInputDesc(0);
  OP_TILING_CHECK(tempGetInputDesc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "inputDesc is nullptr."),
        return ge::GRAPH_FAILED);
  _Params.dataType = tempGetInputDesc->GetDataType();
  // calculate tilingdata
  // printf("step 3 in MoeFFNTiling::runTiling\n");
  OP_TILING_CHECK(!GetTilingData(context),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "get tiling data fail."),
                  return ge::GRAPH_FAILED);
  // tilingdata
  // printf("step 4 in MoeFFNTiling::runTiling\n");
  OP_TILING_CHECK(!SetTilingData(context),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "set tiling data fail."),
                  return ge::GRAPH_FAILED);
  // launchinfo: tilingkey, workspace, blockdim
  // printf("step 5 in MoeFFNTiling::runTiling\n");
  OP_TILING_CHECK(!SetLaunchInfo(context),
                  VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "set launchinfo fail."),
                  return ge::GRAPH_FAILED);
  // printf("step 6 in MoeFFNTiling::runTiling\n");
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingMoeFFN(gert::TilingContext* context) {
  MoeFFNTiling tiling_handle;
  return tiling_handle.runTiling(context);
}
}  // namespace optiling

namespace ops {
class MoeFFN : public OpDef {
   public:
    explicit MoeFFN(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("quantized_weight_13")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight_scale_13")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("quantized_weight_2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight_scale_2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("expanded_expert_idx")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("sorted_row_idx")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("expanded_row_idx")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("topk_weights")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight_offset_13")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight_offset_2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("scale_group_size")
            .AttrType(OPTIONAL)
            .Int(0);
        this->Attr("topk")
            .AttrType(OPTIONAL)
            .Int(0);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");

        this->AICore()
            .SetTiling(optiling::TilingMoeFFN)
            .AddConfig("ascend910b", aicore_config);
    }
};

OP_ADD(MoeFFN);
}  // namespace ops

