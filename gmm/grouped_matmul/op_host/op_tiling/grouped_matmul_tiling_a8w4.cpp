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
 * \file grouped_matmul_tiling_a8w4.cpp
 * \brief
 */
#include "grouped_matmul_tiling_utils.h"

namespace optiling {
static void SetA8W4HPTiling(A8W4HPTiling *tiling_data, uint32_t aicNum)
{
  constexpr int SIZE_TWO = 2;
  constexpr int SIZE_THREE = 3;
  constexpr int IDX_ZERO = 0;
  constexpr int IDX_ONE = 1;
  constexpr int IDX_TWO = 2;
  constexpr uint32_t SINGLE_CORE_TILING_0 = 128;
  constexpr uint32_t SINGLE_CORE_TILING_1 = 256;
  constexpr uint32_t SINGLE_CORE_BASE_TILING_0 = 128;
  constexpr uint32_t SINGLE_CORE_BASE_TILING_1 = 256;
  constexpr uint32_t TOTAL_K_THRESHOLD_6656 = 6656;

  uint32_t *ori_in0_shape = tiling_data->get_ori_in0_shape();
  uint32_t total_K = ori_in0_shape[IDX_ONE];
  uint32_t core_num = aicNum;
  uint32_t splitRecord[SIZE_THREE] = {1, 1, 1};
  uint32_t single_core_tiling[SIZE_THREE] = {SINGLE_CORE_TILING_0, SINGLE_CORE_TILING_1, total_K};
  uint32_t single_core_base_tiling[SIZE_TWO] = {SINGLE_CORE_BASE_TILING_0, SINGLE_CORE_BASE_TILING_1};

  tiling_data->set_required_core_num(core_num);
  tiling_data->set_kernel_index(0);
  tiling_data->set_splitTimes(0);

  if (total_K > TOTAL_K_THRESHOLD_6656) {
    splitRecord[IDX_ZERO] = CeilDiv(total_K, TOTAL_K_THRESHOLD_6656);
    single_core_tiling[IDX_TWO] = TOTAL_K_THRESHOLD_6656;
  }

  tiling_data->set_splitRecord(splitRecord);
  tiling_data->set_single_core_tiling(single_core_tiling);
  tiling_data->set_single_core_base_tiling(single_core_base_tiling);
}

static void PrintA8W4HPTiling(gert::TilingContext* context, A8W4HPTiling *data)
{
  constexpr int TWO = 2;
  OP_LOGD(context->GetNodeName(), "Tiling data strategy: ");
  OP_LOGD(context->GetNodeName(), "  group_num=%u", data->get_group_num());
  OP_LOGD(context->GetNodeName(), "  group_type=%hhd", data->get_group_type());
  OP_LOGD(context->GetNodeName(), "  required_core_num=%u", data->get_required_core_num());
  OP_LOGD(context->GetNodeName(), "  format_in=%f", data->get_format_in());
  OP_LOGD(context->GetNodeName(), "  format_out=%f", data->get_format_out());
  OP_LOGD(context->GetNodeName(), "  numAic=%u", data->get_numAic());
  OP_LOGD(context->GetNodeName(), "  numAiv=%u", data->get_numAiv());
  OP_LOGD(context->GetNodeName(), "  szUb=%llu", static_cast<long long unsigned int>(data->get_szUb()));
  OP_LOGD(context->GetNodeName(), "  szL0A=%llu", static_cast<long long unsigned int>(data->get_szL0A()));
  OP_LOGD(context->GetNodeName(), "  szL0C=%llu", static_cast<long long unsigned int>(data->get_szL0C()));
  OP_LOGD(context->GetNodeName(), "  pattern=%hhu", data->get_pattern());
  OP_LOGD(context->GetNodeName(), "  kernel_index=%hhu", data->get_kernel_index());
  OP_LOGD(context->GetNodeName(), "  splitTimes=%u", data->get_splitTimes());
  OP_LOGD(context->GetNodeName(), "  output_type=%hhd", data->get_output_type());
  OP_LOGD(context->GetNodeName(), "  ori_in0_shape=[%u,%u]", data->get_ori_in0_shape()[0],
            data->get_ori_in0_shape()[1]);
  OP_LOGD(context->GetNodeName(), "  ori_in1_shape=[%u,%u]", data->get_ori_in1_shape()[0],
            data->get_ori_in1_shape()[1]);
  OP_LOGD(context->GetNodeName(), "  ori_out_shape=[%u,%u]", data->get_ori_out_shape()[0],
            data->get_ori_out_shape()[1]);
  OP_LOGD(context->GetNodeName(), "  single_core_tiling=[%u,%u,%u]", data->get_single_core_tiling()[0],
            data->get_single_core_tiling()[1], data->get_single_core_tiling()[TWO]);
  OP_LOGD(context->GetNodeName(), "  single_core_base_tiling=[%u,%u]", data->get_single_core_base_tiling()[0],
            data->get_single_core_base_tiling()[1]);
  OP_LOGD(context->GetNodeName(), "  splitRecord=[%u,%u,%u]", data->get_splitRecord()[0],
            data->get_splitRecord()[1], data->get_splitRecord()[TWO]);
  OP_LOGD(context->GetNodeName(), "  workspaceOffset=%llu", static_cast<long long unsigned int>(data->get_workspaceOffset()));
}

constexpr uint32_t A8W4_CV_PARALL_NUM = 4U;
constexpr uint32_t A8W4_VECTOR_BASE_M = 32U;
constexpr uint32_t A8W4_UB_CAL_SIZE = 32U * 256U;
constexpr uint32_t A8W4_UB_REST_BYTES = 9U * 32U * 256U;
constexpr uint32_t A8W4_DIM_N = 2U;
constexpr uint32_t A8W4_DIM_K = 1U;
constexpr uint32_t A8W4_DIM_M = 0U;
constexpr uint32_t A8W4_ONE = 1U;
constexpr uint32_t A8W4_TWO = 2U;
constexpr uint32_t A8W4_FOUR = 4U;
constexpr uint32_t A8W4_EIGHT = 8U;
constexpr uint32_t A8W4_BASE_M = 128U;
constexpr uint32_t A8W4_BASE_K = 64U;
constexpr uint32_t A8W4_BASE_N = 128U;
constexpr uint32_t A8W4_WORKSPACE_SINGLE_N = 256U;
constexpr uint32_t A8W4_WORKSPACE_SINGLE_M = 128U;
constexpr uint32_t A8W4_QUANT_GROUP_SIZE = 256U;
constexpr uint32_t A8W4_K_UNIT = 64U;
constexpr uint32_t A8W4_MAX_K_MSD = 18432U;
constexpr uint32_t A8W4_MSD_BASE_M_SHORT = 64U;
constexpr uint32_t A8W4_MSD_BASE_M_DEFAULT = 128U;
constexpr uint32_t A8W4_MSD_BASE_K = 256U;
constexpr uint32_t A8W4_MSD_BASE_N_SHORT = 512U;
constexpr uint32_t A8W4_MSD_BASE_N_DEFAULT = 256U;
constexpr uint32_t A8W4_MSD_WHITE_BASE_M = 64U;
constexpr uint32_t A8W4_MSD_WHITE_BASE_N = 512U;
constexpr uint32_t A8W4_MSD_WHITE_BASE_K = 256U;
constexpr uint32_t A8W4_AVG_M_SHORT_THRESHOLD = 32U;
constexpr uint32_t A8W4_CV11_AVG_M_THRESHOLD = 128U;
constexpr uint32_t A8W4_CV11_GROUP_THRESHOLD = 4U;
constexpr uint32_t A8W4_CV11_KN_THRESHOLD = 8192U;
constexpr uint32_t A8W4_MSD_N_AVG_RATIO_THRESHOLD = 4U;

struct A8W4ShapeInfo {
  uint32_t m = 0U;
  uint32_t k = 0U;
  uint32_t n = 0U;
  uint32_t groupNum = 0U;
  uint32_t quantGroupNum = 0U;
};
static inline bool IsA8W4HighPerfEnabled(const gert::ContinuousVector* tuningConfigPtr)
{
  return (tuningConfigPtr != nullptr && tuningConfigPtr->GetSize() > TUNING_CONFIG_A8W4_SPEC_SCENARIO_INDEX) ?
      ((reinterpret_cast<const int64_t *>(tuningConfigPtr->GetData()))[TUNING_CONFIG_A8W4_SPEC_SCENARIO_INDEX] == 1) :
      false;
}

static inline int64_t GetA8W4TuningConfig(const gert::ContinuousVector* tuningConfigPtr)
{
  return (tuningConfigPtr != nullptr && tuningConfigPtr->GetSize() > TUNING_CONFIG_TOKEN_PER_EXPECT_INDEX) ?
      (reinterpret_cast<const int64_t *>(tuningConfigPtr->GetData()))[TUNING_CONFIG_TOKEN_PER_EXPECT_INDEX] : 0;
}

static inline uint64_t MakeA8W4TilingKey(uint32_t yDtype, uint32_t a8w4KernelTemplate, uint32_t aivAicRatio)
{
  return GET_TPL_TILING_KEY(GMM_TPL_INT8, GMM_TPL_INT4, yDtype, 0, 0, GROUPED_MATMUL_GROUP_LIST_TYPE_COUNT, 0,
                            a8w4KernelTemplate, GROUPED_MATMUL_A16W8_KERNEL_TEMPLATE_NONE, aivAicRatio, 0);
}

static inline void SaveA8W4TilingData(gert::TilingContext* context, GMMTilingData& tilingDataA8W4)
{
  tilingDataA8W4.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tilingDataA8W4.GetDataSize());
}

static inline A8W4ShapeInfo GetA8W4ShapeInfo(const gert::TilingContext* context)
{
  A8W4ShapeInfo shape;
  shape.n = context->GetDynamicInputTensor(SCALE_INDEX, 0)->GetStorageShape().GetDim(A8W4_DIM_N);
  shape.k = context->GetDynamicInputTensor(X_INDEX, 0)->GetStorageShape().GetDim(A8W4_DIM_K);
  shape.m = context->GetDynamicInputTensor(X_INDEX, 0)->GetStorageShape().GetDim(A8W4_DIM_M);
  shape.groupNum = context->GetDynamicInputTensor(WEIGHT_INDEX, 0)->GetStorageShape().GetDim(A8W4_DIM_M);
  shape.quantGroupNum = context->GetDynamicInputTensor(SCALE_INDEX, 0)->GetStorageShape().GetDim(A8W4_DIM_K);
  return shape;
}

static inline uint32_t GetA8W4WithOffset(const gert::TilingContext* context, const A8W4ShapeInfo& shape)
{
  auto offset = context->GetDynamicInputTensor(OFFSET_INDEX, 0);
  uint32_t withOffset = 0U;
  if (offset != nullptr) {
    auto &offsetShape = offset->GetStorageShape();
    const size_t offsetDimNum = offsetShape.GetDimNum();
    auto offsetDim0 = offsetShape.GetDim(0);
    auto offsetDim1 = offsetShape.GetDim(1);
    auto offsetDim2 = offsetShape.GetDim(A8W4_DIM_N);
    if (offsetDimNum == OFFSET_DIM_A8W4 && offsetDim0 == shape.groupNum && offsetDim1 == 1 && offsetDim2 == shape.n) {
      withOffset = 1U;
      OP_LOGD(context->GetNodeName(), "GMM A8W4: offset is enable .");
    } else {
      OP_LOGW(context->GetNodeName(),
              "GMM A8W4: offset's shape is invalid, If you want to enable offset, the expected shape is (%u,1,%u), "
              "and the current shape is (%ld,%ld,%ld).",
              shape.groupNum, shape.n, offsetDim0, offsetDim1, offsetDim2);
    }
  }
  return withOffset;
}

static inline ge::graphStatus ValidateA8W4Shape(const gert::TilingContext* context, const A8W4ShapeInfo& shape)
{
  if (shape.quantGroupNum == 0U || shape.k % shape.quantGroupNum != 0U) {
    OP_LOGE(context->GetNodeName(),
            "GMM tiling: invalid A8W4 shape, k=%u must be divisible by quantGroupNum=%u.",
            shape.k, shape.quantGroupNum);
    return ge::GRAPH_FAILED;
  }
  if (shape.k % A8W4_K_UNIT != 0) {
    OP_LOGE(context->GetNodeName(), "GMM tiling: invalid A8W4 shape, k=%u must be divisible by %u.",
            shape.k, A8W4_K_UNIT);
    return ge::GRAPH_FAILED;
  }
  if (shape.k > A8W4_MAX_K_MSD) {
    OP_LOGE(context->GetNodeName(), "GMM tiling: invalid A8W4 shape, k=%u must not exceed %u.",
            shape.k, A8W4_MAX_K_MSD);
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

static inline void SetA8W4CommonBaseParams(GMMTilingData& tilingDataA8W4, const A8W4ShapeInfo& shape,
                                           uint32_t aicNum, uint32_t withOffset)
{
  tilingDataA8W4.gmmBaseParams.set_coreNum(aicNum);
  tilingDataA8W4.gmmBaseParams.set_groupNum(shape.groupNum);
  tilingDataA8W4.gmmBaseParams.set_totalInGroup(shape.m);
  tilingDataA8W4.gmmBaseParams.set_k(shape.k);
  tilingDataA8W4.gmmBaseParams.set_n(shape.n);
  tilingDataA8W4.gmmBaseParams.set_vBaseM(A8W4_VECTOR_BASE_M);
  tilingDataA8W4.gmmBaseParams.set_ubCalSize(A8W4_UB_CAL_SIZE);
  tilingDataA8W4.gmmBaseParams.set_ubRestBytes(A8W4_UB_REST_BYTES);
  tilingDataA8W4.gmmBaseParams.set_parallNum(A8W4_CV_PARALL_NUM);
  tilingDataA8W4.gmmBaseParams.set_quantGroupNum(shape.quantGroupNum);
  tilingDataA8W4.gmmBaseParams.set_m(shape.m);
  tilingDataA8W4.gmmBaseParams.set_withOffset(withOffset);
}

static inline void SetA8W4BType(matmul_tiling::MultiCoreMatmulTiling& mm, bool wNZ,
                                matmul_tiling::DataType dtype)
{
  mm.SetBType(matmul_tiling::TPosition::GM,
              wNZ ? matmul_tiling::CubeFormat::NZ : matmul_tiling::CubeFormat::ND, dtype, false);
}

static ge::graphStatus RunA8W4HighPerfTiling(gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr,
                                             uint32_t yDtype)
{
  OP_LOGD(context->GetNodeName(), "Enter GMM A8W4 MSD high performance path...");
  constexpr int CASE_ZERO = 0;
  constexpr int CASE_ONE = 1;
  constexpr int CASE_TWO = 2;
  constexpr int CASE_THREE = 3;
  constexpr float INT4_TYPE_COUNT = 0.5f;
  constexpr float FP16_TYPE_COUNT = 2.0f;

  GMMTilingData tilingDataA8W4;
  uint32_t groupNum = context->GetDynamicInputTensor(WEIGHT_INDEX, 0)->GetStorageShape().GetDim(A8W4_DIM_M);
  uint32_t n = context->GetDynamicInputTensor(WEIGHT_INDEX, 0)->GetStorageShape().GetDim(A8W4_DIM_N);
  uint32_t k = context->GetDynamicInputTensor(X_INDEX, 0)->GetStorageShape().GetDim(A8W4_DIM_K);
  uint32_t m = A8W4_TWO * context->GetDynamicInputTensor(X_INDEX, 0)->GetStorageShape().GetDim(A8W4_DIM_M);

  auto attr = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attr);
  const auto transposeWeightPtr = attr->GetAttrPointer<bool>(ATTR_INDEX_TRANS_W);
  const auto transposeXPtr = attr->GetAttrPointer<bool>(ATTR_INDEX_TRANS_X);
  auto transposeWeight = transposeWeightPtr != nullptr ? *transposeWeightPtr : false;
  auto transposeX = transposeXPtr != nullptr ? *transposeXPtr : false;
  int pattern = transposeX ? (transposeWeight ? CASE_TWO : CASE_THREE) : (transposeWeight ? CASE_ZERO : CASE_ONE);
  pattern = CASE_ZERO;

  tilingDataA8W4.hpTilingData.set_pattern(static_cast<uint8_t>(pattern));
  tilingDataA8W4.hpTilingData.set_kernel_index(0);
  tilingDataA8W4.hpTilingData.set_format_in(INT4_TYPE_COUNT);
  tilingDataA8W4.hpTilingData.set_format_out(FP16_TYPE_COUNT);

  std::vector<uint32_t> oriIn0Shape;
  std::vector<uint32_t> oriIn1Shape;
  std::vector<uint32_t> oriOutShape;
  switch (pattern) {
    case CASE_ZERO:
      oriIn0Shape = {m, k};
      oriIn1Shape = {n, k};
      oriOutShape = {m, n};
      break;
    case CASE_ONE:
      oriIn0Shape = {m, k};
      oriIn1Shape = {k, n};
      oriOutShape = {m, n};
      break;
    case CASE_TWO:
      oriIn0Shape = {k, m};
      oriIn1Shape = {n, k};
      oriOutShape = {m, n};
      break;
    case CASE_THREE:
      oriIn0Shape = {k, m};
      oriIn1Shape = {k, n};
      oriOutShape = {m, n};
      break;
    default:
      return ge::GRAPH_FAILED;
  }
  tilingDataA8W4.hpTilingData.set_ori_in0_shape(oriIn0Shape.data());
  tilingDataA8W4.hpTilingData.set_ori_in1_shape(oriIn1Shape.data());
  tilingDataA8W4.hpTilingData.set_ori_out_shape(oriOutShape.data());

  uint32_t aic = compileInfoPtr->aicNum;
  uint32_t aiv = compileInfoPtr->aivNum;
  SetA8W4HPTiling(&tilingDataA8W4.hpTilingData, aic);

  tilingDataA8W4.hpTilingData.set_group_num(groupNum);
  tilingDataA8W4.hpTilingData.set_group_type(0);
  aic = tilingDataA8W4.hpTilingData.get_required_core_num();
  if (aic == 0U) {
    return ge::GRAPH_FAILED;
  }
  context->SetBlockDim(aic);
  tilingDataA8W4.hpTilingData.set_numAic(aic);
  tilingDataA8W4.hpTilingData.set_numAiv(aiv);
  tilingDataA8W4.hpTilingData.set_szUb(compileInfoPtr->ubSize);
  tilingDataA8W4.hpTilingData.set_szL0A(compileInfoPtr->l0ASize);
  tilingDataA8W4.hpTilingData.set_szL0C(compileInfoPtr->l0CSize);
  auto yDtypeLocal = context->GetOutputDesc(Y_INDEX)->GetDataType();
  tilingDataA8W4.hpTilingData.set_output_type(yDtypeLocal == ge::DT_FLOAT16 ? 0 : 1);

  size_t workspaceSize = m * n * sizeof(int16_t) +
      (static_cast<size_t>(SixteenAlign(m, true)) * k / A8W4_TWO * sizeof(uint8_t));
  context->SetScheduleMode(1);
  context->SetTilingKey(MakeA8W4TilingKey(yDtype, GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_AUTOTILING,
                                          GROUPED_MATMUL_AIV_AIC_RATIO_2));

  size_t *workspaces = context->GetWorkspaceSizes(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
  workspaces[0] = SYS_WORKSPACE_SIZE;
  workspaces[0] += workspaceSize;

  SaveA8W4TilingData(context, tilingDataA8W4);
  PrintA8W4HPTiling(context, &tilingDataA8W4.hpTilingData);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RunA8W4PergroupTiling(gert::TilingContext* context,
                                             matmul_tiling::MultiCoreMatmulTiling& mm,
                                             GMMTilingData& tilingDataA8W4, const A8W4ShapeInfo& shape,
                                             uint32_t yDtype, uint32_t aicNum, bool wNZ, bool isPerchannel)
{
  mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8, false);
  SetA8W4BType(mm, wNZ, matmul_tiling::DataType::DT_INT8);
  mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
  mm.SetBias(false);
  mm.SetOrgShape(A8W4_BASE_M, shape.n, shape.k);
  mm.SetShape(A8W4_BASE_M, A8W4_BASE_N, shape.k);
  mm.SetFixSplit(A8W4_BASE_M, A8W4_BASE_N, A8W4_BASE_K);
  if (mm.GetTiling(tilingDataA8W4.mmTilingData) == -1) {
    return ge::GRAPH_FAILED;
  }
  context->SetTilingKey(MakeA8W4TilingKey(yDtype, GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_PERGROUP_ANTIQUANT,
                                          GROUPED_MATMUL_AIV_AIC_RATIO_2));
  SaveA8W4TilingData(context, tilingDataA8W4);

  size_t* workspaces = context->GetWorkspaceSizes(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
  workspaces[0] = SYS_WORKSPACE_SIZE;
  workspaces[0] += static_cast<size_t>(shape.groupNum * shape.k * shape.n * static_cast<uint32_t>(sizeof(int8_t)) +
      (A8W4_CV_PARALL_NUM * aicNum * A8W4_WORKSPACE_SINGLE_N * A8W4_WORKSPACE_SINGLE_M *
       static_cast<uint32_t>(sizeof(int32_t)) * A8W4_EIGHT));
  return isPerchannel ? ge::GRAPH_PARAM_INVALID : ge::GRAPH_SUCCESS;
}

static ge::graphStatus RunA8W4MsdTiling(gert::TilingContext* context,
                                        matmul_tiling::MultiCoreMatmulTiling& mm,
                                        GMMTilingData& tilingDataA8W4, const A8W4ShapeInfo& shape,
                                        uint32_t yDtype, uint32_t aicNum, bool wNZ, uint32_t avgM,
                                        int64_t tuningConfig, bool isInA8W4WhiteList, bool enableCV11)
{
  const bool isShortM = avgM < A8W4_AVG_M_SHORT_THRESHOLD;
  uint32_t msdBaseM = isShortM ? A8W4_MSD_BASE_M_SHORT : A8W4_MSD_BASE_M_DEFAULT;
  uint32_t msdBaseN = isShortM ? A8W4_MSD_BASE_N_SHORT : A8W4_MSD_BASE_N_DEFAULT;
  uint32_t msdSingleN = FindBestSingleNA8W4(A8W4_MSD_WHITE_BASE_M, A8W4_MSD_WHITE_BASE_N, avgM, shape.n,
                                            shape.groupNum, aicNum);

  mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT4, false);
  SetA8W4BType(mm, wNZ, matmul_tiling::DataType::DT_INT4);
  mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
  mm.SetBias(false);
  if (isInA8W4WhiteList) {
    mm.SetOrgShape(A8W4_MSD_WHITE_BASE_M, shape.n, shape.k);
    mm.SetShape(A8W4_MSD_WHITE_BASE_M, msdSingleN, shape.k);
    mm.SetFixSplit(A8W4_MSD_WHITE_BASE_M, A8W4_MSD_WHITE_BASE_N, A8W4_MSD_WHITE_BASE_K);
    OP_LOGI(context->GetNodeName(),
            "GMM A8W4 tiling: baseM is %u, baseN is %u, baseK is %u, tuningConfig is %ld.",
            A8W4_MSD_WHITE_BASE_M, A8W4_MSD_WHITE_BASE_N, A8W4_MSD_WHITE_BASE_K, tuningConfig);
  } else {
    mm.SetOrgShape(msdBaseM, shape.n, shape.k);
    mm.SetShape(msdBaseM, msdBaseN, shape.k);
    mm.SetFixSplit(msdBaseM, msdBaseN, A8W4_MSD_BASE_K);
    OP_LOGI(context->GetNodeName(),
            "GMM A8W4 tiling: baseM is %u, baseN is %u, baseK is %u, tuningConfig is %ld.",
            msdBaseM, msdBaseN, A8W4_MSD_BASE_K, tuningConfig);
  }
  if (mm.GetTiling(tilingDataA8W4.mmTilingData) == -1) {
    return ge::GRAPH_FAILED;
  }

  if (isInA8W4WhiteList) {
    tilingDataA8W4.mmTilingData.set_dbL0A(A8W4_ONE);
    tilingDataA8W4.mmTilingData.set_dbL0B(A8W4_ONE);
    tilingDataA8W4.mmTilingData.set_dbL0C(A8W4_ONE);
    tilingDataA8W4.mmTilingData.set_stepKa(A8W4_ONE);
    tilingDataA8W4.mmTilingData.set_stepKb(A8W4_ONE);
    tilingDataA8W4.mmTilingData.set_depthA1(A8W4_TWO);
    tilingDataA8W4.mmTilingData.set_depthB1(A8W4_TWO);
    tilingDataA8W4.mmTilingData.set_baseK(A8W4_MSD_BASE_K);
    tilingDataA8W4.mmTilingData.set_stepM(A8W4_ONE);
    tilingDataA8W4.mmTilingData.set_stepN(A8W4_ONE);
  } else {
    tilingDataA8W4.mmTilingData.set_dbL0C(A8W4_ONE);
    tilingDataA8W4.mmTilingData.set_stepKa(A8W4_FOUR);
    tilingDataA8W4.mmTilingData.set_stepKb(A8W4_FOUR);
    tilingDataA8W4.mmTilingData.set_depthA1(A8W4_EIGHT);
    tilingDataA8W4.mmTilingData.set_depthB1(A8W4_EIGHT);
    tilingDataA8W4.mmTilingData.set_stepM(A8W4_ONE);
    tilingDataA8W4.mmTilingData.set_stepN(A8W4_ONE);
  }
  OP_LOGI(context->GetNodeName(), "GMM_tiling: baseM is %u, baseK is %u, baseN is %u.", msdBaseM,
          A8W4_MSD_BASE_K, msdBaseN);
  context->SetScheduleMode(1);

  uint32_t a8w4KernelTemplate = GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_API_DEQUANT;
  if (isInA8W4WhiteList) {
    a8w4KernelTemplate = static_cast<uint32_t>(GROUPED_MATMUL_A8W4_KERNEL_TEMPLATE_MSD_VECTOR_DEQUANT);
  }
  context->SetTilingKey(MakeA8W4TilingKey(
      yDtype, a8w4KernelTemplate,
      enableCV11 ? GROUPED_MATMUL_AIV_AIC_RATIO_1 : GROUPED_MATMUL_AIV_AIC_RATIO_2));
  SaveA8W4TilingData(context, tilingDataA8W4);

  size_t* workspaces = context->GetWorkspaceSizes(1);
  OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
  workspaces[0] = SYS_WORKSPACE_SIZE;
  if (isInA8W4WhiteList) {
    workspaces[0] += static_cast<size_t>(
        (A8W4_CV_PARALL_NUM * aicNum * msdBaseM * msdBaseN * static_cast<uint32_t>(sizeof(short))) * A8W4_TWO);
  } else {
    workspaces[0] += static_cast<size_t>(
        A8W4_CV_PARALL_NUM * aicNum * A8W4_WORKSPACE_SINGLE_N * A8W4_WORKSPACE_SINGLE_M *
        static_cast<uint32_t>(sizeof(int32_t)) * A8W4_EIGHT);
  }
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RunA8W4DefaultTiling(gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr,
                                            const gert::ContinuousVector* tuningConfigPtr, uint32_t yDtype, bool wNZ)
{
  GMMTilingData tilingDataA8W4;
  const A8W4ShapeInfo shape = GetA8W4ShapeInfo(context);
  std::array<int64_t, 5> mKNList = {shape.groupNum, shape.m, shape.k, shape.n, wNZ};
  const uint32_t withOffset = GetA8W4WithOffset(context, shape);
  const int64_t tuningConfig = GetA8W4TuningConfig(tuningConfigPtr);

  const bool isInA8W4WhiteList = ((tuningConfig != 0 && tuningConfig <= 64) || A8W4_PRETILING_WHITE_LIST.count(mKNList))
      && shape.quantGroupNum != 0 && shape.k / shape.quantGroupNum == A8W4_QUANT_GROUP_SIZE
      && shape.k % shape.quantGroupNum == 0 && withOffset == 0 && shape.k % A8W4_K_UNIT == 0
      && shape.n % A8W4_K_UNIT == 0;

  uint32_t calcM = 1U;
  if (shape.groupNum != 0U) {
    calcM = shape.m / shape.groupNum;
  }
  const uint32_t avgM = tuningConfig != 0L ? static_cast<uint32_t>(tuningConfig) : calcM;
  const bool enableCV11 = avgM <= A8W4_CV11_AVG_M_THRESHOLD && shape.groupNum <= A8W4_CV11_GROUP_THRESHOLD &&
      shape.k <= A8W4_CV11_KN_THRESHOLD && shape.n <= A8W4_CV11_KN_THRESHOLD;

  SetA8W4CommonBaseParams(tilingDataA8W4, shape, compileInfoPtr->aicNum, withOffset);
  context->SetBlockDim(compileInfoPtr->aicNum);

  if (ValidateA8W4Shape(context, shape) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  matmul_tiling::PlatformInfo platformInfo;
  InitPlatformInfo(compileInfoPtr, platformInfo);
  matmul_tiling::MultiCoreMatmulTiling mm(platformInfo);

  const bool isPerchannel = shape.quantGroupNum == 1U;
  auto biasPtr = context->GetDynamicInputTensor(BIAS_INDEX, 0);
  const bool isMSD = (tuningConfig == 0L || avgM == 0U || shape.n / avgM > A8W4_MSD_N_AVG_RATIO_THRESHOLD ||
      withOffset != 0U) && !(biasPtr == nullptr || biasPtr->GetStorageShape().GetShapeSize() == 0);
  if (withOffset != 0U && !isMSD) {
    OP_LOGW(context->GetNodeName(), "GMM W4A8: with-offset mode requires non-empty bias, but got an empty tensor.");
  }

  if (!isMSD) {
    return RunA8W4PergroupTiling(context, mm, tilingDataA8W4, shape, yDtype, compileInfoPtr->aicNum, wNZ,
                                 isPerchannel);
  }
  return RunA8W4MsdTiling(context, mm, tilingDataA8W4, shape, yDtype, compileInfoPtr->aicNum, wNZ, avgM,
                          tuningConfig, isInA8W4WhiteList, enableCV11);
}

ge::graphStatus GMMTiling::A8W4Tiling(gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr) {
  OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
  const uint32_t& aicNum = compileInfoPtr->aicNum;
  if (aicNum == 0U) {
    return ge::GRAPH_FAILED;
  }

  auto yDesc = context->GetOutputDesc(Y_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
  uint32_t yDtype = GMM_TPL_FLOAT16;
  if (yDesc->GetDataType() == ge::DT_BF16) {
    yDtype = static_cast<uint32_t>(GMM_TPL_BF16);
  }

  auto w0Desc = context->GetDynamicInputDesc(WEIGHT_INDEX, 0);
  OP_CHECK_NULL_WITH_CONTEXT(context, w0Desc);
  auto wFormat0 = static_cast<ge::Format>(ge::GetPrimaryFormat(w0Desc->GetStorageFormat()));
  const bool wNZ = wFormat0 == ge::FORMAT_FRACTAL_NZ;

  auto attr = context->GetAttrs();
  const auto tuningConfigPtr = attr != nullptr ?
      attr->GetAttrPointer<gert::ContinuousVector>(ATTR_INDEX_TUNING_CONFIG) : nullptr;
  if (IsA8W4HighPerfEnabled(tuningConfigPtr)) {
    return RunA8W4HighPerfTiling(context, compileInfoPtr, yDtype);
  }
  return RunA8W4DefaultTiling(context, compileInfoPtr, tuningConfigPtr, yDtype, wNZ);
}

}  // namespace optiling
