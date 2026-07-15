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
 * \file compressor_grad_tiling.h
 * \brief
 */

#ifndef COMPRESSOR_GRAD_TILING_H
#define COMPRESSOR_GRAD_TILING_H

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <sstream>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"
#include "../../op_kernel/arch35/compressor_grad_template_tiling_key.h"
#include "../../op_kernel/arch35/compressor_grad_tiling_data.h"
#include "platform/platform_info.h"

#ifdef ASCENDC_OP_TEST
#define CMPG_EXTERN_C extern "C"
#else
#define CMPG_EXTERN_C
#endif

namespace optiling {

// INPUT
constexpr uint32_t CMPG_TOKEN_X_INPUT_INDEX = 0;
constexpr uint32_t CMPG_WEIGHT_KV_INPUT_INDEX = 1;
constexpr uint32_t CMPG_WEIGHT_WGATE_INPUT_INDEX = 2;
constexpr uint32_t CMPG_D_CPM_KV_INPUT_INDEX = 3;
constexpr uint32_t CMPG_SOFTMAX_SCORE_INPUT_INDEX = 4;
constexpr uint32_t CMPG_KV_INPUT_INDEX = 5;

// INPUT(OPTION)
constexpr uint32_t CMPG_CU_SEQ_LEN_INPUT_INDEX = 6;
constexpr uint32_t CMPG_SEQ_USED_INPUT_INDEX = 7;
constexpr uint32_t CMPG_START_POS_INPUT_INDEX = 8;

// ATTR
constexpr uint32_t CMPG_CMP_RATIO_ATTR_INDEX = 0;
constexpr uint32_t CMPG_COFF_ATTR_INDEX = 1;

// OUTPUT
constexpr uint32_t CMPG_D_X_OUTPUT_INDEX = 0;
constexpr uint32_t CMPG_D_WKV_OUTPUT_INDEX = 1;
constexpr uint32_t CMPG_D_WGATE_OUTPUT_INDEX = 2;
constexpr uint32_t CMPG_D_APE_OUTPUT_INDEX = 3;

constexpr uint32_t CMPG_DIM_NUM_1 = 1;
constexpr uint32_t CMPG_DIM_NUM_2 = 2;
constexpr uint32_t CMPG_DIM_NUM_3 = 3;
constexpr uint32_t CMPG_DIM_NUM_4 = 4;
constexpr uint32_t CMPG_DIM_INDEX_0 = 0;
constexpr uint32_t CMPG_DIM_INDEX_1 = 1;
constexpr uint32_t CMPG_DIM_INDEX_2 = 2;
constexpr uint32_t CMPG_DIM_INDEX_3 = 3;

// CONSTRAINTS
constexpr uint32_t CMPG_MAX_HIDDEN_SIZE = 10240;
constexpr uint32_t CMPG_MIN_HIDDEN_SIZE = 1024;
constexpr uint32_t CMPG_ALIGN_FACTOR_HIDDEN_SIZE = 512;

constexpr uint32_t CMPG_BATCH_MODE_SCHEDULE = 1;

static const std::string CMPG_X_NAME = "x";
static const std::string CMPG_WKV_NAME = "wkv";
static const std::string CMPG_WGATE_NAME = "wgate";
static const std::string CMPG_D_CPM_KV_NAME = "d_cpm_kv";
static const std::string CMPG_SOFTMAX_SCORE_NAME = "softmax_score";
static const std::string CMPG_KV_NAME = "kv";
static const std::string CMPG_CU_SEQLENS_NAME = "cu_seqlens";
static const std::string CMPG_SEQUSED_NAME = "seqused";
static const std::string CMPG_START_POS_NAME = "start_pos";
static const std::string CMPG_CMP_RATIO_NAME = "cmp_ratio";
static const std::string CMPG_COFF_NAME = "coff";
static const std::string CMPG_D_X_NAME = "d_x";
static const std::string CMPG_D_WKV_NAME = "d_wkv";
static const std::string CMPG_D_WGATE_NAME = "d_wgate";
static const std::string CMPG_D_APE_NAME = "d_ape";

static std::string DataTypeToSerialString(ge::DataType type);

const std::map<std::string, std::vector<ge::DataType>> CMPG_DTYPE_SUPPORT_MAP = {
    {CMPG_X_NAME,             {ge::DT_BF16, ge::DT_FLOAT16}},
    {CMPG_WKV_NAME,           {ge::DT_BF16, ge::DT_FLOAT16}},
    {CMPG_WGATE_NAME,         {ge::DT_BF16, ge::DT_FLOAT16}},
    {CMPG_D_CPM_KV_NAME,      {ge::DT_BF16, ge::DT_FLOAT16}},
    {CMPG_SOFTMAX_SCORE_NAME,   {ge::DT_FLOAT}},
    {CMPG_KV_NAME,         {ge::DT_FLOAT}},
    {CMPG_CU_SEQLENS_NAME,    {ge::DT_INT32}},
    {CMPG_SEQUSED_NAME,       {ge::DT_INT32}},
    {CMPG_START_POS_NAME,     {ge::DT_INT32}},
    {CMPG_D_X_NAME,           {ge::DT_BF16, ge::DT_FLOAT16}},
    {CMPG_D_WKV_NAME,         {ge::DT_BF16, ge::DT_FLOAT16}},
    {CMPG_D_WGATE_NAME,       {ge::DT_BF16, ge::DT_FLOAT16}},
    {CMPG_D_APE_NAME,         {ge::DT_FLOAT}},
};

const std::map<std::string, std::vector<uint32_t>> CMPG_DIM_NUM_MAP = {
    {CMPG_X_NAME,            {CMPG_DIM_NUM_2, CMPG_DIM_NUM_3}},
    {CMPG_WKV_NAME,          {CMPG_DIM_NUM_2}},
    {CMPG_WGATE_NAME,        {CMPG_DIM_NUM_2}},
    {CMPG_D_CPM_KV_NAME,     {CMPG_DIM_NUM_2, CMPG_DIM_NUM_3}},
    {CMPG_SOFTMAX_SCORE_NAME,  {CMPG_DIM_NUM_3, CMPG_DIM_NUM_4}},
    {CMPG_KV_NAME,        {CMPG_DIM_NUM_3, CMPG_DIM_NUM_4}},
    {CMPG_CU_SEQLENS_NAME,   {CMPG_DIM_NUM_1}},
    {CMPG_SEQUSED_NAME,      {CMPG_DIM_NUM_1}},
    {CMPG_START_POS_NAME,    {CMPG_DIM_NUM_1}},
    {CMPG_D_X_NAME,          {CMPG_DIM_NUM_2, CMPG_DIM_NUM_3}},
    {CMPG_D_WKV_NAME,        {CMPG_DIM_NUM_2}},
    {CMPG_D_WGATE_NAME,      {CMPG_DIM_NUM_2}},
    {CMPG_D_APE_NAME,        {CMPG_DIM_NUM_2}},
};

static const std::map<std::string, uint32_t> CMPG_LAYOUT_DIM_MAP = {
    {"BSH", CMPG_DIM_NUM_3},
    {"TH", CMPG_DIM_NUM_2},
};

const std::map<ge::DataType, std::string> CMPG_DATATYPE_TO_STRING_MAP = {
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},
    {ge::DT_FLOAT, "DT_FLOAT"},
    {ge::DT_FLOAT16, "DT_FLOAT16"},
    {ge::DT_INT8, "DT_INT8"},
    {ge::DT_INT16, "DT_INT16"},
    {ge::DT_UINT16, "DT_UINT16"},
    {ge::DT_UINT8, "DT_UINT8"},
    {ge::DT_INT32, "DT_INT32"},
    {ge::DT_INT64, "DT_INT64"},
    {ge::DT_UINT32, "DT_UINT32"},
    {ge::DT_UINT64, "DT_UINT64"},
    {ge::DT_BOOL, "DT_BOOL"},
    {ge::DT_DOUBLE, "DT_DOUBLE"},
    {ge::DT_DUAL, "DT_DUAL"},
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"},
    {ge::DT_COMPLEX32, "DT_COMPLEX32"},
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},
    {ge::DT_QINT8, "DT_QINT8"},
    {ge::DT_QINT16, "DT_QINT16"},
    {ge::DT_QINT32, "DT_QINT32"},
    {ge::DT_QUINT8, "DT_QUINT8"},
    {ge::DT_QUINT16, "DT_QUINT16"},
    {ge::DT_RESOURCE, "DT_RESOURCE"},
    {ge::DT_STRING_REF, "DT_STRING_REF"},
    {ge::DT_STRING, "DT_STRING"},
    {ge::DT_VARIANT, "DT_VARIANT"},
    {ge::DT_BF16, "DT_BFLOAT16"},
    {ge::DT_INT4, "DT_INT4"},
    {ge::DT_UINT1, "DT_UINT1"},
    {ge::DT_INT2, "DT_INT2"},
    {ge::DT_UINT2, "DT_UINT2"}
};

struct CompressorGradCompileInfo {
    int64_t core_num;
};

struct CmpGradRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct CmpGradOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
    const gert::Tensor *tensor;
};

enum class CmpGradLayoutType {
    LAYOUT_BSH,
    LAYOUT_TH
};

enum class CmpGradTemplateId:uint8_t {
    EMPTY_X = 0,
    NORMAL = 1
};

CMPG_EXTERN_C ge::graphStatus TilingCompressorGrad(gert::TilingContext *context);

struct CompressorGradBaseShapeInfo {
    uint32_t bSize = 0;
    uint32_t sSize = 0;
    uint32_t hSize = 0;
    uint32_t tSize = 0;
    uint32_t dSize = 0;
    uint32_t coffSize = 0;
    uint32_t rSize = 0;
};

const std::vector<int> CMPG_COFF {1, 2};
const std::vector<int> CMPG_CMP_RATIO {2, 4, 8, 16, 32, 64, 128};
const std::vector<uint32_t> CMPG_HEAD_DIM {128, 512};

struct CompressorGradContext {
    const char *opName;
    const char *opType;
    fe::PlatFormInfos *platformInfo;

    CmpGradRequiredParaInfo x;
    CmpGradRequiredParaInfo wkv;
    CmpGradRequiredParaInfo wgate;
    CmpGradRequiredParaInfo dCpmKv;
    CmpGradRequiredParaInfo softmaxScore;
    CmpGradRequiredParaInfo kv;
    CmpGradOptionalParaInfo cuSeqlens;
    CmpGradOptionalParaInfo seqused;
    CmpGradOptionalParaInfo startPos;
    CmpGradRequiredParaInfo dX;
    CmpGradRequiredParaInfo dWkv;
    CmpGradRequiredParaInfo dWgate;
    CmpGradRequiredParaInfo dApe;

    const int *coff;
    const int *cmpRatio;
    CmpGradTemplateId templateId;

    ge::DataType dtype = ge::DT_BF16;
    CmpGradLayoutType layout = CmpGradLayoutType::LAYOUT_BSH;

    size_t *workSpaces;
    uint64_t tilingKey;
    uint32_t blockDim;
};

class CompressorGradTiling {
public:
    explicit CompressorGradTiling(CompressorGradContext *context) : context_(context) {}
    ~CompressorGradTiling() = default;

    static ge::graphStatus ConvertContext(gert::TilingContext &context, CompressorGradContext &compressorGradContext);
    ge::graphStatus RunBigKernelTiling(CompressorGradTilingData* tilingData);

private:
    static void ConvertRequiredParams(gert::TilingContext &context, CompressorGradContext &compressorGradContext);
    static void ConvertOptionalParams(gert::TilingContext &context, CompressorGradContext &compressorGradContext);
    ge::graphStatus GetNpuInfo();
    ge::graphStatus SetBaseInfo();
    ge::graphStatus SetInnerSplitInfo();
    ge::graphStatus SetWorkSpaceInfo();
    ge::graphStatus CalcWorkSpace();
    ge::graphStatus CheckSinglePara() const;
    ge::graphStatus GenTilingKey() const;
    template <typename T>
    ge::graphStatus CheckFeatureValueSupport(const T *featureValue, const std::vector<T> &expectFeatureValList,
                                             const std::string &name) const;
    template <typename T>
    ge::graphStatus CheckAttrValueSupport(const T *attrValue, const std::vector<T> &expectAttrValList,
                                          const std::string &name) const;
    template <typename T>
    void LogErrorNumberSupport(const std::vector<T> &expectNumberList, const T &actualValue, const std::string &name,
                               const std::string subName) const;
    ge::graphStatus CheckDimNumInLayoutSupport(const std::string &layout, const gert::StorageShape *shape,
                                               const std::string &name) const;
    ge::graphStatus CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc, const std::string &name) const;
    void LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList, const ge::DataType &actualDtype,
                              const std::string &name) const;
    ge::graphStatus CheckDimNumSupport(const gert::StorageShape *shape, const std::string &name) const;
    ge::graphStatus LogErrorShapeConsistency(const std::string &name, const gert::StorageShape *shape,
                                             const uint32_t &dimNum, const std::string &subName,
                                             const uint32_t &expectNum) const;
    ge::graphStatus CheckSingleParaX() const;
    ge::graphStatus CheckSingleParaWkv() const;
    ge::graphStatus CheckSingleParaWgate() const;
    ge::graphStatus CheckSingleParaDCpmKv() const;
    ge::graphStatus CheckSingleParaScKv() const;
    ge::graphStatus CheckSingleParaSoftmaxOut() const;
    ge::graphStatus CheckSingleParaStartPos() const;
    ge::graphStatus CheckSingleParaCuSeqlens() const;
    ge::graphStatus CheckSingleParaSeqused() const;
    ge::graphStatus CheckSingleParaDX() const;
    ge::graphStatus CheckSingleParaDWkv() const;
    ge::graphStatus CheckSingleParaDWgate() const;
    ge::graphStatus CheckSingleParaDApe() const;
    ge::graphStatus CheckSingleParaCmpRatio() const;
    ge::graphStatus CheckSingleParaCoff() const;
    ge::graphStatus CheckRequiredParaExistence() const;
    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckFeature() const;
    ge::graphStatus CheckShapeConsistency() const;
    ge::graphStatus CheckDtypeConsistencyX(const gert::CompileTimeTensorDesc *desc, const std::string &name) const;
    ge::graphStatus CheckDtypeConsistency() const;
    ge::graphStatus CheckMultiParaConsistency() const;
    ge::graphStatus CheckDimNumConsistency() const;
    ge::graphStatus CheckEmptyTensor() const;
    ge::graphStatus CheckBlockDimConstrain() const;

    size_t ubSize_ = 0;
    size_t l1Size_ = 0;
    size_t l0cSize_ = 0;
    size_t l0bSize_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    size_t libapiSize_ = 0;
    size_t workspaceSize_ = 0;
    uint8_t coff = 1;

    uint32_t mBaseSize = 0;
    uint32_t dBaseSize = 0;

    CompressorGradBaseShapeInfo baseShapeInfo_;
    CompressorGradContext *context_ = nullptr;
    CompressorGradBaseParams *baseParams_ = nullptr;
    CompressorGradInnerSplitParams *innerSplitParams_ = nullptr;
    CompressorGradWorkspaceParams *workspaceParams_ = nullptr;
};

} // optiling

#endif
