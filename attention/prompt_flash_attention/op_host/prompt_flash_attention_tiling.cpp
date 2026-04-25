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
 * \file prompt_flash_attention_tiling.cpp
 * \brief
 */
#include <queue>
#include <numeric>
#include <graph/utils/type_utils.h>

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling_base/data_copy_transpose_tiling.h"
#include "log/log.h"
#include "err/ops_err.h"
#include "prompt_flash_attention_tiling.h"
#include "register/tilingdata_base.h"

using namespace ge;
using namespace AscendC;
using namespace matmul_tiling;
namespace optiling {
constexpr uint32_t BYTE_BLOCK = 32; // The block size of datacopy, which moves data at the block granularity.
constexpr uint32_t SOFTMAX_BUFFER_NUM = 3;

constexpr uint32_t NUM_0 = 0;
constexpr uint32_t NUM_1 = 1;
constexpr uint32_t NUM_2 = 2;
constexpr uint32_t NUM_3 = 3;
constexpr uint32_t NUM_4 = 4;
constexpr uint32_t NUM_5 = 5;
constexpr uint32_t INDEX_2 = 2;
constexpr uint32_t INDEX_3 = 3;
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t VALUE_INDEX = 2;
constexpr uint32_t QUERY_ROPE_INDEX = 3;
constexpr uint32_t KEY_ROPE_INDEX = 4;
constexpr uint32_t ATTENTION_OUT_INDEX = 0;
constexpr uint32_t PSE_SHIFT_INDEX = 3;
constexpr uint32_t ATTEN_MASK_INDEX = 4;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 5;
constexpr uint32_t ACTUAL_SEQ_KV_INDEX = 6;
constexpr uint32_t DEQ_SCALE1_INDEX = 7;
constexpr uint32_t QUANT_SCALE1_INDEX = 8;
constexpr uint32_t DEQ_SCALE2_INDEX = 9;
constexpr uint32_t QUANT_SCALE2_INDEX = 10;
constexpr uint32_t QUANT_OFFSET2_INDEX = 11;
constexpr uint32_t ANTIQUANT_SCALE_INDEX = 12;
constexpr uint32_t ANTIQUANT_OFFSET_INDEX = 13;

constexpr uint32_t INPUT_QKV_SHAPE_MIN_DIMS = 2;
constexpr uint32_t INPUT_QKV_SHAPE_MAX_DIMS = 5;
constexpr uint32_t INPUT_KV_SHAPE_MAX_DIMS = 5;

constexpr uint32_t ATTR_N_INDEX = 0;
constexpr uint32_t ATTR_SCALE_INDEX = 1;
constexpr uint32_t ATTR_PRE_TOKEN_INDEX = 2;
constexpr uint32_t ATTR_NEXT_TOKEN_INDEX = 3;
constexpr uint32_t ATTR_INPUT_LAYOUT_INDEX = 4;
constexpr uint32_t ATTR_NUM_KV_HEADS_INDEX = 5;
constexpr uint32_t ATTR_SPARSE_MODE = 6;
constexpr uint32_t ATTR_INNER_PRECISE = 7;

constexpr uint64_t EMPTY_KV_TILING_KEY = 20;
constexpr uint64_t KVCACHE_NZ_TILING_KEY = 20000000;
constexpr uint64_t KVCACHE_ND_TILING_KEY = 10000000;
constexpr uint32_t LOOP_BEGIN_NUM = 0;
constexpr uint32_t SPARSE_MODE_NO_MASK = 0;
constexpr uint32_t SPARSE_MODE_ALL_MASK = 1;
constexpr uint32_t SPARSE_MODE_LEFT_UP = 2;
constexpr uint32_t SPARSE_MODE_RIGHT_DOWN = 3;
constexpr uint32_t SPARSE_MODE_BAND = 4;

constexpr int32_t SPARSE_MODE_NONE = 20;
constexpr int32_t SPARSE_MODE_NORM = 21;
constexpr int32_t SPARSE_MODE_ALIBI = 22;

constexpr uint32_t SPARSE_MODE_INT_MAX = 214748647;
constexpr uint32_t SPARSE_OPTIMIZE_ATTENTION_SIZE = 2048;
constexpr uint32_t PSE_SHIFT_DIM = 4;
constexpr uint32_t ATTENTION_MASK_DIM2 = 2;
constexpr uint32_t ATTENTION_MASK_DIM3 = 3;
constexpr uint32_t ATTENTION_MASK_DIM4 = 4;
constexpr int32_t BLOCK_SIZE_BASE = 128;  // The current requirement is a multiple of 128, and to prevent cross block handling, the mm base is also set to 128.
constexpr int32_t BLOCK_SIZE_MAX = 1024;
constexpr int32_t LONG_SEQ_ALIBI_LEN = 256;
constexpr int32_t PA_BBH_DIMS = 3;
constexpr int32_t PA_BNBD_DIMS = 4;
constexpr int32_t PA_NZ_DIMS = 5;

constexpr uint32_t CVDIFF_S2_THRESHOLDS = 1;
constexpr uint32_t CVDIFF_SMALL_QS_THRESHOLDS = 16;
constexpr uint32_t CVDIFF_MM1RES_UB_SIZE = 16384; // 128 * 128
constexpr uint32_t CVDIFF_SOUTER_FACTOR_DEFAULT = 128;
constexpr uint32_t CVDIFF_SMALL_KV_THRESHOLDS = 1024;
constexpr uint32_t CVDIFF_SINNER_FACTOR_SMALL_KVS = 512;   // kv_s <= 512 scene sinner slice size
constexpr uint32_t CVDIFF_SINNER_FACTOR_DEFAULT = 1024;    // CV diff general scene sinner slice size
constexpr uint32_t CVDIFF_SINNER_FACTOR_SMALL_QS = 2048;   // q_s <= 16 scene sinner slice size
constexpr uint32_t CVDIFF_MSD_BUFFER_SIZE_512B = 512; // 0.5k
constexpr uint32_t CVDIFF_MSD_BUFFER_SIZE_1024B = 1024; // 0.5k

constexpr uint32_t SPLIT_DOUBLE_UB = 2;
constexpr uint32_t DSPLIT_THRESHOLDS_512 = 512;
constexpr uint64_t DSPLIT_S2_D_TILING_KEY = 400;
constexpr uint64_t DSPLIT_S2_TILING_KEY = 300;
constexpr uint32_t UB_ALIGN = 32;
uint64_t BENCHMARK_TILING_KEY = 1000000000000000000;
constexpr uint64_t BENCHMARK_TILING_KEY_0 = 3000000000000000000;
constexpr uint64_t BENCHMARK_TILING_KEY_1 = 3000000000000000001;
constexpr uint64_t BENCHMARK_TILING_KEY_2 = 3000000000000000002;
constexpr uint64_t BENCHMARK_TILING_KEY_3 = 3000000000000000003;
constexpr uint64_t BENCHMARK_TILING_KEY_4 = 2000000000004000000U;
constexpr uint64_t tiling_key_10000U = 10000U;
constexpr uint64_t tiling_key_20000U = 20000U;

constexpr uint32_t THIRTY_ONE = 31;
constexpr uint32_t FROM_FUSED_FLAG = 71;
constexpr uint32_t MATMUL_NORM_MIN_SEQ = 128;
constexpr uint32_t MATMUL_NORM_MIN_HEADSIZE = 128;

constexpr uint32_t BLIMIT = 65536;
constexpr uint32_t NLIMIT = 256;  // n <= 256
constexpr uint32_t SLIMIT = 20971520;  // s、kvs <= 20M
constexpr uint32_t DLIMIT = 512; // D <= 512

constexpr uint32_t MSD_UB_BASE_WIDTH = 16;
constexpr uint32_t MSD_UB_HEGHT = 256;
constexpr uint32_t MSD_UB_INQUEUE = 8;
constexpr uint32_t MSD_UB_TMP_NM = 16;
constexpr uint32_t ONE_BLK_SIZE_PFA = 32;
constexpr uint32_t COMPUTELINE_FOR_BIG_D = 1;
constexpr uint32_t MAX_COMPUTELINES = 16;
constexpr uint32_t MSD_BIG_D = 256;
constexpr uint32_t CV_RATIO = 2;

constexpr int64_t HIGH_PERF_BUFFER_NUM = 6L;
constexpr int64_t HIGH_PERF_API_BUFFER_MULTIPLE = 2L;
constexpr int64_t FRACTAL_NUM = 16L;
constexpr int64_t AIV_AIC_NUM_RATIO = 2L;
constexpr int64_t S1_VEC2_BASE_SIZE_MAX = 512L;
constexpr int64_t BMM_BASICBLOCK_M_128 = 128L;
constexpr int64_t BMM_BASICBLOCK_N_128 = 128L;
constexpr int64_t BMM1_DEPTH_A1_2 = 2L;
constexpr int64_t BMM1_DEPTH_A1_3 = 3L;
constexpr size_t WORK_SPACE_RESERVE_SIZE = 16 * 1024 * 1024;
constexpr int64_t MAX_AIC_NUM = 24L;
constexpr int64_t MAX_AIV_NUM = 48L;

const uint32_t MAX_BLOCK_DIM = 50;
constexpr int32_t MLA_THRESHOLD = 256;
constexpr int32_t BASE_DEFAULT_VALUE_0 = 0;
constexpr int32_t PING_PONG_BUFFER_SIZE = 32768 * 8;
constexpr int32_t BLOCK_SIZE = 16;
constexpr int32_t LONG_SEQ_LEN = 128;
const int32_t PP_BLOCK_BUFFER_SIZE = 128 * 128;
const int32_t PP_MM_NUM = 8;
const int32_t PP_INDEX = 16;
constexpr std::array<int32_t, PP_MM_NUM> PP_MM = {16, 32, 48, 64, 80, 96, 112, 128};

constexpr int32_t QUERY_INPUT_DIM_NUM_HOST_TILING = 4;
constexpr int32_t ZERO_HOST_TILING = 0;
constexpr int32_t ONE_HOST_TILING = 1;
constexpr int32_t TWO_HOST_TILING = 2;
constexpr int32_t THREE_HOST_TILING = 3;
constexpr int32_t FOUR_HOST_TILING = 4;
constexpr int32_t SIXTEEN_HOST_TILING = 16;

constexpr int32_t BMM_BLOCK_SIZE_THRESHOLD_LOWER_64_HOST_TILING = 64;
constexpr int32_t BMM_BLOCK_SIZE_THRESHOLD_UPPER_128_HOST_TILING = 128;
constexpr int32_t INPUT_LAYOUT_TILING_KEY_FACTOR_HOST_TILING = 100000;

constexpr int64_t BYTES_PER_KB_HOST_TILING = 1024;
constexpr int64_t ATTEN_MASK_S2_SIZE_DEFAULT_HOST_TILING = 2048;
constexpr int64_t S1_VEC2_BASE_8_HOST_TILING = 8;
constexpr int64_t S1_VEC2_MULTIPLIER_2_HOST_TILING = 2;

constexpr uint32_t BATCH_MODE_SCHEDULE = 1;

constexpr uint32_t BATCH_LIMIT_128 = 128;
constexpr uint32_t BATCH_LIMIT_300 = 300;

inline int32_t ConvertValueToIndexMM(int32_t val, int32_t idxBound)
{
    return (val > PP_MM[idxBound]) ? idxBound : (val / PP_INDEX - 1);
}
const int32_t PP_NN_NUM = 16;
constexpr std::array<int32_t, PP_NN_NUM> PP_NN = {16, 32, 48, 64, 80, 96, 112, 128,
    144, 160, 176, 192, 208, 224, 240, 256};
inline int32_t ConvertValueToIndexNN(int32_t val, int32_t idxBound)
{
    return (val > PP_NN[idxBound]) ? idxBound : (val / PP_INDEX - 1);
}

enum class MaskType : uint8_t {
    MASK_TYPE_NONE = 0,
    MASK_TYPE_NORM = 1,
    MASK_TYPE_ALIBI = 2,
    MASK_TYPE_LOOK_AHEAD = 3,
    MASK_TYPE_SWA_NORM = 4,
    MASK_TYPE_SWA_COMPRESS = 5
};

enum class QuantType : uint8_t {
    TYPE_QUANT_UNDEFINED = 0,
    TYPE_DEQUANT_FUSION,
    TYPE_QUANT_QKV_OFFLINE,
    TYPE_QUANT_QKV_ONLINE
};

enum class AttenMaskCompressMode : uint8_t {
    NO_COMPRESS_MODE = 0,
    LEFT_UP_CAUSAL_MODE,
    RIGHT_DOWN_CAUSAL_MODE,
    BAND_MODE,
    PREFIX_MODE,
    RIGHT_DOWN_CAUSAL_BAND_MODE = 5,
    BAND_LEFT_UP_CAUSAL_MODE
};

enum class SparseTypeEnum : uint8_t {
    ALL = 0,
    NONE = 1,
    ANY = 2,
    CAUSAL = 3,
    BAND = 4,
    PREFIX = 5,
    BAND_COMPRE = 6,
    RIGHT_DOWN_CAUSAL = 7,
    RIGHT_DOWN_CAUSAL_BAND = 8,
    BAND_LEFT_UP_CAUSAL = 9
};

static const std::unordered_map<ge::DataType, string> g_strDataTypePfa = {
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
    {ge::DT_STRING, "DT_STRING"},
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8V"},
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},
    {ge::DT_QINT8, "DT_QINT8"},
    {ge::DT_QINT16, "DT_QINT16"},
    {ge::DT_QINT32, "DT_QINT32"},
    {ge::DT_QUINT8, "DT_QUINT8"},
    {ge::DT_QUINT16, "DT_QUINT16"},
    {ge::DT_RESOURCE, "DT_RESOURCE"},
    {ge::DT_STRING_REF, "DT_STRING_REF"},
    {ge::DT_DUAL, "DT_DUAL"},
    {ge::DT_VARIANT, "DT_VARIANT"},
    {ge::DT_BF16, "DT_BF16"},
    {ge::DT_HIFLOAT8, "DT_HIFLOAT8"},
    {ge::DT_FLOAT8_E4M3FN, "DT_FLOAT8_E4M3FN"},
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},
};

template <typename T>
static auto AlignUp(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

template <typename T>
static auto AlignDown(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return num1 / num2 * num2;
}

template <typename T>
static auto CeilDivision(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
static auto CalcTailSize(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    T mod = num1 % num2;
    return mod != 0 ? mod : num2;
}

static uint32_t PromptGcd(uint32_t a, uint32_t b)
{
    if (b == 0U) {
        return b;
    }
    if (static_cast<uint32_t>(a % b) == 0U) {
        return b;
    }
    return PromptGcd(b, static_cast<uint32_t>(a % b));
}

static ge::DataType ValidPfaDataType(ge::DataType type)
{
    return (g_strDataTypePfa.find(type) == g_strDataTypePfa.end()) ? ge::DT_UNDEFINED : type;
}

namespace v2 {
std::string GetPfaDataTypeStr(ge::DataType type) {
    ge::DataType findDype = (g_strDataTypePfa.find(type) == g_strDataTypePfa.end()) ? ge::DT_UNDEFINED : type;
    return g_strDataTypePfa.at(findDype);
}
} // namespace v2

namespace arch38 {
std::string GetPfaDataTypeStr(ge::DataType type) {
    ge::DataType findDype = (g_strDataTypePfa.find(type) == g_strDataTypePfa.end()) ? ge::DT_UNDEFINED : type;
    return g_strDataTypePfa.at(findDype);
}
} // namespace arch38

ge::graphStatus PromptFlashAttentionTiling::ConvertContextToPFAParams(gert::TilingContext* context, ContextParamsForPFATiling& contextKeyParams) const
{
    contextKeyParams.opName = context->GetNodeName();
    bool inputOutputIsNullPtr = (context->GetInputDesc(QUERY_INDEX) == nullptr) || (context->GetInputDesc(KEY_INDEX) == nullptr) ||
        (context->GetInputDesc(VALUE_INDEX) == nullptr) || (context->GetOutputDesc(ATTENTION_OUT_INDEX) == nullptr) ||
        (context->GetInputShape(QUERY_INDEX) == nullptr) || (context->GetInputShape(KEY_INDEX) == nullptr) ||
        (context->GetInputShape(VALUE_INDEX) == nullptr) || (context->GetOutputShape(ATTENTION_OUT_INDEX) == nullptr);
    OP_CHECK_IF(inputOutputIsNullPtr,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "q, k, v or attenOut is nullptr!"),
        return ge::GRAPH_FAILED);

    contextKeyParams.isKvContinuous = 1U;
    contextKeyParams.emptyTensor = 0U;
    contextKeyParams.fromTilingSink = 0U;
    contextKeyParams.pseShift = context->GetOptionalInputTensor(PSE_SHIFT_INDEX);
    contextKeyParams.attentionMask = context->GetOptionalInputTensor(ATTEN_MASK_INDEX);
    contextKeyParams.actualSequenceLengthQ = context->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    contextKeyParams.actualSequenceLengthKV = context->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    contextKeyParams.antiquantScale = context->GetOptionalInputTensor(ANTIQUANT_SCALE_INDEX);
    contextKeyParams.antiquantOffset = context->GetOptionalInputTensor(ANTIQUANT_OFFSET_INDEX);
    contextKeyParams.inputDataType = context->GetInputDesc(QUERY_INDEX)->GetDataType();
    contextKeyParams.kDataType = context->GetInputDesc(KEY_INDEX)->GetDataType();
    contextKeyParams.vDataType = context->GetInputDesc(VALUE_INDEX)->GetDataType();
    contextKeyParams.blockTable = nullptr;
    contextKeyParams.keySharedPrefix = (nullptr);
    contextKeyParams.valueSharedPrefix = (nullptr);
    contextKeyParams.actualSharedPrefixLen = (nullptr);
    contextKeyParams.pseShiftDataType = (contextKeyParams.pseShift != nullptr) ?
    context->GetOptionalInputDesc(PSE_SHIFT_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.maskDataType = (contextKeyParams.attentionMask != nullptr) ?
    context->GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.outputDataType = context->GetOutputDesc(ATTENTION_OUT_INDEX)->GetDataType();
    contextKeyParams.queryInputShape = context->GetInputShape(QUERY_INDEX);
    contextKeyParams.keyInputShape = context->GetInputShape(KEY_INDEX);
    contextKeyParams.valueInputShape = context->GetInputShape(VALUE_INDEX);
    contextKeyParams.pseShiftShape = context->GetOptionalInputShape(PSE_SHIFT_INDEX);
    contextKeyParams.attentionMaskShape = context->GetOptionalInputShape(ATTEN_MASK_INDEX);
    contextKeyParams.deqScale1Shape = context->GetOptionalInputShape(DEQ_SCALE1_INDEX);
    contextKeyParams.scale1Shape = context->GetOptionalInputShape(QUANT_SCALE1_INDEX);
    contextKeyParams.deqScale2Shape = context->GetOptionalInputShape(DEQ_SCALE2_INDEX);
    contextKeyParams.scale2Shape = context->GetOptionalInputShape(QUANT_SCALE2_INDEX);
    contextKeyParams.offset2Shape = context->GetOptionalInputShape(QUANT_OFFSET2_INDEX);
    contextKeyParams.antiquantScaleShape = context->GetOptionalInputShape(ANTIQUANT_SCALE_INDEX);
    contextKeyParams.antiquantOffsetShape = context->GetOptionalInputShape(ANTIQUANT_OFFSET_INDEX);
    contextKeyParams.outputShape = context->GetOutputShape(0);
    auto attrs = context->GetAttrs();
    contextKeyParams.innerPrecisePtr = attrs->GetAttrPointer<int64_t>(ATTR_INNER_PRECISE);
    contextKeyParams.headsNumber = attrs->GetAttrPointer<int64_t>(ATTR_N_INDEX);
    contextKeyParams.sparseMode = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_MODE);
    contextKeyParams.preToken = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKEN_INDEX);
    contextKeyParams.nextToken = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKEN_INDEX);
    contextKeyParams.scaleValue = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    contextKeyParams.layout = attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX);
    contextKeyParams.numKeyValueHeads = attrs->GetAttrPointer<int64_t>(ATTR_NUM_KV_HEADS_INDEX);
    contextKeyParams.workspaceSize = context->GetWorkspaceSizes(1);
    contextKeyParams.compileInfoPtr = static_cast<const PromptFlashAttentionCompileInfo *>(context->GetCompileInfo());
    contextKeyParams.isBSNDOut = (string(contextKeyParams.layout) == "BNSD_BSND") ? 1U : 0U;
    contextKeyParams.fromFused = NUM_0;

    contextKeyParams.deqScaleType = (context->GetOptionalInputDesc(DEQ_SCALE1_INDEX) != nullptr) ?
    context->GetOptionalInputDesc(DEQ_SCALE1_INDEX)->GetDataType() : contextKeyParams.inputDataType;
    contextKeyParams.deqScale2Type = (context->GetOptionalInputDesc(DEQ_SCALE2_INDEX) != nullptr) ?
    context->GetOptionalInputDesc(DEQ_SCALE2_INDEX)->GetDataType() : contextKeyParams.inputDataType;

    contextKeyParams.quantScale2Type = (context->GetOptionalInputDesc(QUANT_SCALE2_INDEX) != nullptr) ?
        context->GetOptionalInputDesc(QUANT_SCALE2_INDEX)->GetDataType() : ge::DT_FLOAT;
    contextKeyParams.quantOffset2Type = (context->GetOptionalInputDesc(QUANT_OFFSET2_INDEX) != nullptr) ?
        context->GetOptionalInputDesc(QUANT_OFFSET2_INDEX)->GetDataType() : ge::DT_FLOAT;

    OP_CHECK_IF(contextKeyParams.workspaceSize == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "workSpaceSize got from ge is nullptr"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void PromptFlashAttentionTiling::UpdateTilingKeyFlag(const ContextParamsForPFATiling& contextKeyParams, uint64_t& tilingKey)
{
    uint64_t binaryFlag = 0;
    auto queryDtype = contextKeyParams.inputDataType;
    auto kvDtype = contextKeyParams.kDataType;

    if ((queryDtype == ge::DT_FLOAT16) && (kvDtype == ge::DT_INT8) && !(enableMsd)) {
        binaryFlag += 8UL;    // 4bit flag bit, the leftmost side indicates whether to perform inverse quantization operation, with a corresponding value of 2**3 = 8, and the remaining 3bit is reserved
    }
    tilingKey += (binaryFlag * 100000000000UL); // If inverse quantization is performed, tilingKey should increase by 8*100000000000.
    return;
}

bool PromptFlashAttentionTiling::GetApiTmpSize(const uint32_t sOuterFactor, const uint32_t sInnerFactor, const uint32_t typeByteSize)
{
    auto tmpShape = Shape({sOuterFactor, sInnerFactor});
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        apiTmpSize = GetSoftMaxFlashV2MinTmpSize(tmpShape, typeByteSize, typeByteSize, true, true);
        return true;
    }
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND910B) {
        uint32_t softmaxTmpSize = GetSoftMaxMinTmpSize(tmpShape, typeByteSize, true);
        uint32_t softmaxFlashTmpSize = GetSoftMaxFlashMinTmpSize(tmpShape, typeByteSize, true, true);
        if ((softmaxTmpSize == 0U) || (softmaxFlashTmpSize == 0U)) {
            return false;
        }
        apiTmpSize = std::max(softmaxTmpSize, softmaxFlashTmpSize);
    }
    return false;
}

size_t PromptFlashAttentionTiling::GetPFAWorkSpaceSize(PromptFlashAttentionTilingData& tilingData)
{
    size_t sysWorkspaceSize, workspaceSize;
    const uint64_t defaultSysWorkspaceSize910B = 16U * 1024U * 1024U;
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        sysWorkspaceSize = defaultSysWorkspaceSize;  // sys workspace size default value
        return sysWorkspaceSize;
    } else { // 910b
        uint64_t maxSpmSize = tilingData.promptAttentionTensorSizeRect.get_spmTmpSize();
        sysWorkspaceSize = defaultSysWorkspaceSize910B;  // sys workspace size default value
        if (tilingMod == TilingMod::CVDIFF) {
            int64_t mm1ResSize = tilingData.promptAttentionSingleCoreParams.get_singleProcessSOuterSize() * \
                tilingData.promptAttentionSingleCoreParams.get_singleProcessSInnerSize();
            int64_t mm2ResSize = tilingData.promptAttentionSingleCoreParams.get_singleProcessSOuterSize() * \
                tilingData.promptAttentionBaseParams.get_headSize();

            int64_t mdsExpandNumber = MSD_HIGH_PERFORMANCE_EXPEND_NUM;
            if (innerPrecise == static_cast<decltype(innerPrecise)>(HIGH_PRECISION)) {
                mdsExpandNumber = static_cast<int64_t>(MSD_HIGH_PRECISION_EXPEND_NUM);
            }
            if (enableMsd) {
                workspaceSize = sysWorkspaceSize + static_cast<uint64_t>(coreNum) * softmaxDataTypeSize * (maxSpmSize
                + static_cast<uint64_t>(mm1ResSize) * NUM_2 * static_cast<uint64_t>(mdsExpandNumber)
                + static_cast<uint64_t>(mm2ResSize) * NUM_2 * static_cast<uint64_t>(mdsExpandNumber)); // 2:use 2mm ub
            } else {
                workspaceSize = sysWorkspaceSize + static_cast<uint64_t>(coreNum) * softmaxDataTypeSize * (maxSpmSize + static_cast<uint64_t>(mm1ResSize) * NUM_2 + static_cast<uint64_t>(mm2ResSize) * NUM_2); // 2:use 2mm ub
            }

            if (enableKvAntiquant) {
                int32_t KvAntiquantSize = tilingData.promptAttentionSingleCoreParams.get_singleProcessSInnerSize() * \
                    tilingData.promptAttentionBaseParams.get_alignedHeadSize();
                workspaceSize += static_cast<uint64_t>(coreNum) * dataTypeSize * static_cast<uint64_t>(KvAntiquantSize) * 2U;  // key value, 2 is used to ensure alignment
            }
            if (enablePA) {
                workspaceSize += static_cast<uint64_t>(coreNum) * 2U * 2U * 64U;  // 2 bmm, db, ensure alignment of each structure 64B, dcci cacheline needs to
            }

            if (enableMsd) {
                workspaceSize = workspaceSize + static_cast<uint64_t>(coreNum) * mdsExpandNumber * tilingData.promptAttentionBaseParams.get_seqSize() * tilingData.promptAttentionBaseParams.get_headSize();
            }
         } else {
            if ((splitS2 == 1U) && (splitD == 1U)) {
                workspaceSize = sysWorkspaceSize + static_cast<uint64_t>(coreNum) * softmaxDataTypeSize * (maxSpmSize + \
                    NUM_2 * tilingData.promptAttentionTensorSizeRect.get_mmResUbSize() * // 2 : 2 mm ub
                    tilingData.promptAttentionSingleCoreParams.get_multiSmaxsInnerLoopTimes());
            } else {
                workspaceSize = sysWorkspaceSize + static_cast<uint64_t>(coreNum) * softmaxDataTypeSize * (maxSpmSize + \
                    tilingData.promptAttentionTensorSizeRect.get_bmm2ResUbSize() + \
                    tilingData.promptAttentionTensorSizeRect.get_mmResUbSize() * \
                    tilingData.promptAttentionSingleCoreParams.get_multiSmaxsInnerLoopTimes());
            }
        }
        return workspaceSize;
    }
}

size_t PromptFlashAttentionTiling::GetPFABaseApiWorkSpaceSize(const uint32_t& numBlocksToBeSet) {
    size_t sysWorkspaceSize, workspaceSize;
    const uint64_t defaultSysWorkspaceSize910B = 16U * 1024U * 1024U;
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        sysWorkspaceSize = defaultSysWorkspaceSize;
        return sysWorkspaceSize;
    } else {
        sysWorkspaceSize = defaultSysWorkspaceSize910B;
        uint64_t dataLenFloat = sizeof(float);
        uint64_t workSize = static_cast<uint64_t>(numBlocksToBeSet) * static_cast<uint64_t>(PING_PONG_BUFFER_SIZE) * dataLenFloat;
        baseApiTilingData.promptAttentionBaseApiBaseParams.set_workSize(workSize);
        workspaceSize = tilingMod == TilingMod::CVDIFF_BASE_API ? defaultSysWorkspaceSize910B + workSize * 3U : defaultSysWorkspaceSize910B + workSize * 9U;
        return workspaceSize;
    }
}

ge::graphStatus PromptFlashAttentionTiling::TilingGetTilingKeyAttentionAscendC(uint64_t& tilingKey,
    ContextParamsForPFATiling& contextKeyParams, bool useNewTiling, PromptFlashAttentionTilingData &tilingData) {
    auto inputDataType = contextKeyParams.inputDataType;  // input q
    auto attenMaskElemType = contextKeyParams.maskDataType;
    auto outputDataType = contextKeyParams.outputDataType;  // output tensor
    tilingData.promptAttentionBaseParams.set_attenMaskElemType(attenMaskElemType);

    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        tilingKey = 12288U; // 12288: 310p tiling
        if (inputLayout != InputLayout::BNSD) {
            tilingKey += tiling_key_10000U;
        }  // 10000 : BSH/BSND 22288
        if ((inputDataType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) {
            tilingKey += 600U;
        } // innerPrecise 0, add 600

        return ge::GRAPH_SUCCESS;
    }
    tilingKey = 0U;
    // If not in CV diff template,when there is a tail block, tilingKey should increase by 1.
    if (!((tilingMod == TilingMod::CVDIFF) || (isSOuterNoTail && isSInnerNoTail && isDNoTail))) {
        tilingKey += 1U;
    }
    if (inputDataType == ge::DT_BF16) {
        tilingKey += 100U;
    } // When the input qkv is BF16, add 100.
    if (inputDataType == ge::DT_INT8) {
        tilingKey += 200U;
    } // When the input qkv is INT8, add 200.
    if (tilingMod == TilingMod::CVDIFF) {
        tilingKey += 1002U;
    } // 1002: Add 1000 when using CV diff; Without distinguishing between tail and no tail, add 2 uniformly.
    if (outputDataType == ge::DT_INT8 ) {
        tilingKey += tiling_key_20000U;
    } // When output is INT8, add 20000.

    if (!useNewTiling) {
        return ge::GRAPH_SUCCESS; // The old template does not consider NSD differences, only 0, 1, 100, 101
    }

    tilingKey += 10U; // New Template 10、11、15、16、110、111、115、116.
    tilingKey += (inputLayout == InputLayout::BNSD) || (inputLayout == InputLayout::NSD) ? 5U : 0U;

    // The KV cache inverse quantization for CV diff currently only handles the case where Q in the CV diff template is FP16.
    if ((inputDataType == ge::DT_FLOAT16 || inputDataType == ge::DT_BF16) && (tilingMod == TilingMod::CVDIFF)) {
        tilingKey = 1012U;    // 1012：CV diff, +1000; new_tiling, +10; not distinguishing between tail and total, +2.
        tilingKey += ((inputDataType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) || (enableMsd && contextKeyParams.inputDataType == ge::DT_FLOAT16 && contextKeyParams.kDataType == ge::DT_INT8) ? 600U : 0U;  // fp16 high precision mode, regarded as a type 600.
        tilingKey += (inputDataType == ge::DT_BF16) ? 100U : 0U;    // 100: bf16
        tilingKey += (outputDataType == ge::DT_BF16) ? tiling_key_10000U : 0U; // When the output dtype is bf16, tilingKey should increase by 10000.
        tilingKey += (outputDataType == ge::DT_INT8) ? tiling_key_20000U : 0U; // 20000: The situation of outputDataType == ge::DT_INT8
        if ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::SH) || (inputLayout == InputLayout::BSND)) {
            tilingKey += 100000U;
        }   // When the inputLayout is BSH, SH or BSND, plus 100000.
        if ((splitCoreMode != SplitCoreMode::SPLIT_NBS_CUBE && splitCoreMode != SplitCoreMode::SPLIT_ONEN_CUBE) && enableMatmulNorm) {
            tilingKey += 1000000U;
        }  // Only enable matmul tiling optimization, do not enable l1reuse, add 1000000, mutually exclusive with the following situation of 2000000.
        if ((splitCoreMode == SplitCoreMode::SPLIT_NBS_CUBE || splitCoreMode == SplitCoreMode::SPLIT_ONEN_CUBE)) {
            tilingKey += 2000000U;
        }   // l1reuse defaults to enabling matmul tiling optimization, with an additional 2000000, which is mutually exclusive from the 1000000 situation mentioned above.
        UpdateTilingKeyFlag(contextKeyParams, tilingKey);         // Determine whether to perform inverse quantization and generate a binary number by combining it with the remaining reserved bits, and take its decimal representation.
    }

    if (enablePA) {
        tilingKey += 10000000U;  // 10000000: the situation of PA
    }

    if (isKVHasPrefix) {
        tilingKey += 100000000U;  // 100000000: the situation of prefix
    }

    if (enableMsd) {
        tilingKey += 400200000000U;  // 400200000000: for msd
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::TilingGetBaseApiTilingKeyAttentionAscendC(uint64_t& tilingKey, ContextParamsForPFATiling& contextKeyParams) {
    auto inputDataType = contextKeyParams.inputDataType;  // input q
    auto kDataType = contextKeyParams.kDataType;  // input k
    auto vDataType = contextKeyParams.vDataType;  // input v
    auto outputDataType = contextKeyParams.outputDataType;  // output tensor

    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        tilingKey = 12288U; // 12288: 310p tiling
        if (inputLayout != InputLayout::BNSD) {
            tilingKey += tiling_key_10000U;
        }  // 10000 : BSH/BSND 22288
        if ((inputDataType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) {
            tilingKey += 600U;
        } // innerPrecise 0, add 600
        if (atbRunFlag_) {
            tilingKey += 100000U;
        }
        return ge::GRAPH_SUCCESS;
    } else {
        tilingKey = 12U;
        BENCHMARK_TILING_KEY = BENCHMARK_TILING_KEY_4;
        tilingKey += (inputDataType == ge::DT_BF16) ? 100U : 0U;
        tilingKey += (inputDataType == ge::DT_INT8) ? 200U : 0U;
        if (innerPrecise == static_cast<decltype(innerPrecise)>(HIGH_PERFORMANCE)) {
            tilingKey += 1000U;
        }
        tilingKey += (outputDataType == ge::DT_BF16) ? tiling_key_10000U : 0U;
        tilingKey += (outputDataType == ge::DT_INT8) ? tiling_key_20000U : 0U;
        if (tilingMod == TilingMod::CVDIFF_BASE_API && innerPrecise == HIGH_PRECISION && !contextKeyParams.pseShift
            && !contextKeyParams.deqScale1Shape && !contextKeyParams.deqScale2Shape
            && !contextKeyParams.scale1Shape && !contextKeyParams.scale2Shape
            && !contextKeyParams.offset2Shape) {
            tilingKey += 2000000000U;
        }
        if (tilingMod == TilingMod::CVDIFF_MLA) {
            tilingKey += 10000000000U;
        }
        tilingKey += (kDataType == ge::DT_INT8 && vDataType == ge::DT_INT8) ? 100000000000U : 0U;
    }
    return ge::GRAPH_SUCCESS;
}

bool PromptFlashAttentionTiling::EnableMTE2BmmPipe(PromptFlashAttentionTilingData& tilingData,
    matmul_tiling::MatmulApiTiling& bmm, uint32_t sOuterFactor, uint32_t sInnerFactor) {
    if (tilingData.promptAttentionBaseParams.get_seqSize() > 16) { // When the size is greater than 16, use xiaoe speculative inference.
        return true;
    }
    uint32_t baseK = 32U;
    uint32_t head_size = tilingData.promptAttentionBaseParams.get_headSize();
    if (head_size%baseK != 0U) {
        return true;
    }

    uint32_t baseM = std::min(uint32_t(128), sOuterFactor);
    uint32_t baseN = std::min(uint32_t(512), sInnerFactor);
    if (enablePA) {
        baseN = static_cast<uint32_t>(BLOCK_SIZE_BASE);
    }
    int32_t ret = 0;
    ret = bmm.SetFixSplit(baseM, baseN, baseK);
    OP_CHECK_IF(ret != 0,
        OPS_REPORT_VECTOR_INNER_ERR("PromptFlashAttention", "bmm SetFixSplit failed, ret = %d!", ret),
        return false);
    return false;
}

void PromptFlashAttentionTiling::PromptFlashAttention310PSetBmm1(matmul_tiling::MatmulApiTiling& bmm1)
{
    matmul_tiling::DataType mmOutputType = matmul_tiling::DataType::DT_FLOAT16;
    if (inputType == ge::DT_FLOAT16 && innerPrecise == HIGH_PRECISION) {
        mmOutputType = matmul_tiling::DataType::DT_FLOAT;
    }
    // 310p mm1: A gm ND, B gm ND, C vec NZ
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_FLOAT16, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_FLOAT16, true);
    bmm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::NZ,
        mmOutputType);
}

void PromptFlashAttentionTiling::PromptFlashAttention310PSetBmm2(matmul_tiling::MatmulApiTiling& bmm2)
{
    // 310p mm2: A vec NZ, B gm ND, C vec ND
    bmm2.SetAType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::NZ,
        matmul_tiling::DataType::DT_FLOAT16, false);
    bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_FLOAT16, false);
    bmm2.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_FLOAT16);
}

ge::graphStatus PromptFlashAttentionTiling::CheckKeyValueParamsConsistency(const ContextParamsForPFATiling& contextKeyParams) {
    if (!contextKeyParams.isKvContinuous) {
        return GRAPH_SUCCESS;
    }

    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    const uint32_t keyDimNum = keyShape->GetStorageShape().GetDimNum();
    const uint32_t valueDimNum = valueShape->GetStorageShape().GetDimNum();

    OP_CHECK_IF(contextKeyParams.kDataType != contextKeyParams.vDataType,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "tensor key dtype(%d) must be consistent with tensor value dtype(%d)!", contextKeyParams.kDataType, contextKeyParams.vDataType),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyDimNum != valueDimNum,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "tensor key shape dimNum(%u) must be consistent with tensor value shape dimNum(%u)!", keyDimNum, valueDimNum),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((keyDimNum < INPUT_QKV_SHAPE_MIN_DIMS) || (keyDimNum > INPUT_QKV_SHAPE_MAX_DIMS),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "tensor key shape dimNum(%u) is invalid! Only support range [%u, %u]", keyDimNum, INPUT_QKV_SHAPE_MIN_DIMS, INPUT_QKV_SHAPE_MAX_DIMS),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool PromptFlashAttentionTiling::PromptFlashAttentionCheckBmm1(PromptFlashAttentionTilingData& tilingData,
    int64_t l1SizeRemain, int64_t l0CSize,
    uint32_t sOuterFactor, uint32_t sInnerFactor, bool allGM, bool autoBaseMNK) {
    int32_t ret = 0;
    matmul_tiling::MatmulApiTiling bmm1(ascendPlatformInfo);
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        PromptFlashAttention310PSetBmm1(bmm1);
    } else { // 910b
        matmul_tiling::DataType bmm1InputType = matmul_tiling::DataType::DT_FLOAT16;
        matmul_tiling::DataType bmm1OutputType = matmul_tiling::DataType::DT_FLOAT16;
        GetMatMulType(bmm1InputType, bmm1OutputType);
        matmul_tiling::TPosition cPosition = allGM ? matmul_tiling::TPosition::GM : matmul_tiling::TPosition::VECCALC;
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1InputType, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1InputType, true);
        bmm1.SetCType(cPosition, matmul_tiling::CubeFormat::ND, bmm1OutputType);
    }
    ret = bmm1.SetShape(sOuterFactor, sInnerFactor, tilingData.promptAttentionBaseParams.get_headSize());
    OP_CHECK_IF(ret != 0,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm1 SetShape failed, ret = %d!", ret),
        return false);
    int32_t ratio = tilingData.promptAttentionBaseParams.get_headNumRatio();
    if (ratio == 0) {
        return false;
    }
    int32_t strideQ = tilingData.promptAttentionBaseParams.get_headSize() *
        tilingData.promptAttentionBaseParams.get_headNumSize();
    int32_t strideK = strideQ / ratio;
    if ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::SH) ||
        (inputLayout == InputLayout::BSND)) {
        bmm1.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
            tilingData.promptAttentionBaseParams.get_seqInnerSize(),
            strideQ, strideK);

        if (enableKvAntiquant) {
            bmm1.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
                tilingData.promptAttentionBaseParams.get_seqInnerSize(),
                strideQ, tilingData.promptAttentionBaseParams.get_headSize());
        } else if (enableMsd) {
            // Left input BNSD, right input BSH
            bmm1.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
                tilingData.promptAttentionBaseParams.get_seqInnerSize(),
                tilingData.promptAttentionBaseParams.get_headSize(), strideK);
        }
    } else if ((inputLayout == InputLayout::BNSD) || (inputLayout == InputLayout::NSD)) {
        if (enablePA && PAlayoutType == 1U) {  // The left matrix of PA is BNSD, and the right matrix is BSH.
            bmm1.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
                tilingData.promptAttentionBaseParams.get_seqInnerSize(),
                tilingData.promptAttentionBaseParams.get_headSize(), strideK);
        } else {
            bmm1.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
                tilingData.promptAttentionBaseParams.get_seqInnerSize(),
                tilingData.promptAttentionBaseParams.get_headSize());
        }
    }

    bmm1.SetBias(false);
    ret = bmm1.SetBufferSpace(l1SizeRemain, l0CSize);
    OP_CHECK_IF(ret != 0,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm1 SetBufferSpace failed, l1SizeRemain = %ld, l0CSize = %ld, ret = %d!",
        l1SizeRemain, l0CSize, ret),
        return false);
    if (enablePA) {
        ret = bmm1.SetFixSplit(sOuterFactor, BLOCK_SIZE_BASE);
    } else {
        ret = bmm1.SetFixSplit(sOuterFactor, sInnerFactor);
    }
    OP_CHECK_IF(ret != 0,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm1 SetFixSplit failed, l1SizeRemain = %ld, l0CSize = %ld, sOuterFactor = %u, sInnerFactor = %u, ret = %d!",
        l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, ret),
        return false);
    if (inputType == ge::DT_INT8) {
        bmm1.SetDequantType(matmul_tiling::DequantType::SCALAR);
    }

    if (autoBaseMNK) {
        if (enableMatmulNorm || splitCoreMode == SplitCoreMode::SPLIT_NBS_CUBE || splitCoreMode == SplitCoreMode::SPLIT_ONEN_CUBE) {
            uint32_t baseM = std::min(uint32_t(128), sOuterFactor);
            uint32_t baseN = std::min(uint32_t(128), sInnerFactor);
            uint32_t baseK = 128U;
            ret = bmm1.SetFixSplit(baseM, baseN, baseK);
            OP_CHECK_IF(ret != 0,
                OPS_REPORT_VECTOR_INNER_ERR("PromptFlashAttention", "bmm1 SetFixSplit failed, ret = %d!", ret),
                return false);
        } else {
            uint32_t baseM = std::min(uint32_t(128), sOuterFactor);
            uint32_t baseN = std::min(uint32_t(256), sInnerFactor);
            uint32_t baseK = 64U;
            if (enablePA) {
                baseN = static_cast<uint32_t>(BLOCK_SIZE_BASE);
            }
            if (ret != 0) {
                ret = bmm1.SetFixSplit(baseM, baseN, baseK);
                OP_CHECK_IF(ret != 0,
                    OPS_REPORT_VECTOR_INNER_ERR("PromptFlashAttention", "bmm1 SetFixSplit failed, ret = %d!", ret),
                    return false);
            }
        }
    }

    OP_CHECK_IF(ret != 0, // Get tiling fail for bmm1.
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm1 GetTiling failed, l1SizeRemain = %ld, l0CSize = %ld, sOuterFactor = %u, sInnerFactor = %u, autoBaseMNK = %d, ret = %d!",
        l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, autoBaseMNK, ret),
        return false);

    bool res = EnableMTE2BmmPipe(tilingData, bmm1, sOuterFactor, sInnerFactor);  // Open MTE2 Matmul pipeline.

    OP_CHECK_IF(res == false,     // EnableMTE2BmmPipe fail.
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "EnableMTE2BmmPipe failed!"),
        return false);

    return true;
}

void PromptFlashAttentionTiling::GetMatMulType(matmul_tiling::DataType &mmInputType,
    matmul_tiling::DataType &mmOutputType) {
    if (inputType == ge::DT_FLOAT16 && innerPrecise == HIGH_PRECISION) {
        mmInputType = matmul_tiling::DataType::DT_FLOAT16;
        mmOutputType = matmul_tiling::DataType::DT_FLOAT;
    } else if (inputType == ge::DT_BF16) {
        mmInputType = matmul_tiling::DataType::DT_BF16;
        mmOutputType = matmul_tiling::DataType::DT_FLOAT;
    } else if (inputType == ge::DT_INT8) {
        mmInputType = matmul_tiling::DataType::DT_INT8;
        mmOutputType = matmul_tiling::DataType::DT_FLOAT16;
    }
}

bool PromptFlashAttentionTiling::PromptFlashAttentionCheckBmm2(PromptFlashAttentionTilingData& tilingData,
    int64_t l1SizeRemain, int64_t l0CSize,
    uint32_t sOuterFactor, uint32_t sInnerFactor, uint32_t dSplitFactor, bool allGM, bool autoBaseMNK) {
    int32_t ret = 0;
    matmul_tiling::MatmulApiTiling bmm2(ascendPlatformInfo);
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        PromptFlashAttention310PSetBmm2(bmm2);
        ret = bmm2.SetShape(sOuterFactor, tilingData.promptAttentionBaseParams.get_headSize(), sInnerFactor);
        if ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND)) {
            int32_t ratio = tilingData.promptAttentionBaseParams.get_headNumRatio();
            if (ratio == 0) {
                return false;
            }
            int32_t strideQ = tilingData.promptAttentionBaseParams.get_headSize() *
                tilingData.promptAttentionBaseParams.get_headNumSize();
            int32_t strideV = strideQ / ratio;
            bmm2.SetOrgShape(sOuterFactor, strideV, sInnerFactor,
                tilingData.promptAttentionBaseParams.get_seqInnerSize());
        } else if ((inputLayout == InputLayout::BNSD) || (inputLayout == InputLayout::NSD)) { // M, N, KA, KB
            bmm2.SetOrgShape(sOuterFactor, tilingData.promptAttentionBaseParams.get_headSize(), sInnerFactor,
                tilingData.promptAttentionBaseParams.get_seqInnerSize());
        }
    } else { // This is for 910B.
        matmul_tiling::DataType bmm2InputType = matmul_tiling::DataType::DT_FLOAT16;
        matmul_tiling::DataType bmm2OutputType = matmul_tiling::DataType::DT_FLOAT16;
        GetMatMulType(bmm2InputType, bmm2OutputType);
        if ((splitS2 == 1U) && (splitD == 1U)) {
            bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2InputType, false);
            bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2InputType, false);
            bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutputType);
            ret = bmm2.SetShape(sOuterFactor, tilingData.promptAttentionBaseParams.get_headSize(),
                tilingData.promptAttentionBaseParams.get_seqInnerSize());
        } else {
            matmul_tiling::TPosition aPosition = allGM ? matmul_tiling::TPosition::GM : matmul_tiling::TPosition::TSCM;
            matmul_tiling::TPosition cPosition = allGM ? matmul_tiling::TPosition::GM : matmul_tiling::TPosition::VECCALC;
            bmm2.SetAType(aPosition, matmul_tiling::CubeFormat::NZ, bmm2InputType, false);
            bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2InputType, false);
            bmm2.SetCType(cPosition, matmul_tiling::CubeFormat::ND_ALIGN, bmm2OutputType);
            ret = bmm2.SetShape(sOuterFactor, tilingData.promptAttentionBaseParams.get_headSize(), sInnerFactor);
        }
    OP_CHECK_IF(ret != 0,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm2 set SetShape failed, sOuterFactor = %u, sInnerFactor = %u, ret = %d!",
        sOuterFactor, sInnerFactor, ret),
        return false);
    int32_t ratio = tilingData.promptAttentionBaseParams.get_headNumRatio();
    if (ratio == 0) {
        return false;
    }
    int32_t strideQ = tilingData.promptAttentionBaseParams.get_headSize() *
        tilingData.promptAttentionBaseParams.get_headNumSize();
    int32_t strideV = strideQ / ratio;
        if ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND) ||
            (inputLayout == InputLayout::SH)) {
            bmm2.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(), strideV,
                tilingData.promptAttentionBaseParams.get_seqInnerSize());
            if (enableKvAntiquant) {
                bmm2.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
                    tilingData.promptAttentionBaseParams.get_headSize(),
                    tilingData.promptAttentionBaseParams.get_seqInnerSize());
            }
        } else if ((inputLayout == InputLayout::BNSD) || (inputLayout == InputLayout::NSD)) {
            if (enablePA && PAlayoutType == 1U) {  // The left matrix of PA is BNSD, and the right matrix is of PA is BSH.
                bmm2.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(), strideV,
                    tilingData.promptAttentionBaseParams.get_seqInnerSize());
            } else {
                bmm2.SetOrgShape(tilingData.promptAttentionBaseParams.get_seqSize(),
                    tilingData.promptAttentionBaseParams.get_headSize(),
                    tilingData.promptAttentionBaseParams.get_seqInnerSize());
            }
        }
    }

    bmm2.SetBias(false);
    ret = bmm2.SetBufferSpace(l1SizeRemain, l0CSize);
    OP_CHECK_IF(ret != 0,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm2 set SetBufferSpace failed, l1SizeRemain = %ld, l0CSize = %ld, sOuterFactor = %u, sInnerFactor = %u, ret = %d!",
        l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, ret),
        return false);
    if (inputType == ge::DT_INT8) {
        bmm2.SetDequantType(matmul_tiling::DequantType::SCALAR);
    }

    if (autoBaseMNK) {
        if (enableMatmulNorm || splitCoreMode == SplitCoreMode::SPLIT_NBS_CUBE || splitCoreMode == SplitCoreMode::SPLIT_ONEN_CUBE) {
            uint32_t baseM = std::min(uint32_t(128), sOuterFactor);
            uint32_t baseN = static_cast<uint32_t>(std::min(uint64_t(128), tilingData.promptAttentionBaseParams.get_headSize()));
            uint32_t baseK = 128U;
            ret = bmm2.SetFixSplit(baseM, baseN, baseK);
            OP_CHECK_IF(ret != 0,
                OPS_REPORT_VECTOR_INNER_ERR("PromptFlashAttention", "bmm2 SetFixSplit failed, ret = %d!", ret),
                return false);
        }
    } else {
        if ((isDNoTail) || (splitS2 == 0U) || (splitD == 1U)) {
            ret = bmm2.SetFixSplit(sOuterFactor, dSplitFactor);
        } else {
            ret = bmm2.SetFixSplit(sOuterFactor, tilingData.promptAttentionBaseParams.get_alignedHeadSize());
        }
        OP_CHECK_IF(ret != 0,
            OPS_REPORT_VECTOR_INNER_ERR("PromptFlashAttention", "bmm2 SetFixSplit failed, ret = %d!", ret),
            return false);
    }
    OP_CHECK_IF(ret != 0,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm2 set GetTiling failed, l1SizeRemain = %ld, l0CSize = %ld, sOuterFactor = %u, sInnerFactor = %u, autoBaseMNK = %d, ret = %d!",
        l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, autoBaseMNK, ret),
        return false);
    OP_CHECK_IF(ret != 0,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "bmm2 set shareL0CSize failed, l1SizeRemain = %ld, l0CSize = %ld, sOuterFactor = %u, sInnerFactor = %u, autoBaseMNK = %d, ret = %d!",
        l1SizeRemain, l0CSize, sOuterFactor, sInnerFactor, autoBaseMNK, ret),
        return false);
    return true;
}

uint32_t PromptFlashAttentionTiling::CalculateL1SizeUsed(PromptFlashAttentionTilingData& tilingData, const uint32_t typeByteSize)
{
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        return (typeByteSize * tilingData.promptAttentionTensorSizeRect.get_scmTmpSize() * 3); // 3：Two extra tscm buffers are needed for a1, b1 or b1, b2.
    }
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND910B) {
        return (typeByteSize * tilingData.promptAttentionTensorSizeRect.get_scmTmpSize());
    }
    return 0;
}

ge::graphStatus PromptFlashAttentionTiling::PromptFlashAttentionApiTiling(PromptFlashAttentionTilingData& tilingData,
    uint32_t typeSize,  uint32_t sOuterFactor, uint32_t softmaxSInnerFactor, uint32_t softmaxSOuterFactor) {
    auto softmaxShapeRect = Shape({softmaxSOuterFactor, softmaxSInnerFactor});

    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        uint32_t sftV2Size = GetSoftMaxFlashV2MinTmpSize(softmaxShapeRect, softmaxDataTypeNZ_, softmaxDataTypeNZ_, true);
        if ((ubSizeRemain + apiTmpSize) < static_cast<int64_t>(sftV2Size)) {
            return ge::GRAPH_FAILED;
        }
        SoftMaxFlashV2TilingFunc(softmaxShapeRect, softmaxDataTypeNZ_, softmaxDataTypeNZ_, (ubSizeRemain + apiTmpSize) / UB_ALIGN * UB_ALIGN,
            tilingData.softmaxTilingDataRect, true);
    } else {
        SoftMaxTilingFunc(softmaxShapeRect, sizeof(float), ubSizeRemain + apiTmpSize, tilingData.softmaxTilingDataRect);
        SoftMaxFlashV2TilingFunc(softmaxShapeRect, softmaxDataTypeSize, sizeof(float), ubSizeRemain + apiTmpSize,
            tilingData.softmaxFlashTilingDataRect, true, true);
    }

    auto transposeSrcShapeRect = Shape({1, 1, sOuterFactor,
        tilingData.promptAttentionBaseParams.get_headSize()});
    auto transposeDstShape = Shape({tilingData.promptAttentionBaseParams.get_batchSize(),
        tilingData.promptAttentionBaseParams.get_headNumSize(),
        tilingData.promptAttentionBaseParams.get_seqSize(),
        tilingData.promptAttentionBaseParams.get_headSize() *
        tilingData.promptAttentionBaseParams.get_headNumSize()});

    GetDataCopyTransposeTiling(transposeDstShape, transposeSrcShapeRect, typeSize, tilingData.transposeTilingDataRect);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::PromptFlashAttentionSetTilingData(gert::TilingContext* context,
    PromptFlashAttentionTilingData& tilingData) {
    if (atbRunFlag_) {
        baseApiTilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(baseApiTilingData.GetDataSize());
    } else if (mlaRunFlag_) {
        mlaTilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(mlaTilingData.GetDataSize());
    } else {
        tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::GetRectangleFactor(uint32_t seqFactorThreshold,
    std::queue<uint32_t>& sQueue, int32_t threshold) {
    for (int i = static_cast<int64_t>(seqFactorThreshold); i >= threshold ; i = (i - threshold)) { // threshold 16
        sQueue.push(i);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::SetInputLayout(const char* layout) {
    if (layout == nullptr) {
        inputLayout = InputLayout::BSH;
        return ge::GRAPH_SUCCESS;
    }

    std::string layoutStr(layout);
    if (layoutStr == "") {
        inputLayout = InputLayout::BSH;
    } else if (layoutStr == "SH") {
        inputLayout = InputLayout::SH;
    } else if (layoutStr == "BSH") {
        inputLayout = InputLayout::BSH;
    } else if (layoutStr == "NSD") {
        inputLayout = InputLayout::NSD;
    } else if (layoutStr == "BSND") {
        inputLayout = InputLayout::BSND;
    } else if (layoutStr == "BNSD") {
        inputLayout = InputLayout::BNSD;
    } else if (layoutStr == "BNSD_BSND") { // Reuse BNSD process for BNSD_BSND
        inputLayout = InputLayout::BNSD;
    } else if (layoutStr == "TND") {
        inputLayout = InputLayout::TND;
    } else if (layoutStr == "NTD_TND") {
        inputLayout = InputLayout::NTD_TND;
    } else {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

bool PromptFlashAttentionTiling::CheckInputDimAndHeadNum(ContextParamsForPFATiling& contextKeyParams, const uint32_t nQAttr, const uint32_t nKVAttr) {
    uint32_t nQ = nQAttr;
    uint32_t nKV = nKVAttr;
    if (nKVAttr == 0U) { // Detected that nKVAttr is the default value, which means that the customer did not pass in.
        nKV = nQAttr;
    }

    const gert::StorageShape* queryShape = contextKeyParams.queryInputShape;
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    uint32_t queryShapeHeadNum = nQ;
    uint32_t keyShapeHeadNum = nKV;
    uint32_t valueShapeHeadNum = nKV;
    const uint32_t queryDim = queryShape->GetStorageShape().GetDimNum();
    const uint32_t keyDim = keyShape->GetStorageShape().GetDimNum();
    const uint32_t valueDim = valueShape->GetStorageShape().GetDimNum();
    const uint32_t nIdx = inputLayout == InputLayout::BNSD ? 1U : 2U; // BNSD: 1; BSND:2

    if (((inputLayout == InputLayout::BNSD) || (inputLayout == InputLayout::BSND)) && (!enablePA)) {
        if ((queryDim == 4U) && (keyDim == 4U) && (valueDim == 4U)) { // dim num: 4
            queryShapeHeadNum = queryShape->GetStorageShape().GetDim(nIdx);
            keyShapeHeadNum = keyShape->GetStorageShape().GetDim(nIdx);
            valueShapeHeadNum = valueShape->GetStorageShape().GetDim(nIdx);
        } else {
            OP_LOGE(contextKeyParams.opName, "input dim of q(%u), k(%u), v(%u) must be 4 for BNSD or BSND format!", queryDim, keyDim, valueDim);
            return false;
        }
    } else if ((inputLayout == InputLayout::NSD) && (!enablePA)) {
        if ((queryDim == 3U) && (keyDim == 3U) && (valueDim == 3U)) { // dim num: 3
            queryShapeHeadNum = queryShape->GetStorageShape().GetDim(0);
            keyShapeHeadNum = keyShape->GetStorageShape().GetDim(0);
            valueShapeHeadNum = valueShape->GetStorageShape().GetDim(0);
        } else {
            OP_LOGE(contextKeyParams.opName, "input dim of q(%u), k(%u), v(%u) must be 3 for NSD format!", queryDim, keyDim, valueDim);
            return false;
        }
    }

    OP_CHECK_IF(nQ > 256U,   // The maximum limit for head is 256.
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "the query's heads num(%u) should not be more than 256!", nQ),
        return false);

    OP_CHECK_IF(queryShapeHeadNum != nQ,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "numHeads(%u) in query shape must be equal to numHeads(%u) in attr!", queryShapeHeadNum, nQ),
        return false);
    OP_CHECK_IF(keyShapeHeadNum != nKV,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "numHeads(%u) in key shape do not match numKeyValueHeads(%u) in attr!", keyShapeHeadNum, nKV),
        return false);
    OP_CHECK_IF(valueShapeHeadNum != nKV,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "numHeads(%u) in value shape do not match numKeyValueHeads(%u) in attr!", valueShapeHeadNum, nKV),
        return false);
    return true;
}

bool PromptFlashAttentionTiling::SetTilingHeadNumRatio(ContextParamsForPFATiling& contextKeyParams,
    const int64_t* numQueryHeads, const int64_t* numKeyValueHeads,
    PromptFlashAttentionTilingData& tilingData) {
    const int64_t nQ = *numQueryHeads;
    const int64_t nKV = *numKeyValueHeads;

    if ((nQ < 0) || (nKV < 0)) {
        OP_LOGE(contextKeyParams.opName, "numHeads(%d) or numKeyValueHeads(%d) is negative!", nQ, nKV);
        return false;
    }

    if (!CheckInputDimAndHeadNum(contextKeyParams, nQ, nKV)) {
        return false;
    }

    if (nKV == 0) { // Detected that nKV is the default value, which means that the customer did not pass in.
        tilingData.promptAttentionBaseParams.set_headNumRatio(1);
        return true;
    }

    if (nQ % nKV != 0) {
        OP_LOGE(contextKeyParams.opName, "numHeads(%d) must be divisible by numKeyValueHeads(%d)!", nQ, nKV);
        return false;
    } else {
        if (nQ / nKV > 64) {   // G cannot be greater than 64.
            OP_LOGE(contextKeyParams.opName, "numHeads / numKeyValueHeads = %d, cannot be larger than 64", nQ / nKV);
            return false;
        }
        tilingData.promptAttentionBaseParams.set_headNumRatio(nQ / nKV);
        return true;
    }
}

bool PromptFlashAttentionTiling::CheckNonEmptyShapeExceptions(ContextParamsForPFATiling& contextKeyParams,
    const gert::StorageShape* shape,
    const std::string &sName) {
    OP_CHECK_IF(shape == nullptr,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "%s shape is null.", sName.c_str()),
                    return true);
    OP_CHECK_IF(shape->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "Shape size of %s is overflow.", sName.c_str()),
                    return true);
    return false;
}

bool PromptFlashAttentionTiling::CheckBaseApiNonEmptyShapeExceptions(ContextParamsForPFATiling& contextKeyParams,
    const gert::StorageShape* shape,
    const std::string &sName) {
    OP_CHECK_IF(shape == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "%s shape is null.", sName.c_str()),
        return true);
    OP_CHECK_IF(shape->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue || shape->GetStorageShape().GetShapeSize() == 0,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "Shape size of %s is invalid.", sName.c_str()),
        return true);
    return false;
}

bool PromptFlashAttentionTiling::CheckActualSeqLength(ContextParamsForPFATiling& contextKeyParams, uint32_t b, uint32_t sQ, uint32_t sKV,
        const gert::Tensor* actualSeqLenQ, const gert::Tensor* actualSeqLenKV,
        InputLayout inLayout, PromptFlashAttentionTilingData& tilingData) {
    if (contextKeyParams.fromTilingSink != 0U) {
        return true;
    }
    uint64_t actualLenDimsQ  = (actualSeqLenQ  != nullptr) ? actualSeqLenQ->GetShapeSize()  : 0;
    uint64_t actualLenDimsKV = (actualSeqLenKV != nullptr) ? actualSeqLenKV->GetShapeSize() : 0;
    bool inputActualSeqQ  = !((actualLenDimsQ  == 0) || (actualSeqLenQ  == nullptr) || (actualSeqLenQ->GetData<int64_t>()  == nullptr));
    bool inputActualSeqKV = !((actualLenDimsKV == 0) || (actualSeqLenKV == nullptr) || (actualSeqLenKV->GetData<int64_t>() == nullptr));
    int64_t actualSeqQSum = 0;
    int64_t actualSeqTmp = 0; // The element of actualSeq.
    constexpr uint64_t actualLenDimsQMin = 1; // The length of actualSeqQ is 1
    constexpr uint64_t actualLenDimsKVMin = 1; // The length of actualSeqKV is 1

    // SH format verification separately.
    if (inLayout == InputLayout::SH) {
        if (inputActualSeqQ) {
            for (uint32_t i = LOOP_BEGIN_NUM; i < b; ++i) {
                actualSeqQSum = actualSeqQSum + static_cast<uint32_t>(actualSeqLenQ->GetData<int64_t>()[i]);
            }
            OP_CHECK_IF(actualSeqQSum != sQ,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "SH format sum of actual_seq_q(%ld) do not match s_q(%u)!", actualSeqQSum, sQ),
                return false);
        }
        return true;
    }
    // Pass the length of actSeqlen to kernel.
    tilingData.promptAttentionBaseParams.set_actualSeqLengthsSize(actualLenDimsQ);
    tilingData.promptAttentionBaseParams.set_actualSeqLengthsKVSize(actualLenDimsKV);

    if (inputActualSeqQ) {   // check the length of actual_seq_lengthsQ, whether is 1 or batch size
        OP_CHECK_IF(actualLenDimsQ < b && actualLenDimsQ > actualLenDimsQMin,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Dim(%lu) of the query's actual sequence lengths must equal to 1 or greater than or equal to batch size(%u)!", actualLenDimsQ, b),
            return false);
        uint32_t actualSeqQLength = std::min(static_cast<uint32_t>(actualLenDimsQ), b); // actual_seq_lengths is 1 or batch size
        for (uint32_t i = LOOP_BEGIN_NUM; i < actualSeqQLength; ++i) {
            actualSeqTmp = static_cast<int64_t>(actualSeqLenQ->GetData<int64_t>()[i]);
            OP_CHECK_IF(actualSeqTmp < 0 || actualSeqTmp > sQ,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "The query's actual sequence lengths[%u](%ld) must be in range[0, %u]!", i, actualSeqTmp, sQ),
                return false);
        }
    }

    if (inputActualSeqKV) {  // check the length of actual_seq_lengthsKV,whether is 1 or batch size
        OP_CHECK_IF(actualLenDimsKV < b && actualLenDimsKV > actualLenDimsKVMin,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Dim(%lu) of the key/value's actual sequence lengths must equal to 1 or greater than or equal to batch size(%u)!", actualLenDimsKV, b),
            return false);
        uint32_t actualSeqKVLength = std::min(static_cast<uint32_t>(actualLenDimsKV), b); // actual_seq_lengths_KV is 1 or batch size
        for (uint32_t i = LOOP_BEGIN_NUM; i < actualSeqKVLength; ++i) {
            actualSeqTmp = static_cast<int64_t>(actualSeqLenKV->GetData<int64_t>()[i]);
            if (contextKeyParams.isKvContinuous == 1U) {
                if (!enablePA) {
                    OP_CHECK_IF(actualSeqTmp < 0 || actualSeqTmp > sKV,
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "The key/value's actual sequence lengths[%u](%ld) must be in range[0, %u]!", i, actualSeqTmp, sKV),
                        return false);
                } else {
                    OP_CHECK_IF(actualSeqTmp < 0,
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "The key/value's actual sequence lengths[%u](%ld) must >= 0", i, actualSeqTmp),
                        return false);
                }
            } else {
                if ((inLayout == InputLayout::BSND) || (inLayout == InputLayout::BSH)) {
                    OP_CHECK_IF(actualSeqTmp < 0 || actualSeqTmp > contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(1),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "The key/value's actual sequence lengths[%u](%ld) must be in range[0, %li]!", i, actualSeqTmp,
                            contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(1)),
                        return false);
                } else {
                    OP_CHECK_IF(actualSeqTmp < 0 || actualSeqTmp > contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(2),
                        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "The key/value's actual sequence lengths[%u](%ld) must be in range[0, %li]!", i, actualSeqTmp,
                            contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(2)),
                        return false);
                }
            }
        }
    }

    return true;
}

bool PromptFlashAttentionTiling::CheckPseShiftTypeAndShape(ContextParamsForPFATiling& contextKeyParams,
    const gert::StorageShape *pseShiftShape, uint32_t b, uint64_t n, uint32_t s1, uint32_t s2) {
    if (contextKeyParams.fromTilingSink != 0U) {
        return true;
    }
    pseShiftElemType = contextKeyParams.pseShiftDataType;

    OP_CHECK_IF((curShortSocName == platform_ascendc::SocVersion::ASCEND310P),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "not support 310P when pse is not null"),
        return false);

    OP_CHECK_IF((inputType == ge::DT_FLOAT16 && pseShiftElemType != ge::DT_FLOAT16),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "q type is fp16, but pse shift type is not fp16, pse shift type = %s",
        g_strDataTypePfa.at(ValidPfaDataType(pseShiftElemType)).c_str()),
        return false);

    OP_CHECK_IF((inputType == ge::DT_BF16 && pseShiftElemType != ge::DT_BF16),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "q type is bf16, but pse shift type is not bf16, pse shift type = %s",
        g_strDataTypePfa.at(ValidPfaDataType(pseShiftElemType)).c_str()),
        return false);

    OP_CHECK_IF((inputType == ge::DT_INT8 && pseShiftElemType != ge::DT_FLOAT16),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "q type is int8, but pse shift type is not fp16, pse shift type = %s",
        g_strDataTypePfa.at(ValidPfaDataType(pseShiftElemType)).c_str()),
        return false);

    // Currently does not support D has super large size.
     OP_CHECK_IF((n == 0),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "num head is zero"),
        return false);

    // If pse is empty, there is no need to perform PSE actions.
    if (((pseShiftShape != nullptr) && (pseShiftShape->GetStorageShape().GetShapeSize() == 0)) ||
        (pseShiftShape == nullptr)) {
            usePseShift = 0U;
            return true;
    }

    if (pseShiftElemType == ge::DT_FLOAT16) {
        pseShiftElemSize = FLOAT16SIZE;
    } else if (pseShiftElemType == ge::DT_BF16) {
        pseShiftElemSize = BFLOAT16SIZE;
    }
    pseShiftTypeByteNum = BYTE_BLOCK / pseShiftElemSize;

    uint32_t pseShiftDim = pseShiftShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF((pseShiftDim != PSE_SHIFT_DIM),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "pse shift shape must be 4 dimension, rather than %u dimension", pseShiftDim),
        return false);

    pseShiftBatch = pseShiftShape->GetStorageShape().GetDim(0);
    uint32_t pseShiftN = pseShiftShape->GetStorageShape().GetDim(1);  // 1: The first dimension is N.
    pseShiftS1 = pseShiftShape->GetStorageShape().GetDim(2);          // 2: The second dimension is S1.
    pseShiftS2 = pseShiftShape->GetStorageShape().GetDim(3);          // 3: The third dimension is S2.
    OP_CHECK_IF(((pseShiftBatch != 1 && pseShiftBatch != b) || (pseShiftN != n) ||
        (pseShiftS1 < s1) || (pseShiftS2 < s2)),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "pse shift shape must be [1 or %u, %u, >=%u, >=%u], but now it is [%u, %u, %u, %u]",
        b, n ,s1, s2, pseShiftBatch, pseShiftN, pseShiftS1, pseShiftS2),
        return false);

    return true;
}

ge::graphStatus PromptFlashAttentionTiling::processPageAttentionInputFlag(ContextParamsForPFATiling& contextKeyParams) {
    std::string tempLayoutStr(contextKeyParams.layout);
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    int32_t keyDimNum = keyShape->GetStorageShape().GetDimNum();
    if (keyDimNum == PA_BBH_DIMS) { // BBH
        inputKvLayout = InputLayout::BBH; // only used for TND/NTD_TND, not used in any other layouts
        mlaTilingData.PFAinputParams.set_paCacheLayoutType(NUM_0);
    } else if (keyDimNum == PA_BNBD_DIMS) { // BNBD
        inputKvLayout = InputLayout::BNBD;
        mlaTilingData.PFAinputParams.set_paCacheLayoutType(NUM_1);
    } else if (keyDimNum == PA_NZ_DIMS) { // NZ
        inputKvLayout = InputLayout::NZ;
        mlaTilingData.PFAinputParams.set_paCacheLayoutType(NUM_2);
    } else {
        OP_CHECK_IF(((keyDimNum != PA_BBH_DIMS) && (keyDimNum != PA_BNBD_DIMS) && (keyDimNum != PA_NZ_DIMS)),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "keyDimNum(%d) invalid, When layout is %s and PA enable, only support keyDimNum in [3, 4, 5].", keyDimNum, tempLayoutStr.c_str()),
            return false);
    }
    return ge::GRAPH_SUCCESS;
}

bool PromptFlashAttentionTiling::checkPABlockSizeAndBlockTable(ContextParamsForPFATiling& contextKeyParams, const gert::Tensor* actualSeqLenKV, 
    const int32_t* curBlockSize, int64_t b) {
    OP_CHECK_IF((*curBlockSize % BLOCK_SIZE_BASE != 0U || *curBlockSize < BLOCK_SIZE_BASE || *curBlockSize > BLOCK_SIZE_MAX),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "block size(%d) should be a multiple of %d, and can't greater than %d when PA enable",
        *curBlockSize, BLOCK_SIZE_BASE, BLOCK_SIZE_MAX),
        return false);
    const gert::StorageShape* blockTableShape = contextKeyParams.blockTableShape;
    OP_CHECK_IF((((blockTableShape != nullptr) && (blockTableShape->GetStorageShape().GetShapeSize() == 0)) ||
        (blockTableShape == nullptr)),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "blockTable can't be empty when PA enable"),
            return false);
    ge::DataType blockTableType = contextKeyParams.blockTableType;
    OP_CHECK_IF((blockTableType != ge::DT_INT32),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "blockTable only support int32 when PA enable"),
        return false);
    int32_t blockTableDim = static_cast<int32_t>(blockTableShape->GetStorageShape().GetDimNum());
    OP_CHECK_IF(blockTableDim != NUM_2,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "the dim of block table must be 2 when PA enable"),
        return false);
    OP_CHECK_IF((actualSeqLenKV->GetShapeSize() != b),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "dims of actualSeqLenKV(%ld) should be equal to batch(%ld) when PA enable",
        actualSeqLenKV->GetShapeSize(), b),
        return false);
    return true;
}

bool PromptFlashAttentionTiling::culActSeqLenParamsWhenPA(ContextParamsForPFATiling& contextKeyParams, const gert::Tensor* actualSeqLenKV, int64_t& blockNumValid, 
    int32_t& maxBlockNumPerBatch, int32_t tempBlockSize, const int32_t* curBlockSize, int64_t b) {
    int32_t actualSeqKVPerBatch = 0;
    int32_t blockNumPerBatch = 0;
    int32_t maxKvSeqLen = 0;
    for (int32_t i = 0; i < b; i++) {
        actualSeqKVPerBatch = static_cast<int32_t>(actualSeqLenKV->GetData<int64_t>()[i]);
        maxKvSeqLen = std::max(maxKvSeqLen, static_cast<int32_t>(actualSeqLenKV->GetData<int64_t>()[i]));
        blockNumPerBatch = (actualSeqKVPerBatch + *curBlockSize - 1) / *curBlockSize;
        blockNumValid += blockNumPerBatch;
        if (blockNumPerBatch > maxBlockNumPerBatch) {
            maxBlockNumPerBatch = blockNumPerBatch;
        }
    }
    OP_CHECK_IF(maxKvSeqLen > (maxBlockNumPerBatch * tempBlockSize),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "maxKvSeqLen(%d) should be <= (maxBlockNumPerBatch * tempBlockSize(%d))",
        maxKvSeqLen, maxBlockNumPerBatch * tempBlockSize),
        return false);
    return true;
}

bool PromptFlashAttentionTiling::checkPAKeyValueDimsWhenBBH(ContextParamsForPFATiling& contextKeyParams, int32_t keyDim1, int32_t keyDim2, int32_t keyDim3, 
    int64_t blockNumValid, const int32_t* curBlockSize, int64_t h, int64_t headNumRatio) {
    auto hKV = h / headNumRatio; // the function SetTilingHeadNumRatio ensures that headNumRatio is not 0.
    int32_t tempBlockSize = keyDim2;
    int32_t tempKVH = keyDim3;
    PAlayoutType = 1;  // If it is three-dimensional, PAlayoutType = 1
    OP_CHECK_IF(((tempBlockSize != *curBlockSize) || (tempKVH * headNumRatio != h)), 
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "the dim of key [%d, %d, %d] is invalid, which should be [>=%ld, %d, %d] when PA enable", keyDim1,
        keyDim2, keyDim3, blockNumValid, *curBlockSize, hKV), // When assigning headNumRatio, it is guaranteed that it will not be 0
        return false);
    // In the BSH input of the PA scenario, it is required that the h of the KV matrix does not exceed 65535.  The dim and dim3 of the K/V have already been verified to be equal, so only the K matrix is verified here.
    OP_CHECK_IF(keyDim3 > 65535,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "layout of key/value is BSH, the h of key/value %d should not > 65535 when PA enable", keyDim3),
        return false);
    return true;
}

bool PromptFlashAttentionTiling::checkPAKeyValueDimsWhenBNBD(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* keyShape, const gert::StorageShape* valueShape, 
    int32_t keyDim1, int32_t keyDim2, int32_t keyDim3, int64_t blockNumValid, const int32_t* curBlockSize, int64_t n, int64_t h, int64_t headNumRatio) {
    if (n == 0) {
        return false;
    }
    auto nKV = n / headNumRatio;  // the function SetTilingHeadNumRatio ensures that headNumRatio is not 0.
    int32_t valueDim3 = valueShape->GetStorageShape().GetDim(THIRD_DIM);
    int32_t keyDim4 = keyShape->GetStorageShape().GetDim(FOURTH_DIM);
    OP_CHECK_IF((keyDim3 != valueDim3),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "the dim of key and value are inconsistent when PA enable"), return false);
    int32_t tempKVN = keyDim2;
    int32_t tempBlockSize = keyDim3;
    int32_t tempKD = keyDim4;
    PAlayoutType = 0;  // If it is four-dimensional, PAlayoutType = 0
    OP_CHECK_IF(((tempKVN * headNumRatio != n) || (tempBlockSize != *curBlockSize) || (tempKD != (h / n))),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "the dim of key [%d, %d, %d, %d] is invalid, which should be [>=%ld, %d, %d, %d] when PA enable",
        keyDim1, keyDim2, keyDim3, keyDim4, blockNumValid, nKV, *curBlockSize, tempKD), return false);
    return true;
}

bool PromptFlashAttentionTiling::checkPAKeyValueDimsWhenNZ(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* keyShape, const gert::StorageShape* valueShape, 
    int32_t keyDim1, int32_t keyDim2, int32_t keyDim3, int64_t blockNumValid, const int32_t* curBlockSize, int64_t h, int64_t n, int64_t headNumRatio) {
    if (n == 0) {
        return false;
    }
    auto nKV = n / headNumRatio;  // the function SetTilingHeadNumRatio ensures that headNumRatio is not 0.
    auto hKV = h / headNumRatio;
    int32_t keyDim4 = keyShape->GetStorageShape().GetDim(FOURTH_DIM);
    int32_t keyDim5 = keyShape->GetStorageShape().GetDim(FIFTH_DIM);
    int32_t valueDim4 = valueShape->GetStorageShape().GetDim(FOURTH_DIM);
    int32_t valueDim5 = valueShape->GetStorageShape().GetDim(FIFTH_DIM);
    OP_CHECK_IF(((keyDim4 != valueDim4) || (keyDim5 != valueDim5)),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "the dim of key and value are inconsistent when PA enable"), return false);
    int32_t tempKVN = keyDim2;
    int32_t tempBlockSize = keyDim4;
    auto kDataType = contextKeyParams.kDataType;
    OP_CHECK_IF((keyShape->GetStorageShape().GetDim(FIFTH_DIM) == 0),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "the fifth dim of key can not be 0"), return false);
    int32_t lastDim = NUM_32 / (sizeof(keyShape->GetStorageShape().GetDim(FIFTH_DIM)) / sizeof(kDataType));
    int32_t tempKD = keyDim3 * lastDim;
    OP_CHECK_IF(((tempKVN * headNumRatio != n) || (tempBlockSize != *curBlockSize) || (tempKD != (h / n))),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "the dim of key [%d, %d, %d, %d, %d] is invalid, which should be [>=%ld, %d, %d, %d, %d] when PA enable",
        keyDim1, keyDim2, keyDim3, keyDim4, keyDim5, blockNumValid, nKV, (hKV / nKV) / lastDim, *curBlockSize, lastDim), return false);
    return true;
}

bool PromptFlashAttentionTiling::CheckPAKeyValueParams(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* keyShape, const gert::StorageShape* valueShape, 
    int64_t blockNumValid, const int32_t* curBlockSize, int64_t n, int64_t h, int64_t headNumRatio) {
    int32_t keyDim = keyShape->GetStorageShape().GetDimNum();
    int32_t valueDim = valueShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(keyDim != valueDim,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "the dim num of key(%d) and value(%d) are inconsistent when PA enable", keyDim, valueDim),
        return false);
    OP_CHECK_IF(((keyDim != 3) && (keyDim != 4) && (keyDim != 5)),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "the dim of key and value must be in [3, 4, 5] when PA enable"),
        return false);
    int32_t keyDim1 = keyShape->GetStorageShape().GetDim(0);  // block_num_sum
    int32_t keyDim2 = keyShape->GetStorageShape().GetDim(1);
    int32_t keyDim3 = keyShape->GetStorageShape().GetDim(2);
    int32_t valueDim1 = valueShape->GetStorageShape().GetDim(0);
    int32_t valueDim2 = valueShape->GetStorageShape().GetDim(1);
    OP_CHECK_IF(((keyDim1 != valueDim1) || (keyDim2 != valueDim2)),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "the dim of key and value are inconsistent when PA enable"), return false);
    if (keyDim == NUM_4) {  // dim num: 4
        if (!checkPAKeyValueDimsWhenBNBD(contextKeyParams, keyShape, valueShape, keyDim1, keyDim2, keyDim3, blockNumValid, curBlockSize, n, h, headNumRatio)) {
            return false;
        }
    } else if (keyDim == NUM_5) { // dim num: 4
        if (!checkPAKeyValueDimsWhenNZ(contextKeyParams, keyShape, valueShape, keyDim1, keyDim2, keyDim3, blockNumValid, curBlockSize, h, n, headNumRatio)) {
            return false;
        }
    } else { // dim num: 3
        if (!checkPAKeyValueDimsWhenBBH(contextKeyParams, keyDim1, keyDim2, keyDim3, blockNumValid, curBlockSize, h, headNumRatio)) {
            return false;
        }
    }
    return true;
}

bool PromptFlashAttentionTiling::CheckPASparseMode(ContextParamsForPFATiling& contextKeyParams) {
    const int32_t* sparseMode = contextKeyParams.sparseMode;
    const gert::StorageShape* attenMaskShape = contextKeyParams.attentionMaskShape;
    OP_CHECK_IF((sparseMode == nullptr) || (*sparseMode != SPARSE_MODE_NO_MASK && *sparseMode != SPARSE_MODE_RIGHT_DOWN && *sparseMode != SPARSE_MODE_BAND),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When Layout is TND/NTD_TND and PA enabled, sparseMode only support 0 or 3 or 4, but sparseMode = %d",
        *sparseMode),
        return false);
    OP_CHECK_IF((*sparseMode == SPARSE_MODE_NO_MASK && attenMaskShape != nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When Layout is TND/NTD_TND and PA enabled, sparseMode = 0, not support attentionMask"),
        return false);
    return true;
}

bool PromptFlashAttentionTiling::CheckPAWhenBaseApi(ContextParamsForPFATiling& contextKeyParams, const gert::Tensor* actualSeqLenQ,
    const gert::Tensor* actualSeqLenKV, int64_t n, int64_t h, int64_t headNumRatio) {
    const int32_t* curBlockSize = contextKeyParams.blockSize;
    auto b = actualSeqLenQ->GetShapeSize();
    if (!checkPABlockSizeAndBlockTable(contextKeyParams, actualSeqLenKV, curBlockSize, b)) {
        return false;
    }
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    int64_t blockNumValid = 0;
    int32_t maxBlockNumPerBatch = 0;
    int32_t tempBlockSize = 0;
    if (inputKvLayout == InputLayout::BBH) {
        tempBlockSize = keyShape->GetStorageShape().GetDim(SECOND_DIM);
    } else if (inputKvLayout == InputLayout::BNBD) {
        tempBlockSize = keyShape->GetStorageShape().GetDim(THIRD_DIM);
    } else if (inputKvLayout == InputLayout::NZ) { // B N D/16 B 16
        tempBlockSize = keyShape->GetStorageShape().GetDim(FOURTH_DIM);
    }
    if (tempBlockSize == 0 || !culActSeqLenParamsWhenPA(contextKeyParams, actualSeqLenKV, blockNumValid, maxBlockNumPerBatch, tempBlockSize, curBlockSize, b)) {
        return false;
    }
    if (blockNumValid == 0 || !CheckPAKeyValueParams(contextKeyParams, keyShape, valueShape, blockNumValid, curBlockSize, n, h, headNumRatio)) {
        return false;
    }
    const gert::StorageShape* blockTableShape = contextKeyParams.blockTableShape;
    int32_t blockTableDim1 = static_cast<int32_t>(blockTableShape->GetStorageShape().GetDim(FIRST_DIM));
    blockTableDim2 = static_cast<int32_t>(blockTableShape->GetStorageShape().GetDim(SECOND_DIM));
    // When blockTableDim2>maxBlockNumPerBatch, the kernel should use blockTableDim2 as the second dimension when indexing block id in blockTable.
    // But for the verification of mask S2 axis, maxBlockNumPerBatch * tempBlockSize should still be used as the verification benchmark.
    if (contextKeyParams.fromTilingSink != 0) {
        tmpS2 = blockTableDim2 * (*curBlockSize); // Tiling sinking scene, workspace needs to be calculated, at this time, blockTableDim2 * blockSize is used as S2.
        return true;
    }
    OP_CHECK_IF((maxBlockNumPerBatch == 0),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "maxBlockNumPerBatch[%d] invalid, can not be 0",
        maxBlockNumPerBatch),
        return false);
    OP_CHECK_IF(((blockTableDim1 != b) || (blockTableDim2 < maxBlockNumPerBatch)),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "block table shape should be [%ld, >=%d], now is [%d, %d] when PA enable",
        b, maxBlockNumPerBatch, blockTableDim1, blockTableDim2),
        return false);
    int32_t keyDim1 = keyShape->GetStorageShape().GetDim(FIRST_DIM);
    PABlockNumSum = keyDim1;
    tmpS2 = maxBlockNumPerBatch * tempBlockSize;
    return true;
}

bool PromptFlashAttentionTiling::CheckPATypeAndShape(ContextParamsForPFATiling& contextKeyParams,
    const gert::Tensor* actualSeqLenKV, int32_t b, int64_t n, int64_t h, int64_t headNumRatio) {
    const int32_t* curBlockSize = contextKeyParams.blockSize;
    if (n == 0) {
        return false;
    }    
    OP_CHECK_IF((*curBlockSize % BLOCK_SIZE_BASE != 0 || *curBlockSize < BLOCK_SIZE_BASE || *curBlockSize > BLOCK_SIZE_MAX),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "block size(%d) should be a multiple of %d, and can't greater than %d when PA enable",
        *curBlockSize, BLOCK_SIZE_BASE, BLOCK_SIZE_MAX),
        return false);

    const gert::StorageShape* blockTableShape = contextKeyParams.blockTableShape;
    OP_CHECK_IF((((blockTableShape != nullptr) && (blockTableShape->GetStorageShape().GetShapeSize() == 0)) ||
        (blockTableShape == nullptr)),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "blockTable can't be empty when PA enable"),
            return false);
    int32_t blockTableDim1 = static_cast<int32_t>(blockTableShape->GetStorageShape().GetDim(0));
    blockTableDim2 = static_cast<int32_t>(blockTableShape->GetStorageShape().GetDim(1));
    // When blockTableDim2>maxBlockNumPerBatch, the kernel should use blockTableDim2 as the second dimension when indexing block id in blockTable.
    // But for the verification of mask S2 axis, maxBlockNumPerBatch * tempBlockSize should still be used as the verification benchmark.

    if (contextKeyParams.fromTilingSink != 0U) {
        tmpS2 = blockTableDim2 * (*curBlockSize); // Tiling sinking scene, workspace needs to be calculated, at this time, blockTableDim2 * blockSize is used as S2.
        return true;
    }
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    int32_t keyDim = keyShape->GetStorageShape().GetDimNum();
    int32_t valueDim = valueShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(keyDim != valueDim,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "the dim num of key(%d) and value(%d) are inconsistent when PA enable", keyDim, valueDim),
        return false);
    OP_CHECK_IF(((keyDim != 3) && (keyDim != 4)),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "the dim of key and value must be 3 or 4 when PA enable"),
        return false);

    int32_t keyDim1 = keyShape->GetStorageShape().GetDim(0);  // block_num_sum
    int32_t keyDim2 = keyShape->GetStorageShape().GetDim(1);
    int32_t keyDim3 = keyShape->GetStorageShape().GetDim(2);
    int32_t keyDim4 = 0;
    int32_t valueDim1 = valueShape->GetStorageShape().GetDim(0);
    int32_t valueDim2 = valueShape->GetStorageShape().GetDim(1);
    int32_t valueDim3 = valueShape->GetStorageShape().GetDim(2);
    int32_t valueDim4 = 0;
    int32_t tempBlockSize = keyDim2;
    int32_t tempH = keyDim3;
    int32_t tempN = 0;
    int32_t tempD = 0;

    if (keyDim == 4) {  // dim num: 4
        keyDim4 = keyShape->GetStorageShape().GetDim(3);  // 3: The third dimension.
        valueDim4 = valueShape->GetStorageShape().GetDim(3);  // 3: The third dimension.
        tempN = keyDim2;
        tempBlockSize = keyDim3;
        tempD = keyDim4;
    }

    OP_CHECK_IF(((keyDim1 != valueDim1) || (keyDim2 != valueDim2) || (keyDim3 != valueDim3) || (keyDim4 != valueDim4)),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "the dim of key and value are inconsistent when PA enable"),
        return false);

    int32_t actualSeqKVPerBatch = 0;
    int32_t blockNumPerBatch = 0;
    int64_t blockNumValid = 0;
    int32_t maxBlockNumPerBatch = 0;
    for (int32_t i = 0; i < b; i++) {
        actualSeqKVPerBatch = actualSeqLenKV->GetShapeSize() > 1 ? static_cast<int32_t>(actualSeqLenKV->GetData<int64_t>()[i]) :
            static_cast<int32_t>(actualSeqLenKV->GetData<int64_t>()[0]);
        blockNumPerBatch = (actualSeqKVPerBatch + *curBlockSize - 1) / *curBlockSize;
        blockNumValid += blockNumPerBatch;
        if (blockNumPerBatch > maxBlockNumPerBatch) {
            maxBlockNumPerBatch = blockNumPerBatch;
        }
    }

    if (keyDim == NUM_3) {  // dim num: 3
        PAlayoutType = 1U;  // If it is three-dimensional, PAlayoutType = 1
        OP_CHECK_IF(((tempBlockSize != *curBlockSize) || (tempH * headNumRatio != h)),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "the dim of key [%d, %d, %d] is invalid, which should be [>=%ld, %d, %d] when PA enable", keyDim1,
            keyDim2, keyDim3, blockNumValid, *curBlockSize, h / headNumRatio),  // When assigning headNumRatio, it is guaranteed that it will not be 0
            return false);
        // In the BSH input of the PA scenario, it is required that the h of the KV matrix does not exceed 65535.  The dim and dim3 of the K/V have already been verified to be equal, so only the K matrix is verified here.
        OP_CHECK_IF(keyDim3 > 65535,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "layout of key/value is BSH, the h of key/value %d should not > 65535 when PA enable",
            keyDim3),
            return false);
    } else {
        PAlayoutType = 0U;  // If it is four-dimensional, PAlayoutType = 0
        OP_CHECK_IF(((tempN * headNumRatio != n) || (tempBlockSize != *curBlockSize) || (tempD != (h / n))),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "the dim of key [%d, %d, %d, %d] is invalid, which should be [>=%ld, %d, %d, %d] when PA enable",
            keyDim1, keyDim2, keyDim3, keyDim4, blockNumValid, n / headNumRatio, *curBlockSize, (h / n)),
            return false);
    }

    std::string layoutStr(contextKeyParams.layout);
    if (layoutStr == "BNSD" || layoutStr == "BNSD_BSND" || layoutStr == "NSD") {
        OP_CHECK_IF(((keyDim != 3) && (keyDim != 4)),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "the layout of query is %s, key and value layout should be [>=%ld, %d, %d] or [>=%ld, %d, %d, %d] when PA enable",
            layoutStr.c_str(), blockNumValid, *curBlockSize, h, blockNumValid, n, *curBlockSize, (h / n)),
            return false);
    } else if (layoutStr == "BSH" || layoutStr == "BSND") {
        OP_CHECK_IF(keyDim != 3,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "the layout of query is %s, key and value layout should be [>=%ld, %d, %d] when PA enable",
            layoutStr.c_str(), blockNumValid, *curBlockSize, h),
            return false);
    } else {
        OP_LOGE(contextKeyParams.opName, "unsupported input data layout when PA enable");
        return false;
    }

    ge::DataType blockTableType = contextKeyParams.blockTableType;
    OP_CHECK_IF((blockTableType != ge::DT_INT32),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "blockTable only support int32 when PA enable"),
        return false);

    int32_t blockTableDim = static_cast<int32_t>(blockTableShape->GetStorageShape().GetDimNum());
    OP_CHECK_IF(blockTableDim != 2,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "the dim of block table must be 2 when PA enable"),
        return false);

    OP_CHECK_IF(((blockTableDim1 != b) || (blockTableDim2 < maxBlockNumPerBatch)),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "block table shape should be [%d, >=%d], now is [%d, %d] when PA enable",
        b, maxBlockNumPerBatch, blockTableDim1, blockTableDim2),
        return false);
    PABlockNumSum = keyDim1;
    tmpS2 = maxBlockNumPerBatch * tempBlockSize;
    return true;
}

bool PromptFlashAttentionTiling::CheckAttenMaskShape(ContextParamsForPFATiling& contextKeyParams,
    const int32_t* sparseMode,
    const gert::StorageShape* attenMaskShape,
    const uint64_t sQ, const uint64_t sK, const uint32_t batchSize) {
    if (contextKeyParams.fromTilingSink != 0U) {
        return true;
    }
    // Attention mask empty Tensor scene, no need to verify attention mask shape based on sparse mode value
    if (((attenMaskShape != nullptr) && (attenMaskShape->GetStorageShape().GetShapeSize() == 0)) ||
        (attenMaskShape == nullptr)) {
        return true;
    }
    uint32_t attenMaskDim = attenMaskShape->GetStorageShape().GetDimNum();
    uint32_t attenMaskBatch = 1U;
    uint32_t attenMaskS1, attenMaskS2;
    int32_t checkShapeRet = 0;
    if (attenMaskDim == ATTENTION_MASK_DIM2) {
        attenMaskS1 = attenMaskShape->GetStorageShape().GetDim(0);
        attenMaskS2 = attenMaskShape->GetStorageShape().GetDim(1);
        if ((sparseMode == nullptr) || *sparseMode == SPARSE_MODE_NO_MASK || *sparseMode == SPARSE_MODE_ALL_MASK) {
            checkShapeRet = static_cast<int32_t>((attenMaskS1 >= sQ) && (attenMaskS2 >= sK) &&
                (attenMaskBatch == 1U || attenMaskBatch == batchSize));
        }

        if ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_LEFT_UP || *sparseMode == SPARSE_MODE_RIGHT_DOWN || *sparseMode == SPARSE_MODE_BAND)) {
            checkShapeRet = attenMaskS1 == SPARSE_OPTIMIZE_ATTENTION_SIZE &&
                attenMaskS2 == SPARSE_OPTIMIZE_ATTENTION_SIZE;
        }
    } else if (attenMaskDim == ATTENTION_MASK_DIM3) {
        attenMaskBatch = attenMaskShape->GetStorageShape().GetDim(0);
        attenMaskS1 = attenMaskShape->GetStorageShape().GetDim(1);
        attenMaskS2 = attenMaskShape->GetStorageShape().GetDim(2);  // 2: When the dim is 3, the second dimension is S2.
        if ((sparseMode == nullptr) || *sparseMode == SPARSE_MODE_NO_MASK || *sparseMode == SPARSE_MODE_ALL_MASK) {
            checkShapeRet = static_cast<int32_t>((attenMaskS1 >= sQ) && (attenMaskS2 >= sK) &&
                (attenMaskBatch == 1U || attenMaskBatch == batchSize));
        }
        if ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_LEFT_UP || *sparseMode == SPARSE_MODE_RIGHT_DOWN || *sparseMode == SPARSE_MODE_BAND)) {
            checkShapeRet = static_cast<int32_t>(attenMaskBatch == 1U &&
                attenMaskS1 == SPARSE_OPTIMIZE_ATTENTION_SIZE &&
                attenMaskS2 == SPARSE_OPTIMIZE_ATTENTION_SIZE);
        }
    } else if (attenMaskDim == ATTENTION_MASK_DIM4) {
        uint32_t attenMaskN = 1U;
        attenMaskBatch = attenMaskShape->GetStorageShape().GetDim(0);
        attenMaskN = attenMaskShape->GetStorageShape().GetDim(1);
        attenMaskS1 = attenMaskShape->GetStorageShape().GetDim(2);  // 2: When the dim is 4, the second dimension is S1.
        attenMaskS2 = attenMaskShape->GetStorageShape().GetDim(3);  // 3: When the dim is 4, the third dimension is S2.
        if ((sparseMode == nullptr) || *sparseMode == SPARSE_MODE_NO_MASK || *sparseMode == SPARSE_MODE_ALL_MASK) {
            checkShapeRet = static_cast<int32_t>((attenMaskS1 >= sQ) && (attenMaskS2 >= sK) &&
                (attenMaskBatch == 1U || attenMaskBatch == batchSize));
        }
        if ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_LEFT_UP || *sparseMode == SPARSE_MODE_RIGHT_DOWN || *sparseMode == SPARSE_MODE_BAND)) {
            checkShapeRet = static_cast<int32_t>(attenMaskBatch == 1U && attenMaskN == 1U &&
                attenMaskS1 == SPARSE_OPTIMIZE_ATTENTION_SIZE &&
                attenMaskS2 == SPARSE_OPTIMIZE_ATTENTION_SIZE);
        }
    } else {
        OP_LOGE(contextKeyParams.opName, "attenMask dim(%u) must be 2 or 3 or 4!", attenMaskDim);
        return false;
    }
    if ((sparseMode == nullptr) || ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_NO_MASK)) ||
        ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_ALL_MASK))) {
        OP_CHECK_IF(checkShapeRet != 1, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "attenMask batch(%u) must be 1 or %u, attenMask Q_S(%u) must be larger than sQ(%u), attenMask KV_S(%u) must be larger than sK(%u), please check",
            attenMaskBatch, batchSize, attenMaskS1, sQ, attenMaskS2, sK), return false);
    }
    if ((sparseMode != nullptr) && ((*sparseMode == SPARSE_MODE_LEFT_UP) || (*sparseMode == SPARSE_MODE_RIGHT_DOWN) || (*sparseMode == SPARSE_MODE_BAND))) {
        OP_CHECK_IF(checkShapeRet != 1, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "attenMask shape must be (2048, 2048) or (1, 2048, 2048) or (1, 1, 2048, 2048) when sparse mode = %d",
            *sparseMode), return false);
    }
    return true;
}

bool PromptFlashAttentionTiling::CheckPAAntiquantSupportScenarios(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingData& tilingData) {
    OP_CHECK_IF(tilingData.promptAttentionBaseParams.get_isAntiPerchannel() != 1U,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "not support antiquant when blockTable is not null and antiquant mode is not perchannel"), return false);
    OP_CHECK_IF(contextKeyParams.inputDataType != ge::DT_FLOAT16 || contextKeyParams.kDataType != ge::DT_INT8,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "not support antiquant when q(%s) is not fp16 or kv(%s) is not int8",
        g_strDataTypePfa.at(ValidPfaDataType(contextKeyParams.inputDataType)).c_str(), g_strDataTypePfa.at(ValidPfaDataType(contextKeyParams.kDataType)).c_str()), return false);
    std::string layoutStr(contextKeyParams.layout);
    OP_CHECK_IF(layoutStr != "BSH" && layoutStr != "BSND",
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "not support antiquant when layout is not BSH or BSND"), return false);
    const int64_t* innerPrecisePtr = contextKeyParams.innerPrecisePtr;
    int64_t innerPreciseMode = innerPrecisePtr ? *innerPrecisePtr : HIGH_PERFORMANCE;
    innerPrecise = static_cast<uint64_t>(innerPreciseMode);
    OP_CHECK_IF(innerPrecise != HIGH_PERFORMANCE,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "PA perchannel antiquant only support high performance"), return false);
    OP_CHECK_IF(contextKeyParams.sparseMode == nullptr || *contextKeyParams.sparseMode != SPARSE_MODE_RIGHT_DOWN,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "PA perchannel antiquant only support sparse mode 3, but input sparseMode is (%d)", *contextKeyParams.sparseMode), return false);
    const gert::StorageShape* attenMaskShape = contextKeyParams.attentionMaskShape;
    OP_CHECK_IF((attenMaskShape != nullptr && attenMaskShape->GetStorageShape().GetShapeSize() == 0) || attenMaskShape == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "PA perchannel antiquant only support sparse mode 3, maskShape is null"), return false);
    OP_CHECK_IF(contextKeyParams.pseShift != nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "PA perchannel antiquant not support PSE"), return false);
    return true;
}

bool PromptFlashAttentionTiling::CheckPerchannelAntiquantParamsShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* antiquantScaleShape, const gert::StorageShape* antiquantOffsetShape,
        const uint64_t n, const uint32_t d, const uint64_t h, uint32_t paramFirstDim) const {
    OP_CHECK_IF(antiquantScaleShape == nullptr, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant scale is nullptr"), return false);
    if ((inputLayout == InputLayout::BNSD) || (inputLayout == InputLayout::NSD)) {
        OP_CHECK_IF(antiquantScaleShape->GetStorageShape().GetDimNum() != 4,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "the antiquant scale dim num[%zu] should be 4 if layout is BNSD or NSD!", antiquantScaleShape->GetStorageShape().GetDimNum()),
            return false);
        OP_CHECK_IF(antiquantScaleShape->GetStorageShape().GetDim(0) != paramFirstDim || antiquantScaleShape->GetStorageShape().GetDim(1) != n ||
            antiquantScaleShape->GetStorageShape().GetDim(2) != 1 || antiquantScaleShape->GetStorageShape().GetDim(3) != d,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant scale dim [%ld, %ld, %ld, %ld] is wrong!", antiquantScaleShape->GetStorageShape().GetDim(0),
            antiquantScaleShape->GetStorageShape().GetDim(1), antiquantScaleShape->GetStorageShape().GetDim(2), antiquantScaleShape->GetStorageShape().GetDim(3)),
            return false);
        OP_CHECK_IF(antiquantOffsetShape != nullptr && antiquantOffsetShape->GetStorageShape().GetDimNum() != 4,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant offset dim num[%zu] should be 4 if layout is BNSD or NSD!", antiquantOffsetShape->GetStorageShape().GetDimNum()),
            return false);
        OP_CHECK_IF(antiquantOffsetShape != nullptr && (antiquantOffsetShape->GetStorageShape().GetDim(0) != paramFirstDim || antiquantOffsetShape->GetStorageShape().GetDim(1) != n ||
            antiquantOffsetShape->GetStorageShape().GetDim(2) != 1 || antiquantOffsetShape->GetStorageShape().GetDim(3) != d),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant offset dim [%ld, %ld, %ld, %ld] is wrong!", antiquantOffsetShape->GetStorageShape().GetDim(0),
            antiquantOffsetShape->GetStorageShape().GetDim(1), antiquantOffsetShape->GetStorageShape().GetDim(2), antiquantOffsetShape->GetStorageShape().GetDim(3)),
            return false);
    } else if ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::SH)) {
        OP_CHECK_IF(antiquantScaleShape->GetStorageShape().GetDimNum() != 2,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant scale dim num[%zu] should be 2 if layout is BSH or SH!", antiquantScaleShape->GetStorageShape().GetDimNum()),
            return false);
        OP_CHECK_IF(antiquantScaleShape->GetStorageShape().GetDim(0) != paramFirstDim || antiquantScaleShape->GetStorageShape().GetDim(1) != h,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant scale dim [%ld, %ld] is wrong!", antiquantScaleShape->GetStorageShape().GetDim(0),
            antiquantScaleShape->GetStorageShape().GetDim(1)),
            return false);
        OP_CHECK_IF(antiquantOffsetShape != nullptr && antiquantOffsetShape->GetStorageShape().GetDimNum() != 2,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant offset dim num[%zu] should be 2 if layout is BSH or SH!", antiquantOffsetShape->GetStorageShape().GetDimNum()),
            return false);
        OP_CHECK_IF(antiquantOffsetShape != nullptr && (antiquantOffsetShape->GetStorageShape().GetDim(0) != paramFirstDim || antiquantOffsetShape->GetStorageShape().GetDim(1) != h),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant offset dim [%ld, %ld] is wrong!", antiquantOffsetShape->GetStorageShape().GetDim(0),
            antiquantOffsetShape->GetStorageShape().GetDim(1)),
            return false);
    } else if (inputLayout == InputLayout::BSND && !CheckPerchannelBSNDParamsShape(contextKeyParams, antiquantScaleShape, antiquantOffsetShape, n, d, paramFirstDim)) {
        return false;
    }
    return true;
}

bool PromptFlashAttentionTiling::CheckPerchannelBSNDParamsShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* antiquantScaleShape, const gert::StorageShape* antiquantOffsetShape,
    const uint64_t n, const uint32_t d, uint32_t paramFirstDim) const {
    OP_CHECK_IF(antiquantScaleShape->GetStorageShape().GetDimNum() != 3,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant scale dim num[%zu] should be 3 if layout is BSND!", antiquantScaleShape->GetStorageShape().GetDimNum()),
        return false);
    OP_CHECK_IF(antiquantScaleShape->GetStorageShape().GetDim(0) != paramFirstDim || antiquantScaleShape->GetStorageShape().GetDim(1) != n ||
        antiquantScaleShape->GetStorageShape().GetDim(2) != d,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant scale dim [%ld, %ld, %ld] is wrong!", antiquantScaleShape->GetStorageShape().GetDim(0),
        antiquantScaleShape->GetStorageShape().GetDim(1), antiquantScaleShape->GetStorageShape().GetDim(2)),
        return false);
    OP_CHECK_IF(antiquantOffsetShape != nullptr && antiquantOffsetShape->GetStorageShape().GetDimNum() != 3,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant offset dim num[%zu] should be 3 if layout is BSND!", antiquantOffsetShape->GetStorageShape().GetDimNum()),
        return false);
    OP_CHECK_IF(antiquantOffsetShape != nullptr && (antiquantOffsetShape->GetStorageShape().GetDim(0) != paramFirstDim || antiquantOffsetShape->GetStorageShape().GetDim(1) != n ||
        antiquantOffsetShape->GetStorageShape().GetDim(2) != d),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant offset dim [%ld, %ld, %ld] is wrong!", antiquantOffsetShape->GetStorageShape().GetDim(0),
        antiquantOffsetShape->GetStorageShape().GetDim(1), antiquantOffsetShape->GetStorageShape().GetDim(2)),
        return false);
    return true;
}

bool PromptFlashAttentionTiling::CheckAntiquantParamsShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* antiquantScaleShape,
    const gert::StorageShape* antiquantOffsetShape, const uint64_t n, const uint32_t d, const uint64_t h,
    PromptFlashAttentionTilingData& tilingData) {
    if (antiquantScaleShape != nullptr && antiquantScaleShape->GetStorageShape().GetDimNum() == 1) {
        tilingData.promptAttentionBaseParams.set_isAntiPerchannel(0);
        OP_CHECK_IF(antiquantScaleShape->GetStorageShape().GetDim(0) != 2,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant scale dim[0] = %ld, but it should be 2 under Per-Tensor mode!", antiquantScaleShape->GetStorageShape().GetDim(0)),
            return false);
        OP_CHECK_IF(antiquantOffsetShape != nullptr && antiquantOffsetShape->GetStorageShape().GetDim(0) != 2,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "antiquant offset dim[0] = %ld, but it should be 2 under Per-Tensor mode!", antiquantOffsetShape->GetStorageShape().GetDim(0)),
            return false);
    } else {
        tilingData.promptAttentionBaseParams.set_isAntiPerchannel(1);
        if (contextKeyParams.blockTable != nullptr) {
            OP_CHECK_IF(contextKeyParams.keyAntiquantMode == nullptr || contextKeyParams.valueAntiquantMode == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "k v antiquant mode is nullptr"),
                return false);
            int64_t keyAntiquantMode = *contextKeyParams.keyAntiquantMode;
            int64_t valueAntiquantMode = *contextKeyParams.valueAntiquantMode;
            OP_CHECK_IF(keyAntiquantMode != 0 || valueAntiquantMode != 0,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "keyAntiquantMode(%ld) or valueAntiquantMode(%ld) is not correct, keyAntiquantMode and valueAntiquantMode only support per-token and per-channel when keyAntiquantScale or valueAntiquantScale is enabled",
                keyAntiquantMode, valueAntiquantMode), return false);
            OP_CHECK_IF(!CheckPerchannelAntiquantParamsShape(contextKeyParams, contextKeyParams.KeyAntiquantScaleShape, contextKeyParams.KeyAntiquantOffsetShape, n, d, h, 1),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "key antiquant scale or offset shape is wrong"), return false);
            OP_CHECK_IF(!CheckPerchannelAntiquantParamsShape(contextKeyParams, contextKeyParams.valueAntiquantScaleShape, contextKeyParams.valueAntiquantOffsetShape, n, d, h, 1),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "value antiquant scale or offset shape is wrong"), return false);
        } else {
            uint32_t kvAntiquantParamFirstDim = 2;
            return CheckPerchannelAntiquantParamsShape(contextKeyParams, antiquantScaleShape, antiquantOffsetShape, n, d, h, kvAntiquantParamFirstDim);
        }
    }
    return true;
}

ge::graphStatus PromptFlashAttentionTiling::CheckPostQuantParams(const ContextParamsForPFATiling& contextKeyParams, uint64_t h, uint64_t n) const {
    const gert::StorageShape* quantScale2Shape = contextKeyParams.scale2Shape;
    const gert::StorageShape* quantOffset2Shape = contextKeyParams.offset2Shape;
    const ge::DataType quantScale2Type = contextKeyParams.quantScale2Type;
    const ge::DataType quantOffset2Type = contextKeyParams.quantOffset2Type;
    if (n == 0U) {
        return ge::GRAPH_FAILED;
    }
    int64_t quantScale2ShapeSize = 0;
    int64_t quantOffset2ShapeSize = 0;
    uint32_t quantD = 0;
    uint64_t queryD = h / n;

    if (outputType == ge::DT_INT8) {
        // Basic verification: quantScale2 must be inputted and not an empty tensor
        OP_CHECK_IF(quantScale2Shape == nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant scale is nullptr when output type is int8."),
            return ge::GRAPH_FAILED);
        quantScale2ShapeSize = quantScale2Shape->GetStorageShape().GetShapeSize();
        quantD = quantScale2ShapeSize / n;
        OP_CHECK_IF(quantScale2ShapeSize == 0,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "quant_scale2 is empty tensor when output type is int8."),
            return ge::GRAPH_FAILED);

        // altert unsupported situation(post quant per-tensor + BF16 + BSH + D unalign)
        if ((contextKeyParams.inputDataType == ge::DT_BF16) && (quantScale2ShapeSize == 1) && (inputLayout == InputLayout::BSH) && (queryD % BYTE_BLOCK != 0)) {
            OP_LOGW(contextKeyParams.opName, "post quant per-tensor doesn't support D unaligned(%u), when qkv is bf16 and layout is BSH.", queryD);
        }

        // Cross characteristic verification: The After Quant per-channel does not currently support left padding, ring attention, and D non 32B alignment
        if (quantScale2ShapeSize != 1) {
            OP_CHECK_IF((contextKeyParams.queryPaddingSize != nullptr) || (contextKeyParams.kvPaddingSize != nullptr),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant per-channel do not support left padding."),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(contextKeyParams.isSoftMaxLseEnable == true,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant per-channel do not support ring attention."),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(quantD % BYTE_BLOCK != 0,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant per-channel do not support D(%u) non-32-byte aligned.", quantD),
                return ge::GRAPH_FAILED);
        }

        // dtype verification
        OP_CHECK_IF((quantScale2Type != ge::DT_BF16) && (quantScale2Type != ge::DT_FLOAT) && (quantScale2Type != ge::DT_FLOAT16),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant scale dtype(%s) only support bf16, fp16 and fp32.",
            g_strDataTypePfa.at(ValidPfaDataType(quantScale2Type)).c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((quantOffset2Shape != nullptr) && (quantScale2Type != quantOffset2Type),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant scale dtype(%s) and offset dtype(%s) must be consistent.",
            g_strDataTypePfa.at(ValidPfaDataType(quantScale2Type)).c_str(), g_strDataTypePfa.at(ValidPfaDataType(quantOffset2Type)).c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((inputType != ge::DT_BF16) && (quantScale2Type == ge::DT_BF16),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant scale and offset support bf16 only if input dtype(%s) is bf16.",
            g_strDataTypePfa.at(ValidPfaDataType(inputType)).c_str()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((inputType != ge::DT_FLOAT16) && (quantScale2Type == ge::DT_FLOAT16),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "post quant scale and offset support fp16 only if input dtype(%s) is fp16.",
            g_strDataTypePfa.at(ValidPfaDataType(inputType)).c_str()),
            return ge::GRAPH_FAILED);

        // shape verification
        if (quantOffset2Shape != nullptr) {
            quantOffset2ShapeSize = quantOffset2Shape->GetStorageShape().GetShapeSize();
            OP_CHECK_IF(quantScale2ShapeSize != quantOffset2ShapeSize,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "the output's dequant scale dimension multiply result(%ld) do not equal the output's dequant offset dimension multiply result(%ld).",
                quantScale2ShapeSize, quantOffset2ShapeSize), return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF((quantScale2ShapeSize != 1) && (quantScale2ShapeSize != h),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "post the output's dequant scale/offset dimension multiply result only support 1 and H(%u), now is (%ld). "
            "The shape of scale2/offset2 do not match that of query, or D is not 32 Byte aligned, "
            "which post quant per-channel do not support.", h, quantScale2ShapeSize), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void PromptFlashAttentionTiling::Align(uint32_t &num) {
    num = (num + typeByteNum - 1U) / typeByteNum * typeByteNum;
}

// Code for ut, no pratical to use.
ge::graphStatus PromptFlashAttentionTiling::GetBasicShape310P(uint32_t &b,
    uint32_t &bKV, uint32_t &s, uint64_t &h, uint32_t &seqInnerSize, const gert::StorageShape *queryShape,
    const gert::StorageShape *keyShape, const uint64_t n, size_t actualLenDims, size_t actualLenDimsKV) {
    if (n == 0U) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(queryShape == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("GetBasicShape310P", "queryShape is null."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyShape == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("GetBasicShape310P", "keyShape is null."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(n == 0,
        OPS_REPORT_VECTOR_INNER_ERR("GetBasicShape310P", "n is 0."),
        return ge::GRAPH_FAILED);
    if (inputLayout == InputLayout::NSD) {
        uint32_t d;
        b = 1U;
        bKV = 1U;
        s = queryShape->GetStorageShape().GetDim(1);
        seqInnerSize = keyShape->GetStorageShape().GetDim(1);
        d = queryShape->GetStorageShape().GetDim(2); // dim num: 2
        Align(d);
        h = (d * n);
        return ge::GRAPH_SUCCESS;
    }

    if (inputLayout == InputLayout::BNSD) {
        uint32_t d;
        b = queryShape->GetStorageShape().GetDim(0);
        bKV = keyShape->GetStorageShape().GetDim(0);
        s = queryShape->GetStorageShape().GetDim(2); // dim num: 2
        seqInnerSize = keyShape->GetStorageShape().GetDim(2); // dim num: 2
        d = queryShape->GetStorageShape().GetDim(3); // dim num: 3
        Align(d);
        h = (queryShape->GetStorageShape().GetDim(1) * d);
        return ge::GRAPH_SUCCESS;
    }

    if (inputLayout == InputLayout::SH) {
        b = ((actualLenDims == 0) ? 1 : actualLenDims); // When the input layout is SH and actual_seq is not input, the batch of query is set to 1.
        bKV = ((actualLenDimsKV == 0) ? 1 : actualLenDimsKV); // When the input layout is SH and actual_seqkv is not input, the batch of key/value is set to 1.
        uint32_t d;
        s = queryShape->GetStorageShape().GetDim(0);
        h = queryShape->GetStorageShape().GetDim(1);
        seqInnerSize = keyShape->GetStorageShape().GetDim(0);

        Align(s);
        Align(seqInnerSize);
        d = (h / n);
        Align(d);
        h = (d * n);
        return ge::GRAPH_SUCCESS;
    }

    if (inputLayout == InputLayout::BSH) {
        uint32_t d;
        b = queryShape->GetStorageShape().GetDim(0); // dim num: 0, btach of query
        bKV = keyShape->GetStorageShape().GetDim(0); // dim num: 0, btach of kv
        s = queryShape->GetStorageShape().GetDim(1); // dim num: 0, s of query
        h = queryShape->GetStorageShape().GetDim(2); // dim num: 2
        seqInnerSize = keyShape->GetStorageShape().GetDim(1);
        d = h / n;
        Align(d);
        h = d * n;
        return ge::GRAPH_SUCCESS;
    }

    if (inputLayout == InputLayout::BSND) {
        uint32_t d;
        b = (queryShape->GetStorageShape().GetDim(0));
        bKV = (keyShape->GetStorageShape().GetDim(0));
        s = (queryShape->GetStorageShape().GetDim(1));
        d = (queryShape->GetStorageShape().GetDim(INDEX_3));
        seqInnerSize = (keyShape->GetStorageShape().GetDim(1));
        Align(d);
        h = (d * n);
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus PromptFlashAttentionTiling::GetAndCheckEmptyQueryShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape *queryShape) const {
    OP_CHECK_IF(queryShape == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "queryShape is null."),
        return ge::GRAPH_FAILED);
    uint32_t b = 0;
    uint32_t n = 0;
    uint32_t s = 0;
    uint32_t d = 0;
    uint32_t h = 0;
    if ((inputLayout == InputLayout::BNSD) || (inputLayout == InputLayout::NSD)) {
        if (queryShape->GetStorageShape().GetDimNum() == 3) { // dim num: 3
            b = 1;
            n = queryShape->GetStorageShape().GetDim(0);
            s = queryShape->GetStorageShape().GetDim(1);
            d = queryShape->GetStorageShape().GetDim(2); // dim num: 2
        } else {
            b = (queryShape->GetStorageShape().GetDim(0));
            n = (queryShape->GetStorageShape().GetDim(1));
            s = (queryShape->GetStorageShape().GetDim(2)); // dim num: 2
            d = (queryShape->GetStorageShape().GetDim(3)); // dim num: 3
        }
    } else if ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND) ||
        (inputLayout == InputLayout::SH)) {
        if (queryShape->GetStorageShape().GetDimNum() == NUM_2) { // dim num: 2
            b = 1; // Process according to batch = 1.
            s = (queryShape->GetStorageShape().GetDim(0));
            h = (queryShape->GetStorageShape().GetDim(1));
        } else if (queryShape->GetStorageShape().GetDimNum() == 3) { // 3 : BSH
            b = (queryShape->GetStorageShape().GetDim(0));
            s = (queryShape->GetStorageShape().GetDim(1));
            h = (queryShape->GetStorageShape().GetDim(2)); // dim num: 2
        } else { // BSND
            b = queryShape->GetStorageShape().GetDim(0);
            s = queryShape->GetStorageShape().GetDim(1);
            n = queryShape->GetStorageShape().GetDim(2); // dim num: 2
            d = queryShape->GetStorageShape().GetDim(3); // dim num: 3
        }
    } else if (InputLayoutIsTNDLike()) {
        b = 1u;
        n = GetNFromInputShape(QUERY_INDEX, queryShape);
        s = 1u;
        d = queryShape->GetStorageShape().GetDim(THIRD_DIM);
    } else {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(b > BLIMIT, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "batch size should <= 65536, but batch size = %u", b), return ge::GRAPH_FAILED);
    if (s > SLIMIT) {
        OP_LOGW(contextKeyParams.opName,
            "seq should <= 20m, but seq = %u", s);
    }
    if (inputLayout == InputLayout::BSH || inputLayout == InputLayout::SH) {
        OP_CHECK_IF(h > DLIMIT * NLIMIT, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "h should <= 512 * 256, but h = %u", h), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(n > NLIMIT, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "n should <= 256, but n = %u", n), return ge::GRAPH_FAILED);
        OP_CHECK_IF(d > DLIMIT, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "D should <= 512, but d = %u", d), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

void PromptFlashAttentionTiling::SetBaseApiTilingData(ContextParamsForPFATiling& contextKeyParams, std::vector<int64_t>& actualSeqLengths,
    std::vector<int64_t>& actualSeqLengthsKV) {
    const gert::StorageShape* queryShape = contextKeyParams.queryInputShape;
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;

    PromptAttentionBaseApiBaseParams* baseParams = &baseApiTilingData.promptAttentionBaseApiBaseParams;
    uint32_t batchSize = 0U;
    uint32_t headSize = 0U;
    uint32_t headSizeV = 0U;
    uint32_t s = 0U;
    uint32_t seqInnerSize = 0U;
    const uint64_t headNum = *contextKeyParams.headsNumber;
    const uint64_t kvHeadNum = (*contextKeyParams.numKeyValueHeads != 0) ? *contextKeyParams.numKeyValueHeads : headNum;
    uint32_t maxSeqLen = 0;
    uint32_t maxKvSeqLen = 0;
    const gert::Tensor* tempData = contextKeyParams.actualSequenceLengthQ;
    const gert::Tensor* tempDataKV = contextKeyParams.actualSequenceLengthKV;
    size_t actualLenDims = (tempData != nullptr) ? tempData->GetShapeSize() : 0;
    size_t actualLenDimsKV = (tempDataKV != nullptr) ? tempDataKV->GetShapeSize() : 0;
    uint32_t inLayoutType = 1U;

    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        if (inputLayout == InputLayout::BSH) {
            batchSize = queryShape->GetStorageShape().GetDim(ZERO_HOST_TILING);
            s = queryShape->GetStorageShape().GetDim(TWO_HOST_TILING);
            seqInnerSize = queryShape->GetStorageShape().GetDim(TWO_HOST_TILING);
            headSize = queryShape->GetStorageShape().GetDim(ONE_HOST_TILING) * SIXTEEN_HOST_TILING / headNum;
            headSizeV = valueShape->GetStorageShape().GetDim(ONE_HOST_TILING) * SIXTEEN_HOST_TILING / kvHeadNum;
        } else if (inputLayout == InputLayout::BNSD) {
            inLayoutType = 3U;
            batchSize = queryShape->GetStorageShape().GetDim(ZERO_HOST_TILING) / headNum;
            s = queryShape->GetStorageShape().GetDim(TWO_HOST_TILING);
            seqInnerSize = queryShape->GetStorageShape().GetDim(TWO_HOST_TILING);
            headSize = queryShape->GetStorageShape().GetDim(ONE_HOST_TILING) * SIXTEEN_HOST_TILING;
            headSizeV = valueShape->GetStorageShape().GetDim(ONE_HOST_TILING) * SIXTEEN_HOST_TILING;
        }
    } else {
        if (inputLayout == InputLayout::SH) {
            inLayoutType = 0U;
            batchSize = actualLenDims == static_cast<size_t>(ZERO_HOST_TILING) ? ONE_HOST_TILING : actualLenDims; // When the input layout is SH and actual_seq is not input, the batch of query is set to 1.
            s = queryShape->GetStorageShape().GetDim(ZERO_HOST_TILING);
            seqInnerSize = keyShape->GetStorageShape().GetDim(ZERO_HOST_TILING);
            headSize = queryShape->GetStorageShape().GetDim(ONE_HOST_TILING) / headNum;
            headSizeV = valueShape->GetStorageShape().GetDim(ONE_HOST_TILING) / kvHeadNum;
        } else if (inputLayout == InputLayout::BSH) {
            batchSize = queryShape->GetStorageShape().GetDim(ZERO_HOST_TILING);
            s = queryShape->GetStorageShape().GetDim(ONE_HOST_TILING);
            seqInnerSize = keyShape->GetStorageShape().GetDim(ONE_HOST_TILING);
            headSize = queryShape->GetStorageShape().GetDim(TWO_HOST_TILING) / headNum;
            headSizeV = valueShape->GetStorageShape().GetDim(TWO_HOST_TILING) / kvHeadNum;
        } else if (inputLayout == InputLayout::BSND) {
            batchSize = queryShape->GetStorageShape().GetDim(ZERO_HOST_TILING);
            s = queryShape->GetStorageShape().GetDim(ONE_HOST_TILING);
            seqInnerSize = keyShape->GetStorageShape().GetDim(ONE_HOST_TILING);
            headSize = queryShape->GetStorageShape().GetDim(THREE_HOST_TILING);
            headSizeV = valueShape->GetStorageShape().GetDim(THREE_HOST_TILING);
        } else if (inputLayout == InputLayout::BNSD) {
            batchSize = queryShape->GetStorageShape().GetDim(ZERO_HOST_TILING);
            s = queryShape->GetStorageShape().GetDim(TWO_HOST_TILING); // dim num: 2
            seqInnerSize = keyShape->GetStorageShape().GetDim(TWO_HOST_TILING); // dim num: 2
            headSize = queryShape->GetStorageShape().GetDim(THREE_HOST_TILING);
            headSizeV = valueShape->GetStorageShape().GetDim(THREE_HOST_TILING);
        }
    }
    actualSeqLengths.resize(batchSize);
    actualSeqLengthsKV.resize(batchSize);
    for (size_t i = LOOP_BEGIN_NUM; i < batchSize; i++) {
        if ((actualLenDims == 0) || (tempData == nullptr) || (tempData->GetData<int64_t>() == nullptr)) {
            actualSeqLengths[i] = s;
        } else {
            actualSeqLengths[i] = (actualLenDims > 1) ? static_cast<uint32_t>(tempData->GetData<int64_t>()[i]) :
                static_cast<uint32_t>(tempData->GetData<int64_t>()[0]);
        }
        if ((actualLenDimsKV == 0) || (tempDataKV == nullptr) || (tempDataKV->GetData<int64_t>() == nullptr)) {       // The user did not input act_seq_kv
            actualSeqLengthsKV[i] = seqInnerSize;
        } else {
            actualSeqLengthsKV[i] = (actualLenDimsKV > 1) ? static_cast<uint32_t>(tempDataKV->GetData<int64_t>()[i]) :
                static_cast<uint32_t>(tempDataKV->GetData<int64_t>()[0]);
        }
        maxSeqLen = std::max(maxSeqLen, static_cast<uint32_t>(actualSeqLengths[i]));
        maxKvSeqLen = std::max(maxKvSeqLen, static_cast<uint32_t>(actualSeqLengthsKV[i]));
    }

    baseParams->set_maskTypeByteNum(BYTE_BLOCK / dataTypeSize);
    baseParams->set_batchSize(batchSize);
    baseParams->set_headNumSize(headNum);
    baseParams->set_headSize(headSize);
    baseParams->set_kvHeadNumSize(kvHeadNum);
    baseParams->set_embeddingSizeV(headSizeV);
    baseParams->set_maxSeqLen(maxSeqLen);
    baseParams->set_maxKvSeqLen(maxKvSeqLen);
    baseParams->set_inputLayoutType(inLayoutType);

    tilingMod = (headSize != headSizeV || headSize > MLA_THRESHOLD) ? TilingMod::CVDIFF_MLA : TilingMod::CVDIFF_BASE_API;
    atbRunFlag_ = true;
}

void PromptFlashAttentionTiling::SetBaseApiSeqTilingData(ContextParamsForPFATiling& contextKeyParams, std::vector<int64_t>& actualSeqLengths,
        std::vector<int64_t>& actualSeqLengthsKV) {
    PromptAttentionBaseApiBaseParams* baseParams = &baseApiTilingData.promptAttentionBaseApiBaseParams;

    const float* scaleValue = contextKeyParams.scaleValue;

    uint32_t ppMScalar = 0;
    uint32_t ppNScalar = 0;
    int64_t totalQBlkNum = 0;
    uint32_t totalQBlkNumFirst = 0;

    baseParams->set_isClamp(BASE_DEFAULT_VALUE_0);
    baseParams->set_clampMin(BASE_DEFAULT_VALUE_0);
    baseParams->set_clampMax(BASE_DEFAULT_VALUE_0);
    baseParams->set_dataShapeType(BASE_DEFAULT_VALUE_0);
    baseParams->set_scaleType(BASE_DEFAULT_VALUE_0);
    baseParams->set_tor(*scaleValue);

    for (uint32_t i = LOOP_BEGIN_NUM; i < baseParams->get_batchSize(); i++) {
        int seqLen = actualSeqLengths[i];
        int subSeqInnerLen = actualSeqLengthsKV[i];

        int32_t qSeqlenAligned = (seqLen + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        int32_t kvSeqlenAligned = (subSeqInnerLen + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        int32_t embeddingSizeAligned = (baseParams->get_headSize() + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        int32_t tilingK = embeddingSizeAligned < LONG_SEQ_LEN ? LONG_SEQ_LEN : embeddingSizeAligned;
        int32_t nUbd = tilingMod == TilingMod::CVDIFF_BASE_API ? std::min(LONG_SEQ_LEN, kvSeqlenAligned)
            : std::min((PP_BLOCK_BUFFER_SIZE / tilingK / BLOCK_SIZE) * BLOCK_SIZE, kvSeqlenAligned);
        int32_t nIbd = ConvertValueToIndexNN(nUbd, PP_NN_NUM - 1);
        nIbd = nIbd < 0 ? 0 : nIbd;
        int32_t mUbd = tilingMod == TilingMod::CVDIFF_BASE_API ? std::min(LONG_SEQ_LEN, qSeqlenAligned)
            : std::min((PP_BLOCK_BUFFER_SIZE / std::max(embeddingSizeAligned, PP_NN[nIbd]) / BLOCK_SIZE) *
                BLOCK_SIZE, qSeqlenAligned);
        int32_t mIbd = ConvertValueToIndexMM(mUbd, PP_MM_NUM - 1);
        mIbd = mIbd < 0 ? 0 : mIbd;
        int32_t curQBlockNum = (seqLen != 0) ? ((seqLen + PP_MM[mIbd] - 1) / PP_MM[mIbd]) : 0;
        totalQBlkNum += curQBlockNum;
        if (i == 0) {
            ppMScalar = PP_MM[mIbd];
            ppNScalar = PP_NN[nIbd];
            totalQBlkNumFirst = totalQBlkNum;
        }
    }

    QuantType quantType = QuantType::TYPE_QUANT_UNDEFINED;
    const gert::StorageShape* deqScale1Shape = contextKeyParams.deqScale1Shape;
    const gert::StorageShape* quantScale1Shape = contextKeyParams.scale1Shape;
    const gert::StorageShape* deqScale2Shape = contextKeyParams.deqScale2Shape;
    auto qDataType = contextKeyParams.inputDataType;
    auto kDataType = contextKeyParams.kDataType;
    auto vDataType = contextKeyParams.vDataType;
    auto outputDataType = contextKeyParams.outputDataType;
    if ((qDataType == ge::DT_INT8) && (kDataType == ge::DT_INT8) && (vDataType == ge::DT_INT8) &&
        (outputDataType == ge::DT_FLOAT16 || outputDataType == ge::DT_BF16)) {
        if ((deqScale1Shape != nullptr && deqScale1Shape->GetStorageShape().GetShapeSize() > 0) &&
            (quantScale1Shape != nullptr && quantScale1Shape->GetStorageShape().GetShapeSize() > 0) &&
            (deqScale2Shape != nullptr && deqScale2Shape->GetStorageShape().GetShapeSize() > 0)) {
            quantType = QuantType::TYPE_QUANT_QKV_OFFLINE;
        } else {
            quantType = QuantType::TYPE_QUANT_QKV_ONLINE;
        }
    }

    baseParams->set_quantType(static_cast<uint32_t>(quantType));
    baseParams->set_totalQBlkNum(totalQBlkNum);
    baseParams->set_totalQBlkNumFirst(totalQBlkNumFirst);
    baseParams->set_ppMScalar(ppMScalar);
    baseParams->set_ppNScalar(ppNScalar);
}

ge::graphStatus PromptFlashAttentionTiling::CheckBaseAPISupportScenarios(ContextParamsForPFATiling& contextKeyParams) {
    const int32_t* sparseMode = contextKeyParams.sparseMode;

    const std::unordered_set<int32_t> sparseModeRange = {20, 21, 22};
    if (sparseModeRange.find(*sparseMode) == sparseModeRange.end()) {
        return ge::GRAPH_FAILED;
    }

    // 310 PFA 增加
    std::string layoutStr(contextKeyParams.layout);
    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P && (layoutStr == "BSH" || layoutStr == "BNSD") &&
        (contextKeyParams.queryInputShape->GetStorageShape().GetDimNum() == QUERY_INPUT_DIM_NUM_HOST_TILING && contextKeyParams.queryInputShape->GetStorageShape().GetDim(THREE_HOST_TILING) == BLOCK_SIZE)) {
        atbRunFlag_ = true;
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::AtbSplitBlock(ContextParamsForPFATiling& contextKeyParams)
{
    PromptAttentionBaseApiBaseParams* baseParams = &baseApiTilingData.promptAttentionBaseApiBaseParams;
    int32_t headNum = *contextKeyParams.headsNumber;
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "platformInfoPtr is null"), return ge::GRAPH_FAILED);
    auto ascendcPlatformLocal = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint32_t nzRealCoreNum = ascendcPlatformLocal.CalcTschBlockDim(aivNum, aicNum, aivNum);
    if (nzRealCoreNum == 0U) {
        return ge::GRAPH_FAILED;
    }
    uint32_t procNum = static_cast<uint32_t>((headNum * baseParams->get_totalQBlkNum()));
    uint32_t procPerBlk = nzRealCoreNum == 0U ? procNum : procNum / nzRealCoreNum;
    uint32_t tailBlks = nzRealCoreNum == 0U ? procNum : procNum % nzRealCoreNum;

    uint32_t startBlk = 0;
    uint32_t endBlk = 0;

    uint32_t numBlocks = procNum < nzRealCoreNum ? procNum : nzRealCoreNum;
    if (numBlocks > MAX_BLOCK_DIM) {
        return ge::GRAPH_FAILED;
    }
    std::vector<uint32_t> startBlkArray(MAX_BLOCK_DIM, 0U);
    std::vector<uint32_t> endBlkArray(MAX_BLOCK_DIM, 0U);

    for (uint32_t i = 0; i < numBlocks; i++) {
        startBlk = endBlk;
        endBlk = i < tailBlks ? endBlk + procPerBlk + 1U : endBlk + procPerBlk;
        startBlkArray[i] = static_cast<uint32_t>(startBlk);
        endBlkArray[i] = static_cast<uint32_t>(endBlk);
    }
    PromptAttentionSplitCoreParams* splitCoreParams = &baseApiTilingData.promptAttentionSplitCoreParams;
    splitCoreParams->set_startBlkArray(startBlkArray.data());
    splitCoreParams->set_endBlkArray(endBlkArray.data());
    return ge::GRAPH_SUCCESS;
}

void PromptFlashAttentionTiling::SetTensorSizeParams()
{
    auto &tensorSizeParams = mlaTilingData.PFAtensorSizeParams;
    auto &coreParams = mlaTilingData.PFAcoreParams;
    int64_t batchInnerSize = coreParams.get_bBaseSize() * coreParams.get_n2BaseSize() * coreParams.get_gBaseSize();
    tensorSizeParams.set_bmm1ResUbSize(batchInnerSize * s1BasicBlock * s2BasicBlock);
    tensorSizeParams.set_attenMaskUbSize(batchInnerSize * s1BasicBlock * s2BasicBlock);
    tensorSizeParams.set_castUbSize(batchInnerSize * s1BasicBlock * std::max(s2BasicBlock, dBasicBlock));
    tensorSizeParams.set_softmaxMaxUbSize(batchInnerSize * s1BasicBlock * (BYTE_BLOCK / sizeof(float)));
    tensorSizeParams.set_softmaxSumUbSize(batchInnerSize * s1BasicBlock * (BYTE_BLOCK / sizeof(float)));
    tensorSizeParams.set_softmaxExpUbSize(batchInnerSize * s1BasicBlock * (BYTE_BLOCK / softmaxDataTypeSize));
    tensorSizeParams.set_apiTmpBufferBytes(apiMaxUBSize);

    tensorSizeParams.set_bmm2ResUbSize(batchInnerSize * s1BasicBlock * dBasicBlock);
}

// TND 新增
void PromptFlashAttentionTiling::SetMultiCoreParamsTND()
{
    auto &multiCoreParams = mlaTilingData.PFAmultiCoreParams;
    auto &coreParams = mlaTilingData.PFAcoreParams;
    accumS1BlockNum = 0;
    for (int64_t i = 0; i < bSize; i++) {
        OP_LOGD(contextKeyParamsPtr->opName, "[%s]actualSeqLenData data %ld is %ld.", "PFA_TND", i, actualSeqLenData[i]);
        OP_LOGD(contextKeyParamsPtr->opName, "[%s]actualSeqLenKvData data %ld is %ld.", "PFA_TND", i, actualSeqLenKvData[i]);
        accumS1BlockNum += CeilDivision(actualSeqLenData[i], s1BasicBlock);
    }
    int64_t totalSize = accumS1BlockNum * coreParams.get_n2OuterSize() * coreParams.get_gOuterSize();
    int64_t actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));
    int64_t actualUsedAicNum = std::min(totalSize, static_cast<int64_t>(aicNum));
    int64_t actualUsedAiCoreNum = isSameAB ? actualUsedAicNum * 2 : actualUsedAivNum;
    int64_t actualSplitAiCoreNum = isSameAB ? actualUsedAicNum : actualUsedAivNum;
    multiCoreParams.set_coreNum(static_cast<int32_t>(actualUsedAiCoreNum));
    multiCoreParams.set_totalSize(totalSize);
    multiCoreParams.set_splitFactorSize(CeilDivision(totalSize, actualSplitAiCoreNum));
    multiCoreParams.set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParams.get_splitFactorSize()));
}

bool PromptFlashAttentionTiling::InputLayoutIsTNDLike() const
{
    return inputLayout == InputLayout::TND || inputLayout == InputLayout::NTD_TND;
}

int64_t PromptFlashAttentionTiling::GetTFromInputShape(uint32_t inputIdx, const gert::StorageShape *shape) const
{
    auto actualSeqLenTensor = contextKeyParamsPtr->actualSequenceLengthQ;
    bool isKV = (inputIdx != QUERY_INDEX && inputIdx != QUERY_ROPE_INDEX);
    if (isKV) {
        actualSeqLenTensor = contextKeyParamsPtr->actualSequenceLengthKV;
    }
    auto &actualSeqLenShape = actualSeqLenTensor->GetShape().GetStorageShape();
    const int64_t *value = actualSeqLenTensor->GetData<int64_t>();
    if (enablePA && isKV) {
        int64_t tempT = 0;
        for (int64_t i = 0; i < actualSeqLenShape.GetDim(0); ++i) {
            tempT += value[i];
        }
        return tempT;
    }

    if (inputLayout == InputLayout::TND) {
        return shape->GetStorageShape().GetDim(FIRST_DIM);
    } else {
        return shape->GetStorageShape().GetDim(SECOND_DIM);
    }
}

int64_t PromptFlashAttentionTiling::GetNFromInputShape(uint32_t inputIdx, const gert::StorageShape *shape) const
{
    bool isKV = (inputIdx != QUERY_INDEX && inputIdx != QUERY_ROPE_INDEX);
    auto nKV = contextKeyParamsPtr->numKeyValueHeads == nullptr ? *contextKeyParamsPtr->headsNumber : *contextKeyParamsPtr->numKeyValueHeads;
    if (enablePA && (isKV)) {
        if (inputKvLayout == InputLayout::BBH) {
            return static_cast<int64_t>(nKV);
        } else {
            return shape->GetStorageShape().GetDim(SECOND_DIM);
        }
    }

    if (inputLayout == InputLayout::TND) {
        return shape->GetStorageShape().GetDim(SECOND_DIM);
    } else {
        return shape->GetStorageShape().GetDim(FIRST_DIM);
    }
}

int64_t PromptFlashAttentionTiling::GetDFromInputShape(uint32_t inputIdx, const gert::StorageShape *shape) const
{
    bool isKV = (inputIdx != QUERY_INDEX && inputIdx != QUERY_ROPE_INDEX);
    auto nKV = contextKeyParamsPtr->numKeyValueHeads == nullptr ? *contextKeyParamsPtr->headsNumber : *contextKeyParamsPtr->numKeyValueHeads;
    if (enablePA && isKV) {
        if (inputKvLayout == InputLayout::NZ) {
            return shape->GetStorageShape().GetDim(FIFTH_DIM) * shape->GetStorageShape().GetDim(THIRD_DIM);
        } else if (inputKvLayout == InputLayout::BBH) {
            return shape->GetStorageShape().GetDim(THIRD_DIM) / nKV;
        } else if (inputKvLayout == InputLayout::BNBD) {
            return shape->GetStorageShape().GetDim(FOURTH_DIM);
        }
    }
    return shape->GetStorageShape().GetDim(THIRD_DIM);
}

int64_t PromptFlashAttentionTiling::GetTFromOutputShape(const gert::StorageShape *shape) const
{
    return shape->GetStorageShape().GetDim(0);
}

int64_t PromptFlashAttentionTiling::GetNFromOutputShape(const gert::StorageShape *shape) const
{
    return shape->GetStorageShape().GetDim(1);
}

void PromptFlashAttentionTiling::GetActualSeqLenData(int64_t inputIdx,
    std::array<int64_t, MAX_VAR_LEN_SEQ_LEN> &res, int64_t &actualLen) const
{
    auto actualSeqLenTensor = contextKeyParamsPtr->actualSequenceLengthQ;
    if (inputIdx == ACTUAL_SEQ_KV_INDEX) {
        actualSeqLenTensor = contextKeyParamsPtr->actualSequenceLengthKV;
    }
    if (actualSeqLenTensor == nullptr) {
        if (inputIdx == ACTUAL_SEQ_KV_INDEX) {
            OP_LOGE(contextKeyParamsPtr->opName, "[%s]the key/value's actual sequence lengths can not be null pointer", "PFA_TND");
        } else {
            OP_LOGE(contextKeyParamsPtr->opName, "[%s]the query's actual sequence lengths can not be null pointer", "PFA_TND");
        }
        return;
    }
    auto &actualSeqLenShape = actualSeqLenTensor->GetShape().GetStorageShape();
    if (actualSeqLenShape.GetDimNum() != 1) {
        OP_LOGW(contextKeyParamsPtr->opName, "[%s]actualSeqLenShape is invalid %lu %ld", "PFA_TND", actualSeqLenShape.GetDimNum(),
            actualSeqLenShape.GetDim(0));
        return;
    }
    /* Get Data from tensor. */
    const int64_t *value = actualSeqLenTensor->GetData<int64_t>();
    if (value == nullptr) {
        OP_LOGE(contextKeyParamsPtr->opName, "[%s]actualSeqLenTensor data is null pointer", "PFA_TND");
        return;
    }
    res[0] = value[0];
    actualLen++;
    if (enablePA && inputIdx == ACTUAL_SEQ_KV_INDEX) {
        for (int64_t i = 1; i < actualSeqLenShape.GetDim(0); ++i) {
            auto qLen = value[i];
            res[i] = qLen < 0 ? 0 : qLen;
            actualLen++;
        }
    } else {
        for (int64_t i = 1; i < actualSeqLenShape.GetDim(0); ++i) {
            auto qLen = value[i] - value[i - 1];
            res[i] = qLen < 0 ? 0 : qLen;
            actualLen++;
        }
    }
}

bool PromptFlashAttentionTiling::BalanceLoad(const std::vector<int64_t> &sparseValidArray,
    PFAMultiCoreParams &multiCoreParams, std::vector<int64_t> &localValue,
    std::vector<int64_t> &sparseStartIdx) const
{
    // to avoid buffer overflow, or maybe sometimes we want to only verify single core
    int64_t validAiCoreNum = isSameAB ? std::min(static_cast<int64_t>(multiCoreParams.get_coreNum() / 2), MAX_AIC_NUM)
        :std::min(static_cast<int64_t>(multiCoreParams.get_coreNum()), MAX_AIV_NUM);
    int64_t totalSize = multiCoreParams.get_totalSize();
    int64_t maxVal = *std::max_element(localValue.begin(), localValue.end());
    int64_t tmpMaxVal = maxVal;

    // 从前往后遍历
    for (int64_t idx = 1; idx < validAiCoreNum; ++idx) {
        int64_t start = sparseStartIdx[idx];
        if (start < totalSize && start > 0 && ((localValue[idx - 1] + sparseValidArray[start]) < maxVal)) {
            localValue[idx - 1] += sparseValidArray[start];
            localValue[idx] -= sparseValidArray[start];
            sparseStartIdx[idx] += 1;
        } else if (start == totalSize) {
            break;
        }
    }
    tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

    // 从后往前遍历
    for (int64_t idx = validAiCoreNum - 1; idx > 0; --idx) {
        int64_t start = sparseStartIdx[idx];
        if (start == totalSize) {
            if (sparseStartIdx[idx - 1] == totalSize) {
                continue;
            }
            localValue[idx - 1] -= sparseValidArray[start - 1];
            localValue[idx] = sparseValidArray[start - 1];
            sparseStartIdx[idx] -= 1;
        } else if (start > 0) {
            if ((localValue[idx] + sparseValidArray[start - 1]) >= tmpMaxVal) {
                continue;
            }
            localValue[idx - 1] -= sparseValidArray[start - 1];
            localValue[idx] += sparseValidArray[start - 1];
            sparseStartIdx[idx] -= 1;
        } else {
            break;
        }
    }
    tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

    return tmpMaxVal < maxVal;
}

void PromptFlashAttentionTiling::InitLoadValue(const std::vector<int64_t> &sparseValidArray, int64_t validAivNum,
    int64_t totalSize, const std::vector<int64_t> &sparseStartIdx,
    std::vector<int64_t> &localValue) const
{
    for (int64_t idx = 0; idx < validAivNum; ++idx) {
        int64_t start = sparseStartIdx[idx];
        int64_t end = ((idx + 1) < validAivNum) ? sparseStartIdx[idx + 1] : totalSize;
        if (start < totalSize) {
            localValue[idx] =
                std::accumulate(sparseValidArray.begin() + start, sparseValidArray.begin() + end, 0LL);
        } else {
            break;
        }
    }
}

bool PromptFlashAttentionTiling::SetSparseStartIdxTND(const std::vector<int64_t> &sparseValidArray, PFAMultiCoreParams &multiCoreParams) const
{
    // to avoid buffer overflow, or maybe sometimes we want to only verify single core
    int64_t validAiCoreNum = isSameAB ? std::min(static_cast<int64_t>(multiCoreParams.get_coreNum() / 2), MAX_AIC_NUM)
        :std::min(static_cast<int64_t>(multiCoreParams.get_coreNum()), MAX_AIV_NUM);
    int64_t totalSize = multiCoreParams.get_totalSize(); // BN2GS1.o
    int64_t *sparseStartIdx = multiCoreParams.get_sparseStartIdx();
    int64_t maxAiCoreNum = isSameAB ? MAX_AIC_NUM : MAX_AIV_NUM;
    OP_CHECK_IF(totalSize <= 0, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParamsPtr->opName, "totalSize should be larger than 0."), return false);

    // initLoad: 使用均分策略, 保证后续不会比均分差
    int64_t splitFactorSize = multiCoreParams.get_splitFactorSize();
    std::vector<int64_t> localSparseStartIdx(maxAiCoreNum, totalSize);
    for (int64_t idx = 0; idx < maxAiCoreNum; ++idx) {
        localSparseStartIdx[idx] = std::min((idx * splitFactorSize), totalSize);
    }
    std::vector<int64_t> localValue(validAiCoreNum, 0);
    InitLoadValue(sparseValidArray, validAiCoreNum, totalSize, localSparseStartIdx, localValue);

    // 负载均衡粗调
    std::vector<int64_t> tmpLocalValue(validAiCoreNum, 0);
    std::vector<int64_t> tmpsparseStartIdx(maxAiCoreNum, totalSize);
    int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0LL);
    int64_t avgVal = CeilDivision(sparseArraySum, validAiCoreNum);

    tmpsparseStartIdx[0] = 0;
    for (int64_t idx = 1; idx < maxAiCoreNum; ++idx) {
        int64_t start = tmpsparseStartIdx[idx - 1];
        int64_t singleLoadValue = 0;
        tmpsparseStartIdx[idx] = start;
        while (singleLoadValue < avgVal && tmpsparseStartIdx[idx] < totalSize) {
            singleLoadValue += sparseValidArray[tmpsparseStartIdx[idx]];
            tmpsparseStartIdx[idx] += 1;
        }

        if ((start + 1) < tmpsparseStartIdx[idx]) {
            int64_t redoSingleLoadValue = singleLoadValue - sparseValidArray[tmpsparseStartIdx[idx] - 1];
            bool loadValueFlag = (singleLoadValue - avgVal) > (avgVal - redoSingleLoadValue);
            tmpsparseStartIdx[idx] = loadValueFlag ? (tmpsparseStartIdx[idx] - 1) : (tmpsparseStartIdx[idx]);
            singleLoadValue = loadValueFlag ? redoSingleLoadValue : singleLoadValue;
            sparseArraySum -= singleLoadValue;
            avgVal = CeilDivision(sparseArraySum, (validAiCoreNum - idx));
        }
    }

    InitLoadValue(sparseValidArray, validAiCoreNum, totalSize, tmpsparseStartIdx, tmpLocalValue);

    // 负载均衡精调
    while (BalanceLoad(sparseValidArray, multiCoreParams, tmpLocalValue, tmpsparseStartIdx)) {
        // 根据负载均衡是否能得到更好预测结果决定是否结束循环
    }

    // exchange initLoad and 负载均衡
    if ((*std::max_element(localValue.begin(), localValue.end())) >
        (*std::max_element(tmpLocalValue.begin(), tmpLocalValue.end()))) {
        localSparseStartIdx.swap(tmpsparseStartIdx);
        localValue.swap(tmpLocalValue);
    }
    for (int64_t idx = 0; idx < maxAiCoreNum; ++idx) {
        sparseStartIdx[idx] = localSparseStartIdx[idx];
    }
    return true;
}

int64_t PromptFlashAttentionTiling::GetS2RealSize(uint8_t sparseType, int32_t bOutIdx, int64_t s1OutIdx)
{
    int64_t s2RealSize = s2Size;
    int64_t actualS1Len = actualSeqLenData[bOutIdx];
    int64_t actualS2Len = actualSeqLenKvData[bOutIdx];

    if (sparseType == static_cast<uint8_t>(SparseTypeEnum::CAUSAL)) {
        s2RealSize = s1BasicBlock * (s1OutIdx + 1);
    } else if (sparseType == static_cast<uint8_t>(SparseTypeEnum::RIGHT_DOWN_CAUSAL)) {
        s2RealSize = s1BasicBlock * (s1OutIdx + 1) + actualS2Len - actualS1Len;
        s2RealSize = std::max(s2RealSize, static_cast<int64_t>(0));
    }
    return std::min(s2RealSize, actualSeqLenKvData[bOutIdx]);
}

void PromptFlashAttentionTiling::InitSparseValidArrayTND(std::vector<int64_t> &sparseValidArray)
{
    uint8_t sparseType = mlaTilingData.PFAinputParams.get_sparseType();
    auto &coreParams = mlaTilingData.PFAcoreParams;
    for (int32_t i = 0; i < bSize; i++) {
        int64_t n2G = coreParams.get_n2OuterSize() * coreParams.get_gOuterSize();
        int64_t s1BlockNum = CeilDivision(actualSeqLenData[i], s1BasicBlock);
        // 每个s1方向上切分块的计算量
        for (int64_t k = 0; k < n2G; ++k) {
            for (int64_t j = 0; j < s1BlockNum; ++j) {
                // 此处暂时设置为1, 由于实测尾块1和128性能差距不大，理论上应该如下所示
                // 理论值: s1RealSize为std::min(s1BasicBlock, (actualSeqLenData[i] - s1BasicBlock * j))
                int64_t s1RealSize = 1;
                int64_t s2RealSize = GetS2RealSize(sparseType, i, j);
                // 新增一个系数, 解决理论和实际的差异
                int64_t s2RemainSize = s2RealSize % s2sizeLimitMax;
                s2RealSize = (s2RealSize / s2sizeLimitMax) * s2sizeLimitMax;
                s2RealSize += ((s2RemainSize > 0) ? COF[CeilDivision(s2RemainSize, 128L) - 1] : 0);
                sparseValidArray.emplace_back(s1RealSize * s2RealSize);
            }
        }
    }
}

void PromptFlashAttentionTiling::SetSparseParamsTND()
{
    auto &coreParams = mlaTilingData.PFAcoreParams;
    auto &multiCoreParams = mlaTilingData.PFAmultiCoreParams;
    std::vector<int64_t> sparseValidArray;
    sparseValidArray.reserve(multiCoreParams.get_totalSize());
    InitSparseValidArrayTND(sparseValidArray);
    SetSparseStartIdxTND(sparseValidArray, multiCoreParams);

    coreParams.set_s1SparseValidSize(s1SparseValidSize);
    coreParams.set_s2SparseValidSize(s2SparseValidSize);
}

uint32_t PromptFlashAttentionTiling::CalcTschNumBlocks(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum) const
{
    uint32_t ration;
    if (aicCoreNum == 0 || aivCoreNum == 0 || aicCoreNum > aivCoreNum) {
        return sliceNum;
    }
    ration = aivCoreNum / aicCoreNum;
    return (sliceNum + (ration - 1U)) / ration;
}

void PromptFlashAttentionTiling::CalcUBSize()
{
    apiMaxUBSize = HIGH_PERF_API_BUFFER_MULTIPLE * s1BasicBlock * s2BasicBlock * sizeof(float);
}

bool PromptFlashAttentionTiling::SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
    matmul_tiling::MatmulApiTiling &bmm1)
{
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16, true);
    bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    // 分不满核，且稀疏场景，shape设置的较小能产生更好的tiling
    bmm1.SetShape(std::min(tmpS1BasicBlock, s1Size),
        std::min(tmpS2BasicBlock * mlaTilingData.PFAcoreParams.get_nRatio(), s2Size), dSize);
    if (InputLayoutIsTNDLike()) {
        bmm1.SetOrgShape(s1Size, tmpS2BasicBlock * mlaTilingData.PFAcoreParams.get_nRatio(), s1StrideSize, s2StrideSize);
        bmm1.SetBias(false);
        if (bmm1.SetBufferSpace(ascendPlatformInfo.l1Size, ascendPlatformInfo.l0CSize) != 0) {
            return false;
        }
        return true;
    } else {
        bmm1.SetOrgShape(s1Size, tmpS2BasicBlock * mlaTilingData.PFAcoreParams.get_nRatio(), dSize, dSize);
        bmm1.SetBias(false);
        if (bmm1.SetBufferSpace(ascendPlatformInfo.l1Size, ascendPlatformInfo.l0CSize) != 0) {
            return false;
        }

        int64_t baseM = std::min(BMM_BASICBLOCK_M_128, AlignUp(s1Size, FRACTAL_NUM));
        int64_t baseN = std::min(BMM_BASICBLOCK_N_128, AlignUp(s2Size, FRACTAL_NUM));
        bmm1.SetFixSplit(baseM, baseN, alignedD);

        return true;
    }
}

bool PromptFlashAttentionTiling::SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
    matmul_tiling::MatmulApiTiling &bmm2)
{
    int64_t singleM = std::min(tmpS1BasicBlock, s1Size);
    bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16, false);
    bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16, false);
    bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    if (InputLayoutIsTNDLike()) {
        bmm2.SetShape(std::min(tmpS1BasicBlock, s1Size), dSize,
            std::min(tmpS2BasicBlock * mlaTilingData.PFAcoreParams.get_nRatio(), s2Size));
        bmm2.SetOrgShape(s1Size, s2StrideSize, std::min(tmpS2BasicBlock * mlaTilingData.PFAcoreParams.get_nRatio(), s2Size),
            s2StrideSize);
        bmm2.SetBias(false);
        if (bmm2.SetBufferSpace(ascendPlatformInfo.l1Size, ascendPlatformInfo.l0CSize) != 0) {
            return false;
        }
        return true;
    } else {
        bmm2.SetShape(singleM, valueDSize,
            std::min(tmpS2BasicBlock * mlaTilingData.PFAcoreParams.get_nRatio(), s2Size));
        bmm2.SetOrgShape(s1Size, valueDSize, std::min(tmpS2BasicBlock * mlaTilingData.PFAcoreParams.get_nRatio(), s2Size),
            valueDSize);
        bmm2.SetBias(false);
        if (bmm2.SetBufferSpace(ascendPlatformInfo.l1Size, ascendPlatformInfo.l0CSize) != 0) {
            return false;
        }
        if (valueDSize > BMM_BLOCK_SIZE_THRESHOLD_LOWER_64_HOST_TILING && valueDSize <= BMM_BLOCK_SIZE_THRESHOLD_UPPER_128_HOST_TILING) {
            int64_t baseM = std::min(BMM_BASICBLOCK_M_128, AlignUp(s1Size, FRACTAL_NUM));
            int64_t baseN = std::min(BMM_BASICBLOCK_N_128, AlignUp(valueDSize, FRACTAL_NUM));
            bmm2.SetFixSplit(baseM, baseN);
        }
        int64_t baseM = std::min(BMM_BASICBLOCK_M_128, AlignUp(s1Size, FRACTAL_NUM));
        int64_t baseN = AlignUp(valueDSize, FRACTAL_NUM);
        bmm2.SetFixSplit(baseM, baseN);
        return true;
    }
}

bool PromptFlashAttentionTiling::SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
    matmul_tiling::MatmulApiTiling &bmm1,
    matmul_tiling::MatmulApiTiling &bmm2)
{
    if (!SetBmm1TilingInput(tmpS1BasicBlock, tmpS2BasicBlock, bmm1) ||
        !SetBmm2TilingInput(tmpS1BasicBlock, tmpS2BasicBlock, bmm2)) {
        return false;
    }

    if (bmm1.GetTiling(mlaTilingData.bmm1TilingData) == -1) {
        return false;
    }

    if (mlaTilingData.bmm1TilingData.get_baseK() >= mlaTilingData.bmm1TilingData.get_singleCoreK() &&
        mlaTilingData.bmm1TilingData.get_depthA1() == BMM1_DEPTH_A1_2) {
        mlaTilingData.bmm1TilingData.set_depthA1(BMM1_DEPTH_A1_3);
    }

    // 当D > 128，由于L0 DB开启的限制，在设置tiling时把baseK切小，在MatmulPolicy中仍然按照不切K搬运
    if (!InputLayoutIsTNDLike()) {
        mlaTilingData.bmm1TilingData.set_baseK(AlignUp(dSize / NUM_2, FRACTAL_NUM));
    }

    mlaTilingData.bmm1TilingData.set_shareMode(0);
    mlaTilingData.bmm1TilingData.set_shareL1Size(ascendPlatformInfo.l1Size);
    mlaTilingData.bmm1TilingData.set_shareL0CSize(ascendPlatformInfo.l0CSize);

    if (bmm2.GetTiling(mlaTilingData.bmm2TilingData) == -1) {
        return false;
    }

    mlaTilingData.bmm2TilingData.set_shareMode(0);
    mlaTilingData.bmm2TilingData.set_shareL1Size(ascendPlatformInfo.l1Size);
    mlaTilingData.bmm2TilingData.set_shareL0CSize(ascendPlatformInfo.l0CSize);
    return true;
}

bool PromptFlashAttentionTiling::SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock) {
    matmul_tiling::MatmulApiTiling bmm1(ascendPlatformInfo);
    matmul_tiling::MatmulApiTiling bmm2(ascendPlatformInfo);
    return SetMatMulTiling(tmpS1BasicBlock, tmpS2BasicBlock, bmm1, bmm2);
}

int64_t PromptFlashAttentionTiling::CalcMaxS1BasicBlockSize(int64_t actualD, const BufferNum &bufferNum)
{
    // if S2 basic block is min value 16, s1 basic block can reach max value, then we get:
    // s1 * 16 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp =>
    // s1 * (16 * X + D * Y + (expNum + 2) * (32 / sizeof(T))) * sizeof(T) + apiTmp
    // just ignore apiTmp now, consider it at last
    int64_t alignUnit = BYTE_BLOCK / dataTypeSize;
    int64_t maxS1BasicBlock = ascendPlatformInfo.ubSize / dataTypeSize /
        (FRACTAL_NUM * bufferNum.bufferS1S2Num + actualD * bufferNum.bufferS1DNum +
         (bufferNum.bufferExpNum + TWO_HOST_TILING) * alignUnit); // here 2 means FlashSoftMax sum and max output
    return AlignDown(maxS1BasicBlock, FRACTAL_NUM);
}

int64_t PromptFlashAttentionTiling::CalcMaxS2BasicBlockSize(const BufferNum &bufferNum,
    int64_t tmpS1BasicBlock)
{
    // used UB: s1s2 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp
    // if D full load, use alignedD in above formula
    // if D not full load, use S2 basic block var in above formula
    // just ignore apiTmp now, consider it at last
    int64_t tmpS2BasicBlock = (ascendPlatformInfo.ubSize - tmpS1BasicBlock * (bufferNum.bufferExpNum + 2) * BYTE_BLOCK -
        tmpS1BasicBlock * alignedD * bufferNum.bufferS1DNum * dataTypeSize) /
        (tmpS1BasicBlock * bufferNum.bufferS1S2Num * dataTypeSize);
    return std::min(AlignDown(tmpS2BasicBlock, FRACTAL_NUM), alignedS2);
}

bool PromptFlashAttentionTiling::IsBasicBlockInSoftMax(const ge::Shape &shape) const
{
    // 2 axes at least
    if (shape.GetDimNum() < 2) {
        return false;
    }

    int64_t lastAxis = shape.GetDim(shape.GetDimNum() - 1);
    // last axis should be less than 2048 and fullfil 64 times
    int64_t basicLastAxis = 64;
    int64_t lastAxisNum = 2048;
    if (lastAxis > lastAxisNum || lastAxis % basicLastAxis != 0) {
        return false;
    }

    int64_t preAxes = 1;
    for (size_t idx = 0; idx < shape.GetDimNum() - 1; ++idx) {
        preAxes *= shape.GetDim(idx);
    }

    // all axes except last one should be 8 times
    return preAxes % 8 == 0;
}

void PromptFlashAttentionTiling::GetBufferNum(BufferNum &bufferNum) const
{
    bufferNum.bufferS1S2Num = HIGH_PERF_BUFFER_NUM;
}

void PromptFlashAttentionTiling::MatchTemplate(uint32_t valueD)
{
    // UB Size calc logic: s1s2 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp
    BufferNum bufferNum;
    GetBufferNum(bufferNum);

    s1BasicBlock = std::numeric_limits<int64_t>::max();
    s2BasicBlock = std::numeric_limits<int64_t>::max();
    s1BasicBlockBest = (InputLayoutIsTNDLike() && (valueD <= D_SIZE_128)) ? 512L : 256L; // TND基本块设为(512, 512)
    s1BasicBlock = std::min(s1BasicBlockBest, alignedS1);
    s2BasicBlock = std::min(128L, alignedS2);
    s1VecBasicBlock = s1BasicBlock / AIV_AIC_NUM_RATIO;
    nRatio = (InputLayoutIsTNDLike() && (valueD <= D_SIZE_128)) ? 4L : 8L; // TND基本块设为(512, 128*4)
    dBasicBlock = std::min(128L, alignedD);
    CalcUBSize();
}

ge::graphStatus PromptFlashAttentionTiling::CheckLearnableSinkWhenLayoutIsTND(ContextParamsForPFATiling& contextKeyParams) const {
    const gert::StorageShape* queryShape = contextKeyParams.queryInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    int64_t queryN = GetNFromInputShape(QUERY_INDEX, queryShape);
    int64_t valueD = GetDFromInputShape(VALUE_INDEX, valueShape);

    if (contextKeyParams.hasLearnableSink == false) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::StorageShape* learnableSinkShape = contextKeyParams.learnableSinkShape;
    OP_CHECK_IF(learnableSinkShape->GetStorageShape().GetDimNum() != DIM_NUM_1,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "learnable_sink enable, learnable_sink dim(%zu) must be 1!", learnableSinkShape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);

    int64_t sinkN = learnableSinkShape->GetStorageShape().GetDim(FIRST_DIM);
    OP_CHECK_IF(sinkN != queryN,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "learnable_sink enable, sinkN(%ld) must be same equal queryN(%ld)!", sinkN, queryN),
        return ge::GRAPH_FAILED);
    // learnable sink场景，只支持valueD<=128的baseapi模板
    OP_CHECK_IF(valueD != D_SIZE_128 && valueD != D_SIZE_64,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When Layout is TND, learnable_sink only support valueD equal 64/128."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::CheckInputShapeWhenLayoutIsTND(ContextParamsForPFATiling& contextKeyParams) const {
    std::string layoutStr(contextKeyParams.layout);
    const gert::Tensor* actSeqLenData = contextKeyParams.actualSequenceLengthQ;
    const gert::Tensor* actSeqLenDataKV = contextKeyParams.actualSequenceLengthKV;
    OP_CHECK_IF((actSeqLenData == nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is %s, actualSequenceLengthQ is required, but now is nullptr!", layoutStr.c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((actSeqLenDataKV == nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is %s, actualSequenceLengthKV is required, but now is nullptr!", layoutStr.c_str()),
        return ge::GRAPH_FAILED);
    int64_t actSeqLenDims = actSeqLenData->GetShapeSize();
    int64_t actSeqLenKVDims = actSeqLenDataKV->GetShapeSize();
    // PFA TND tiling下沉和acl graph当前不支持
    OP_CHECK_IF(((actSeqLenData->GetData<int64_t>() == nullptr) || (actSeqLenDataKV->GetData<int64_t>() == nullptr)),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is %s, not support tiling_schedule_optimize = True or config mode is reduce-overhead!", layoutStr.c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((actSeqLenDims == 0),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is %s, actualSequenceLengthQ is required, but the number of element in it is 0!", layoutStr.c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((actSeqLenKVDims == 0),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is %s, actualSequenceLengthKV is required, but the number of element in it is 0!", layoutStr.c_str()),
        return ge::GRAPH_FAILED);
    int64_t lastSeqLen = static_cast<int64_t>(actSeqLenData->GetData<int64_t>()[actSeqLenDims - 1]);
    int64_t lastSeqLenKV = static_cast<int64_t>(actSeqLenDataKV->GetData<int64_t>()[actSeqLenKVDims - 1]);

    const gert::StorageShape* queryShape = contextKeyParams.queryInputShape;
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    const gert::StorageShape* queryRopeShape = contextKeyParams.queryRope;
    const gert::StorageShape* keyRopeShape = contextKeyParams.keyRope;
    int64_t queryT = GetTFromInputShape(QUERY_INDEX, queryShape);
    int64_t keyT = GetTFromInputShape(KEY_INDEX, keyShape);
    int64_t valueT = GetTFromInputShape(VALUE_INDEX, valueShape);

    if (enablePA && (inputKvLayout == InputLayout::NZ || inputKvLayout == InputLayout::BNBD || inputKvLayout == InputLayout::BBH)) {
        OP_CHECK_IF((queryShape->GetStorageShape().GetDimNum() != DIM_NUM_3 || keyShape->GetStorageShape().GetDimNum() != valueShape->GetStorageShape().GetDimNum()),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is %s, querDim(%zu) must be 3, keyDim(%zu) valueDim(%zu) must be equal.", layoutStr.c_str(),
            queryShape->GetStorageShape().GetDimNum(), keyShape->GetStorageShape().GetDimNum(), valueShape->GetStorageShape().GetDimNum()),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((queryShape->GetStorageShape().GetDimNum() != DIM_NUM_3 || keyShape->GetStorageShape().GetDimNum() != DIM_NUM_3 ||
            valueShape->GetStorageShape().GetDimNum() != DIM_NUM_3),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is %s, querDim(%zu) keyDim(%zu) valueDim(%zu) must be 3.", layoutStr.c_str(),
            queryShape->GetStorageShape().GetDimNum(), keyShape->GetStorageShape().GetDimNum(), valueShape->GetStorageShape().GetDimNum()),
            return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(queryT != lastSeqLen,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is %s, queryT(%ld) must be equal to the last element of actualSequenceLengthQ(%ld)", layoutStr.c_str(), queryT, lastSeqLen),
        return ge::GRAPH_FAILED);
    if (!enablePA) {
        OP_CHECK_IF((keyT != lastSeqLenKV) || (valueT != lastSeqLenKV),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is %s and PA not enabled, keyT(%ld) and valueT(%ld) must be equal to the last element of actualSeqenceLengthKV(%ld)", layoutStr.c_str(), keyT, valueT, lastSeqLenKV),
            return ge::GRAPH_FAILED);
    }

    int64_t queryN = GetNFromInputShape(QUERY_INDEX, queryShape);
    int64_t keyN = GetNFromInputShape(KEY_INDEX, keyShape);
    int64_t valueN = GetNFromInputShape(VALUE_INDEX, valueShape);

    OP_CHECK_IF((keyN != valueN),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is %s, the values of keyN(%ld), valueN(%ld) must be equal", layoutStr.c_str(), keyN, valueN),
        return ge::GRAPH_FAILED);
    // already ensure not nullptr
    OP_CHECK_IF((queryN != *(contextKeyParams.headsNumber)),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is %s, the values of queryN(%ld) not equal to Attr the query's heads num(%d)", layoutStr.c_str(), queryN, *(contextKeyParams.headsNumber)),
        return ge::GRAPH_FAILED);
    // GQA场景 G是整数判断
    OP_CHECK_IF(keyN == 0, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "keyN is zero."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((queryN % keyN != 0),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is %s, the values of queryN(%ld) should be a multiple of keyN(%ld).", layoutStr.c_str(), queryN, keyN),
        return ge::GRAPH_FAILED);
    int64_t queryD = GetDFromInputShape(QUERY_INDEX, queryShape);
    int64_t keyD = GetDFromInputShape(KEY_INDEX, keyShape);
    int64_t valueD = GetDFromInputShape(VALUE_INDEX, valueShape);

    if ((queryRopeShape != nullptr) || (keyRopeShape != nullptr)) {
        OP_CHECK_IF((queryRopeShape == nullptr),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is %s, queryRope is null, but keyRope exists, they should be both null or exist!", layoutStr.c_str()),
                return ge::GRAPH_FAILED);
        OP_CHECK_IF((keyRopeShape == nullptr),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is %s, keyRope is null, but queryRope exists, they should be both null or exist!", layoutStr.c_str()),
                return ge::GRAPH_FAILED);
        OP_CHECK_IF((queryRopeShape->GetStorageShape().GetDimNum() != queryShape->GetStorageShape().GetDimNum() ||
                keyRopeShape->GetStorageShape().GetDimNum() != keyShape->GetStorageShape().GetDimNum()),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "When layout is %s, queryRope/keyRope is not nullptr, queryRopeDims(%zu)/keyRopeDims(%zu) must be equal to queryDims(%zu)/keyDims(%zu)!", layoutStr.c_str(),
                queryRopeShape->GetStorageShape().GetDimNum(), keyRopeShape->GetStorageShape().GetDimNum(), queryShape->GetStorageShape().GetDimNum(), keyShape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
        int64_t queryRopeT = GetTFromInputShape(QUERY_ROPE_INDEX, queryRopeShape);
        int64_t keyRopeT = GetTFromInputShape(KEY_ROPE_INDEX, keyRopeShape);
        int64_t queryRopeN = GetNFromInputShape(QUERY_ROPE_INDEX, queryRopeShape);
        int64_t keyRopeN = GetNFromInputShape(KEY_ROPE_INDEX, keyRopeShape);
        OP_CHECK_IF((queryRopeT != queryT) || (queryRopeN != queryN),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "When layout is %s and queryRope is not nullptr, queryRopeT(%ld)/queryRopeN(%ld) must be equal to queryT(%ld)/queryN(%ld)!",
                layoutStr.c_str(), queryRopeT, queryRopeN, queryT, queryN),
                return ge::GRAPH_FAILED);
        if (!enablePA) {
            OP_CHECK_IF((keyRopeT != keyT),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "When layout is %s and keyRope is not nullptr, keyRopeT(%ld) must be equal to keyT(%ld)!",
                    layoutStr.c_str(), keyRopeT, keyT),
                    return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF((keyRopeN != keyN),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "When layout is %s and keyRope is not nullptr, keyRopeN(%ld) must be equal to keyN(%ld)!",
                layoutStr.c_str(), keyRopeN, keyN),
                return ge::GRAPH_FAILED);
        int64_t queryRopeD = GetDFromInputShape(QUERY_ROPE_INDEX, queryRopeShape);
        int64_t keyRopeD = GetDFromInputShape(KEY_ROPE_INDEX, keyRopeShape);
        OP_CHECK_IF((queryD != D_SIZE_128) || (keyD != D_SIZE_128) || (valueD != D_SIZE_128) || (queryRopeD != D_SIZE_64) || (keyRopeD != D_SIZE_64),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When layout is %s and queryRope/keyRope is not nullptr, queryRopeD(%ld)/keyRopeD(%ld) must be 64, queryD(%ld)/keyD(%ld)/valueD(%ld) must be 128!", layoutStr.c_str(),
            queryRopeD, keyRopeD, queryD, keyD, valueD), return ge::GRAPH_FAILED);
    } else {
        // QKD=192&&VD=192/128 || QKV D等长且等于128
        OP_CHECK_IF(!(((queryD == D_SIZE_192) && (keyD == D_SIZE_192) && ((valueD == D_SIZE_192) || (valueD == D_SIZE_128))) || ((queryD == keyD) && (queryD == valueD) && (valueD == D_SIZE_128 || valueD == D_SIZE_64))),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is %s, queryD(%ld), keyD(%ld) and valueD(%ld) must be same equal 192/128/64, or queryD and keyD equal 192 and valueD equal 128.",
            layoutStr.c_str(), queryD, keyD, valueD), return ge::GRAPH_FAILED);
    }
    // G不等于1时，VD不能等于192
    OP_CHECK_IF((queryN / keyN != 1) && (valueD == D_SIZE_192),
               OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is %s, When G(%ld) not 1, valueD(%ld) should not be 192.", layoutStr.c_str(), queryN / keyN, valueD),
               return ge::GRAPH_FAILED);

    if (CheckLearnableSinkWhenLayoutIsTND(contextKeyParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(contextKeyParams.inputDataType != ge::DT_BF16,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is %s, inputDataType should be bf16", layoutStr.c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputLayout == InputLayout::NTD_TND && valueD == D_SIZE_192,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout is NTD_TND, valueD(%ld) must be 128!", valueD),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::CheckActSeqWhenLayoutIsTND(ContextParamsForPFATiling& contextKeyParams) const {
    std::string layoutStr(contextKeyParams.layout);
    const gert::Tensor* actSeqLen = contextKeyParams.actualSequenceLengthQ;
    const gert::Tensor* actSeqLenKV = contextKeyParams.actualSequenceLengthKV;
    size_t batchOfQuery = actSeqLen->GetShapeSize();
    size_t batchOfKey = actSeqLenKV->GetShapeSize();
    OP_CHECK_IF(batchOfQuery != batchOfKey,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "When layout is %s, the length of actualSequenceLengthQ(%zu) and actualSequenceLengthKV(%zu) must be equal.",
            layoutStr.c_str(), batchOfQuery, batchOfKey),
        return ge::GRAPH_FAILED);
    for (uint32_t i = LOOP_BEGIN_NUM; i < batchOfQuery; ++i) {
        int64_t curActSeq = actSeqLen->GetData<int64_t>()[i];
        int64_t curActSeqKV = actSeqLenKV->GetData<int64_t>()[i];
        OP_CHECK_IF(curActSeq < NUM_0 || curActSeqKV < NUM_0,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "When layout is %s, actualSequenceLengthQ[%u]=%ld and actualSequenceLengthKV[%u]=%ld must >= 0",
                layoutStr.c_str(), i, curActSeq, i, curActSeqKV),
            return ge::GRAPH_FAILED);
    }
    int64_t lastActSeq = 0;
    int64_t lastActSeqKV = 0;
    for (uint32_t i = LOOP_BEGIN_NUM; i < batchOfQuery; ++i) {
        int64_t curActSeq = actSeqLen->GetData<int64_t>()[i];
        int64_t curActSeqKV = actSeqLenKV->GetData<int64_t>()[i];
        OP_CHECK_IF(curActSeq < lastActSeq,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "When layout is %s, the query's actual sequence lengths must be not decreasing, but it's not at %u, actSeqLen[%u]=%ld, actSeqLen[%u]=%ld",
                layoutStr.c_str(), i, i, curActSeq, i - 1, lastActSeq),
            return ge::GRAPH_FAILED);
        if (!enablePA) {
            OP_CHECK_IF(curActSeqKV < lastActSeqKV,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "When layout is %s, the key/value's actual sequence lengths must be not decreasing, but it's not at %u, actSeqLenKV[%u]=%ld, actSeqLenKV[%u]=%ld",
                    layoutStr.c_str(), i, i, curActSeqKV, i - 1, lastActSeqKV),
                return ge::GRAPH_FAILED);
        }
        lastActSeq = curActSeq;
        lastActSeqKV = curActSeqKV;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::CheckVarLenPreNextToken(ContextParamsForPFATiling& contextKeyParams,
    int32_t sparseMode, int64_t sparsePreTokens, int64_t sparseNextTokens)
{
    // Band下不能出现：
    // preToken < -q_s
    // nextToken < -kv_s
    // preToken + nextToken < 0
    if (sparseMode != SPARSE_MODE_BAND) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(sparsePreTokens + sparseNextTokens < 0,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "The sum of preTokens[%ld] and nextTokens[%ld] must be greater than or equal to 0.",
            sparsePreTokens, sparseNextTokens),
        return ge::GRAPH_FAILED);

    for (int64_t i = 0; i < bSize; ++i) {
        if (actualSeqLenData[i] == 0 || actualSeqLenKvData[i] == 0) {
            continue;
        }
        OP_CHECK_IF((sparsePreTokens + actualSeqLenData[i] < 0 || sparseNextTokens + actualSeqLenKvData[i] < 0),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "Batch[%ld], actualSeqLen[%ld] + preTokens[%ld] must be greater than or equal to 0 and actualSeqLenKv[%ld] + nextTokens[%ld] must be greater than or equal to 0",
                i, actualSeqLenData[i], sparsePreTokens, actualSeqLenKvData[i], sparseNextTokens),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::CheckIOType(ContextParamsForPFATiling& contextKeyParams, PromptFlashAttentionTilingData& tilingData, int32_t& outputDataTypeSize) {
    outputType = contextKeyParams.outputDataType;
    inputType = contextKeyParams.inputDataType;
    intputKeyType = contextKeyParams.kDataType;
    intputValueType = contextKeyParams.vDataType;
    std::string tempLayoutStr(contextKeyParams.layout);
    OP_CHECK_IF((tempLayoutStr == "TND") &&
        (inputType == ge::DT_INT8 && intputKeyType == ge::DT_INT8 && intputValueType == ge::DT_INT8),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When Layout is TND, not support QKV dataType is all int8!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(tempLayoutStr == "TND" && outputType == ge::DT_INT8,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When Layout is TND, not support attention out dataType is int8!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempLayoutStr == "TND") &&
        (inputType != ge::DT_INT8 && intputKeyType == ge::DT_INT8 && intputValueType == ge::DT_INT8),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When Layout is TND, not support KV Antiquant!"),
        return ge::GRAPH_FAILED);

    if (inputType == ge::DT_FLOAT16 && contextKeyParams.kDataType == ge::DT_INT8) {
        enableKvAntiquant = true;

        if (contextKeyParams.hasKeyAntiquantScale || contextKeyParams.hasValueAntiquantScale) {
            enableKvAntiquant = false;
        }
    }

    if (contextKeyParams.hasKeyAntiquantScale || contextKeyParams.hasValueAntiquantScale) {
        enableMsd = true;
        tilingData.promptAttentionBaseParams.set_isMsd(1);
    } else {
        enableMsd = false;
        tilingData.promptAttentionBaseParams.set_isMsd(0);
        OP_CHECK_IF(inputType == ge::DT_BF16 && contextKeyParams.kDataType == ge::DT_INT8,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "The key's dequant scale and the value's dequant scale should not be null, when data type of query is bf16 and data type of key/value is int8"),
            return ge::GRAPH_FAILED);
    }

    if (inputType == ge::DT_FLOAT16) {
        dataTypeSize = FLOAT16SIZE;
    } else if (inputType == ge::DT_BF16) {
        dataTypeSize = BFLOAT16SIZE;
    } else if (inputType == ge::DT_INT8) {
        dataTypeSize = INT8SIZE;
    }
    if (outputType == ge::DT_FLOAT16) {
        outputDataTypeSize = FLOAT16SIZE;
    } else if (outputType == ge::DT_BF16) {
        outputDataTypeSize = BFLOAT16SIZE;
    } else if (outputType == ge::DT_INT8) {
        outputDataTypeSize = INT8SIZE;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::CheckDimNums(ContextParamsForPFATiling& contextKeyParams) {
    std::string layoutStr(contextKeyParams.layout);
    const gert::StorageShape* queryShape = contextKeyParams.queryInputShape;
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    const gert::StorageShape* outShape = contextKeyParams.outputShape;
    const uint32_t keyDimNum = keyShape->GetStorageShape().GetDimNum();
    bool isPageAttention = contextKeyParams.blockTable != nullptr;
    if (CheckNonEmptyShapeExceptions(contextKeyParams, queryShape, "query")) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNonEmptyShapeExceptions(contextKeyParams, keyShape, "key")) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNonEmptyShapeExceptions(contextKeyParams, valueShape, "value")) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNonEmptyShapeExceptions(contextKeyParams, outShape, "out")) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(keyDimNum > INPUT_KV_SHAPE_MAX_DIMS,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "Tensor key shape dimNum(%u) is invalid! When antiquant, only support KdimNum in [3, 4, 5]", keyDimNum),
            return ge::GRAPH_FAILED);
    // key value dims must be equal!
    OP_CHECK_IF((keyShape->GetStorageShape().GetDimNum() != valueShape->GetStorageShape().GetDimNum()),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Layout %s keydims(%zu) and valuedims(%zu) must be equal!",
            layoutStr.c_str(), keyShape->GetStorageShape().GetDimNum(),
            valueShape->GetStorageShape().GetDimNum()),
            return ge::GRAPH_FAILED);
    if (enablePA && (inputKvLayout == InputLayout::NZ || inputKvLayout == InputLayout::BBH || inputKvLayout == InputLayout::BNBD)) {
        OP_CHECK_IF((queryShape->GetStorageShape().GetDimNum() != NUM_3 ||
            keyShape->GetStorageShape().GetDimNum() != valueShape->GetStorageShape().GetDimNum()),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "When layout TND/NTD_TND/BSH and kvcache NZ/BBH/BNBD, queryDims(%zu) must be 3, keydims(%zu), "
                "valueDims(%zu) must be equal!", queryShape->GetStorageShape().GetDimNum(),
                keyShape->GetStorageShape().GetDimNum(), valueShape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    } else {
        if (layoutStr == "TND" || layoutStr == "NTD_TND" || layoutStr == "BSH") {
            OP_CHECK_IF((queryShape->GetStorageShape().GetDimNum() != NUM_3 ||
                keyShape->GetStorageShape().GetDimNum() != NUM_3 ||
                valueShape->GetStorageShape().GetDimNum() != NUM_3),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Layout TND/NTD_TND/BSH queryDims(%zu), keydims(%zu), "
                    "valueDims(%zu) must be 3!", queryShape->GetStorageShape().GetDimNum(),
                    keyShape->GetStorageShape().GetDimNum(), valueShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
        }
    }
    // page attention the dimNum of KV-Cache may be 3 or 4!
    if (layoutStr == "BNSD" || layoutStr == "BSND" || layoutStr == "BNSD_BSND") {
        OP_CHECK_IF((queryShape->GetStorageShape().GetDimNum() != NUM_4),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Layout BNSD/BSND/BNSD_BSND queryDims(%zu) must be 4!",
                queryShape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
        if (isPageAttention) {
            uint32_t seqSize = 0U;
            if (layoutStr == "BSND") {
                seqSize = queryShape->GetStorageShape().GetDim(DIM_1);
            } else {
                seqSize = queryShape->GetStorageShape().GetDim(DIM_2);
            }
            OP_CHECK_IF((keyShape->GetStorageShape().GetDimNum() != NUM_3 && keyShape->GetStorageShape().GetDimNum() != NUM_4),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "when s(%u) > 16 and Page attention enable, "
                    "Layout BNSD/BSND/BNSD_BSND keyDims(%zu) valueDims(%zu) must be all 3/4!",
                    seqSize, keyShape->GetStorageShape().GetDimNum(), valueShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF((keyShape->GetStorageShape().GetDimNum() != NUM_4),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "Layout BNSD/BSND/BNSD_BSND keyDims(%zu) valueDims(%zu) must be all 4!",
                    keyShape->GetStorageShape().GetDimNum(), valueShape->GetStorageShape().GetDimNum()),
                    return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::CheckD(ContextParamsForPFATiling& contextKeyParams) const {
    std::string layoutStr(contextKeyParams.layout);
    if (layoutStr == "TND" || layoutStr == "NTD_TND" || layoutStr == "SH" || layoutStr == "NSD") {
        return ge::GRAPH_SUCCESS;
    }
    const gert::StorageShape* queryShape = contextKeyParams.queryInputShape;
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    int64_t queryD = 0;
    int64_t keyD = 0;
    int64_t valueD = 0;
    int64_t tempQN = 1;
    int64_t tempKVN = 1;
    // KV may be BSH or BBH(page attention)
    if (keyShape->GetStorageShape().GetDimNum() == NUM_3) {
        OP_CHECK_IF((contextKeyParams.headsNumber == nullptr),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "the query's heads num can not be nullptr!"),
                return ge::GRAPH_FAILED);
        OP_CHECK_IF((*(contextKeyParams.headsNumber) == 0),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "the query's heads num can not be 0!"),
                return ge::GRAPH_FAILED);
        tempQN = *(contextKeyParams.headsNumber);
        tempKVN = (contextKeyParams.numKeyValueHeads != nullptr && *(contextKeyParams.numKeyValueHeads) != 0) ?
            *(contextKeyParams.numKeyValueHeads) : tempQN;
    }
    if (layoutStr == "BSH") {
        queryD = queryShape->GetStorageShape().GetDim(DIM_2) / tempQN;
        keyD = keyShape->GetStorageShape().GetDim(DIM_2) / tempKVN;
        valueD = valueShape->GetStorageShape().GetDim(DIM_2) / tempKVN;
        if ((queryD != keyD) || (queryD != valueD) || (keyD != valueD)) {
            return ge::GRAPH_FAILED;
        }
    } else if (layoutStr == "BNSD" || layoutStr == "BNSD_BSND" || layoutStr == "BSND") {
        queryD = queryShape->GetStorageShape().GetDim(DIM_3);
        if (keyShape->GetStorageShape().GetDimNum() == NUM_3) {
            keyD = keyShape->GetStorageShape().GetDim(DIM_2) / tempKVN;
            valueD = valueShape->GetStorageShape().GetDim(DIM_2) / tempKVN;
        } else {
            keyD = keyShape->GetStorageShape().GetDim(DIM_3);
            valueD = valueShape->GetStorageShape().GetDim(DIM_3);
        }
        if ((queryD != keyD) || (queryD != valueD) || (keyD != valueD)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::CheckMaskType(ContextParamsForPFATiling &contextKeyParams,
    PromptFlashAttentionTilingData &tilingData,
    uint32_t &maskElemSize) const {
    if (contextKeyParams.attentionMask != nullptr) {
        auto maskDataType = contextKeyParams.maskDataType;
        if (maskDataType == ge::DT_FLOAT16) {
            maskElemSize = FLOAT16SIZE;
        } else if (maskDataType == ge::DT_BOOL) {
            maskElemSize = BOOLSIZE;
        } else if (maskDataType == ge::DT_INT8) {
            // Adapt to static graph mode, bool type attentionmask is converted to int8.
            maskElemSize = INT8SIZE;
        } else if (maskDataType == ge::DT_UINT8) {
            maskElemSize = UINT8SIZE;
        }
        // FP32 mask type does not support.
        OP_CHECK_IF(maskDataType == ge::DT_FLOAT,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "invalid maskType dtype[%s], maskType should not be float[%s]",
                    g_strDataTypePfa.at(ValidPfaDataType(maskDataType)).c_str(),
                    g_strDataTypePfa.at(ValidPfaDataType(ge::DT_FLOAT)).c_str()),
                return ge::GRAPH_FAILED);
        // When in fp16 high-precision mode, the mask type only supports bool or int8.
        OP_CHECK_IF(((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) &&
                (maskDataType != ge::DT_BOOL) && (maskDataType != ge::DT_INT8) && (maskDataType != ge::DT_UINT8),
            OPS_REPORT_VECTOR_INNER_ERR(
                contextKeyParams.opName,
                "invalid maskType dtype[%s], maskType should be bool, int8 or uint8 when fp16 high-precision mode",
                g_strDataTypePfa.at(ValidPfaDataType(maskDataType)).c_str()),
            return ge::GRAPH_FAILED);
        // When bf16, the mask type only supports bool or int8.
        OP_CHECK_IF((inputType == ge::DT_BF16) && (maskDataType != ge::DT_BOOL) && (maskDataType != ge::DT_INT8) &&
                (maskDataType != ge::DT_UINT8),
            OPS_REPORT_VECTOR_INNER_ERR(
                contextKeyParams.opName,
                "invalid maskType dtype[%s], maskType should be bool, int8 or uint8 when input type is bfloat16",
                g_strDataTypePfa.at(ValidPfaDataType(maskDataType)).c_str()),
            return ge::GRAPH_FAILED);
        // FP16 mask type does not support invalid line correction.
        OP_CHECK_IF((maskDataType == ge::DT_FLOAT16 && (tilingData.promptAttentionBaseParams.get_isRowInvalid() == 1U)),
            OPS_REPORT_VECTOR_INNER_ERR(
                contextKeyParams.opName,
                "invalid maskType dtype[%s], maskType should not be float16 when innerPrecise = 2 or 3",
                g_strDataTypePfa.at(ValidPfaDataType(maskDataType)).c_str()),
            return ge::GRAPH_FAILED);
        if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
            OP_CHECK_IF(maskDataType != ge::DT_BOOL,
                OPS_REPORT_VECTOR_INNER_ERR(
                    contextKeyParams.opName,
                    "invalid maskType dtype[%s], maskType should be bool when socVersion is 310p",
                    g_strDataTypePfa.at(ValidPfaDataType(maskDataType)).c_str()),
                return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::CheckBaseApiMaskVal(ContextParamsForPFATiling& contextKeyParams,
    const gert::StorageShape* pseShiftShape,
    const std::pair<std::vector<int64_t>, std::string> maskShape) const {
    int32_t maskDim = static_cast<int32_t>(pseShiftShape->GetStorageShape().GetDimNum());
    std::vector<int64_t> supportShape = maskShape.first;
    string maskTypeName = maskShape.second;
    if (maskDim == static_cast<int32_t>(supportShape.size())) {
        for (int32_t i = maskDim - 1; i >= 0; i--) {
            uint32_t supportShapeVal = supportShape[i];
            uint32_t actualShapeVal = pseShiftShape->GetStorageShape().GetDim(i);
            if (i >= maskDim - TWO_HOST_TILING) {
                OP_CHECK_IF(actualShapeVal < supportShapeVal,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                        "%s mask dim %d shape invalid, should be greater than %u, actual is %u",
                            maskTypeName.c_str(), i, supportShapeVal, actualShapeVal), return ge::GRAPH_FAILED);
            } else {
                OP_CHECK_IF(actualShapeVal != supportShapeVal,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                        "%s mask dim %d shape invalid, should be equal to %u, actual is %u",
                            maskTypeName.c_str(), i, supportShapeVal, actualShapeVal), return ge::GRAPH_FAILED);
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::CheckBaseApiAlibiMask(ContextParamsForPFATiling& contextKeyParams,
    const gert::StorageShape* pseShiftShape,
    uint32_t batchSize, int32_t maxSeqLen, int32_t maxKvSeqLen,
    uint32_t kvHead, bool compressHead) const {
    uint32_t maskDim = pseShiftShape->GetStorageShape().GetDimNum();
    uint32_t headNum = *contextKeyParams.headsNumber;
    OP_CHECK_IF(maskDim < NUM_2 || maskDim > NUM_4,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Alibi mask dim num invalid, should be 2, 3 or 4."),
        return ge::GRAPH_FAILED);
    bool isAlibiCompress = (pseShiftShape->GetStorageShape().GetDim(maskDim - NUM_1) == LONG_SEQ_LEN) &&
        (pseShiftShape->GetStorageShape().GetDim(maskDim - NUM_2) != LONG_SEQ_LEN);
    if (maskDim == 2U) {
        OP_CHECK_IF(CheckBaseApiMaskVal(contextKeyParams, pseShiftShape, {{LONG_SEQ_ALIBI_LEN, LONG_SEQ_ALIBI_LEN}, "alibi dim 2"}) != ge::GRAPH_SUCCESS,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Alibi long seq mask invalid."),
                return ge::GRAPH_FAILED);
    } else {
        if (isAlibiCompress) {
            OP_CHECK_IF(maskDim != NUM_3,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Alibi compress mask dim num invalid, should be 3."),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(CheckBaseApiMaskVal(contextKeyParams, pseShiftShape, {{headNum, maxSeqLen, LONG_SEQ_LEN}, "alibi longseq"}) != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Alibi compress mask shape invalid."),
                return ge::GRAPH_FAILED);
        } else {
            if (!compressHead) {
                OP_CHECK_IF(CheckBaseApiMaskVal(contextKeyParams, pseShiftShape, {{headNum, maxSeqLen, maxKvSeqLen}, "alibi"}) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Alibi mask shape invalid."),
                        return ge::GRAPH_FAILED);
                OP_CHECK_IF(CheckBaseApiMaskVal(contextKeyParams, pseShiftShape, {{batchSize, headNum, maxSeqLen, maxKvSeqLen}, "alibi"}) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Alibi mask shape invalid."),
                        return ge::GRAPH_FAILED);
            } else {
                OP_CHECK_IF(CheckBaseApiMaskVal(contextKeyParams, pseShiftShape, {{headNum, maxSeqLen, maxKvSeqLen}, "alibi with compress"}) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Alibi compress head mask shape invalid."),
                        return ge::GRAPH_FAILED);
                OP_CHECK_IF(CheckBaseApiMaskVal(contextKeyParams, pseShiftShape, {{batchSize / kvHead, headNum, maxSeqLen, maxKvSeqLen}, "alibi with compress"}) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Alibi compress head mask shape invalid."),
                        return ge::GRAPH_FAILED);
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::CheckBaseApiNormMask(ContextParamsForPFATiling& contextKeyParams,
    const gert::StorageShape* pseShiftShape,
    int32_t maskType, uint32_t batchSize,
    int32_t maxSeqLen, int32_t maxKvSeqLen,
    bool compressHead) const {
    if (maskType == SPARSE_MODE_NORM) {
        OP_CHECK_IF(pseShiftShape->GetStorageShape().GetDimNum() != NUM_4 &&
            pseShiftShape->GetStorageShape().GetDimNum() != NUM_3 &&
            pseShiftShape->GetStorageShape().GetDimNum() != NUM_2,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "norm mask dim num invalid, should be 2 or 3 or 4."),
            return ge::GRAPH_FAILED);
        if (!compressHead) {
            OP_CHECK_IF(CheckBaseApiMaskVal(contextKeyParams, pseShiftShape, {{maxSeqLen, maxKvSeqLen}, "norm"}) != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Norm compress mask shape invalid."),
                    return ge::GRAPH_FAILED);
            OP_CHECK_IF(CheckBaseApiMaskVal(contextKeyParams, pseShiftShape, {{batchSize, maxSeqLen, maxKvSeqLen}, "norm"}) != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Norm compress mask shape invalid."),
                    return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::CheckBaseApiMaskBasic(ContextParamsForPFATiling& contextKeyParams,
    const gert::StorageShape* pseShiftShape, bool isLongSeq,
    uint32_t batchSize) const
{
    OP_CHECK_IF(contextKeyParams.pseShift == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "maskType is not no mask, but mask is nullptr."),
        return ge::GRAPH_FAILED);
    // mask dtype
    OP_CHECK_IF(contextKeyParams.pseShiftDataType != ge::DT_FLOAT16 &&
        contextKeyParams.pseShiftDataType != ge::DT_BF16,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Mask datatype invalid."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(contextKeyParams.pseShiftDataType != contextKeyParams.inputDataType &&
        contextKeyParams.inputDataType != ge::DT_INT8,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Mask datatype not consist with query."),
        return ge::GRAPH_FAILED);
    // mask dim
    OP_CHECK_IF(pseShiftShape->GetStorageShape().GetDimNum() < NUM_2,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Mask dim num invalid, should be over 2."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(pseShiftShape->GetStorageShape().GetDimNum() > NUM_4,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Mask dim num invalid, should be less or equal than 4."),
        return ge::GRAPH_FAILED);
    if (isLongSeq) {
        OP_CHECK_IF(pseShiftShape->GetStorageShape().GetDimNum() != NUM_2 &&
            pseShiftShape->GetStorageShape().GetDimNum() != NUM_3,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "long seq mask dim num invalid, should be 2 or 3."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckBaseApiMaskVal(contextKeyParams, pseShiftShape, {{LONG_SEQ_LEN, LONG_SEQ_LEN}, "longseq"}) != ge::GRAPH_SUCCESS,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "long seq mask shape invalid."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckBaseApiMaskVal(contextKeyParams, pseShiftShape, {{batchSize, LONG_SEQ_LEN, LONG_SEQ_LEN}, "longseq"}) != ge::GRAPH_SUCCESS,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "long seq mask shape invalid."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::SetBaseApiAlibiMaskInfo(ContextParamsForPFATiling &contextKeyParams,
    const gert::StorageShape* pseShiftShape)
{
    uint32_t maskDim = pseShiftShape->GetStorageShape().GetDimNum();
    PromptAttentionBaseApiBaseParams* baseParams = &baseApiTilingData.promptAttentionBaseApiBaseParams;
    uint32_t maxSeqLen = baseParams->get_maxSeqLen();
    if (maskDim == NUM_3) {
        baseParams->set_headStride(maxSeqLen);
        baseParams->set_maskStride(NUM_0);
    } else if (maskDim == NUM_4) {
        OP_CHECK_IF(pseShiftShape->GetStorageShape().GetDim(TWO_HOST_TILING) * pseShiftShape->GetStorageShape().GetDim(ONE_HOST_TILING) > UINT32_MAX,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Alibi mask shape value invalid."),
            return ge::GRAPH_FAILED);
        baseParams->set_maskStride(pseShiftShape->GetStorageShape().GetDim(ONE_HOST_TILING) * pseShiftShape->GetStorageShape().GetDim(TWO_HOST_TILING));
        baseParams->set_headStride(maxSeqLen);
    } else if (maskDim == NUM_2) {
        OP_CHECK_IF(maxSeqLen != LONG_SEQ_ALIBI_LEN,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Long seq alibi mask shape value invalid."),
            return ge::GRAPH_FAILED);
    }
    OP_LOGI(contextKeyParams.opName, "maskstride %u, headstride %u", baseParams->get_maskStride(), baseParams->get_headStride());
    return ge::GRAPH_SUCCESS;
}

void PromptFlashAttentionTiling::SetBaseApiOtherMaskInfo(ContextParamsForPFATiling &contextKeyParams, const gert::StorageShape* pseShiftShape)
{
    uint32_t maskDim = pseShiftShape->GetStorageShape().GetDimNum();
    PromptAttentionBaseApiBaseParams* baseParams = &baseApiTilingData.promptAttentionBaseApiBaseParams;
    uint32_t maxSeqLen = baseParams->get_maxSeqLen();
    if (maskDim == NUM_3) {
        baseParams->set_maskStride(pseShiftShape->GetStorageShape().GetDim(1));
        baseParams->set_headStride(NUM_0);
    } else if (maskDim == NUM_2) {
        baseParams->set_maskStride(NUM_0);
        baseParams->set_headStride(NUM_0);
    } else if (maskDim == NUM_4) {
       baseParams->set_maskType(static_cast<uint32_t>(MaskType::MASK_TYPE_NORM));
       baseParams->set_headStride(maxSeqLen);
       baseParams->set_maskStride(baseParams->get_headNumSize() * baseParams->get_maxKvSeqLen());
    }
    if (maxSeqLen == LONG_SEQ_LEN && baseParams->get_isTriuMask() != 0) {
        baseParams->set_isLongSeq(NUM_1);
    }
    OP_LOGI(contextKeyParams.opName, "maskstride %u, headstride %u", baseParams->get_maskStride(), baseParams->get_headStride());
}

ge::graphStatus PromptFlashAttentionTiling::SetBaseApiPseInfo(ContextParamsForPFATiling &contextKeyParams, const gert::StorageShape* pseShiftShape)
{
    const int32_t* sparseMode = contextKeyParams.sparseMode;
    uint32_t maskType = *sparseMode;
    PromptAttentionBaseApiBaseParams* baseParams = &baseApiTilingData.promptAttentionBaseApiBaseParams;
    uint32_t maxSeqLen = baseParams->get_maxSeqLen();
    bool isAlibi = maskType == SPARSE_MODE_ALIBI;
    if (maskType >= SPARSE_MODE_NORM) {
        uint32_t maskDim = pseShiftShape->GetStorageShape().GetDimNum();
        if (maxSeqLen == LONG_SEQ_LEN && pseShiftShape->GetStorageShape().GetDim(maskDim - NUM_2) != maxSeqLen) {
            baseParams->set_alibiCompressOffset(static_cast<uint32_t>(pseShiftShape->GetStorageShape().GetDim(maskDim - NUM_2)));
        }
        if (isAlibi) {
            baseParams->set_maskType(static_cast<uint32_t>(MaskType::MASK_TYPE_ALIBI));
            OP_CHECK_IF(SetBaseApiAlibiMaskInfo(contextKeyParams, pseShiftShape) != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "set alibi mask attrs failed."),
                return ge::GRAPH_FAILED);
        } else {
            SetBaseApiOtherMaskInfo(contextKeyParams, pseShiftShape);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::CheckBaseApiPse(ContextParamsForPFATiling &contextKeyParams,
        const gert::StorageShape* pseShiftShape) {
    const int32_t* sparseMode = contextKeyParams.sparseMode;
    if (pseShiftShape == nullptr) {
        OP_CHECK_IF(*sparseMode != SPARSE_MODE_NONE,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "when mask is nullptr, sparseMode should be 20."),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    PromptAttentionBaseApiBaseParams* baseParams = &baseApiTilingData.promptAttentionBaseApiBaseParams;
    uint32_t batchSize = baseParams->get_batchSize();
    uint32_t kvHead = baseParams->get_kvHeadNumSize();
    uint32_t maxSeqLen = baseParams->get_maxSeqLen();
    uint32_t maxKvSeqLen = baseParams->get_maxKvSeqLen();
    bool compressHead = false;

    if (tilingMod == TilingMod::CVDIFF_MLA) {
        OP_CHECK_IF(*sparseMode >= SPARSE_MODE_ALIBI,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "MLA not support lookahead, alibi, swa mask"), return ge::GRAPH_FAILED);
    }
    if (*sparseMode >= SPARSE_MODE_NORM) {
        uint32_t maskDim = pseShiftShape->GetStorageShape().GetDimNum();
        bool isLongSeq = (baseParams->get_isTriuMask() == 1U)
            && (pseShiftShape->GetStorageShape().GetDim(maskDim - 1) == LONG_SEQ_LEN);
        OP_CHECK_IF(baseParams->get_maxSeqLen() > UINT32_MAX, OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "maxseq more than UINT32_MAX."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckBaseApiMaskBasic(contextKeyParams, pseShiftShape, isLongSeq, batchSize) != ge::GRAPH_SUCCESS,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Check mask basic info failed."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckBaseApiNormMask(contextKeyParams, pseShiftShape, *sparseMode, batchSize, maxSeqLen, maxKvSeqLen, compressHead) != ge::GRAPH_SUCCESS,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Norm mask check shape failed."),
            return ge::GRAPH_FAILED);
        if (*sparseMode == SPARSE_MODE_ALIBI) {
            OP_CHECK_IF(CheckBaseApiAlibiMask(contextKeyParams, pseShiftShape, batchSize, maxSeqLen, maxKvSeqLen, kvHead, compressHead) != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Alibi mask check shape failed."),
                return ge::GRAPH_FAILED);
        }
    }
    OP_CHECK_IF(SetBaseApiPseInfo(contextKeyParams, pseShiftShape) != ge::GRAPH_SUCCESS,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "Set Pse Info failed."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void PromptFlashAttentionTiling::SetMaskSize(const gert::StorageShape* attenMaskShape, PromptFlashAttentionTilingData& tilingData) const {
    auto maskKVsSize = 2048; // 2048 : default the last frist dim.
    auto maskQsSize = 2048; // 2048 : default the last second dim.
    if (attenMaskShape != nullptr) {
        maskKVsSize = attenMaskShape->GetStorageShape().GetDim(attenMaskShape->GetStorageShape().GetDimNum() - 1); // 1: last frist dim
        maskQsSize = attenMaskShape->GetStorageShape().GetDim(attenMaskShape->GetStorageShape().GetDimNum() - 2); // 2: last second dim
    }

    tilingData.promptAttentionBaseParams.set_maskKVsSize(maskKVsSize);
    tilingData.promptAttentionBaseParams.set_maskQsSize(maskQsSize);
}

ge::graphStatus PromptFlashAttentionTiling::CheckShape(ContextParamsForPFATiling& contextKeyParams, const gert::StorageShape* queryShape,
    const gert::StorageShape* keyShape, const gert::StorageShape* valueShape, const gert::StorageShape* outShape,
    const gert::StorageShape* pseShiftShape, const gert::StorageShape* attenMaskShape) {
    if (CheckNonEmptyShapeExceptions(contextKeyParams, queryShape, "query")) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNonEmptyShapeExceptions(contextKeyParams, keyShape, "key")) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNonEmptyShapeExceptions(contextKeyParams, valueShape, "value")) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNonEmptyShapeExceptions(contextKeyParams, outShape, "out")) {
        return ge::GRAPH_FAILED;
    }
    // Optional input can be empty.
    OP_CHECK_IF((pseShiftShape != nullptr) &&
        (pseShiftShape->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "Shape size of pseShift is overflow."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((attenMaskShape != nullptr) &&
        (attenMaskShape->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "Shape size of attenMask is overflow."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((outShape->GetStorageShape().GetShapeSize() != 0) &&
        (queryShape->GetStorageShape().GetShapeSize() == 0),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "query is empty tensor."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((queryShape->GetStorageShape().GetDimNum() < NUM_2) || (queryShape->GetStorageShape().GetDimNum() > 4),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "queryShape dim num is error, queryShape dim num = %lu", queryShape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(SetInputLayout(contextKeyParams.layout) == GRAPH_FAILED,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "when D != 512 only support layout BSH, "
        "BSND, BNSD, BNSD_BSND, TND, NTD_TND."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((inputLayout != InputLayout::NTD_TND) && (inputLayout != InputLayout::TND) &&
        (contextKeyParams.queryRope != nullptr || contextKeyParams.keyRope != nullptr),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "Only layout TND, NTD_TND support query/key D = 128, queryRopr/keyRope D = 64!"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(((contextKeyParams.fromFused == 0) && (inputLayout == InputLayout::NTD_TND)),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "NTD_TND layout is only supported in FIA, not in PFA."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::CheckBaseApiRequiredInput(ContextParamsForPFATiling& contextKeyParams) {
    const gert::StorageShape* queryShape = contextKeyParams.queryInputShape;
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    const gert::StorageShape* attenMaskShape = contextKeyParams.attentionMaskShape;
    const gert::StorageShape* quantScale2Shape = contextKeyParams.scale2Shape;
    const gert::StorageShape* quantOffset2Shape = contextKeyParams.offset2Shape;
    const gert::StorageShape* antiquantScaleShape = contextKeyParams.antiquantScaleShape;
    const gert::StorageShape* antiquantOffsetShape = contextKeyParams.antiquantOffsetShape;
    const gert::StorageShape* outShape = contextKeyParams.outputShape;

    if (CheckBaseApiNonEmptyShapeExceptions(contextKeyParams, queryShape, "query")) {
        return ge::GRAPH_FAILED;
    }
    if (CheckBaseApiNonEmptyShapeExceptions(contextKeyParams, keyShape, "key")) {
        return ge::GRAPH_FAILED;
    }
    if (CheckBaseApiNonEmptyShapeExceptions(contextKeyParams, valueShape, "value")) {
        return ge::GRAPH_FAILED;
    }
    if (CheckBaseApiNonEmptyShapeExceptions(contextKeyParams, outShape, "out")) {
        return ge::GRAPH_FAILED;
    }

    if (contextKeyParams.inputDataType == ge::DT_FLOAT16 &&
        !(contextKeyParams.kDataType == ge::DT_FLOAT16 &&
        contextKeyParams.vDataType == ge::DT_FLOAT16 &&
        contextKeyParams.outputDataType == ge::DT_FLOAT16)) {
        return ge::GRAPH_FAILED;
    }

    if (contextKeyParams.inputDataType == ge::DT_BF16 &&
        !(contextKeyParams.kDataType == DT_BF16 &&
        contextKeyParams.vDataType == DT_BF16 &&
        (contextKeyParams.outputDataType == ge::DT_FLOAT16 || contextKeyParams.outputDataType == ge::DT_BF16))) {
        return ge::GRAPH_FAILED;
    }

    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P) {
        // 1、校验 NZ 格式的 DIM
        static const size_t Q_CACHE_DIM_NUM = 4;
        static const size_t KV_CACHE_DIM_NUM = 4;
        OP_CHECK_IF((queryShape->GetStorageShape().GetDimNum() != Q_CACHE_DIM_NUM),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "query dim num %lu, invalid, should be %lu", queryShape->GetStorageShape().GetDimNum(), Q_CACHE_DIM_NUM),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((keyShape->GetStorageShape().GetDimNum() != KV_CACHE_DIM_NUM),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "key dim num %lu, invalid, should be %lu", keyShape->GetStorageShape().GetDimNum(), KV_CACHE_DIM_NUM),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((valueShape->GetStorageShape().GetDimNum() != KV_CACHE_DIM_NUM),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                "value dim num %lu, invalid, should be %lu", valueShape->GetStorageShape().GetDimNum(), KV_CACHE_DIM_NUM),
            return ge::GRAPH_FAILED);
        // 2、校验 scaleValue
        OP_CHECK_IF((contextKeyParams.scaleValue == nullptr),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "scaleValue should not be null"),
            return ge::GRAPH_FAILED);
        // 3、校验 HeadNum
        OP_CHECK_IF((contextKeyParams.headsNumber == nullptr),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "headNum should not be null"),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF((contextKeyParams.numKeyValueHeads == nullptr),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "kvHeadNum should not be null"),
            return ge::GRAPH_FAILED);

        const int64_t headNum = *contextKeyParams.headsNumber;
        const int64_t kvHeadNum = (*contextKeyParams.numKeyValueHeads != 0) ? *contextKeyParams.numKeyValueHeads : headNum;

        OP_CHECK_IF((headNum < 0),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "headNum can not be less than 0"),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF((kvHeadNum < 0),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "the key/value's heads num can not be less than 0"),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF((kvHeadNum > headNum),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "the query's heads num can not be less than the key/value's heads num"),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF((headNum % kvHeadNum != 0),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "the query's heads num must be an integer multiple of the key/value's heads num"),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF((headNum / kvHeadNum > 64),
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "GQA does not support groups larger than 64"),
            return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF((attenMaskShape != nullptr || quantScale2Shape != nullptr || quantOffset2Shape != nullptr || antiquantScaleShape != nullptr || antiquantOffsetShape != nullptr),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "PFA SH layout do not support attenMask or antiquant"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(SetInputLayout(contextKeyParams.layout) == GRAPH_FAILED,
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "invalid input layout:%s.", contextKeyParams.layout),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((outShape->GetStorageShape().GetShapeSize() != 0) &&
        (queryShape->GetStorageShape().GetShapeSize() == 0),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "query is empty tensor."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((queryShape->GetStorageShape().GetDimNum() < NUM_2) || (queryShape->GetStorageShape().GetDimNum() > 4),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName, "queryShape dim num is error, queryShape dim num = %lu", queryShape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PromptFlashAttentionTiling::CheckBaseApiOptionalInput(ContextParamsForPFATiling& contextKeyParams) {
    const int64_t* innerPrecisePtr = contextKeyParams.innerPrecisePtr;
    innerPrecise = innerPrecisePtr ? *innerPrecisePtr : HIGH_PERFORMANCE;
    if (contextKeyParams.inputDataType == ge::DT_BF16) {
        innerPrecise = HIGH_PRECISION;
    }

    OP_CHECK_IF((innerPrecise != HIGH_PRECISION) && (innerPrecise != HIGH_PERFORMANCE),
        OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
        "innerPrecise need to be HIGH_PRECISION or HIGH_PERFORMANCE"),
        return ge::GRAPH_FAILED);

    if (curShortSocName == platform_ascendc::SocVersion::ASCEND310P && innerPrecise != HIGH_PERFORMANCE) {
        return ge::GRAPH_FAILED;
    }

    if (tilingMod == TilingMod::CVDIFF_BASE_API && curShortSocName != platform_ascendc::SocVersion::ASCEND310P) {
        OP_CHECK_IF(inputLayout != InputLayout::SH,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "inputLayout need to be SH"),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(contextKeyParams.actualSequenceLengthQ == nullptr || contextKeyParams.actualSequenceLengthKV == nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
            "actualSequenceLengthQ and actualSequenceLengthKV can't be null"),
            return ge::GRAPH_FAILED);
    }

    const gert::StorageShape* pseShiftShape = contextKeyParams.pseShiftShape;
    if (CheckBaseApiPse(contextKeyParams, pseShiftShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto qDataType = contextKeyParams.inputDataType;
    auto kDataType = contextKeyParams.kDataType;
    auto vDataType = contextKeyParams.vDataType;
    auto outputDataType = contextKeyParams.outputDataType;
    if ((qDataType == ge::DT_INT8) && (kDataType == ge::DT_INT8) && (vDataType == ge::DT_INT8) &&
        (outputDataType == ge::DT_FLOAT16 || outputDataType == ge::DT_BF16)) {
            const gert::StorageShape* deqScale1Shape = contextKeyParams.deqScale1Shape;
            const gert::StorageShape* quantScale1Shape = contextKeyParams.scale1Shape;
            const gert::StorageShape* deqScale2Shape = contextKeyParams.deqScale2Shape;
            PromptAttentionBaseApiBaseParams* baseParams = &baseApiTilingData.promptAttentionBaseApiBaseParams;
            uint32_t qHeads = baseParams->get_headSize();

            OP_CHECK_IF((deqScale1Shape != nullptr && (deqScale1Shape->GetStorageShape().GetShapeSize() != 1 ||
                deqScale1Shape->GetStorageShape().GetDim(0) != qHeads)),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "invalid deqScale1Shape size or value! "
                    "The size must equal 1 and the value must equal Q headNums(%u)", qHeads),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF((quantScale1Shape != nullptr && (quantScale1Shape->GetStorageShape().GetShapeSize() != 1 ||
            quantScale1Shape->GetStorageShape().GetDim(0) != qHeads)),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "invalid quantScale1Shape size or value! "
                    "The size must equal 1 and the value must equal Q headNums(%u)", qHeads),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF((deqScale2Shape != nullptr && (deqScale2Shape->GetStorageShape().GetShapeSize() != 1 ||
                deqScale2Shape->GetStorageShape().GetDim(0) != qHeads)),
                OPS_REPORT_VECTOR_INNER_ERR(contextKeyParams.opName,
                    "invalid deqScale2Shape size or value! "
                    "The size must equal 1 and the value must equal Q headNums(%u)", qHeads),
                return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

void PromptFlashAttentionTiling::InferTilingMod(const ContextParamsForPFATiling& contextKeyParams, const std::vector<int64_t>& actualSeqLengths, const std::vector<int64_t>& actualSeqLengthsKV,
    uint32_t actualSeqArrayLen, uint32_t hDivN, uint32_t seqInnerSize, int32_t sparseModeVal)
{
    if (hDivN > DSPLIT_THRESHOLDS_512) {   // D segmentation threshold // S1S2D splits into fp16 and int8 types
        splitD = 1U;
    }

    if ((seqInnerSize <= DSPLIT_THRESHOLDS_512) && (splitD == 1U)) {
        splitS2 = 0U;
    }

    if ((curShortSocName != platform_ascendc::SocVersion::ASCEND310P) &&
        (splitD != 1U) && (isDNoTail == true)) {
        tilingMod = TilingMod::CVDIFF;
    }

    // Determine whether to use the norm template
    if (curShortSocName != platform_ascendc::SocVersion::ASCEND310P) {
        int64_t minActualSeqLengths = INT64_MAX;
        int64_t minActualSeqLengthsKV = INT64_MAX;
        for (uint32_t i = 0; i < actualSeqArrayLen; ++i) {
            minActualSeqLengths = std::min(minActualSeqLengths, actualSeqLengths[i]);
            minActualSeqLengthsKV = std::min(minActualSeqLengthsKV, actualSeqLengthsKV[i]);
        }
        if (minActualSeqLengths >= MATMUL_NORM_MIN_SEQ && minActualSeqLengthsKV >= MATMUL_NORM_MIN_SEQ && hDivN == MATMUL_NORM_MIN_HEADSIZE &&
            inputType == ge::DT_FLOAT16 && contextKeyParams.kDataType == ge::DT_FLOAT16 &&
            contextKeyParams.maskDataType == ge::DT_BOOL && outputType == ge::DT_FLOAT16 && usePseShift == 0 &&
            inputLayout == InputLayout::BNSD && sparseModeVal == SPARSE_MODE_BAND && (!enablePA)) {
            enableMatmulNorm = true;
        }
    }
}

ge::graphStatus TilingPromptFlashAttention(gert::TilingContext* context) {
    if (context == nullptr) {
            OP_LOGE("PromptFlashAttention", "tiling context is nullptr!");
            return ge::GRAPH_FAILED;
    }
    auto resultCode = FiaTilingRegistry::GetInstance().DoTilingImpl(context, nullptr);
    return resultCode;
}

REGISTER_TILING_TEMPLATE_FIA(PromptFlashAttention, PromptFlashAttentionTiling, std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201), static_cast<int32_t>(NpuArch::DAV_2002), static_cast<int32_t>(NpuArch::DAV_3003), static_cast<int32_t>(NpuArch::DAV_3113)}), 91);
}