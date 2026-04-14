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
 * \file resize_upsample_trilinear_tiling.cpp
 * \brief
 */
#include <vector>
#include "register/op_def_registry.h"
#include "resize_upsample_trilinear_tiling.h"

using namespace std;

#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGI(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGW(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define unlikely(x) __builtin_expect((x), 0)
#define OP_CHECK_IF(condition, log, return_expr) \
    do { \
        if (unlikely(condition)) { \
            log; \
            return_expr; \
        } \
    } while (0)

namespace optiling {

static constexpr int OUTPUT_SIZE_IDX = 0;
static constexpr int SCALES_IDX = 1;
static constexpr int ALIGN_CORNERS_IDX = 2;
static constexpr int INPUT_X_IDX = 0;
static constexpr int OUTPUT_Y_IDX = 0;

static constexpr int OUTPUT_WEIGHT_IDX = 2;
static constexpr int OUTPUT_HEIGHT_IDX = 1;
static constexpr int OUTPUT_DEPTH_IDX = 0;

static constexpr int SCALE_W_IDX = 4;
static constexpr int SCALE_H_IDX = 3;
static constexpr int SCALE_D_IDX = 2;

static constexpr int DIREC_WIDTH = 0;
static constexpr int DIREC_HEIGHT = 1;
static constexpr int DIREC_DEEPTH = 2;
static constexpr int REDUNDANCE = 2;

static constexpr int FLOAT_BYTE_LENGTH = 4;
static constexpr int FLOAT16_BYTE_LENGTH = 2;
static constexpr int BYTE_SIZE = 8;

static constexpr int TILING_DATATYPE_FP16 = 1000;
static constexpr int TILING_DATATYPE_FP32 = 3000;
static constexpr int TILING_DATATYPE_BF16 = 2000;

static constexpr int SLIDE_SIZE = 16;      // best matmul paradigm.bf16,f16:16*16 x 16*16 ,f32:16*8 x 8*16  value of N
static constexpr uint32_t BATCH_SIZE = 16; // value of M
static constexpr int RESERVED_WORKSPACE_SIZE = 32 * 1024 * 1024;

constexpr float MAX_SUPPORT_SCALE = 50.0f;
constexpr float BEST_PERFORMANCE_SCALE_1 = 27.0f;
constexpr float BEST_PERFORMANCE_SCALE_2 = 11.0f;
constexpr float BEST_PERFORMANCE_SCALE_3 = 3.0f;

constexpr uint32_t MIN_BATCH_SIZE = 128;
constexpr uint32_t BEST_PERFORMANCE_BATCH_SIZE_1 = 256;
constexpr uint32_t BEST_PERFORMANCE_BATCH_SIZE_2 = 512;
constexpr uint32_t BEST_PERFORMANCE_BATCH_SIZE_3 = 1024;

constexpr uint32_t MAX_DATA_NUM = 8192;
constexpr uint32_t RESERVED_NUM = 4;
constexpr uint8_t SCHEDULE_MODE = 1;

template <typename T>
inline uint32_t CEIL(T x)
{
    int32_t floor_v = int32_t(x);
    if (x == floor_v) {
        return floor_v;
    }
    return floor_v + 1;
}

template <typename T1, typename T2>
inline uint32_t CEILA2B(T1 a, T2 b)
{
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

inline bool AllTrue(bool a, bool b)
{
    return a && b;
}

inline uint32_t GET_DATATYPE_SIZE(ge::DataType dataType)
{
    if (dataType == ge::DT_FLOAT) {
        return FLOAT_BYTE_LENGTH;
    } else if (dataType == ge::DT_FLOAT16) {
        return FLOAT16_BYTE_LENGTH;
    } else if (dataType == ge::DT_BF16) {
        return FLOAT16_BYTE_LENGTH;
    } else {
        return FLOAT_BYTE_LENGTH;
    }
}

template <typename T>
auto GetOptionalAttr(const gert::RuntimeAttrs* attrs, const int idx, const T& defaultValue) -> T
{
    const T* attrPtr = attrs->GetAttrPointer<T>(idx);
    T outValue = (nullptr == attrPtr) ? defaultValue : (*attrPtr);
    return outValue;
}

static float ComputeScales(float scale, uint32_t input_size, uint32_t output_size)
{
    auto zero = static_cast<float>(0.);
    if (scale > zero) {
        return scale;
    } else if (output_size == 0) {
        return zero;
    } else {
        return (static_cast<float>(input_size) / output_size);
    }
}

static inline bool FloatEqual(float a, float b)
{
    float closeTo0 = float(1e-6);
    if (a > b) {
        return a - b < closeTo0;
    } else {
        return b - a < closeTo0;
    }
}

static float AreaPixelComputeScale(bool alignCorner, int64_t inputSize, int64_t outputSize, float scale)
{
    if (outputSize == inputSize) {
        return static_cast<float>(1);
    }
    if (alignCorner) {
        if (outputSize > 1) {
            return static_cast<float>(inputSize - 1) / (outputSize - 1);
        } else {
            return static_cast<float>(0);
        }
    } else {
        return ComputeScales(scale, inputSize, outputSize);
    }
}

static bool CheckShapes(gert::TilingContext* context, int64_t w, int64_t h, int64_t d)
{
    OP_CHECK_IF(
        !(w > 0 && h > 0 && d > 0),
        OP_LOGE(context->GetNodeName(), "shape should bigger than 0, but got shape (D: %ld,H: %ld, W: %ld)", d, h, w),
        return false);
    return true;
}

static bool CheckShapeMaxLimit(gert::TilingContext* context, UpsampleTrilinearTilingData& tiling)
{
    OP_CHECK_IF(
        tiling.get_input_w() > INT32_MAX,
        OP_LOGE(context->GetNodeName(), "The shape exceeds the limit. Input_w must less than 2147483647."),
        return false);
    OP_CHECK_IF(
        tiling.get_output_d() > INT32_MAX,
        OP_LOGE(context->GetNodeName(), "The shape exceeds the limit. Output_d must less than 2147483647."),
        return false);
    OP_CHECK_IF(
        tiling.get_output_w() * tiling.get_output_h() > INT32_MAX,
        OP_LOGE(context->GetNodeName(), "The shape exceeds the limit. Output_h * output_w must less than 2147483647."),
        return false);
    OP_CHECK_IF(
        tiling.get_batches() * tiling.get_input_d() * tiling.get_input_h() > INT32_MAX,
        OP_LOGE(
            context->GetNodeName(),
            "The shape exceeds the limit. Output_n * output_c * output_d * output_h must less than 2147483647."),
        return false);
    return true;
}

static bool CheckScales(gert::TilingContext* context, float scalesW, float scalesH, float scalesD)
{
    OP_CHECK_IF(
        (scalesW > MAX_SUPPORT_SCALE || scalesH > MAX_SUPPORT_SCALE || scalesD > MAX_SUPPORT_SCALE),
        OP_LOGE(
            context->GetNodeName(),
            "Scales should less than 50, but got scale (scalesW: %f, scalesH: %f, scalesD: %f) ", scalesW, scalesH,
            scalesD),
        return false);
    return true;
}

static bool SetTCubeTilingD(
    gert::TilingContext* context, uint32_t batches, UpsampleTrilinearTilingData& tiling, ge::DataType dataType)
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTiling_d;
    mmTiling_d.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_d.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_d.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTiling_d.SetShape(
        tiling.get_output_d(), tiling.get_output_w() * tiling.get_output_h(),
        CEIL(SLIDE_SIZE * tiling.get_scale_d()) + REDUNDANCE);
    mmTiling_d.SetOrgShape(
        tiling.get_output_d(), tiling.get_output_w() * tiling.get_output_h(), batches * tiling.get_input_d());
    OP_CHECK_IF(
        (mmTiling_d.GetTiling(tiling.matmul_tiling_d) == -1),
        OP_LOGE(context->GetNodeName(), "MatmulApiTiling_d set failed"), return false);
    return true;
}

static bool SetTCubeTilingH(
    gert::TilingContext* context, uint32_t batches, UpsampleTrilinearTilingData& tiling, ge::DataType dataType)
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTiling_h;
    mmTiling_h.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_h.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_h.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTiling_h.SetShape(
        tiling.get_output_h(), tiling.get_output_w(), CEIL(SLIDE_SIZE * tiling.get_scale_h()) + REDUNDANCE);
    mmTiling_h.SetOrgShape(
        tiling.get_output_h(), tiling.get_output_w(), batches * tiling.get_input_d() * tiling.get_input_h());
    OP_CHECK_IF(
        (mmTiling_h.GetTiling(tiling.matmul_tiling_h) == -1),
        OP_LOGE(context->GetNodeName(), "MatmulApiTiling_h set failed"), return false);
    return true;
}

static bool SetTCubeTilingW(
    gert::TilingContext* context, uint32_t batches, UpsampleTrilinearTilingData& tiling, ge::DataType dataType)
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTiling_w;
    mmTiling_w.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_w.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTiling_w.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTiling_w.SetShape(
        batches * tiling.get_input_d() * tiling.get_input_h(), SLIDE_SIZE,
        CEIL(SLIDE_SIZE * tiling.get_scale_w()) + REDUNDANCE);
    mmTiling_w.SetOrgShape(
        batches * tiling.get_input_d() * tiling.get_input_h(), tiling.get_output_w(), tiling.get_input_w());
    OP_CHECK_IF(
        (mmTiling_w.GetTiling(tiling.matmul_tiling_w) == -1),
        OP_LOGE(context->GetNodeName(), "MatmulApiTiling_w set failed"), return false);
    return true;
}

static void SetTailGroupList(
    UpsampleTrilinearTilingData& tiling, int direction, uint32_t* tailGroupSlideStartInxList,
    uint32_t* tailGroupSlideEndInxList)
{
    if (DIREC_WIDTH == direction) {
        tiling.set_tail_group_slide_start_inx_w_list(tailGroupSlideStartInxList);
        tiling.set_tail_group_slide_end_inx_w_list(tailGroupSlideEndInxList);
    } else if (DIREC_HEIGHT == direction) {
        tiling.set_tail_group_start_inx_h_list(tailGroupSlideStartInxList);
        tiling.set_tail_group_end_inx_h_list(tailGroupSlideEndInxList);
    } else {
        tiling.set_tail_group_start_inx_d_list(tailGroupSlideStartInxList);
        tiling.set_tail_group_end_inx_d_list(tailGroupSlideEndInxList);
    }
}

static void SetTailBatchList(
    UpsampleTrilinearTilingData& tiling, int direction, uint32_t* tailGroupBlockStartInxList,
    uint32_t* tailGroupBlockEndInxList)
{
    if (DIREC_WIDTH == direction) {
        tiling.set_tail_group_start_inx_w_list(tailGroupBlockStartInxList);
        tiling.set_tail_group_end_inx_w_list(tailGroupBlockEndInxList);
    } else if (DIREC_HEIGHT == direction) {
        tiling.set_tail_group_batch_start_inx_h_list(tailGroupBlockStartInxList);
        tiling.set_tail_group_batch_end_inx_h_list(tailGroupBlockEndInxList);
    } else {
        tiling.set_tail_group_batch_start_inx_d_list(tailGroupBlockStartInxList);
        tiling.set_tail_group_batch_end_inx_d_list(tailGroupBlockEndInxList);
    }
}

static uint32_t calcIndxPerCoreInTailBlockW(
    UpsampleTrilinearTilingData& tiling, uint32_t core_num, uint32_t group_size, uint32_t* core_per_group,
    uint32_t group_tail_w)
{
    uint32_t batch = tiling.get_batches() * tiling.get_input_d() * tiling.get_input_h();
    vector<uint32_t> tailGroupBlockStartInxList(MAX_CORE_COUNT, 0);
    vector<uint32_t> tailGroupBlockEndInxList(MAX_CORE_COUNT, 0);
    vector<uint32_t> tailGroupSlideStartInxList(MAX_CORE_COUNT, 0);
    vector<uint32_t> tailGroupSlideEndInxList(MAX_CORE_COUNT, 0);
    uint32_t slide_num = tiling.get_output_w() / (SLIDE_SIZE * core_num);
    uint32_t tail_block_start;
    uint32_t last_slide_idx = slide_num > 0 ? core_num * slide_num * SLIDE_SIZE - 1 : 0;
    uint32_t core_num_cur_group;
    uint32_t coreRealNeedNum = 0;
    size_t idx = 0;
    for (size_t i = 0; i < group_size; i++) {
        if (i == 0) {
            core_num_cur_group = core_per_group[i] + 1;
        } else {
            core_num_cur_group = core_per_group[i] - core_per_group[i - 1];
        }
        uint32_t avg_blocks_per_core = std::max(CEILA2B(batch, core_num_cur_group), BATCH_SIZE);
        for (size_t j = 0; j < core_num_cur_group; j++) {
            tail_block_start = j * avg_blocks_per_core;
            if (tail_block_start >= batch) {
                break;
            }
            tailGroupBlockStartInxList[idx] = tail_block_start;
            tailGroupBlockEndInxList[idx] =
                std::min(batch - 1, tailGroupBlockStartInxList[idx] + avg_blocks_per_core - 1);
            if (i == 0) {
                tailGroupSlideStartInxList[idx] = last_slide_idx == 0 ? 0 : last_slide_idx + i * SLIDE_SIZE + 1;
            } else {
                tailGroupSlideStartInxList[idx] =
                    last_slide_idx == 0 ? i * SLIDE_SIZE : last_slide_idx + i * SLIDE_SIZE + 1;
            }
            if (i == group_size - 1) {
                uint32_t group_width = group_tail_w == 0 ? SLIDE_SIZE : group_tail_w;
                tailGroupSlideEndInxList[idx] = tailGroupSlideStartInxList[idx] + group_width - 1;
            } else {
                tailGroupSlideEndInxList[idx] = tailGroupSlideStartInxList[idx] + SLIDE_SIZE - 1;
            }
            coreRealNeedNum++;
            idx++;
        }
    }
    SetTailGroupList(tiling, DIREC_WIDTH, tailGroupSlideStartInxList.data(), tailGroupSlideEndInxList.data());
    SetTailBatchList(tiling, DIREC_WIDTH, tailGroupBlockStartInxList.data(), tailGroupBlockEndInxList.data());
    return coreRealNeedNum;
}

static uint32_t calcIndxPerCoreInTailBlock(
    UpsampleTrilinearTilingData& tiling, uint32_t core_num, int direction, uint32_t group_size,
    uint32_t* core_per_group, uint32_t group_tail)
{
    uint32_t batch;
    uint32_t last_slide_index;
    uint32_t slide_num;
    if (DIREC_HEIGHT == direction) {
        batch = tiling.get_batches() * tiling.get_input_d();
        slide_num = tiling.get_output_h() / (SLIDE_SIZE * core_num);
    } else {
        batch = tiling.get_batches();
        slide_num = tiling.get_output_d() / (SLIDE_SIZE * core_num);
    }
    last_slide_index = slide_num > 0 ? core_num * slide_num * SLIDE_SIZE - 1 : 0;
    vector<uint32_t> tailGroupBlockStartInxList(MAX_CORE_COUNT, 0);
    vector<uint32_t> tailGroupBlockEndInxList(MAX_CORE_COUNT, 0);
    vector<uint32_t> tailGroupSlideStartInxList(MAX_CORE_COUNT, 0);
    vector<uint32_t> tailGroupSlideEndInxList(MAX_CORE_COUNT, 0);
    uint32_t core_num_cur_group;
    uint32_t coreRealNeedNum = 0;
    size_t free_core = 0;
    size_t idx = 0;
    for (size_t i = 0; i < group_size; i++) {
        uint32_t tail;
        uint32_t tail_temp;
        if (i == 0) {
            core_num_cur_group = core_per_group[i] + 1;
        } else {
            core_num_cur_group = core_per_group[i] - core_per_group[i - 1];
        }
        if (core_num_cur_group == 0) {
            continue;
        }
        uint32_t avg_batch_per_core = batch / core_num_cur_group;
        tail = batch % core_num_cur_group;
        uint32_t real_core = avg_batch_per_core > 0 ? core_num_cur_group : tail;
        tail_temp = tail;
        idx = idx - free_core;
        free_core = 0;
        for (size_t j = 0; j < core_num_cur_group; j++) {
            if (i == 0) {
                tailGroupSlideStartInxList[idx] = last_slide_index == 0 ? 0 : last_slide_index + i * SLIDE_SIZE + 1;
            } else {
                tailGroupSlideStartInxList[idx] =
                    last_slide_index == 0 ? i * SLIDE_SIZE : last_slide_index + i * SLIDE_SIZE + 1;
            }
            if (i == group_size - 1) {
                uint32_t group_width = group_tail == 0 ? SLIDE_SIZE : group_tail;
                tailGroupSlideEndInxList[idx] = tailGroupSlideStartInxList[idx] + group_width - 1;
            } else {
                tailGroupSlideEndInxList[idx] = tailGroupSlideStartInxList[idx] + SLIDE_SIZE - 1;
            }
            if (j < real_core) {
                if (tail_temp > 0) {
                    tail_temp--;
                    if (avg_batch_per_core == 0) {
                        tailGroupBlockStartInxList[idx] = j;
                        tailGroupBlockEndInxList[idx] = tailGroupBlockStartInxList[idx] + 1;
                    } else {
                        tailGroupBlockStartInxList[idx] = j == 0 ? 0 : j * avg_batch_per_core + tail - tail_temp - 1;
                        tailGroupBlockEndInxList[idx] = tailGroupBlockStartInxList[idx] + avg_batch_per_core + 1;
                    }
                } else {
                    tailGroupBlockStartInxList[idx] = j == 0 ? 0 : j * avg_batch_per_core + tail;
                    tailGroupBlockEndInxList[idx] = tailGroupBlockStartInxList[idx] + avg_batch_per_core;
                }
                coreRealNeedNum++;
            } else {
                free_core++;
            }
            idx++;
        }
    }
    SetTailGroupList(tiling, direction, tailGroupSlideStartInxList.data(), tailGroupSlideEndInxList.data());
    SetTailBatchList(tiling, direction, tailGroupBlockStartInxList.data(), tailGroupBlockEndInxList.data());
    return coreRealNeedNum;
}

static uint32_t calcIndxPerCore(UpsampleTrilinearTilingData& tiling, uint32_t core_num, int direction)
{
    int64_t direc_length;
    vector<uint32_t> tailGroupBlockStartInxList(MAX_CORE_COUNT, 0);
    vector<uint32_t> tailGroupBlockEndInxList(MAX_CORE_COUNT, 0);
    vector<uint32_t> tailGroupSlideStartInxList(MAX_CORE_COUNT, 0);
    vector<uint32_t> tailGroupSlideEndInxList(MAX_CORE_COUNT, 0);
    if ((DIREC_WIDTH == direction && FloatEqual(tiling.get_scale_w(), 1.0)) ||
        (DIREC_HEIGHT == direction && FloatEqual(tiling.get_scale_h(), 1.0))) {
        SetTailGroupList(tiling, direction, tailGroupSlideStartInxList.data(), tailGroupSlideEndInxList.data());
        SetTailBatchList(tiling, direction, tailGroupBlockStartInxList.data(), tailGroupBlockEndInxList.data());
        return 1;
    }
    if (AllTrue(DIREC_DEEPTH == direction, FloatEqual(tiling.get_scale_d(), 1.0)) &&
        !AllTrue(FloatEqual(tiling.get_scale_w(), 1.0), FloatEqual(tiling.get_scale_h(), 1.0))) {
        SetTailGroupList(tiling, direction, tailGroupSlideStartInxList.data(), tailGroupSlideEndInxList.data());
        SetTailBatchList(tiling, direction, tailGroupBlockStartInxList.data(), tailGroupBlockEndInxList.data());
        return 1;
    }
    if (DIREC_WIDTH == direction) {
        direc_length = tiling.get_output_w();
    } else if (DIREC_HEIGHT == direction) {
        direc_length = tiling.get_output_h();
    } else {
        direc_length = tiling.get_output_d();
    }
    uint32_t slide_num = direc_length / (SLIDE_SIZE * core_num);
    uint32_t slide_tail_size = direc_length % (SLIDE_SIZE * core_num);
    uint32_t group_size = CEILA2B(slide_tail_size, SLIDE_SIZE);
    uint32_t group_tail_size = slide_tail_size % SLIDE_SIZE;
    uint32_t core_num_per_group = 0;
    uint32_t core_num_per_group_tail = 0;
    uint32_t core_real_need_num = core_num;
    if (group_size != 0) {
        core_num_per_group = core_num / group_size;
        core_num_per_group_tail = core_num % group_size;
    }
    uint32_t core_num_per_group_tail_temp = core_num_per_group_tail;
    vector<uint32_t> core_per_group(group_size, 0);
    for (size_t i = 0; i < group_size; i++) {
        if (core_num_per_group_tail_temp > 0) {
            core_num_per_group_tail_temp--;
            core_per_group[i] =
                (i + 1) * core_num_per_group + core_num_per_group_tail - core_num_per_group_tail_temp - 1;
        } else {
            core_per_group[i] = (i + 1) * core_num_per_group + core_num_per_group_tail - 1;
        }
    }
    if (DIREC_WIDTH == direction) {
        core_real_need_num =
            calcIndxPerCoreInTailBlockW(tiling, core_num, group_size, core_per_group.data(), group_tail_size);
    } else {
        core_real_need_num =
            calcIndxPerCoreInTailBlock(tiling, core_num, direction, group_size, core_per_group.data(), group_tail_size);
    }
    return slide_num > 0 ? core_num : core_real_need_num;
}

static uint64_t GetWorkSpaceSize(
    UpsampleTrilinearTilingData& tiling, uint32_t batches, uint32_t coreRealNeedNum, uint32_t data_type_size)
{
    uint64_t temp_result_w_size;
    uint64_t temp_result_h_size;
    uint64_t ratio_size_w;
    uint64_t ratio_size_h;
    uint64_t ratio_size_d;
    temp_result_w_size = FloatEqual(tiling.get_scale_w(), 1.0) ?
                             0 :
                             batches * tiling.get_input_d() * tiling.get_input_h() * tiling.get_output_w();
    temp_result_h_size = FloatEqual(tiling.get_scale_h(), 1.0) ?
                             0 :
                             batches * tiling.get_input_d() * tiling.get_output_h() * tiling.get_output_w();
    ratio_size_w = (CEIL(SLIDE_SIZE * tiling.get_scale_w()) + REDUNDANCE) * SLIDE_SIZE;
    ratio_size_h = (CEIL(SLIDE_SIZE * tiling.get_scale_h()) + REDUNDANCE) * SLIDE_SIZE;
    ratio_size_d = (CEIL(SLIDE_SIZE * tiling.get_scale_d()) + REDUNDANCE) * SLIDE_SIZE;
    uint64_t max_ratio_size = std::max(ratio_size_w, std::max(ratio_size_h, ratio_size_d));

    tiling.set_ratio_metrix_size(max_ratio_size);
    return (temp_result_w_size + temp_result_h_size + max_ratio_size * coreRealNeedNum) * data_type_size;
}

static ge::graphStatus Tiling4UpsampleTrilinear(gert::TilingContext* context)
{
    // Use PlatformAscendC to get hardware info
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto maxCoreNum = ascendcPlatform.GetCoreNum();

    int batchIdx = OUTPUT_DEPTH_IDX;
    int channelIdx = OUTPUT_HEIGHT_IDX;
    int weightIdx = SCALE_W_IDX;
    int heightIdx = SCALE_H_IDX;
    int depthIdx = SCALE_D_IDX;
    int outwIdx = SCALE_W_IDX;
    int outhIdx = SCALE_H_IDX;
    int outdIdx = SCALE_D_IDX;

    UpsampleTrilinearTilingData tiling;

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const auto align_corners = GetOptionalAttr<bool>(attrs, SCALES_IDX, false);

    auto output_storage_shape = context->GetOutputShape(0)->GetStorageShape();
    if (!CheckShapes(
            context, output_storage_shape.GetDim(outwIdx), output_storage_shape.GetDim(outhIdx),
            output_storage_shape.GetDim(outdIdx))) {
        return ge::GRAPH_FAILED;
    }
    tiling.set_output_w(output_storage_shape.GetDim(outwIdx));
    tiling.set_output_h(output_storage_shape.GetDim(outhIdx));
    tiling.set_output_d(output_storage_shape.GetDim(outdIdx));
    auto input_storage_shape = context->GetInputShape(0)->GetStorageShape();
    tiling.set_scale_w(AreaPixelComputeScale(
        align_corners, input_storage_shape.GetDim(weightIdx), output_storage_shape.GetDim(outwIdx),
        GetOptionalAttr<float>(attrs, SCALE_W_IDX, static_cast<float>(1.0))));
    tiling.set_scale_h(AreaPixelComputeScale(
        align_corners, input_storage_shape.GetDim(heightIdx), output_storage_shape.GetDim(outhIdx),
        GetOptionalAttr<float>(attrs, SCALE_H_IDX, static_cast<float>(1.0))));
    tiling.set_scale_d(AreaPixelComputeScale(
        align_corners, input_storage_shape.GetDim(depthIdx), output_storage_shape.GetDim(outdIdx),
        GetOptionalAttr<float>(attrs, SCALE_D_IDX, static_cast<float>(1.0))));
    if (!CheckScales(context, tiling.get_scale_w(), tiling.get_scale_h(), tiling.get_scale_d())) {
        return ge::GRAPH_FAILED;
    }
    tiling.set_align_corners(align_corners ? 1 : 0);

    tiling.set_input_w(input_storage_shape.GetDim(weightIdx));
    tiling.set_input_h(input_storage_shape.GetDim(heightIdx));
    tiling.set_input_d(input_storage_shape.GetDim(depthIdx));
    if (!CheckShapes(context, tiling.get_input_w(), tiling.get_input_h(), tiling.get_input_d())) {
        return ge::GRAPH_FAILED;
    }
    // data type:fp32,fp16,bf16
    auto dataType = context->GetInputDesc(0)->GetDataType();
    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT16) {
        tilingKey = TILING_DATATYPE_FP16;
    } else if (dataType == ge::DT_FLOAT) {
        tilingKey = TILING_DATATYPE_FP32;
    } else if (dataType == ge::DT_BF16) {
        tilingKey = TILING_DATATYPE_BF16;
    }
    context->SetTilingKey(tilingKey);
    uint32_t data_type_size = GET_DATATYPE_SIZE(dataType);

    // Block Tiling
    tiling.set_total_core_num(maxCoreNum);
    const int64_t batch = input_storage_shape.GetDim(batchIdx);
    const int64_t channel = input_storage_shape.GetDim(channelIdx);
    tiling.set_batches(batch * channel);

    if (!CheckShapeMaxLimit(context, tiling)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t core_real_need_num_w = calcIndxPerCore(tiling, maxCoreNum, DIREC_WIDTH);
    uint32_t core_real_need_num_h = calcIndxPerCore(tiling, maxCoreNum, DIREC_HEIGHT);
    uint32_t core_real_need_num_d = calcIndxPerCore(tiling, maxCoreNum, DIREC_DEEPTH);
    uint32_t coreRealNeedNum = std::max(core_real_need_num_w, std::max(core_real_need_num_h, core_real_need_num_d));

    if (!SetTCubeTilingW(context, batch * channel, tiling, dataType)) {
        return ge::GRAPH_FAILED;
    }
    if (!SetTCubeTilingH(context, batch * channel, tiling, dataType)) {
        return ge::GRAPH_FAILED;
    }
    if (!SetTCubeTilingD(context, batch * channel, tiling, dataType)) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(coreRealNeedNum);
    context->SetScheduleMode(SCHEDULE_MODE);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] =
        GetWorkSpaceSize(tiling, batch * channel, coreRealNeedNum, data_type_size) + RESERVED_WORKSPACE_SIZE;

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ResizeUpsampleTrilinear)
    .Tiling(Tiling4UpsampleTrilinear);
} // namespace optiling
