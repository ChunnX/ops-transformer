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
 * \file add_lora_tiling.cpp
 * \brief
 */

#include "add_lora_tiling.h"
#include "log/log.h"
#include "op_host/tiling_util.h"

#include <cmath>
#include <cstring>

using namespace ge;

namespace optiling {

// cnt_seg/segbase 段行宽：SPEC 约束上界 W_MAX=32（编译期常量，非运行时 W——
// 与 op_kernel/add_lora.py 模块常量 W_MAX 逐字一致）。
constexpr int64_t W_MAX = 32;

// 32B 对齐（DESIGN §1：workspace 各段 32B 对齐下界；镜像 wrapper _align32）。
static inline int64_t Align32(int64_t n)
{
    return (n + 31) / 32 * 32;
}

// 64B 对齐（偏离 D8：lora 段基址提升用；仍满足 32B 下界；镜像 wrapper _align64）。
static inline int64_t Align64(int64_t n)
{
    return (n + 63) / 64 * 64;
}

ge::graphStatus AddLoraCompileInfo::ParamCheck(gert::TilingContext *context, bool hasWeightA, int64_t layerIdx,
                                               float scale, int64_t yOffset, int64_t ySliceSize, int64_t bSize,
                                               int64_t wSize, int64_t lSize, int64_t rSize, int64_t h1Size,
                                               int64_t h2Size, int64_t h3Size, int64_t xDim0, int64_t indicesDim0,
                                               int64_t waW, int64_t waL, int64_t waR, int64_t waH1)
{
    // —— 以下逐条镜像 SPEC §8 shape_constraints（源：custom/addlora/test_addlora.py
    //    wrapper 参数校验，Stage-4 已验证口径）——
    OP_CHECK_IF(bSize < 1 || bSize > 4096,
                OP_LOGE(context->GetNodeName(), "B must be in [1, 4096], but got %ld.", bSize),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(wSize < 1 || wSize > 32 || lSize < 1 || lSize > 32,
                OP_LOGE(context->GetNodeName(), "W and L must be in [1, 32], but got W=%ld, L=%ld.", wSize, lSize),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(h1Size > 131072 || h2Size > 131072 || h3Size > 131072,
                OP_LOGE(context->GetNodeName(),
                        "H1/H2/H3 must not exceed 131072, but got H1=%ld, H2=%ld, H3=%ld.", h1Size, h2Size, h3Size),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(xDim0 != bSize || indicesDim0 != bSize,
                OP_LOGE(context->GetNodeName(),
                        "y.dim0 == x.dim0 == indices.dim0 == B violated, but got B=%ld, x.dim0=%ld, "
                        "indices.dim0=%ld.",
                        bSize, xDim0, indicesDim0),
                return ge::GRAPH_PARAM_INVALID);

    // R ≤ 128（weightA 在场，双矩阵乘路径）或 R == H1（缺省，BGMV 路径）
    if (hasWeightA) {
        OP_CHECK_IF(waW != wSize || waL != lSize,
                    OP_LOGE(context->GetNodeName(),
                            "weightA/weightB (W, L) mismatch, but got weightA=(%ld, %ld), W=%ld, L=%ld.",
                            waW, waL, wSize, lSize),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(waR != rSize || waH1 != h1Size,
                    OP_LOGE(context->GetNodeName(),
                            "weightA (R, H1) mismatch, but got weightA=(%ld, %ld), R=%ld, H1=%ld.",
                            waR, waH1, rSize, h1Size),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(rSize > 128,
                    OP_LOGE(context->GetNodeName(),
                            "R <= 128 is required when weightA is present, but got R=%ld.", rSize),
                    return ge::GRAPH_PARAM_INVALID);
    } else {
        OP_CHECK_IF(rSize != h1Size,
                    OP_LOGE(context->GetNodeName(),
                            "R == H1 is required when weightA is absent (BGMV), but got R=%ld, H1=%ld.",
                            rSize, h1Size),
                    return ge::GRAPH_PARAM_INVALID);
    }

    // R/H1/H2/H3 16 对齐
    OP_CHECK_IF(rSize % 16 != 0,
                OP_LOGE(context->GetNodeName(), "R=%ld must be a multiple of 16.", rSize),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(h1Size % 16 != 0,
                OP_LOGE(context->GetNodeName(), "H1=%ld must be a multiple of 16.", h1Size),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(h2Size % 16 != 0,
                OP_LOGE(context->GetNodeName(), "H2=%ld must be a multiple of 16.", h2Size),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(h3Size % 16 != 0,
                OP_LOGE(context->GetNodeName(), "H3=%ld must be a multiple of 16.", h3Size),
                return ge::GRAPH_PARAM_INVALID);

    // 属性约束
    OP_CHECK_IF(layerIdx < 0 || layerIdx >= lSize,
                OP_LOGE(context->GetNodeName(), "0 <= layer_idx < L violated, but got layer_idx=%ld, L=%ld.",
                        layerIdx, lSize),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(std::isfinite(scale) == 0,
                OP_LOGE(context->GetNodeName(), "scale must be finite and non-NaN, but got %f.", scale),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(yOffset < 0 || yOffset >= h3Size,
                OP_LOGE(context->GetNodeName(), "0 <= y_offset < H3 violated, but got y_offset=%ld, H3=%ld.",
                        yOffset, h3Size),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(yOffset + h2Size > h3Size,
                OP_LOGE(context->GetNodeName(),
                        "y_offset + H2 <= H3 violated, but got y_offset=%ld, H2=%ld, H3=%ld.",
                        yOffset, h2Size, h3Size),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(ySliceSize != -1 && (ySliceSize < 0 || ySliceSize > h2Size),
                OP_LOGE(context->GetNodeName(),
                        "y_slice_size must be -1 or in [0, H2], but got y_slice_size=%ld, H2=%ld.",
                        ySliceSize, h2Size),
                return ge::GRAPH_PARAM_INVALID);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AddLoraCompileInfo::CheckShapeDims(gert::TilingContext *context, const gert::StorageShape *yShape,
                                                   const gert::StorageShape *xShape,
                                                   const gert::StorageShape *weightBShape,
                                                   const gert::StorageShape *indicesShape,
                                                   const gert::StorageShape *weightAShape)
{
    OP_CHECK_IF(yShape->GetStorageShape().GetDimNum() != 2,
                OP_LOGE(context->GetNodeName(), "y must be 2-D, but got %ldD.",
                        static_cast<int64_t>(yShape->GetStorageShape().GetDimNum())),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(xShape->GetStorageShape().GetDimNum() != 2,
                OP_LOGE(context->GetNodeName(), "x must be 2-D, but got %ldD.",
                        static_cast<int64_t>(xShape->GetStorageShape().GetDimNum())),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(weightBShape->GetStorageShape().GetDimNum() != 4,
                OP_LOGE(context->GetNodeName(), "weightB must be 4-D [W, L, H2, R], but got %ldD.",
                        static_cast<int64_t>(weightBShape->GetStorageShape().GetDimNum())),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(indicesShape->GetStorageShape().GetDimNum() != 1,
                OP_LOGE(context->GetNodeName(), "indices must be 1-D, but got %ldD.",
                        static_cast<int64_t>(indicesShape->GetStorageShape().GetDimNum())),
                return ge::GRAPH_PARAM_INVALID);
    if (weightAShape != nullptr) {
        OP_CHECK_IF(weightAShape->GetStorageShape().GetDimNum() != 4,
                    OP_LOGE(context->GetNodeName(), "weightA must be 4-D [W, L, R, H1], but got %ldD.",
                            static_cast<int64_t>(weightAShape->GetStorageShape().GetDimNum())),
                    return ge::GRAPH_PARAM_INVALID);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus AddLoraTilingFunc(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context->GetNodeName(), "GetPlatformInfo is nullptr."),
                return ge::GRAPH_PARAM_INVALID);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0,
                OP_LOGE(context->GetNodeName(), "num of core obtained is 0, aicNum=%u, aivNum=%u.", aicNum, aivNum),
                return ge::GRAPH_PARAM_INVALID);

    auto socVersion = ascendcPlatform.GetSocVersion();
    if (!Ops::Transformer::OpTiling::IsRegbaseSocVersion(context)) {
        OP_LOGE(context->GetNodeName(), "SOC Version[%d] is not support.", static_cast<int32_t>(socVersion));
        return ge::GRAPH_PARAM_INVALID;
    }

    OP_CHECK_IF(context->GetWorkspaceSizes(1) == nullptr,
                OP_LOGE(context->GetNodeName(), "workSpaceSize got from ge is nullptr."),
                return ge::GRAPH_PARAM_INVALID);

    // blockDim 语义（MIX kernel）：CalcTschBlockDim(sliceNum=aivNum, aicCoreNum=aicNum,
    // aivCoreNum=aivNum) 在 950PR（AIC:AIV = 28:56 = 1:2）上返回 28——与 Stage-4 验证的
    // python launch `add_lora[None, BLOCK_DIM=28, key]` 同值。kernel 侧（生成入口为
    // KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)）：AIC 上 get_block_num() == 28；
    // AIV 上 aiv_num = get_block_num() * get_subblock_num() == 28 * 2 == 56。两侧取值与
    // DESIGN §5 分核策略一致（本 kernel 内切分一律以 get_block_num() 运行时值为准）。
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);

    // Extract attrs (declaration order: layer_idx=0, scale=1, y_offset=2, y_slice_size=3).
    auto attrs = context->GetAttrs();
    int64_t layerIdx = 0;
    float scale = 0.001f;
    int64_t yOffset = 0;
    int64_t ySliceSize = -1;
    if (attrs != nullptr) {
        auto layerIdxPtr = attrs->GetAttrPointer<int64_t>(0);
        auto scalePtr = attrs->GetAttrPointer<float>(1);
        auto yOffsetPtr = attrs->GetAttrPointer<int64_t>(2);
        auto ySliceSizePtr = attrs->GetAttrPointer<int64_t>(3);
        if (layerIdxPtr != nullptr) {
            layerIdx = *layerIdxPtr;
        }
        if (scalePtr != nullptr) {
            scale = *scalePtr;
        }
        if (yOffsetPtr != nullptr) {
            yOffset = *yOffsetPtr;
        }
        if (ySliceSizePtr != nullptr) {
            ySliceSize = *ySliceSizePtr;
        }
    }

    // Extract runtime dimensions from input shapes (input order: y=0, x=1, weightB=2,
    // indices=3; optional weightA=4).
    auto yShape = context->GetInputShape(0);
    auto xShape = context->GetInputShape(1);
    auto weightBShape = context->GetInputShape(2);
    auto indicesShape = context->GetInputShape(3);
    OP_CHECK_IF(yShape == nullptr || xShape == nullptr || weightBShape == nullptr || indicesShape == nullptr,
                OP_LOGE(context->GetNodeName(), "y, x, weightB or indices shape is null."),
                return ge::GRAPH_PARAM_INVALID);
    auto weightAShape = context->GetOptionalInputShape(4);
    const bool hasWeightA = (weightAShape != nullptr);

    // dtype 检查（SPEC 单档：y/x/weightB/weightA FP16、indices INT32）。
    auto yDesc = context->GetInputDesc(0);
    auto xDesc = context->GetInputDesc(1);
    auto weightBDesc = context->GetInputDesc(2);
    auto indicesDesc = context->GetInputDesc(3);
    OP_CHECK_IF(yDesc == nullptr || xDesc == nullptr || weightBDesc == nullptr || indicesDesc == nullptr,
                OP_LOGE(context->GetNodeName(), "y, x, weightB or indices desc is null."),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(yDesc->GetDataType() != ge::DT_FLOAT16 || xDesc->GetDataType() != ge::DT_FLOAT16,
                OP_LOGE(context->GetNodeName(),
                        "y and x dtype must be float16, but got y=%d, x=%d.",
                        static_cast<int32_t>(yDesc->GetDataType()), static_cast<int32_t>(xDesc->GetDataType())),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(weightBDesc->GetDataType() != ge::DT_FLOAT16,
                OP_LOGE(context->GetNodeName(), "weightB dtype must be float16, but got %d.",
                        static_cast<int32_t>(weightBDesc->GetDataType())),
                return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(indicesDesc->GetDataType() != ge::DT_INT32,
                OP_LOGE(context->GetNodeName(), "indices dtype must be int32, but got %d.",
                        static_cast<int32_t>(indicesDesc->GetDataType())),
                return ge::GRAPH_PARAM_INVALID);
    if (hasWeightA) {
        auto weightADesc = context->GetOptionalInputDesc(4);
        OP_CHECK_IF(weightADesc == nullptr,
                    OP_LOGE(context->GetNodeName(), "weightA desc is null."),
                    return ge::GRAPH_PARAM_INVALID);
        OP_CHECK_IF(weightADesc->GetDataType() != ge::DT_FLOAT16,
                    OP_LOGE(context->GetNodeName(), "weightA dtype must be float16, but got %d.",
                            static_cast<int32_t>(weightADesc->GetDataType())),
                    return ge::GRAPH_PARAM_INVALID);
    }

    auto outputShape = context->GetOutputShape(0);
    OP_CHECK_IF(outputShape == nullptr, OP_LOGE(context->GetNodeName(), "output shape is null."),
                return ge::GRAPH_PARAM_INVALID);

    auto dimRet = AddLoraCompileInfo::CheckShapeDims(context, yShape, xShape, weightBShape, indicesShape,
                                                     weightAShape);
    OP_CHECK_IF(dimRet != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "AddLoraCompileInfo::CheckShapeDims failed."),
                return ge::GRAPH_PARAM_INVALID);

    int64_t bSize = yShape->GetStorageShape().GetDim(0);
    int64_t h3Size = yShape->GetStorageShape().GetDim(1);
    int64_t xDim0 = xShape->GetStorageShape().GetDim(0);
    int64_t h1Size = xShape->GetStorageShape().GetDim(1);
    int64_t wSize = weightBShape->GetStorageShape().GetDim(0);
    int64_t lSize = weightBShape->GetStorageShape().GetDim(1);
    int64_t h2Size = weightBShape->GetStorageShape().GetDim(2);
    int64_t rSize = weightBShape->GetStorageShape().GetDim(3);
    int64_t indicesDim0 = indicesShape->GetStorageShape().GetDim(0);
    int64_t waW = 0;
    int64_t waL = 0;
    int64_t waR = 0;
    int64_t waH1 = 0;
    if (hasWeightA) {
        waW = weightAShape->GetStorageShape().GetDim(0);
        waL = weightAShape->GetStorageShape().GetDim(1);
        waR = weightAShape->GetStorageShape().GetDim(2);
        waH1 = weightAShape->GetStorageShape().GetDim(3);
    }

    // Unified parameter validation (SPEC §8 full mirror).
    auto ret = AddLoraCompileInfo::ParamCheck(context, hasWeightA, layerIdx, scale, yOffset, ySliceSize, bSize, wSize,
                                              lSize, rSize, h1Size, h2Size, h3Size, xDim0, indicesDim0, waW, waL,
                                              waR, waH1);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "AddLoraCompileInfo::ParamCheck failed."),
                return ge::GRAPH_PARAM_INVALID);

    // 输出 shape 防御性核对（y_out ≡ y in-place，infershape 产 [B, H3]）。
    OP_CHECK_IF(outputShape->GetStorageShape().GetDimNum() != 2 ||
                    outputShape->GetStorageShape().GetDim(0) != bSize ||
                    outputShape->GetStorageShape().GetDim(1) != h3Size,
                OP_LOGE(context->GetNodeName(),
                        "y_out shape must equal y shape [B, H3] = [%ld, %ld].", bSize, h3Size),
                return ge::GRAPH_PARAM_INVALID);

    // TilingKey：唯一字段 addLoraFlag（AddLoraTilingKey bits=1，AddLoraTilingKey_tilingkey.h
    // 生成 ASCENDC_TPL 两档选择）。weightA 在场（optional input index 4）→ 1（双矩阵乘
    // 路径，R ≤ 128）；缺省 → 0（BGMV 单矩阵乘路径，R == H1）。判定方式与样板对 optional
    // 的判定一致（GetOptionalInputShape(4) != nullptr）。
    uint64_t tilingKey = static_cast<uint64_t>(hasWeightA ? 1 : 0);
    context->SetTilingKey(tilingKey);

    // ── TilingData 填充（镜像 test_addlora.py wrapper 的纯 Python 推导）──
    // AddLoraTiling comes from the pypto kernel (.py dataclass) via force-include.
    AddLoraTiling *tilingData = context->GetTilingData<AddLoraTiling>();
    tilingData->B = bSize;
    tilingData->W = wSize;
    tilingData->L = lSize;
    tilingData->R = rSize;
    tilingData->H1 = h1Size;
    tilingData->H2 = h2Size;
    tilingData->H3 = h3Size;
    tilingData->layer_idx = layerIdx;

    // scale_bits：scale 的 IEEE-754 binary32 位模式以 int32 承载（SPEC「纯 int 字段」冻结
    // 契约；镜像 wrapper `struct.unpack("!i", struct.pack("!f", scale))[0]`——pack/unpack
    // 两次字节序反转相消，即本征位重解释）。kernel 入口 restore_scale_vf 一次性还原。
    float scaleF = scale;
    int32_t scaleBits = 0;
    memcpy(&scaleBits, &scaleF, sizeof(scaleBits));
    tilingData->scale_bits = static_cast<int64_t>(scaleBits);

    tilingData->y_offset = yOffset;
    // s 规约：y_slice_size == -1 → H2；0 ≤ s ≤ H2
    tilingData->s = (ySliceSize == -1) ? h2Size : ySliceSize;
    // cnt_seg/segbase 段表容量 = 2 * blockDim（= NSEG_MAX = 56 @ 950PR；host 已知 block_dim）
    const int64_t nsegCap = 2 * static_cast<int64_t>(blockDim);
    tilingData->nseg_cap = nsegCap;

    // ── workspace 段偏移（逐字段镜像 wrapper _ws_offsets 的数学；各段 32B 对齐，
    //    lora 段 64B 对齐——偏离 D8）──
    // x_g 段 [B, H1] FP16（第一段，偏移 0）
    const int64_t wsXgOff = 0;
    // t 段 [B, R] FP16（仅 addLoraFlag=1 使用；段大小 B*R*2 字节）
    const int64_t wsTOff = Align32(bSize * h1Size * 2);
    // lora 段 [B, H2] FP32：基址 64B 对齐（D8——段基址 ≡ 32 (mod 64) 且 store 窗覆盖物理
    // 末行时段末 32B 不被写入的硬件行为，64B 对齐后触发类整体消除；H2 16 对齐使 B*H2*4
    // 为 64B 整数倍、nseg_cap*W_MAX*4 = 7168B 亦为 64B 整数倍，故后续段等距顺延不交叠）
    const int64_t wsLoraOff = Align64(wsTOff + bSize * rSize * 2);
    // cnt_seg 段 [nseg_cap, W_MAX] INT32（段计数，行 stride=W_MAX——用编译期上界而非运行时 W）
    const int64_t wsCntOff = Align32(wsLoraOff + bSize * h2Size * 4);
    // segbase 段 [nseg_cap, W_MAX] INT32（段基址）
    const int64_t wsSegbaseOff = Align32(wsCntOff + nsegCap * W_MAX * 4);
    // rowpos 段 [B] INT32（段内 rank）
    const int64_t wsRowposOff = Align32(wsSegbaseOff + nsegCap * W_MAX * 4);
    // tbl 段（ops 适配 A3：原独立 tbl tensor 折叠进 workspace；布局 offsets[W+1] |
    // m_start[W+1] | inv_map[B]，共 2*(W+1)+B 个 INT32，32B 对齐）
    const int64_t wsTblOff = Align32(wsRowposOff + bSize * 4);
    const int64_t tblBytes = (2 * (wSize + 1) + bSize) * 4;
    // 用户段总量（wrapper 无额外余量——如实镜像，不加 1MB 余量）
    const int64_t wsUserBytes = Align32(wsTblOff + tblBytes);

    tilingData->ws_xg_off = wsXgOff;
    tilingData->ws_t_off = wsTOff;
    tilingData->ws_lora_off = wsLoraOff;
    tilingData->ws_cnt_off = wsCntOff;
    tilingData->ws_segbase_off = wsSegbaseOff;
    tilingData->ws_rowpos_off = wsRowposOff;
    tilingData->ws_tbl_off = wsTblOff;

    // Workspace：系统保留段 + 用户段总量。kernel 收到的 workspace 指针经生成入口的
    // AscendC::GetUserWorkspace 路由到用户段基址（上述偏移均相对用户段基址）。
    size_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    workspaceSize += static_cast<size_t>(wsUserBytes);
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    OP_LOGI(context->GetNodeName(),
            "AddLora tiling completed: blockDim=%u, aivNum=%u, aicNum=%u, hasWeightA=%d, tilingKey=%lu, "
            "B=%ld, W=%ld, L=%ld, R=%ld, H1=%ld, H2=%ld, H3=%ld, layer_idx=%ld, y_offset=%ld, "
            "y_slice_size=%ld, s=%ld, scale_bits=%d, nseg_cap=%ld, ws_xg_off=%ld, ws_t_off=%ld, "
            "ws_lora_off=%ld, ws_cnt_off=%ld, ws_segbase_off=%ld, ws_rowpos_off=%ld, ws_tbl_off=%ld, "
            "workspaceSize=%zu.",
            blockDim, aivNum, aicNum, static_cast<int32_t>(hasWeightA), tilingKey, bSize, wSize, lSize, rSize, h1Size,
            h2Size, h3Size, layerIdx, yOffset, ySliceSize, tilingData->s, scaleBits, nsegCap, wsXgOff, wsTOff,
            wsLoraOff, wsCntOff, wsSegbaseOff, wsRowposOff, wsTblOff, workspaceSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAddLora(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AddLora)
    .Tiling(AddLoraTilingFunc)
    .TilingParse<AddLoraCompileInfo>(TilingParseForAddLora);

} // namespace optiling
