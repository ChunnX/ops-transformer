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
 * \file ffn_arch35_tiling.cpp
 * \brief FFN arch35 (ascend950) fused 路径 tiling：布局识别 + up/down 融合 tiling。
 *        从 ffn_tiling.cpp 拆出（行为不变），legacy 路径仍在 ffn_tiling.cpp。
 */

#include "ffn_arch35_tiling.h"

#include "ffn_layout.h"

#include <algorithm>
#include <climits>

#include "err/ops_err.h"
#include "log/log.h"
#include "register/tilingdata_base.h"

#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_aswt_tiling.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_streamk_tiling.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_data.h"
#include "../op_kernel/ffn_arch35_tiling_key.h"

using namespace ge;
using namespace AscendC;

namespace optiling {
namespace {
using namespace optiling;
using namespace optiling::matmul_v3_advanced;

// 官方 ops-nn mat_mul_v3 Basic ASWT tiling：与 fused_mat_mul 同源。
// 只计算 runInfo，不写 context（两个 matmul 合并在同一个 FFN tiling 里）。
class FfnMatMulV3Tiling : public MatMulV3BasicAswtTiling {
public:
    FfnMatMulV3Tiling(gert::TilingContext *ctx, MatMulTilingCfg &cfg, bool allowFullLoad = true)
        : MatMulV3BasicAswtTiling(ctx, cfg), allowFullLoad_(allowFullLoad) {}

    ge::graphStatus ComputeOnly()
    {
        ge::graphStatus ret = GetShapeAttrsInfo();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        if (!IsCapable()) {
            return ge::GRAPH_PARAM_INVALID;
        }
        ret = DoOpTiling();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        return AdjustOpTiling();
    }

    ge::graphStatus GetBasicTiling(MatMulV3BasicTilingData &out) const
    {
        ge::graphStatus ret = GetTilingDataProcess(out);
        if (ret == ge::GRAPH_SUCCESS) {
            // 官方 DoOpTiling 已按资格算出 fullLoad_（0/1/2），透传给 kernel 分派全载变体
            out.fullLoad = static_cast<uint8_t>(fullLoad_);
        }
        return ret;
    }

protected:
    ge::graphStatus DoOpTiling() override
    {
        // up 的全载不可用：官方全载内核不支持 gelu/silu 融合（cmct 全载 block 白名单仅
        // 无融合/ADD/MUL/RELU），host 若按官方资格给 up 下发全载 tiling、kernel 会以
        // 非全载内核执行 → 结果错配。故 up 强制关全载，仅 down 允许官方资格。
        // 走官方"非全载"分支重算 stepKa/depthA1/B1（对齐 MatMulV3BasicAswtTiling::DoOpTiling
        // 的 else 分支），保证关掉全载后 tiling 参数与非全载内核一致。
        const bool forceOff = !allowFullLoad_;
        if (forceOff) {
            MatMulV3AswTiling::DoOpTiling();
            isSlice_ = MatMulV3TilingHelper::IsSelfNonContiguous(context_);
            l0C2Out_ = MatMulV3TilingHelper::GetL0C2Out(compileInfo_, args_, runInfo_);
            fullLoad_ = MatMulV3FullLoad::NONE_FULL_LOAD;
            if (l0C2Out_ == MatMulV3L0C2Out::ON_THE_FLY) {
                uint64_t remainSizeForAL1BL1 = args_.hasBias ?
                    (compileInfo_.l1Size - BIAS_TABLE_NUM * DATA_SIZE_FP32) : compileInfo_.l1Size;
                runInfo_.stepKa = remainSizeForAL1BL1 / NUM_TWO / ((runInfo_.baseM + runInfo_.baseN) * runInfo_.baseK) /
                                  args_.aDtypeSize;
                runInfo_.stepKb = runInfo_.stepKa;
                runInfo_.depthA1 = runInfo_.stepKa * DB_SIZE;
                runInfo_.depthB1 = runInfo_.stepKb * DB_SIZE;
                CheckFp32SplitK();
                CheckApiLevelAndModel();
            } else {
                CheckApiLevelAndModel();
            }
            return ge::GRAPH_SUCCESS;
        }
        return MatMulV3BasicAswtTiling::DoOpTiling();
    }

private:
    bool allowFullLoad_ = true;
};

// 官方 ops-nn mat_mul_v3 Basic StreamK tiling（split-K 场景），同样只算 runInfo 不写 context。
class FfnMatMulV3StreamKTiling : public MatMulV3BasicStreamKTiling {
public:
    FfnMatMulV3StreamKTiling(gert::TilingContext *ctx, MatMulTilingCfg &cfg)
        : MatMulV3BasicStreamKTiling(ctx, cfg) {}

    ge::graphStatus ComputeOnly()
    {
        ge::graphStatus ret = GetShapeAttrsInfo();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        if (!IsCapable()) {
            return ge::GRAPH_PARAM_INVALID;
        }
        ret = DoOpTiling();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        return AdjustOpTiling();
    }

    ge::graphStatus GetBasicTiling(MatMulV3BasicTilingData &out) const
    {
        return GetTilingDataProcess(out);
    }
};
} // namespace

bool FFNArch35Tiling::DetectLayout(const gert::TilingContext *context,
                                   const gert::StorageShape *weight1Shape,
                                   const gert::StorageShape *weight2Shape,
                                   int64_t xK, bool isSwiglu, bool &isLinear)
{
    isLinear = false;
    if (weight1Shape == nullptr || weight2Shape == nullptr ||
        weight1Shape->GetStorageShape().GetDimNum() < 2 || weight2Shape->GetStorageShape().GetDimNum() < 2) {
        return false;
    }
    // 维度统一取最后两维（与 infershape/aclnn 一致，为将来 >2 维权重留门）。
    const auto &w1Shape = weight1Shape->GetStorageShape();
    const auto &w2Shape = weight2Shape->GetStorageShape();
    const int64_t w1Dim0 = w1Shape.GetDim(w1Shape.GetDimNum() - 2);
    const int64_t w1Dim1 = w1Shape.GetDim(w1Shape.GetDimNum() - 1);
    const int64_t w2Dim0 = w2Shape.GetDim(w2Shape.GetDimNum() - 2);
    const int64_t w2Dim1 = w2Shape.GetDim(w2Shape.GetDimNum() - 1);

    // 布局识别走公共规则（ffn_layout.h）；swiglu 隐藏宽=w1 边长一半，
    // isSwiglu 由调用方按 ActiveType 枚举传入。
    const ffnlayout::FfnLayout layout =
        ffnlayout::FfnDetectLayout(w1Dim0, w1Dim1, w2Dim0, w2Dim1, xK, isSwiglu);
    if (layout == ffnlayout::FfnLayout::INVALID) {
        OP_LOGE(context, "FFN arch35 weight1 shape [%ld, %ld] does not match x K=%ld",
                static_cast<long>(w1Dim0), static_cast<long>(w1Dim1), static_cast<long>(xK));
        return false;
    }
    isLinear = (layout == ffnlayout::FfnLayout::LINEAR);
    return true;
}

ge::graphStatus FFNArch35Tiling::Tiling(gert::TilingContext *context,
                                        const FFNCompileInfo *compileInfoPtr,
                                        FFNTilingData &tilingData,
                                        const FFNArch35TilingParams &params)
{
    // arch35 fused 路径约束：ascend950、单 expert、bf16、gelu/silu/swiglu；
    // bias1/bias2 两者同时有或同时无（swiglu 模型场景为无 bias）。
    // 权重支持 canonical [K, N] 与 linear [N, K]（out, in）两种布局（GetInputShape 已识别）；
    // bias 支持 bf16 / float32。不满足时由调用方（ascend950 分支）直接返回失败。
    const uint32_t activeType = params.activeType;
    const uint32_t &expertNum = params.expertNum;
    const uint32_t &bs = params.bs;
    const uint32_t &k1 = params.k1;
    const uint32_t &n1 = params.n1;
    const uint32_t &n2 = params.n2;
    const bool &isFfnTransB_ = params.isFfnTransB;
    const ge::DataType &xDataType = params.xDataType;
    const uint32_t &xDataTypeSize = params.xDataTypeSize;

    if (compileInfoPtr->socVersion != platform_ascendc::SocVersion::ASCEND950 || expertNum > 1 ||
        (xDataType != ge::DT_BF16 && xDataType != ge::DT_FLOAT16) ||
        (activeType != static_cast<uint32_t>(ActiveType::GELU) &&
         activeType != static_cast<uint32_t>(ActiveType::SILU) &&
         activeType != static_cast<uint32_t>(ActiveType::SWIGLU))) {
        return ge::GRAPH_NOT_CHANGED;
    }
    const bool isSwiglu = (activeType == static_cast<uint32_t>(ActiveType::SWIGLU));
    const bool isSilu = (activeType == static_cast<uint32_t>(ActiveType::SILU));
    if (isSwiglu && !isFfnTransB_) {
        // canonical [K,2H] 的 swiglu 需要 B 右半列切片，kernel 暂未实现，先拒绝
        return ge::GRAPH_NOT_CHANGED;
    }
    auto bias1Desc = context->GetOptionalInputDesc(BIAS1_INDEX);
    auto bias2Desc = context->GetOptionalInputDesc(BIAS2_INDEX);
    const bool hasBias1 = (bias1Desc != nullptr);
    const bool hasBias2 = (bias2Desc != nullptr);
    if (hasBias1 != hasBias2) {
        // 只支持两者同时有/同时无，混合情况回退
        return ge::GRAPH_NOT_CHANGED;
    }
    ge::DataType biasDtype = xDataType; // 无 bias 时按 x dtype 下发，kernel 按字段分派
    bool biasIsBf16 = (xDataType == ge::DT_BF16);
    bool biasIsFp16 = (xDataType == ge::DT_FLOAT16);
    if (hasBias1) {
        biasDtype = bias1Desc->GetDataType();
        if (biasDtype != ge::DT_BF16 && biasDtype != ge::DT_FLOAT16 && biasDtype != ge::DT_FLOAT) {
            OP_LOGE(context, "FFN arch35 only supports bf16/fp16/float32 bias, got %d",
                    static_cast<int>(biasDtype));
            return ge::GRAPH_FAILED;
        }
        if (bias2Desc->GetDataType() != biasDtype) {
            OP_LOGE(context, "FFN arch35 bias1/bias2 dtype mismatch");
            return ge::GRAPH_FAILED;
        }
        biasIsBf16 = (biasDtype == ge::DT_BF16);
        biasIsFp16 = (biasDtype == ge::DT_FLOAT16);
        // fp16/bf16 x 允许同 dtype 或 fp32 bias；fp16 x + bf16 bias / bf16 x + fp16 bias 拒绝
        if ((xDataType == ge::DT_FLOAT16 && biasDtype == ge::DT_BF16) ||
            (xDataType == ge::DT_BF16 && biasDtype == ge::DT_FLOAT16) ||
            // fp16 x 仅支持 fp16 bias；bf16 x 支持 bf16/fp32 bias
            (xDataType == ge::DT_FLOAT16 && biasDtype == ge::DT_FLOAT)) {
            OP_LOGE(context, "FFN arch35 unsupported bias dtype (fp16 x: fp16 only; bf16 x: bf16/float32)");
            return ge::GRAPH_FAILED;
        }
    }

    // 任意 H（含非对齐）均可走 fused 路径：fp32 gate 走 DataCopyPad 32B 自动对齐，
    // FusionSwiglu 帧已 16 对齐。

    using namespace matmul_v3_advanced;
    MatmulV3CompileInfo mmCompileInfo{};
    OP_CHECK_IF(InitCompileInfo(context->GetPlatformInfo(), &mmCompileInfo) != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "init mat_mul_v3 compile info failed"),
                return ge::GRAPH_FAILED);

    auto computeOneStage = [&](uint64_t m, uint64_t k, uint64_t n, bool isDown, bool useStreamK,
                               MatMulV3BasicTilingData &out) -> ge::graphStatus {
        MatMulV3Args args{};
        args.opName = "FFN";
        args.isATrans = false;
        args.isBTrans = isFfnTransB_; // linear 布局 [N, K] 时 kernel 内做 transB
        args.isHf32 = false;
        args.hasBias = hasBias1;
        args.hasScale = false;
        args.aType = xDataType;
        args.bType = xDataType;
        args.cType = xDataType;
        args.x3Type = xDataType;
        args.biasType = biasDtype;
        args.aFormat = ge::FORMAT_ND;
        args.bFormat = ge::FORMAT_ND;
        args.outFormat = ge::FORMAT_ND;
        args.mValue = m;
        args.mOriValue = m;
        args.kValue = k;
        args.nValue = n;
        args.nOriValue = n;
        args.aDtypeSize = xDataTypeSize;
        args.bDtypeSize = xDataTypeSize;
        args.fusedOpType = 0;
        args.batchX3 = 1;
        args.hasX3Input = false;
        args.isForceGrpAccForFp32 = false;
        args.isAvoidTensorApi = false;
        args.batchInfo = nullptr;
        args.preferL0cDB2 = false;
        args.preferL0cMSplitDB2 = false;
        // swiglu 单 matmul（n=n1=2H）：tiling 搜索内强制 baseN 32 对齐 + 尾波禁 N 拆分，
        // 使 M=640（2H=9728，自然搜索 baseN=272 仅 16 对齐）也能走两段式 single 路径。
        // 仅影响 n==n1 的 up 计算，旧三段（n=H）与 gelu/silu/down 不受影响。
        args.swigluSingleNAlign32 = isSwiglu && n == n1;
        args.preferNoMSplit = false;
        MatMulTilingCfg cfg(false, &mmCompileInfo, &args, nullptr);
        if (useStreamK) {
            // 官方 StreamK split-K tiling（对齐 built-in MatMulV3 的 down 切分）
            FfnMatMulV3StreamKTiling tiling(context, cfg);
            ge::graphStatus ret = tiling.ComputeOnly();
            if (ret != ge::GRAPH_SUCCESS) {
                return ret;
            }
            return tiling.GetBasicTiling(out);
        }
        {
            // 仅 down 允许官方 L1 全载（up 的融合内核不支持全载，见 FfnMatMulV3Tiling::DoOpTiling）
            FfnMatMulV3Tiling tiling(context, cfg, isDown);
            ge::graphStatus ret = tiling.ComputeOnly();
            if (ret != ge::GRAPH_SUCCESS) {
                OP_LOGE(context, "FFN arch35 mat_mul_v3 tiling failed: m=%lu k=%lu n=%lu ret=%d", m, k, n,
                        static_cast<int>(ret));
                return ret;
            }
            return tiling.GetBasicTiling(out);
        }
    };

    MatMulV3BasicTilingData up{};
    MatMulV3BasicTilingData down{};
    const uint32_t hiddenCols = isSwiglu ? (n1 / 2) : n1; // swiglu: raw 2H -> hidden H
    const uint32_t downK = hiddenCols;
    // swiglu：优先单 matmul（N=2H，B 装载交错 gate/up）；约束不满足则回退旧双半宽。
    // 约束：无 N 尾块且 tile 32 对齐（epilogue 以 32 列为一对）。
    uint32_t swigluSingle = 0; // 0=回退两段式(gate+up+down)，1=single(2H)
    OP_CHECK_IF(computeOneStage(bs, k1, n1, false, false, up) != ge::GRAPH_SUCCESS,
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "compute up matmul tiling failed"),
                return ge::GRAPH_FAILED);
    if (isSwiglu) {
        // 半 tile 布局：要求 L0 tile N == L1 tile N（g/u 各占一半且在同一 tile），
        // 且整 tile 32 对齐（半宽 bf16 输出行须 32B 对齐）。
        // 2026-08-17：放开 n1%32==0——非 32 对齐的尾块（宽 16k+16）由 kernel 按
        // padded-half 处理：mmad/UB 帧宽 Align(尾块,32)，epilogue 只写有效半宽。
        // nTailCnt 是"尾块分摊数"（默认 1，非宽度尾块），不参与判断。
        swigluSingle = (up.n == n1 && up.nL1 % 32 == 0 && up.baseN % 32 == 0 && up.nL1 == up.baseN);
        if (!swigluSingle) {
            OP_CHECK_IF(computeOneStage(bs, k1, hiddenCols, false, false, up) != ge::GRAPH_SUCCESS,
                        OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "compute up matmul tiling failed"),
                        return ge::GRAPH_FAILED);
        }
    }
    bool downStreamK = false;
    if (!isSwiglu) {
        // gelu/silu down：先算官方 basic 切分作基准，再决定是否切换官方 StreamK split-K。
        OP_CHECK_IF(computeOneStage(bs, downK, n2, true, false, down) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "compute down matmul tiling failed"),
                    return ge::GRAPH_FAILED);
        const MatMulV3BasicTilingData basicDown = down;
        // 结构门控：只有当 basic 把 M 拆成多块（B/w2 按 M 块重复搬运）时，streamk 才可能
        // 通过"单 M 块"消除重复搬运；且 streamk 必须真的用单 M 块覆盖整个 M。
        // 否则 split-K 归约 + AIV epilogue 的开销超过收益。
        const uint64_t fullM = ((static_cast<uint64_t>(bs) + 15U) / 16U) * 16U;
        if (basicDown.baseM < fullM) {
            MatMulV3BasicTilingData skDown{};
            ge::graphStatus skRet = computeOneStage(bs, downK, n2, true, true, skDown);
            if (skRet == ge::GRAPH_SUCCESS && skDown.baseM >= fullM) {
                down = skDown;
                downStreamK = true;
                OP_LOGI(context, "FFN down uses official StreamK split-K (usedCore=%u skSingleCoreK=%u)",
                        down.usedCoreNum, down.skSingleCoreK);
            } else if (skRet != ge::GRAPH_SUCCESS && skRet != ge::GRAPH_PARAM_INVALID) {
                OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "compute down stream-k tiling failed");
                return ge::GRAPH_FAILED;
            }
        }
    } else {
        OP_CHECK_IF(computeOneStage(bs, downK, n2, true, false, down) != ge::GRAPH_SUCCESS,
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "compute down matmul tiling failed"),
                    return ge::GRAPH_FAILED);
    }

    tilingData.set_v1UpUsedCoreNum(up.usedCoreNum);
    tilingData.set_v1UpM(up.m);
    tilingData.set_v1UpN(up.n);
    tilingData.set_v1UpK(up.k);
    tilingData.set_v1UpML1(up.mL1);
    tilingData.set_v1UpNL1(up.nL1);
    tilingData.set_v1UpKL1(up.kL1);
    tilingData.set_v1UpBaseM(up.baseM);
    tilingData.set_v1UpBaseN(up.baseN);
    tilingData.set_v1UpBaseK(up.baseK);
    tilingData.set_v1UpSkSingleCoreK(up.skSingleCoreK);
    tilingData.set_v1UpMTailCnt(up.mTailCnt);
    tilingData.set_v1UpNTailCnt(up.nTailCnt);
    tilingData.set_v1UpMBaseTailSplitCnt(up.mBaseTailSplitCnt);
    tilingData.set_v1UpNBaseTailSplitCnt(up.nBaseTailSplitCnt);
    tilingData.set_v1UpMTailMain(up.mTailMain);
    tilingData.set_v1UpNTailMain(up.nTailMain);
    tilingData.set_v1UpIsHf32(up.mmadParam);
    tilingData.set_v1UpL1BufferNum(up.l1BufferNum);
    tilingData.set_v1UpL0cDB(up.l0cDB);
    tilingData.set_v1UpUbDB(up.ubDB);
    tilingData.set_v1UpFullLoad(up.fullLoad);

    tilingData.set_v1DownUsedCoreNum(down.usedCoreNum);
    tilingData.set_v1DownM(down.m);
    tilingData.set_v1DownN(down.n);
    tilingData.set_v1DownK(down.k);
    tilingData.set_v1DownML1(down.mL1);
    tilingData.set_v1DownNL1(down.nL1);
    tilingData.set_v1DownKL1(down.kL1);
    tilingData.set_v1DownBaseM(down.baseM);
    tilingData.set_v1DownBaseN(down.baseN);
    tilingData.set_v1DownBaseK(down.baseK);
    tilingData.set_v1DownSkSingleCoreK(down.skSingleCoreK);
    tilingData.set_v1DownMTailCnt(down.mTailCnt);
    tilingData.set_v1DownNTailCnt(down.nTailCnt);
    tilingData.set_v1DownMBaseTailSplitCnt(down.mBaseTailSplitCnt);
    tilingData.set_v1DownNBaseTailSplitCnt(down.nBaseTailSplitCnt);
    tilingData.set_v1DownMTailMain(down.mTailMain);
    tilingData.set_v1DownNTailMain(down.nTailMain);
    tilingData.set_v1DownIsHf32(down.mmadParam);
    tilingData.set_v1DownL1BufferNum(down.l1BufferNum);
    tilingData.set_v1DownL0cDB(down.l0cDB);
    tilingData.set_v1DownUbDB(down.ubDB);
    tilingData.set_v1DownFullLoad(down.fullLoad);

    // swiglu：user workspace 起始处放 gate fp32 [M, H]（供 up 半块 epilogue 读取），
    // hidden bf16 [M, H] 放在其后；gelu/silu：起始处留给 fused gelu mix kernel
    // 的 fp32 workspace（仿官方 m*n*4），hidden bf16 [M, N1] 放在其后。
    // swiglu 两段式：user workspace 起始处放 gate fp32 [M,H]；
    // gelu/silu：起始处留给 fused up kernel 的 fp32 workspace（仿官方 m*n*4）。
    uint32_t hiddenOffset = 0;
    if (isSwiglu && !swigluSingle) {
        hiddenOffset = (static_cast<uint32_t>(bs) * hiddenCols * sizeof(float) + 127) & ~127U;
    } else if (!isSwiglu) {
        hiddenOffset = (static_cast<uint32_t>(bs) * n1 * 4 + 127) & ~127U;
    }
    tilingData.set_v1HiddenOffset(hiddenOffset);
    tilingData.set_v1HiddenRows(bs);
    tilingData.set_v1HiddenCols(hiddenCols);
    tilingData.set_v1IsFp16(xDataType == ge::DT_FLOAT16 ? 1 : 0);
    tilingData.set_v1BiasIsBf16(biasIsBf16 ? 1 : 0);
    tilingData.set_v1BiasIsFp16(biasIsFp16 ? 1 : 0);
    tilingData.set_v1HasBias(hasBias1 ? 1 : 0);
    tilingData.set_v1TransB(isFfnTransB_ ? 1 : 0);
    tilingData.set_v1SwigluSingle(swigluSingle);
    // up/down 都用 cmct Basic kernel，blockDim 用官方 tiling 的核数
    const uint32_t blockDim = std::max(up.usedCoreNum, down.usedCoreNum);
    context->SetBlockDim(blockDim);
    // TilingKey 模板化：DTYPE(bf16/fp16) × ACT(gelu/silu/swiglu) × MODE(basic/streamK)
    const uint8_t tplDtype = (xDataType == ge::DT_FLOAT16) ? FFN_TPL_DTYPE_FP16 : FFN_TPL_DTYPE_BF16;
    uint8_t tplAct = FFN_TPL_ACT_GELU;
    if (activeType == static_cast<uint32_t>(ActiveType::SILU)) {
        tplAct = FFN_TPL_ACT_SILU;
    } else if (activeType == static_cast<uint32_t>(ActiveType::SWIGLU)) {
        tplAct = FFN_TPL_ACT_SWIGLU;
    }
    const uint8_t tplMode = downStreamK ? FFN_TPL_MODE_STREAMK : FFN_TPL_MODE_BASIC;
    const uint64_t tilingKey = GET_TPL_TILING_KEY(tplDtype, tplAct, tplMode);
    context->SetTilingKey(tilingKey);

    size_t *workspaces = context->GetWorkspaceSizes(1);
    if (downStreamK) {
        // StreamK split-K：分片归约区替换原 down fp32 区。
        // 官方布局：每个 (空间tile,K路) 一个固定 256×256 fp32 槽（kernel BLOCK_BASE_M/N=256），
        // 按 aicNum 个槽兜底（官方 GetWorkspaceSize = aicNum*256*256*4）。
        workspaces[0] = compileInfoPtr->sysWorkspaceSize + hiddenOffset +
                        static_cast<size_t>(bs) * hiddenCols * xDataTypeSize + 128 +
                        static_cast<uint64_t>(down.usedCoreNum) * 256 * 256 * sizeof(float);
    } else {
        workspaces[0] = compileInfoPtr->sysWorkspaceSize + hiddenOffset +
                        static_cast<size_t>(bs) * hiddenCols * xDataTypeSize + 128 +
                        static_cast<size_t>(bs) * n2 * 4;
    }

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    OP_LOGI(context,
            "chw FFN arch35 fused tiling set(act=%u): M=%u K=%u H=%u N=%u blockDim=%u upCore=%u downCore=%u "
            "swigluSingle=%u up=(n=%u,%u,%u,%u,dbL0C=%u,ubDB=%u) down=(k=%u,%u,%u,%u,dbL0C=%u,ubDB=%u)",
            activeType, bs, k1, hiddenCols, n2, blockDim, up.usedCoreNum, down.usedCoreNum, swigluSingle,
            up.n, up.baseM, up.baseN, up.baseK, up.l0cDB, up.ubDB, down.k, down.baseM, down.baseN, down.baseK,
            down.l0cDB, down.ubDB);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
