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
 * \file all_gather_fit_balance_tiling.h
 * \brief
 */
#ifndef __ALL_GATHER_FIT_BALANCE_TILING_H__
#define __ALL_GATHER_FIT_BALANCE_TILING_H__

#pragma once
#include "op_host/op_tiling/mc2_fit_based_balance_tiling.h"

class AllGatherMMFitBalanceTiling : public Mc2FitBasedBalanceTiling
{
public:
    double frontMMTime_ = 0;

    explicit AllGatherMMFitBalanceTiling(const mc2tiling::TilingArgs& args, KernelType kernelType,
        TopoType topoType = TopoType::STANDARD_CARD, SocVersion socVersion = SocVersion::SOC950) :
        Mc2FitBasedBalanceTiling(args, kernelType, topoType, socVersion)
    {
        commPerf_.SetCommShapeLen(mmInfo_.kValue);
        // mxfp4(DT_FLOAT4_E2M1)每元素占0.5字节，uint64_t无法表达0.5；GetDataTypeSize对该dtype返回0，
        // 会导致公式化性能模型出现×0/÷0(CommTime恒为常数、InverseCommTime/GetLinearThresholdLen除零)。
        // 此处按"1字节打包2个fp4元素"建模：通信侧用SetCommDTypeSize(1)+ExpansionFraction(2)表达0.5字节/元素
        // (复用HCCL现成子字节机制，CommTime/InverseCommTime/GetLinearThresholdLen均经GetRealDtypeSizes()取0.5)；
        // matmul侧inMatrixADtypeSize置1(L2占用按2x高估，避免0导致memUsage/kLength退化为0)。
        if (args.geAType == ge::DataType::DT_FLOAT4_E2M1) {
            commPerf_.SetCommDTypeSize(1);
            commPerf_.SetCommDtypeSizeExpansionFraction(2);
            matmulPerf_.mmShapeInfo_.inMatrixADtypeSize = 1;
            matmulPerf_.mmShapeInfo_.inMatrixBDtypeSize = 1;
            mmInfo_.inMatrixADtypeSize = 1;
            mmInfo_.inMatrixBDtypeSize = 1;
        } else {
            commPerf_.SetCommDTypeSize(mmInfo_.inMatrixADtypeSize);
        }
        tilingM_.SetMinLenByMax(matmulPerf_.GetBaseM());
        tilingM_.SetAlignLength(matmulPerf_.GetBaseM());
    }

    void EstimateMMCommTime() override;
    void SetShortTileLen() override;
    void SetLongTileLen() override;
    void AdjustLongShortTileLen() override;
};

#endif // __ALL_GATHER_FIT_BALANCE_TILING_H__
