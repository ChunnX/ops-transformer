/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file torch_interface.cpp
 * \brief Complete PyTorch entry for flash_attn_example.
 *        The host call chain (validation -> tiling -> tiling copy -> kernel
 *        launch -> sync) is fully implemented; the tiling strategy and the
 *        kernel computation it drives are skeletons (see op_host / op_kernel).
 */

#include <ATen/Operators.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>
#include <cstdlib>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "tiling/platform/platform_ascendc.h"
#include "op_host/fa_tiling.h"

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "op_kernel/fa_kernel_interface.h"

namespace ascend_ops {
namespace FA {

using ascend_ops::fa_host::ContextParamsForTiling;
using ascend_ops::fa_host::FaTiling;

constexpr int64_t SPARSE_MODE_3_MASK_SIZE = 2048;

template <bool hasAttnMask, bool isCombine, config::S1TemplateType s1TemplateType = config::S1TemplateType::Aligned128,
          fa_kernel::config::S2TemplateType s2TemplateType = fa_kernel::config::S2TemplateType::Aligned128>
__global__ __aicore__ void FaKernel(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR attnMask, GM_ADDR attentionOut,
                                    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    FaKernelInterface<hasAttnMask, isCombine, s1TemplateType, s2TemplateType>(query, key, value, attnMask, attentionOut,
                                                                              workspace, tiling);
    return;
}

template <bool hasAttnMask, config::S1TemplateType S1, fa_kernel::config::S2TemplateType S2>
void DispatchKernel(bool isCombine, uint32_t blockDim, void *aclstream, GM_ADDR gq, GM_ADDR gk, GM_ADDR gv, GM_ADDR gm,
                    GM_ADDR go, GM_ADDR gws, GM_ADDR gtl)
{
    if (isCombine) {
        FaKernel<hasAttnMask, true, S1, S2><<<blockDim, nullptr, aclstream>>>(gq, gk, gv, gm, go, gws, gtl);
    } else {
        FaKernel<hasAttnMask, false, S1, S2><<<blockDim, nullptr, aclstream>>>(gq, gk, gv, gm, go, gws, gtl);
    }
}

// 对输入进行校验拦截：典型 case 为 非量化 / isCausal=true / D=128。
bool CheckInput(ContextParamsForTiling &contextKeyParams, const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                const at::Tensor &attnMask, double softmaxScale, bool isCausal)
{
    // q、k、v、attnMask 不允许传 None
    if (!q.defined()) {
        printf("Error: q can't be None.\n");
        return false;
    }
    if (!k.defined()) {
        printf("Error: k can't be None.\n");
        return false;
    }
    if (!v.defined()) {
        printf("Error: v can't be None.\n");
        return false;
    }
    if (!attnMask.defined()) {
        printf("Error: mask can't be None.\n");
        return false;
    }

    // 空 Tensor 检查
    if (q.numel() == 0) {
        printf("Error: query empty tensor is not supported.\n");
        return false;
    }
    if (k.numel() == 0) {
        printf("Error: key empty tensor is not supported.\n");
        return false;
    }
    if (v.numel() == 0) {
        printf("Error: value empty tensor is not supported.\n");
        return false;
    }

    // 非量化：q/k/v 仅支持 bfloat16
    if (q.scalar_type() != at::kBFloat16) {
        printf("Error: query dtype only support bfloat16.\n");
        return false;
    }
    if (k.scalar_type() != at::kBFloat16) {
        printf("Error: key dtype only support bfloat16.\n");
        return false;
    }
    if (v.scalar_type() != at::kBFloat16) {
        printf("Error: value dtype only support bfloat16.\n");
        return false;
    }
    if (attnMask.scalar_type() != at::kBool) {
        printf("Error: mask dtype only support bool.\n");
        return false;
    }

    // 典型 case：仅支持 isCausal=true (CAUSAL)
    if (!isCausal) {
        printf("Error: only support isCausal=true.\n");
        return false;
    }

    // GQA 场景 Qn 与 KVn 检查
    int64_t numHeads = q.size(2);
    int64_t numKeyValueHeads = k.size(2);
    if (numKeyValueHeads == 0 || numHeads % numKeyValueHeads != 0) {
        printf("Error: query N must be divided evenly by kv n.\n");
        return false;
    }

    // BSND: q/k/v 维度为 [B, S, N, D]
    if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4) {
        printf("Error: query/key/value only support 4-dim BSND layout.\n");
        return false;
    }
    if (attnMask.dim() != 2 || attnMask.size(0) != SPARSE_MODE_3_MASK_SIZE ||
        attnMask.size(1) != SPARSE_MODE_3_MASK_SIZE) {
        printf("Error: causal bool mask only support shape [2048, 2048].\n");
        return false;
    }
    if (q.size(0) != k.size(0) || q.size(0) != v.size(0)) {
        printf("Error: query/key/value batch size must be same.\n");
        return false;
    }
    if (k.size(1) != v.size(1)) {
        printf("Error: key/value S must be same.\n");
        return false;
    }

    // QKV 的 D 检查（仅支持 D=128）
    if (q.size(3) != 128) {
        printf("Error: the value of query D only support 128.\n");
        return false;
    }
    if (k.size(3) != 128) {
        printf("Error: the value of key D only support 128.\n");
        return false;
    }
    if (v.size(3) != 128) {
        printf("Error: the value of value D only support 128.\n");
        return false;
    }
    return true;
}

bool ConvertContextToParams(ContextParamsForTiling &contextKeyParams, const at::Tensor &q, const at::Tensor &k,
                            const at::Tensor &v, const at::Tensor &attnMask, double softmaxScale, bool isCausal)
{
    if (!CheckInput(contextKeyParams, q, k, v, attnMask, softmaxScale, isCausal)) {
        return false;
    }
    contextKeyParams.attentionMask = &attnMask;
    contextKeyParams.attentionMaskShape = attnMask.sizes();
    contextKeyParams.maskDataType = attnMask.scalar_type();
    contextKeyParams.inputDataType = q.scalar_type();
    contextKeyParams.kDataType = k.scalar_type();
    contextKeyParams.vDataType = v.scalar_type();
    contextKeyParams.outputDataType = contextKeyParams.inputDataType;
    contextKeyParams.queryInputShape = q.sizes();
    contextKeyParams.keyInputShape = k.sizes();
    contextKeyParams.valueInputShape = v.sizes();
    contextKeyParams.outputShape = contextKeyParams.queryInputShape;
    contextKeyParams.headsNumber = q.size(2);
    contextKeyParams.softmaxScale = softmaxScale;
    contextKeyParams.numKeyValueHeads = k.size(2);

    return true;
}

at::Tensor FlashAttnNpu(const at::Tensor &q,        // 查询张量 [B, S, N, D]
                        const at::Tensor &k,        // 键张量 [B, S, N_kv, D]
                        const at::Tensor &v,        // 值张量 [B, S, N_kv, D]
                        const at::Tensor &attnMask, // 注意力掩码
                        double softmaxScale,        // 注意力缩放系数
                        bool isCausal               // 因果掩码
)
{
    // 获取 npu 信息
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    // stream
    int devidx = q.device().index();
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void *aclstream = stream.stream(true);
    // 计算输出 shape（BSND 输入 -> 同 shape 输出）
    at::Tensor output = at::empty(q.sizes(), q.options());
    // tilingData
    ContextParamsForTiling contextParamsForTiling;
    TORCH_CHECK(ConvertContextToParams(contextParamsForTiling, q, k, v, attnMask, softmaxScale, isCausal),
                "FA input validation failed");
    FaTiling tiling;
    tiling.DoTiling(contextParamsForTiling);

    optiling::FlashAttnTilingData tilingData = tiling.tilingData_;

    const uint32_t cvRatio =
        tiling.platformInfo_.aicNum == 0 ? 0 : tiling.platformInfo_.aivNum / tiling.platformInfo_.aicNum;
    const uint32_t usedAicNum = tilingData.baseTiling.faBaseParams.coreNum;
    const uint32_t usedAivNum = usedAicNum * cvRatio;
    TORCH_CHECK(usedAicNum > 0 && usedAivNum > 0, "FA tiling generated invalid core num, aic=", usedAicNum,
                ", aiv=", usedAivNum);
    uint32_t blockDimToBeSet =
        ascendcPlatform->CalcTschBlockDim(usedAivNum, tiling.platformInfo_.aicNum, tiling.platformInfo_.aivNum);

    int64_t workspaceSize = contextParamsForTiling.workspaceSize;
    auto workspaceTensor =
        at::empty({workspaceSize}, at::TensorOptions().dtype(at::kByte).device(q.options().device()));

    int64_t tilingFileSize = sizeof(optiling::FlashAttnTilingData);
    auto tilingTensor = at::empty({tilingFileSize}, at::TensorOptions().dtype(at::kByte).device(q.options().device()));
    auto copyRet =
        aclrtMemcpy(tilingTensor.data_ptr(), tilingFileSize, &tilingData, tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(copyRet == ACL_ERROR_NONE, "FA tiling copy failed, aclrtMemcpy ret=", copyRet);

    constexpr bool hasAttnMask = true; // 使能 mask
    const bool isCombine = tiling.tilingKeyInfo_.isCombine;
    auto aclCal = [=]() -> int {
        GM_ADDR gq = (GM_ADDR)(q.data_ptr());
        GM_ADDR gk = (GM_ADDR)(k.data_ptr());
        GM_ADDR gv = (GM_ADDR)(v.data_ptr());
        GM_ADDR gm = (GM_ADDR)(attnMask.data_ptr());
        GM_ADDR go = (GM_ADDR)(output.data_ptr());
        GM_ADDR gws = (GM_ADDR)(workspaceTensor.data_ptr());
        GM_ADDR gtl = (GM_ADDR)(tilingTensor.data_ptr());

        DispatchKernel<hasAttnMask, config::S1TemplateType::Aligned128, fa_kernel::config::S2TemplateType::Aligned128>(
            isCombine, blockDimToBeSet, aclstream, gq, gk, gv, gm, go, gws, gtl);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("FA", aclCal);

    auto syncRet = aclrtSynchronizeStream(aclstream);
    TORCH_CHECK(syncRet == ACL_ERROR_NONE, "FA kernel synchronize failed, aclrtSynchronizeStream ret = ", syncRet);

    return output;
}

TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
{
    m.impl("flash_attn_example", FlashAttnNpu);
}

TORCH_LIBRARY(ascend_ops, m)
{
    m.def(R"(flash_attn_example(Tensor q,
             Tensor k,
             Tensor v,
             Tensor attnMask,
             float softmaxScale = 0,
             bool isCausal = True) -> Tensor)");
}
} // namespace FA
} // namespace ascend_ops

extern "C" {
PyObject *PyInit__C(void)
{
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT, "_C", NULL, -1, NULL,
    };
    return PyModule_Create(&module_def);
}
}
