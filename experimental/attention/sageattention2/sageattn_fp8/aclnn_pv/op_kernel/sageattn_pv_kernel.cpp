/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * aclnn 骨架：统一 FP8 PV 入口 kernel 占位
 * 待 P1 修通后接真 kernel（FAInferInt8QK 或 mm1_qk_int8 + FA pipeline 组合）
 */

#include "kernel_operator.h"
using namespace AscendC;

// kernel 占位：仅打印参数，不做实际计算
extern "C" __attribute__((visibility("default"))) void SageAttnPvKernel(uint8_t* q, uint8_t* k, uint8_t* v,
    uint8_t* qScale, uint8_t* kScale, uint8_t* vScale, uint8_t* o, uint8_t* lse, uint8_t* tiling) {
    (void)q; (void)k; (void)v;
    (void)qScale; (void)kScale; (void)vScale;
    (void)o; (void)lse; (void)tiling;
    // 占位：空操作，只验证编译通过
    // TODO: 接真 kernel（FAInferInt8QK 或 mm1_qk_int8 + softmax + PV + rescale）
}