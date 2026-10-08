/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <functional>
#include <cstdint>
#include <algorithm>
#include <cstring>

#ifndef USE_MOCK_ACLNN
#include "acl/acl.h"
#include "aclnn_causal_conv1d_grad.h"
#endif

// ============================================================================
// 宏定义
// ============================================================================
#define LOG_PRINT(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)

// ============================================================================
// FP16/BF16 辅助类型（简化表示）
// ============================================================================
struct float16_t {
    uint16_t data;
    float16_t() : data(0) {}
    explicit float16_t(float f);
    operator float() const;
};

struct bfloat16_t {
    uint16_t data;
    bfloat16_t() : data(0) {}
    explicit bfloat16_t(float f);
    operator float() const;
};

// FP16 转换（真实 IEEE 754 half）
float16_t::float16_t(float f) {
    uint32_t fbits;
    memcpy(&fbits, &f, sizeof(float));

    uint32_t sign = (fbits >> 31) & 0x1;
    uint32_t exp = (fbits >> 23) & 0xFF;
    uint32_t mantissa = fbits & 0x7FFFFF;

    uint16_t hexp, hmantissa;
    if (exp == 0) {
        hexp = 0; hmantissa = 0;
    } else if (exp == 0xFF) {
        hexp = 0x1F; hmantissa = (mantissa != 0) ? 0x200 : 0;
    } else {
        int32_t newexp = (int32_t)exp - 127 + 15;
        if (newexp >= 0x1F) {
            hexp = 0x1F; hmantissa = 0;
        } else if (newexp <= 0) {
            hexp = 0; hmantissa = 0;
        } else {
            hexp = (uint16_t)newexp;
            hmantissa = (uint16_t)(mantissa >> 13);
        }
    }
    data = (sign << 15) | (hexp << 10) | hmantissa;
}

float16_t::operator float() const {
    uint32_t sign = (data >> 15) & 0x1;
    uint32_t exp = (data >> 10) & 0x1F;
    uint32_t mantissa = data & 0x3FF;

    uint32_t fexp, fmantissa;
    if (exp == 0) {
        if (mantissa == 0) {
            fexp = 0; fmantissa = 0;
        } else {
            fexp = 0; fmantissa = 0;
        }
    } else if (exp == 0x1F) {
        fexp = 0xFF;
        fmantissa = (mantissa != 0) ? (1 << 22) : 0;
    } else {
        fexp = exp - 15 + 127;
        fmantissa = mantissa << 13;
    }

    uint32_t fbits = (sign << 31) | (fexp << 23) | fmantissa;
    float result;
    memcpy(&result, &fbits, sizeof(float));
    return result;
}

// BF16 转换
bfloat16_t::bfloat16_t(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    data = static_cast<uint16_t>(bits >> 16);
}

bfloat16_t::operator float() const {
    uint32_t bits = static_cast<uint32_t>(data) << 16;
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

// ============================================================================
// 辅助函数
// ============================================================================

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t size = 1;
    for (auto dim : shape) size *= dim;
    return size;
}

// ============================================================================
// CPU Golden 计算函数（全程 fp32 累加）
// ============================================================================

template<typename DType>
void ComputeGoldenCausalConv1dGrad(
    const DType* grad_y, const DType* x, const DType* weight, const DType* bias,
    DType* dx, DType* dweight, DType* dbias,
    int64_t B, int64_t T, int64_t D, int64_t kW = 4)
{
    // 转换为 fp32
    std::vector<float> grad_y_fp32(B * T * D);
    std::vector<float> x_fp32(B * T * D);
    std::vector<float> weight_fp32(kW * D);
    std::vector<float> bias_fp32(D);

    for (int64_t i = 0; i < B * T * D; ++i) {
        grad_y_fp32[i] = static_cast<float>(grad_y[i]);
        x_fp32[i] = static_cast<float>(x[i]);
    }
    for (int64_t i = 0; i < kW * D; ++i) {
        weight_fp32[i] = static_cast<float>(weight[i]);
    }
    for (int64_t i = 0; i < D; ++i) {
        bias_fp32[i] = static_cast<float>(bias[i]);
    }

    // 左侧 padding（冷启动）
    std::vector<float> x_pad(B * (T + kW - 1) * D, 0.0f);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t d = 0; d < D; ++d) {
                x_pad[b * (T + kW - 1) * D + (t + kW - 1) * D + d] =
                    x_fp32[b * T * D + t * D + d];
            }
        }
    }

    // 重算 pre-activation z
    std::vector<float> z(B * T * D, 0.0f);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t d = 0; d < D; ++d) {
                float sum = bias_fp32[d];
                for (int64_t j = 0; j < kW; ++j) {
                    sum += weight_fp32[j * D + d] *
                           x_pad[b * (T + kW - 1) * D + (t + kW - 1 - j) * D + d];
                }
                z[b * T * D + t * D + d] = sum;
            }
        }
    }

    // SiLU 导数
    std::vector<float> grad_z(B * T * D);
    for (int64_t i = 0; i < B * T * D; ++i) {
        float z_val = z[i];
        float sigmoid_z = 1.0f / (1.0f + std::exp(-z_val));
        float silu_deriv = sigmoid_z * (1.0f + z_val * (1.0f - sigmoid_z));
        grad_z[i] = grad_y_fp32[i] * silu_deriv;
    }

    // 计算 dbias
    std::vector<float> dbias_fp32(D, 0.0f);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t d = 0; d < D; ++d) {
                dbias_fp32[d] += grad_z[b * T * D + t * D + d];
            }
        }
    }

    // 计算 dweight
    std::vector<float> dweight_fp32(kW * D, 0.0f);
    for (int64_t j = 0; j < kW; ++j) {
        for (int64_t d = 0; d < D; ++d) {
            float sum = 0.0f;
            for (int64_t b = 0; b < B; ++b) {
                for (int64_t t = 0; t < T; ++t) {
                    sum += grad_z[b * T * D + t * D + d] *
                           x_pad[b * (T + kW - 1) * D + (t + kW - 1 - j) * D + d];
                }
            }
            dweight_fp32[j * D + d] = sum;
        }
    }

    // 计算 dx（转置卷积）
    std::vector<float> gz_rpad(B * (T + kW - 1) * D, 0.0f);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t d = 0; d < D; ++d) {
                gz_rpad[b * (T + kW - 1) * D + t * D + d] =
                    grad_z[b * T * D + t * D + d];
            }
        }
    }

    std::vector<float> dx_fp32(B * T * D, 0.0f);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t d = 0; d < D; ++d) {
                float sum = 0.0f;
                for (int64_t j = 0; j < kW; ++j) {
                    sum += weight_fp32[j * D + d] *
                           gz_rpad[b * (T + kW - 1) * D + (t + j) * D + d];
                }
                dx_fp32[b * T * D + t * D + d] = sum;
            }
        }
    }

    // Cast 回原始 dtype
    for (int64_t i = 0; i < B * T * D; ++i) {
        dx[i] = static_cast<DType>(dx_fp32[i]);
    }
    for (int64_t i = 0; i < kW * D; ++i) {
        dweight[i] = static_cast<DType>(dweight_fp32[i]);
    }
    for (int64_t i = 0; i < D; ++i) {
        dbias[i] = static_cast<DType>(dbias_fp32[i]);
    }
}

// ============================================================================
// 精度比对函数（混合容差）
// ============================================================================
// float16:  rtol=1e-3, atol=1e-3, matched_ratio≥0.99, max_abs_error≤0.1
// bfloat16: rtol=4e-3, atol=4e-3, matched_ratio≥0.99, max_abs_error≤1.0
// ============================================================================

template<typename T>
bool CompareResults(const T* golden, const T* actual, size_t size,
                    double rtol, double atol,
                    double matched_ratio_threshold = 0.99,
                    double max_abs_error_limit = 0.1) {
    if (golden == nullptr || actual == nullptr) {
        LOG_PRINT("  [ERROR] null pointer input");
        return false;
    }
    if (size == 0) return true;

    size_t matched = 0;
    double max_abs_error = 0.0;
    double sum_abs_error = 0.0;

    for (size_t i = 0; i < size; ++i) {
        double g = static_cast<double>(golden[i]);
        double a = static_cast<double>(actual[i]);
        double abs_error = std::abs(a - g);
        double threshold = atol + rtol * std::abs(g);

        if (abs_error <= threshold) {
            matched++;
        }
        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
        }
        sum_abs_error += abs_error;
    }

    double matched_ratio = static_cast<double>(matched) / static_cast<double>(size);
    double mean_abs_error = sum_abs_error / static_cast<double>(size);

    bool pass = (matched_ratio >= matched_ratio_threshold) &&
                (max_abs_error <= max_abs_error_limit);

    if (pass) {
        LOG_PRINT("  [PASS] matched_ratio=%.4f, max_abs_error=%.2e, mean_abs_error=%.2e",
                  matched_ratio, max_abs_error, mean_abs_error);
    } else {
        LOG_PRINT("  [FAIL] matched_ratio=%.4f (>= %.2f), max_abs_error=%.2e (<= %.2e)",
                  matched_ratio, matched_ratio_threshold, max_abs_error, max_abs_error_limit);
        int shown = 0;
        for (size_t i = 0; i < size && shown < 5; ++i) {
            double g = static_cast<double>(golden[i]);
            double a = static_cast<double>(actual[i]);
            double abs_error = std::abs(a - g);
            double threshold = atol + rtol * std::abs(g);
            if (abs_error > threshold) {
                LOG_PRINT("  不匹配 [%zu]: 期望=%.6f, 实际=%.6f, abs_err=%.2e",
                          i, g, a, abs_error);
                shown++;
            }
        }
    }
    return pass;
}

// ============================================================================
// CPU Golden 自测
// ============================================================================

void TestGoldenCorrectness() {
    LOG_PRINT("\n========================================");
    LOG_PRINT("CPU Golden 正确性自测");
    LOG_PRINT("========================================");

    int64_t B = 1, T = 4, D = 12288, kW = 4;
    size_t input_size = B * T * D;
    size_t weight_size = kW * D;
    size_t bias_size = D;

    std::vector<float> grad_y(input_size);
    std::vector<float> x(input_size);
    std::vector<float> weight(weight_size);
    std::vector<float> bias(bias_size);
    std::vector<float> dx(input_size);
    std::vector<float> dweight(weight_size);
    std::vector<float> dbias(bias_size);

    // 初始化测试数据（缩小范围避免溢出）
    // 注意：i 是 size_t（无符号），必须先转 int 再减，否则负数下溢成 2^64 级大数
    for (size_t i = 0; i < input_size; ++i) {
        grad_y[i] = 0.01f * (static_cast<int>(i % 10) - 5);
        x[i] = 0.1f * (static_cast<int>(i % 8) - 4);
    }
    for (size_t i = 0; i < weight_size; ++i) {
        weight[i] = 0.01f * (static_cast<int>(i % 10) - 5);
    }
    for (size_t i = 0; i < bias_size; ++i) {
        bias[i] = 0.001f * (static_cast<int>(i % 10) - 5);
    }

    LOG_PRINT("\n测试 1: 基础反向传播（B=%ld, T=%ld, D=%ld）", B, T, D);
    ComputeGoldenCausalConv1dGrad(
        grad_y.data(), x.data(), weight.data(), bias.data(),
        dx.data(), dweight.data(), dbias.data(),
        B, T, D, kW);

    // 验证非零输出
    bool dx_nonzero = false, dw_nonzero = false, db_nonzero = false;
    for (size_t i = 0; i < input_size && !dx_nonzero; ++i) {
        if (std::abs(dx[i]) > 1e-6f) dx_nonzero = true;
    }
    for (size_t i = 0; i < weight_size && !dw_nonzero; ++i) {
        if (std::abs(dweight[i]) > 1e-6f) dw_nonzero = true;
    }
    for (size_t i = 0; i < bias_size && !db_nonzero; ++i) {
        if (std::abs(dbias[i]) > 1e-6f) db_nonzero = true;
    }

    if (dx_nonzero && dw_nonzero && db_nonzero) {
        LOG_PRINT("  [PASS] 所有梯度输出非零");
    } else {
        LOG_PRINT("  [FAIL] 部分梯度为零: dx=%d, dw=%d, db=%d",
                  dx_nonzero, dw_nonzero, db_nonzero);
    }

    LOG_PRINT("  dx 范围: [%.6f, %.6f]",
              *std::min_element(dx.begin(), dx.end()),
              *std::max_element(dx.begin(), dx.end()));
    LOG_PRINT("  dweight 范围: [%.6f, %.6f]",
              *std::min_element(dweight.begin(), dweight.end()),
              *std::max_element(dweight.begin(), dweight.end()));
    LOG_PRINT("  dbias 范围: [%.6f, %.6f]",
              *std::min_element(dbias.begin(), dbias.end()),
              *std::max_element(dbias.begin(), dbias.end()));

    LOG_PRINT("\n测试 2: grad_y 全零（应输出全零梯度）");
    std::fill(grad_y.begin(), grad_y.end(), 0.0f);
    ComputeGoldenCausalConv1dGrad(
        grad_y.data(), x.data(), weight.data(), bias.data(),
        dx.data(), dweight.data(), dbias.data(),
        B, T, D, kW);

    bool all_zero = true;
    for (size_t i = 0; i < input_size; ++i) {
        if (std::abs(dx[i]) > 1e-6f) { all_zero = false; break; }
    }
    for (size_t i = 0; i < weight_size && all_zero; ++i) {
        if (std::abs(dweight[i]) > 1e-6f) { all_zero = false; break; }
    }
    for (size_t i = 0; i < bias_size && all_zero; ++i) {
        if (std::abs(dbias[i]) > 1e-6f) { all_zero = false; break; }
    }

    LOG_PRINT("  结果: %s", all_zero ? "PASS" : "FAIL");

    LOG_PRINT("\n========================================");
}

// ============================================================================
// Real 模式辅助函数
// ============================================================================

#ifndef USE_MOCK_ACLNN

std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

template<typename T>
int CreateAclTensor(const std::vector<T>& hostData,
                   const std::vector<int64_t>& shape,
                   void** deviceAddr,
                   aclDataType dataType,
                   aclTensor** tensor) {
    size_t size = GetShapeSize(shape) * sizeof(T);

    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) return ret;

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) { aclrtFree(*deviceAddr); return ret; }

    auto strides = ComputeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(),
                              0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

#endif

// ============================================================================
// 统一测试执行器
// ============================================================================

#ifdef USE_MOCK_ACLNN

template<typename DType>
bool RunTest(const std::vector<DType>& grad_y, const std::vector<DType>& x,
             const std::vector<DType>& weight, const std::vector<DType>& bias,
             int64_t B, int64_t T, int64_t D, int64_t kW,
             const char* dtype, double rtol, double atol,
             double matched_ratio_threshold, double max_abs_error_limit) {
    LOG_PRINT("\n[Mock] 测试 - dtype=%s, shape=[%ld,%ld,%ld]", dtype, B, T, D);

    size_t input_size = B * T * D;
    size_t weight_size = kW * D;
    size_t bias_size = D;

    std::vector<DType> dx(input_size);
    std::vector<DType> dweight(weight_size);
    std::vector<DType> dbias(bias_size);

    std::vector<DType> dx_golden(input_size);
    std::vector<DType> dweight_golden(weight_size);
    std::vector<DType> dbias_golden(bias_size);

    ComputeGoldenCausalConv1dGrad(
        grad_y.data(), x.data(), weight.data(), bias.data(),
        dx_golden.data(), dweight_golden.data(), dbias_golden.data(),
        B, T, D, kW);

    // Mock 模式：直接使用 golden 作为输出
    dx = dx_golden;
    dweight = dweight_golden;
    dbias = dbias_golden;

    bool pass_dx = CompareResults(dx_golden.data(), dx.data(), input_size,
                                   rtol, atol, matched_ratio_threshold, max_abs_error_limit);
    bool pass_dw = CompareResults(dweight_golden.data(), dweight.data(), weight_size,
                                   rtol, atol, matched_ratio_threshold, max_abs_error_limit);
    bool pass_db = CompareResults(dbias_golden.data(), dbias.data(), bias_size,
                                   rtol, atol, matched_ratio_threshold, max_abs_error_limit);

    return pass_dx && pass_dw && pass_db;
}

#else

// Real 模式实现
template<typename DType>
bool RunTest(const std::vector<DType>& grad_y, const std::vector<DType>& x,
             const std::vector<DType>& weight, const std::vector<DType>& bias,
             int64_t B, int64_t T, int64_t D, int64_t kW,
             aclDataType dtype, aclrtStream stream,
             double rtol, double atol,
             double matched_ratio_threshold, double max_abs_error_limit) {
    LOG_PRINT("\n[Real] 测试 - shape=[%ld,%ld,%ld]", B, T, D);

    size_t input_size = B * T * D;
    size_t weight_size = kW * D;
    size_t bias_size = D;

    // 计算 golden
    std::vector<DType> dx_golden(input_size);
    std::vector<DType> dweight_golden(weight_size);
    std::vector<DType> dbias_golden(bias_size);
    ComputeGoldenCausalConv1dGrad(
        grad_y.data(), x.data(), weight.data(), bias.data(),
        dx_golden.data(), dweight_golden.data(), dbias_golden.data(),
        B, T, D, kW);

    // 创建输入 tensor
    void *gradY_dev = nullptr, *x_dev = nullptr, *weight_dev = nullptr, *bias_dev = nullptr;
    void *dx_dev = nullptr, *dweight_dev = nullptr, *dbias_dev = nullptr;
    aclTensor *gradY_tensor = nullptr, *x_tensor = nullptr, *weight_tensor = nullptr, *bias_tensor = nullptr;
    aclTensor *dx_tensor = nullptr, *dweight_tensor = nullptr, *dbias_tensor = nullptr;

    std::vector<int64_t> input_shape = {B, T, D};
    std::vector<int64_t> weight_shape = {kW, D};
    std::vector<int64_t> bias_shape = {D};

    auto ret = CreateAclTensor(grad_y, input_shape, &gradY_dev, dtype, &gradY_tensor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [ERROR] CreateAclTensor(gradY) failed: %d", ret);
        return false;
    }
    ret = CreateAclTensor(x, input_shape, &x_dev, dtype, &x_tensor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [ERROR] CreateAclTensor(x) failed: %d", ret);
        aclrtFree(gradY_dev); aclDestroyTensor(gradY_tensor);
        return false;
    }
    ret = CreateAclTensor(weight, weight_shape, &weight_dev, dtype, &weight_tensor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [ERROR] CreateAclTensor(weight) failed: %d", ret);
        aclrtFree(gradY_dev); aclDestroyTensor(gradY_tensor);
        aclrtFree(x_dev); aclDestroyTensor(x_tensor);
        return false;
    }
    ret = CreateAclTensor(bias, bias_shape, &bias_dev, dtype, &bias_tensor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [ERROR] CreateAclTensor(bias) failed: %d", ret);
        aclrtFree(gradY_dev); aclDestroyTensor(gradY_tensor);
        aclrtFree(x_dev); aclDestroyTensor(x_tensor);
        aclrtFree(weight_dev); aclDestroyTensor(weight_tensor);
        return false;
    }

    // [DBG] 输入 H2D 回读校验 + 设备指针记录（定位输入地址错位问题）
    {
        DType rb[4] = {DType(0), DType(0), DType(0), DType(0)};
        aclrtMemcpy(rb, 4 * sizeof(DType), gradY_dev, 4 * sizeof(DType), ACL_MEMCPY_DEVICE_TO_HOST);
        float g0 = static_cast<float>(rb[0]), g1 = static_cast<float>(rb[1]);
        aclrtMemcpy(rb, 4 * sizeof(DType), x_dev, 4 * sizeof(DType), ACL_MEMCPY_DEVICE_TO_HOST);
        float x0 = static_cast<float>(rb[0]), x1 = static_cast<float>(rb[1]);
        LOG_PRINT("  [DBG] devptr gy=%p x=%p w=%p b=%p", gradY_dev, x_dev, weight_dev, bias_dev);
        LOG_PRINT("  [DBG] readback gy=%.6f,%.6f x=%.6f,%.6f", g0, g1, x0, x1);
    }

    // 创建输出 tensor
    std::vector<DType> dx_init(input_size, DType(0));
    std::vector<DType> dweight_init(weight_size, DType(0));
    std::vector<DType> dbias_init(bias_size, DType(0));
    ret = CreateAclTensor(dx_init, input_shape, &dx_dev, dtype, &dx_tensor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [ERROR] CreateAclTensor(dx) failed: %d", ret);
        aclrtFree(gradY_dev); aclDestroyTensor(gradY_tensor);
        aclrtFree(x_dev); aclDestroyTensor(x_tensor);
        aclrtFree(weight_dev); aclDestroyTensor(weight_tensor);
        aclrtFree(bias_dev); aclDestroyTensor(bias_tensor);
        return false;
    }
    ret = CreateAclTensor(dweight_init, weight_shape, &dweight_dev, dtype, &dweight_tensor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [ERROR] CreateAclTensor(dweight) failed: %d", ret);
        aclrtFree(gradY_dev); aclDestroyTensor(gradY_tensor);
        aclrtFree(x_dev); aclDestroyTensor(x_tensor);
        aclrtFree(weight_dev); aclDestroyTensor(weight_tensor);
        aclrtFree(bias_dev); aclDestroyTensor(bias_tensor);
        aclrtFree(dx_dev); aclDestroyTensor(dx_tensor);
        return false;
    }
    ret = CreateAclTensor(dbias_init, bias_shape, &dbias_dev, dtype, &dbias_tensor);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [ERROR] CreateAclTensor(dbias) failed: %d", ret);
        aclrtFree(gradY_dev); aclDestroyTensor(gradY_tensor);
        aclrtFree(x_dev); aclDestroyTensor(x_tensor);
        aclrtFree(weight_dev); aclDestroyTensor(weight_tensor);
        aclrtFree(bias_dev); aclDestroyTensor(bias_tensor);
        aclrtFree(dx_dev); aclDestroyTensor(dx_tensor);
        aclrtFree(dweight_dev); aclDestroyTensor(dweight_tensor);
        return false;
    }

    // 调用 ACLNN L2 API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto aclnnRet = aclnnCausalConv1dGradGetWorkspaceSize(
        gradY_tensor, x_tensor, weight_tensor, bias_tensor,
        dx_tensor, dweight_tensor, dbias_tensor,
        &workspaceSize, &executor);

    if (aclnnRet != ACL_SUCCESS) {
        LOG_PRINT("  [ERROR] aclnnCausalConv1dGradGetWorkspaceSize failed: %d", aclnnRet);
        aclrtFree(gradY_dev); aclDestroyTensor(gradY_tensor);
        aclrtFree(x_dev); aclDestroyTensor(x_tensor);
        aclrtFree(weight_dev); aclDestroyTensor(weight_tensor);
        aclrtFree(bias_dev); aclDestroyTensor(bias_tensor);
        aclrtFree(dx_dev); aclDestroyTensor(dx_tensor);
        aclrtFree(dweight_dev); aclDestroyTensor(dweight_tensor);
        aclrtFree(dbias_dev); aclDestroyTensor(dbias_tensor);
        return false;
    }

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  [ERROR] workspace malloc failed: %d", ret);
            aclrtFree(gradY_dev); aclDestroyTensor(gradY_tensor);
            aclrtFree(x_dev); aclDestroyTensor(x_tensor);
            aclrtFree(weight_dev); aclDestroyTensor(weight_tensor);
            aclrtFree(bias_dev); aclDestroyTensor(bias_tensor);
            aclrtFree(dx_dev); aclDestroyTensor(dx_tensor);
            aclrtFree(dweight_dev); aclDestroyTensor(dweight_tensor);
            aclrtFree(dbias_dev); aclDestroyTensor(dbias_tensor);
            return false;
        }
    }

    aclnnRet = aclnnCausalConv1dGrad(workspace, workspaceSize, executor, stream);
    if (aclnnRet != ACL_SUCCESS) {
        LOG_PRINT("  [ERROR] aclnnCausalConv1dGrad failed: %d", aclnnRet);
        if (workspace) aclrtFree(workspace);
        aclrtFree(gradY_dev); aclDestroyTensor(gradY_tensor);
        aclrtFree(x_dev); aclDestroyTensor(x_tensor);
        aclrtFree(weight_dev); aclDestroyTensor(weight_tensor);
        aclrtFree(bias_dev); aclDestroyTensor(bias_tensor);
        aclrtFree(dx_dev); aclDestroyTensor(dx_tensor);
        aclrtFree(dweight_dev); aclDestroyTensor(dweight_tensor);
        aclrtFree(dbias_dev); aclDestroyTensor(dbias_tensor);
        return false;
    }

    aclrtSynchronizeStream(stream);

    // D2H
    std::vector<DType> dx_npu(input_size);
    std::vector<DType> dweight_npu(weight_size);
    std::vector<DType> dbias_npu(bias_size);
    aclrtMemcpy(dx_npu.data(), input_size * sizeof(DType), dx_dev, input_size * sizeof(DType), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(dweight_npu.data(), weight_size * sizeof(DType), dweight_dev, weight_size * sizeof(DType), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(dbias_npu.data(), bias_size * sizeof(DType), dbias_dev, bias_size * sizeof(DType), ACL_MEMCPY_DEVICE_TO_HOST);

    // 清理
    if (workspace) aclrtFree(workspace);
    aclrtFree(gradY_dev); aclDestroyTensor(gradY_tensor);
    aclrtFree(x_dev); aclDestroyTensor(x_tensor);
    aclrtFree(weight_dev); aclDestroyTensor(weight_tensor);
    aclrtFree(bias_dev); aclDestroyTensor(bias_tensor);
    aclrtFree(dx_dev); aclDestroyTensor(dx_tensor);
    aclrtFree(dweight_dev); aclDestroyTensor(dweight_tensor);
    aclrtFree(dbias_dev); aclDestroyTensor(dbias_tensor);

    // 比对
    LOG_PRINT("  dx:");
    bool pass_dx = CompareResults(dx_golden.data(), dx_npu.data(), input_size,
                                   rtol, atol, matched_ratio_threshold, max_abs_error_limit);
    LOG_PRINT("  dweight:");
    bool pass_dw = CompareResults(dweight_golden.data(), dweight_npu.data(), weight_size,
                                   rtol, atol, matched_ratio_threshold, max_abs_error_limit);
    LOG_PRINT("  dbias:");
    bool pass_db = CompareResults(dbias_golden.data(), dbias_npu.data(), bias_size,
                                   rtol, atol, matched_ratio_threshold, max_abs_error_limit);

    return pass_dx && pass_dw && pass_db;
}

#endif

// ============================================================================
// 测试用例定义
// ============================================================================

struct TestCase {
    const char* name;
    int64_t B, T, D, kW;
    bool is_fp16;
    double rtol, atol;
    double matched_ratio_threshold;
    double max_abs_error_limit;
    bool zero_grad_y = false;  // EI1: grad_y 全零（输出应全零）
};

std::vector<TestCase> GetTestCases() {
    return {
        // 本次 NPU 验证任务范围：仅 bfloat16（fp16 用例按任务要求跳过/注释）
        // {"L0-001 float16 基础", 1, 4, 12288, 4, true, 1e-3, 1e-3, 0.99, 0.1},
        // {"L0-002 bfloat16 基础", 2, 8, 12288, 4, false, 4e-3, 4e-3, 0.99, 1.0},

        // L1 边界条件（bf16）
        // {"BC1 最小 batch/seq (B=1,T=1)", 1, 1, 12288, 4, true, 1e-3, 1e-3, 0.99, 0.1},
        // {"BC2 T=kW=4", 1, 4, 12288, 4, true, 1e-3, 1e-3, 0.99, 0.1},
        // {"BC3 典型训练 (B=2,T=16)", 2, 16, 12288, 4, true, 1e-3, 1e-3, 0.99, 0.1},
        {"BC1_bf16 最小 batch/seq (B=1,T=1)", 1, 1, 12288, 4, false, 4e-3, 4e-3, 0.99, 1.0},
        {"BC2_bf16 T=kW 边界 (B=1,T=4)", 1, 4, 12288, 4, false, 4e-3, 4e-3, 0.99, 1.0},
        {"BC3_bf16 典型 shape (B=2,T=16)", 2, 16, 12288, 4, false, 4e-3, 4e-3, 0.99, 1.0},

        // L1 极端输入（bf16）：grad_y 全零，dx/dweight/dbias 输出应全零
        {"EI1_bf16 grad_y 全零", 1, 8, 12288, 4, false, 4e-3, 4e-3, 0.99, 1.0, true},
    };
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char* argv[]) {
    LOG_PRINT("\n========================================");
    LOG_PRINT("causal_conv1d_grad 算子 ST 测试");
    LOG_PRINT("========================================");

#ifdef USE_MOCK_ACLNN
    LOG_PRINT("模式: Mock (CPU golden)");
#else
    LOG_PRINT("模式: Real (NPU)");
#endif

    TestGoldenCorrectness();

    int passed = 0, failed = 0;

#ifndef USE_MOCK_ACLNN
    // 设置环境变量（必须在 aclInit 之前）
    // ASCEND_CUSTOM_OPP_PATH 必须指向 vendors/<vendor_name> 目录
    const char* custom_opp_vendor = "/workspace/huangqiang/hq_claude/code/cannbot-skills/plugins-official/ops-registry-invoke/operators/causal_conv1d_grad/custom_opp/vendors/causal_conv1d_grad_custom";
    setenv("ASCEND_CUSTOM_OPP_PATH", custom_opp_vendor, 1);
    LOG_PRINT("设置 ASCEND_CUSTOM_OPP_PATH=%s", custom_opp_vendor);

    int32_t deviceId = 0;
    aclrtStream stream;

    auto initRet = aclInit(nullptr);
    if (initRet != ACL_SUCCESS) {
        LOG_PRINT("[FATAL] aclInit失败: %d", initRet);
        return 1;
    }
    initRet = aclrtSetDevice(deviceId);
    if (initRet != ACL_SUCCESS) {
        LOG_PRINT("[FATAL] aclrtSetDevice(%d)失败: %d", deviceId, initRet);
        aclFinalize();
        return 1;
    }
    initRet = aclrtCreateStream(&stream);
    if (initRet != ACL_SUCCESS) {
        LOG_PRINT("[FATAL] aclrtCreateStream失败: %d", initRet);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }
#endif

    LOG_PRINT("\n执行内置测试用例...");

    auto test_cases = GetTestCases();

    for (const auto& tc : test_cases) {
        LOG_PRINT("\n测试: %s", tc.name);

        size_t input_size = tc.B * tc.T * tc.D;
        size_t weight_size = tc.kW * tc.D;
        size_t bias_size = tc.D;

        // 生成测试数据（缩小范围避免 sigmoid/exp 溢出）
        std::vector<float> grad_y_fp32(input_size);
        std::vector<float> x_fp32(input_size);
        std::vector<float> weight_fp32(weight_size);
        std::vector<float> bias_fp32(bias_size);

        // grad_y: ±0.01~0.05（小梯度）；EI1 用例 grad_y 全零
        // 注意：i 是 size_t（无符号），必须先转 int 再减，否则负数下溢成 2^64 级大数
        for (size_t i = 0; i < input_size; ++i) {
            grad_y_fp32[i] = tc.zero_grad_y ? 0.0f : (0.01f * (static_cast<int>(i % 10) - 5));
        }
        // x: ±0.1~0.4（中等输入）
        for (size_t i = 0; i < input_size; ++i) {
            x_fp32[i] = 0.1f * (static_cast<int>(i % 8) - 4);
        }
        // weight: ±0.01~0.05（小权重）
        for (size_t i = 0; i < weight_size; ++i) {
            weight_fp32[i] = 0.01f * (static_cast<int>(i % 10) - 5);
        }
        // bias: ±0.001~0.005（极小偏置）
        for (size_t i = 0; i < bias_size; ++i) {
            bias_fp32[i] = 0.001f * (static_cast<int>(i % 10) - 5);
        }

#ifdef USE_MOCK_ACLNN
        if (tc.is_fp16) {
            if (RunTest(grad_y_fp32, x_fp32, weight_fp32, bias_fp32,
                        tc.B, tc.T, tc.D, tc.kW, "float16",
                        tc.rtol, tc.atol, tc.matched_ratio_threshold, tc.max_abs_error_limit)) {
                passed++;
            } else {
                failed++;
            }
        } else {
            std::vector<bfloat16_t> grad_y_bf16(input_size);
            std::vector<bfloat16_t> x_bf16(input_size);
            std::vector<bfloat16_t> weight_bf16(weight_size);
            std::vector<bfloat16_t> bias_bf16(bias_size);
            for (size_t i = 0; i < input_size; ++i) {
                grad_y_bf16[i] = bfloat16_t(grad_y_fp32[i]);
                x_bf16[i] = bfloat16_t(x_fp32[i]);
            }
            for (size_t i = 0; i < weight_size; ++i) {
                weight_bf16[i] = bfloat16_t(weight_fp32[i]);
            }
            for (size_t i = 0; i < bias_size; ++i) {
                bias_bf16[i] = bfloat16_t(bias_fp32[i]);
            }
            if (RunTest(grad_y_bf16, x_bf16, weight_bf16, bias_bf16,
                        tc.B, tc.T, tc.D, tc.kW, "bfloat16",
                        tc.rtol, tc.atol, tc.matched_ratio_threshold, tc.max_abs_error_limit)) {
                passed++;
            } else {
                failed++;
            }
        }
#else
        // Real 模式
        aclDataType dtype = tc.is_fp16 ? ACL_FLOAT16 : ACL_BF16;
        if (tc.is_fp16) {
            std::vector<float16_t> grad_y_fp16(input_size);
            std::vector<float16_t> x_fp16(input_size);
            std::vector<float16_t> weight_fp16(weight_size);
            std::vector<float16_t> bias_fp16(bias_size);
            for (size_t i = 0; i < input_size; ++i) {
                grad_y_fp16[i] = float16_t(grad_y_fp32[i]);
                x_fp16[i] = float16_t(x_fp32[i]);
            }
            for (size_t i = 0; i < weight_size; ++i) {
                weight_fp16[i] = float16_t(weight_fp32[i]);
            }
            for (size_t i = 0; i < bias_size; ++i) {
                bias_fp16[i] = float16_t(bias_fp32[i]);
            }
            if (RunTest(grad_y_fp16, x_fp16, weight_fp16, bias_fp16,
                        tc.B, tc.T, tc.D, tc.kW, dtype, stream,
                        tc.rtol, tc.atol, tc.matched_ratio_threshold, tc.max_abs_error_limit)) {
                passed++;
            } else {
                failed++;
            }
        } else {
            std::vector<bfloat16_t> grad_y_bf16(input_size);
            std::vector<bfloat16_t> x_bf16(input_size);
            std::vector<bfloat16_t> weight_bf16(weight_size);
            std::vector<bfloat16_t> bias_bf16(bias_size);
            for (size_t i = 0; i < input_size; ++i) {
                grad_y_bf16[i] = bfloat16_t(grad_y_fp32[i]);
                x_bf16[i] = bfloat16_t(x_fp32[i]);
            }
            for (size_t i = 0; i < weight_size; ++i) {
                weight_bf16[i] = bfloat16_t(weight_fp32[i]);
            }
            for (size_t i = 0; i < bias_size; ++i) {
                bias_bf16[i] = bfloat16_t(bias_fp32[i]);
            }
            if (RunTest(grad_y_bf16, x_bf16, weight_bf16, bias_bf16,
                        tc.B, tc.T, tc.D, tc.kW, dtype, stream,
                        tc.rtol, tc.atol, tc.matched_ratio_threshold, tc.max_abs_error_limit)) {
                passed++;
            } else {
                failed++;
            }
        }
#endif
    }

#ifndef USE_MOCK_ACLNN
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
#endif

    LOG_PRINT("\n========================================");
    LOG_PRINT("测试报告");
    LOG_PRINT("========================================");
    LOG_PRINT("总计: %d", passed + failed);
    LOG_PRINT("通过: %d", passed);
    LOG_PRINT("失败: %d", failed);
    LOG_PRINT("========================================\n");

    return failed == 0 ? 0 : 1;
}
