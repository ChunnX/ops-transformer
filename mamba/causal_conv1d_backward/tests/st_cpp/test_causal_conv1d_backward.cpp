/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_causal_conv1d_backward.cpp
 * @brief CausalConv1dBackward 算子 ST 测试
 *
 * CPU golden 与标准 06 §七 参考实现同语义：qsl 分段、weight[W-1] 配当前 token、
 * 无 bias、activation silu/none，全程 fp32 累加后 cast 一次。
 * Mock 模式（--mock）：仅 CPU golden 自检；Real 模式（默认）：NPU 执行比对。
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <functional>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <cstdlib>

#ifndef USE_MOCK_ACLNN
#include "acl/acl.h"
#include "aclnnop/aclnn_causal_conv1d_backward.h"
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
        fexp = 0; fmantissa = 0;
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
// CPU Golden 计算函数（标准 06 §七 参考实现同语义，全程 fp32 累加）
//
// token 轴展平为 [tokenCnt, D]，按 qsl 分段（qsl 为空时整轴一段）；
// weight[W-1] 配当前 token x[t]（与正向 causal_conv1d_fn 同序）；
// 段内左补 W-1 个零（冷启动），右侧补零（右边界截断）。
// ============================================================================

template<typename DType>
void ComputeGoldenCausalConv1dBackward(
    const DType* grad_y, const DType* x, const DType* weight,
    const int32_t* qsl, int64_t seqCnt,
    DType* grad_x, DType* grad_weight,
    int64_t tokenCnt, int64_t D, int64_t kW, bool silu)
{
    // 转换为 fp32
    std::vector<float> gy_fp32(tokenCnt * D);
    std::vector<float> x_fp32(tokenCnt * D);
    std::vector<float> w_fp32(kW * D);
    for (int64_t i = 0; i < tokenCnt * D; ++i) {
        gy_fp32[i] = static_cast<float>(grad_y[i]);
        x_fp32[i] = static_cast<float>(x[i]);
    }
    for (int64_t i = 0; i < kW * D; ++i) {
        w_fp32[i] = static_cast<float>(weight[i]);
    }

    std::vector<float> gx_fp32(tokenCnt * D, 0.0f);
    std::vector<float> gw_fp32(kW * D, 0.0f);

    // 分段边界
    std::vector<int64_t> bounds;
    if (qsl == nullptr || seqCnt <= 1) {
        bounds = {0, tokenCnt};
    } else {
        for (int64_t i = 0; i <= seqCnt; ++i) {
            bounds.push_back(static_cast<int64_t>(qsl[i]));
        }
    }

    for (size_t seg = 0; seg + 1 < bounds.size(); ++seg) {
        int64_t lo = bounds[seg];
        int64_t hi = bounds[seg + 1];
        int64_t n = hi - lo;
        if (n <= 0) {
            continue;
        }

        // 段内左补 W-1 个零：padded[k] = x[lo + k - (W-1)]（k >= W-1 时）
        std::vector<float> padded((n + kW - 1) * D, 0.0f);
        for (int64_t t = 0; t < n; ++t) {
            for (int64_t d = 0; d < D; ++d) {
                padded[(t + kW - 1) * D + d] = x_fp32[(lo + t) * D + d];
            }
        }

        // 重算 pre = Σ_k weight[k]·padded[k : k+n]
        std::vector<float> pre(n * D, 0.0f);
        for (int64_t k = 0; k < kW; ++k) {
            for (int64_t t = 0; t < n; ++t) {
                for (int64_t d = 0; d < D; ++d) {
                    pre[t * D + d] += w_fp32[k * D + d] * padded[(k + t) * D + d];
                }
            }
        }

        // dout = grad_y · Act'(pre)
        std::vector<float> dout(n * D, 0.0f);
        for (int64_t i = 0; i < n * D; ++i) {
            if (!silu) {
                dout[i] = gy_fp32[(lo) * D + i];
            } else {
                float z = pre[i];
                float sig = 1.0f / (1.0f + std::exp(-z));
                dout[i] = gy_fp32[lo * D + i] * (sig * (1.0f + z * (1.0f - sig)));
            }
        }

        // grad_x[s] = Σ_k weight[k]·dout[s + W-1-k]（右侧补零 = 段右边界）
        std::vector<float> dout_pad((n + kW - 1) * D, 0.0f);
        for (int64_t t = 0; t < n; ++t) {
            for (int64_t d = 0; d < D; ++d) {
                dout_pad[t * D + d] = dout[t * D + d];
            }
        }
        for (int64_t k = 0; k < kW; ++k) {
            for (int64_t t = 0; t < n; ++t) {
                for (int64_t d = 0; d < D; ++d) {
                    gx_fp32[(lo + t) * D + d] +=
                        w_fp32[k * D + d] * dout_pad[(kW - 1 - k + t) * D + d];
                }
            }
        }

        // grad_weight[k] = Σ_t dout[t]·x[t-(W-1-k)] = Σ_t dout[t]·padded[k+t]
        for (int64_t k = 0; k < kW; ++k) {
            for (int64_t t = 0; t < n; ++t) {
                for (int64_t d = 0; d < D; ++d) {
                    gw_fp32[k * D + d] += dout[t * D + d] * padded[(k + t) * D + d];
                }
            }
        }
    }

    // Cast 回原始 dtype（一次性）
    for (int64_t i = 0; i < tokenCnt * D; ++i) {
        grad_x[i] = static_cast<DType>(gx_fp32[i]);
    }
    for (int64_t i = 0; i < kW * D; ++i) {
        grad_weight[i] = static_cast<DType>(gw_fp32[i]);
    }
}

// ============================================================================
// 精度比对函数（混合容差）
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

    int64_t tokenCnt = 8, D = 12288, kW = 4;
    size_t input_size = tokenCnt * D;
    size_t weight_size = kW * D;

    std::vector<float> grad_y(input_size);
    std::vector<float> x(input_size);
    std::vector<float> weight(weight_size);
    std::vector<float> grad_x(input_size);
    std::vector<float> grad_weight(weight_size);

    for (size_t i = 0; i < input_size; ++i) {
        grad_y[i] = 0.01f * (static_cast<int>(i % 10) - 5);
        x[i] = 0.1f * (static_cast<int>(i % 8) - 4);
    }
    for (size_t i = 0; i < weight_size; ++i) {
        weight[i] = 0.01f * (static_cast<int>(i % 10) - 5);
    }

    LOG_PRINT("\n测试 1: 3D 单段（qsl=None, tokenCnt=%ld, D=%ld）", tokenCnt, D);
    ComputeGoldenCausalConv1dBackward(
        grad_y.data(), x.data(), weight.data(), nullptr, 1,
        grad_x.data(), grad_weight.data(), tokenCnt, D, kW, true);

    bool gx_nonzero = false, gw_nonzero = false;
    for (size_t i = 0; i < input_size && !gx_nonzero; ++i) {
        if (std::abs(grad_x[i]) > 1e-6f) gx_nonzero = true;
    }
    for (size_t i = 0; i < weight_size && !gw_nonzero; ++i) {
        if (std::abs(grad_weight[i]) > 1e-6f) gw_nonzero = true;
    }
    LOG_PRINT("  结果: %s", (gx_nonzero && gw_nonzero) ? "PASS" : "FAIL");

    LOG_PRINT("\n测试 2: grad_y 全零（应输出全零梯度）");
    std::fill(grad_y.begin(), grad_y.end(), 0.0f);
    ComputeGoldenCausalConv1dBackward(
        grad_y.data(), x.data(), weight.data(), nullptr, 1,
        grad_x.data(), grad_weight.data(), tokenCnt, D, kW, true);
    bool all_zero = true;
    for (size_t i = 0; i < input_size && all_zero; ++i) {
        if (std::abs(grad_x[i]) > 1e-6f) all_zero = false;
    }
    for (size_t i = 0; i < weight_size && all_zero; ++i) {
        if (std::abs(grad_weight[i]) > 1e-6f) all_zero = false;
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

struct TestCase {
    const char* name;
    int64_t tokenCnt;          // 展平 token 数
    int64_t dim;
    int64_t kW;
    bool is3d;                 // true: 3D [B,L,D]（内部按 B*L 记）；false: 2D + qsl
    int64_t b = 1;             // 3D 时的 B
    const int32_t* qsl = nullptr;  // 非 3D-None 时的分段
    int64_t seqCnt = 1;
    bool is_fp16 = false;
    const char* activation = "silu";
    bool zero_grad_y = false;
    double rtol;
    double atol;
    double matched_ratio_threshold;
    double max_abs_error_limit;
};

namespace {
const int32_t QSL_VARLEN[] = {0, 5, 13, 20};          // 3 段：5/8/7
const int32_t QSL_PACKED_3D[] = {0, 3, 9, 16};        // 3D + 跨行分段
const int32_t QSL_W2[] = {0, 11, 24};                 // W=2 两段
} // namespace

std::vector<TestCase> GetTestCases() {
    return {
        // bf16 主力 dtype（ Ascend950 训练配置）
        {"BC1_bf16 最小 batch/seq (B=1,T=1)", 1, 12288, 4, true, 1, nullptr, 1, false, "silu", false, 4e-3, 4e-3, 0.99, 1.0},
        {"BC2_bf16 T=kW 边界 (B=1,T=4)", 4, 12288, 4, true, 1, nullptr, 1, false, "silu", false, 4e-3, 4e-3, 0.99, 1.0},
        {"BC3_bf16 典型 shape (B=2,T=16)", 32, 12288, 4, true, 2, nullptr, 1, false, "silu", false, 4e-3, 4e-3, 0.99, 1.0},
        {"EI1_bf16 grad_y 全零", 8, 12288, 4, true, 1, nullptr, 1, false, "silu", true, 4e-3, 4e-3, 0.99, 1.0},
        {"NC1_bf16 变长 2D+qsl (5/8/7, W=4)", 20, 12288, 4, false, 1, QSL_VARLEN, 3, false, "silu", false, 4e-3, 4e-3, 0.99, 1.0},
        {"NC2_bf16 3D+qsl 跨行分段 (B=1,T=16)", 16, 12288, 4, true, 1, QSL_PACKED_3D, 3, false, "silu", false, 4e-3, 4e-3, 0.99, 1.0},
        {"NC3_bf16 activation=none (B=2,T=8)", 16, 12288, 4, true, 2, nullptr, 1, false, "none", false, 4e-3, 4e-3, 0.99, 1.0},
        {"NC4_bf16 W=2 变长 (11/13)", 24, 12288, 2, false, 1, QSL_W2, 2, false, "silu", false, 4e-3, 4e-3, 0.99, 1.0},
        {"NC5_fp16 变长 2D+qsl (5/8/7, W=4)", 20, 12288, 4, false, 1, QSL_VARLEN, 3, true, "silu", false, 1e-3, 1e-3, 0.99, 0.1},
        {"NC6_bf16 尾 tile (D=4160, 尾=64)", 8, 4160, 4, true, 1, nullptr, 1, false, "silu", false, 4e-3, 4e-3, 0.99, 1.0},
    };
}

#ifdef USE_MOCK_ACLNN

template<typename DType>
bool RunTest(const std::vector<DType>& grad_y, const std::vector<DType>& x,
             const std::vector<DType>& weight, const TestCase& tc,
             const char* dtype) {
    LOG_PRINT("\n[Mock] 测试 - dtype=%s, %s, tokenCnt=%ld, D=%ld, W=%ld, act=%s",
              dtype, tc.is3d ? "3D" : "2D+qsl", tc.tokenCnt, tc.dim, tc.kW, tc.activation);

    size_t input_size = tc.tokenCnt * tc.dim;
    size_t weight_size = tc.kW * tc.dim;

    std::vector<DType> grad_x(input_size);
    std::vector<DType> grad_weight(weight_size);
    std::vector<DType> grad_x_golden(input_size);
    std::vector<DType> grad_weight_golden(weight_size);

    ComputeGoldenCausalConv1dBackward(
        grad_y.data(), x.data(), weight.data(), tc.qsl, tc.seqCnt,
        grad_x_golden.data(), grad_weight_golden.data(),
        tc.tokenCnt, tc.dim, tc.kW, std::string(tc.activation) == "silu");

    // Mock 模式：直接使用 golden 作为输出
    grad_x = grad_x_golden;
    grad_weight = grad_weight_golden;

    bool pass_gx = CompareResults(grad_x_golden.data(), grad_x.data(), input_size,
                                   tc.rtol, tc.atol, tc.matched_ratio_threshold, tc.max_abs_error_limit);
    bool pass_gw = CompareResults(grad_weight_golden.data(), grad_weight.data(), weight_size,
                                   tc.rtol, tc.atol, tc.matched_ratio_threshold, tc.max_abs_error_limit);
    return pass_gx && pass_gw;
}

#else

template<typename DType>
bool RunTest(const std::vector<DType>& grad_y, const std::vector<DType>& x,
             const std::vector<DType>& weight, const TestCase& tc,
             aclDataType dtype, aclrtStream stream) {
    LOG_PRINT("\n[Real] 测试 - %s, tokenCnt=%ld, D=%ld, W=%ld, act=%s",
              tc.is3d ? "3D" : "2D+qsl", tc.tokenCnt, tc.dim, tc.kW, tc.activation);

    size_t input_size = tc.tokenCnt * tc.dim;
    size_t weight_size = tc.kW * tc.dim;

    // 计算 golden
    std::vector<DType> grad_x_golden(input_size);
    std::vector<DType> grad_weight_golden(weight_size);
    ComputeGoldenCausalConv1dBackward(
        grad_y.data(), x.data(), weight.data(), tc.qsl, tc.seqCnt,
        grad_x_golden.data(), grad_weight_golden.data(),
        tc.tokenCnt, tc.dim, tc.kW, std::string(tc.activation) == "silu");

    // 输入 shape：3D [B, L, D]（L = tokenCnt / B）或 2D [tokenCnt, D]
    int64_t l = tc.is3d ? (tc.tokenCnt / tc.b) : tc.tokenCnt;
    std::vector<int64_t> input_shape = tc.is3d ? std::vector<int64_t>{tc.b, l, tc.dim}
                                               : std::vector<int64_t>{tc.tokenCnt, tc.dim};
    std::vector<int64_t> weight_shape = {tc.kW, tc.dim};
    std::vector<int64_t> qsl_shape = {tc.seqCnt + 1};

    void *gradY_dev = nullptr, *x_dev = nullptr, *weight_dev = nullptr, *qsl_dev = nullptr;
    void *gradX_dev = nullptr, *gradWeight_dev = nullptr;
    aclTensor *gradY_tensor = nullptr, *x_tensor = nullptr, *weight_tensor = nullptr, *qsl_tensor = nullptr;
    aclTensor *gradX_tensor = nullptr, *gradWeight_tensor = nullptr;
    auto cleanup = [&]() {
        if (gradY_dev) aclrtFree(gradY_dev);
        if (x_dev) aclrtFree(x_dev);
        if (weight_dev) aclrtFree(weight_dev);
        if (qsl_dev) aclrtFree(qsl_dev);
        if (gradX_dev) aclrtFree(gradX_dev);
        if (gradWeight_dev) aclrtFree(gradWeight_dev);
        if (gradY_tensor) aclDestroyTensor(gradY_tensor);
        if (x_tensor) aclDestroyTensor(x_tensor);
        if (weight_tensor) aclDestroyTensor(weight_tensor);
        if (qsl_tensor) aclDestroyTensor(qsl_tensor);
        if (gradX_tensor) aclDestroyTensor(gradX_tensor);
        if (gradWeight_tensor) aclDestroyTensor(gradWeight_tensor);
    };

    auto ret = CreateAclTensor(grad_y, input_shape, &gradY_dev, dtype, &gradY_tensor);
    if (ret != ACL_SUCCESS) { LOG_PRINT("  [ERROR] CreateAclTensor(gradY) failed: %d", ret); cleanup(); return false; }
    ret = CreateAclTensor(x, input_shape, &x_dev, dtype, &x_tensor);
    if (ret != ACL_SUCCESS) { LOG_PRINT("  [ERROR] CreateAclTensor(x) failed: %d", ret); cleanup(); return false; }
    ret = CreateAclTensor(weight, weight_shape, &weight_dev, dtype, &weight_tensor);
    if (ret != ACL_SUCCESS) { LOG_PRINT("  [ERROR] CreateAclTensor(weight) failed: %d", ret); cleanup(); return false; }
    if (tc.qsl != nullptr) {
        std::vector<int32_t> qsl_host(tc.qsl, tc.qsl + tc.seqCnt + 1);
        ret = CreateAclTensor(qsl_host, qsl_shape, &qsl_dev, ACL_INT32, &qsl_tensor);
        if (ret != ACL_SUCCESS) { LOG_PRINT("  [ERROR] CreateAclTensor(qsl) failed: %d", ret); cleanup(); return false; }
    }

    std::vector<DType> zero_in(input_size, DType(0));
    std::vector<DType> zero_w(weight_size, DType(0));
    ret = CreateAclTensor(zero_in, input_shape, &gradX_dev, dtype, &gradX_tensor);
    if (ret != ACL_SUCCESS) { LOG_PRINT("  [ERROR] CreateAclTensor(gradX) failed: %d", ret); cleanup(); return false; }
    ret = CreateAclTensor(zero_w, weight_shape, &gradWeight_dev, dtype, &gradWeight_tensor);
    if (ret != ACL_SUCCESS) { LOG_PRINT("  [ERROR] CreateAclTensor(gradWeight) failed: %d", ret); cleanup(); return false; }

    // 调用 ACLNN L2 API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto aclnnRet = aclnnCausalConv1dBackwardGetWorkspaceSize(
        gradY_tensor, x_tensor, weight_tensor, qsl_tensor, tc.activation,
        gradX_tensor, gradWeight_tensor, &workspaceSize, &executor);
    if (aclnnRet != ACL_SUCCESS) {
        LOG_PRINT("  [ERROR] aclnnCausalConv1dBackwardGetWorkspaceSize failed: %d", aclnnRet);
        cleanup();
        return false;
    }

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  [ERROR] workspace malloc failed: %d", ret);
            cleanup();
            return false;
        }
    }

    aclnnRet = aclnnCausalConv1dBackward(workspace, workspaceSize, executor, stream);
    if (aclnnRet != ACL_SUCCESS) {
        LOG_PRINT("  [ERROR] aclnnCausalConv1dBackward failed: %d", aclnnRet);
        if (workspace) aclrtFree(workspace);
        cleanup();
        return false;
    }

    aclrtSynchronizeStream(stream);

    // D2H
    std::vector<DType> grad_x_npu(input_size);
    std::vector<DType> grad_weight_npu(weight_size);
    aclrtMemcpy(grad_x_npu.data(), input_size * sizeof(DType), gradX_dev,
                input_size * sizeof(DType), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(grad_weight_npu.data(), weight_size * sizeof(DType), gradWeight_dev,
                weight_size * sizeof(DType), ACL_MEMCPY_DEVICE_TO_HOST);

    if (workspace) aclrtFree(workspace);
    cleanup();

    // 比对
    LOG_PRINT("  grad_x:");
    bool pass_gx = CompareResults(grad_x_golden.data(), grad_x_npu.data(), input_size,
                                   tc.rtol, tc.atol, tc.matched_ratio_threshold, tc.max_abs_error_limit);
    LOG_PRINT("  grad_weight:");
    bool pass_gw = CompareResults(grad_weight_golden.data(), grad_weight_npu.data(), weight_size,
                                   tc.rtol, tc.atol, tc.matched_ratio_threshold, tc.max_abs_error_limit);
    return pass_gx && pass_gw;
}

#endif

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char* argv[]) {
    LOG_PRINT("\n========================================");
    LOG_PRINT("causal_conv1d_backward 算子 ST 测试");
    LOG_PRINT("========================================");

#ifdef USE_MOCK_ACLNN
    LOG_PRINT("模式: Mock (CPU golden)");
#else
    LOG_PRINT("模式: Real (NPU)");
#endif

    TestGoldenCorrectness();

    int passed = 0, failed = 0;

#ifndef USE_MOCK_ACLNN
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

        size_t input_size = tc.tokenCnt * tc.dim;
        size_t weight_size = tc.kW * tc.dim;

        // 生成测试数据（缩小范围避免 sigmoid/exp 溢出）
        std::vector<float> grad_y_fp32(input_size);
        std::vector<float> x_fp32(input_size);
        std::vector<float> weight_fp32(weight_size);

        for (size_t i = 0; i < input_size; ++i) {
            grad_y_fp32[i] = tc.zero_grad_y ? 0.0f : (0.01f * (static_cast<int>(i % 10) - 5));
        }
        for (size_t i = 0; i < input_size; ++i) {
            x_fp32[i] = 0.1f * (static_cast<int>(i % 8) - 4);
        }
        for (size_t i = 0; i < weight_size; ++i) {
            weight_fp32[i] = 0.01f * (static_cast<int>(i % 10) - 5);
        }

        bool ok = false;
#ifdef USE_MOCK_ACLNN
        if (tc.is_fp16) {
            std::vector<float16_t> gy(input_size), xin(input_size);
            std::vector<float16_t> w(weight_size);
            for (size_t i = 0; i < input_size; ++i) { gy[i] = float16_t(grad_y_fp32[i]); xin[i] = float16_t(x_fp32[i]); }
            for (size_t i = 0; i < weight_size; ++i) { w[i] = float16_t(weight_fp32[i]); }
            ok = RunTest(gy, xin, w, tc, "float16");
        } else {
            std::vector<bfloat16_t> gy(input_size), xin(input_size);
            std::vector<bfloat16_t> w(weight_size);
            for (size_t i = 0; i < input_size; ++i) { gy[i] = bfloat16_t(grad_y_fp32[i]); xin[i] = bfloat16_t(x_fp32[i]); }
            for (size_t i = 0; i < weight_size; ++i) { w[i] = bfloat16_t(weight_fp32[i]); }
            ok = RunTest(gy, xin, w, tc, "bfloat16");
        }
#else
        aclDataType dtype = tc.is_fp16 ? ACL_FLOAT16 : ACL_BF16;
        if (tc.is_fp16) {
            std::vector<float16_t> gy(input_size), xin(input_size);
            std::vector<float16_t> w(weight_size);
            for (size_t i = 0; i < input_size; ++i) { gy[i] = float16_t(grad_y_fp32[i]); xin[i] = float16_t(x_fp32[i]); }
            for (size_t i = 0; i < weight_size; ++i) { w[i] = float16_t(weight_fp32[i]); }
            ok = RunTest(gy, xin, w, tc, dtype, stream);
        } else {
            std::vector<bfloat16_t> gy(input_size), xin(input_size);
            std::vector<bfloat16_t> w(weight_size);
            for (size_t i = 0; i < input_size; ++i) { gy[i] = bfloat16_t(grad_y_fp32[i]); xin[i] = bfloat16_t(x_fp32[i]); }
            for (size_t i = 0; i < weight_size; ++i) { w[i] = bfloat16_t(weight_fp32[i]); }
            ok = RunTest(gy, xin, w, tc, dtype, stream);
        }
#endif
        if (ok) {
            passed++;
        } else {
            failed++;
        }
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
