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
#include <cstdlib>
#include <ctime>
#include <gtest/gtest.h>
#include "acl/acl.h"
#include "../../../../../mhc_pre_sinkhorn_backward/op_host/op_api/aclnn_mhc_pre_sinkhorn_backward.h"
#include "../../../../op_host/op_api/aclnn_mhc_pre_clamp_sinkhorn_backward.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

int InitAcl(int32_t device_id, aclrtContext &context, aclrtStream &stream)
{
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed, error: %d\n", ret); return -1);

    ret = aclrtSetDevice(device_id);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed, error: %d\n", ret); return -1);

    ret = aclrtCreateContext(&context, device_id);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed, error: %d\n", ret); return -1);

    ret = aclrtSetCurrentContext(context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed, error: %d\n", ret); return -1);

    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed, error: %d\n", ret); return -1);

    return 0;
}

int CreateAclTensorFloat32(const std::vector<float> &host_data, const std::vector<int64_t> &shape, void *&device_addr,
                           aclTensor *&tensor)
{
    int64_t size = GetShapeSize(shape) * sizeof(float);

    aclError ret = aclrtMalloc(&device_addr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    ret = aclrtMemcpy(device_addr, size, host_data.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret); return -1);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

int CreateAclTensorBfloat16(const std::vector<float> &host_data, const std::vector<int64_t> &shape, void *&device_addr,
                            aclTensor *&tensor)
{
    int64_t size = GetShapeSize(shape);
    std::vector<uint16_t> host_data_bf16(size);
    for (int64_t i = 0; i < size; ++i) {
        host_data_bf16[i] = static_cast<uint16_t>(static_cast<int16_t>(host_data[i] * 32767));
    }

    int64_t byte_size = size * sizeof(uint16_t);

    aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    ret = aclrtMemcpy(device_addr, byte_size, host_data_bf16.data(), byte_size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret); return -1);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_BF16, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

int CreateAclTensorBfloat16Output(const std::vector<int64_t> &shape, void *&device_addr, aclTensor *&tensor)
{
    int64_t size = GetShapeSize(shape);
    int64_t byte_size = size * sizeof(uint16_t);

    aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_BF16, nullptr, 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

int CreateAclTensorFloat32Output(const std::vector<int64_t> &shape, void *&device_addr, aclTensor *&tensor)
{
    int64_t size = GetShapeSize(shape);
    int64_t byte_size = size * sizeof(float);

    aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); return -1);

    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, shape.data(),
                             shape.size(), device_addr);
    CHECK_RET(tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);

    return 0;
}

struct BackwardTensors {
    void *grad_hin_addr = nullptr, *grad_h_post_addr = nullptr, *grad_h_res_addr = nullptr;
    void *x_addr = nullptr, *phi_addr = nullptr, *alpha_addr = nullptr, *bias_addr = nullptr;
    void *h_pre_addr = nullptr, *hc_before_norm_addr = nullptr, *inv_rms_addr = nullptr;
    void *sum_out_addr = nullptr, *norm_out_addr = nullptr;
    void *grad_x_addr = nullptr, *grad_phi_addr = nullptr, *grad_alpha_addr = nullptr, *grad_bias_addr = nullptr;
    void *reduce_out_addr = nullptr;
    aclTensor *grad_hin = nullptr, *grad_h_post = nullptr, *grad_h_res = nullptr;
    aclTensor *x = nullptr, *phi = nullptr, *alpha = nullptr, *bias = nullptr;
    aclTensor *h_pre = nullptr, *hc_before_norm = nullptr, *inv_rms = nullptr;
    aclTensor *sum_out = nullptr, *norm_out = nullptr;
    aclTensor *grad_x = nullptr, *grad_phi = nullptr, *grad_alpha = nullptr, *grad_bias = nullptr;
    aclTensor *reduce_out = nullptr;
};

int CreateBackwardInputTensors(const std::vector<float> &grad_hin_host_data,
                               const std::vector<float> &grad_h_post_host_data,
                               const std::vector<float> &grad_h_res_host_data,
                               const std::vector<float> &x_host_data,
                               const std::vector<float> &phi_host_data,
                               const std::vector<float> &alpha_host_data,
                               const std::vector<float> &bias_host_data,
                               const std::vector<float> &h_pre_host_data,
                               const std::vector<float> &hc_before_norm_host_data,
                               const std::vector<float> &inv_rms_host_data,
                               const std::vector<float> &sum_out_host_data,
                               const std::vector<float> &norm_out_host_data,
                               const std::vector<float> &reduce_out_host_data,
                               const std::vector<int64_t> &grad_hin_shape,
                               const std::vector<int64_t> &grad_h_post_shape,
                               const std::vector<int64_t> &grad_h_res_shape,
                               const std::vector<int64_t> &x_shape,
                               const std::vector<int64_t> &phi_shape,
                               const std::vector<int64_t> &alpha_shape,
                               const std::vector<int64_t> &bias_shape,
                               const std::vector<int64_t> &h_pre_shape,
                               const std::vector<int64_t> &hc_before_norm_shape,
                               const std::vector<int64_t> &inv_rms_shape,
                               const std::vector<int64_t> &sum_out_shape,
                               const std::vector<int64_t> &norm_out_shape,
                               const std::vector<int64_t> &reduce_out_shape,
                               BackwardTensors &tensors)
{
    int ret = CreateAclTensorBfloat16(grad_hin_host_data, grad_hin_shape, tensors.grad_hin_addr, tensors.grad_hin);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_hin_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(grad_h_post_host_data, grad_h_post_shape, tensors.grad_h_post_addr, tensors.grad_h_post);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_h_post_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(grad_h_res_host_data, grad_h_res_shape, tensors.grad_h_res_addr, tensors.grad_h_res);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_h_res_tensor failed\n"); return -1);
    ret = CreateAclTensorBfloat16(x_host_data, x_shape, tensors.x_addr, tensors.x);
    CHECK_RET(ret == 0, LOG_PRINT("Create x_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(phi_host_data, phi_shape, tensors.phi_addr, tensors.phi);
    CHECK_RET(ret == 0, LOG_PRINT("Create phi_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(alpha_host_data, alpha_shape, tensors.alpha_addr, tensors.alpha);
    CHECK_RET(ret == 0, LOG_PRINT("Create alpha_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(bias_host_data, bias_shape, tensors.bias_addr, tensors.bias);
    CHECK_RET(ret == 0, LOG_PRINT("Create bias_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(h_pre_host_data, h_pre_shape, tensors.h_pre_addr, tensors.h_pre);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_pre_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(hc_before_norm_host_data, hc_before_norm_shape, tensors.hc_before_norm_addr, tensors.hc_before_norm);
    CHECK_RET(ret == 0, LOG_PRINT("Create hc_before_norm_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(inv_rms_host_data, inv_rms_shape, tensors.inv_rms_addr, tensors.inv_rms);
    CHECK_RET(ret == 0, LOG_PRINT("Create inv_rms_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(sum_out_host_data, sum_out_shape, tensors.sum_out_addr, tensors.sum_out);
    CHECK_RET(ret == 0, LOG_PRINT("Create sum_out_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(norm_out_host_data, norm_out_shape, tensors.norm_out_addr, tensors.norm_out);
    CHECK_RET(ret == 0, LOG_PRINT("Create norm_out_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(reduce_out_host_data, reduce_out_shape, tensors.reduce_out_addr, tensors.reduce_out);
    CHECK_RET(ret == 0, LOG_PRINT("Create reduce_out_tensor failed\n"); return -1);
    return 0;
}

int CreateBackwardOutputTensors(const std::vector<int64_t> &grad_x_shape,
                                const std::vector<int64_t> &grad_phi_shape,
                                const std::vector<int64_t> &grad_alpha_shape,
                                const std::vector<int64_t> &grad_bias_shape,
                                BackwardTensors &tensors)
{
    int ret = CreateAclTensorBfloat16Output(grad_x_shape, tensors.grad_x_addr, tensors.grad_x);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_x_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(grad_phi_shape, tensors.grad_phi_addr, tensors.grad_phi);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_phi_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(grad_alpha_shape, tensors.grad_alpha_addr, tensors.grad_alpha);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_alpha_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(grad_bias_shape, tensors.grad_bias_addr, tensors.grad_bias);
    CHECK_RET(ret == 0, LOG_PRINT("Create grad_bias_tensor failed\n"); return -1);
    return 0;
}

void DestroyBackwardTensors(BackwardTensors &tensors)
{
    aclDestroyTensor(tensors.grad_hin);
    aclDestroyTensor(tensors.grad_h_post);
    aclDestroyTensor(tensors.grad_h_res);
    aclDestroyTensor(tensors.x);
    aclDestroyTensor(tensors.phi);
    aclDestroyTensor(tensors.alpha);
    aclDestroyTensor(tensors.bias);
    aclDestroyTensor(tensors.h_pre);
    aclDestroyTensor(tensors.hc_before_norm);
    aclDestroyTensor(tensors.inv_rms);
    aclDestroyTensor(tensors.sum_out);
    aclDestroyTensor(tensors.norm_out);
    aclDestroyTensor(tensors.grad_x);
    aclDestroyTensor(tensors.grad_phi);
    aclDestroyTensor(tensors.grad_alpha);
    aclDestroyTensor(tensors.grad_bias);
    aclDestroyTensor(tensors.reduce_out);
}

void FreeBackwardDeviceMemory(BackwardTensors &tensors)
{
    aclrtFree(tensors.grad_hin_addr);
    aclrtFree(tensors.grad_h_post_addr);
    aclrtFree(tensors.grad_h_res_addr);
    aclrtFree(tensors.x_addr);
    aclrtFree(tensors.phi_addr);
    aclrtFree(tensors.alpha_addr);
    aclrtFree(tensors.bias_addr);
    aclrtFree(tensors.h_pre_addr);
    aclrtFree(tensors.hc_before_norm_addr);
    aclrtFree(tensors.inv_rms_addr);
    aclrtFree(tensors.sum_out_addr);
    aclrtFree(tensors.norm_out_addr);
    aclrtFree(tensors.grad_x_addr);
    aclrtFree(tensors.grad_phi_addr);
    aclrtFree(tensors.grad_alpha_addr);
    aclrtFree(tensors.grad_bias_addr);
    aclrtFree(tensors.reduce_out_addr);
}

template <typename T>
struct CompareResult {
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    int64_t max_abs_idx = -1;
    int64_t max_rel_idx = -1;
};

template <typename T>
CompareResult<T> CompareArrays(const T *ref, const T *tgt, int64_t size, double rtol = 1e-3, double atol = 1e-4)
{
    CompareResult<T> result;
    for (int64_t i = 0; i < size; ++i) {
        double abs_diff = std::abs(static_cast<double>(tgt[i]) - static_cast<double>(ref[i]));
        double rel_diff = (std::abs(static_cast<double>(ref[i])) > 1e-8)
                              ? abs_diff / std::abs(static_cast<double>(ref[i]))
                              : abs_diff;
        if (abs_diff > result.max_abs_diff) {
            result.max_abs_diff = abs_diff;
            result.max_abs_idx = i;
        }
        if (rel_diff > result.max_rel_diff) {
            result.max_rel_diff = rel_diff;
            result.max_rel_idx = i;
        }
    }
    return result;
}

template <typename T>
bool AssertPrecision(const T *ref, const T *tgt, int64_t size, const std::string &name,
                     double rtol = 1e-3, double atol = 1e-4)
{
    auto result = CompareArrays(ref, tgt, size, rtol, atol);
    bool pass = (result.max_abs_diff <= atol || result.max_rel_diff <= rtol);
    LOG_PRINT("[%s] max_abs_diff=%.6e at idx=%ld, max_rel_diff=%.6e at idx=%ld, %s\n",
              name.c_str(), result.max_abs_diff, result.max_abs_idx,
              result.max_rel_diff, result.max_rel_idx,
              pass ? "PASS" : "FAIL");
    return pass;
}

int ReadTensorFromDevice(void *device_addr, void *host_data, int64_t size, aclDataType data_type)
{
    int64_t byte_size = 0;
    if (data_type == ACL_BF16) {
        byte_size = size * sizeof(uint16_t);
    } else if (data_type == ACL_FLOAT) {
        byte_size = size * sizeof(float);
    } else {
        LOG_PRINT("Unsupported data type\n");
        return -1;
    }
    aclError ret = aclrtMemcpy(host_data, byte_size, device_addr, byte_size, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy from device failed, error: %d\n", ret); return -1);
    return 0;
}

class MhcPreClampSinkhornBackwardPrecisionTest : public testing::Test {
protected:
    aclrtContext context_ = nullptr;
    aclrtStream stream_ = nullptr;
    int32_t device_id_ = 0;

    void SetUp() override
    {
        int ret = InitAcl(device_id_, context_, stream_);
        ASSERT_EQ(ret, 0) << "InitAcl failed";
    }

    void TearDown() override
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
        }
        if (context_ != nullptr) {
            aclrtDestroyContext(context_);
        }
        aclrtResetDevice(device_id_);
        aclFinalize();
    }
};

TEST_F(MhcPreClampSinkhornBackwardPrecisionTest, CompareWithSinkhornBackward)
{
    const int64_t batchSize = 1;
    const int64_t seqLength = 64;
    const int64_t n = 1;
    const int64_t c = 128;

    const std::vector<int64_t> bf16_shape = {batchSize, seqLength, n, c};
    const std::vector<int64_t> fp32_shape = {batchSize, seqLength, n, c};
    const std::vector<int64_t> reduce_out_shape = {batchSize, seqLength, n, 1};

    int64_t bf16_size = GetShapeSize(bf16_shape);
    int64_t fp32_size = GetShapeSize(fp32_shape);
    int64_t reduce_out_size = GetShapeSize(reduce_out_shape);

    std::srand(std::time(nullptr));

    auto gen_float = []() { return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f; };

    std::vector<float> grad_hin_host(bf16_size);
    std::vector<float> grad_h_post_host(fp32_size);
    std::vector<float> grad_h_res_host(fp32_size);
    std::vector<float> x_host(bf16_size);
    std::vector<float> phi_host(fp32_size);
    std::vector<float> alpha_host(fp32_size);
    std::vector<float> bias_host(fp32_size);
    std::vector<float> h_pre_host(fp32_size);
    std::vector<float> hc_before_norm_host(fp32_size);
    std::vector<float> inv_rms_host(fp32_size);
    std::vector<float> sum_out_host(fp32_size);
    std::vector<float> norm_out_host(fp32_size);
    std::vector<float> reduce_out_host(reduce_out_size, 0.0f);

    for (int64_t i = 0; i < bf16_size; ++i) {
        grad_hin_host[i] = gen_float();
        x_host[i] = gen_float();
    }
    for (int64_t i = 0; i < fp32_size; ++i) {
        grad_h_post_host[i] = gen_float();
        grad_h_res_host[i] = gen_float();
        phi_host[i] = gen_float();
        alpha_host[i] = gen_float();
        bias_host[i] = gen_float();
        h_pre_host[i] = gen_float();
        hc_before_norm_host[i] = gen_float();
        inv_rms_host[i] = gen_float();
        sum_out_host[i] = gen_float();
        norm_out_host[i] = gen_float();
    }

    BackwardTensors ref_tensors;
    BackwardTensors tgt_tensors;

    int ret = 0;

    ret = CreateBackwardInputTensors(grad_hin_host, grad_h_post_host, grad_h_res_host,
                                     x_host, phi_host, alpha_host, bias_host,
                                     h_pre_host, hc_before_norm_host, inv_rms_host,
                                     sum_out_host, norm_out_host, reduce_out_host,
                                     bf16_shape, fp32_shape, fp32_shape,
                                     bf16_shape, fp32_shape, fp32_shape, fp32_shape,
                                     fp32_shape, fp32_shape, fp32_shape,
                                     fp32_shape, fp32_shape, reduce_out_shape,
                                     ref_tensors);
    ASSERT_EQ(ret, 0) << "Create ref input tensors failed";

    ret = CreateBackwardInputTensors(grad_hin_host, grad_h_post_host, grad_h_res_host,
                                     x_host, phi_host, alpha_host, bias_host,
                                     h_pre_host, hc_before_norm_host, inv_rms_host,
                                     sum_out_host, norm_out_host, reduce_out_host,
                                     bf16_shape, fp32_shape, fp32_shape,
                                     bf16_shape, fp32_shape, fp32_shape, fp32_shape,
                                     fp32_shape, fp32_shape, fp32_shape,
                                     fp32_shape, fp32_shape, reduce_out_shape,
                                     tgt_tensors);
    ASSERT_EQ(ret, 0) << "Create tgt input tensors failed";

    ret = CreateBackwardOutputTensors(bf16_shape, fp32_shape, fp32_shape, fp32_shape, ref_tensors);
    ASSERT_EQ(ret, 0) << "Create ref output tensors failed";

    ret = CreateBackwardOutputTensors(bf16_shape, fp32_shape, fp32_shape, fp32_shape, tgt_tensors);
    ASSERT_EQ(ret, 0) << "Create tgt output tensors failed";

    const double hc_eps = 1e-6;
    const double min_val = 0.0;
    const double max_val = 1.0;

    aclnnStatus aclnn_ret = ACL_SUCCESS;
    void *ref_workspace = nullptr;
    void *tgt_workspace = nullptr;
    aclOpExecutor *ref_executor = nullptr;
    aclOpExecutor *tgt_executor = nullptr;
    uint64_t ref_workspace_size = 0;
    uint64_t tgt_workspace_size = 0;

    aclnn_ret = aclnnMhcPreSinkhornBackwardGetWorkspaceSize(
        ref_tensors.grad_hin, ref_tensors.grad_h_post, ref_tensors.grad_h_res,
        ref_tensors.x, ref_tensors.phi, ref_tensors.alpha, ref_tensors.bias,
        ref_tensors.h_pre, ref_tensors.hc_before_norm, ref_tensors.inv_rms,
        ref_tensors.sum_out, ref_tensors.norm_out,
        hc_eps,
        ref_tensors.grad_x, ref_tensors.grad_phi, ref_tensors.grad_alpha, ref_tensors.grad_bias,
        &ref_workspace_size, &ref_executor);
    ASSERT_EQ(aclnn_ret, ACL_SUCCESS) << "ref aclnnMhcPreSinkhornBackwardGetWorkspaceSize failed";

    aclnn_ret = aclnnMhcPreClampSinkhornBackwardGetWorkspaceSize(
        tgt_tensors.grad_hin, tgt_tensors.grad_h_post, tgt_tensors.grad_h_res,
        tgt_tensors.x, tgt_tensors.phi, tgt_tensors.alpha, tgt_tensors.bias,
        tgt_tensors.h_pre, tgt_tensors.hc_before_norm, tgt_tensors.inv_rms,
        tgt_tensors.sum_out, tgt_tensors.norm_out,
        tgt_tensors.reduce_out,
        min_val, max_val, hc_eps,
        tgt_tensors.grad_x, tgt_tensors.grad_phi, tgt_tensors.grad_alpha, tgt_tensors.grad_bias,
        &tgt_workspace_size, &tgt_executor);
    ASSERT_EQ(aclnn_ret, ACL_SUCCESS) << "tgt aclnnMhcPreClampSinkhornBackwardGetWorkspaceSize failed";

    if (ref_workspace_size > 0) {
        aclrtMalloc(&ref_workspace, ref_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    if (tgt_workspace_size > 0) {
        aclrtMalloc(&tgt_workspace, tgt_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    }

    aclnn_ret = aclnnMhcPreSinkhornBackward(ref_workspace, ref_workspace_size, ref_executor, stream_);
    ASSERT_EQ(aclnn_ret, ACL_SUCCESS) << "ref aclnnMhcPreSinkhornBackward failed";

    aclnn_ret = aclnnMhcPreClampSinkhornBackward(tgt_workspace, tgt_workspace_size, tgt_executor, stream_);
    ASSERT_EQ(aclnn_ret, ACL_SUCCESS) << "tgt aclnnMhcPreClampSinkhornBackward failed";

    aclrtSynchronizeStream(stream_);

    int64_t bf16_byte_size = bf16_size * sizeof(uint16_t);
    int64_t fp32_byte_size = fp32_size * sizeof(float);

    std::vector<uint16_t> ref_grad_x_bf16(bf16_size);
    std::vector<uint16_t> tgt_grad_x_bf16(bf16_size);
    std::vector<float> ref_grad_phi(fp32_size);
    std::vector<float> tgt_grad_phi(fp32_size);
    std::vector<float> ref_grad_alpha(fp32_size);
    std::vector<float> tgt_grad_alpha(fp32_size);
    std::vector<float> ref_grad_bias(fp32_size);
    std::vector<float> tgt_grad_bias(fp32_size);

    aclrtMemcpy(ref_grad_x_bf16.data(), bf16_byte_size, ref_tensors.grad_x_addr, bf16_byte_size, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(tgt_grad_x_bf16.data(), bf16_byte_size, tgt_tensors.grad_x_addr, bf16_byte_size, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(ref_grad_phi.data(), fp32_byte_size, ref_tensors.grad_phi_addr, fp32_byte_size, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(tgt_grad_phi.data(), fp32_byte_size, tgt_tensors.grad_phi_addr, fp32_byte_size, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(ref_grad_alpha.data(), fp32_byte_size, ref_tensors.grad_alpha_addr, fp32_byte_size, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(tgt_grad_alpha.data(), fp32_byte_size, tgt_tensors.grad_alpha_addr, fp32_byte_size, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(ref_grad_bias.data(), fp32_byte_size, ref_tensors.grad_bias_addr, fp32_byte_size, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(tgt_grad_bias.data(), fp32_byte_size, tgt_tensors.grad_bias_addr, fp32_byte_size, ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<float> ref_grad_x(bf16_size);
    std::vector<float> tgt_grad_x(bf16_size);
    for (int64_t i = 0; i < bf16_size; ++i) {
        ref_grad_x[i] = static_cast<float>(static_cast<int16_t>(ref_grad_x_bf16[i])) / 32767.0f;
        tgt_grad_x[i] = static_cast<float>(static_cast<int16_t>(tgt_grad_x_bf16[i])) / 32767.0f;
    }

    bool all_pass = true;

    all_pass &= AssertPrecision<float>(ref_grad_x.data(), tgt_grad_x.data(), bf16_size, "grad_x", 1e-2, 1e-2);
    all_pass &= AssertPrecision<float>(ref_grad_phi.data(), tgt_grad_phi.data(), fp32_size, "grad_phi");
    all_pass &= AssertPrecision<float>(ref_grad_alpha.data(), tgt_grad_alpha.data(), fp32_size, "grad_alpha");
    all_pass &= AssertPrecision<float>(ref_grad_bias.data(), tgt_grad_bias.data(), fp32_size, "grad_bias");

    if (ref_workspace != nullptr) {
        aclrtFree(ref_workspace);
    }
    if (tgt_workspace != nullptr) {
        aclrtFree(tgt_workspace);
    }

    DestroyBackwardTensors(ref_tensors);
    DestroyBackwardTensors(tgt_tensors);
    FreeBackwardDeviceMemory(ref_tensors);
    FreeBackwardDeviceMemory(tgt_tensors);

    EXPECT_TRUE(all_pass) << "Precision comparison failed";
}