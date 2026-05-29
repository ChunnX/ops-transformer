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
#include "aclnnop/aclnn_mhc_pre_sinkhorn.h"
#include "aclnnop/aclnn_mhc_pre_clamp_sinkhorn.h"

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

struct Tensors {
    void *x_addr = nullptr, *phi_addr = nullptr, *alpha_addr = nullptr, *bias_addr = nullptr;
    void *hin_addr = nullptr, *h_post_addr = nullptr, *h_res_addr = nullptr;
    void *h_pre_addr = nullptr, *hc_before_norm_addr = nullptr, *inv_rms_addr = nullptr;
    void *sum_out_addr = nullptr, *norm_out_addr = nullptr, *reduce_out_addr = nullptr;
    aclTensor *x = nullptr, *phi = nullptr, *alpha = nullptr, *bias = nullptr;
    aclTensor *hin = nullptr, *h_post = nullptr, *h_res = nullptr;
    aclTensor *h_pre = nullptr, *hc_before_norm = nullptr, *inv_rms = nullptr;
    aclTensor *sum_out = nullptr, *norm_out = nullptr, *reduce_out = nullptr;
};

int CreateInputTensors(const std::vector<float> &x_host_data, const std::vector<float> &phi_host_data,
                       const std::vector<float> &alpha_host_data, const std::vector<float> &bias_host_data,
                       const std::vector<int64_t> &x_shape, const std::vector<int64_t> &phi_shape,
                       const std::vector<int64_t> &alpha_shape, const std::vector<int64_t> &bias_shape,
                       Tensors &tensors)
{
    int ret = CreateAclTensorBfloat16(x_host_data, x_shape, tensors.x_addr, tensors.x);
    CHECK_RET(ret == 0, LOG_PRINT("Create x_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(phi_host_data, phi_shape, tensors.phi_addr, tensors.phi);
    CHECK_RET(ret == 0, LOG_PRINT("Create phi_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(alpha_host_data, alpha_shape, tensors.alpha_addr, tensors.alpha);
    CHECK_RET(ret == 0, LOG_PRINT("Create alpha_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32(bias_host_data, bias_shape, tensors.bias_addr, tensors.bias);
    CHECK_RET(ret == 0, LOG_PRINT("Create bias_tensor failed\n"); return -1);
    return 0;
}

int CreateOutputTensors(const std::vector<int64_t> &hin_shape, const std::vector<int64_t> &h_post_shape,
                        const std::vector<int64_t> &h_res_shape, const std::vector<int64_t> &h_pre_shape,
                        const std::vector<int64_t> &hc_before_norm_shape, const std::vector<int64_t> &inv_rms_shape,
                        const std::vector<int64_t> &sum_out_shape, const std::vector<int64_t> &norm_out_shape,
                        const std::vector<int64_t> &reduce_out_shape,
                        Tensors &tensors)
{
    int ret = CreateAclTensorBfloat16Output(hin_shape, tensors.hin_addr, tensors.hin);
    CHECK_RET(ret == 0, LOG_PRINT("Create hin_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(h_post_shape, tensors.h_post_addr, tensors.h_post);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_post_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(h_res_shape, tensors.h_res_addr, tensors.h_res);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_res_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(h_pre_shape, tensors.h_pre_addr, tensors.h_pre);
    CHECK_RET(ret == 0, LOG_PRINT("Create h_pre_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(hc_before_norm_shape, tensors.hc_before_norm_addr, tensors.hc_before_norm);
    CHECK_RET(ret == 0, LOG_PRINT("Create hc_before_norm_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(inv_rms_shape, tensors.inv_rms_addr, tensors.inv_rms);
    CHECK_RET(ret == 0, LOG_PRINT("Create inv_rms_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(sum_out_shape, tensors.sum_out_addr, tensors.sum_out);
    CHECK_RET(ret == 0, LOG_PRINT("Create sum_out_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(norm_out_shape, tensors.norm_out_addr, tensors.norm_out);
    CHECK_RET(ret == 0, LOG_PRINT("Create norm_out_tensor failed\n"); return -1);
    ret = CreateAclTensorFloat32Output(reduce_out_shape, tensors.reduce_out_addr, tensors.reduce_out);
    CHECK_RET(ret == 0, LOG_PRINT("Create reduce_out_tensor failed\n"); return -1);
    return 0;
}

void DestroyTensors(Tensors &tensors)
{
    aclDestroyTensor(tensors.x);
    aclDestroyTensor(tensors.phi);
    aclDestroyTensor(tensors.alpha);
    aclDestroyTensor(tensors.bias);
    aclDestroyTensor(tensors.hin);
    aclDestroyTensor(tensors.h_post);
    aclDestroyTensor(tensors.h_res);
    aclDestroyTensor(tensors.h_pre);
    aclDestroyTensor(tensors.hc_before_norm);
    aclDestroyTensor(tensors.inv_rms);
    aclDestroyTensor(tensors.sum_out);
    aclDestroyTensor(tensors.norm_out);
    aclDestroyTensor(tensors.reduce_out);
}

void FreeDeviceMemory(Tensors &tensors)
{
    aclrtFree(tensors.x_addr);
    aclrtFree(tensors.phi_addr);
    aclrtFree(tensors.alpha_addr);
    aclrtFree(tensors.bias_addr);
    aclrtFree(tensors.hin_addr);
    aclrtFree(tensors.h_post_addr);
    aclrtFree(tensors.h_res_addr);
    aclrtFree(tensors.h_pre_addr);
    aclrtFree(tensors.hc_before_norm_addr);
    aclrtFree(tensors.inv_rms_addr);
    aclrtFree(tensors.sum_out_addr);
    aclrtFree(tensors.norm_out_addr);
    aclrtFree(tensors.reduce_out_addr);
}

template <typename T>
struct CompareResult {
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    int64_t total_elements = 0;
    int64_t diff_count = 0;
    int64_t first_diff_idx = -1;
    T first_diff_val_ref = 0;
    T first_diff_val_test = 0;
};

template <typename T>
CompareResult<T> CompareTensors(const std::vector<T> &ref_data, const std::vector<T> &test_data,
                                const std::string &name, double rel_threshold = 1e-3, double abs_threshold = 1e-6)
{
    CompareResult<T> result;
    int64_t size = static_cast<int64_t>(ref_data.size());
    result.total_elements = size;

    for (int64_t i = 0; i < size; ++i) {
        double ref_val = static_cast<double>(ref_data[i]);
        double test_val = static_cast<double>(test_data[i]);
        double abs_diff = std::fabs(ref_val - test_val);
        double rel_diff = (std::fabs(ref_val) > 1e-10) ? abs_diff / std::fabs(ref_val) : abs_diff;

        if (abs_diff > result.max_abs_diff) {
            result.max_abs_diff = abs_diff;
        }
        if (rel_diff > result.max_rel_diff) {
            result.max_rel_diff = rel_diff;
        }

        if (abs_diff > abs_threshold && rel_diff > rel_threshold) {
            if (result.diff_count == 0) {
                result.first_diff_idx = i;
                result.first_diff_val_ref = ref_data[i];
                result.first_diff_val_test = test_data[i];
            }
            result.diff_count++;
        }
    }

    LOG_PRINT("  [%s] max_abs_diff=%.6e, max_rel_diff=%.6e, diff_count=%ld/%ld",
              name.c_str(), result.max_abs_diff, result.max_rel_diff,
              result.diff_count, result.total_elements);
    if (result.diff_count > 0) {
        LOG_PRINT(", first_diff@[%ld]: ref=%f vs test=%f",
                  result.first_diff_idx,
                  static_cast<double>(result.first_diff_val_ref),
                  static_cast<double>(result.first_diff_val_test));
    }
    LOG_PRINT("\n");

    return result;
}

void CompareBf16Tensors(const std::vector<uint16_t> &ref_data, const std::vector<uint16_t> &test_data,
                        const std::string &name)
{
    int64_t size = static_cast<int64_t>(ref_data.size());
    int64_t diff_count = 0;
    int64_t first_diff_idx = -1;
    uint16_t first_diff_val_ref = 0;
    uint16_t first_diff_val_test = 0;

    for (int64_t i = 0; i < size; ++i) {
        if (ref_data[i] != test_data[i]) {
            if (diff_count == 0) {
                first_diff_idx = i;
                first_diff_val_ref = ref_data[i];
                first_diff_val_test = test_data[i];
            }
            diff_count++;
        }
    }

    LOG_PRINT("  [%s] BF16 bit-exact compare: diff_count=%ld/%ld", name.c_str(), diff_count, size);
    if (diff_count > 0) {
        LOG_PRINT(", first_diff@[%ld]: ref=0x%04x vs test=0x%04x",
                  first_diff_idx, first_diff_val_ref, first_diff_val_test);
    }
    LOG_PRINT("\n");
}

int RunMhcPreSinkhorn(Tensors &tensors,
                      int64_t hc_mult, int64_t num_iters,
                      float hc_eps, float norm_eps, bool need_backward,
                      aclrtStream stream)
{
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclnn_ret = aclnnMhcPreSinkhornGetWorkspaceSize(
        tensors.x, tensors.phi, tensors.alpha, tensors.bias,
        hc_mult, num_iters, hc_eps, norm_eps, need_backward,
        tensors.hin, tensors.h_post, tensors.h_res,
        tensors.h_pre, tensors.hc_before_norm, tensors.inv_rms,
        tensors.sum_out, tensors.norm_out,
        &workspace_size, &executor);
    CHECK_RET(aclnn_ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMhcPreSinkhornGetWorkspaceSize failed, error: %d\n", aclnn_ret); return -1);

    void *workspace_addr = nullptr;
    if (workspace_size > 0) {
        aclError ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed, error: %d\n", ret); return -1);
    }

    aclnn_ret = aclnnMhcPreSinkhorn(workspace_addr, workspace_size, executor, stream);
    CHECK_RET(aclnn_ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreSinkhorn failed, error: %d\n", aclnn_ret);
              aclrtFree(workspace_addr); return -1);

    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed\n"); aclrtFree(workspace_addr); return -1);

    if (workspace_size > 0) {
        aclrtFree(workspace_addr);
    }
    return 0;
}

int RunMhcPreClampSinkhorn(Tensors &tensors,
                           float min_val, float max_val,
                           int64_t hc_mult, int64_t num_iters,
                           float hc_eps, float norm_eps, bool need_backward,
                           aclrtStream stream)
{
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus aclnn_ret = aclnnMhcPreClampSinkhornGetWorkspaceSize(
        tensors.x, tensors.phi, tensors.alpha, tensors.bias,
        min_val, max_val,
        hc_mult, num_iters, hc_eps, norm_eps, need_backward,
        tensors.hin, tensors.h_post, tensors.h_res,
        tensors.h_pre, tensors.hc_before_norm, tensors.inv_rms,
        tensors.sum_out, tensors.norm_out, tensors.reduce_out,
        &workspace_size, &executor);
    CHECK_RET(aclnn_ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMhcPreClampSinkhornGetWorkspaceSize failed, error: %d\n", aclnn_ret); return -1);

    void *workspace_addr = nullptr;
    if (workspace_size > 0) {
        aclError ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed, error: %d\n", ret); return -1);
    }

    aclnn_ret = aclnnMhcPreClampSinkhorn(workspace_addr, workspace_size, executor, stream);
    CHECK_RET(aclnn_ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreClampSinkhorn failed, error: %d\n", aclnn_ret);
              aclrtFree(workspace_addr); return -1);

    CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed\n"); aclrtFree(workspace_addr); return -1);

    if (workspace_size > 0) {
        aclrtFree(workspace_addr);
    }
    return 0;
}

void CopyTensorToHostFloat(const std::vector<int64_t> &shape, void *device_addr, std::vector<float> &host_data)
{
    int64_t size = GetShapeSize(shape);
    host_data.resize(size);
    aclrtMemcpy(host_data.data(), size * sizeof(float), device_addr, size * sizeof(float),
                ACL_MEMCPY_DEVICE_TO_HOST);
}

void CopyTensorToHostBf16(const std::vector<int64_t> &shape, void *device_addr, std::vector<uint16_t> &host_data)
{
    int64_t size = GetShapeSize(shape);
    host_data.resize(size);
    aclrtMemcpy(host_data.data(), size * sizeof(uint16_t), device_addr, size * sizeof(uint16_t),
                ACL_MEMCPY_DEVICE_TO_HOST);
}

class MhcPreClampSinkhornPrecision : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MhcPreClampSinkhornPrecision SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MhcPreClampSinkhornPrecision TearDown" << std::endl;
    }
};

TEST_F(MhcPreClampSinkhornPrecision, compare_with_mhc_pre_sinkhorn_min0_max1)
{
    int32_t device_id = 0;
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;

    int64_t bs = 2, seq_len = 128, n = 4, c = 256;
    int64_t hc_mult = n;
    int64_t hc_mix = n * n + 2 * n;
    int64_t num_iters = 20;
    float hc_eps = 1e-6f, norm_eps = 1e-6f;
    bool need_backward = true;
    float min_val = 0.0f;
    float max_val = 1.0f;
    float abs_threshold = 1e-3f;
    float rel_threshold = 1e-3f;

    std::vector<int64_t> x_shape = {bs, seq_len, n, c};
    std::vector<int64_t> phi_shape = {hc_mix, n * c};
    std::vector<int64_t> alpha_shape = {3};
    std::vector<int64_t> bias_shape = {hc_mix};

    std::vector<int64_t> hin_shape = {bs, seq_len, c};
    std::vector<int64_t> h_post_shape = {bs, seq_len, n};
    std::vector<int64_t> h_res_shape = {bs, seq_len, n * n};
    std::vector<int64_t> h_pre_shape = {bs, seq_len, n};
    std::vector<int64_t> hc_before_norm_shape = {bs, seq_len, hc_mix};
    std::vector<int64_t> inv_rms_shape = {bs, seq_len, 1};
    std::vector<int64_t> sum_out_shape = {num_iters * 2, bs, seq_len, n};
    std::vector<int64_t> norm_out_shape = {num_iters * 2, bs, seq_len, n, n};
    std::vector<int64_t> reduce_out_shape = {num_iters * 2, bs, seq_len, n, n};

    int ret = InitAcl(device_id, context, stream);
    ASSERT_EQ(ret, 0) << "InitAcl failed";

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    int64_t x_size = GetShapeSize(x_shape);
    std::vector<float> x_host_data(x_size);
    for (int64_t i = 0; i < x_size; ++i) {
        x_host_data[i] = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    int64_t phi_size = GetShapeSize(phi_shape);
    std::vector<float> phi_host_data(phi_size);
    for (int64_t i = 0; i < phi_size; ++i) {
        phi_host_data[i] = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    std::vector<float> alpha_host_data(3, 1.0f);

    int64_t bias_size = GetShapeSize(bias_shape);
    std::vector<float> bias_host_data(bias_size);
    for (int64_t i = 0; i < bias_size; ++i) {
        bias_host_data[i] = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    LOG_PRINT("Creating tensors for mhc_pre_sinkhorn (reference)...\n");
    Tensors ref_tensors;
    ret = CreateInputTensors(x_host_data, phi_host_data, alpha_host_data, bias_host_data,
                             x_shape, phi_shape, alpha_shape, bias_shape, ref_tensors);
    ASSERT_EQ(ret, 0) << "CreateInputTensors for ref failed";

    ret = CreateOutputTensors(hin_shape, h_post_shape, h_res_shape, h_pre_shape,
                              hc_before_norm_shape, inv_rms_shape,
                              sum_out_shape, norm_out_shape, reduce_out_shape, ref_tensors);
    ASSERT_EQ(ret, 0) << "CreateOutputTensors for ref failed";

    LOG_PRINT("Creating tensors for mhc_pre_clamp_sinkhorn (test, min=0, max=1)...\n");
    Tensors test_tensors;
    ret = CreateInputTensors(x_host_data, phi_host_data, alpha_host_data, bias_host_data,
                             x_shape, phi_shape, alpha_shape, bias_shape, test_tensors);
    ASSERT_EQ(ret, 0) << "CreateInputTensors for test failed";

    ret = CreateOutputTensors(hin_shape, h_post_shape, h_res_shape, h_pre_shape,
                              hc_before_norm_shape, inv_rms_shape,
                              sum_out_shape, norm_out_shape, reduce_out_shape, test_tensors);
    ASSERT_EQ(ret, 0) << "CreateOutputTensors for test failed";

    LOG_PRINT("Running mhc_pre_sinkhorn (reference)...\n");
    ret = RunMhcPreSinkhorn(ref_tensors, hc_mult, num_iters, hc_eps, norm_eps, need_backward, stream);
    ASSERT_EQ(ret, 0) << "mhc_pre_sinkhorn failed";

    LOG_PRINT("Running mhc_pre_clamp_sinkhorn (test, min=0, max=1)...\n");
    ret = RunMhcPreClampSinkhorn(test_tensors, min_val, max_val,
                                  hc_mult, num_iters, hc_eps, norm_eps, need_backward, stream);
    ASSERT_EQ(ret, 0) << "mhc_pre_clamp_sinkhorn failed";

    LOG_PRINT("\n========== Precision Comparison Results ==========\n");
    LOG_PRINT("min=%.1f, max=%.1f, abs_threshold=%.0e, rel_threshold=%.0e\n",
              min_val, max_val, abs_threshold, rel_threshold);
    LOG_PRINT("When min=0, max=1, clamp does not change any value (sigmoid output is in (0,1)).\n");
    LOG_PRINT("The two operators should produce identical results.\n\n");

    int total_pass = 0;
    int total_fail = 0;

    {
        std::vector<uint16_t> ref_data, test_data;
        CopyTensorToHostBf16(hin_shape, ref_tensors.hin_addr, ref_data);
        CopyTensorToHostBf16(hin_shape, test_tensors.hin_addr, test_data);
        CompareBf16Tensors(ref_data, test_data, "hin (BF16)");
    }

    auto compare_float = [&](const std::vector<int64_t> &shape, void *ref_addr, void *test_addr,
                             const std::string &name) {
        std::vector<float> ref_data, test_data;
        CopyTensorToHostFloat(shape, ref_addr, ref_data);
        CopyTensorToHostFloat(shape, test_addr, test_data);
        auto result = CompareTensors<float>(ref_data, test_data, name, rel_threshold, abs_threshold);
        if (result.diff_count == 0) {
            total_pass++;
        } else {
            total_fail++;
        }
    };

    compare_float(h_post_shape, ref_tensors.h_post_addr, test_tensors.h_post_addr, "hPost (float)");
    compare_float(h_res_shape, ref_tensors.h_res_addr, test_tensors.h_res_addr, "hRes (float)");
    compare_float(h_pre_shape, ref_tensors.h_pre_addr, test_tensors.h_pre_addr, "hPre (float)");
    compare_float(sum_out_shape, ref_tensors.sum_out_addr, test_tensors.sum_out_addr, "sumOut (float)");
    compare_float(norm_out_shape, ref_tensors.norm_out_addr, test_tensors.norm_out_addr, "normOut (float)");

    LOG_PRINT("\n========== Summary ==========\n");
    LOG_PRINT("Total tensor groups compared: %d\n", total_pass + total_fail);
    LOG_PRINT("Passed: %d, Failed: %d\n", total_pass, total_fail);

    if (total_fail == 0) {
        LOG_PRINT("RESULT: PASS - mhc_pre_clamp_sinkhorn (min=0, max=1) matches mhc_pre_sinkhorn.\n");
    } else {
        LOG_PRINT("RESULT: FAIL - mhc_pre_clamp_sinkhorn (min=0, max=1) differs from mhc_pre_sinkhorn.\n");
    }

    DestroyTensors(ref_tensors);
    FreeDeviceMemory(ref_tensors);
    DestroyTensors(test_tensors);
    FreeDeviceMemory(test_tensors);

    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(device_id);
    aclFinalize();

    EXPECT_EQ(total_fail, 0) << "Precision comparison failed: " << total_fail << " tensor groups differ";
}