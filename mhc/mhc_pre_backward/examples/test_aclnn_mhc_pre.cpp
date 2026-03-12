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
 * \file test_aclnn_mhc_pre.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_pre.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

// 计算Tensor形状对应的总元素数
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t size = 1;
  for (int64_t dim : shape) {
    size *= dim;
  }
  return size;
}

// 将Device侧Tensor数据拷贝到Host侧并打印（float类型）
void PrintTensorDataFloat(const std::vector<int64_t>& shape, void* device_addr) {
  int64_t size = GetShapeSize(shape);
  std::vector<float> host_data(size, 0.0f);
  
  aclError ret = aclrtMemcpy(
      host_data.data(), size * sizeof(float),
      device_addr, size * sizeof(float),
      ACL_MEMCPY_DEVICE_TO_HOST
  );
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("Memcpy device to host failed, error: %d\n", ret); 
            return);

  LOG_PRINT("Tensor data (first 10 elements): ");
  for (int i = 0; i < std::min((int64_t)10, size); ++i) {
    LOG_PRINT("%f ", host_data[i]);
  }
  LOG_PRINT("\n");
}

// 将Device侧Tensor数据拷贝到Host侧并打印（float16类型）
void PrintTensorDataFloat16(const std::vector<int64_t>& shape, void* device_addr) {
  int64_t size = GetShapeSize(shape);
  std::vector<float> host_data(size, 0.0f);
  
  aclError ret = aclrtMemcpy(
      host_data.data(), size * sizeof(float),
      device_addr, size * sizeof(aclFloat16),
      ACL_MEMCPY_DEVICE_TO_HOST
  );
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("Memcpy device to host failed, error: %d\n", ret); 
            return);

  LOG_PRINT("Tensor data (first 10 elements): ");
  for (int i = 0; i < std::min((int64_t)10, size); ++i) {
    LOG_PRINT("%f ", host_data[i]);
  }
  LOG_PRINT("\n");
}

// 初始化AscendCL环境（Device/Context/Stream）
int InitAcl(int32_t device_id, aclrtContext& context, aclrtStream& stream) {
  aclError ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclInit failed, error: %d\n", ret); 
            return -1);

  ret = aclrtSetDevice(device_id);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtSetDevice failed, error: %d\n", ret); 
            return -1);

  ret = aclrtCreateContext(&context, device_id);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtCreateContext failed, error: %d\n", ret); 
            return -1);

  ret = aclrtSetCurrentContext(context);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtSetCurrentContext failed, error: %d\n", ret); 
            return -1);

  ret = aclrtCreateStream(&stream);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtCreateStream failed, error: %d\n", ret); 
            return -1);

  return 0;
}

// 创建FLOAT32类型Device侧aclTensor（含数据拷贝）
int CreateAclTensorFloat32(
    const std::vector<float>& host_data,
    const std::vector<int64_t>& shape,
    void*& device_addr,
    aclTensor*& tensor) {
  int64_t size = GetShapeSize(shape) * sizeof(float);

  aclError ret = aclrtMalloc(&device_addr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); 
            return -1);

  ret = aclrtMemcpy(
      device_addr, size,
      host_data.data(), size,
      ACL_MEMCPY_HOST_TO_DEVICE
  );
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret); 
            return -1);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  tensor = aclCreateTensor(
      shape.data(), shape.size(),
      ACL_FLOAT, strides.data(), 0,
      ACL_FORMAT_ND, shape.data(), shape.size(),
      device_addr
  );
  CHECK_RET(tensor != nullptr, 
            LOG_PRINT("aclCreateTensor failed\n"); 
            return -1);

  return 0;
}

// 创建FLOAT16类型Device侧aclTensor（含数据拷贝）
int CreateAclTensorFloat16(
    const std::vector<float>& host_data,
    const std::vector<int64_t>& shape,
    void*& device_addr,
    aclTensor*& tensor) {
  int64_t size = GetShapeSize(shape);
  std::vector<aclFloat16> host_data_fp16(size);
  for (int64_t i = 0; i < size; ++i) {
    host_data_fp16[i] = aclFloat16(host_data[i]);
  }

  int64_t byte_size = size * sizeof(aclFloat16);

  aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); 
            return -1);

  ret = aclrtMemcpy(
      device_addr, byte_size,
      host_data_fp16.data(), byte_size,
      ACL_MEMCPY_HOST_TO_DEVICE
  );
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret); 
            return -1);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  tensor = aclCreateTensor(
      shape.data(), shape.size(),
      ACL_FLOAT16, strides.data(), 0,
      ACL_FORMAT_ND, shape.data(), shape.size(),
      device_addr
  );
  CHECK_RET(tensor != nullptr, 
            LOG_PRINT("aclCreateTensor failed\n"); 
            return -1);

  return 0;
}

// 创建FLOAT16类型输出aclTensor（仅申请内存）
int CreateAclTensorFloat16Output(
    const std::vector<int64_t>& shape,
    void*& device_addr,
    aclTensor*& tensor) {
  int64_t size = GetShapeSize(shape);
  int64_t byte_size = size * sizeof(aclFloat16);

  aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); 
            return -1);

  tensor = aclCreateTensor(
      shape.data(), shape.size(),
      ACL_FLOAT16, nullptr, 0,
      ACL_FORMAT_ND, shape.data(), shape.size(),
      device_addr
  );
  CHECK_RET(tensor != nullptr, 
            LOG_PRINT("aclCreateTensor failed\n"); 
            return -1);

  return 0;
}

// 创建FLOAT32类型输出aclTensor（仅申请内存）
int CreateAclTensorFloat32Output(
    const std::vector<int64_t>& shape,
    void*& device_addr,
    aclTensor*& tensor) {
  int64_t size = GetShapeSize(shape);
  int64_t byte_size = size * sizeof(float);

  aclError ret = aclrtMalloc(&device_addr, byte_size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); 
            return -1);

  tensor = aclCreateTensor(
      shape.data(), shape.size(),
      ACL_FLOAT, nullptr, 0,
      ACL_FORMAT_ND, shape.data(), shape.size(),
      device_addr
  );
  CHECK_RET(tensor != nullptr, 
            LOG_PRINT("aclCreateTensor failed\n"); 
            return -1);

  return 0;
}

int main() {
  int32_t device_id = 0;
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;
  
  int ret = InitAcl(device_id, context, stream);
  CHECK_RET(ret == 0, 
            LOG_PRINT("InitAcl failed, error: %d\n", ret); 
            return -1);

  int B = 1;
  int S = 2048;
  int n = 4;
  int D = 2560;

  std::vector<int64_t> x_shape = {B * S, n, D};
  int64_t x_size = GetShapeSize(x_shape);
  std::vector<float> x_host_data(x_size, 1.0f);

  std::vector<int64_t> phi_shape = {n * n + 2 * n, n * D};
  std::vector<float> phi_host_data(GetShapeSize(phi_shape), 1.0f);

  std::vector<int64_t> alpha_shape = {3};
  std::vector<float> alpha_host_data(3, 1.0f);

  std::vector<int64_t> bias_shape = {n * n + 2 * n};
  std::vector<float> bias_host_data(GetShapeSize(bias_shape), 1.0f);

  std::vector<int64_t> gamma_shape = {n, D};
  std::vector<float> gamma_host_data(GetShapeSize(gamma_shape), 1.0f);

  std::vector<int64_t> output_hin_shape = {B * S, D};
  std::vector<int64_t> output_h_post_shape = {B * S, n};
  std::vector<int64_t> output_h_res_shape = {B * S, n, n};
  std::vector<int64_t> output_inv_rms_shape = {B * S};
  std::vector<int64_t> output_h_mix_shape = {B * S, n * n + 2 * n};
  std::vector<int64_t> output_h_pre_shape = {B * S, n};

  void* x_device_addr = nullptr;
  aclTensor* x_tensor = nullptr;
  ret = CreateAclTensorFloat16(x_host_data, x_shape, x_device_addr, x_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create x_tensor failed\n"); return -1);

  void* phi_device_addr = nullptr;
  aclTensor* phi_tensor = nullptr;
  ret = CreateAclTensorFloat32(phi_host_data, phi_shape, phi_device_addr, phi_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create phi_tensor failed\n"); return -1);

  void* alpha_device_addr = nullptr;
  aclTensor* alpha_tensor = nullptr;
  ret = CreateAclTensorFloat32(alpha_host_data, alpha_shape, alpha_device_addr, alpha_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create alpha_tensor failed\n"); return -1);

  void* bias_device_addr = nullptr;
  aclTensor* bias_tensor = nullptr;
  ret = CreateAclTensorFloat32(bias_host_data, bias_shape, bias_device_addr, bias_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create bias_tensor failed\n"); return -1);

  void* gamma_device_addr = nullptr;
  aclTensor* gamma_tensor = nullptr;
  ret = CreateAclTensorFloat32(gamma_host_data, gamma_shape, gamma_device_addr, gamma_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create gamma_tensor failed\n"); return -1);

  void* output_hin_device_addr = nullptr;
  aclTensor* output_hin_tensor = nullptr;
  ret = CreateAclTensorFloat16Output(output_hin_shape, output_hin_device_addr, output_hin_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create output_hin_tensor failed\n"); return -1);

  void* output_h_post_device_addr = nullptr;
  aclTensor* output_h_post_tensor = nullptr;
  ret = CreateAclTensorFloat32Output(output_h_post_shape, output_h_post_device_addr, output_h_post_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create output_h_post_tensor failed\n"); return -1);

  void* output_h_res_device_addr = nullptr;
  aclTensor* output_h_res_tensor = nullptr;
  ret = CreateAclTensorFloat32Output(output_h_res_shape, output_h_res_device_addr, output_h_res_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create output_h_res_tensor failed\n"); return -1);

  void* output_inv_rms_device_addr = nullptr;
  aclTensor* output_inv_rms_tensor = nullptr;
  ret = CreateAclTensorFloat32Output(output_inv_rms_shape, output_inv_rms_device_addr, output_inv_rms_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create output_inv_rms_tensor failed\n"); return -1);

  void* output_h_mix_device_addr = nullptr;
  aclTensor* output_h_mix_tensor = nullptr;
  ret = CreateAclTensorFloat32Output(output_h_mix_shape, output_h_mix_device_addr, output_h_mix_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create output_h_mix_tensor failed\n"); return -1);

  void* output_h_pre_device_addr = nullptr;
  aclTensor* output_h_pre_tensor = nullptr;
  ret = CreateAclTensorFloat32Output(output_h_pre_shape, output_h_pre_device_addr, output_h_pre_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create output_h_pre_tensor failed\n"); return -1);

  double norm_eps = 1e-6;
  double hc_eps = 1e-6;
  int64_t out_flag = 1;

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  
  aclnnStatus aclnn_ret = aclnnMhcPreGetWorkspaceSize(
    x_tensor, phi_tensor, alpha_tensor, bias_tensor, gamma_tensor,
    out_flag, norm_eps, hc_eps,
    output_hin_tensor, output_h_post_tensor, output_h_res_tensor,
    output_inv_rms_tensor, output_h_mix_tensor, output_h_pre_tensor,
    &workspace_size, &executor);

  CHECK_RET(aclnn_ret == ACL_SUCCESS, 
            LOG_PRINT("aclnnMhcPreGetWorkspaceSize failed, error: %d\n", aclnn_ret); 
            return -1);
  
  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, 
              LOG_PRINT("aclrtMalloc workspace failed, error: %d\n", ret); 
              return -1);
  }

  aclnn_ret = aclnnMhcPre(
      workspace_addr,
      workspace_size,
      executor,
      stream
  );
  CHECK_RET(aclnn_ret == ACL_SUCCESS, 
            LOG_PRINT("aclnnMhcPre failed, error: %d\n", aclnn_ret); 
            return -1);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtSynchronizeStream failed, error: %d\n", ret); 
            return -1);

  LOG_PRINT("MhcPre compute success!\n");
  LOG_PRINT("Output tensor data: \n");

  PrintTensorDataFloat16(output_hin_shape, output_hin_device_addr);
  PrintTensorDataFloat(output_h_post_shape, output_h_post_device_addr);
  PrintTensorDataFloat(output_h_res_shape, output_h_res_device_addr);
  PrintTensorDataFloat(output_inv_rms_shape, output_inv_rms_device_addr);
  PrintTensorDataFloat(output_h_mix_shape, output_h_mix_device_addr);
  PrintTensorDataFloat(output_h_pre_shape, output_h_pre_device_addr);

  aclDestroyTensor(x_tensor);
  aclDestroyTensor(phi_tensor);
  aclDestroyTensor(alpha_tensor);
  aclDestroyTensor(bias_tensor);
  aclDestroyTensor(gamma_tensor);

  aclDestroyTensor(output_hin_tensor);
  aclDestroyTensor(output_h_post_tensor);
  aclDestroyTensor(output_h_res_tensor);
  aclDestroyTensor(output_inv_rms_tensor);
  aclDestroyTensor(output_h_mix_tensor);
  aclDestroyTensor(output_h_pre_tensor);

  aclrtFree(x_device_addr);
  aclrtFree(phi_device_addr);
  aclrtFree(alpha_device_addr);
  aclrtFree(bias_device_addr);
  aclrtFree(gamma_device_addr);

  aclrtFree(output_hin_device_addr);
  aclrtFree(output_h_post_device_addr);
  aclrtFree(output_h_res_device_addr);
  aclrtFree(output_inv_rms_device_addr);
  aclrtFree(output_h_mix_device_addr);
  aclrtFree(output_h_pre_device_addr);

  if (workspace_size > 0) {
    aclrtFree(workspace_addr);
  }

  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(device_id);
  aclFinalize();

  LOG_PRINT("All resources released successfully!\n");
  return 0;
}
