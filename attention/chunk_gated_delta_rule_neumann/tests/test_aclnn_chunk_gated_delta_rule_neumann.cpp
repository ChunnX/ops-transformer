/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_chunk_gated_delta_rule_neumann.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) shapeSize *= i;
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) strides[i] = shape[i + 1] * strides[i + 1];

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

struct TestParams {
    int64_t B;
    int64_t T;
    int64_t nk;
    int64_t nv;
    int64_t dk;
    int64_t dv;
    float scale;
};

bool read_test_inputs(const char* filename, TestParams& params,
                      std::vector<uint16_t>& q_data, std::vector<uint16_t>& k_data,
                      std::vector<uint16_t>& v_data, std::vector<uint16_t>& beta_data,
                      std::vector<float>& g_data, std::vector<uint16_t>& initial_state_data,
                      std::vector<int32_t>& seqlens_data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        LOG_PRINT("Failed to open: %s\n", filename);
        return false;
    }

    file.read(reinterpret_cast<char*>(&params.B), sizeof(int64_t));
    file.read(reinterpret_cast<char*>(&params.T), sizeof(int64_t));
    file.read(reinterpret_cast<char*>(&params.nk), sizeof(int64_t));
    file.read(reinterpret_cast<char*>(&params.nv), sizeof(int64_t));
    file.read(reinterpret_cast<char*>(&params.dk), sizeof(int64_t));
    file.read(reinterpret_cast<char*>(&params.dv), sizeof(int64_t));
    file.read(reinterpret_cast<char*>(&params.scale), sizeof(float));

    LOG_PRINT("Parameters: B=%ld, T=%ld, nk=%ld, nv=%ld, dk=%ld, dv=%ld, scale=%f\n",
              params.B, params.T, params.nk, params.nv, params.dk, params.dv, params.scale);

    int64_t q_size = params.T * params.nk * params.dk;
    int64_t k_size = params.T * params.nk * params.dk;
    int64_t v_size = params.T * params.nv * params.dv;
    int64_t beta_size = params.T * params.nv;
    int64_t g_size = params.T * params.nv;
    int64_t state_size = params.B * params.nv * params.dv * params.dk;

    q_data.resize(q_size);
    k_data.resize(k_size);
    v_data.resize(v_size);
    beta_data.resize(beta_size);
    g_data.resize(g_size);
    initial_state_data.resize(state_size);
    seqlens_data.resize(params.B);

    std::vector<uint32_t> g_raw(g_size);
    
    file.read(reinterpret_cast<char*>(q_data.data()), q_size * sizeof(uint16_t));
    file.read(reinterpret_cast<char*>(k_data.data()), k_size * sizeof(uint16_t));
    file.read(reinterpret_cast<char*>(v_data.data()), v_size * sizeof(uint16_t));
    file.read(reinterpret_cast<char*>(beta_data.data()), beta_size * sizeof(uint16_t));
    file.read(reinterpret_cast<char*>(g_raw.data()), g_size * sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(initial_state_data.data()), state_size * sizeof(uint16_t));
    file.read(reinterpret_cast<char*>(seqlens_data.data()), params.B * sizeof(int32_t));

    for (size_t i = 0; i < g_size; ++i) {
        g_data[i] = *reinterpret_cast<float*>(&g_raw[i]);
    }

    LOG_PRINT("Data loaded: q=%ld, k=%ld, v=%ld, beta=%ld, g=%ld, state=%ld\n",
              q_size, k_size, v_size, beta_size, g_size, state_size);
    return true;
}

bool read_golden_output(const char* filename, std::vector<uint16_t>& out_golden,
                        std::vector<uint16_t>& state_golden, int64_t out_size, int64_t state_size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        LOG_PRINT("Failed to open golden: %s\n", filename);
        return false;
    }

    out_golden.resize(out_size);
    state_golden.resize(state_size);

    file.read(reinterpret_cast<char*>(out_golden.data()), out_size * sizeof(uint16_t));
    file.read(reinterpret_cast<char*>(state_golden.data()), state_size * sizeof(uint16_t));

    LOG_PRINT("Golden loaded: out=%ld, state=%ld\n", out_size, state_size);
    return true;
}

float bf16_to_float(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    return *reinterpret_cast<float*>(&bits);
}

void compute_error(const std::vector<uint16_t>& npu_data,
                   const std::vector<uint16_t>& golden_data,
                   const char* name) {
    if (npu_data.size() != golden_data.size()) {
        LOG_PRINT("Size mismatch for %s\n", name);
        return;
    }

    float max_err = 0.0f;
    float max_rel = 0.0f;
    int max_idx = 0;

    for (size_t i = 0; i < npu_data.size(); ++i) {
        float npu_val = bf16_to_float(npu_data[i]);
        float golden_val = bf16_to_float(golden_data[i]);
        float err = fabs(npu_val - golden_val);
        float rel = (fabs(golden_val) > 0.0f) ? err / fabs(golden_val) : err;

        if (err > max_err) {
            max_err = err;
            max_rel = rel;
            max_idx = i;
        }
    }

    LOG_PRINT("\n%s precision:\n", name);
    LOG_PRINT("  max_abs_err: %.6f at idx %d\n", max_err, max_idx);
    LOG_PRINT("  max_rel_err: %.6f%%\n", max_rel * 100);
    
    float threshold = 0.01f;
    if (max_err < threshold) {
        LOG_PRINT("  PASS (threshold %.6f)\n", threshold);
    } else {
        LOG_PRINT("  FAIL (threshold %.6f)\n", threshold);
        LOG_PRINT("  npu[idx] = %.6f, golden[idx] = %.6f\n",
                  bf16_to_float(npu_data[max_idx]), bf16_to_float(golden_data[max_idx]));
    }
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  LOG_PRINT("Reading test data from binary files...\n");
  
  TestParams params;
  std::vector<uint16_t> q_data, k_data, v_data, beta_data, initial_state_data;
  std::vector<float> g_data;
  std::vector<int32_t> seqlens_data;
  std::vector<uint16_t> out_golden, state_golden;
  int64_t out_size = 0;
  int64_t state_size = 0;

  const char* input_files[] = {"test_data/test_inputs.bin", "test_inputs.bin", "test_inputs_multi.bin", "test_inputs_simple.bin"};
  const char* golden_files[] = {"test_data/golden_output.bin", "golden_output.bin", "golden_output_multi.bin", "golden_output_simple.bin"};
  bool loaded = false;
  
  for (int i = 0; i < 4 && !loaded; i++) {
    if (read_test_inputs(input_files[i], params, q_data, k_data, v_data,
                          beta_data, g_data, initial_state_data, seqlens_data)) {
      out_size = params.T * params.nv * params.dv;
      state_size = params.B * params.nv * params.dv * params.dk;
      if (read_golden_output(golden_files[i], out_golden, state_golden, out_size, state_size)) {
        loaded = true;
        LOG_PRINT("Using: %s and %s\n", input_files[i], golden_files[i]);
      }
    }
  }
  
  if (!loaded) {
    LOG_PRINT("Failed to read any test inputs\n");
    aclFinalize();
    return 1;
  }

  LOG_PRINT("Creating tensors...\n");

  std::vector<int64_t> qShape = {params.T, params.nk, params.dk};
  std::vector<int64_t> kShape = {params.T, params.nk, params.dk};
  std::vector<int64_t> vShape = {params.T, params.nv, params.dv};
  std::vector<int64_t> betaShape = {params.T, params.nv};
  std::vector<int64_t> gShape = {params.T, params.nv};
  std::vector<int64_t> stateShape = {params.B, params.nv, params.dv, params.dk};
  std::vector<int64_t> outShape = {params.T, params.nv, params.dv};

  std::vector<int64_t> seqlensShape = {params.B};

  void* qDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* betaDeviceAddr = nullptr;
  void* gDeviceAddr = nullptr;
  void* initialStateDeviceAddr = nullptr;
  void* seqlensDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* finalStateDeviceAddr = nullptr;

  aclTensor* qTensor = nullptr;
  aclTensor* kTensor = nullptr;
  aclTensor* vTensor = nullptr;
  aclTensor* betaTensor = nullptr;
  aclTensor* gTensor = nullptr;
  aclTensor* initialStateTensor = nullptr;
  aclTensor* seqlensTensor = nullptr;
  aclTensor* outTensor = nullptr;
  aclTensor* finalStateTensor = nullptr;

  CreateAclTensor(q_data, qShape, &qDeviceAddr, aclDataType::ACL_BF16, &qTensor);
  CreateAclTensor(k_data, kShape, &kDeviceAddr, aclDataType::ACL_BF16, &kTensor);
  CreateAclTensor(v_data, vShape, &vDeviceAddr, aclDataType::ACL_BF16, &vTensor);
  CreateAclTensor(beta_data, betaShape, &betaDeviceAddr, aclDataType::ACL_BF16, &betaTensor);
  CreateAclTensor(g_data, gShape, &gDeviceAddr, aclDataType::ACL_FLOAT, &gTensor);
  CreateAclTensor(initial_state_data, stateShape, &initialStateDeviceAddr, aclDataType::ACL_BF16, &initialStateTensor);
  CreateAclTensor(seqlens_data, seqlensShape, &seqlensDeviceAddr, aclDataType::ACL_INT32, &seqlensTensor);
  CreateAclTensor(std::vector<uint16_t>(out_size, 0), outShape, &outDeviceAddr, aclDataType::ACL_BF16, &outTensor);
  CreateAclTensor(std::vector<uint16_t>(state_size, 0), stateShape, &finalStateDeviceAddr, aclDataType::ACL_BF16, &finalStateTensor);

  LOG_PRINT("Calling aclnnChunkGatedDeltaRuleNeumannGetWorkspaceSize...\n");

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnChunkGatedDeltaRuleNeumannGetWorkspaceSize(
      qTensor, kTensor, vTensor, betaTensor, initialStateTensor,
      seqlensTensor, gTensor, params.scale,
      outTensor, finalStateTensor, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  LOG_PRINT("Workspace size: %lu bytes\n", workspaceSize);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for workspace failed. ERROR: %d\n", ret); return ret);
  }

  LOG_PRINT("Calling aclnnChunkGatedDeltaRuleNeumann...\n");
  ret = aclnnChunkGatedDeltaRuleNeumann(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnn execution failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("SynchronizeStream failed. ERROR: %d\n", ret); return ret);

  LOG_PRINT("aclnn execution succeeded\n");

  std::vector<uint16_t> out_npu(out_size);
  std::vector<uint16_t> state_npu(state_size);

  aclrtMemcpy(out_npu.data(), out_size * sizeof(uint16_t), outDeviceAddr,
              out_size * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
  aclrtMemcpy(state_npu.data(), state_size * sizeof(uint16_t), finalStateDeviceAddr,
              state_size * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);

  LOG_PRINT("\n========== Precision Comparison ==========\n");
  compute_error(out_npu, out_golden, "Output");
  compute_error(state_npu, state_golden, "FinalState");

  aclDestroyTensor(qTensor);
  aclDestroyTensor(kTensor);
  aclDestroyTensor(vTensor);
  aclDestroyTensor(betaTensor);
  aclDestroyTensor(gTensor);
  aclDestroyTensor(initialStateTensor);
  aclDestroyTensor(seqlensTensor);
  aclDestroyTensor(outTensor);
  aclDestroyTensor(finalStateTensor);

  if (workspaceAddr) aclrtFree(workspaceAddr);
  aclrtFree(qDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(betaDeviceAddr);
  aclrtFree(gDeviceAddr);
  aclrtFree(initialStateDeviceAddr);
  aclrtFree(seqlensDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(finalStateDeviceAddr);

  aclrtDestroyStream(stream);
  aclFinalize();

  LOG_PRINT("\nTest completed\n");
  return 0;
}