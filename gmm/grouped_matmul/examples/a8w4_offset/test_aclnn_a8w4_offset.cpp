#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_v5.h"
#include "aclnnop/aclnn_trans_quant_param.h"

namespace {
constexpr int64_t E = 2, M = 8, K = 512, N = 16, GS = 256;

#define ACL_CHECK(expr) do { auto rc = (expr); if (rc != ACL_SUCCESS) { \
  std::cerr << #expr << " failed: " << rc << "\n" << aclGetRecentErrMsg() << std::endl; return false; } } while (0)

struct TensorOwner {
  aclTensor *tensor = nullptr;
  void *device = nullptr;
  ~TensorOwner() { if (tensor) aclDestroyTensor(tensor); if (device) aclrtFree(device); }
};

struct ListOwner {
  aclTensorList *list = nullptr;
  ~ListOwner() { if (list) aclDestroyTensorList(list); }
};

bool MakeTensor(const void *host, size_t bytes, const std::vector<int64_t> &shape,
                aclDataType dtype, TensorOwner &out) {
  ACL_CHECK(aclrtMalloc(&out.device, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMemcpy(out.device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  std::vector<int64_t> strides(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) strides[i] = strides[i + 1] * shape[i + 1];
  out.tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                               shape.data(), shape.size(), out.device);
  if (!out.tensor) { std::cerr << "aclCreateTensor failed\n"; return false; }
  return true;
}

bool MakeList(TensorOwner &tensor, ListOwner &out) {
  const aclTensor *items[] = {tensor.tensor};
  out.list = aclCreateTensorList(items, 1);
  return out.list != nullptr;
}

uint16_t FloatToHalf(float f) {
  uint32_t x; std::memcpy(&x, &f, sizeof(x));
  uint32_t sign = (x >> 16) & 0x8000, mant = x & 0x7fffff;
  int exp = static_cast<int>((x >> 23) & 0xff) - 127 + 15;
  if (exp <= 0) return static_cast<uint16_t>(sign);
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00);
  return static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
}

float HalfToFloat(uint16_t h) {
  uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu, mant = h & 0x3ffu, x;
  if (exp == 0) x = sign;
  else if (exp == 31) x = sign | 0x7f800000u | (mant << 13);
  else x = sign | ((exp + 112) << 23) | (mant << 13);
  float f; std::memcpy(&f, &x, sizeof(f)); return f;
}

bool RunCase(bool withOffset) {
  const int64_t G = withOffset ? 1 : K / GS;
  std::vector<int8_t> x(M * K), logicalW(E * K * N);
  for (int64_t i = 0; i < M * K; ++i) x[i] = static_cast<int8_t>((i * 5 + 3) % 16 - 8);
  for (int64_t i = 0; i < E * K * N; ++i) logicalW[i] = static_cast<int8_t>((i * 7 + 1) % 16 - 8);

  // ACL_INT4 ND storage: two signed nibbles per byte, low nibble first.
  std::vector<uint8_t> packedW(logicalW.size() / 2);
  for (size_t i = 0; i < packedW.size(); ++i)
    packedW[i] = static_cast<uint8_t>((logicalW[2 * i] & 0xf) | ((logicalW[2 * i + 1] & 0xf) << 4));

  std::vector<float> floatScale(E * G * N), fusedOffset(withOffset ? E * N : 0), bias(E * N, 0.0f);
  // Binary-exact scales keep the CPU reference independent of FP19 truncation.
  for (size_t i = 0; i < floatScale.size(); ++i) floatScale[i] = static_cast<float>(1 + i % 3) / 128.0f;
  if (withOffset) {
    for (int64_t e = 0; e < E; ++e) for (int64_t n = 0; n < N; ++n)
      fusedOffset[e * N + n] = static_cast<float>((e + n) % 5 - 2) * floatScale[e * N + n];
  }
  for (int64_t e = 0; e < E; ++e) for (int64_t k = 0; k < K; ++k) for (int64_t n = 0; n < N; ++n) {
    float s = floatScale[(e * G + (withOffset ? 0 : k / GS)) * N + n];
    bias[e * N + n] += 8.0f * logicalW[(e * K + k) * N + n] * s;
  }

  uint64_t *quantRaw = nullptr, quantSize = 0;
  ACL_CHECK(aclnnTransQuantParam(floatScale.data(), floatScale.size(), nullptr, 0, &quantRaw, &quantSize));
  std::vector<uint64_t> quantScale(quantRaw, quantRaw + quantSize);
  free(quantRaw);
  std::vector<float> perToken(M, 1.0f);
  std::vector<int64_t> groups = {M / E, M / E};
  std::vector<uint16_t> output(M * N, FloatToHalf(0.0f));

  TensorOwner tx, tw, tb, ts, toff, tpt, tgroup, ty;
  if (!MakeTensor(x.data(), x.size(), {M, K}, ACL_INT8, tx) ||
      !MakeTensor(packedW.data(), packedW.size(), {E, K, N}, ACL_INT4, tw) ||
      !MakeTensor(bias.data(), bias.size() * 4, {E, N}, ACL_FLOAT, tb) ||
      !MakeTensor(quantScale.data(), quantScale.size() * 8, {E, G, N}, ACL_UINT64, ts) ||
      !MakeTensor(perToken.data(), perToken.size() * 4, {M, 1}, ACL_FLOAT, tpt) ||
      !MakeTensor(groups.data(), groups.size() * 8, {E}, ACL_INT64, tgroup) ||
      !MakeTensor(output.data(), output.size() * 2, {M, N}, ACL_FLOAT16, ty)) return false;
  if (withOffset && !MakeTensor(fusedOffset.data(), fusedOffset.size() * 4, {E, 1, N}, ACL_FLOAT, toff)) return false;

  ListOwner lx, lw, lb, ls, loff, lpt, ly;
  if (!MakeList(tx, lx) || !MakeList(tw, lw) || !MakeList(tb, lb) || !MakeList(ts, ls) ||
      !MakeList(tpt, lpt) || !MakeList(ty, ly) || (withOffset && !MakeList(toff, loff))) return false;

  const int64_t tuningData[] = {M / E, 0, -1};
  aclIntArray *tuning = aclCreateIntArray(tuningData, 3);
  uint64_t workspaceSize = 0; aclOpExecutor *executor = nullptr;
  auto rc = aclnnGroupedMatmulV5GetWorkspaceSize(
      lx.list, lw.list, lb.list, ls.list, withOffset ? loff.list : nullptr,
      nullptr, nullptr, lpt.list, tgroup.tensor, nullptr, nullptr, nullptr,
      3, 0, 1, 0, tuning, ly.list, nullptr, nullptr, &workspaceSize, &executor);
  aclDestroyIntArray(tuning);
  if (rc != ACL_SUCCESS) { std::cerr << "GetWorkspace failed: " << rc << "\n" << aclGetRecentErrMsg() << std::endl; return false; }
  void *workspace = nullptr;
  if (workspaceSize) ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
  aclrtStream stream = nullptr; ACL_CHECK(aclrtCreateStream(&stream));
  rc = aclnnGroupedMatmulV5(workspace, workspaceSize, executor, stream);
  if (rc == ACL_SUCCESS) rc = aclrtSynchronizeStream(stream);
  aclrtDestroyStream(stream); if (workspace) aclrtFree(workspace);
  if (rc != ACL_SUCCESS) { std::cerr << "Execute failed: " << rc << "\n" << aclGetRecentErrMsg() << std::endl; return false; }
  ACL_CHECK(aclrtMemcpy(output.data(), output.size() * 2, ty.device, output.size() * 2, ACL_MEMCPY_DEVICE_TO_HOST));

  float maxErr = 0.0f;
  for (int64_t m = 0; m < M; ++m) for (int64_t n = 0; n < N; ++n) {
    int64_t e = m < M / E ? 0 : 1; float ref = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
      float s = floatScale[(e * G + (withOffset ? 0 : k / GS)) * N + n];
      float w = logicalW[(e * K + k) * N + n] * s + (withOffset ? fusedOffset[e * N + n] : 0.0f);
      ref += x[m * K + k] * w;
    }
    maxErr = std::max(maxErr, std::fabs(HalfToFloat(output[m * N + n]) - ref));
  }
  std::cout << "ACLNN offset=" << (withOffset ? "non-null" : "null") << " max_abs_error=" << maxErr << std::endl;
  return maxErr < 0.2f;
}
}  // namespace

int main(int argc, char **argv) {
  if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS) return 1;
  bool ok = RunCase(argc > 1 && std::string(argv[1]) == "non-null");
  aclrtResetDevice(0); aclFinalize();
  std::cout << (ok ? "PASS" : "FAIL") << std::endl;
  return ok ? 0 : 1;
}
