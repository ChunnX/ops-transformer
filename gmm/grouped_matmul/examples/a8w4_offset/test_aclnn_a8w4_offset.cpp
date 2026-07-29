#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_matmul_v5.h"
#include "aclnnop/aclnn_grouped_matmul_weight_nz.h"
#include "aclnnop/aclnn_trans_quant_param.h"

namespace {
void Stage(const char *name) { std::cerr << "[stage] " << name << std::endl; }

constexpr int64_t DEFAULT_E = 2;
constexpr int64_t DEFAULT_M = 8;
constexpr int64_t DEFAULT_K = 512;
constexpr int64_t DEFAULT_N = 16;
constexpr int64_t GS = 256;

int64_t GetEnvInt64(const char *name, int64_t defaultValue) {
  const char *value = std::getenv(name);
  if (value == nullptr || *value == '\0') return defaultValue;
  char *end = nullptr;
  const long long parsed = std::strtoll(value, &end, 10);
  if (end == value || *end != '\0') {
    std::cerr << "invalid integer environment variable " << name << '=' << value << std::endl;
    return -1;
  }
  return static_cast<int64_t>(parsed);
}

bool GetEnvFlag(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && std::string(value) == "1";
}

std::string GetEnvString(const char *name, const char *defaultValue) {
  const char *value = std::getenv(name);
  return value == nullptr ? std::string(defaultValue) : std::string(value);
}

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

bool MakeTensor(const void *host, size_t bytes, const std::vector<int64_t> &logicalShape,
                const std::vector<int64_t> &storageShape, aclDataType dtype,
                aclFormat format, TensorOwner &out) {
  ACL_CHECK(aclrtMalloc(&out.device, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMemcpy(out.device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  std::vector<int64_t> strides(logicalShape.size(), 1);
  for (int i = static_cast<int>(logicalShape.size()) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * logicalShape[i + 1];
  }
  out.tensor = aclCreateTensor(logicalShape.data(), logicalShape.size(), dtype, strides.data(), 0, format,
                               storageShape.data(), storageShape.size(), out.device);
  if (!out.tensor) { std::cerr << "aclCreateTensor failed\n"; return false; }
  return true;
}

bool MakeTensor(const void *host, size_t bytes, const std::vector<int64_t> &shape,
                aclDataType dtype, TensorOwner &out) {
  return MakeTensor(host, bytes, shape, shape, dtype, ACL_FORMAT_ND, out);
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

float HalfUlp(float value) {
  value = std::fabs(value);
  if (value == 0.0F || value < std::ldexp(1.0F, -14)) {
    return std::ldexp(1.0F, -24);
  }
  int exponent = 0;
  std::frexp(value, &exponent);
  return std::ldexp(1.0F, exponent - 11);
}

bool RunCase(bool withOffset) {
  Stage("prepare host data");
  const bool fixedScaleOne = GetEnvFlag("A8W4_FIXED_SCALE_ONE");
  const bool oddM = GetEnvFlag("A8W4_ODD_M");
  const bool randomData = GetEnvFlag("A8W4_RANDOM_DATA");
  const bool verboseScale = GetEnvFlag("A8W4_VERBOSE_SCALE");
  const bool weightNz = GetEnvFlag("A8W4_WEIGHT_NZ");
  const bool packedInt32 = GetEnvFlag("A8W4_PACKED_INT32");
  const bool specialNz = GetEnvFlag("A8W4_SPECIAL_NZ");
  const std::string dataMode = GetEnvString("A8W4_DATA_MODE", randomData ? "random" : "pattern");
  const std::string scaleProfile =
      GetEnvString("A8W4_SCALE_PROFILE", fixedScaleOne ? "one" : "varying");
  const std::string groupMode = GetEnvString("A8W4_GROUP_MODE", "random");
  const char *apiEnv = std::getenv("A8W4_API");
  const std::string api = apiEnv == nullptr ? "V5" : apiEnv;
  if (api != "V5" && api != "WeightNz") {
    std::cerr << "A8W4_API must be V5 or WeightNz, got: " << api << std::endl;
    return false;
  }
  if (api == "WeightNz" && !weightNz) {
    std::cerr << "A8W4_API=WeightNz requires A8W4_WEIGHT_NZ=1" << std::endl;
    return false;
  }
  constexpr uint32_t defaultSeed = 20260723U;
  const char *seedEnv = std::getenv("A8W4_RANDOM_SEED");
  const uint32_t randomSeed = seedEnv == nullptr ? defaultSeed :
      static_cast<uint32_t>(std::strtoul(seedEnv, nullptr, 10));
  const int64_t mTotal = GetEnvInt64("A8W4_M", oddM ? 9 : DEFAULT_M);
  const int64_t K = GetEnvInt64("A8W4_K", DEFAULT_K);
  const int64_t N = GetEnvInt64("A8W4_N", DEFAULT_N);
  const int64_t E = GetEnvInt64("A8W4_E", DEFAULT_E);
  if (E <= 0 || E > 128 || mTotal <= 0 || K <= 0 || K > 18432 || N <= 0 || N % 8 != 0) {
    std::cerr << "invalid case: E=" << E << " M=" << mTotal << " K=" << K << " N=" << N
              << " (requires 1<=E<=128, M>0, 0<K<=18432, and N%8==0)"
              << std::endl;
    return false;
  }
  if (weightNz && (K % 16 != 0 || N % 64 != 0)) {
    std::cerr << "NZ case requires K%16==0 and N%64==0" << std::endl;
    return false;
  }
  if (specialNz && (!weightNz || K % 64 != 0 || N % 16 != 0)) {
    std::cerr << "special NZ requires NZ weight, K%64==0 and N%16==0" << std::endl;
    return false;
  }

  std::vector<int64_t> groups(E, 0);
  const int64_t firstGroupOverride = GetEnvInt64("A8W4_FIRST_GROUP_M", -1);
  if (firstGroupOverride > 0 && E == 2 && firstGroupOverride < mTotal) {
    groups[0] = firstGroupOverride;
    groups[1] = mTotal - firstGroupOverride;
  } else if (groupMode == "balanced" || GetEnvFlag("A8W4_BALANCED_GROUPS")) {
    for (int64_t token = 0; token < mTotal; ++token) {
      ++groups[token % E];
    }
  } else if (groupMode == "one-heavy" || groupMode == "last-heavy") {
    const int64_t heavy = groupMode == "last-heavy" ? E - 1 : 0;
    const int64_t lightBudget = std::min<int64_t>(mTotal / 10, E - 1);
    for (int64_t e = 0; e < lightBudget; ++e) {
      const int64_t expert = heavy == 0 ? e + 1 : e;
      ++groups[expert];
    }
    groups[heavy] = mTotal - lightBudget;
  } else if (groupMode == "alternating") {
    int64_t remaining = mTotal;
    for (int64_t e = 0; e < E && remaining > 0; ++e) {
      const int64_t requested = e % 2 == 0 ? 1 : std::min<int64_t>(remaining, 17);
      groups[e] = std::min(remaining, requested);
      remaining -= groups[e];
    }
    groups[E - 1] += remaining;
  } else if (groupMode == "zero-expert") {
    const int64_t activeExperts = std::max<int64_t>(1, E / 2);
    for (int64_t token = 0; token < mTotal; ++token) {
      ++groups[(token * 2) % activeExperts * 2 % E];
    }
  } else if (groupMode == "zipf") {
    uint64_t weightSum = 0;
    for (int64_t e = 0; e < E; ++e) weightSum += static_cast<uint64_t>(E / (e + 1) + 1);
    int64_t assigned = 0;
    for (int64_t e = 0; e < E; ++e) {
      groups[e] = static_cast<int64_t>(
          static_cast<uint64_t>(mTotal) * static_cast<uint64_t>(E / (e + 1) + 1) / weightSum);
      assigned += groups[e];
    }
    groups[0] += mTotal - assigned;
  } else {
    std::mt19937 groupRng(randomSeed ^ 0x9e3779b9U);
    std::uniform_int_distribution<int64_t> expertDist(0, E - 1);
    for (int64_t token = 0; token < mTotal; ++token) {
      ++groups[expertDist(groupRng)];
    }
  }
  std::vector<int64_t> expertForRow(mTotal);
  int64_t row = 0;
  for (int64_t e = 0; e < E; ++e) {
    for (int64_t token = 0; token < groups[e]; ++token) {
      expertForRow[row++] = e;
    }
  }
  std::cout << "scale_mode=" << scaleProfile << std::endl;
  std::cout << "shape=E" << E << " M" << mTotal << " N" << N << " K" << K << " groups=[";
  for (int64_t e = 0; e < E; ++e) {
    std::cout << (e == 0 ? "" : ",") << groups[e];
  }
  std::cout << "]" << std::endl;
  std::cout << "data_mode=" << dataMode << " group_mode=" << groupMode
            << " seed=" << randomSeed << std::endl;
  std::cout << "api=" << api << std::endl;
  std::cout << "weight_format=" << (weightNz ? "FRACTAL_NZ" : "ND") << std::endl;
  std::cout << "weight_carrier=" << (packedInt32 ? "packed-INT32" : "INT4")
            << " special_nz=" << (specialNz ? 1 : 0) << std::endl;
  const int64_t G = withOffset ? 1 : (K + GS - 1) / GS;
  std::vector<int8_t> x(mTotal * K);
  std::mt19937 rng(randomSeed);
  auto hash32 = [](uint64_t value) {
    value ^= value >> 33;
    value *= 0xff51afd7ed558ccdULL;
    value ^= value >> 33;
    value *= 0xc4ceb9fe1a85ec53ULL;
    value ^= value >> 33;
    return static_cast<uint32_t>(value);
  };
  auto randomInt4At = [&](uint64_t index, uint64_t salt) {
    return static_cast<int8_t>(static_cast<int32_t>(
        hash32(index ^ (static_cast<uint64_t>(randomSeed) << 32) ^ salt) & 0x0fU) - 8);
  };
  auto logicalWeightAt = [&](int64_t e, int64_t k, int64_t n) {
    const uint64_t index =
        (static_cast<uint64_t>(e) * static_cast<uint64_t>(K) + static_cast<uint64_t>(k)) *
            static_cast<uint64_t>(N) +
        static_cast<uint64_t>(n);
    if (dataMode == "zero") return static_cast<int8_t>(0);
    if (dataMode == "one") return static_cast<int8_t>(1);
    if (dataMode == "extreme") return static_cast<int8_t>(index % 3 == 0 ? -8 : 7);
    if (dataMode == "pattern") return static_cast<int8_t>((index * 7 + 1) % 16 - 8);
    return randomInt4At(index, 0x3c6ef372fe94f82bULL);
  };
  if (dataMode == "zero") {
    std::fill(x.begin(), x.end(), 0);
  } else if (dataMode == "one") {
    std::fill(x.begin(), x.end(), 1);
  } else if (dataMode == "extreme") {
    for (size_t i = 0; i < x.size(); ++i) x[i] = i % 2 == 0 ? -8 : 7;
  } else if (dataMode == "row-varying") {
    for (int64_t m = 0; m < mTotal; ++m) {
      const int32_t amplitude = 1 << (m % 4);
      for (int64_t k = 0; k < K; ++k) {
        const uint32_t value = hash32(
            static_cast<uint64_t>(m * K + k) ^ (static_cast<uint64_t>(randomSeed) << 32));
        x[m * K + k] = static_cast<int8_t>(
            static_cast<int32_t>(value % static_cast<uint32_t>(2 * amplitude)) - amplitude);
      }
    }
  } else if (dataMode == "random") {
    for (size_t i = 0; i < x.size(); ++i) {
      x[i] = randomInt4At(i, 0xa54ff53a5f1d36f1ULL);
    }
  } else if (dataMode == "pattern") {
    for (int64_t i = 0; i < mTotal * K; ++i) x[i] = static_cast<int8_t>((i * 5 + 3) % 16 - 8);
  } else {
    std::cerr << "unsupported A8W4_DATA_MODE=" << dataMode << std::endl;
    return false;
  }

  std::vector<int64_t> weightStorageShape = {E, K, N};
  int64_t physicalScalarCount = E * K * N;
  if (weightNz) {
    if (specialNz) {
      const int64_t kBlocks = (K + 63) / 64;
      const int64_t nBlocks = (N + 15) / 16;
      weightStorageShape = {E, kBlocks, nBlocks, 16, 64};
      physicalScalarCount = E * kBlocks * nBlocks * 16 * 64;
      std::cout << "weight_storage_shape=[" << E << ',' << kBlocks << ',' << nBlocks
                << ",16,64]" << std::endl;
    } else {
      const int64_t nBlocks = (N + 63) / 64;
      const int64_t kBlocks = (K + 15) / 16;
      weightStorageShape = {E, nBlocks, kBlocks, 16, 64};
      physicalScalarCount = E * nBlocks * kBlocks * 16 * 64;
      std::cout << "weight_storage_shape=[" << E << ',' << nBlocks << ',' << kBlocks
                << ",16,64]" << std::endl;
    }
  }
  if (physicalScalarCount % 2 != 0) {
    std::cerr << "physical INT4 scalar count must be even" << std::endl;
    return false;
  }

  // Pack directly into the final ND/NZ storage. Keeping a second full-size
  // physical INT8 weight would add hundreds of MiB for the NET shapes.
  std::vector<uint8_t> packedW(static_cast<size_t>(physicalScalarCount / 2), 0);
  auto setPackedInt4 = [&](int64_t physicalIndex, int8_t value) {
    const size_t byteIndex = static_cast<size_t>(physicalIndex / 2);
    const uint8_t nibble = static_cast<uint8_t>(value) & 0x0fU;
    if ((physicalIndex & 1) == 0) {
      packedW[byteIndex] = static_cast<uint8_t>((packedW[byteIndex] & 0xf0U) | nibble);
    } else {
      packedW[byteIndex] = static_cast<uint8_t>((packedW[byteIndex] & 0x0fU) | (nibble << 4));
    }
  };
  if (weightNz) {
    if (specialNz) {
      const int64_t kBlocks = (K + 63) / 64;
      const int64_t nBlocks = (N + 15) / 16;
      for (int64_t e = 0; e < E; ++e) {
        for (int64_t n = 0; n < N; ++n) {
          for (int64_t k = 0; k < K; ++k) {
            const int64_t physicalIndex =
                ((((e * kBlocks + k / 64) * nBlocks + n / 16) * 16 + n % 16) * 64 + k % 64);
            setPackedInt4(physicalIndex, logicalWeightAt(e, k, n));
          }
        }
      }
    } else {
      const int64_t nBlocks = (N + 63) / 64;
      const int64_t kBlocks = (K + 15) / 16;
      for (int64_t e = 0; e < E; ++e) {
        for (int64_t n = 0; n < N; ++n) {
          for (int64_t k = 0; k < K; ++k) {
            const int64_t physicalIndex =
                ((((e * nBlocks + n / 64) * kBlocks + k / 16) * 16 + k % 16) * 64 + n % 64);
            setPackedInt4(physicalIndex, logicalWeightAt(e, k, n));
          }
        }
      }
    }
  } else {
    for (int64_t e = 0; e < E; ++e) {
      for (int64_t k = 0; k < K; ++k) {
        const int64_t rowBase = (e * K + k) * N;
        for (int64_t n = 0; n < N; ++n) {
          setPackedInt4(rowBase + n, logicalWeightAt(e, k, n));
        }
      }
    }
  }

  std::vector<float> floatScale(E * G * N), fusedOffset(withOffset ? E * N : 0), bias(E * N, 0.0f);
  // Binary-exact scales keep the CPU reference independent of FP19 truncation.
  std::uniform_int_distribution<int32_t> scaleNumeratorDist(1, 7);
  static constexpr float SCALE_TIERS[] = {0.125F, 0.25F, 0.5F, 0.75F, 1.0F};
  for (size_t i = 0; i < floatScale.size(); ++i) {
    const int32_t numerator = dataMode == "random" || dataMode == "row-varying" ?
        scaleNumeratorDist(rng) : static_cast<int32_t>(1 + i % 3);
    if (scaleProfile == "zero") {
      floatScale[i] = 0.0F;
    } else if (scaleProfile == "one") {
      floatScale[i] = 1.0F;
    } else if (scaleProfile == "tiers") {
      floatScale[i] = SCALE_TIERS[i % (sizeof(SCALE_TIERS) / sizeof(SCALE_TIERS[0]))];
    } else if (scaleProfile == "varying") {
      floatScale[i] = static_cast<float>(numerator) / 128.0F;
    } else {
      std::cerr << "unsupported A8W4_SCALE_PROFILE=" << scaleProfile << std::endl;
      return false;
    }
  }
  if (withOffset) {
    std::uniform_int_distribution<int32_t> offsetMultiplierDist(-3, 3);
    for (int64_t e = 0; e < E; ++e) for (int64_t n = 0; n < N; ++n)
      fusedOffset[e * N + n] = scaleProfile == "zero" ? 0.0F :
          static_cast<float>(dataMode == "random" || dataMode == "row-varying" ?
                                 offsetMultiplierDist(rng) :
                                 static_cast<int32_t>((e + n) % 5 - 2)) *
          floatScale[e * N + n];
  }

  uint64_t *quantRaw = nullptr, quantSize = 0;
  ACL_CHECK(aclnnTransQuantParam(floatScale.data(), floatScale.size(), nullptr, 0, &quantRaw, &quantSize));
  std::vector<uint64_t> quantScale(quantRaw, quantRaw + quantSize);
  free(quantRaw);
  std::cout << "quantScale size=" << quantScale.size() << std::endl;
  if (verboseScale) {
    for (size_t i = 0; i < quantScale.size(); ++i) {
      std::cout << "quantScale[" << i << "]=" << quantScale[i]
                << " (0x" << std::hex << quantScale[i] << std::dec << ")\n";
    }
  }
  std::vector<float> perToken(mTotal, scaleProfile == "zero" ? 0.0F : 1.0F);
  std::vector<uint16_t> output(mTotal * N, FloatToHalf(0.0f));

  TensorOwner tx, tw, tb, ts, toff, tpt, tgroup, ty;
  std::vector<int64_t> weightLogicalShape = specialNz ?
      std::vector<int64_t>{E, N, K} : std::vector<int64_t>{E, K, N};
  std::vector<int64_t> carrierStorageShape = weightStorageShape;
  if (packedInt32) {
    weightLogicalShape.back() /= 8;
    carrierStorageShape.back() /= 8;
  }
  if (!MakeTensor(x.data(), x.size(), {mTotal, K}, ACL_INT8, tx) ||
      !(weightNz
            ? MakeTensor(packedW.data(), packedW.size(), weightLogicalShape, carrierStorageShape,
                         packedInt32 ? ACL_INT32 : ACL_INT4, ACL_FORMAT_FRACTAL_NZ, tw)
            : MakeTensor(packedW.data(), packedW.size(), weightLogicalShape,
                         packedInt32 ? ACL_INT32 : ACL_INT4, tw)) ||
      !MakeTensor(bias.data(), bias.size() * 4, {E, N}, ACL_FLOAT, tb) ||
      !MakeTensor(quantScale.data(), quantScale.size() * 8, {E, G, N}, ACL_UINT64, ts) ||
      !MakeTensor(perToken.data(), perToken.size() * 4, {mTotal}, ACL_FLOAT, tpt) ||
      !MakeTensor(groups.data(), groups.size() * 8, {E}, ACL_INT64, tgroup) ||
      !MakeTensor(output.data(), output.size() * 2, {mTotal, N}, ACL_FLOAT16, ty)) return false;
  if (withOffset && !MakeTensor(fusedOffset.data(), fusedOffset.size() * 4, {E, 1, N}, ACL_FLOAT, toff)) return false;

  ListOwner lx, lw, lb, ls, loff, lpt, ly;
  if (!MakeList(tx, lx) || !MakeList(tw, lw) || !MakeList(tb, lb) || !MakeList(ts, ls) ||
      !MakeList(tpt, lpt) || !MakeList(ty, ly) || (withOffset && !MakeList(toff, loff))) return false;

  const int64_t tuningData[] = {
      *std::max_element(groups.begin(), groups.end()), specialNz ? 1 : 0, -1};
  aclIntArray *tuning = aclCreateIntArray(tuningData, 3);
  uint64_t workspaceSize = 0; aclOpExecutor *executor = nullptr;
  Stage("GetWorkspaceSize begin");
  aclnnStatus rc = ACL_SUCCESS;
  if (api == "WeightNz") {
    rc = aclnnGroupedMatmulWeightNzGetWorkspaceSize(
        lx.list, lw.list, lb.list, ls.list, withOffset ? loff.list : nullptr,
        nullptr, nullptr, lpt.list, tgroup.tensor, nullptr, nullptr, nullptr,
        3, 0, 1, 0, tuning, withOffset ? 0 : GS, ly.list, nullptr, nullptr,
        &workspaceSize, &executor);
  } else {
    rc = aclnnGroupedMatmulV5GetWorkspaceSize(
        lx.list, lw.list, lb.list, ls.list, withOffset ? loff.list : nullptr,
        nullptr, nullptr, lpt.list, tgroup.tensor, nullptr, nullptr, nullptr,
        3, 0, 1, 0, tuning, ly.list, nullptr, nullptr, &workspaceSize, &executor);
  }
  Stage("GetWorkspaceSize end");
  aclDestroyIntArray(tuning);
  if (rc != ACL_SUCCESS) { std::cerr << "GetWorkspace failed: " << rc << "\n" << aclGetRecentErrMsg() << std::endl; return false; }
  void *workspace = nullptr;
  if (workspaceSize) ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
  aclrtStream stream = nullptr; ACL_CHECK(aclrtCreateStream(&stream));
  Stage("kernel launch begin");
  rc = api == "WeightNz" ?
      aclnnGroupedMatmulWeightNz(workspace, workspaceSize, executor, stream) :
      aclnnGroupedMatmulV5(workspace, workspaceSize, executor, stream);
  Stage("kernel launch end");
  if (rc == ACL_SUCCESS) {
    Stage("stream sync begin");
    rc = aclrtSynchronizeStream(stream);
    Stage("stream sync end");
  }
  aclrtDestroyStream(stream); if (workspace) aclrtFree(workspace);
  if (rc != ACL_SUCCESS) { std::cerr << "Execute failed: " << rc << "\n" << aclGetRecentErrMsg() << std::endl; return false; }
  Stage("D2H begin");
  ACL_CHECK(aclrtMemcpy(output.data(), output.size() * 2, ty.device, output.size() * 2, ACL_MEMCPY_DEVICE_TO_HOST));
  Stage("D2H end");

  uint64_t outputChecksum = 1469598103934665603ULL;
  uint64_t nonFiniteCount = 0;
  for (uint16_t value : output) {
    outputChecksum ^= static_cast<uint64_t>(value);
    outputChecksum *= 1099511628211ULL;
    if (!std::isfinite(HalfToFloat(value))) ++nonFiniteCount;
  }

  const uint64_t outputElements = static_cast<uint64_t>(mTotal) * static_cast<uint64_t>(N);
  const uint64_t requestedSamples = static_cast<uint64_t>(
      std::max<int64_t>(1, GetEnvInt64("A8W4_GOLDEN_SAMPLES", 1024)));
  std::set<uint64_t> sampleIndices;
  auto addSample = [&](int64_t m, int64_t n) {
    if (m >= 0 && m < mTotal && n >= 0 && n < N) {
      sampleIndices.insert(static_cast<uint64_t>(m) * static_cast<uint64_t>(N) +
                           static_cast<uint64_t>(n));
    }
  };
  if (GetEnvFlag("A8W4_FULL_GOLDEN") || outputElements <= requestedSamples) {
    for (uint64_t index = 0; index < outputElements; ++index) sampleIndices.insert(index);
  } else {
    const int64_t nBoundaries[] = {
        0, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, N - 1};
    std::vector<int64_t> mBoundaries = {0, 1, 7, 8, 15, 16, 17, 255, 256, 257, mTotal - 1};
    int64_t groupStart = 0;
    for (int64_t e = 0; e < E; ++e) {
      if (groups[e] > 0) {
        mBoundaries.push_back(groupStart);
        mBoundaries.push_back(groupStart + groups[e] - 1);
      }
      groupStart += groups[e];
    }
    for (int64_t m : mBoundaries) {
      for (int64_t n : nBoundaries) addSample(m, n);
    }
    std::mt19937 sampleRng(randomSeed ^ 0x6a09e667U);
    std::uniform_int_distribution<int64_t> mDist(0, mTotal - 1);
    std::uniform_int_distribution<int64_t> nDist(0, N - 1);
    while (sampleIndices.size() < std::min<uint64_t>(requestedSamples, outputElements)) {
      addSample(mDist(sampleRng), nDist(sampleRng));
    }
  }

  float maxErr = 0.0f;
  float maxAllowed = 0.0f;
  int64_t maxErrM = 0, maxErrN = 0;
  float maxErrGot = 0.0f, maxErrRef = 0.0f;
  bool samplesPass = true;
  std::ofstream goldenFile;
  std::string goldenPath;
  const char *goldenPathEnv = std::getenv("A8W4_GOLDEN_PATH");
  if (goldenPathEnv != nullptr && *goldenPathEnv != '\0') {
    goldenPath = goldenPathEnv;
    goldenFile.open(goldenPath, std::ios::out | std::ios::trunc);
    if (!goldenFile) {
      std::cerr << "failed to open golden file: " << goldenPath << std::endl;
      return false;
    }
    goldenFile << "seed,m,n,expert,raw_mmad,row_sum,scale,quant_scale_hex,fused_offset,"
                  "fixpipe_golden,offset_correction,final_golden,device_output,inferred_fixpipe,error,tolerance\n";
    goldenFile << std::setprecision(10);
  }
  uint64_t printedSamples = 0;
  for (uint64_t sampleIndex : sampleIndices) {
    const int64_t m = static_cast<int64_t>(sampleIndex / static_cast<uint64_t>(N));
    const int64_t n = static_cast<int64_t>(sampleIndex % static_cast<uint64_t>(N));
    const int64_t e = expertForRow[m];
    int64_t rawMmad = 0;
    int64_t rowSum = 0;
    float ref = 0.0f;
    std::vector<int64_t> groupMmad(static_cast<size_t>(G), 0);
    for (int64_t k = 0; k < K; ++k) {
      float s = floatScale[(e * G + (withOffset ? 0 : k / GS)) * N + n];
      const int8_t logicalWeight = logicalWeightAt(e, k, n);
      float w = logicalWeight * s + (withOffset ? fusedOffset[e * N + n] : 0.0f);
      ref += x[m * K + k] * w;
      const int64_t mmadValue = static_cast<int64_t>(x[m * K + k]) * logicalWeight;
      rawMmad += mmadValue;
      groupMmad[static_cast<size_t>(withOffset ? 0 : k / GS)] += mmadValue;
      if (n == 0) rowSum += x[m * K + k];
    }
    if (n != 0) {
      for (int64_t k = 0; k < K; ++k) rowSum += x[m * K + k];
    }
    const float channelScale = floatScale[e * G * N + n];
    float fixpipeGolden = 0.0F;
    float fixpipeUlpSum = 0.0F;
    for (int64_t g = 0; g < G; ++g) {
      const float partial = static_cast<float>(groupMmad[static_cast<size_t>(g)]) *
                            floatScale[(e * G + g) * N + n];
      fixpipeGolden += partial;
      fixpipeUlpSum += HalfUlp(partial);
    }
    const float offsetCorrection = withOffset ? static_cast<float>(rowSum) * fusedOffset[e * N + n] : 0.0f;
    const float got = HalfToFloat(output[m * N + n]);
    const float inferredFixpipe = got - offsetCorrection;
    const float err = std::fabs(got - ref);
    // Each per-group partial is independently converted by FixPipe before the
    // FP16 partials are accumulated. Sum their ULPs so cancellation and a
    // large number of quant groups receive a scale-aware tolerance.
    const float tolerance = 0.2F + 1.5F * (fixpipeUlpSum + HalfUlp(ref));
    if (printedSamples < 8) {
      std::cout << "value[" << m << ',' << n << "] raw=" << rawMmad
                << " scale=" << channelScale
                << " fix_golden=" << fixpipeGolden
                << " fix_inferred=" << inferredFixpipe
                << " got=" << got << " ref=" << ref << '\n';
      ++printedSamples;
    }
    if (goldenFile) {
      goldenFile << randomSeed << ',' << m << ',' << n << ',' << e << ',' << rawMmad << ',' << rowSum << ','
                 << channelScale << ",0x" << std::hex << quantScale[e * G * N + n] << std::dec << ','
                 << (withOffset ? fusedOffset[e * N + n] : 0.0f) << ',' << fixpipeGolden << ','
                 << offsetCorrection << ',' << ref << ',' << got << ',' << inferredFixpipe << ',' << err << ','
                 << tolerance << '\n';
    }
    if (!std::isfinite(got) || err > tolerance) samplesPass = false;
    if (err > maxErr) {
      maxErr = err;
      maxAllowed = tolerance;
      maxErrM = m;
      maxErrN = n;
      maxErrGot = got;
      maxErrRef = ref;
    }
  }
  std::cout << "max_error_at=[" << maxErrM << ',' << maxErrN << "] got=" << maxErrGot
            << " ref=" << maxErrRef << " tolerance=" << maxAllowed << std::endl;
  std::cout << "ACLNN offset=" << (withOffset ? "non-null" : "null") << " max_abs_error=" << maxErr << std::endl;
  std::cout << "sample_count=" << sampleIndices.size() << " output_elements=" << outputElements
            << " non_finite=" << nonFiniteCount << " output_checksum=0x"
            << std::hex << outputChecksum << std::dec << std::endl;
  if (!goldenPath.empty()) std::cout << "golden_csv=" << goldenPath << std::endl;
  return samplesPass && nonFiniteCount == 0;
}
}  // namespace

int main(int argc, char **argv) {
  Stage("aclInit begin");
  if (aclInit(nullptr) != ACL_SUCCESS) return 1;
  Stage("aclInit end");
  if (aclrtSetDevice(0) != ACL_SUCCESS) return 1;
  Stage("setDevice end");
  bool ok = RunCase(argc > 1 && std::string(argv[1]) == "non-null");
  Stage("RunCase end");
  aclrtResetDevice(0); aclFinalize();
  std::cout << (ok ? "PASS" : "FAIL") << std::endl;
  // Some CANN test environments corrupt or block during process-global library
  // teardown after ACL resources have already been reset/finalized. Matrix
  // testing may opt out of that unrelated libc teardown path.
  if (GetEnvFlag("A8W4_FAST_EXIT")) std::_Exit(ok ? 0 : 1);
  return ok ? 0 : 1;
}
