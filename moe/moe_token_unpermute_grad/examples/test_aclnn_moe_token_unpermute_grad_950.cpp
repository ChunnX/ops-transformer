#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_unpermute_grad.h"

#define CHECK_RET(cond, return_expr) \
  do { \
    if (!(cond)) { return_expr; } \
  } while (0)

#define LOG_PRINT(message, ...) \
  do { printf(message, ##__VA_ARGS__); } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) shapeSize *= i;
  return shapeSize;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) strides[i] = shape[i + 1] * strides[i + 1];
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int ReadResult(void **deviceAddr, std::vector<float> &resultData, int64_t size) {
  resultData.resize(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), *deviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);
  return 0;
}

int Init(int32_t deviceId, aclrtStream *stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

// Golden computation for probs_not_none path
// probs is 2D [numTokens, topK], flattened to [numExpertTokens] = numTokens * topK
void ComputeGoldenProbNotNone(const std::vector<float> &permutedTokens,
                              const std::vector<float> &unpermutedTokensGrad,
                              const std::vector<int32_t> &sortedIndices,
                              const std::vector<float> &probs,
                              int64_t numTokens, int64_t topK, int64_t hidden,
                              std::vector<float> &goldenGradX,
                              std::vector<float> &goldenProbsGrad) {
  int64_t numExpertTokens = numTokens * topK;
  goldenGradX.assign(numExpertTokens * hidden, 0.0f);
  goldenProbsGrad.assign(numExpertTokens, 0.0f);
  
  for (int64_t i = 0; i < numExpertTokens; i++) {
    int32_t origIdx = sortedIndices[i];
    if (origIdx < 0 || origIdx >= numTokens) continue;
    
    // probs is 2D [numTokens, topK], flattened as probs[tokenIdx * topK + k]
    // For expert token i, the corresponding prob is probs[i] (flat)
    float prob = probs[i];
    
    // grad_x[i] = grad_y[origIdx] * prob
    for (int64_t h = 0; h < hidden; h++) {
      goldenGradX[i * hidden + h] = unpermutedTokensGrad[origIdx * hidden + h] * prob;
    }
    
    // grad_probs[i] = sum(grad_y[origIdx] * x[i], axis=hidden)
    float dot = 0.0f;
    for (int64_t h = 0; h < hidden; h++) {
      dot += unpermutedTokensGrad[origIdx * hidden + h] * permutedTokens[i * hidden + h];
    }
    goldenProbsGrad[i] = dot;
  }
}

// Golden computation for probs_none path
void ComputeGoldenProbNone(const std::vector<float> &unpermutedTokensGrad,
                           const std::vector<int32_t> &sortedIndices,
                           int64_t numTokens, int64_t hidden,
                           std::vector<float> &goldenGradX) {
  int64_t numExpertTokens = sortedIndices.size();
  goldenGradX.assign(numExpertTokens * hidden, 0.0f);
  
  for (int64_t i = 0; i < numExpertTokens; i++) {
    int32_t origIdx = sortedIndices[i];
    if (origIdx < 0 || origIdx >= numTokens) continue;
    for (int64_t h = 0; h < hidden; h++) {
      goldenGradX[i * hidden + h] = unpermutedTokensGrad[origIdx * hidden + h];
    }
  }
}

bool CompareResult(const std::vector<float> &actual, const std::vector<float> &golden, float atol, float rtol) {
  if (actual.size() != golden.size()) {
    printf("  Size mismatch: actual=%zu, golden=%zu\n", actual.size(), golden.size());
    return false;
  }
  float maxDiff = 0.0f;
  int failCount = 0;
  for (size_t i = 0; i < actual.size(); i++) {
    float diff = std::abs(actual[i] - golden[i]);
    float threshold = atol + rtol * std::abs(golden[i]);
    if (diff > threshold) {
      failCount++;
      if (failCount <= 5) {
        printf("  Mismatch at [%zu]: actual=%f, golden=%f, diff=%f\n", i, actual[i], golden[i], diff);
      }
    }
    if (diff > maxDiff) maxDiff = diff;
  }
  printf("  Max diff: %f, fail count: %d/%zu\n", maxDiff, failCount, actual.size());
  return failCount == 0;
}

int RunTestCase(int64_t numTokens, int64_t topK, int64_t hidden, bool hasProbs) {
  printf("\n=== Test: tokens=%ld, topK=%ld, hidden=%ld, hasProbs=%d ===\n", 
         numTokens, topK, hidden, hasProbs);
  
  int64_t numExpertTokens = numTokens * topK;
  
  // Generate random inputs
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::uniform_int_distribution<int32_t> idxDist(0, numTokens - 1);
  
  std::vector<float> permutedTokens(numExpertTokens * hidden);
  std::vector<float> unpermutedTokensGrad(numTokens * hidden);
  std::vector<int32_t> sortedIndices(numExpertTokens);
  std::vector<float> probs(numTokens * topK);
  
  for (auto &v : permutedTokens) v = dist(rng);
  for (auto &v : unpermutedTokensGrad) v = dist(rng);
  for (auto &v : sortedIndices) v = idxDist(rng);
  for (auto &v : probs) v = dist(rng);
  
  // Compute golden
  std::vector<float> goldenGradX, goldenProbsGrad;
  if (hasProbs) {
    ComputeGoldenProbNotNone(permutedTokens, unpermutedTokensGrad, sortedIndices, probs,
                             numTokens, topK, hidden, goldenGradX, goldenProbsGrad);
  } else {
    ComputeGoldenProbNone(unpermutedTokensGrad, sortedIndices, numTokens, hidden, goldenGradX);
  }
  
  // Create ACL tensors
  std::vector<int64_t> permutedTokensShape = {numExpertTokens, hidden};
  std::vector<int64_t> unpermutedGradShape = {numTokens, hidden};
  std::vector<int64_t> sortedIndicesShape = {numExpertTokens};
  std::vector<int64_t> probsShape = {numTokens, topK};
  std::vector<int64_t> gradXShape = {numExpertTokens, hidden};
  std::vector<int64_t> probsGradShape = {numTokens, topK};
  
  void *permutedTokensAddr = nullptr, *unpermutedGradAddr = nullptr;
  void *sortedIndicesAddr = nullptr, *probsAddr = nullptr;
  void *gradXAddr = nullptr, *probsGradAddr = nullptr;
  aclTensor *permutedTokensTensor = nullptr, *unpermutedGradTensor = nullptr;
  aclTensor *sortedIndicesTensor = nullptr, *probsTensor = nullptr;
  aclTensor *gradXTensor = nullptr, *probsGradTensor = nullptr;
  
  auto ret = CreateAclTensor(permutedTokens, permutedTokensShape, &permutedTokensAddr, aclDataType::ACL_FLOAT, &permutedTokensTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(unpermutedTokensGrad, unpermutedGradShape, &unpermutedGradAddr, aclDataType::ACL_FLOAT, &unpermutedGradTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sortedIndices, sortedIndicesShape, &sortedIndicesAddr, aclDataType::ACL_INT32, &sortedIndicesTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  if (hasProbs) {
    ret = CreateAclTensor(probs, probsShape, &probsAddr, aclDataType::ACL_FLOAT, &probsTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }
  
  std::vector<float> gradXHost(numExpertTokens * hidden, 0.0f);
  std::vector<float> probsGradHost(numExpertTokens, 0.0f);
  ret = CreateAclTensor(gradXHost, gradXShape, &gradXAddr, aclDataType::ACL_FLOAT, &gradXTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(probsGradHost, probsGradShape, &probsGradAddr, aclDataType::ACL_FLOAT, &probsGradTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // Call aclnnMoeTokenUnpermuteGrad
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor = nullptr;
  bool paddedMode = false;
  
  ret = aclnnMoeTokenUnpermuteGradGetWorkspaceSize(
      permutedTokensTensor, unpermutedGradTensor, sortedIndicesTensor, probsTensor,
      paddedMode, nullptr, gradXTensor, probsGradTensor, &workspaceSize, &executor);
  if (ret != ACL_SUCCESS) {
    printf("  aclnnMoeTokenUnpermuteGradGetWorkspaceSize failed: %d\n", ret);
    return ret;
  }
  
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, printf("malloc workspace failed\n"); return ret);
  }
  
  aclrtStream stream;
  ret = aclrtCreateStream(&stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  ret = aclnnMoeTokenUnpermuteGrad(workspaceAddr, workspaceSize, executor, stream);
  if (ret != ACL_SUCCESS) {
    printf("  aclnnMoeTokenUnpermuteGrad failed: %d\n", ret);
    return ret;
  }
  aclrtSynchronizeStream(stream);
  
  // Read results
  std::vector<float> actualGradX, actualProbsGrad;
  ReadResult(&gradXAddr, actualGradX, numExpertTokens * hidden);
  if (hasProbs) {
    ReadResult(&probsGradAddr, actualProbsGrad, numExpertTokens);
  }
  
  // Compare
  bool gradOk = CompareResult(actualGradX, goldenGradX, 1e-4f, 1e-4f);
  bool probsOk = true;
  if (hasProbs) {
    probsOk = CompareResult(actualProbsGrad, goldenProbsGrad, 1e-4f, 1e-4f);
  }
  
  if (gradOk && probsOk) {
    printf("  PASS\n");
  } else {
    printf("  FAILED: gradOk=%d, probsOk=%d\n", gradOk, probsOk);
  }
  
  // Cleanup
  aclDestroyTensor(permutedTokensTensor);
  aclDestroyTensor(unpermutedGradTensor);
  aclDestroyTensor(sortedIndicesTensor);
  if (probsTensor) aclDestroyTensor(probsTensor);
  aclDestroyTensor(gradXTensor);
  aclDestroyTensor(probsGradTensor);
  aclrtFree(permutedTokensAddr);
  aclrtFree(unpermutedGradAddr);
  aclrtFree(sortedIndicesAddr);
  if (probsAddr) aclrtFree(probsAddr);
  aclrtFree(gradXAddr);
  aclrtFree(probsGradAddr);
  if (workspaceAddr) aclrtFree(workspaceAddr);
  aclrtDestroyStream(stream);
  
  return (gradOk && probsOk) ? 0 : 1;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  int passed = 0, failed = 0;
  
  // Test cases
  struct TestCase { int64_t tokens; int64_t topK; int64_t hidden; bool hasProbs; };
  TestCase tests[] = {
    {10, 1, 64, false},   // prob_none
    {10, 1, 64, true},     // prob_not_none (topk=1)
    {10, 3, 64, true},     // prob_not_none (topk=3)
    {100, 2, 128, true},   // larger shape
    {10, 1, 64, false},    // prob_none repeat
    {50, 4, 256, true},    // larger topk
  };
  
  for (auto &tc : tests) {
    int r = RunTestCase(tc.tokens, tc.topK, tc.hidden, tc.hasProbs);
    if (r == 0) passed++; else failed++;
  }
  
  printf("\n============================================================\n");
  printf("Results: %d passed, %d failed out of %d\n", passed, failed, (int)(sizeof(tests)/sizeof(tests[0])));
  
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  
  return (failed > 0) ? 1 : 0;
}
