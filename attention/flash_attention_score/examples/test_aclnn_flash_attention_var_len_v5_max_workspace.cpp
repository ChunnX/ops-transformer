/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * 测试 aclnnFlashAttentionVarLenScoreV5GetMaxWorkspaceSize 返回值
 * >= aclnnFlashAttentionVarLenScoreV5GetWorkspaceSize 返回值
 *
 * 测试维度:
 *   - 数据类型: BF16 / FP16 / FP32
 *   - 可选输入: attenMask, pse(pseType=1/2/3), rope, prefix, qStartIdx/kvStartIdx
 *   - sparseMode: 0~3, 6
 *   - GQA(N1!=N2), MLA(Dv<D), 多 batch, softmaxOutLayout
 *   - attenMask dtype: UINT8 / BOOL
 * 不测试: dropout, sink, paddingMask(预留未使用)
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_flash_attention_score.h"

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
  for (auto i : shape) {
    shapeSize *= i;
  }
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
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

// ==================== 测试配置 ====================
struct TestConfig {
  const char* name;
  aclDataType dtype;         // Q/K/V 数据类型
  int64_t N1;                // query 头数
  int64_t N2;                // kv 头数
  int64_t D;                 // Q/K head dim
  int64_t Dv;                // V head dim
  bool hasRope;              // queryRope/keyRope (仅 BF16)
  int64_t D_rope;            // rope head dim
  bool hasAttenMask;
  aclDataType maskDtype;     // ACL_UINT8 或 ACL_BOOL
  int64_t sparseMode;        // 0~8, 不含 5
  bool hasPse;
  int64_t pseType;           // 0=outer_mul_add, 1=outer_add_mul, 2=inner_mul_add, 3=inner_mul_add_sqrt
  bool hasPrefix;
  bool hasStartIdx;          // qStartIdx + kvStartIdx
  bool softmaxSameAsInput;   // true="same_as_input"(TND), false=""(NTD)
  int numBatches;
};

static int passedCount = 0;
static int failedCount = 0;
static int errorCount = 0;

int RunSingleTest(const TestConfig& cfg, aclrtStream stream) {
  LOG_PRINT("\n=== Running: %s ===\n", cfg.name);

  const int64_t T1 = 256, T2 = 256;

  // ---- 根据 config 计算所有 shape ----
  std::vector<int64_t> qShape  = {T1, cfg.N1, cfg.D};
  std::vector<int64_t> kShape  = {T2, cfg.N2, cfg.D};
  std::vector<int64_t> vShape  = {T2, cfg.N2, cfg.Dv};
  std::vector<int64_t> ropeQShape, ropeKShape;
  if (cfg.hasRope) {
    ropeQShape = {T1, cfg.N1, cfg.D_rope};
    ropeKShape = {T2, cfg.N2, cfg.D_rope};
  }
  std::vector<int64_t> maskShape = {T1, T2};

  // PSE shape 依赖 pseType
  std::vector<int64_t> pseShape;
  if (cfg.hasPse) {
    if (cfg.pseType == 2 || cfg.pseType == 3) {
      // inner pse (alibi slope): [N2]
      pseShape = {cfg.N2};
    } else {
      // outer pse: [B, N2, S1, S2]
      pseShape = {1, cfg.N2, T1, T2};
    }
  }

  // 输出 shape
  std::vector<int64_t> outShape = {T1, cfg.N1, cfg.Dv};
  std::vector<int64_t> smShape;
  if (cfg.softmaxSameAsInput) {
    smShape = {T1, cfg.N1, 8};     // TND: [T, N, 8]
  } else {
    smShape = {cfg.N1, T1, 8};     // NTD: [N, T, 8]
  }

  // ---- device 地址 & tensor 句柄 ----
  void* qAddr = nullptr;       void* kAddr = nullptr;       void* vAddr = nullptr;
  void* ropeQAddr = nullptr;   void* ropeKAddr = nullptr;
  void* maskAddr = nullptr;    void* pseAddr = nullptr;
  void* outAddr = nullptr;     void* smMaxAddr = nullptr;   void* smSumAddr = nullptr;

  aclTensor* q = nullptr;      aclTensor* k = nullptr;      aclTensor* v = nullptr;
  aclTensor* ropeQ = nullptr;  aclTensor* ropeK = nullptr;
  aclTensor* mask = nullptr;   aclTensor* pse = nullptr;
  aclTensor* out = nullptr;    aclTensor* smMax = nullptr;  aclTensor* smSum = nullptr;

  auto ret = ACL_SUCCESS;
  uint64_t ws = 0;
  aclOpExecutor* executor = nullptr;
  uint64_t wsMax = 0;
  aclOpExecutor* executorMax = nullptr;

  // ---- Q / K / V ----
  {
    std::vector<float> qData(GetShapeSize(qShape), 1);
    std::vector<float> kData(GetShapeSize(kShape), 1);
    std::vector<float> vData(GetShapeSize(vShape), 1);
    ret = CreateAclTensor(qData, qShape, &qAddr, cfg.dtype, &q);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  create q failed\n"); goto cleanup);
    ret = CreateAclTensor(kData, kShape, &kAddr, cfg.dtype, &k);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  create k failed\n"); goto cleanup);
    ret = CreateAclTensor(vData, vShape, &vAddr, cfg.dtype, &v);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  create v failed\n"); goto cleanup);
  }

  // ---- Rope ----
  if (cfg.hasRope) {
    std::vector<float> rqData(GetShapeSize(ropeQShape), 1);
    std::vector<float> rkData(GetShapeSize(ropeKShape), 1);
    ret = CreateAclTensor(rqData, ropeQShape, &ropeQAddr, ACL_BF16, &ropeQ);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  create ropeQ failed\n"); goto cleanup);
    ret = CreateAclTensor(rkData, ropeKShape, &ropeKAddr, ACL_BF16, &ropeK);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  create ropeK failed\n"); goto cleanup);
  }

  // ---- AttenMask ----
  if (cfg.hasAttenMask) {
    std::vector<uint8_t> maskData(GetShapeSize(maskShape), 0);
    ret = CreateAclTensor(maskData, maskShape, &maskAddr, cfg.maskDtype, &mask);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  create mask failed\n"); goto cleanup);
  }

  // ---- PSE ----
  if (cfg.hasPse) {
    // pseType=2/3: FP32 slope; pseType=0/1: 与 attentionOut 同类型
    aclDataType pseDtype = (cfg.pseType == 2 || cfg.pseType == 3) ? ACL_FLOAT : cfg.dtype;
    std::vector<float> pseData(GetShapeSize(pseShape), 1);
    ret = CreateAclTensor(pseData, pseShape, &pseAddr, pseDtype, &pse);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  create pse failed\n"); goto cleanup);
  }

  // ---- 输出张量 ----
  {
    std::vector<float> outData(GetShapeSize(outShape), 0);
    std::vector<float> smData(GetShapeSize(smShape), 0);
    ret = CreateAclTensor(outData, outShape, &outAddr, cfg.dtype, &out);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  create out failed\n"); goto cleanup);
    ret = CreateAclTensor(smData, smShape, &smMaxAddr, ACL_FLOAT, &smMax);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  create smMax failed\n"); goto cleanup);
    ret = CreateAclTensor(smData, smShape, &smSumAddr, ACL_FLOAT, &smSum);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("  create smSum failed\n"); goto cleanup);
  }

  // ---- IntArray 参数 ----
  aclIntArray* prefix = nullptr;
  if (cfg.hasPrefix) {
    std::vector<int64_t> prefixData(cfg.numBatches, 0);
    prefix = aclCreateIntArray(prefixData.data(), prefixData.size());
  }

  std::vector<int64_t> seqQLenData, seqKvLenData;
  if (cfg.numBatches == 1) {
    seqQLenData = {T1};
    seqKvLenData = {T2};
  } else {
    seqQLenData = {128, T1};
    seqKvLenData = {128, T2};
  }
  aclIntArray* acSeqQLen = aclCreateIntArray(seqQLenData.data(), seqQLenData.size());
  aclIntArray* acSeqKvLen = aclCreateIntArray(seqKvLenData.data(), seqKvLenData.size());

  aclIntArray* qStartIdx = nullptr;
  aclIntArray* kvStartIdx = nullptr;
  if (cfg.hasStartIdx) {
    std::vector<int64_t> qIdx(cfg.numBatches, 0);
    std::vector<int64_t> kvIdx(cfg.numBatches, 0);
    qStartIdx = aclCreateIntArray(qIdx.data(), qIdx.size());
    kvStartIdx = aclCreateIntArray(kvIdx.data(), kvIdx.size());
  }

  // ---- 公共参数 ----
  char layOut[] = "TND";
  char softmaxLayoutSame[] = "same_as_input";
  char softmaxLayoutEmpty[] = "";
  char* softmaxOutLayout = cfg.softmaxSameAsInput ? softmaxLayoutSame : softmaxLayoutEmpty;
  double scaleValue = 0.088388;
  double keepProb = 1.0;
  int64_t preTokens = 65536;
  int64_t nextTokens = 65536;
  int64_t headNum = cfg.N1;
  int64_t innerPrecise = 0;

  // ==================== 调用 GetWorkspaceSize ====================
  ret = aclnnFlashAttentionVarLenScoreV5GetWorkspaceSize(
      q, ropeQ, k, ropeK, v, pse,
      nullptr,    // dropMask
      nullptr,    // paddingMask (预留未使用)
      mask,       // attenMask
      nullptr,    // sink
      prefix, acSeqQLen, acSeqKvLen, qStartIdx, kvStartIdx,
      scaleValue, keepProb, preTokens, nextTokens, headNum, layOut,
      innerPrecise, cfg.sparseMode, cfg.pseType, softmaxOutLayout,
      smMax, smSum, nullptr, out,
      &ws, &executor);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("  [ERROR] GetWorkspaceSize failed: %d\n", ret);
    errorCount++;
    goto cleanup;
  }

  // ==================== 调用 GetMaxWorkspaceSize ====================
  ret = aclnnFlashAttentionVarLenScoreV5GetMaxWorkspaceSize(
      q, ropeQ, k, ropeK, v, pse,
      nullptr, nullptr, mask, nullptr,
      prefix, acSeqQLen, acSeqKvLen, qStartIdx, kvStartIdx,
      scaleValue, keepProb, preTokens, nextTokens, headNum, layOut,
      innerPrecise, cfg.sparseMode, cfg.pseType, softmaxOutLayout,
      smMax, smSum, nullptr, out,
      &wsMax, &executorMax);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("  [ERROR] GetMaxWorkspaceSize failed: %d\n", ret);
    errorCount++;
    goto cleanup;
  }

  // ==================== 比较结果 ====================
  LOG_PRINT("  WorkspaceSize     = %lu\n", ws);
  LOG_PRINT("  MaxWorkspaceSize  = %lu\n", wsMax);
  if (wsMax >= ws) {
    LOG_PRINT("  [PASS] %s\n", cfg.name);
    passedCount++;
  } else {
    LOG_PRINT("  [FAIL] %s: MaxWorkspace(%lu) < Workspace(%lu)\n", cfg.name, wsMax, ws);
    failedCount++;
  }

  // ==================== 用真实 workspace 执行计算 ====================
  {
    void* wsAddr = nullptr;
    if (ws > 0) {
      ret = aclrtMalloc(&wsAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
      if (ret != ACL_SUCCESS) {
        LOG_PRINT("  [WARN] workspace malloc failed\n");
        goto cleanup;
      }
    }
    ret = aclnnFlashAttentionVarLenScoreV5(wsAddr, ws, executor, stream);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("  [WARN] execution failed: %d\n", ret);
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
      LOG_PRINT("  [WARN] sync failed: %d\n", ret);
    }
    if (wsAddr != nullptr) {
      aclrtFree(wsAddr);
    }
  }

cleanup:
  if (q)       aclDestroyTensor(q);
  if (k)       aclDestroyTensor(k);
  if (v)       aclDestroyTensor(v);
  if (ropeQ)   aclDestroyTensor(ropeQ);
  if (ropeK)   aclDestroyTensor(ropeK);
  if (mask)    aclDestroyTensor(mask);
  if (pse)     aclDestroyTensor(pse);
  if (out)     aclDestroyTensor(out);
  if (smMax)   aclDestroyTensor(smMax);
  if (smSum)   aclDestroyTensor(smSum);

  if (qAddr)      aclrtFree(qAddr);
  if (kAddr)      aclrtFree(kAddr);
  if (vAddr)      aclrtFree(vAddr);
  if (ropeQAddr)  aclrtFree(ropeQAddr);
  if (ropeKAddr)  aclrtFree(ropeKAddr);
  if (maskAddr)   aclrtFree(maskAddr);
  if (pseAddr)    aclrtFree(pseAddr);
  if (outAddr)    aclrtFree(outAddr);
  if (smMaxAddr)  aclrtFree(smMaxAddr);
  if (smSumAddr)  aclrtFree(smSumAddr);

  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // =====================================================================
  //  20 个测试用例: 覆盖数据类型 / 可选输入 / sparseMode / GQA / MLA 等
  //  不含 dropout, sink, paddingMask(预留未使用)
  //
  //  {name, dtype, N1, N2, D, Dv, hasRope, D_rope,
  //   hasAttenMask, maskDtype, sparseMode,
  //   hasPse, pseType, hasPrefix, hasStartIdx,
  //   softmaxSameAsInput, numBatches}
  // =====================================================================
  TestConfig tests[] = {
    // ---- Group 1: BF16 + sparseMode 覆盖 ----
    // 最小输入: 只有 Q/K/V, 无可选输入
    {"BF16_minimal",             ACL_BF16, 1,1, 128,128, false,0,  false,ACL_BOOL,0,  false,1, false,false, true, 1},
    // + attenMask(UINT8), sparseMode=1(ALL_MASK)
    {"BF16_attenMask_sparse1",   ACL_BF16, 1,1, 128,128, false,0,  true, ACL_UINT8,1, false,1, false,false, true, 1},
    // + attenMask(UINT8), sparseMode=2(LEFT_UP_CAUSAL)
    {"BF16_attenMask_sparse2",   ACL_BF16, 1,1, 128,128, false,0,  true, ACL_UINT8,2, false,1, false,false, true, 1},
    // + attenMask(UINT8), sparseMode=3(RIGHT_DOWN_CAUSAL, 要求 Sq<=Skv)
    {"BF16_attenMask_sparse3",   ACL_BF16, 1,1, 128,128, false,0,  true, ACL_UINT8,3, false,1, false,false, true, 1},

    // ---- Group 2: PSE 类型覆盖 ----
    // + attenMask + pse(pseType=1, outer_add_mul, PSE dtype=BF16, shape=[1,N2,S1,S2])
    {"BF16_attenMask_pse1",      ACL_BF16, 1,1, 128,128, false,0,  true, ACL_UINT8,1, true, 1, false,false, true, 1},
    // + attenMask + pse(pseType=2, inner_mul_add, PSE dtype=FP32, slope shape=[N2])
    {"BF16_attenMask_pse2",      ACL_BF16, 1,1, 128,128, false,0,  true, ACL_UINT8,1, true, 2, false,false, true, 1},
    // + attenMask + pse(pseType=3, inner_mul_add_sqrt, PSE dtype=FP32, slope shape=[N2])
    {"BF16_attenMask_pse3",      ACL_BF16, 1,1, 128,128, false,0,  true, ACL_UINT8,1, true, 3, false,false, true, 1},

    // ---- Group 3: 特殊可选输入 ----
    // + attenMask + rope (要求 BF16 + attenMask + pse=null + keepProb=1, pseType 仅支持 1)
    {"BF16_attenMask_rope",      ACL_BF16, 1,1, 128,128, true, 64,true, ACL_UINT8,0, false,1, false,false, true, 1},
    // + attenMask + prefix, sparseMode=6(PREFIX_COMPRESS)
    {"BF16_attenMask_prefix",    ACL_BF16, 1,1, 128,128, false,0,  true, ACL_UINT8,6, false,1, true, false, true, 1},
    // + attenMask + qStartIdx/kvStartIdx
    {"BF16_attenMask_startIdx",  ACL_BF16, 1,1, 128,128, false,0,  true, ACL_UINT8,0, false,1, false,true,  true, 1},
    // + attenMask(BOOL dtype), 验证 BOOL 掩码
    {"BF16_attenMask_BOOL",      ACL_BF16, 1,1, 128,128, false,0,  true, ACL_BOOL, 1, false,1, false,false, true, 1},

    // ---- Group 4: Shape 变化 ----
    // GQA: N1=4, N2=2, Nq/Nkv=2
    {"BF16_GQA",                 ACL_BF16, 4,2, 128,128, false,0,  true, ACL_UINT8,1, false,1, false,false, true, 1},
    // MLA: D=128, Dv=64 (kD >= vD)
    {"BF16_MLA",                 ACL_BF16, 1,1, 128,64,  false,0,  true, ACL_UINT8,1, false,1, false,false, true, 1},
    // 多 batch: actualSeqQLen={128,256}, actualSeqKvLen={128,256}
    {"BF16_attenMask_2batch",    ACL_BF16, 1,1, 128,128, false,0,  true, ACL_UINT8,1, false,1, false,false, true, 2},

    // ---- Group 5: softmaxOutLayout ----
    // softmaxOutLayout="" → 输出 NTD 排布, smMax/smSum shape=[N,T,8]
    {"BF16_softmaxNTD",          ACL_BF16, 1,1, 128,128, false,0,  true, ACL_UINT8,1, false,1, false,false, false,1},

    // ---- Group 6: 其他数据类型 ----
    // FP16
    {"FP16_attenMask",           ACL_FLOAT16, 1,1, 128,128, false,0, true, ACL_UINT8,1, false,1, false,false, true, 1},
    // FP16 + pse(pseType=1, PSE dtype=FP16, shape=[1,N2,S1,S2])
    {"FP16_attenMask_pse1",      ACL_FLOAT16, 1,1, 128,128, false,0, true, ACL_UINT8,1, true, 1, false,false, true, 1},
    // FP32
    {"FP32_attenMask",           ACL_FLOAT, 1,1, 128,128, false,0,   true, ACL_UINT8,1, false,1, false,false, true, 1},
    // FP32 + pse(pseType=2, PSE dtype=FP32, slope shape=[N2])
    {"FP32_attenMask_pse2",      ACL_FLOAT, 1,1, 128,128, false,0,   true, ACL_UINT8,1, true, 2, false,false, true, 1},

    // ---- Group 7: 组合 ----
    // attenMask + prefix + qStartIdx/kvStartIdx, sparseMode=6
    {"BF16_combo",               ACL_BF16, 1,1, 128,128, false,0,  true, ACL_UINT8,6, false,1, true, true,  true, 1},
  };

  int totalTests = sizeof(tests) / sizeof(tests[0]);
  for (int i = 0; i < totalTests; i++) {
    RunSingleTest(tests[i], stream);
  }

  LOG_PRINT("\n==============================\n");
  LOG_PRINT("Summary: Total=%d  Passed=%d  Failed=%d  Error=%d\n",
            totalTests, passedCount, failedCount, errorCount);
  LOG_PRINT("==============================\n");

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return (failedCount + errorCount) > 0 ? 1 : 0;
}
