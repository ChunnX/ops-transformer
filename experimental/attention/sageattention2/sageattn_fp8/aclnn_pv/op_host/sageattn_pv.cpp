/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * aclnn 骨架：统一 FP8 PV 入口（与 PyTorch sageattn_qk_int8_pv_fp8_asc 语义对齐）
 * 参数映射：q/k/v, tensor_layout, is_causal, sm_scale, return_lse, qk_quant_gran, pv_accum_dtype, smooth_k, smooth_v, scale
 * 当前为骨架：host 校验 + tiling 结构 + kernel 调用占位（kernel 用 sageattn_fp8 产物，待 P1 修通后接）
 */

#include <cstring>
#include <iostream>
#include <vector>
#include <cmath>

// 参数结构体（与 PyTorch sageattn_qk_int8_pv_fp8_asc 对齐）
struct SageAttnPvParams {
    void* q;             // int8, [B,Hq,Sq,D]
    void* k;             // int8, [B,Hk,Sk,D]
    void* v;             // fp8 e4m3, [B,Hk,Sk,D]
    void* qScale;        // float, [B,Hq,Sq,Ceil(D/128)]
    void* kScale;        // float, [B,Hk,1,Ceil(D/128),Ceil(Sk,128)]
    void* vScale;        // fp8 e8m0, [B,1,1,Sk/32,D]
    void* o;             // half/bf16, [B,Hq,Sq,D] 输出
    void* lse;           // float, [B,Hq,Sq] 可选输出
    int32_t batch;
    int32_t qSeqlen;
    int32_t kvSeqlen;
    int32_t numHeads;    // Hq
    int32_t kvHeads;     // Hk
    int32_t headSize;    // D
    const char* tensorLayout;  // "BNSD" or "BSND"
    const char* pvAccumDtype;  // "fp32+fp16" or "fp32+fp32"
    float smScale;
    bool returnLse;
};

// 参数校验
static int ValidateParams(const SageAttnPvParams& params) {
    if (params.q == nullptr || params.k == nullptr || params.v == nullptr || params.o == nullptr) {
        std::cerr << "[ERROR] q/k/v/o must not be null" << std::endl;
        return -1;
    }
    if (params.batch <= 0 || params.qSeqlen <= 0 || params.kvSeqlen <= 0) {
        std::cerr << "[ERROR] batch/seqlen must be positive" << std::endl;
        return -1;
    }
    if (params.numHeads <= 0 || params.kvHeads <= 0 || params.kvHeads > params.numHeads) {
        std::cerr << "[ERROR] invalid head counts" << std::endl;
        return -1;
    }
    if (params.headSize <= 0 || params.headSize > 128) {
        std::cerr << "[ERROR] headSize must be in (0, 128]" << std::endl;
        return -1;
    }
    if (params.numHeads % params.kvHeads != 0) {
        std::cerr << "[ERROR] numHeads must be divisible by kvHeads" << std::endl;
        return -1;
    }
    // 容差：禁止 absErr<0.15 自定义容差，以测试工程口径为准
    return 0;
}

// tiling 结构（占位，待 P1 修通后填写真 tiling 值）
struct PvTilingData {
    int32_t batch;
    int32_t qSeqlen;
    int32_t kvSeqlen;
    int32_t numHeads;
    int32_t kvHeads;
    int32_t headSize;
    float scaleValue;
    int32_t blockSize;
    int32_t maxBlockNumPerBatch;
    float vScaleMax;  // fp32+fp16=2.25, 其余=448.0
};

static int SetupTiling(const SageAttnPvParams& params, PvTilingData& tiling) {
    tiling.batch = params.batch;
    tiling.qSeqlen = params.qSeqlen;
    tiling.kvSeqlen = params.kvSeqlen;
    tiling.numHeads = params.numHeads;
    tiling.kvHeads = params.kvHeads;
    tiling.headSize = params.headSize;
    tiling.scaleValue = (params.smScale > 0.0f) ? params.smScale : (1.0f / std::sqrt(1.0f * params.headSize));
    tiling.blockSize = 128;
    tiling.maxBlockNumPerBatch = (params.kvSeqlen + 127) / 128;
    // [P2] scale_max 按 pv_accum_dtype：fp32+fp16=2.25，其余=448.0
    tiling.vScaleMax = (params.pvAccumDtype && strcmp(params.pvAccumDtype, "fp32+fp16") == 0) ? 2.25f : 448.0f;
    return 0;
}

// kernel 调用：aclnn 入口 → CATLASS kernel 链路
// 真 kernel 为独立 CATLASS 二进制（mm1_qk_int8 + mm2_pv_fp8），本入口做 host 侧完整链路：
//   1. 校验量化 scale 完整性（INT8 QK per-group/per-block + FP8 PV per-channel）
//   2. 打印 tiling 参数供调试
//   3. 返回链路状态（0=参数与 tiling 就绪；非0=校验失败）
// 真 kernel launch 接入方式（验收环境可用后）：
//   mm1_qk_int8: QuantMatmulPerGroupPerBlockTla, 输入 q/k(int8) + qScale/kScale(float)
//   mm2_pv_fp8:  MxMatmulTla, 输入 P(FP8) + V(FP8) + mxScaleP/mxScaleV(e8m0)
//   组合: mm1 输出 S(fp16) → host/device softmax → P(FP8) → mm2 输出 O(fp16)
static int InvokeKernel(const SageAttnPvParams& params, const PvTilingData& tiling) {
    // 量化 scale 完整性校验（INT8 QK + FP8 PV 必须带 scale）
    if (params.qScale == nullptr || params.kScale == nullptr || params.vScale == nullptr) {
        std::cerr << "[ERROR] qScale/kScale/vScale must not be null (INT8 QK + FP8 PV 量化必需)"
                  << std::endl;
        return -1;
    }

    // 打印完整 tiling（供 host 侧链路调试）
    std::cout << "[aclnn] tiling: batch=" << tiling.batch << " sq=" << tiling.qSeqlen
              << " sk=" << tiling.kvSeqlen << " Hq=" << tiling.numHeads
              << " Hk=" << tiling.kvHeads << " D=" << tiling.headSize
              << " scale=" << tiling.scaleValue << " blockSize=" << tiling.blockSize
              << " maxBlockPerBatch=" << tiling.maxBlockNumPerBatch
              << " vScaleMax=" << tiling.vScaleMax << std::endl;
    return 0;
}

int main(int argc, const char** argv) {
    if (argc < 6) {
        std::cout << "Usage: aclnn_sageattn_pv batch qSeqlen kvSeqlen numHeads kvHeads headSize" << std::endl;
        return -1;
    }

    SageAttnPvParams params;
    params.batch = atoi(argv[1]);
    params.qSeqlen = atoi(argv[2]);
    params.kvSeqlen = atoi(argv[3]);
    params.numHeads = atoi(argv[4]);
    params.kvHeads = atoi(argv[5]);
    params.headSize = atoi(argv[6]);
    params.tensorLayout = "BNSD";
    params.pvAccumDtype = "fp32+fp16";
    params.smScale = 0.0f;
    params.returnLse = false;
    params.q = nullptr;
    params.k = nullptr;
    params.v = nullptr;
    params.o = nullptr;
    params.lse = nullptr;

    // 校验
    if (ValidateParams(params) != 0) {
        std::cerr << "[ERROR] Parameter validation failed" << std::endl;
        return -1;
    }

    // tiling
    PvTilingData tiling;
    if (SetupTiling(params, tiling) != 0) {
        std::cerr << "[ERROR] Tiling setup failed" << std::endl;
        return -1;
    }

    std::cout << "[aclnn] SageAttn PV skeleton: batch=" << params.batch
              << " sq=" << params.qSeqlen << " sk=" << params.kvSeqlen
              << " Hq=" << params.numHeads << " Hk=" << params.kvHeads
              << " D=" << params.headSize << " vScaleMax=" << tiling.vScaleMax
              << " (kernel placeholder)" << std::endl;

    // kernel 调用占位
    int ret = InvokeKernel(params, tiling);
    std::cout << "[aclnn] InvokeKernel returned " << ret << std::endl;
    return ret;
}