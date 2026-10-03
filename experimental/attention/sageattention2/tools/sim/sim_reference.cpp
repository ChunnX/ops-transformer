/**
 * \file sim_reference.cpp
 * \brief B2 门禁 golden 参考程序（容器内 BiSheng 宿主模式编译，SAGE_ATTN_DEVICE_COMPILE=0 路径）。
 *
 * 职责：
 * 1. 按 sim_case_common.hpp 冻结用例生成 fp16 Q/K/V 与 key_mean，写 bin；
 * 2. 经 ComputeHostTiling 单点计算 TilingData，写 160 字节 tiling bin；
 * 3. twin_reference.h RunSequential（宿主孪生唯一真值）计算 O/LSE，写 golden bin。
 *
 * 产物清单（工作目录）：input_q_fp16.bin / input_k_fp16.bin / input_v_fp16.bin /
 * input_key_mean_f32.bin / tiling_160.bin / golden_o_fp16.bin / golden_lse_f32.bin
 *
 * 编译（容器内，见 run_sim_e2e.sh 步骤 1）：
 *   bisheng -std=c++17 -O2 -DSAGE_ATTN_DEVICE_COMPILE=0
 *       -I sageattn_fp8/ascendc -I aclnn -I quant -I tools/sim
 *       sim_reference.cpp quant/quant_int8.cpp quant/quant_fp8.cpp
 */
#include <cstdio>
#include <vector>

#include "sim_case_common.hpp"

// 宿主孪生唯一真值（仅本宿主 TU 引入，见 sim_case_common.hpp 的注记）
#include "../../sageattn_fp8/ascendc/twin_reference.h"

int main(int argc, char **argv) {
    // 工作目录约定：argv[1]（缺省 "."），bin 全部落于此
    const std::string dir = (argc > 1) ? argv[1] : std::string(".");
    const auto P = [&dir](const char *name) { return dir + "/" + name; };
    using namespace sage_attn_sim;

    // 1. 输入生成（确定性，与设备侧共享同一份 bin）
    std::vector<uint16_t> q_bits, k_bits, v_bits;
    GenerateFp16Tensor(kSeed + 1u, q_bits);
    GenerateFp16Tensor(kSeed + 2u, k_bits);
    GenerateFp16Tensor(kSeed + 3u, v_bits);
    std::vector<float> km, corr;
    ComputeKeyMean(k_bits, km);
    ComputeLseCorrection(q_bits, km, corr);

    if (!WriteBin(P(kQBin), q_bits) || !WriteBin(P(kKBin), k_bits) || !WriteBin(P(kVBin), v_bits) ||
        !WriteBin(P(kKmBin), km) || !WriteBin(P(kLseCorrBin), corr)) {
        std::fprintf(stderr, "[FAIL] 写输入 bin 失败（目录 %s）\n", dir.c_str());
        return 1;
    }

    // 2. Host Tiling（单点）
    sage_attn::SageAttention2TilingData tiling{};
    uint64_t ws_bytes = 0;
    if (!BuildTiling(tiling, ws_bytes)) {
        std::fprintf(stderr, "[FAIL] ComputeHostTiling 拒绝用例\n");
        return 1;
    }
    if (!WriteBinBytes(P(kTilingBin), &tiling, sizeof(tiling))) {
        std::fprintf(stderr, "[FAIL] 写 tiling bin 失败\n");
        return 1;
    }
    std::printf("[INFO] tiling: m_per_core=%u block_n=%u k_step=%u core_num_used=%u ws_bytes=%llu\n",
                tiling.m_per_core, tiling.block_n, tiling.k_step, tiling.core_num_used,
                static_cast<unsigned long long>(ws_bytes));

    // 3. golden：宿主孪生全流水（smooth_k 的 LSE 校正项经 io.lse_correction 外部传入，
    //    与设备核 GM 入参同源同值）
    std::vector<uint16_t> out_bits(static_cast<size_t>(kB) * kSq * kHq * kDOg, 0);
    std::vector<float> lse(static_cast<size_t>(kB) * kHq * kSq, 0.0f);

    sage_attn::twin::TwinIo io{};
    io.q = q_bits.data();
    io.k = k_bits.data();
    io.v = v_bits.data();
    io.out = out_bits.data();
    io.lse = lse.data();
    io.key_mean = km.data();
    io.lse_correction = corr.data();
    io.b = kB;
    io.hq = kHq;
    io.hkv = kHkv;
    io.sq = kSq;
    io.sk = kSk;
    io.d_og = kDOg;
    io.tensor_layout = kLayout;
    io.qk_quant_gran = kGran;
    io.pv_accum_dtype = kPvAccum;
    io.is_causal = kIsCausal;
    io.smooth_k = kSmoothK;
    io.smooth_v = 0;  // 冻结用例为 fp32+fp16 档，smooth_v 不生效
    io.dtype_mode = kDtypeMode;
    io.sm_scale = kSmScale;

    const int rc = sage_attn::twin::RunSequential(io, nullptr);
    if (rc != 0) {
        std::fprintf(stderr, "[FAIL] RunSequential 返回 %d\n", rc);
        return 1;
    }
    if (!WriteBin(P(kGoldenOBin), out_bits) || !WriteBin(P(kGoldenLseBin), lse)) {
        std::fprintf(stderr, "[FAIL] 写 golden bin 失败\n");
        return 1;
    }
    std::printf("[PASS] golden 参考产物齐备（O=%zu half, LSE=%zu float）\n",
                out_bits.size(), lse.size());
    return 0;
}
