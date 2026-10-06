#ifndef SAGE_ATTN_SIM_CASE_COMMON_HPP
#define SAGE_ATTN_SIM_CASE_COMMON_HPP

/**
 * \file sim_case_common.hpp
 * \brief SageAttention2 camodel 仿真 harness 的共享用例定义（B2 门禁数值比对）。
 *
 * 唯一定义点：用例形状/模式参数、输入生成算法（固定种子 std::mt19937 + 整数域
 * 确定性映射，杜绝跨编译器 uniform_real_distribution 实现差异）、bin 文件读写辅助。
 *
 * smooth_k 生效门控（op_host 实况同式）：标准 7 参收敛后入口无 key_mean/
 * lse_correction 独立张量，op_host/sage_attention2_tiling.cpp 以 has_key_mean=false
 * 门控——tiling.smooth_k = smoothK && has_key_mean = 0，核内 K 不减均值、LSE 无
 * 校正项；孪生 golden 同口径（io.smooth_k=0），设备与参考逐位可比。
 *
 * 被两个编译域共享：
 * - sim_reference.cpp（容器内 BiSheng 宿主模式编译，产出输入 bin + tiling bin + golden bin）；
 * - sim_main.cpp（bisheng --run-mode=sim 混编，读同一批 bin 驱动设备核）。
 * 两端输入位级同源是数值比对成立的前提，因此设备侧一律从 bin 读输入，不在
 * 设备 harness 内重新生成。
 */

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "../../sageattn_fp8/ascendc/tiling.h"

#include "../../aclnn/sage_attn_proto.h"
#include "../../aclnn/sage_attn_host_tiling.h"
#include "../../quant/quant_common.h"

// 注意：twin_reference.h（宿主孪生真值）仅在宿主编译域被 include：sim_reference.cpp、
// 自身实现 twin_reference.cpp，以及 op_kernel/sageattention2.cpp 的宿主分支
// （!SAGE_ATTN_DEVICE_COMPILE 门控）。它携带 std::vector 载体且数值实现在宿主分支，
// 不得混入 bisheng 设备编译域。

namespace sage_attn_sim {

// -------------------------------------------------------------------------
// 用例参数（任务书冻结小 case）：B=1、Hq=Hkv=1、Sq=Sk=128、D=64、fp16、
// per_thread、pv_accum=fp32+fp16、非因果、sm_scale=1/8；smooth_k 属性冻结为 1，
// 但标准 7 参收敛后无 key_mean 供给，生效口径恒为关（见文件头门控注记）。
// -------------------------------------------------------------------------
constexpr uint32_t kB = 1;
constexpr uint32_t kHq = 1;
constexpr uint32_t kHkv = 1;
constexpr uint32_t kSq = 128;
constexpr uint32_t kSk = 128;
constexpr uint32_t kDOg = 64;
constexpr uint32_t kLayout = sage_attn::LAYOUT_NHD;           // 0 = NHD
constexpr uint32_t kGran = sage_attn::QUANT_GRAN_PER_THREAD;  // 3 = per_thread
constexpr uint32_t kPvAccum = sage_attn::PV_ACCUM_FP32_FP16;  // 2 = fp32+fp16
constexpr uint32_t kIsCausal = 0;
constexpr uint32_t kSmoothK = 1;    // smoothK 属性值（任务书冻结；生效门控见 BuildTiling）
constexpr uint32_t kDtypeMode = 1;  // 1 = fp16
constexpr float kSmScale = 0.125f;  // 1/8
constexpr uint32_t kSeed = 20261003u;

// bin 文件名（相对工作目录，由 run 脚本约定 cwd）
constexpr const char *kQBin = "input_q_fp16.bin";
constexpr const char *kKBin = "input_k_fp16.bin";
constexpr const char *kVBin = "input_v_fp16.bin";
constexpr const char *kTilingBin = "tiling_160.bin";
constexpr const char *kGoldenOBin = "golden_o_fp16.bin";
constexpr const char *kGoldenLseBin = "golden_lse_f32.bin";
constexpr const char *kDeviceOBin = "device_o_fp16.bin";
constexpr const char *kDeviceLseBin = "device_lse_f32.bin";

/**
 * 确定性输入生成：mt19937 原始 32 位 → [0,2^24) 整数 → [-2,2) float。
 * 全程整数与单次乘法，IEEE-754 下跨编译器位级一致（不使用
 * uniform_real_distribution——其算法是实现定义的）。
 */
inline void GenerateFp16Tensor(uint32_t seed, std::vector<uint16_t> &out_bits) {
    std::mt19937 rng(seed);
    out_bits.resize(static_cast<size_t>(kB) * kSq * kHq * kDOg);
    for (size_t i = 0; i < out_bits.size(); ++i) {
        const uint32_t raw = rng();
        const int32_t m = static_cast<int32_t>(raw >> 8) - (1 << 23);  // [-2^23, 2^23)
        const float v = static_cast<float>(m) * (4.0f / static_cast<float>(1 << 24));
        out_bits[i] = sage_attn::quant::Fp32ToFp16(v);
    }
}

/**
 * Host Tiling 单点计算（复用 aclnn/sage_attn_host_tiling.h 的 ComputeHostTiling，
 * 与 op_host/aclnn 两形态共享同一闭式实现），并冻结语义对齐本 harness 的调用面。
 */
inline bool BuildTiling(sage_attn::SageAttention2TilingData &tiling, uint64_t &ws_bytes) {
    const aclnnStatus st = sage_attn::ComputeHostTiling(
        kB, kSq, kSk, kHq, kHkv, kDtypeMode, kDOg, static_cast<double>(kSmScale),
        kGran, kPvAccum, kIsCausal != 0, kLayout,
        kSmoothK != 0, false,            // smooth_v=false（非 fp32 档被 host 拒绝）
        true,                            // hasLse
        tiling, ws_bytes);
    return st == ACLNN_SUCCESS;
}

// -------------------------------------------------------------------------
// bin 读写辅助（小端原样字节，无元数据头；长度由读方按用例常量解释）
// -------------------------------------------------------------------------
template <typename T>
inline bool WriteBin(const std::string &path, const std::vector<T> &data) {
    FILE *f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) return false;
    const size_t wrote = std::fwrite(data.data(), sizeof(T), data.size(), f);
    std::fclose(f);
    return wrote == data.size();
}

inline bool WriteBinBytes(const std::string &path, const void *data, size_t bytes) {
    FILE *f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) return false;
    const size_t wrote = std::fwrite(data, 1, bytes, f);
    std::fclose(f);
    return wrote == bytes;
}

template <typename T>
inline bool ReadBin(const std::string &path, std::vector<T> &data, size_t expect_count) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return false;
    data.resize(expect_count);
    const size_t got = std::fread(data.data(), sizeof(T), expect_count, f);
    std::fclose(f);
    return got == expect_count;
}

}  // namespace sage_attn_sim

#endif  // SAGE_ATTN_SIM_CASE_COMMON_HPP
