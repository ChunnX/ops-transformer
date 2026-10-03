#ifndef SAGE_ATTN_SIM_CASE_COMMON_HPP
#define SAGE_ATTN_SIM_CASE_COMMON_HPP

/**
 * \file sim_case_common.hpp
 * \brief SageAttention2 camodel 仿真 harness 的共享用例定义（B2 门禁数值比对）。
 *
 * 唯一定义点：用例形状/模式参数、输入生成算法（固定种子 std::mt19937 + 整数域
 * 确定性映射，杜绝跨编译器 uniform_real_distribution 实现差异）、smooth_k 的
 * key_mean 预计算（与 twin_reference.h RunSequential 内置口径一致：double 累加
 * 求均值后舍入回 fp16）、bin 文件读写辅助。
 *
 * 被两个编译域共享：
 * - sim_reference.cpp（容器内 BiSheng 宿主模式编译，产出输入 bin + tiling bin + golden bin）；
 * - sim_main.cpp（bisheng --run-mode=sim 混编，读同一批 bin 驱动设备核）。
 * 两端输入位级同源是数值比对成立的前提，因此设备侧一律从 bin 读输入，不在
 * 设备 harness 内重新生成。
 */

#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "../../sageattn_fp8/ascendc/tiling.h"

#include "../../aclnn/sage_attn_proto.h"
#include "../../aclnn/sage_attn_host_tiling.h"

// aclnn 错误码补齐：sim 编译域带 -I$ASC/include，本容器 CANN 9.2.0-beta.1 的
// aclnn/aclnn_base.h 有 aclnnStatus 类型而无 ACLNN_* 常量（无 opdev/op_errno.h），
// host_tiling.h 的本地桩分支被 SAGE_HAS_CANN_ACLNN_BASE 关闭，常量处于未定义态
// （ComputeHostTiling 返回值与 BuildTiling 判定直接引用）。此处按桩分支同值补
// 定义——须置于 host_tiling.h 之后（守卫宏由其定义），仅覆盖本 harness 编译域，
// 不动仓内既有文件。
#if defined(SAGE_HAS_CANN_ACLNN_BASE)
#ifndef ACLNN_SUCCESS
constexpr aclnnStatus ACLNN_SUCCESS = 0;
constexpr aclnnStatus ACLNN_ERR_PARAM_NULLPTR = 161001;
constexpr aclnnStatus ACLNN_ERR_PARAM_INVALID = 161002;
#endif
#endif

#include "../../quant/quant_common.h"

// 注意：twin_reference.h（宿主孪生真值）仅由 sim_reference.cpp include——
// 它携带 std::vector 载体且数值实现在宿主分支，不得混入 bisheng 设备编译域。

namespace sage_attn_sim {

// -------------------------------------------------------------------------
// 用例参数（任务书冻结小 case）：B=1、Hq=Hkv=1、Sq=Sk=128、D=64、fp16、
// per_thread、pv_accum=fp32+fp16、非因果、smooth_k=1、sm_scale=1/8。
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
constexpr uint32_t kSmoothK = 1;
constexpr uint32_t kDtypeMode = 1;  // 1 = fp16
constexpr float kSmScale = 0.125f;  // 1/8
constexpr uint32_t kSeed = 20261003u;

// bin 文件名（相对工作目录，由 run 脚本约定 cwd）
constexpr const char *kQBin = "input_q_fp16.bin";
constexpr const char *kKBin = "input_k_fp16.bin";
constexpr const char *kVBin = "input_v_fp16.bin";
constexpr const char *kKmBin = "input_key_mean_f32.bin";
constexpr const char *kLseCorrBin = "input_lse_correction_f32.bin";
constexpr const char *kTilingBin = "tiling_160.bin";
constexpr const char *kGoldenOBin = "golden_o_fp16.bin";
constexpr const char *kGoldenLseBin = "golden_lse_f32.bin";
constexpr const char *kDeviceOBin = "device_o_fp16.bin";
constexpr const char *kDeviceLseBin = "device_lse_f32.bin";

// NHD 布局元素偏移：[b, s, h, d]
inline size_t NhdOffset(uint32_t b, uint32_t s, uint32_t h, uint32_t d,
                        uint32_t /*dim_b*/, uint32_t dim_s, uint32_t dim_h, uint32_t dim_d) {
    return ((static_cast<size_t>(b) * dim_s + s) * dim_h + h) * dim_d + d;
}

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
 * smooth_k 的 key_mean 预计算（单点，设备与 golden 共用同一份）：
 * 按 K 逐 (b, h, d) 通道做 double 累加均值，舍入回 fp16 后以 float 返回，
 * 与 twin_reference.h 步骤 2 的内置口径逐位一致。
 */
inline void ComputeKeyMean(const std::vector<uint16_t> &k_bits, std::vector<float> &km) {
    km.assign(static_cast<size_t>(kB) * kHkv * 128u, 0.0f);  // d_pad=128 上界，尾段 0
    for (uint32_t bi = 0; bi < kB; ++bi) {
        for (uint32_t hi = 0; hi < kHkv; ++hi) {
            for (uint32_t di = 0; di < kDOg; ++di) {
                double sum = 0.0;
                for (uint32_t si = 0; si < kSk; ++si) {
                    const size_t off = NhdOffset(bi, si, hi, di, kB, kSk, kHkv, kDOg);
                    sum += sage_attn::quant::Fp16ToFp32(k_bits[off]);
                }
                const float mean = static_cast<float>(sum / kSk);
                km[(static_cast<size_t>(bi) * kHkv + hi) * 128u + di] =
                    sage_attn::quant::Fp16ToFp32(sage_attn::quant::Fp32ToFp16(mean));
            }
        }
    }
}

/**
 * smooth_k 的行级 lse_correction 预计算（[b*hq*sq] fp32，开源契约 q·km_broadcast^T）：
 * 与 key_mean 同为"上层算好传入"的外部张量——设备核经 GM 入参 5 直读、孪生经
 * TwinIo.lse_correction 消费，本端产出保证两侧同源（gqa=1，km 广播为恒等）。
 */
inline void ComputeLseCorrection(const std::vector<uint16_t> &q_bits, const std::vector<float> &km,
                                 std::vector<float> &corr) {
    corr.assign(static_cast<size_t>(kB) * kHq * kSq, 0.0f);
    for (uint32_t bi = 0; bi < kB; ++bi) {
        for (uint32_t hi = 0; hi < kHq; ++hi) {
            const float *km_head = km.data() + (static_cast<size_t>(bi) * kHkv + hi) * 128u;
            for (uint32_t si = 0; si < kSq; ++si) {
                float acc = 0.0f;
                for (uint32_t di = 0; di < kDOg; ++di) {
                    const size_t off = NhdOffset(bi, si, hi, di, kB, kSq, kHq, kDOg);
                    acc += sage_attn::quant::Fp16ToFp32(q_bits[off]) * km_head[di];
                }
                corr[(static_cast<size_t>(bi) * kHq + hi) * kSq + si] = acc;
            }
        }
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
        kSmoothK != 0,                   // hasKeyMean
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
