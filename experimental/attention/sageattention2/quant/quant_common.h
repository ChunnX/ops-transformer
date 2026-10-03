#ifndef SAGE_ATTN_QUANT_COMMON_H
#define SAGE_ATTN_QUANT_COMMON_H

/**
 * \file quant_common.h
 * \brief INT8 (Q/K) 与 FP8 (V) 两条量化路径共享的宿主机参考实现基础件：
 *        1. __aicore__ 宏桩（本地编译 / CCE 测试环境支持）
 *        2. FP16 / BF16 位级转换 (RNE 舍入)
 *        3. 输入读取抽象 Reader (float / fp16 / bf16 减法语义对齐)
 *        唯一来源：quant_int8.cpp 与 quant_fp8.cpp 均从此包含，禁止再各自复制。
 */

#include <cstdint>
#include <cstring>
#include <cmath>

#include "../sageattn_fp8/ascendc/tiling.h"


#ifndef __host__
#define __host__
#endif

namespace sage_attn {
namespace quant {

/**
 * 浮点格式转换函数 (FP16 / BF16 到 FP32 及逆向转换)
 */
__aicore__ __host__ inline float Fp16ToFp32(uint16_t h) noexcept {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    if (exp == 0) {
        if (mant == 0) {
            uint32_t u = sign << 31;
            float f; std::memcpy(&f, &u, sizeof(float)); return f;
        }
        int32_t e = -14;
        uint32_t m = mant;
        while ((m & 0x400) == 0) {
            m <<= 1;
            e -= 1;
        }
        m &= ~0x400;
        uint32_t f_exp = static_cast<uint32_t>(e + 127);
        uint32_t u = (sign << 31) | (f_exp << 23) | (m << 13);
        float f; std::memcpy(&f, &u, sizeof(float)); return f;
    } else if (exp == 31) {
        uint32_t u = (sign << 31) | 0x7f800000 | (mant << 13);
        float f; std::memcpy(&f, &u, sizeof(float)); return f;
    }
    uint32_t f_exp = exp - 15 + 127;
    uint32_t u = (sign << 31) | (f_exp << 23) | (mant << 13);
    float f; std::memcpy(&f, &u, sizeof(float)); return f;
}

__aicore__ __host__ inline uint16_t Fp32ToFp16(float f) noexcept {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(float));
    uint32_t sign = (u >> 31) & 0x1;
    uint32_t exp = (u >> 23) & 0xff;
    uint32_t mant = u & 0x7fffff;
    if (exp == 0) return static_cast<uint16_t>(sign << 15);
    int32_t new_exp = static_cast<int32_t>(exp) - 127 + 15;
    if (new_exp >= 31) return static_cast<uint16_t>((sign << 15) | 0x7c00);
    if (new_exp <= 0) {
        // 次正规域：统一 uint32 移位并对 shift>24 直接舍入到零
        const uint32_t shift = static_cast<uint32_t>(1 - new_exp) + 13u;
        if (shift > 24u) return static_cast<uint16_t>(sign << 15);
        mant |= 0x800000u;
        const uint32_t round_bit = 1u << (shift - 1u);
        uint32_t quotient = mant >> shift;
        const uint32_t remainder = mant & ((1u << shift) - 1u);
        if (remainder > round_bit || (remainder == round_bit && (quotient & 1))) quotient++;
        return static_cast<uint16_t>((sign << 15) | quotient);
    }
    uint32_t quotient = mant >> 13;
    uint32_t remainder = mant & 0x1fff;
    if (remainder > 0x1000 || (remainder == 0x1000 && (quotient & 1))) {
        quotient++;
        if (quotient == 0x400) {
            quotient = 0;
            new_exp++;
            if (new_exp >= 31) return static_cast<uint16_t>((sign << 15) | 0x7c00);
        }
    }
    return static_cast<uint16_t>((sign << 15) | (new_exp << 10) | quotient);
}

__aicore__ __host__ inline float Bf16ToFp32(uint16_t b) noexcept {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f; std::memcpy(&f, &u, sizeof(float)); return f;
}

__aicore__ __host__ inline uint16_t Fp32ToBf16(float f) noexcept {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(float));
    uint32_t lsb = (u >> 16) & 1;
    uint32_t rounding_bias = 0x7fff + lsb;
    u += rounding_bias;
    return static_cast<uint16_t>(u >> 16);
}

/**
 * INT8 舍入原语（严格区分 half-away-from-zero 与 RNE，策略八 / task.md 2.2 策略四）
 * - per_thread 路径: half-away-from-zero (floor(|x|+0.5)*sign，饱和至 [-127, 127])，
 *   对齐 GPU quant_per_thread.py 的 +0.5*sign(x)；
 * - per_warp 路径: Round-to-Nearest-Even (对齐 cvt.rni.sat.s8，饱和至 [-128, 127])。
 * 唯一定义纪律（2026-10-03）：本组原语同时服务宿主参考（quant_int8/quant_fp8/
 * twin_reference）与设备核（op_kernel/sageattention2.cpp 的 sage_attention2 入口），实现不依赖 std::floor/std::nearbyint
 *（bisheng dav-3510 设备侧 <cmath> 无这些符号，链接期 undefined symbol），以显式
 * 整数运算表达同一舍入语义；量化域 |x| 有界（|x/scale| 至多百余量级），
 * float 24 位尾数内整数 cast 精确，与 std 版本逐位一致（tests/test_quant.py 锁定）。
 */
__aicore__ __host__ inline int32_t FloorToInt(float v) noexcept {
    int32_t t = static_cast<int32_t>(v);
    return (v < static_cast<float>(t)) ? (t - 1) : t;
}

__aicore__ __host__ inline int8_t RoundHalfAwayFromZero(float val) noexcept {
    const bool neg = (val < 0.0f);
    const float abs_val = neg ? -val : val;
    int32_t rounded = FloorToInt(abs_val + 0.5f);
    // 饱和界 INT8_QUANT_MAX 唯一定义位于 tiling.h
    if (rounded > static_cast<int32_t>(INT8_QUANT_MAX)) rounded = static_cast<int32_t>(INT8_QUANT_MAX);
    return static_cast<int8_t>(neg ? -rounded : rounded);
}

__aicore__ __host__ inline int8_t RoundNearestEven(float val) noexcept {
    const int32_t lower = FloorToInt(val);
    const float frac = val - static_cast<float>(lower);
    int32_t result;
    if (frac > 0.5f) {
        result = lower + 1;
    } else if (frac < 0.5f) {
        result = lower;
    } else {
        result = (lower & 1) ? (lower + 1) : lower;  // 半数点 ties-to-even
    }
    if (result > static_cast<int32_t>(INT8_QUANT_MAX)) result = static_cast<int32_t>(INT8_QUANT_MAX);
    if (result < -128) result = -128; // int8 下界（INT8_MIN，tiling.h 无对应常量）
    return static_cast<int8_t>(result);
}

/**
 * FP8 e4m3fn 常量：1 位符号、4 位指数（bias 7）、3 位尾数；
 * 最大有限值 448.0f (0x7E)，无无穷表示，0x7F/0xFF 为 NaN。
 * FP8_E4M3_MAX 的唯一定义位于 tiling.h，此处经 using 引入。
 */
using sage_attn::FP8_E4M3_MAX;

/**
 * IEEE 754 float32 到 FP8 e4m3fn 的精确位级 RNE + Satfinite 转换
 * （与 GPU cvt.rn.satfinite.e4m3 逐位一致，全仓唯一定义点，禁止外部别名）
 */
__aicore__ __host__ inline uint8_t FloatToE4m3fn(float f) noexcept {
    if (f != f) return 0x7f;  // NaN（设备侧无 std::isnan，位比较等价）
    if (f > FP8_E4M3_MAX) f = FP8_E4M3_MAX;
    if (f < -FP8_E4M3_MAX) f = -FP8_E4M3_MAX;

    uint32_t u;
    std::memcpy(&u, &f, sizeof(float));

    const uint32_t sign = (u >> 31) & 0x1;
    const uint32_t exp = (u >> 23) & 0xff;
    const uint32_t mant = u & 0x7fffff;

    if (exp == 0 && mant == 0) {
        return static_cast<uint8_t>(sign << 7);
    }

    int32_t target_exp = static_cast<int32_t>(exp) - 127 + 7;

    if (target_exp <= 0) {
        // 非规格化数 (Subnormal) 处理
        const int32_t shift = 1 - target_exp + (23 - 3);
        const uint32_t full_mant = mant | 0x800000;
        if (shift > 31) return static_cast<uint8_t>(sign << 7);

        uint32_t quotient = full_mant >> shift;
        const uint32_t remainder = full_mant & ((1U << shift) - 1U);
        const uint32_t half = 1U << (shift - 1);

        // 偶数舍入 (RNE)
        if (remainder > half || (remainder == half && (quotient & 1))) {
            quotient++;
        }
        if (quotient > 7) {
            return static_cast<uint8_t>((sign << 7) | (1 << 3) | (quotient - 8));
        }
        return static_cast<uint8_t>((sign << 7) | quotient);
    }

    if (target_exp >= 15) {
        // 饱和截断到 e4m3 最大正常数 448.0f (0x7E)
        uint32_t quotient = mant >> 20;
        const uint32_t remainder = mant & 0xfffff;
        if (remainder > 0x80000 || (remainder == 0x80000 && (quotient & 1))) {
            quotient++;
        }
        if (target_exp > 15 || quotient >= 7) {
            return static_cast<uint8_t>((sign << 7) | (15 << 3) | 6);
        }
        return static_cast<uint8_t>((sign << 7) | (15 << 3) | quotient);
    }

    // 规格化数 (Normal)
    uint32_t quotient = mant >> 20;
    const uint32_t remainder = mant & 0xfffff;
    if (remainder > 0x80000 || (remainder == 0x80000 && (quotient & 1))) {
        quotient++;
        if (quotient == 8) {
            quotient = 0;
            target_exp++;
            if (target_exp == 15 && quotient > 6) quotient = 6;
        }
    }
    return static_cast<uint8_t>((sign << 7) | (target_exp << 3) | quotient);
}

/**
 * FP8 e4m3fn 到 float32 的精确解码（PV 孪生反量化与单测使用）
 */
__aicore__ __host__ inline float E4m3fnToFloat(uint8_t v) noexcept {
    const uint32_t sign = (v >> 7) & 0x1;
    const uint32_t exp = (v >> 3) & 0xf;
    const uint32_t mant = v & 0x7;
    float f;
    if (exp == 0) {
        if (mant == 0) {
            const uint32_t u = sign << 31;
            std::memcpy(&f, &u, sizeof(float));
            return f;
        }
        // 非规格化：无隐含前导 1，真值 = mant × 2^-9（小整数 × 2 的幂，构造精确）
        const float mag = static_cast<float>(mant) * 0.001953125f;
        return (sign != 0) ? -mag : mag;
    }
    if (exp == 15 && mant == 7) {
        // NaN
        const uint32_t u = (sign << 31) | 0x7f800000u | (1u << 22);
        std::memcpy(&f, &u, sizeof(float));
        return f;
    }
    const uint32_t u = (sign << 31) | ((exp - 7 + 127) << 23) | (mant << 20);
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

/**
 * per_thread / per_warp 分组索引原语（开源算法粒度，单一定义；
 * 组边界语义对齐 git/SageAttention triton/quant_per_thread.py 与 CUDA per_warp kernel）
 * Q：128 token 块；per_thread 每块 32 组（4 子块 × 8 个步长 8 交织组），per_warp 每块 4 组。
 * K：64 token 块；per_thread 每块 4 组（每组合 8 对相邻 token），per_warp 每块 1 组。
 */
constexpr uint32_t Q_BLK = 128;
constexpr uint32_t Q_SUB = 32;
constexpr uint32_t K_BLK = 64;

__aicore__ __host__ constexpr inline uint32_t QBlockNum(uint32_t seq_len) noexcept {
    return (seq_len + Q_BLK - 1) / Q_BLK;
}

__aicore__ __host__ constexpr inline uint32_t KBlockNum(uint32_t seq_len) noexcept {
    return (seq_len + K_BLK - 1) / K_BLK;
}

/**
 * Q 的 scale 组数（每 head）：per_thread 为 blk×32，per_warp 为 blk×4
 */
__aicore__ __host__ constexpr inline uint32_t QScaleCount(uint32_t seq_len, bool per_thread) noexcept {
    return QBlockNum(seq_len) * (per_thread ? 32u : 4u);
}

/**
 * K 的 scale 组数（每 head）：per_thread 为 blk×4，per_warp 为 blk×1
 */
__aicore__ __host__ constexpr inline uint32_t KScaleCount(uint32_t seq_len, bool per_thread) noexcept {
    return KBlockNum(seq_len) * (per_thread ? 4u : 1u);
}

/**
 * token 在其 head 的 Q scale 数组中的线性组号
 */
    // 设备分支 Phase-B 组归并调用（DEFECT-F-1 随动）：不带 __host__——bisheng 对
    // 仅标 __host__ 的函数不在设备 pass 发射函数体，链接期 undefined（归并/定档
    // 为纯整数算术，host/device 双 pass 同结果）
__aicore__ inline uint32_t QScaleIndex(uint32_t token, bool per_thread) noexcept {
    const uint32_t blk = token / Q_BLK;
    const uint32_t in_blk = token % Q_BLK;
    if (per_thread) {
        return blk * 32u + (in_blk / Q_SUB) * 8u + (in_blk % 8u);
    }
    return blk * 4u + in_blk / Q_SUB;
}

/**
 * QK INT8 scale 定档公式唯一定义点（kernel 设备/宿主分支与 quant 参考实现共用）：
 * - per_thread：amax/127 + 1e-7（与 RoundHalfAwayFromZero 舍入路径配对）；
 * - per_warp：clamp(amax, 1e-7)/127（与 RoundNearestEven 舍入路径配对）。
 * 禁止在调用方复制本公式——一处修舍入/兜底，处处生效。
 */
__aicore__ __host__ inline float FinalizeQkScale(float amax, bool per_thread) noexcept {
    return per_thread ? (amax / INT8_QUANT_MAX + SCALE_EPSILON)
                      : ((amax < SCALE_EPSILON ? SCALE_EPSILON : amax) / INT8_QUANT_MAX);
}

// 槽区步长容量域锁定（与 tiling.h kQScaleSlotStride/kKScaleSlotStride 配对）
static_assert(QScaleCount(16384, true) <= kQScaleSlotStride,
              "Q scale slot stride must cover per_thread sq=16384 (P-07 domain)");
static_assert(KScaleCount(32768, true) <= kKScaleSlotStride,
              "K scale slot stride must cover per_thread sk=32768");

/**
 * token 在其 head 的 K scale 数组中的线性组号
 * per_thread 组 g 含 8 对相邻 token：token = blk*64 + p*8 + 2g + e (p∈[0,8), e∈{0,1})，
 * 即组号由序列内 8 交织相的相邻对决定（对齐 quant_per_thread.py 的 (8,4,2) 分解）
 */
__aicore__ inline uint32_t KScaleIndex(uint32_t token, bool per_thread) noexcept {
    const uint32_t blk = token / K_BLK;
    const uint32_t in_blk = token % K_BLK;
    if (per_thread) {
        return blk * 4u + (in_blk % 8u) / 2u;
    }
    return blk;
}

/**
 * V 的 16-token 段内重排模式（对齐 GPU MMA ldmatrix / golden v_fp8 位型）：
 * 物理槽 p 存放 token V_PERM_PATTERN[p]；token j 的物理槽为 V_PERM_INV_PATTERN[j%16]
 */
constexpr uint32_t V_PERM_PATTERN[16] = {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15};
constexpr uint32_t V_PERM_INV_PATTERN[16] = {0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15};
static_assert(V_PERM_PATTERN[V_PERM_INV_PATTERN[3]] == 3, "V perm inverse must be consistent");

/**
 * token 在 16-token 段内的物理 V 槽号（V_PERM_INV_PATTERN 寻址唯一定义点）：
 * golden v_fp8 排布、twin v_deq 读取与 kernel V 量化写槽共用，禁止再展开公式。
 */
__aicore__ __host__ inline uint32_t VSlotOfToken(uint32_t token_in_block) noexcept {
    return (token_in_block / 16u) * 16u + V_PERM_INV_PATTERN[token_in_block % 16u];
}

/**
 * 输入数据读取抽象与减法模式对齐
 */
// Reader 方法与上方转换函数同为 __aicore__（2026-10-03 随 quant 层设备化标注补齐）：
// bisheng 对 TU 内未引用的 inline 函数体也做双 pass 语义检查，host 语境方法调用
// __aicore__ 函数会报 "call to __aicore__ function from __host__ function"。
// 宿主 TU（BiSheng 宿主模式，__aicore__ 展开为空）不受影响。
template <typename T>
struct ElementReader {
    static inline __aicore__ float Read(const T* ptr, size_t idx) noexcept {
        return static_cast<float>(ptr[idx]);
    }
    static inline __aicore__ float Sub(float a, float b) noexcept {
        return a - b;
    }
};

struct HalfReader {
    static inline __aicore__ float Read(const uint16_t* ptr, size_t idx) noexcept {
        return Fp16ToFp32(ptr[idx]);
    }
    static inline __aicore__ float Sub(float a, float b) noexcept {
        return Fp16ToFp32(Fp32ToFp16(a - b));
    }
};

struct Bf16Reader {
    static inline __aicore__ float Read(const uint16_t* ptr, size_t idx) noexcept {
        return Bf16ToFp32(ptr[idx]);
    }
    static inline __aicore__ float Sub(float a, float b) noexcept {
        return Bf16ToFp32(Fp32ToBf16(a - b));
    }
};

/**
 * Data+Scale 相邻紧凑排布单点（quant_int8 KV 块与 quant_fp8 V 头共用）：
 * memcpy(data) + memcpy(scale) + 32B 对齐补零。调用方各自持有数据/尺寸语义，
 * 本函数只承载排布契约——禁止再写第二份三步 memcpy 样板。
 */
inline void PackDataAndScaleAdjacent(const void* data, uint32_t data_bytes,
                                     const void* scale, uint32_t scale_bytes,
                                     void* compact_dst) noexcept {
    std::memcpy(compact_dst, data, data_bytes);
    std::memcpy(static_cast<uint8_t*>(compact_dst) + data_bytes, scale, scale_bytes);
    const uint32_t total_written = data_bytes + scale_bytes;
    const uint32_t aligned_total = (total_written + 31u) / 32u * 32u;
    if (aligned_total > total_written) {
        std::memset(static_cast<uint8_t*>(compact_dst) + total_written, 0,
                    aligned_total - total_written);
    }
}

/**
 * smooth_k 的 K 减均值单点（差值位型舍回，对齐孪生 Half/Bf16Reader::Sub 口径）：
 * km 为 nullptr 时原值直通。kernel 设备/宿主分支与 quant 参考实现共用，
 * 禁止在调用方展开 (dtype==2)?Bf16:Half 三元式。
 */
__aicore__ __host__ inline float SubKm(float val, const float* km_vec, uint32_t c,
                                       uint32_t dtype_mode) noexcept {
    if (km_vec == nullptr) {
        return val;
    }
    return (dtype_mode == 2u) ? Bf16Reader::Sub(val, km_vec[c])
                              : HalfReader::Sub(val, km_vec[c]);
}

} // namespace quant
} // namespace sage_attn

/**
 * quant 层 C-ABI 唯一声明点（定义在 quant_int8.cpp / quant_fp8.cpp；
 * twin_reference.h 与外部单测经本头引用，禁止在他处重复声明）。
 * dtype_mode：0=float32，1=float16，2=bfloat16；布局编码 LAYOUT_HND/LAYOUT_NHD。
 */
extern "C" {
int QuantizeQInt8Ascend(const void* q_in, void* q_out, float* q_scale_out, uint32_t b, uint32_t hq,
                        uint32_t sq, uint32_t d, uint32_t qk_quant_gran, uint32_t tensor_layout,
                        int dtype_mode) noexcept;
int QuantizeKInt8Ascend(const void* k_in, const float* km_in, void* k_out, float* k_scale_out,
                        uint32_t b, uint32_t hkv, uint32_t sk, uint32_t d, uint32_t qk_quant_gran,
                        int smooth_k, uint32_t tensor_layout, int dtype_mode) noexcept;
int QuantizeVFp8Ascend(const void* v_in, void* v_fp8_out, float* v_scale_out, float* vm_out,
                       uint32_t b, uint32_t hkv, uint32_t sk, uint32_t d, uint32_t pv_accum_dtype,
                       int smooth_v, uint32_t tensor_layout, int dtype_mode) noexcept;
int PackKvDataAndScaleAdjacentAscend(const int8_t* k_data_block, const float* k_scale_block,
                                     uint8_t* compact_dst, uint32_t block_k, uint32_t d,
                                     uint32_t qk_quant_gran) noexcept;
int PackVDataAndScaleAdjacentAscend(const uint8_t* v_data_block, const float* v_scale_head,
                                    uint8_t* compact_dst, uint32_t pad_sk_v, uint32_t d) noexcept;
}

#endif // SAGE_ATTN_QUANT_COMMON_H
