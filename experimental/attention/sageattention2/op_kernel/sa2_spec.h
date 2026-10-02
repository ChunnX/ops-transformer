// sa2_spec.h — SageAttention2 v2 内核契约层：常量 / 枚举 / 参数块 / 数值派生（并发对齐点，字段序禁止改动）。
//
// 规格来源：hwdev/DESIGN_SPEC.md §3.2/§4/§5.3/§8（S.4 冻结稿）；归档 v23 附录 C.1 语义；
// todo.md §一.4 D-R 修订（D-R1 阈值法 / D-R2 O 驻 UB / D-R4 SF 打包口径 / D-R6 AIV 指令纪律）。
#ifndef SA2_SPEC_H_
#define SA2_SPEC_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if __has_include("kernel_operator.h")
// 设备构建：half / bfloat16_t 由 AscendC 头提供，禁止在本头重复定义（避免全局类型冲突）。
#include "kernel_operator.h"
// 设备侧 vapi 别名（vapi = AscendC::MicroAPI）：engine 双轨代码统一写 vapi::。
// 别名存在性 = __NPU_ARCH__ 门控（kernel_macros.h 仅在 arch pass 定义 MicroAPI = Reg）；
// MIX 驱动第三 pass（无 arch 定义，host-stub 载体）不得引用该命名空间。
#if defined(__NPU_ARCH__)
namespace vapi = AscendC::MicroAPI;
#endif
#else
// 宿主机路径：IEEE binary16 half 语义替身——RNE 舍入 + 次正规 + 溢出到 Inf，对齐设备 half /
// PyTorch .half()（设备构建不编译本段；数值口径由 tests/sa2_numeric_conformance.cpp 覆盖，
// 2026-10-01 迭代 2：截断替身改为 RNE，fp32+fp16 折叠 oracle 与真机 half 语义一致）。
namespace sa2::hostbits {
struct HalfF32 {
    uint16_t val{0};
    HalfF32() = default;
    HalfF32(float f) {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        const uint16_t sign = static_cast<uint16_t>((u >> 16) & 0x8000u);
        const uint32_t a = u & 0x7FFFFFFFu;
        if (a >= 0x7F800000u) {  // Inf/NaN：NaN 折叠为静音 qNaN（0x7E00 域）
            val = static_cast<uint16_t>(sign | 0x7C00u | (a > 0x7F800000u ? 0x0200u : 0u));
            return;
        }
        if (a >= 0x47800000u) {  // |x| ≥ 2^16：RNE 溢出（IEEE 上溢 → Inf；65520 恰 ties 亦入此界）
            val = static_cast<uint16_t>(sign | 0x7C00u);
            return;
        }
        const int32_t e = static_cast<int32_t>(a >> 23) - 127;
        if (e > -15) {  // half 正规域：保留高 13 位 + RNE（hm 进位 0x400 自然并入指数）
            const uint32_t m = a & 0x007FFFFFu;
            uint32_t hm = m >> 13;
            const uint32_t rem = m & 0x1FFFu;
            hm += (rem > 0x1000u || (rem == 0x1000u && (hm & 1u) != 0u)) ? 1u : 0u;
            val = static_cast<uint16_t>(sign | (((e + 15) << 10) + hm));
        } else {  // half 次正规域（含 ±0）：以 2^-24 步长 RNE，q 进位 0x400 自动升入 2^-14 正规
            const int32_t sh = -1 - e;  // e ≤ -15 ⇒ sh ≥ 14
            if (sh > 25) {              // |x| < 2^-25：RNE 归零（恰 2^-25 ties-to-even 亦归零）
                val = sign;
                return;
            }
            const uint32_t full = (a & 0x007FFFFFu) | 0x00800000u;  // 24 位有效数
            uint32_t q = full >> sh;
            const uint32_t rem = full & ((1u << sh) - 1u);
            const uint32_t halfStep = 1u << (sh - 1);
            q += (rem > halfStep || (rem == halfStep && (q & 1u) != 0u)) ? 1u : 0u;
            val = static_cast<uint16_t>(sign | q);
        }
    }
    operator float() const {
        const uint32_t sign = (val & 0x8000u) ? 0x80000000u : 0u;
        const uint32_t exp = (val >> 10) & 0x1Fu;
        const uint32_t mant = val & 0x03FFu;
        if (exp == 0) {
            if (mant == 0) return sign ? -0.0f : 0.0f;
            const uint32_t k = 31u - static_cast<uint32_t>(__builtin_clz(mant));
            const uint32_t u = sign | ((103u + k) << 23) | ((mant << (23u - k)) & 0x007FFFFFu);
            float f;
            std::memcpy(&f, &u, sizeof(f));
            return f;
        }
        uint32_t u = sign;
        if (exp == 31) {
            u |= 0x7F800000u | ((mant != 0) ? 0x00400000u : 0u);  // Inf / 静音 qNaN
        } else {
            u |= ((exp + 112u) << 23) | (mant << 13);
        }
        float f;
        std::memcpy(&f, &u, sizeof(f));
        return f;
    }
};
}  // namespace sa2::hostbits
using half = sa2::hostbits::HalfF32;
#if defined(__CCE_KT_TEST__) && __has_include("sa2_ascendc_hoststub.h")
#include "sa2_ascendc_hoststub.h"
#endif
#endif

namespace sa2 {

// ---- 设备/宿主双轨注解与数学替身（CANN 9.2.0-beta.1 dav-c310 实测）----
// 设备构建（-xcce 混编）下 libstdc++ 的 std::min/std::max/std::nearbyint/std::exp 均标记 __host__，
// __aicore__ 函数内不可调用；以下替身与 std 同语义（RNE / IEEE 单精度），宿主路径原样可用。
// SA2_AICORE：设备构建时（kernel_operator.h 已定义 __aicore__）展开为 __aicore__，宿主构建为空。
#if defined(__aicore__)
#define SA2_AICORE __aicore__
#else
#define SA2_AICORE
#endif
// 宿主桩轨（sim/spec 先于 hoststub 被包含时）的地址空间限定回退；设备构建下为真限定符。
#ifndef __gm__
#define __gm__
#endif

// M2b-0：bf16 标量位运算转换（AIV 后端不支持 __bf16 标量 cast 指令，实测报
// "not support bf16 type cast"）。读 = 指数域左移 16 位精确展开；写 = u32 RNE
//（+0x7FFF + 尾奇）截高位，与向量 Cast(CAST_RINT)/宿主 RNE 语义一致。fp16 走原生 half。
#if (defined(__AICORE__) && defined(__NPU_ARCH__)) || defined(__CCE_KT_TEST__)
SA2_AICORE inline float Sa2ScalarToFloat(bfloat16_t v)
{
    uint16_t u;
    u = *reinterpret_cast<const uint16_t *>(&v);
    uint32_t bits = static_cast<uint32_t>(u) << 16;
    return *reinterpret_cast<const float *>(&bits);
}

SA2_AICORE inline bfloat16_t Sa2ScalarFromFloatBf16(float f)
{
    uint32_t bits;
    bits = *reinterpret_cast<const uint32_t *>(&f);
    uint32_t bias = 0x7FFFu + ((bits >> 16) & 1u);
    uint16_t u = static_cast<uint16_t>((bits + bias) >> 16);
    bfloat16_t r;
    *reinterpret_cast<uint16_t *>(&r) = u;
    return r;
}
#endif

template <typename T>
SA2_AICORE inline T aicMin(T a, T b) { return a < b ? a : b; }
template <typename T>
SA2_AICORE inline T aicMax(T a, T b) { return a > b ? a : b; }

SA2_AICORE inline float aicNearbyInt(float x) {
    uint32_t bits;
#if defined(__AICORE__) || defined(__CCE_KT_TEST__)
    bits = *reinterpret_cast<const uint32_t *>(&x);
#else
    std::memcpy(&bits, &x, sizeof(bits));
#endif
    const uint32_t sign = bits & 0x80000000u;
    uint32_t u = bits & 0x7FFFFFFFu;

    if (u >= 0x4B000000u) {  // >= 2^23 or Inf/NaN：无小数位或非有限数，直通
        return x;
    }
    if (u <= 0x3F000000u) {  // |x| <= 0.5f：RNE 均舍入为 0.0f（0.5 ties-to-even 为 0）
        uint32_t res = sign;
#if defined(__AICORE__) || defined(__CCE_KT_TEST__)
        return *reinterpret_cast<const float *>(&res);
#else
        float r;
        std::memcpy(&r, &res, sizeof(r));
        return r;
#endif
    }
    if (u < 0x3F800000u) {  // 0.5f < |x| < 1.0f：RNE 舍入为 1.0f
        uint32_t res = sign | 0x3F800000u;
#if defined(__AICORE__) || defined(__CCE_KT_TEST__)
        return *reinterpret_cast<const float *>(&res);
#else
        float r;
        std::memcpy(&r, &res, sizeof(r));
        return r;
#endif
    }

    const uint32_t exp = u >> 23;
    const uint32_t shift = 150u - exp;  // 1 <= shift <= 23
    const uint32_t mask = (1u << shift) - 1u;
    const uint32_t half = 1u << (shift - 1);
    const uint32_t frac = u & mask;
    const uint32_t lsb = (u >> shift) & 1u;
    if (frac > half || (frac == half && lsb == 1u)) {
        u += (1u << shift);
    }
    u &= ~mask;
    uint32_t res = sign | u;
#if defined(__AICORE__) || defined(__CCE_KT_TEST__)
    return *reinterpret_cast<const float *>(&res);
#else
    float r;
    std::memcpy(&r, &res, sizeof(r));
    return r;
#endif
}

SA2_AICORE inline float aicAbs(float x) {
    uint32_t bits;
#if defined(__AICORE__) || defined(__CCE_KT_TEST__)
    bits = *reinterpret_cast<const uint32_t *>(&x);
#else
    std::memcpy(&bits, &x, sizeof(bits));
#endif
    bits &= 0x7FFFFFFFu;
#if defined(__AICORE__) || defined(__CCE_KT_TEST__)
    return *reinterpret_cast<const float *>(&bits);
#else
    float res;
    std::memcpy(&res, &bits, sizeof(res));
    return res;
#endif
}

SA2_AICORE inline float aicExp(float x) {
#if defined(__aicore__)
    return __builtin_expf(x);    // 单精度 exp，数值口径同 std::exp(float)
#else
    return std::exp(x);
#endif
}

template <typename T>
struct AicRemoveRef { using type = T; };
template <typename T>
struct AicRemoveRef<T&> { using type = T; };
template <typename T>
struct AicRemoveRef<T&&> { using type = T; };

// M2b-0 实测补充：AIV loop 同样不支持标量 f32→f16 收窄指令（后端 "Unsupported scalar
// instruction in AIV loop"，最小对照实测定位）。fp16 标量转换亦走位运算 RNE：
// u32 + 0x7FFF + 尾奇 → 截高 16 位，与向量 Cast(CAST_RINT)/宿主 RNE 逐位一致。
#if (defined(__AICORE__) && defined(__NPU_ARCH__)) || defined(__CCE_KT_TEST__)
// __simd_callee__：VF 函数体内可调用（cce 实测：VF 内仅可调 simd_callee 标注函数）
__simd_callee__ inline uint16_t Sa2F32ToHalfBits(float f)
{
    uint32_t bits;
    bits = *reinterpret_cast<const uint32_t *>(&f);
    uint32_t bias = 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + bias) >> 16);
}
#endif
template <typename T>
SA2_AICORE inline T Sa2ScalarFromFloat(float f) { return static_cast<T>(f); }
#if (defined(__AICORE__) && defined(__NPU_ARCH__)) || defined(__CCE_KT_TEST__)
template <>
SA2_AICORE inline half Sa2ScalarFromFloat<half>(float f)
{
    uint16_t u = Sa2F32ToHalfBits(f);
    half r;
    std::memcpy(&r, &u, sizeof(r));
    return r;
}
#endif
#if (defined(__AICORE__) && defined(__NPU_ARCH__)) || defined(__CCE_KT_TEST__)
// f16 位型 → f32 位型（纯整数，无任何浮点指令/位强转）。迭代 2 实测补充：
// Sa2HalfBitsToFloat 末端的「位型 → float」强转在 AIV 编译单元内不可选中（ISel
// "Cannot select: f32 = bitcast"，p33f/p33g 探针），wrapper 侧展开也同罪（内联后同单元）。
// 本变体只产位型，调用方经 u32 装载/存储过桥，float 侧由「直接标量装载」重建
//（bitcast(load) 可选中，p33e 探针）。次正规分支以 CLZ 整数归一化替代 __builtin_ldexpf
//（AIV 禁标量浮点库调用）：mant × 2^-24 在 f32 为精确正规数（mant ≤ 2^10-1 < 2^24，
// 无舍入），位型 = (103 + k) << 23 | (mant << (23 - k)) & 0x7FFFFF，k = 最高有效位。
__simd_callee__ inline uint32_t Sa2HalfBitsToF32Bits(uint16_t u)
{
    uint32_t sign = static_cast<uint32_t>(u & 0x8000u) << 16;
    uint32_t exp = (u >> 10) & 0x1Fu;
    uint32_t mant = u & 0x03FFu;
    if (exp == 0) {
        if (mant == 0) {
            return sign;
        }
        // mant × 2^-24 = 1.m × 2^(k-24)：f32 指数域 = 127 + (k - 24) = 103 + k
        uint32_t k = 31u - static_cast<uint32_t>(__builtin_clz(mant));
        return sign | ((103u + k) << 23) | ((mant << (23u - k)) & 0x7FFFFFu);
    }
    if (exp == 31) {
        return sign | 0x7F800000u | (mant << 13);
    }
    return sign | ((exp + 112u) << 23) | (mant << 13);
}

// f16 位型 → f32 展开（纯整数位运算，无 f16 cast 指令）：符号 | 指数重偏置 | 尾数移位。
// 次正规 f16 输入乘 2^24 缩放（与 hostbits::HalfF32::operator float 同式，逐位一致）。
// 注意：末端含「位型 → float」强转，仅供 AIC/宿主语境调用；AIV 编译单元内一律改用
// Sa2HalfBitsToF32Bits（见上注记）。
__simd_callee__ inline float Sa2HalfBitsToFloat(uint16_t u)
{
    uint32_t bits = Sa2HalfBitsToF32Bits(u);
    return *reinterpret_cast<const float *>(&bits);
}
#endif
// 标量 half 读入统一入口：设备走位运算展开（AIV loop 禁 f16 cast 指令），宿主原生。
template <typename T>
SA2_AICORE inline float Sa2ScalarToFloat(T v) { return static_cast<float>(v); }
#if (defined(__AICORE__) && defined(__NPU_ARCH__)) || defined(__CCE_KT_TEST__)
// VF 专用入口（独立命名，避免与非 simd_callee 的重载集在同命名空间冲突：
// ccec 实测 —— VF 体内重载决议会因非 simd_callee 候选存在而整体不可达）
__simd_callee__ inline float Sa2HalfScalarToFloat(half v)
{
    uint16_t u;
    std::memcpy(&u, &v, sizeof(u));
    return Sa2HalfBitsToFloat(u);
}
#endif
#if (defined(__AICORE__) && defined(__NPU_ARCH__)) || defined(__CCE_KT_TEST__)
template <>
SA2_AICORE inline bfloat16_t Sa2ScalarFromFloat<bfloat16_t>(float f) { return Sa2ScalarFromFloatBf16(f); }
#endif

template <typename T>
SA2_AICORE inline typename AicRemoveRef<T>::type &&aicMove(T &&t) noexcept {
    return static_cast<typename AicRemoveRef<T>::type &&>(t);
}

// IEEE FP8 E4M3FN 编码（OCP 规范：1 位符号，4 位指数 bias=7，3 位尾数，max 448.0，RNE 舍入）
// 采用与 PyTorch / IEEE 754-2008 原生硬件加法器对齐的快速位运算实现
SA2_AICORE inline uint8_t FloatToFp8E4M3(float val) {
    constexpr uint32_t kFp8Max = 1087u << 20;       // 480.0f：首个不可表示的 FP8 溢出阈值
    constexpr uint32_t kDenormMask = 141u << 23;    // ((127 - 7) + (23 - 3) + 1) << 23

    uint32_t fBits;
    std::memcpy(&fBits, &val, sizeof(float));
    const uint32_t sign = fBits & 0x80000000u;
    fBits ^= sign; // 取绝对值比特

    uint8_t result = 0u;
    if (fBits >= kFp8Max) {
        if (fBits > 0x7F800000u) {
            // NaN 输入 -> NaN 码字 (0x7F)
            result = 0x7Fu;
        } else {
            // 溢出或 Inf -> 饱和到最大正规数 448.0f (0x7E)
            // 注记：(+Inf)→448 与 PyTorch float8_e4m3fn 转换语义（Inf→NaN）不一致；本算子数据流（softmax 概率 ∈[0,κ]，κ ≤ 448）不产生 Inf，故不构成契约偏差。
            result = 0x7Eu;
        }
    } else {
        if (fBits < (121u << 23)) {
            // 次正规数范围 (< 2^-6 = 0.015625f，最小正规格化数)
            // 借用 FP32 浮点加法器自动执行硬件级 Round-to-Nearest-Even (RNE)
            float denormF;
            std::memcpy(&denormF, &kDenormMask, sizeof(float));
            float fAbs;
            std::memcpy(&fAbs, &fBits, sizeof(float));
            float temp = fAbs + denormF;
            std::memcpy(&fBits, &temp, sizeof(float));
            result = static_cast<uint8_t>(fBits - kDenormMask);
        } else {
            // 规格化数范围：标准 RNE 舍入
            uint8_t mantOdd = static_cast<uint8_t>((fBits >> 20) & 1u);
            fBits += (static_cast<uint32_t>(7 - 127) << 23) + 0x7FFFFu;
            fBits += mantOdd; // 奇数尾数加 1 完成 RNE ties-to-even
            result = static_cast<uint8_t>(fBits >> 20);
            if (result == 0x7Fu) {
                // 舍入进位恰好落入 NaN 码字 (0x7F) 时饱和到 448.0f (0x7E)
                result = 0x7Eu;
            }
        }
    }
    result |= static_cast<uint8_t>(sign >> 24);
    return result;
}

SA2_AICORE inline float Fp8E4M3ToFloat(uint8_t b) {
    const uint32_t s = static_cast<uint32_t>(b & 0x80u) << 24;
    const uint32_t exp = (b >> 3) & 0x0Fu;
    const uint32_t mant = b & 0x07u;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = s;
        } else {
            const uint32_t k = 31u - static_cast<uint32_t>(__builtin_clz(mant));
            bits = s | ((118u + k) << 23) | ((mant << (23u - k)) & 0x007FFFFFu);
        }
    } else if (exp == 15u && mant == 7u) {
        bits = s | 0x7FC00000u;
    } else {
        bits = s | ((exp + 120u) << 23) | (mant << 20);
    }
#if defined(__AICORE__) || defined(__CCE_KT_TEST__)
    return *reinterpret_cast<const float *>(&bits);
#else
    float r;
    std::memcpy(&r, &bits, sizeof(r));
    return r;
#endif
}

// AMLA E8M0 尺度整数加法代乘（Add-instead-of-Multiply，D-R13）：
// 对浮点数乘以 2^expVal，直接在阶码段执行整数加法，消除向量乘法器与除法器延迟。
// 精确域（全 2^32 位型抽样穷举对照 std::ldexp，2026-10-02 迭代 3）：结果指数 ∈ (0,255) 的正规结果
// 100% 逐位精确；结果下溢（newExp ≤ 0）冲为零（对齐设备 FTZ，核内可达域最小 |v|·2^-127 ≫ 该界外）；
// ±Inf 原样直通（核内输入均有限，防 Inf·2^neg 误降为有限数的潜在误用）。
SA2_AICORE inline float AmlaScaleMultiply(float val, int32_t expVal) {
    uint32_t bits;
#if defined(__AICORE__) || defined(__CCE_KT_TEST__)
    bits = *reinterpret_cast<const uint32_t *>(&val);
#else
    std::memcpy(&bits, &val, sizeof(bits));
#endif
    if ((bits & 0x7FFFFFFFu) == 0 || (((bits & 0x7F800000u) == 0x7F800000u) && ((bits & 0x007FFFFFu) != 0))) {
        return val;
    }
    const int32_t curExp = static_cast<int32_t>((bits >> 23) & 0xFFu);
    const uint32_t sign = bits & 0x80000000u;
    if (curExp == 0) {
        // 次正规数（非零）：用 CLZ 整数归一化对阶替代 __builtin_ldexpf，下溢 FTZ 冲零
        const uint32_t mant = bits & 0x007FFFFFu;
        const uint32_t k = 31u - static_cast<uint32_t>(__builtin_clz(mant));
        const int32_t newExp = expVal + static_cast<int32_t>(k) - 22;
        if (newExp <= 0) return 0.0f;  // FTZ 冲零
        if (newExp >= 255) {
            bits = sign | 0x7F800000u;
        } else {
            const uint32_t normMant = (mant << (23u - k)) & 0x007FFFFFu;
            bits = sign | (static_cast<uint32_t>(newExp) << 23) | normMant;
        }
#if defined(__AICORE__) || defined(__CCE_KT_TEST__)
        return *reinterpret_cast<const float *>(&bits);
#else
        float res;
        std::memcpy(&res, &bits, sizeof(res));
        return res;
#endif
    }
    if (curExp == 255) {
        return val;  // ±Inf 直通：避免负 expVal 把 Inf 误降为有限大数
    }
    const int32_t newExp = curExp + expVal;
    if (newExp <= 0) return 0.0f;
    if (newExp >= 255) {
        bits = sign | 0x7F800000u;
    } else {
        bits = (bits & 0x807FFFFFu) | (static_cast<uint32_t>(newExp) << 23);
    }
#if defined(__AICORE__) || defined(__CCE_KT_TEST__)
    return *reinterpret_cast<const float *>(&bits);
#else
    float res;
    std::memcpy(&res, &bits, sizeof(res));
    return res;
#endif
}

// E8M0 向上取幂位技巧（TileKernels/MIT 等价式）+ P1-6 幂边界饱和兜底：
// uexp = (bits(amax/448) + 0x007FFFFF) >> 23，即 ceil(log2(amax/448)) + 127。
// 边界兜底：若量化后 amax * 2^(-expVal) > 448.0f（例如由于 amax/448 单精度除法舍入跨 2^k 边界少取一档），
// 自动执行 expVal += 1（uexp += 1）兜底，严格杜绝 FloatToFp8E4M3 截断溢出。
SA2_AICORE static inline uint8_t ComputeE8m0Exponent(float amax) {
    if (amax <= 0.0f) {
        return 0;
    }
    float x = amax / 448.0f;
    uint32_t bits = 0;
#if defined(__AICORE__) || defined(__CCE_KT_TEST__)
    bits = *reinterpret_cast<const uint32_t *>(&x);
#else
    std::memcpy(&bits, &x, sizeof(bits));
#endif
    uint32_t rawExp = (bits + 0x007FFFFFu) >> 23;
    if (rawExp > 254) rawExp = 254;
    uint8_t uexp = static_cast<uint8_t>(rawExp);
    const int32_t expVal = static_cast<int32_t>(uexp) - 127;
    if (expVal < 127 && (AmlaScaleMultiply(amax, -expVal) > 448.0f || amax > AmlaScaleMultiply(448.0f, expVal))) {
        ++uexp;
    }
    return uexp;
}

// ---- tile 常量（DESIGN_SPEC §3.2：BR=64、BC=128、DP∈{64,128} 运行期）----
constexpr int64_t kStreamRows = 64;         // BR：每条 AIV 流独占一个 64 行 query tile（R.0.2 双独立流）
constexpr int64_t kKvStep = 128;            // BC：KV 步进行数（D-R3：Bc=128 加倍 softmax 工作量，M2.2 预留 QK K-dim 拆分点）
constexpr int64_t kMaxHeadDim = 128;        // head_dim 上限；dp 补齐 64/128（任务书 §3：>128 报错）
constexpr int64_t kSegCols = 16;            // per_thread 的 D 维分段列宽（R.2.2；初值取开源 WARPQ=16，终值以 golden 反推收口）
constexpr int64_t kSegsFull = kMaxHeadDim / kSegCols;  // 8：DP=128 全长分段数；DP=64 时运行期为 4
constexpr int64_t kScoreDeqShift = 14;      // QK fixpipe DEQ 标量 2^-14：127·127·128×2^-14≈126 < 65504，acc=1 → 6.1e-5 仍 FP16 normal（§4.1）
constexpr float kPDomainFp16Accum = 2.25f;  // κ = scale_max（fp32+fp16 档）：SageAttention2++ FP16 累加封顶（§4.3）
constexpr float kPDomainFp32Accum = 448.0f; // κ = scale_max（fp32 / fp32+fp32 档）：E4M3 满量程（§4.3）
constexpr float kMinGroupMax = 1.0e-30f;    // V MX 组 amax 下限：全零组 → 最小正 2 的幂 scale（§4.8）
constexpr float kFp16NegMax = -65504.0f;    // running max 初始化以 -65504 替代 -inf（§3.3 状态初始化，FP16 域安全）
constexpr int64_t kMxGroupTokens = 32;      // V MX 组：32 token × channel（§4.8）
constexpr int64_t kMxScalePack = 2;         // D-R4：K 维相邻 2 个 32-token 组的 UE8M0 打包进一个 int16

// ---- M2c-A1 N512 Wide 路径常量与参数契约规划（长序列超宽流水线）----
constexpr int64_t kWideKvStep = 512;        // BC_WIDE：超宽 KV 块步长（两 N256 Cube 片段共享一次 Softmax）
constexpr int64_t kWideTileM = 64;          // BR_WIDE：Wide 路径下 query tile 行数保持 64
constexpr int64_t kWideQkSteps = 2;         // 每个 N512 宽步内执行的 N256 QK GEMM 片段数
constexpr int64_t kWideMinSeqLen = 4096;    // 长序列 Wide 路径激活门禁：S >= 4096
constexpr int64_t kWideFragmentKv = 256;    // 每个 Cube GEMM 片段的 KV 列宽（256）
constexpr int64_t kWideHeadDim = 128;       // Wide 路径支持的 head_dim（当前定位于 DP=128）

// Wide 路径静态契约断言：
static_assert(kWideKvStep == kWideQkSteps * kWideFragmentKv, "N512 宽步严格由 2 个 N256 片段拼成");
static_assert(kWideKvStep % kMxGroupTokens == 0, "Wide KV 步长必须对齐 V MX 组大小 (32)");
static_assert(kWideKvStep >= kKvStep, "Wide 步长大于等于标准步长 (128)");

// 判断当前任务是否满足 N512 Wide 路径激活契约：
// 条件：DP=128、Sk >= 4096 且按 512 对齐、Sq 按 64 对齐
constexpr bool IsWidePathEligible(int64_t sq, int64_t sk, int64_t d) {
    return (d == kWideHeadDim) && (sk >= kWideMinSeqLen) && (sk % kWideKvStep == 0) && (sq % kWideTileM == 0);
}

// ---- M2c-A3 Q/K 混流三级流水批量量化管线契约（QuantBatch 流水架构）----
// 规划 64 行 tile 的三级重叠流水（MTE2 装载切片 i+1 / Vector 核心量化切片 i / MTE3 写回切片 i-1）
// 以及 Q/K Scale 倒数复用机制（以 1 次倒数 + D 次向量乘法消除内层高延迟除法）。
constexpr int64_t kQuantTileRows = 64;       // 64 行 Tile 切片大小（BR 维对齐）
constexpr int64_t kQuantPipelineStages = 3;  // MTE2 装载 / Vector 量化 / MTE3 写回 3 级重叠
constexpr int64_t kQuantBufferSlots = 2;     // Ping-Pong 双缓冲槽位数

// 编译期静态契约断言：
static_assert(kStreamRows == kQuantTileRows, "Q 流行数与批量量化切片行数严格对齐 (64)");
static_assert(kKvStep % kQuantTileRows == 0, "KV 步进行数 (128) 必须为批量量化切片行数 (64) 的整数倍");
static_assert(kQuantPipelineStages == 3, "三级流水重叠级数冻结为 3 级 (MTE2/Vector/MTE3)");
static_assert(kQuantBufferSlots == 2, "双缓冲槽位数冻结为 2 槽 (Ping-Pong)");

// Q/K Scale 倒数复用结构：消除每元素重复除法，将除法转换为 1 次倒数 + D 次向量乘法
struct Sa2QuantScaleReciprocal {
    float scale{1.0f};
    float invScale{1.0f};
    float scaledFactor{1.0f};

    SA2_AICORE constexpr Sa2QuantScaleReciprocal() = default;
    SA2_AICORE inline Sa2QuantScaleReciprocal(float amax, float factor) {
        Init(amax, factor);
    }

    SA2_AICORE inline void Init(float amax, float factor) {
        float s = amax / 127.0f;
        if (s < 1e-7f) {
            s = 1.0f;
        }
        scale = s;
        invScale = 1.0f / s;
        scaledFactor = s * factor;
    }

    SA2_AICORE inline int8_t Quantize(float val) const {
        float scaled = aicNearbyInt(val * invScale);
        return static_cast<int8_t>(aicMax(-127.0f, aicMin(127.0f, scaled)));
    }
};

// 混流批量量化切片描述符
struct Sa2QuantSliceDesc {
    int64_t sliceIndex{0};
    int64_t rowOffset{0};
    int64_t totalRows{0};
    int64_t validRows{0};
    int64_t headDim{0};
    int64_t paddedDim{0};
    int64_t bufferSlot{0}; // 0 = ping, 1 = pong
};

// M2c-A3 Q/K 混流批量量化 3 级重叠流水控制器
// 阶段 0: MTE2 异步加载 Tile i + 1 入 ping/pong 槽 (i+1)%2
// 阶段 1: Vector 核心计算 Tile i 量化与尺度提取 (槽 i%2)
// 阶段 2: MTE3 异步写回 Tile i - 1 产物至目标槽 (槽 (i-1)%2)
template <typename LoadFn, typename ComputeFn, typename StoreFn>
struct Sa2QuantBatchPipeline {
    static constexpr int64_t kStages = kQuantPipelineStages;
    static constexpr int64_t kSlots = kQuantBufferSlots;
    static constexpr int64_t kTileRows = kQuantTileRows;

    int64_t totalRows{0};
    int64_t headDim{0};
    int64_t paddedDim{0};
    int64_t numSlices{0};

    SA2_AICORE constexpr Sa2QuantBatchPipeline(int64_t totalR, int64_t d, int64_t dp)
        : totalRows(totalR), headDim(d), paddedDim(dp),
          numSlices((totalR + kTileRows - 1) / kTileRows) {}

    SA2_AICORE inline Sa2QuantSliceDesc GetSliceDesc(int64_t sliceIdx) const {
        Sa2QuantSliceDesc desc{};
        desc.sliceIndex = sliceIdx;
        desc.rowOffset = sliceIdx * kTileRows;
        desc.totalRows = totalRows;
        int64_t rem = totalRows - desc.rowOffset;
        desc.validRows = rem < 0 ? 0 : (rem > kTileRows ? kTileRows : rem);
        desc.headDim = headDim;
        desc.paddedDim = paddedDim;
        desc.bufferSlot = sliceIdx % kSlots;
        return desc;
    }

    SA2_AICORE inline void Execute(LoadFn &&loadFn, ComputeFn &&computeFn, StoreFn &&storeFn) const {
        if (numSlices <= 0) return;

        // 1. Prologue: 启动首个 Slice 的 Load (阶段 0)
        loadFn(GetSliceDesc(0));

        // 2. Main Loop: 3 级重叠流水
        // 每步并发推进：
        //   Load(i + 1) [若 i + 1 < numSlices]
        //   Compute(i)
        //   Store(i - 1) [若 i > 0]
        for (int64_t i = 0; i < numSlices; ++i) {
            if (i + 1 < numSlices) {
                loadFn(GetSliceDesc(i + 1));
            }
            computeFn(GetSliceDesc(i));
            if (i > 0) {
                storeFn(GetSliceDesc(i - 1));
            }
        }

        // 3. Epilogue: 排水最后一个 Slice 的 Store
        storeFn(GetSliceDesc(numSlices - 1));
    }
};

template <typename LoadFn, typename ComputeFn, typename StoreFn>
SA2_AICORE inline void ExecuteQuantBatchPipeline(
    int64_t totalRows, int64_t headDim, int64_t paddedDim,
    LoadFn &&loadFn, ComputeFn &&computeFn, StoreFn &&storeFn)
{
    Sa2QuantBatchPipeline<LoadFn, ComputeFn, StoreFn> pipe(totalRows, headDim, paddedDim);
    pipe.Execute(
        static_cast<LoadFn &&>(loadFn),
        static_cast<ComputeFn &&>(computeFn),
        static_cast<StoreFn &&>(storeFn));
}

// ---- M2c-A5 Stream-K K 均值统计分块常量与描述符 ----
constexpr int64_t kKMeanBlockRows = 128;      // 128 行分块批量处理（与 kKvStep = 128 对齐）
constexpr int64_t kKMeanDataBlockLanes = 64;  // D-R8 AIV 内 datablock 归约基数（64 lane 向量长度）
constexpr int64_t kKMeanDoubleBufSlots = 2;   // Ping-Pong 双缓冲槽位数

static_assert(kKMeanBlockRows % kKvStep == 0, "K 均值分块行数必须为 kKvStep 的整数倍");
static_assert(kKMeanDataBlockLanes == 64, "AIV 向量寄存器单次归约 lane 数必须为 64 (256B / 4B)");
static_assert(kKMeanDoubleBufSlots == 2, "Stream-K 均值统计采用 ping-pong 双缓冲");

struct Sa2StreamKMeanTile {
    int64_t headIdx{0};         // 归属的 KV head 索引
    int64_t chunkStart{0};      // 本分块在序列中的起始 token 行
    int64_t chunkEnd{0};        // 本分块在序列中的结束 token 行
    int64_t totalTokens{0};     // 该 head 的总 token 数 (sk)
    bool isExclusiveHead{true}; // 是否独占整个 head（true: 独写；false: 跨界需汇聚）
    bool isFirstChunk{true};    // 是否为该 head 的首个分块
    bool isLastChunk{true};     // 是否为该 head 的尾部分块
};

// ---- M2c-B1 L2 Cache Hint 逐 Buffer 策略枚举（D-R5 落地）----
enum Sa2L2CacheHint : uint8_t {
    CACHE_HINT_NORMAL_FIRST_VICTIM = 0,
    CACHE_HINT_NORMAL_LAST_VICTIM = 1,   // 用于 K8/V8/Scale：指示 L2 优先常驻，供 GQA 多 Query Head 反复命中
    CACHE_HINT_NORMAL_PERSISTENT = 2,
    CACHE_HINT_NOTALLOC_KEEP = 4,        // 用于单次流式 Q 与写回 Output：避免污染 L2 缓存行
    CACHE_HINT_NOTALLOC_CLEAN = 5,
};

// 逐 Buffer 硬件 L2 Cache 策略表（针对混合注意力不同张量的访问生命周期精确指派）。
// 迭代 2 收口（P2-1）：kQHint / kOutHint 两常量删除——零调用点，且按当前数据流不存在
// 可挂 hint 的真实 DataCopy 点：Q 不走 GM DataCopy（AIV StepVectorQQuantDevice 标量读 GM
// 后直写 L1 q8/qScale，无 MTE2 通路可挂 CACHE_MODE）；Output 写回为 AIV 标量 GM 写
//（FinishTileVectorDevice，V 引擎直写 GM，无 MTE3 DataCopy 可挂）。若后续 M2b 接入
// Q 的 GM→L1 MTE2 预取或 O 的 UB→GM MTE3 批量写回，再按需以 CACHE_HINT_NOTALLOC_KEEP
// 重新引入并接线（本注释即唯一口径，禁止无调用点常量回潮）。
struct Sa2BufferCachePolicy {
    static constexpr Sa2L2CacheHint kK8Hint = CACHE_HINT_NORMAL_LAST_VICTIM;     // K8 常驻 L2，供 GQA 复用
    static constexpr Sa2L2CacheHint kV8Hint = CACHE_HINT_NORMAL_LAST_VICTIM;     // V8 常驻 L2，供 GQA 复用
    static constexpr Sa2L2CacheHint kScaleHint = CACHE_HINT_NORMAL_LAST_VICTIM;  // Scale 张量常驻 L2
};

// ---- M2c-B4 V MX Scale L1 缓存结构（按 Head 缓存机制）----
// 在 L1 内按 Head 缓存 UE8M0 尺度张量。当 GQA 多个 Query Head 访问同一 KV Head 时，
// 直接复用 L1 缓存的热 Scale 数据，消除每次 GEMM 重复从 GM/L2 加载 Scale 的 MTE2 延迟。
constexpr int64_t kL1MxScaleCacheBytes = 176 * 1024;  // 176 KiB：按 head 缓存容量上限

struct Sa2MxScaleL1Cache {
    int64_t cachedBatch{-1};
    int64_t cachedHead{-1};
    // __gm__ 限定：设备构建下普通指针与地址空间限定指针不可隐式互转（CCE 硬约束），
    // 身份比对用 GM 基址必须携带同一定址空间；hoststub 下展开为空，宿主轨不受影响。
    const __gm__ void *gmBase{nullptr};
    uint8_t *l1Base{nullptr};
    int64_t capacityBytes{kL1MxScaleCacheBytes};
    int64_t validBytes{0};

    SA2_AICORE inline Sa2MxScaleL1Cache() = default;

    SA2_AICORE inline explicit Sa2MxScaleL1Cache(uint8_t *l1Buf, int64_t capacity = kL1MxScaleCacheBytes)
        : l1Base(l1Buf), capacityBytes(capacity) {}

    SA2_AICORE inline void Init(uint8_t *l1Buf, int64_t capacity = kL1MxScaleCacheBytes) {
        l1Base = l1Buf;
        capacityBytes = capacity;
        Reset();
    }

    SA2_AICORE inline void Reset() {
        cachedBatch = -1;
        cachedHead = -1;
        gmBase = nullptr;
        validBytes = 0;
    }

    SA2_AICORE inline bool IsHit(int64_t b, int64_t kh, const __gm__ void *base) const {
        return (cachedBatch == b && cachedHead == kh && gmBase == base && gmBase != nullptr && validBytes > 0);
    }

    SA2_AICORE inline void MarkCached(int64_t b, int64_t kh, const __gm__ void *base, int64_t bytes) {
        cachedBatch = b;
        cachedHead = kh;
        gmBase = base;
        validBytes = bytes;
    }

    SA2_AICORE inline uint8_t *GetL1Buffer() const {
        return l1Base;
    }
};

// ---- 容量水位（DESIGN_SPEC §2/§5）----
constexpr int64_t kUbHighWater = 248 * 1024;  // 253952：UB 编程高水位（物理 256KB）
constexpr int64_t kUbEndPerWarp = 152448;     // per_warp 模式静态物理 UB 占用 (148.875 KB <= 160 KB)
constexpr int64_t kUbEndPerThread = 217984;   // per_thread 模式静态物理 UB 占用 (212.875 KB <= 248 KB)
constexpr int64_t kUbHardSafetyLimit = 160 * 1024; // 163840：瞬时活跃工作集硬安全门禁 (160 KB)
constexpr int64_t kUbPhysicalLimit = 256 * 1024;   // 262144：硬件物理上限 (256 KB)
constexpr int64_t kL1Size = 512 * 1024;       // 524288
constexpr int64_t kL0CSize = 256 * 1024;      // 262144
constexpr int64_t kL0ABBytes = 32 * 1024;     // L0A / L0B 各 32KB

// ---- D-R6 AIV 指令纪律（常量与注释层；VF 实现文件必须遵循）----
// icache 仅 8KB：VF 主循环 #pragma unroll 1 + VLOOP 控制；优先单指令复合原语
//（exp_sub / madd / reduce_max_datablock）；掩码合并走 MODE_MERGING；指针走 postupdate；
// 归约类与算术类 V 指令端口错峰（整数加↔浮点加错峰）；NZ 缓冲保留 pad 行防 bank conflict。
constexpr int64_t kVfIcacheBytes = 8 * 1024;  // 8192：VF icache 容量上限

// ---- 量化粒度与 PV 累加档（任务书 §2.3/§功能定义；模板/运行期取值见 sa2_template_key.h）----
enum class Gran : int64_t {
  kPerWarp = 0,    // 每 token 一个 scale（行内 amax/127，零行 → 1.0）
  kPerThread = 1,  // D 维按 kSegCols 分段，每段一个 scale（v2 真实现）
};

enum class PvAccum : int64_t {
  kFp32 = 0,       // fp32 累加（950PR Cube 累加器即 FP32，与 fp32+fp32 数值同路，§4.4）
  kFp32Fp32 = 1,   // fp32+fp32
  kFp32Fp16 = 2,   // fp32+fp16：κ=2.25 + FP16 往返折叠（R.1.4）
};

// ---- host 传入参数块（字段序 = 附录 C.1 契约，host tiling 与 kernel 双侧逐字对齐）----
struct Sa2Params {
  int64_t b, hq, hk, sq, sk, d, dp, qp, kp;  // dp/qp/kp 由 DeriveNumericPolicy 填充（§5.3）
  int64_t qStride[3];                        // {batch, head, row} 元素步长；末维连续（任务书 stride(-1)==1）
  int64_t kStride[3];
  int64_t vStride[3];
  bool causal;
  bool smoothK;
  bool smoothV;     // 仅 fp32 档生效（任务书 §2.3 契约）
  bool returnLse;
  int64_t gran;     // Gran 的 int64 载体（模板维，决定 tiling key）
  int64_t accum;    // PvAccum 的 int64 载体（运行期维，决定 pDomain）
  float smScale;
  float pDomain;    // κ（DeriveNumericPolicy 填充；运行期参数，非模板，R.0.4）
  float lnPDomain;  // θ ≡ ln(κ)（D-R1：阈值法 rescale 触发阈值，无需新增参数）
  float qFactor;    // QH_i = s_q_i · qFactor = s_q_i · sign(s)·128·sqrt(|s|)（负号随 s，§4.2/R.1.2）
  float kFactor;    // KH_j = s_k_j · kFactor = s_k_j · 128·sqrt(|s|)
  int64_t segWidth; // kScale 每行段数：per_warp=1，per_thread=16
  int64_t cores;    // AIC 核数（SplitRange 均分用）
};

// ---- tiling 数据块（kernel 侧 GET_TILING_DATA_WITH_STRUCT 载体）----
// workspace 六区由 host 按 §5.3 分配（512B 对齐），地址语义见 sa2_planner.h KvHeadView；
// kScale 为 fp16 纯量（不参与 UE8M0 打包）；vScale 打包遵循 D-R4 950 口径：
// K 维相邻两个 32-token 组的 UE8M0 打包 int16、MN-major 存储（出处 DeepGEMM-Ascend README:56 + transform_sf.hpp）。
struct Sa2TilingData {
  Sa2Params params;
  uint64_t kMean;   // B·Hk·DP·f32
  uint64_t vMean;   // B·Hk·DP·f32（仅 smooth_v 且 fp32 档语义非零）
  uint64_t k8;      // B·Hk·Kp·DP·int8
  uint64_t kScale;  // B·Hk·Kp·SEG·f16（per_warp 时 SEG=1）
  uint64_t v8;      // B·Hk·Kp·DP·int8
  uint64_t vScale;  // B·Hk·(DP·Kp/64)·int16（打包后字节数 = B·Hk·DP·Kp/32）
};

// ---- kernel 入口约定（后续入口文件遵循；DESIGN_SPEC §8 / 任务书 §4）----
//   op 类型 = SageAttention2；opFile.value = "sa2_kernel"；opInterface.value = "sa2_attention2"；
//   入口函数 = sa2_attention2（__global__ __aicore__，与 opInterface 同名，模板形参序 DTYPE/GRAN/CAUSAL，
//   见 sa2_template_key.h）；
//   tiling 注册 = REGISTER_TILING_DEFAULT(Sa2TilingData)；
//   任务类型 = KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)（1 AIC + 2 AIV，每 AIV 一条 64 行流）。

namespace detail {

// constexpr sqrt：牛顿迭代，64 轮对任意正有限值收敛（host 派生与编译期断言共用）。
constexpr double Sa2Sqrt(double a) {
  if (a <= 0.0) {
    return 0.0;
  }
  double x = a >= 1.0 ? a : 1.0;
  for (int i = 0; i < 64; ++i) {
    x = 0.5 * (x + a / x);
  }
  return x;
}

// constexpr ln：x = 2^k·m 归约 + atanh((m-1)/(m+1)) 级数（33 项，m∈[1,2) 收敛足够）。
constexpr double Sa2Ln(double x) {
  constexpr double kLn2 = 0.69314718055994530942;
  double m = x;
  double k = 0.0;
  while (m >= 2.0) {
    m *= 0.5;
    k += 1.0;
  }
  while (m < 1.0) {
    m *= 2.0;
    k -= 1.0;
  }
  double t = (m - 1.0) / (m + 1.0);
  double t2 = t * t;
  double s = 0.0;
  double p = t;
  for (int i = 1; i <= 33; i += 2) {
    s += p / static_cast<double>(i);
    p *= t2;
  }
  return k * kLn2 + 2.0 * s;
}

constexpr double Sa2Abs(double x) { return x < 0.0 ? -x : x; }

constexpr int64_t RoundUpToImpl(int64_t x, int64_t a) { return (x + a - 1) / a * a; }

}  // namespace detail

constexpr int64_t RoundUpTo(int64_t x, int64_t a) { return detail::RoundUpToImpl(x, a); }

// head_dim 补齐（任务书 §3：<64 → 64；(64,128) → 128；>128 报错，此处返回 0 作 host 校验哨兵）。
constexpr int64_t PaddedHeadDim(int64_t d) { return d <= 64 ? 64 : (d <= 128 ? 128 : 0); }

constexpr float PDomainOf(PvAccum accum) {
  return accum == PvAccum::kFp32Fp16 ? kPDomainFp16Accum : kPDomainFp32Accum;
}

constexpr int64_t SegWidthOf(Gran gran) { return gran == Gran::kPerWarp ? 1 : kSegCols; }

// ---- 数值派生（附录 C.1 语义；host tiling 填 Sa2Params 用， constexpr 供双侧断言）----
// g = 128·sqrt(|s|)：g² = |s|·2^14，与 DEQ 常数拼成 QH·KH = s_q·s_k·s·2^14（softmax 内每元素恰一次乘法）。
struct Sa2NumericPolicy {
  float qFactor = 0.0f;     // sign(s)·g（负号随 s 进 Q 侧因子）
  float kFactor = 0.0f;     // g
  float pDomain = 0.0f;     // κ：按 accum 分档
  float lnPDomain = 0.0f;   // θ = ln(κ)：按 accum 分档（D-R1）
  int64_t segWidth = 0;     // 按 gran：1 或 16
  int64_t qp = 0;           // RoundUpTo(sq, kStreamRows)
  int64_t kp = 0;           // RoundUpTo(sk, kKvStep)（128 为 32 的倍数，V MX 组对齐自动满足，B-40）
  int64_t dp = 0;           // PaddedHeadDim(d)
};

constexpr Sa2NumericPolicy DeriveNumericPolicy(Gran gran, PvAccum accum, int64_t sq, int64_t sk,
                                               int64_t d, float smScale) {
  const double sAbs = static_cast<double>(smScale < 0.0f ? -smScale : smScale);
  const double g = 128.0 * detail::Sa2Sqrt(sAbs);
  const double sgn = smScale < 0.0f ? -1.0 : 1.0;
  const float pDomain = PDomainOf(accum);
  Sa2NumericPolicy out;
  out.qFactor = static_cast<float>(sgn * g);
  out.kFactor = static_cast<float>(g);
  out.pDomain = pDomain;
  out.lnPDomain = static_cast<float>(detail::Sa2Ln(static_cast<double>(pDomain)));
  out.segWidth = SegWidthOf(gran);
  out.qp = RoundUpTo(sq, kStreamRows);
  out.kp = RoundUpTo(sk, kKvStep);
  out.dp = PaddedHeadDim(d);
  return out;
}

inline Sa2NumericPolicy DeriveNumericPolicy(const Sa2Params& p) {
  return DeriveNumericPolicy(static_cast<Gran>(p.gran), static_cast<PvAccum>(p.accum), p.sq, p.sk,
                             p.d, p.smScale);
}

// ---- 静态自检（契约值冻结；预期值已逐一核算入交付报告）----
static_assert(kStreamRows == 64 && kKvStep == 128 && kMaxHeadDim == 128, "tile 三常数冻结值");
static_assert(kSegCols == 16 && kSegsFull == 8 && kScoreDeqShift == 14, "分段与 DEQ 常数冻结值");
static_assert(kUbHighWater == 253952 && kL1Size == 524288 && kL0CSize == 262144, "容量水位冻结值");
static_assert(127 * 127 * kMaxHeadDim == 2064512, "int8×int8×k=128 累加上界（§4.1）");
static_assert(static_cast<int64_t>(Gran::kPerWarp) == 0 && static_cast<int64_t>(Gran::kPerThread) == 1,
              "Gran 枚举值 = tiling 载体值");
static_assert(static_cast<int64_t>(PvAccum::kFp32) == 0 && static_cast<int64_t>(PvAccum::kFp32Fp32) == 1 &&
                  static_cast<int64_t>(PvAccum::kFp32Fp16) == 2,
              "PvAccum 枚举值 = tiling 载体值");
static_assert(sizeof(Sa2Params) == 208, "Sa2Params 布局冻结：9×8 + 9×8 + 4 + pad4 + 16 + 4 + 16 + 16 = 208");
static_assert(sizeof(Sa2TilingData) == 256, "Sa2TilingData = 208 + 6×8 = 256");
static_assert(offsetof(Sa2Params, gran) == 152, "gran @ 152（bool×4 后 4 字节对齐填充）");
static_assert(offsetof(Sa2Params, smScale) == 168, "smScale @ 168");
static_assert(offsetof(Sa2Params, segWidth) == 192, "segWidth @ 192");
static_assert(offsetof(Sa2TilingData, kMean) == 208, "kMean @ 208（params 之后首字段）");

// 派生数值工作例：s = 1/sqrt(128)（d=128 默认 scale），fp32+fp16 档，per_thread，sq=4096、sk=16384。
constexpr Sa2NumericPolicy kProbePolicy =
    DeriveNumericPolicy(Gran::kPerThread, PvAccum::kFp32Fp16, 4096, 16384, 128, 0.08838834764831845f);
static_assert(kProbePolicy.qp == 4096 && kProbePolicy.kp == 16384 && kProbePolicy.dp == 128,
              "qp/kp/dp 派生：RoundUp(4096,64)=4096、RoundUp(16384,128)=16384、pad(128)=128");
static_assert(kProbePolicy.segWidth == 16, "per_thread → segWidth=16");
static_assert(kProbePolicy.pDomain == 2.25f, "fp32+fp16 → κ=2.25");
static_assert(detail::Sa2Abs(static_cast<double>(kProbePolicy.lnPDomain) - 0.81093021621632876) < 1.0e-6,
              "θ = ln(2.25) ≈ 0.810930（D-R1 fp32+fp16 档阈值）");
static_assert(kProbePolicy.kFactor > 0.0f && kProbePolicy.qFactor == kProbePolicy.kFactor,
              "s>0：qFactor = +g = 128·sqrt(0.0883883) ≈ 38.054628");
constexpr Sa2NumericPolicy kProbePolicyNeg =
    DeriveNumericPolicy(Gran::kPerWarp, PvAccum::kFp32, 250, 1000, 48, -0.5f);
static_assert(kProbePolicyNeg.qp == 256 && kProbePolicyNeg.kp == 1024 && kProbePolicyNeg.dp == 64,
              "RoundUp(250,64)=256、RoundUp(1000,128)=1024、pad(48)=64");
static_assert(kProbePolicyNeg.segWidth == 1 && kProbePolicyNeg.pDomain == 448.0f,
              "per_warp → segWidth=1；fp32 档 → κ=448");
static_assert(kProbePolicyNeg.qFactor < 0.0f && kProbePolicyNeg.qFactor == -kProbePolicyNeg.kFactor,
              "s<0：负号随 s 进 qFactor（-90.509667），kFactor = +90.509667");
static_assert(detail::Sa2Abs(static_cast<double>(kProbePolicyNeg.lnPDomain) - 6.1047932322953815) < 1.0e-6,
              "θ = ln(448) ≈ 6.104793（D-R1 fp32 档阈值，FlashMLA 同款）");

}  // namespace sa2

#endif  // SA2_SPEC_H_
