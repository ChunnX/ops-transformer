#pragma once
#include "kernel_operator.h"
#include <type_traits>

/*
combine_log.h（AscendC kernel 侧）

  - KLOG("msg\n")
  - KLOGF("x=%u\n", x)
  - KVAL(x)                                // 自动附带变量名
  - KDump1(xLocal_1D, len)                 // 1维（默认：DumpTensor）
  - KDump2(xLocal_2D, row, col)            // 2维（默认：DumpTensor）
  - KDump3(xLocal_3D, d0, d1, d2)          // 3维（默认：DumpTensor）
  - KDump1_P / KDump2_P / KDump3_P         // printf逐元素打印

说明与约束：
    1) KDump可以打印LocalTensor / GlobalTensor，支持基本数据类型（整数/浮点/half/bfloat16等）。
    2) KDump使用的是AscendC::DumpTensor()，若需要 printf逐元素打印，使用 KDump_P 版本。
        printf-loop 打印依赖 Tensor.GetValue(offset)：
        - LocalTensor::GetValue 在一些 position（如 VECIN/VECCALC/VECOUT）可用，其他位置可能不支持。
        - GlobalTensor::GetValue 会从 GM 读取；若存在 cache 一致性问题需在外侧处理。
    3) AscendC printf 仅支持换行符 '\n'，不要使用 '\t' 等其它转义字符。
    4) 打印量受 1MB/核 限制；printf-loop 很容易爆量。

示例：
    KLOG("Hello, World!\n");
        输出： [Line: 26] Hello, World!
        
    KLOGF("x=%u, y=%f\n", x, y);
        输出： [Line: 28] x=123, y=3.140000

    KVAL(x);
        输出： [Line: 30] x=123

    KDump1(xLocalTensor, len);
        输出： [1.000000, 2.000000, ..., 16.000000]

    KDump2(expandGM_, axisBs, axisH);
        输出：
        [1.000000, 2.000000, ..., 16.000000
         17.000000, 18.000000, ..., 32.000000
         ...
         241.000000, 242.000000, ..., 256.000000]
    KDump3(tensor3D, d0, d1, d2);
    KDump1_P(tensor1D, len);
*/

#ifndef KLOG_ENABLE
#define KLOG_ENABLE 1
#endif
using namespace AscendC;
namespace KLog {

__aicore__ inline uint32_t DescFromLine(uint32_t line)
{
    return (line & 0xFFFFu);
}
__aicore__ inline uint32_t TsIdFromLine(uint32_t line)
{
    return 0x10000u | (line & 0xFFFFu);
}

// ========================
// 1) 字符串 / 格式化
// ========================
__aicore__ inline void LogS(uint32_t line, __gm__ const char *msg)
{
#if KLOG_ENABLE
    AscendC::printf("[Line: %u] %s", (uint32_t)(line & 0xFFFFu), msg);
#else
    (void)line;
    (void)msg;
#endif
}

template <class... Args>
__aicore__ inline void LogF(uint32_t line, __gm__ const char *fmt, Args &&...args)
{
#if KLOG_ENABLE
    AscendC::printf("[Line: %u] ", (uint32_t)(line & 0xFFFFu));
    AscendC::printf(fmt, static_cast<Args &&>(args)...);
#else
    (void)line;
    (void)fmt;
#endif
}

// ========================
// 2) 标量（自动带变量名）
// ========================
__aicore__ inline void ValImpl(uint32_t line, const char *name, const void *v)
{
#if KLOG_ENABLE
    AscendC::printf("[Line: %u] %s=%p\n", (uint32_t)(line & 0xFFFFu), name, v);
#else
    (void)line;
    (void)name;
    (void)v;
#endif
}

__aicore__ inline void ValImpl(uint32_t line, const char *name, int64_t v)
{
#if KLOG_ENABLE
    AscendC::printf("[Line: %u] %s=%lld\n", (uint32_t)(line & 0xFFFFu), name, (long long)v);
#else
    (void)line;
    (void)name;
    (void)v;
#endif
}

__aicore__ inline void ValImpl(uint32_t line, const char *name, uint64_t v)
{
#if KLOG_ENABLE
    AscendC::printf("[Line: %u] %s=%llu\n", (uint32_t)(line & 0xFFFFu), name, (unsigned long long)v);
#else
    (void)line;
    (void)name;
    (void)v;
#endif
}

template <typename T>
__aicore__ inline void ValImpl(uint32_t line, const char *name, T v)
{
    // 浮点/half/bfloat16：建议用 KLOGF("%f\n", (float)v)
    if constexpr (std::is_pointer_v<T>) {
        ValImpl(line, name, (const void *)v);
    } else if constexpr (std::is_unsigned_v<T>) {
        ValImpl(line, name, (uint64_t)v);
    } else {
        ValImpl(line, name, (int64_t)v);
    }
}

// ========================
// 3) printf-loop：逐元素打印
// ========================

// template <typename V>
// __aicore__ inline void PrintOne(V v)
// {
// #if KLOG_ENABLE
//     if constexpr (std::is_floating_point_v<V>) {
//         AscendC::printf("%f", (float)v);
//     } else if constexpr (std::is_integral_v<V> && std::is_unsigned_v<V>) {
//         AscendC::printf("%llu", (unsigned long long)v);
//     } else if constexpr (std::is_integral_v<V>) {
//         AscendC::printf("%lld", (long long)v);
//     } else {
//         // half/bfloat16_t 等：通常可转 float；若环境不支持，请在此处加重载/特化
//         AscendC::printf("%f", (float)v);
//     }
// #else
//     (void)v;
// #endif
// }


// 1) float：直接 %f
__aicore__ inline void PrintOne(float v) {
    AscendC::printf("%f", v);
}

// 2) half：直接 %f（不转 float）
__aicore__ inline void PrintOne(half v) {
    AscendC::printf("%f", v);
}

// 3) bfloat16_t：直接 %f（不转 float）
__aicore__ inline void PrintOne(bfloat16_t v) {
    AscendC::printf("%f", v);
}

// 4) 其它整型
template <typename V>
__aicore__ inline std::enable_if_t<std::is_integral_v<V> && std::is_unsigned_v<V>, void>
PrintOne(V v) { AscendC::printf("%llu", (unsigned long long)v); }

template <typename V>
__aicore__ inline std::enable_if_t<std::is_integral_v<V> && !std::is_unsigned_v<V>, void>
PrintOne(V v) { AscendC::printf("%lld", (long long)v); }


template <typename TensorT>
__aicore__ inline auto TensorGetValue(const TensorT &t, uint64_t idx) -> decltype(t.GetValue(idx))
{
    return t.GetValue(idx);
}

template <typename TensorT>
__aicore__ inline void Print1D(const TensorT &t, uint32_t len)
{
#if KLOG_ENABLE
    AscendC::printf("[");
    for (uint32_t i = 0; i < len; ++i) {
        auto v = TensorGetValue(t, (uint64_t)i);
        PrintOne(v);
        if (i + 1 != len)
            AscendC::printf(", ");
    }
    AscendC::printf("]\n");
#else
    (void)t;
    (void)len;
#endif
}

template <typename TensorT>
__aicore__ inline void Print2D(const TensorT &t, uint32_t rows, uint32_t cols)
{
#if KLOG_ENABLE
    // 2D：每行换行
    for (uint32_t r = 0; r < rows; ++r) {
        uint64_t base = (uint64_t)r * (uint64_t)cols;
        for (uint32_t c = 0; c < cols; ++c) {
            auto v = TensorGetValue(t, base + (uint64_t)c);
            PrintOne(v);
            if (c + 1 != cols)
                AscendC::printf(", ");
        }
        AscendC::printf("\n");
    }
#else
    (void)t;
    (void)rows;
    (void)cols;
#endif
}

template <typename TensorT>
__aicore__ inline void Print3D(const TensorT &t, uint32_t d0, uint32_t d1, uint32_t d2)
{
#if KLOG_ENABLE
    // 3D：按 plane 分段；每个 plane 的每行换行，plane 间额外空行
    uint64_t plane = (uint64_t)d1 * (uint64_t)d2;
    for (uint32_t i = 0; i < d0; ++i) {
        uint64_t off0 = (uint64_t)i * plane;
        for (uint32_t r = 0; r < d1; ++r) {
            uint64_t base = off0 + (uint64_t)r * (uint64_t)d2;
            for (uint32_t c = 0; c < d2; ++c) {
                auto v = TensorGetValue(t, base + (uint64_t)c);
                PrintOne(v);
                if (c + 1 != d2)
                    AscendC::printf(", ");
            }
            AscendC::printf("\n");
        }
        if (i + 1 != d0)
            AscendC::printf("\n");
    }
#else
    (void)t;
    (void)d0;
    (void)d1;
    (void)d2;
#endif
}

// ========================
// 4) DumpTensor 后端（带 shape 的版本内部构造 ShapeInfo）
// ========================
template <typename T>
__aicore__ inline void Dump1D(const LocalTensor<T> &t, uint32_t desc, uint32_t len)
{
#if KLOG_ENABLE
    AscendC::DumpTensor(t, desc, len);
#else
    (void)t;
    (void)desc;
    (void)len;
#endif
}

template <typename T>
__aicore__ inline void Dump1D(const GlobalTensor<T> &t, uint32_t desc, uint32_t len)
{
#if KLOG_ENABLE
    AscendC::DumpTensor(t, desc, len);
#else
    (void)t;
    (void)desc;
    (void)len;
#endif
}

template <typename T>
__aicore__ inline void Dump2D(const LocalTensor<T> &t, uint32_t desc, uint32_t rows, uint32_t cols)
{
#if KLOG_ENABLE
    uint32_t shp[2] = {rows, cols};
    AscendC::ShapeInfo si(2, shp);
    AscendC::DumpTensor(t, desc, rows * cols, si);
#else
    (void)t;
    (void)desc;
    (void)rows;
    (void)cols;
#endif
}

template <typename T>
__aicore__ inline void Dump2D(const GlobalTensor<T> &t, uint32_t desc, uint32_t rows, uint32_t cols)
{
#if KLOG_ENABLE
    uint32_t shp[2] = {rows, cols};
    AscendC::ShapeInfo si(2, shp);
    AscendC::DumpTensor(t, desc, rows * cols, si);
#else
    (void)t;
    (void)desc;
    (void)rows;
    (void)cols;
#endif
}

template <typename T>
__aicore__ inline void Dump3D(const LocalTensor<T> &t, uint32_t desc, uint32_t d0, uint32_t d1, uint32_t d2)
{
#if KLOG_ENABLE
    uint32_t shp[3] = {d0, d1, d2};
    AscendC::ShapeInfo si(3, shp);
    AscendC::DumpTensor(t, desc, d0 * d1 * d2, si);
#else
    (void)t;
    (void)desc;
    (void)d0;
    (void)d1;
    (void)d2;
#endif
}

template <typename T>
__aicore__ inline void Dump3D(const GlobalTensor<T> &t, uint32_t desc, uint32_t d0, uint32_t d1, uint32_t d2)
{
#if KLOG_ENABLE
    uint32_t shp[3] = {d0, d1, d2};
    AscendC::ShapeInfo si(3, shp);
    AscendC::DumpTensor(t, desc, d0 * d1 * d2, si);
#else
    (void)t;
    (void)desc;
    (void)d0;
    (void)d1;
    (void)d2;
#endif
}

// ========================
// 5) 后端选择：默认用 DumpTensor；_P 强制 printf-loop
// ========================
template <typename TensorT>
__aicore__ inline void KDump1_Default(const TensorT &t, uint32_t desc, uint32_t len)
{
    KLog::Dump1D(t, desc, len);
}
template <typename TensorT>
__aicore__ inline void KDump2_Default(const TensorT &t, uint32_t desc, uint32_t r, uint32_t c)
{
    KLog::Dump2D(t, desc, r, c);
}
template <typename TensorT>
__aicore__ inline void KDump3_Default(const TensorT &t, uint32_t desc, uint32_t d0, uint32_t d1, uint32_t d2)
{
    KLog::Dump3D(t, desc, d0, d1, d2);
}

template <typename TensorT>
__aicore__ inline void KDump1_Printf(const TensorT &t, uint32_t desc, uint32_t len)
{
#if KLOG_ENABLE
    AscendC::printf("[Line: %u] PrintfLoop 1D len=%u\n", (uint32_t)(desc & 0xFFFFu), len);
#endif
    KLog::Print1D(t, len);
}
template <typename TensorT>
__aicore__ inline void KDump2_Printf(const TensorT &t, uint32_t desc, uint32_t r, uint32_t c)
{
#if KLOG_ENABLE
    AscendC::printf("[Line: %u] PrintfLoop 2D %ux%u\n", (uint32_t)(desc & 0xFFFFu), r, c);
#endif
    KLog::Print2D(t, r, c);
}
template <typename TensorT>
__aicore__ inline void KDump3_Printf(const TensorT &t, uint32_t desc, uint32_t d0, uint32_t d1, uint32_t d2)
{
#if KLOG_ENABLE
    AscendC::printf("[Line: %u] PrintfLoop 3D %ux%ux%u\n", (uint32_t)(desc & 0xFFFFu), d0, d1, d2);
#endif
    KLog::Print3D(t, d0, d1, d2);
}

} // namespace KLog

// ========================
// 对外宏
// ========================

#if KLOG_ENABLE
#define KLOG(msg)                                                                                                      \
    do {                                                                                                               \
        KLog::LogS(__LINE__, (msg));                                                                                   \
    } while (0)
#define KLOGF(fmt, ...)                                                                                                \
    do {                                                                                                               \
        KLog::LogF(__LINE__, (fmt), ##__VA_ARGS__);                                                                    \
    } while (0)
#define KVAL(x)                                                                                                        \
    do {                                                                                                               \
        KLog::ValImpl(__LINE__, #x, (x));                                                                              \
    } while (0)
#define KTS()                                                                                                          \
    do {                                                                                                               \
        AscendC::PrintTimeStamp(KLog::TsIdFromLine(__LINE__));                                                         \
    } while (0)
#else
#define KLOG(msg)                                                                                                      \
    do {                                                                                                               \
    } while (0)
#define KLOGF(fmt, ...)                                                                                                \
    do {                                                                                                               \
    } while (0)
#define KVAL(x)                                                                                                        \
    do {                                                                                                               \
    } while (0)
#define KTS()                                                                                                          \
    do {                                                                                                               \
    } while (0)
#endif

// 默认：DumpTensor
#if KLOG_ENABLE
#define KDump1(t, len)                                                                                                 \
    do {                                                                                                               \
        KLog::KDump1_Default((t), KLog::DescFromLine(__LINE__), (uint32_t)(len));                                      \
    } while (0)
#define KDump2(t, r, c)                                                                                                \
    do {                                                                                                               \
        KLog::KDump2_Default((t), KLog::DescFromLine(__LINE__), (uint32_t)(r), (uint32_t)(c));                         \
    } while (0)
#define KDump3(t, d0, d1, d2)                                                                                          \
    do {                                                                                                               \
        KLog::KDump3_Default((t), KLog::DescFromLine(__LINE__), (uint32_t)(d0), (uint32_t)(d1), (uint32_t)(d2));       \
    } while (0)
#else
#define KDump1(t, len)                                                                                                 \
    do {                                                                                                               \
    } while (0)
#define KDump2(t, r, c)                                                                                                \
    do {                                                                                                               \
    } while (0)
#define KDump3(t, d0, d1, d2)                                                                                          \
    do {                                                                                                               \
    } while (0)
#endif

// 强制：printf-loop
#if KLOG_ENABLE
#define KDump1_P(t, len)                                                                                               \
    do {                                                                                                               \
        KLog::KDump1_Printf((t), KLog::DescFromLine(__LINE__), (uint32_t)(len));                                       \
    } while (0)
#define KDump2_P(t, r, c)                                                                                              \
    do {                                                                                                               \
        KLog::KDump2_Printf((t), KLog::DescFromLine(__LINE__), (uint32_t)(r), (uint32_t)(c));                          \
    } while (0)
#define KDump3_P(t, d0, d1, d2)                                                                                        \
    do {                                                                                                               \
        KLog::KDump3_Printf((t), KLog::DescFromLine(__LINE__), (uint32_t)(d0), (uint32_t)(d1), (uint32_t)(d2));        \
    } while (0)
#else
#define KDump1_P(t, len)                                                                                               \
    do {                                                                                                               \
    } while (0)
#define KDump2_P(t, r, c)                                                                                              \
    do {                                                                                                               \
    } while (0)
#define KDump3_P(t, d0, d1, d2)                                                                                        \
    do {                                                                                                               \
    } while (0)
#endif