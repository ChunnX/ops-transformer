// sa2_ascendc_hoststub.h — 宿主机语法检查/仿真专用 AscendC 与 CATLASS / tla 替身头。
// 仅当 __has_include("kernel_operator.h") 为假（宿主 g++/clang++ 检查、无 CANN 头路径）时使用；
// 真机 bisheng / CANN 构建走真实头（kernel_operator.h, catlass/catlass.hpp, tla/tensor.hpp）。
#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <cmath>
#include <cstring>
#include <algorithm>

#if !defined(SA2_SPEC_H_) && __has_include("sa2_spec.h")
#include "sa2_spec.h"
#endif

#ifndef __global__
#define __global__
#endif
#ifndef __aicore__
#define __aicore__
#endif
#ifndef __gm__
#define __gm__
#endif
#ifndef GM_ADDR
using GM_ADDR = uint8_t *;
#endif



#ifndef __simd_vf__
#define __simd_vf__
#endif
#ifndef __simt_vf__
#define __simt_vf__
#endif
#ifndef __ubuf__
#define __ubuf__
#endif
#ifndef __simd_callee__
#define __simd_callee__
#endif

enum QuantMode_t : uint32_t {
    NoQuant = 0,
    F322F16 = 1,
    F322BF16 = 2,
    DEQF16 = 3,
    VDEQF16 = 4,
    QF322B8_PRE = 5,
    VQF322B8_PRE = 6,
    REQ8 = 7,
    VREQ8 = 8,
    QS322BF16_PRE = 9,
    VQS322BF16_PRE = 10,
    QF322F16_PRE = 11,
    VQF322F16_PRE = 12,
    QF322BF16_PRE = 13,
    VQF322BF16_PRE = 14,
    QF322F32_PRE = 15,
    VQF322F32_PRE = 16
};

using bfloat16_t = uint16_t;
using float8_e4m3_t = uint8_t;
using float8_e5m2_t = uint8_t;
using float8_e8m0_t = uint8_t;

namespace sa2 {
uint8_t FloatToFp8E4M3(float val);
float Fp8E4M3ToFloat(uint8_t b);
float aicNearbyInt(float x);
float aicLog(float x);
inline bfloat16_t FloatToBf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    u += 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<bfloat16_t>(u >> 16);
}
}

namespace AscendC {

constexpr int32_t AIC = 0;
constexpr int32_t AIV = 1;

enum class CacheMode {
    CACHE_MODE_NORMAL,
    CACHE_MODE_DISABLE,
    CACHE_MODE_LAST,
    CACHE_MODE_PERSISTENT
};

using mx_fp8_e4m3_t = uint8_t;
using mx_fp8_e5m2_t = uint8_t;

enum TPosition : int32_t {
    GM = 0,
    A1 = 1,
    A2 = 2,
    B1 = 3,
    B2 = 4,
    C1 = 5,
    C2 = 6,
    CO1 = 7,
    CO2 = 8,
    VECIN = 9,
    VECOUT = 10,
    VECCALC = 11,
    LCM = VECCALC,
    SPM = 12,
    SHM = SPM,
    TSCM = 13,
    C2PIPE2GM = 14,
    MAX = 15
};

enum class QuePosition : int32_t {
    VECIN = 0,
    VECOUT = 1,
    A1 = 2,
    A2 = 3,
    B1 = 4,
    B2 = 5,
    CO1 = 6,
    CO2 = 7
};

enum class CO2Layout : uint8_t {
    NZ = 0,
    ROW_MAJOR = 1,
    COLUMN_MAJOR = 2
};

struct FixpipeConfig {
    CO2Layout layout{CO2Layout::ROW_MAJOR};
    bool isToUB{true};
};

enum class ReluMode : uint8_t {
    NO_RELU = 0,
    RELU = 1
};

enum class ClipReluMode : uint8_t {
    NOCLIP_RELU = 0,
    CLIP_RELU = 1
};

template <typename T>
class LocalTensor {
public:
    using PrimType = T;
    LocalTensor() = default;
    LocalTensor(T *ptr, uint32_t size = 0, TPosition pos = TPosition::VECCALC)
        : ptr_(ptr), size_(size), pos_(pos) {}

    T *GetData() const { return ptr_; }
    T *data() const { return ptr_; }
    uint64_t GetPhyAddr() const { return reinterpret_cast<uint64_t>(ptr_); }
    uint32_t GetSize() const { return size_; }
    void SetSize(uint32_t size) { size_ = size; }
    TPosition GetPosition() const { return pos_; }

    template <typename U>
    LocalTensor<U> ReinterpretCast() const {
        return LocalTensor<U>(reinterpret_cast<U *>(ptr_), size_ * sizeof(T) / sizeof(U), pos_);
    }

    LocalTensor<T> operator[](uint32_t offset) const {
        return LocalTensor<T>(ptr_ ? ptr_ + offset : nullptr, size_ > offset ? size_ - offset : 0, pos_);
    }
    LocalTensor<T> operator[](int64_t offset) const {
        return LocalTensor<T>(ptr_ ? ptr_ + offset : nullptr, (offset >= 0 && static_cast<int64_t>(size_) > offset) ? static_cast<uint32_t>(static_cast<int64_t>(size_) - offset) : 0, pos_);
    }
    LocalTensor<T> operator[](uint64_t offset) const {
        return LocalTensor<T>(ptr_ ? ptr_ + offset : nullptr, static_cast<uint64_t>(size_) > offset ? static_cast<uint32_t>(static_cast<uint64_t>(size_) - offset) : 0, pos_);
    }

    T &operator()(uint32_t idx) { return ptr_[idx]; }
    const T &operator()(uint32_t idx) const { return ptr_[idx]; }

    T GetValue(uint32_t offset) const { return ptr_ ? ptr_[offset] : T(); }
    void SetValue(uint32_t offset, const T &val) { if (ptr_) ptr_[offset] = val; }

private:
    T *ptr_{nullptr};
    uint32_t size_{0};
    TPosition pos_{TPosition::VECCALC};
};

template <typename T>
class GlobalTensor {
public:
    using PrimType = T;
    GlobalTensor() = default;
    GlobalTensor(T *ptr, uint64_t size = 0) : ptr_(ptr), size_(size) {}

    template <typename U>
    void SetGlobalBuffer(__gm__ U *buffer, uint64_t bufferSize = 0) {
        ptr_ = reinterpret_cast<T *>(buffer);
        size_ = bufferSize;
    }
    T *GetData() const { return ptr_; }
    T *data() const { return ptr_; }
    uint64_t GetSize() const { return size_; }

    template <typename U>
    GlobalTensor<U> ReinterpretCast() const {
        return GlobalTensor<U>(reinterpret_cast<U *>(ptr_), size_ * sizeof(T) / sizeof(U));
    }

    GlobalTensor<T> operator[](uint64_t offset) const {
        return GlobalTensor<T>(ptr_ ? ptr_ + offset : nullptr, size_ > offset ? size_ - offset : 0);
    }

    template <typename H>
    void SetL2CacheHint(H hint) { (void)hint; }

    template <typename H>
    void set_l2_cache_hint(H hint) { (void)hint; }

private:
    T *ptr_{nullptr};
    uint64_t size_{0};
};

template <TPosition pos = TPosition::VECCALC>
class TBuf {
public:
    TBuf() = default;
    ~TBuf() {
        if (buf_) {
            delete[] buf_;
            buf_ = nullptr;
        }
    }
    TBuf(const TBuf &) = delete;
    TBuf &operator=(const TBuf &) = delete;
    TBuf(TBuf &&other) noexcept : buf_(other.buf_), sizeBytes_(other.sizeBytes_) {
        other.buf_ = nullptr;
        other.sizeBytes_ = 0;
    }
    TBuf &operator=(TBuf &&other) noexcept {
        if (this != &other) {
            if (buf_) delete[] buf_;
            buf_ = other.buf_;
            sizeBytes_ = other.sizeBytes_;
            other.buf_ = nullptr;
            other.sizeBytes_ = 0;
        }
        return *this;
    }

    template <typename T>
    LocalTensor<T> Get() const {
        if (buf_ == nullptr || sizeBytes_ == 0) {
            return LocalTensor<T>(nullptr, 0, pos);
        }
        return LocalTensor<T>(reinterpret_cast<T*>(buf_),
                              static_cast<uint32_t>(sizeBytes_ / (sizeof(T) == 0 ? 1 : sizeof(T))), pos);
    }
    template <typename T>
    LocalTensor<T> Allocate(uint32_t size) {
        Init(size * sizeof(T));
        return LocalTensor<T>(reinterpret_cast<T*>(buf_), size, pos);
    }
    void Init(uint32_t bytes) {
        if (buf_) delete[] buf_;
        sizeBytes_ = bytes;
        buf_ = bytes > 0 ? new uint8_t[bytes]() : nullptr;
    }
private:
    uint8_t *buf_{nullptr};
    uint32_t sizeBytes_{0};
};

class TPipe {
public:
    TPipe() = default;
    template <TPosition pos>
    void InitBuffer(TBuf<pos> &buf, uint32_t bytes) {
        buf.Init(bytes);
    }
    void Destroy() {}
};

inline TPipe *GetTPipePtr() {
    static TPipe pipe;
    return &pipe;
}
}  // namespace AscendC

using AscendC::GetTPipePtr;

namespace AscendC {

struct MmadParams {
    uint16_t m = 0;
    uint16_t n = 0;
    uint16_t k = 0;
    bool isBias = false;
    int32_t fmOffset = 0;
    bool enSsparse = false;
    bool enWinogradA = false;
    bool enWinogradB = false;
    uint8_t unitFlag = 0;
    bool kDirectionAlign = false;
    bool cmatrixSource = false;
    bool cmatrixInitVal = true;
};

struct LoadData2DParamsV2 {
    uint32_t mStartPosition = 0;
    uint32_t kStartPosition = 0;
    uint32_t mStep = 0;
    uint32_t kStep = 0;
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
    bool ifTranspose = false;
};

struct LoadData2DMxParams {
    uint32_t xStartPosition = 0;
    uint32_t yStartPosition = 0;
    uint32_t xStep = 0;
    uint32_t yStep = 0;
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
};

struct DataCopyParams {
    uint16_t blockCount = 1;
    uint16_t blockLen = 1;
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
};

struct FixpipeParams {
    uint16_t m = 0;
    uint16_t n = 0;
    uint16_t ndNum = 1;
    uint16_t srcStride = 0;
    uint16_t dstStride = 0;
    bool cmatrixInitVal = false;
};

template <CO2Layout format = CO2Layout::ROW_MAJOR>
struct FixpipeParamsArch3510 {
    FixpipeParamsArch3510() = default;
    FixpipeParamsArch3510(uint16_t nSizeIn, uint16_t mSizeIn, uint16_t srcStrideIn, uint32_t dstStrideIn)
        : nSize(nSizeIn), mSize(mSizeIn), srcStride(srcStrideIn), dstStride(dstStrideIn) {}

    ReluMode preReluMode = ReluMode::NO_RELU;
    ClipReluMode preClipReluMode = ClipReluMode::NOCLIP_RELU;
    uint64_t reluScalar = 0;
    uint64_t vectorRelu = 0;
    uint16_t nSize = 0;
    uint16_t mSize = 0;
    uint16_t srcStride = 0;
    uint32_t dstStride = 0;
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    float deqScalar = 1.0f;
    bool reluEn = false;
    uint8_t unitFlag = 0;
    uint8_t dualDstCtl = 0;
    bool subBlockId = false;
    bool isChannelSplit = false;
};

template <CO2Layout format = CO2Layout::ROW_MAJOR>
using FixpipeParamsC310 = FixpipeParamsArch3510<format>;

inline void SyncAll() {}
inline int64_t GetBlockIdx() { return 0; }
inline int64_t GetSubBlockIdx() { return 0; }
inline GM_ADDR GetUserWorkspace(GM_ADDR ws) { return ws; }

template <int32_t PipeType = 0>
inline void PipeBarrier() {}

enum class HardEvent {
    MTE2_V = 0,
    V_MTE2 = 1,
    MTE1_MTE2 = 2,
    MTE3_V = 3,
    V_MTE3 = 4,
};

template <HardEvent Ev>
inline void SetFlag(uint16_t eventId) { (void)eventId; }

template <HardEvent Ev>
inline void WaitFlag(uint16_t eventId) { (void)eventId; }

template <typename T, typename U, typename S>
inline void Mmad(const LocalTensor<T> &dst, const LocalTensor<U> &fm, const LocalTensor<S> &filter, const MmadParams &params) {
    (void)dst; (void)fm; (void)filter; (void)params;
}

template <typename T, typename U, typename S, typename V>
inline void Mmad(const LocalTensor<T> &dst, const LocalTensor<U> &fm, const LocalTensor<S> &filter, const LocalTensor<V> &bias, const MmadParams &params) {
    (void)dst; (void)fm; (void)filter; (void)bias; (void)params;
}

template <typename T, typename U>
inline void DataCopy(const LocalTensor<T> &dst, const GlobalTensor<U> &src, uint32_t count) {
    (void)dst; (void)src; (void)count;
}

template <typename T, typename U>
inline void DataCopy(const GlobalTensor<T> &dst, const LocalTensor<U> &src, uint32_t count) {
    (void)dst; (void)src; (void)count;
}

template <typename T, typename U>
inline void DataCopy(const LocalTensor<T> &dst, const LocalTensor<U> &src, uint32_t count) {
    (void)dst; (void)src; (void)count;
}

template <typename T, typename U, typename P>
inline void DataCopy(const LocalTensor<T> &dst, const GlobalTensor<U> &src, const P &params) {
    (void)dst; (void)src; (void)params;
}

template <typename T, typename U, typename P>
inline void DataCopy(const GlobalTensor<T> &dst, const LocalTensor<U> &src, const P &params) {
    (void)dst; (void)src; (void)params;
}

template <typename T, typename U, typename P>
inline void DataCopy(const LocalTensor<T> &dst, const LocalTensor<U> &src, const P &params) {
    (void)dst; (void)src; (void)params;
}

template <typename T, typename U, typename P>
inline void LoadData(const LocalTensor<T> &dst, const LocalTensor<U> &src, const P &params) {
    (void)dst; (void)src; (void)params;
}

template <typename T, typename U, typename S, typename P, typename MP>
inline void LoadData(const LocalTensor<T> &dst, const LocalTensor<U> &src, const LocalTensor<S> &scale, const P &params, const MP &mxParams) {
    (void)dst; (void)src; (void)scale; (void)params; (void)mxParams;
}

template <typename T, typename U, typename P>
inline void Fixpipe(const LocalTensor<T> &dst, const LocalTensor<U> &src, const P &params) {
    (void)dst; (void)src; (void)params;
}

template <typename DstT, typename SrcT, const auto &Cfg, typename DstTensor, typename SrcTensor, typename P>
inline void Fixpipe(const DstTensor &dst, const SrcTensor &src, const P &params) {
    (void)dst; (void)src; (void)params;
}

template <typename T, typename U, typename W, typename P>
inline void Fixpipe(const LocalTensor<T> &dst, const LocalTensor<U> &src, const LocalTensor<W> &ws, const P &params) {
    (void)dst; (void)src; (void)ws; (void)params;
}

namespace MicroAPI {

using float8_e4m3_t = ::float8_e4m3_t;

enum class MaskPattern {
    ALL = 0,
    NONE = 1,
};

struct MaskReg {
    uint64_t bits[4]{0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
    uint32_t activeCount{256};

    constexpr MaskReg() = default;

    // activeCount = 掩码有效 lane 数（按元素类型宽度换算，CreateMask<float, ALL> = 64）：
    // 镜像真机掩码语义的必要条件——只查 bits 不查 activeCount 会让 uint8 寄存器的掩码
    // 访存越过 activeCount 写满 256 lane，越槽踩踏相邻 UB 区域（实测 kOut1→VOut0 /
    // vOut1→KScale0 各 192B 越界写，conformance kv_prep 三方对账抓出）。
    inline bool IsActive(size_t idx) const {
        if (idx >= activeCount || idx >= 256) return false;
        return (bits[idx / 64] & (1ULL << (idx % 64))) != 0;
    }
};

template <typename T, MaskPattern Pattern = MaskPattern::ALL>
inline MaskReg CreateMask() {
    MaskReg m;
    if constexpr (Pattern == MaskPattern::ALL) {
        m.bits[0] = m.bits[1] = m.bits[2] = m.bits[3] = 0xFFFFFFFFFFFFFFFFULL;
        m.activeCount = static_cast<uint32_t>(sizeof(T) > 0 ? (256 / sizeof(T)) : 64);
    } else {
        m.bits[0] = m.bits[1] = m.bits[2] = m.bits[3] = 0;
        m.activeCount = 0;
    }
    return m;
}

template <typename T>
inline MaskReg UpdateMask(uint32_t elementNum) {
    MaskReg m;
    m.bits[0] = m.bits[1] = m.bits[2] = m.bits[3] = 0;
    m.activeCount = elementNum;
    for (uint32_t i = 0; i < elementNum && i < 256; ++i) {
        m.bits[i / 64] |= (1ULL << (i % 64));
    }
    return m;
}

enum class LoadDist {
    DIST_NORM_B32 = 0,
    DIST_BRC_B32 = 1,
    DIST_DINTLV_B32 = 2,
    DIST_E2B_B32 = 3,
};

enum class StoreDist {
    DIST_NORM_B32 = 0,
    DIST_PACK4_B32 = 1,
    DIST_FIRST_ELEMENT_B32 = 2,
};

enum class ReduceType {
    MAX = 0,
    MIN = 1,
    SUM = 2,
};

enum class CompareOp {
    LT = 0,
    LE = 1,
    GT = 2,
    GE = 3,
    EQ = 4,
    NE = 5,
};

enum class MaskMergeMode {
    ZEROING = 0,
    MERGING = 1,
};

enum class RegLayout {
    ZERO = 0,
    ONE = 1,
};

enum class SatMode {
    NO_SAT = 0,
    SAT = 1,
    UNKNOWN = 2,
};

enum class DataCopyMode {
    DATA_BLOCK_COPY = 0,
};

enum class PostLiteral {
    POST_MODE_UPDATE = 0,
};

// 舍入模式替身（与真机 AscendC::RoundMode 对位的桩轨枚举；数值语义由 Cast 的实现分支承载，
// 桩轨唯一消费形态 = CAST_RINT（RNE）与 UNKNOWN（无数值含义的宽化））。
enum class RoundMode : int32_t {
    UNKNOWN = 0,
    CAST_RINT = 1,
};

struct CastTrait {
    RegLayout layout{RegLayout::ZERO};
    SatMode sat{SatMode::NO_SAT};
    MaskMergeMode merge{MaskMergeMode::ZEROING};
    RoundMode round{RoundMode::UNKNOWN};
};

template <typename T>
class RegTensor {
public:
    static constexpr size_t kCapacity = (sizeof(T) > 0) ? (256 / sizeof(T)) : 64;
    alignas(64) T data_[kCapacity]{};

    constexpr RegTensor() = default;
    inline T &operator[](size_t idx) { return data_[idx]; }
    inline const T &operator[](size_t idx) const { return data_[idx]; }
    inline T *data() { return data_; }
    inline const T *data() const { return data_; }
    static constexpr size_t size() { return kCapacity; }
};

template <typename T, LoadDist Dist = LoadDist::DIST_NORM_B32, typename SrcPtrT = T>
inline void LoadAlign(RegTensor<T> &dst, const SrcPtrT *src) {
    if (!src) return;
    if constexpr (Dist == LoadDist::DIST_BRC_B32) {
        T val = static_cast<T>(*src);
        for (size_t i = 0; i < dst.kCapacity; ++i) {
            dst[i] = val;
        }
    } else {
        for (size_t i = 0; i < dst.kCapacity; ++i) {
            dst[i] = static_cast<T>(src[i]);
        }
    }
}

template <typename T, LoadDist Dist = LoadDist::DIST_DINTLV_B32, typename SrcPtrT = T>
inline void LoadAlign(RegTensor<T> &dstEven, RegTensor<T> &dstOdd, const SrcPtrT *src) {
    if (!src) return;
    for (size_t i = 0; i < dstEven.kCapacity; ++i) {
        dstEven[i] = static_cast<T>(src[2 * i]);
        dstOdd[i] = static_cast<T>(src[2 * i + 1]);
    }
}

template <typename T, StoreDist Dist = StoreDist::DIST_NORM_B32, typename DstPtrT = T>
inline void StoreAlign(DstPtrT *dst, const RegTensor<T> &src, MaskReg mask) {
    if (!dst) return;
    if constexpr (Dist == StoreDist::DIST_FIRST_ELEMENT_B32) {
        dst[0] = static_cast<DstPtrT>(src[0]);
    } else {
        for (size_t i = 0; i < src.kCapacity; ++i) {
            if (mask.IsActive(i)) {
                dst[i] = static_cast<DstPtrT>(src[i]);
            }
        }
    }
}

template <typename T, auto Mode, auto Post, typename DstPtrT>
inline void StoreAlign(DstPtrT *&dst, const RegTensor<T> &src, uint32_t stride, uint32_t count, MaskReg mask) {
    (void)stride; (void)count;
    StoreAlign<T, StoreDist::DIST_NORM_B32, DstPtrT>(dst, src, mask);
}

template <typename T, typename ScalarT>
inline void Duplicate(RegTensor<T> &dst, ScalarT scalar, MaskReg mask = MaskReg()) {
    T val = static_cast<T>(scalar);
    for (size_t i = 0; i < dst.kCapacity; ++i) {
        if (mask.IsActive(i)) {
            dst[i] = val;
        }
    }
}

template <typename T>
inline void Duplicate(RegTensor<T> &dst, const RegTensor<T> &src, MaskReg mask = MaskReg()) {
    T val = src[0];
    for (size_t i = 0; i < dst.kCapacity; ++i) {
        if (mask.IsActive(i)) {
            dst[i] = val;
        }
    }
}

template <typename T>
inline void Mul(RegTensor<T> &dst, const RegTensor<T> &src1, const RegTensor<T> &src2, MaskReg mask) {
    for (size_t i = 0; i < dst.kCapacity; ++i) {
        if (mask.IsActive(i)) {
            dst[i] = static_cast<T>(src1[i] * src2[i]);
        }
    }
}

template <typename T, typename ScalarT>
inline void Muls(RegTensor<T> &dst, const RegTensor<T> &src, ScalarT scalar, MaskReg mask) {
    T s = static_cast<T>(scalar);
    for (size_t i = 0; i < dst.kCapacity; ++i) {
        if (mask.IsActive(i)) {
            dst[i] = static_cast<T>(src[i] * s);
        }
    }
}

template <typename T>
inline void Add(RegTensor<T> &dst, const RegTensor<T> &src1, const RegTensor<T> &src2, MaskReg mask) {
    for (size_t i = 0; i < dst.kCapacity; ++i) {
        if (mask.IsActive(i)) {
            dst[i] = static_cast<T>(src1[i] + src2[i]);
        }
    }
}

template <typename T, typename ScalarT>
inline void Adds(RegTensor<T> &dst, const RegTensor<T> &src, ScalarT scalar, MaskReg mask) {
    T s = static_cast<T>(scalar);
    for (size_t i = 0; i < dst.kCapacity; ++i) {
        if (mask.IsActive(i)) {
            dst[i] = static_cast<T>(src[i] + s);
        }
    }
}

template <typename T>
inline void Sub(RegTensor<T> &dst, const RegTensor<T> &src1, const RegTensor<T> &src2, MaskReg mask) {
    for (size_t i = 0; i < dst.kCapacity; ++i) {
        if (mask.IsActive(i)) {
            dst[i] = static_cast<T>(src1[i] - src2[i]);
        }
    }
}

template <typename T>
inline void Max(RegTensor<T> &dst, const RegTensor<T> &src1, const RegTensor<T> &src2, MaskReg mask) {
    for (size_t i = 0; i < dst.kCapacity; ++i) {
        if (mask.IsActive(i)) {
            dst[i] = (src1[i] > src2[i]) ? src1[i] : src2[i];
        }
    }
}

template <typename T>
inline void Min(RegTensor<T> &dst, const RegTensor<T> &src1, const RegTensor<T> &src2, MaskReg mask) {
    for (size_t i = 0; i < dst.kCapacity; ++i) {
        if (mask.IsActive(i)) {
            dst[i] = (src1[i] < src2[i]) ? src1[i] : src2[i];
        }
    }
}

template <typename T>
inline void Select(RegTensor<T> &dst, const RegTensor<T> &src1, const RegTensor<T> &src2, MaskReg mask) {
    for (size_t i = 0; i < dst.kCapacity; ++i) {
        dst[i] = mask.IsActive(i) ? src1[i] : src2[i];
    }
}

template <typename T>
inline void Select(RegTensor<T> &dst, MaskReg mask, const RegTensor<T> &src1, const RegTensor<T> &src2) {
    for (size_t i = 0; i < dst.kCapacity; ++i) {
        dst[i] = mask.IsActive(i) ? src1[i] : src2[i];
    }
}

template <typename T, typename ScalarT>
inline void CompareScalar(MaskReg &dst, const RegTensor<T> &src, ScalarT scalar, CompareOp op) {
    dst.bits[0] = dst.bits[1] = dst.bits[2] = dst.bits[3] = 0;
    T s = static_cast<T>(scalar);
    for (size_t i = 0; i < src.kCapacity; ++i) {
        bool cond = false;
        switch (op) {
            case CompareOp::LT: cond = (src[i] < s); break;
            case CompareOp::LE: cond = (src[i] <= s); break;
            case CompareOp::GT: cond = (src[i] > s); break;
            case CompareOp::GE: cond = (src[i] >= s); break;
            case CompareOp::EQ: cond = (src[i] == s); break;
            case CompareOp::NE: cond = (src[i] != s); break;
        }
        if (cond) {
            dst.bits[i / 64] |= (1ULL << (i % 64));
        }
    }
}

template <typename T>
inline void ExpSub(RegTensor<float> &dst, const RegTensor<float> &src, const RegTensor<float> &m, MaskReg mask) {
    for (size_t i = 0; i < dst.kCapacity; ++i) {
        if (mask.IsActive(i)) {
            dst[i] = std::exp(src[i] - m[i]);
        }
    }
}

inline void ExpSub(RegTensor<float> &dst, const RegTensor<float> &src, float mScalar, MaskReg mask) {
    for (size_t i = 0; i < dst.kCapacity; ++i) {
        if (mask.IsActive(i)) {
            dst[i] = std::exp(src[i] - mScalar);
        }
    }
}

inline void Ln(RegTensor<float> &dst, const RegTensor<float> &src, MaskReg mask) {
    for (size_t i = 0; i < dst.kCapacity; ++i) {
        if (mask.IsActive(i)) {
            dst[i] = (src[i] > 0.0f) ? std::log(src[i]) : -INFINITY;
        }
    }
}

template <ReduceType RType, typename T = float, typename SrcT = float, MaskMergeMode Mode = MaskMergeMode::ZEROING>
inline void Reduce(RegTensor<T> &dst, const RegTensor<SrcT> &src, MaskReg mask) {
    if constexpr (RType == ReduceType::MAX) {
        SrcT maxVal = -1e38f;
        bool first = true;
        for (size_t i = 0; i < src.kCapacity; ++i) {
            if (mask.IsActive(i)) {
                if (first || src[i] > maxVal) {
                    maxVal = src[i];
                    first = false;
                }
            }
        }
        dst[0] = static_cast<T>(maxVal);
    } else if constexpr (RType == ReduceType::SUM) {
        double sum = 0.0;
        for (size_t i = 0; i < src.kCapacity; ++i) {
            if (mask.IsActive(i)) {
                sum += static_cast<double>(src[i]);
            }
        }
        dst[0] = static_cast<T>(sum);
    }
}

inline float ReduceMax(const RegTensor<float> &src, MaskReg mask) {
    float maxVal = -1e38f;
    bool first = true;
    for (size_t i = 0; i < src.kCapacity; ++i) {
        if (mask.IsActive(i)) {
            if (first || src[i] > maxVal) {
                maxVal = src[i];
                first = false;
            }
        }
    }
    return maxVal;
}

inline float ReduceSum(const RegTensor<float> &src, MaskReg mask) {
    double sum = 0.0;
    for (size_t i = 0; i < src.kCapacity; ++i) {
        if (mask.IsActive(i)) {
            sum += static_cast<double>(src[i]);
        }
    }
    return static_cast<float>(sum);
}

inline void ReduceMax(RegTensor<float> &dst, const RegTensor<float> &src, MaskReg mask) {
    Reduce<ReduceType::MAX>(dst, src, mask);
}

inline void ReduceSum(RegTensor<float> &dst, const RegTensor<float> &src, MaskReg mask) {
    Reduce<ReduceType::SUM>(dst, src, mask);
}


template <typename DstT, typename SrcT>
inline void Cast(RegTensor<DstT> &dst, const RegTensor<SrcT> &src, MaskReg mask) {
    for (size_t i = 0; i < dst.kCapacity && i < src.kCapacity; ++i) {
        if (mask.IsActive(i)) {
            if constexpr (std::is_same_v<DstT, float8_e4m3_t> && std::is_same_v<SrcT, float>) {
                dst[i] = static_cast<float8_e4m3_t>(sa2::FloatToFp8E4M3(src[i]));
            } else if constexpr (std::is_same_v<DstT, float> && std::is_same_v<SrcT, float8_e4m3_t>) {
                dst[i] = sa2::Fp8E4M3ToFloat(static_cast<uint8_t>(src[i]));
            } else if constexpr (std::is_same_v<DstT, half> && std::is_same_v<SrcT, float>) {
                dst[i] = half(src[i]);
            } else if constexpr (std::is_same_v<DstT, bfloat16_t> && std::is_same_v<SrcT, float>) {
                dst[i] = sa2::FloatToBf16(src[i]);
            } else if constexpr (std::is_same_v<DstT, float> && std::is_same_v<SrcT, half>) {
                dst[i] = static_cast<float>(src[i]);
            } else if constexpr (std::is_same_v<DstT, float> && std::is_same_v<SrcT, bfloat16_t>) {
                // bf16 位型精确展开（指数域左移 16 位），同 sa2::Sa2ScalarToFloat(bfloat16_t)
                uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
                std::memcpy(&dst[i], &bits, sizeof(float));
            } else if constexpr (std::is_integral_v<DstT> && std::is_same_v<SrcT, float>) {
                // 设备 CAST_RINT 语义镜像：RNE 取整后收窄（sa2::aicNearbyInt 同源）
                dst[i] = static_cast<DstT>(sa2::aicNearbyInt(src[i]));
            } else {
                dst[i] = static_cast<DstT>(src[i]);
            }
        }
    }
}

template <typename DstT, typename SrcT, const auto &Trait>
inline void Cast(RegTensor<DstT> &dst, const RegTensor<SrcT> &src, MaskReg mask) {
    Cast<DstT, SrcT>(dst, src, mask);
}

template <typename T>
inline void DataCopy(RegTensor<T> &dst, const T *src) {
    LoadAlign(dst, src);
}

template <typename T, StoreDist Dist = StoreDist::DIST_NORM_B32>
inline void DataCopy(T *dst, const RegTensor<T> &src, MaskReg mask) {
    StoreAlign<T, Dist>(dst, src, mask);
}

template <typename T>
inline void And(RegTensor<T> &dst, const RegTensor<T> &src1, const RegTensor<T> &src2, MaskReg mask) {
    using IntType = std::conditional_t<sizeof(T) == 1, uint8_t,
                    std::conditional_t<sizeof(T) == 2, uint16_t,
                    std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>>;
    for (size_t i = 0; i < dst.kCapacity; ++i) {
        if (mask.IsActive(i)) {
            IntType u1 = 0, u2 = 0;
            std::memcpy(&u1, &src1[i], sizeof(T));
            std::memcpy(&u2, &src2[i], sizeof(T));
            IntType res = u1 & u2;
            std::memcpy(&dst[i], &res, sizeof(T));
        }
    }
}

template <typename T>
inline void Or(RegTensor<T> &dst, const RegTensor<T> &src1, const RegTensor<T> &src2, MaskReg mask) {
    using IntType = std::conditional_t<sizeof(T) == 1, uint8_t,
                    std::conditional_t<sizeof(T) == 2, uint16_t,
                    std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>>;
    for (size_t i = 0; i < dst.kCapacity; ++i) {
        if (mask.IsActive(i)) {
            IntType u1 = 0, u2 = 0;
            std::memcpy(&u1, &src1[i], sizeof(T));
            std::memcpy(&u2, &src2[i], sizeof(T));
            IntType res = u1 | u2;
            std::memcpy(&dst[i], &res, sizeof(T));
        }
    }
}

template <typename T>
inline void Div(RegTensor<T> &dst, const RegTensor<T> &src1, const RegTensor<T> &src2, MaskReg mask) {
    for (size_t i = 0; i < dst.kCapacity; ++i) {
        if (mask.IsActive(i)) {
            dst[i] = static_cast<T>(src1[i] / src2[i]);
        }
    }
}

struct UnalignReg {};
struct UnalignRegForStore {};

template <typename T, auto Post = PostLiteral::POST_MODE_UPDATE, typename DstPtrT>
inline void StoreUnAlign(DstPtrT *&dst, const RegTensor<T> &src, UnalignRegForStore &ureg, uint32_t count) {
    (void)ureg; (void)count;
    if (dst) *dst = static_cast<DstPtrT>(src[0]);
}

template <typename T, auto Post = PostLiteral::POST_MODE_UPDATE, typename DstPtrT>
inline void StoreUnAlignPost(DstPtrT *&dst, UnalignRegForStore &ureg, uint32_t count) {
    (void)dst; (void)ureg; (void)count;
}

}  // namespace MicroAPI

using float8_e4m3_t = ::float8_e4m3_t;
using RoundMode = MicroAPI::RoundMode;
namespace vapi = MicroAPI;

}  // namespace AscendC

namespace vapi = AscendC::MicroAPI;
namespace MicroAPI = AscendC::MicroAPI;

using AscendC::FixpipeParamsArch3510;
using AscendC::FixpipeParamsC310;

namespace Catlass {

namespace Arch {
struct Ascend950 {};

template <AscendC::TPosition POS>
using PositionType = std::integral_constant<AscendC::TPosition, POS>;

using PositionGM = PositionType<AscendC::TPosition::GM>;
using PositionL1 = PositionType<AscendC::TPosition::A1>;
using PositionL0A = PositionType<AscendC::TPosition::A2>;
using PositionL0B = PositionType<AscendC::TPosition::B2>;
using PositionL0C = PositionType<AscendC::TPosition::CO1>;
using PositionBias = PositionType<AscendC::TPosition::C2>;
using PositionUB = PositionType<AscendC::TPosition::VECCALC>;
} // namespace Arch

namespace layout {
struct zN {};
struct nZ {};
struct nN {};
struct zZ {};
struct L0C {};
struct RowMajor {};
struct ColumnMajor {};
} // namespace layout

namespace Gemm {
namespace Tile {

template <class ArchTag, class TensorSrc, class TensorDst, class Enable = void>
struct TileCopyTla {
    TileCopyTla() = default;

    template <class Dst, class Src>
    void operator()(const Dst &dst, const Src &src) const {
        (void)dst; (void)src;
    }

    template <class Dst, class Src, class Scale>
    void operator()(const Dst &dst, const Src &src, const Scale &scale) const {
        (void)dst; (void)src; (void)scale;
    }

    template <class Dst, class Src>
    void operator()(const Dst &dst, const Src &src, uint32_t batch) const {
        (void)dst; (void)src; (void)batch;
    }
};

template <class ArchTag, class ElementA_, class LayoutTagA_, class ElementB_, class LayoutTagB_,
          class ElementC_, class LayoutTagC_, class ElementBias = void, bool ReluEnable_ = false,
          int DEQUANT_GRANULARITY_ = 0, class L0CCopyMode = void>
struct PackedTileCopyTla {
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;
    using CopyL1ToL0A = TileCopyTla<ArchTag, void, void>;
    using CopyL1ToL0B = TileCopyTla<ArchTag, void, void>;
};

template <class ArchTag, class ElementA, class LayoutTagL1A>
struct TileMmadTla {
    TileMmadTla() = default;

    template <class TensorC, class TensorA, class TensorB>
    void operator()(const TensorC &c, const TensorA &a, const TensorB &b,
                    uint32_t m, uint32_t n, uint32_t k, bool initC = true, uint8_t unitFlag = 0) const {
        (void)c; (void)a; (void)b; (void)m; (void)n; (void)k; (void)initC; (void)unitFlag;
    }

    template <class TensorC, class TensorA, class TensorB, class TensorBias>
    void operator()(const TensorC &c, const TensorA &a, const TensorB &b, const TensorBias &bias,
                    uint32_t m, uint32_t n, uint32_t k, bool initC = true, uint8_t unitFlag = 0) const {
        (void)c; (void)a; (void)b; (void)bias; (void)m; (void)n; (void)k; (void)initC; (void)unitFlag;
    }
};

} // namespace Tile
} // namespace Gemm
} // namespace Catlass

namespace tla {

template <int64_t... Dims>
struct Coord {
    int64_t v[sizeof...(Dims) > 0 ? sizeof...(Dims) : 1] = {Dims...};
    constexpr Coord() = default;
    constexpr Coord(int64_t x, int64_t y) : v{x, y} {}
    constexpr Coord(int64_t x) : v{x} {}
};

struct Shape {
    int64_t d0 = 0, d1 = 0, d2 = 0;
    constexpr Shape() = default;
    constexpr Shape(int64_t a, int64_t b) : d0(a), d1(b) {}
    constexpr Shape(int64_t a, int64_t b, int64_t c) : d0(a), d1(b), d2(c) {}
};

struct Layout {
    Shape shape_{};
    Shape originShape_{};
    Shape stride_{};
    constexpr Layout() = default;
    constexpr Layout(Shape s) : shape_(s), originShape_(s), stride_(s) {}
    constexpr Layout(Shape s, Shape st, Shape o) : shape_(s), originShape_(o), stride_(st) {}
    constexpr const Shape &shape() const { return shape_; }
    constexpr const Shape &originShape() const { return originShape_; }
    constexpr const Shape &stride() const { return stride_; }
    constexpr int64_t operator()(Coord<0, 0> /*c*/) const { return 0; }
};

template <typename BuiltinTensor, typename Layout_ = Layout, typename Coord_ = Coord<0, 0>, AscendC::TPosition Pos = AscendC::TPosition::A1>
struct Tensor {
    using PrimType = typename BuiltinTensor::PrimType;
    using Element = PrimType;
    using Layout = Layout_;
    using Coord = Coord_;
    static constexpr AscendC::TPosition position = Pos;

    BuiltinTensor tensor_{};
    Layout layout_{};
    Coord coord_{};

    Tensor() = default;
    Tensor(BuiltinTensor const &bt, Layout const &l = {}, Coord const &c = {})
        : tensor_(bt), layout_(l), coord_(c) {}

    decltype(auto) data() const { return tensor_; }
    decltype(auto) data() { return tensor_; }
    const Layout &layout() const { return layout_; }
    const Coord &coord() const { return coord_; }
    const Shape &shape() const { return layout_.shape(); }
    const Shape &originShape() const { return layout_.originShape(); }
    const Shape &stride() const { return layout_.stride(); }
};

template <typename BuiltinTensor, typename Layout, typename PosType>
inline auto MakeTensor(BuiltinTensor const &bt, Layout const &l, PosType) {
    return Tensor<BuiltinTensor, Layout, Coord<0, 0>, PosType::value>(bt, l);
}

template <typename BuiltinTensor, typename Layout, typename Coord, typename PosType>
inline auto MakeTensor(BuiltinTensor const &bt, Layout const &l, Coord const &c, PosType) {
    return Tensor<BuiltinTensor, Layout, Coord, PosType::value>(bt, l, c);
}

inline auto MakeCoord(int64_t x, int64_t y) { return Coord<0, 0>{x, y}; }
inline auto MakeCoord(int64_t x) { return Coord<0>{x}; }
inline auto MakeShape(int64_t x, int64_t y) { return Shape{x, y}; }
inline auto MakeShape(int64_t x, int64_t y, int64_t z) { return Shape{x, y, z}; }

template <typename Element, typename LayoutTag>
inline Layout MakeLayout(int64_t r, int64_t c) {
    return Layout{Shape{r, c}};
}
inline Layout MakeLayoutL0C(int64_t r, int64_t c) {
    return Layout{Shape{r, c}};
}

template <typename Element, typename LayoutTag, bool isB = false>
inline Layout MakeMxScaleLayout(int64_t r, int64_t c) {
    return Layout{Shape{r, c}};
}

template <typename TensorT, typename CoordT, typename ShapeT>
inline auto GetTile(const TensorT &t, const CoordT &c, const ShapeT &s) {
    return Tensor<typename TensorT::PrimType, Layout, CoordT, TensorT::position>(
        t.data(), Layout{s}, c);
}

template <size_t I0, size_t I1 = 0>
inline int64_t get(const Shape &s) {
    if constexpr (I0 == 0 && I1 == 0) return s.d0;
    else if constexpr (I0 == 1 && I1 == 0) return s.d1;
    else if constexpr (I0 == 2 && I1 == 0) return s.d2;
    else if constexpr (I0 == 0 && I1 == 1) return 1;
    else if constexpr (I0 == 1 && I1 == 1) return 1;
    else return 1;
}

template <size_t I0, size_t I1 = 0>
inline int64_t get(const Coord<0, 0> &c) {
    if constexpr (I0 == 1 || I1 == 1) {
        return c.v[1];
    } else {
        return c.v[0];
    }
}

} // namespace tla

#ifndef g_coreType
#define g_coreType AscendC::AIV
#endif
#ifndef PIPE_ALL
#define PIPE_ALL 0
#endif
#ifndef PIPE_M
#define PIPE_M 1
#endif
#ifndef PIPE_V
#define PIPE_V 2
#endif
#ifndef PIPE_MTE1
#define PIPE_MTE1 3
#endif
#ifndef PIPE_MTE2
#define PIPE_MTE2 4
#endif
#ifndef PIPE_MTE3
#define PIPE_MTE3 5
#endif
#ifndef PIPE_FIX
#define PIPE_FIX 6
#endif

#ifndef KERNEL_TASK_TYPE_DEFAULT
#define KERNEL_TASK_TYPE_DEFAULT(x)
#endif
#ifndef REGISTER_TILING_DEFAULT
#define REGISTER_TILING_DEFAULT(x)
#endif
#ifndef GET_TILING_DATA_WITH_STRUCT
#define GET_TILING_DATA_WITH_STRUCT(type, var, ptr) type var = *reinterpret_cast<const type *>(ptr)
#endif
#ifndef ASCEND_IS_AIC
#define ASCEND_IS_AIC (g_coreType == AscendC::AIC)
#endif
#ifndef ASCEND_IS_AIV
#define ASCEND_IS_AIV (g_coreType == AscendC::AIV)
#endif
#ifndef TILING_KEY_VAR
#define TILING_KEY_VAR 0
#endif
