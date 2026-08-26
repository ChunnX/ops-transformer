## common层
### query拷贝
改造原有的拷贝函数，新增GmCoord定义
改造前：
```
struct GmCoord
{
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t gS1Idx;
    uint32_t dIdx;
    uint32_t gS1DealSize;
    uint32_t dDealSize;
};

template <typename Q_T, GmFormat GM_FORMAT, L1Format L1_FORMAT = L1Format::NZ>
class CopyQueryGmToL1 {
public:
    template <typename FaGmTensorType>
    __aicore__ inline void operator()(FaL1Tensor<Q_T, L1_FORMAT> &dstTensor, FaGmTensorType &srcTensor,
                                      GmCoord &gmCoord)
    {
        if constexpr ((GM_FORMAT == GmFormat::BSNGD) || (GM_FORMAT == GmFormat::TNGD)) {
            ProcessS1G(dstTensor, srcTensor, gmCoord);
        } else if constexpr (GM_FORMAT == GmFormat::BNGSD) {
            auto &offsetCalculator = srcTensor.offsetCalculator;
            if (offsetCalculator.actualSeqLensQParser.GetActualLenDims() != 0) {
                ProcessGS1(dstTensor, srcTensor, gmCoord);
            } else {
                ProcessContinuous(dstTensor, srcTensor, gmCoord);
            }
        } else if constexpr (GM_FORMAT == GmFormat::NGTD) {
            ProcessGS1(dstTensor, srcTensor, gmCoord);
        }
    }
    ...
}
```
改造后：
```
template <InnerMLayout = InnerMLayout::GS1_MERGE_LAYOUT>
struct GmCoord;

template <>
struct GmCoord<InnerMLayout::GS1_MERGE_LAYOUT>
{
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t gS1Idx;
    uint32_t dIdx;
    uint32_t gS1DealSize;
    uint32_t dDealSize;
};

template <>
struct GmCoord<InnerMLayout::S1_ONLY_LAYOUT>
{
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t gIdx;       // 新增gIdx、s1Idx，删除gS1Idx，保证入参原子性
    uint32_t s1Idx;
    uint32_t dIdx;
    uint32_t s1DealSize;  // gS1DealSize改为s1DealSize，代表拷贝行数
    uint32_t dDealSize;
};

template <typename Q_T, GmFormat GM_FORMAT, L1Format L1_FORMAT = L1Format::NZ, InnerMLayout M_LAYOUT = InnerMLayout::GS1_MERGE_LAYOUT>
class CopyQueryGmToL1 {};

template <typename Q_T, GmFormat GM_FORMAT, L1Format::NZ>
class CopyQueryGmToL1<typename Q_T, GmFormat GM_FORMAT, L1Format::NZ, InnerMLayout::GS1_MERGE_LAYOUT> {
public:
    template <typename FaGmTensorType>
    __aicore__ inline void operator()(FaL1Tensor<Q_T, L1_FORMAT> &dstTensor, FaGmTensorType &srcTensor,
                                      GmCoord<InnerMLayout::GS1_MERGE_LAYOUT> &gmCoord)
    {
        if constexpr ((GM_FORMAT == GmFormat::BSNGD) || (GM_FORMAT == GmFormat::TNGD)) {
            ProcessS1G(dstTensor, srcTensor, gmCoord);
        } else if constexpr (GM_FORMAT == GmFormat::BNGSD) {
            auto &offsetCalculator = srcTensor.offsetCalculator;
            if (offsetCalculator.actualSeqLensQParser.GetActualLenDims() != 0) {
                ProcessGS1(dstTensor, srcTensor, gmCoord);
            } else {
                ProcessContinuous(dstTensor, srcTensor, gmCoord);
            }
        } else if constexpr (GM_FORMAT == GmFormat::NGTD) {
            ProcessGS1(dstTensor, srcTensor, gmCoord);
        }
    }
    ...
}

template <typename Q_T, GmFormat GM_FORMAT, L1Format::NZ>
class CopyQueryGmToL1<typename Q_T, GmFormat GM_FORMAT, L1Format::NZ, InnerMLayout::S1_ONLY_LAYOUT> {
public:
    template <typename FaGmTensorType>
    __aicore__ inline void operator()(FaL1Tensor<Q_T, L1_FORMAT> &dstTensor, FaGmTensorType &srcTensor,
                                      GmCoord<InnerMLayout::S1_ONLY_LAYOUT> &gmCoord)
    {
        // 新开发代码，实现不合轴的拷贝
        ...
    }
    ...
}
```
### FA/FD attenOut拷贝
改造原有的拷贝函数，新增UB_FORMAT代表是否合轴
改造前：
```
enum class UbFormat
{
    GS1 = 0,
    S1G = 1
};

template <typename OUT_T, GmFormat GM_FORMAT, UbFormat UB_FORMAT>
class CopyAttenOutUbToGm
{
public:
    template <typename FaGmTensorType>
    __aicore__ inline void operator()(FaGmTensorType &dstTensor,
                                      FaUbTensor<OUT_T> &srcTensor,
                                      GmCoord &gmCoord)
    {
        if constexpr (UB_FORMAT == UbFormat::GS1) {
            ...
        } else if constexpr (UB_FORMAT == UbFormat::S1G) {
            ...
        }
    }
}
```
改造后：
```
enum class UbFormat
{
    GS1 = 0,
    S1G = 1,
    S1_ONLY     // 新增不合轴的format
};

template <typename OUT_T, GmFormat GM_FORMAT, UbFormat UB_FORMAT>
class CopyAttenOutUbToGm {};

template <typename OUT_T, GmFormat GM_FORMAT, UbFormat UB_FORMAT>
class CopyAttenOutUbToGm<typename OUT_T, GmFormat GM_FORMAT, UbFormat::GS1>
{
public:
    template <typename FaGmTensorType>
    __aicore__ inline void operator()(FaGmTensorType &dstTensor,
                                      FaUbTensor<OUT_T> &srcTensor,
                                      GmCoord<InnerMLayout::GS1_MERGE_LAYOUT> &gmCoord)
    {
        // 原始GS1实现
        ...
    }
}

template <typename OUT_T, GmFormat GM_FORMAT, UbFormat UB_FORMAT>
class CopyAttenOutUbToGm<typename OUT_T, GmFormat GM_FORMAT, UbFormat::S1G>
{
public:
    template <typename FaGmTensorType>
    __aicore__ inline void operator()(FaGmTensorType &dstTensor,
                                      FaUbTensor<OUT_T> &srcTensor,
                                      GmCoord<InnerMLayout::GS1_MERGE_LAYOUT> &gmCoord)
    {
        // 原始S1G实现
        ...
    }
}

template <typename OUT_T, GmFormat GM_FORMAT, UbFormat UB_FORMAT>
class CopyAttenOutUbToGm<typename OUT_T, GmFormat GM_FORMAT, UbFormat::S1_ONLY>
{
public:
    template <typename FaGmTensorType>
    __aicore__ inline void operator()(FaGmTensorType &dstTensor,
                                      FaUbTensor<OUT_T> &srcTensor,
                                      GmCoord<InnerMLayout::S1_ONLY_LAYOUT> &gmCoord)
    {
        // 新开发代码，实现不合轴的拷贝
        ...
    }
}
```