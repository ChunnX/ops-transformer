#include <string>
#include "common/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "common/tiling_base.h"
#include "common/tiling_templates_registry.h"
#include "avg_pool_3d_grad_tiling.h"

namespace {
constexpr int32_t kAvgPool3DGradDimLimit = 6;
constexpr int32_t kAvgPool3DGradDedyInputIdx = 1;
} 

namespace optiling {
constexpr int64_t GRAD_SHAPE = 5;
constexpr int64_t ATTR_SIZE = 3;
constexpr int64_t SCALR_UB_SIZE = 20480;
constexpr int64_t BF16_BYTE_SIZE = 2;
constexpr int64_t BF16_UB_PART = 3;
constexpr int64_t BF16_UB_PART_NO_OVERLAP = 4;
constexpr int64_t FP32_UB_PART = 2;
constexpr int64_t D_DIM = 0;
constexpr int64_t H_DIM = 1;
constexpr int64_t W_DIM = 2;
constexpr int64_t W_DIM_OFFSET = 1;
constexpr int64_t H_DIM_OFFSET = 2;
constexpr int64_t D_DIM_OFFSET = 3;
constexpr int64_t N_DIM_OFFSET = 5;
constexpr int64_t COUNT_IDX = 4;
constexpr int64_t DIVISOR_IDX = 5;
constexpr int64_t FORMAT_IDX = 6;
constexpr int64_t ALIGN_BYTES = 32;
constexpr int64_t TILINGKEY_SCATTER = 6000;
constexpr int64_t NDHWC_N_DIM = 0;
constexpr int64_t NDHWC_D_DIM = 1;
constexpr int64_t NCDHW_D_DIM = 2;
constexpr size_t ORIG_INPUT_SHAPE_INDEX = 0;
constexpr size_t COMPATIABLE_PAD_DIM = 6;
constexpr size_t DIM0 = 0;
constexpr size_t DIM2 = 2;
constexpr size_t DIM4 = 4;
constexpr uint64_t DTYPE_LEN_B16 = 2;
constexpr uint64_t DTYPE_LEN_B32 = 4;
constexpr uint64_t FRACTOR_TWO = 2;
constexpr int64_t BATCH_MODE = 1;

struct DHWParam {
    int64_t n = 0;
    int64_t d = 0;
    int64_t h = 0;
    int64_t w = 0;
};

template <typename T>
auto max(T a, T b) -> T
{
    return a >= b ? a : b;
}

class AvgPool3dGradTiling {
public:
    explicit AvgPool3dGradTiling(gert::TilingContext* context) : tilingContext_(context)
    {}
    ge::graphStatus Init();
    ge::graphStatus SetKernelTiling();
    void TilingDataPrint() const;

private:
    inline ge::graphStatus InitDHW();
    inline void Tiling4CParam(
        const uint64_t ubSizePlatform, const int64_t cShape, const int64_t ndhwShape, const ge::DataType& dtype);
    inline void Tiling4HWParam(const uint64_t ubSizePlatform, const ge::DataType& dtype);
    inline void Tiling4CastCopyOut(const int64_t ubSizePlatform, const int64_t ncShape, const int64_t dhwShape);
    inline void SetTilingKey(const ge::DataType& dtype);
    inline int64_t AlignUp(int64_t num, int64_t rnd) const;
    inline int64_t AlignDown(int64_t num, int64_t rnd);
    inline void Tiling4Block(const uint64_t ubSizePlatform, const ge::DataType& dtype);
    inline void BasicBlockSizeGet(const uint64_t ubSizePlatform, const ge::DataType& dtype);
    inline void BasicBlockCore();
    inline void BasicBlockUb(const uint64_t ubSizePlatform, const ge::DataType& dtype);
    inline uint64_t BasicBlockComputerUbSize(
        const uint64_t baseDo, const uint64_t baseHo, const uint64_t baseWo, const uint64_t baseDi,
        const uint64_t baseHi, const uint64_t baseWi, const ge::DataType& dtype);

    ge::graphStatus InitBaseParams(gert::Shape& gradShape, ge::DataType& dtype);
    void CalcShapeVars(const gert::Shape& gradShape, int64_t& ncShape, int64_t& ndhwShape, int64_t& cShape);
    void SetWorkspace(ge::DataType dtype, int64_t ncShape);
    void CalcCoreNumAndDetermineFlag(uint32_t coreNum, int64_t ncShape, int64_t ndhwShape, int64_t gradDim0);
    void DispatchTilingStrategy(uint64_t ubSizePlatform, ge::DataType dtype, 
                                                 int64_t cShape, int64_t ndhwShape);
private:
    gert::TilingContext* tilingContext_ = nullptr;
    AvgPool3dGradTilingParam tilingData_;
    uint64_t coreNum_ = 0;
    uint64_t isDetermine_ = 0;
    uint64_t countIncludePad_ = 0;
    int64_t divisorOverride_ = 0;
    uint64_t isOverlap_ = 0;
    uint64_t isOnlyT_ = 0;
    uint64_t N_ = 0;
    uint64_t C_ = 0;
    std::string dataFormat_;
    DHWParam outDHW_;
    DHWParam inDHW_;
    DHWParam kDHW_;
    DHWParam dDHW_;
    DHWParam padDHW_;

private:
    uint64_t normalCoreCNum_ = 0;
    uint64_t lastCoreCNum_ = 0;
    uint64_t cAlign_ = 0;
    uint64_t cTotal_ = 0;
    uint64_t cNum_ = 0;
    uint64_t cCount_ = 0;
    uint64_t cTail_ = 0;
    uint64_t cLine_ = 0;

private:
    uint64_t normalCoreHWNum_ = 0;
    uint64_t lastCoreHWNum_ = 0;
    uint64_t hwAlign_ = 0;
    uint64_t hwTotal_ = 0;
    uint64_t hwNum_ = 0;
    uint64_t hwCount_ = 0;
    uint64_t hwTail_ = 0;
    uint64_t hwLine_ = 0;

private:
    uint64_t maxDataNumInUb_ = 0;
    uint64_t normalCoreNum_ = 0;
    uint64_t tailCoreNum_ = 0;
    uint64_t normalCoreDataNum_ = 0;
    uint64_t tailCoreDataNum_ = 0;
    uint64_t normalCoreFormerCopyTime_ = 0;
    uint64_t normalCoreTailCopyTime_ = 0;
    uint64_t normalCoreFormerDataNum_ = 0;
    uint64_t normalCoreTailDataNum_ = 0;
    uint64_t tailCoreFormerCopyTime_ = 0;
    uint64_t tailCoreTailCopyTime_ = 0;
    uint64_t tailCoreFormerDataNum_ = 0;
    uint64_t tailCoreTailDataNum_ = 0;

private:
    uint64_t singleCoreNc = 64;
    uint64_t singleCoreDo = 0;
    uint64_t singleCoreHo = 0;
    uint64_t singleCoreWo = 0;
    uint64_t ncCnt = 0;
    uint64_t doCnt = 0;
    uint64_t hoCnt = 0;
    uint64_t woCnt = 0;
    uint64_t totalCnt = 0;
    uint64_t ncTailData = 0;
    uint64_t doTailData = 0;
    uint64_t hoTailData = 0;
    uint64_t woTailData = 0;
    uint64_t baseDo_ = 0;
    uint64_t baseHo_ = 0;
    uint64_t baseWo_ = 0;
    uint64_t baseDi_ = 0;
    uint64_t baseHi_ = 0;
    uint64_t baseWi_ = 0;
    uint64_t isScatter = 0;
    uint64_t ubSize = 0;
};

int64_t AvgPool3dGradTiling::AlignDown(int64_t num, int64_t rnd)
{
    return ((rnd) == 0 ? 0 : ((num / rnd) * rnd));
}

int64_t AvgPool3dGradTiling::AlignUp(int64_t num, int64_t rnd) const
{
    return ((rnd) == 0 ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

ge::graphStatus AvgPool3dGradTiling::InitDHW()
{
    auto inputShape1 = tilingContext_->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape1);
    auto gradShape = inputShape1->GetStorageShape();
    auto attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);
    auto kSize = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    auto pads = attrs->GetAttrPointer<gert::ContinuousVector>(2);
    if (kSize->GetSize() != ATTR_SIZE || strides->GetSize() != ATTR_SIZE ||
        (pads->GetSize() != ATTR_SIZE && pads->GetSize() != COMPATIABLE_PAD_DIM)) {
        OP_LOGE(
            tilingContext_->GetNodeName(),
            "kSize shape %lu"
            "strides shape %lu, pads shape %lu is not same with 3, or pads shape is not same with 6",
            kSize->GetSize(), strides->GetSize(), pads->GetSize());
        return ge::GRAPH_FAILED;
    }
    auto kSizeData = static_cast<const int64_t*>(kSize->GetData());
    auto stridesData = static_cast<const int64_t*>(strides->GetData());
    auto padsData = static_cast<const int64_t*>(pads->GetData());

    auto inputShape0 = tilingContext_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputShape0);
    auto shapeDim = inputShape0->GetStorageShape().GetDim(0);
    const gert::Tensor* shapeTensor = tilingContext_->GetInputTensor(ORIG_INPUT_SHAPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, shapeTensor);
    const uint32_t* shapeValue = shapeTensor->GetData<uint32_t>();
    if (shapeValue == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint64_t gradShapeDimD = (dataFormat_ == "NDHWC") ? NDHWC_D_DIM : NCDHW_D_DIM;
    inDHW_.n = shapeValue[shapeDim - N_DIM_OFFSET];
    inDHW_.d = shapeValue[shapeDim - D_DIM_OFFSET];
    inDHW_.h = shapeValue[shapeDim - H_DIM_OFFSET];
    inDHW_.w = shapeValue[shapeDim - W_DIM_OFFSET];
    outDHW_.n = gradShape.GetDim(NDHWC_N_DIM);
    outDHW_.d = gradShape.GetDim(gradShapeDimD);
    outDHW_.h = gradShape.GetDim(gradShapeDimD + H_DIM);
    outDHW_.w = gradShape.GetDim(gradShapeDimD + W_DIM);
    kDHW_.d = kSizeData[D_DIM];
    kDHW_.h = kSizeData[H_DIM];
    kDHW_.w = kSizeData[W_DIM];
    dDHW_.d = stridesData[D_DIM];
    dDHW_.h = stridesData[H_DIM];
    dDHW_.w = stridesData[W_DIM];

    if (pads->GetSize() != COMPATIABLE_PAD_DIM) {
        padDHW_.d = padsData[D_DIM];
        padDHW_.h = padsData[H_DIM];
        padDHW_.w = padsData[W_DIM];
    } else {
        padDHW_.d = padsData[DIM0];
        padDHW_.h = padsData[DIM2];
        padDHW_.w = padsData[DIM4];
    }
    isOverlap_ = static_cast<uint64_t>(dDHW_.d < kDHW_.d || dDHW_.h < kDHW_.h || dDHW_.w < kDHW_.w);
    isOnlyT_ = static_cast<uint64_t>((kDHW_.h == 1) && (dDHW_.h == 1) && (kDHW_.w == 1) && (dDHW_.w == 1));
    return ge::GRAPH_SUCCESS;
}

void AvgPool3dGradTiling::Tiling4CastCopyOut(
    const int64_t ubSizePlatform, const int64_t ncShape, const int64_t dhwShape)
{
    OP_LOGD(tilingContext_->GetNodeName(), "tiling for cast copy out start");
    auto gradShape = dhwShape * ncShape;
    auto ubSizeLeft = ubSizePlatform - SCALR_UB_SIZE;
    maxDataNumInUb_ = static_cast<uint64_t>(ubSizeLeft / BF16_BYTE_SIZE / BF16_UB_PART);
    normalCoreNum_ = static_cast<uint64_t>(gradShape) % coreNum_;
    tailCoreNum_ = coreNum_ - normalCoreNum_;
    normalCoreDataNum_ = (static_cast<uint64_t>(gradShape) + coreNum_ - 1UL) / coreNum_;
    tailCoreDataNum_ = static_cast<uint64_t>(gradShape) / coreNum_;

    auto normalCoreCopyTime = (normalCoreDataNum_ + maxDataNumInUb_ - 1UL) / maxDataNumInUb_;
    auto tailCoreCopyTime = (tailCoreDataNum_ + maxDataNumInUb_ - 1UL) / maxDataNumInUb_;

    normalCoreFormerCopyTime_ = normalCoreDataNum_ % normalCoreCopyTime;
    normalCoreTailCopyTime_ = normalCoreCopyTime - normalCoreFormerCopyTime_;
    normalCoreFormerDataNum_ = (normalCoreDataNum_ + normalCoreCopyTime - 1UL) / normalCoreCopyTime;
    normalCoreTailDataNum_ = normalCoreDataNum_ / normalCoreCopyTime;

    tailCoreFormerCopyTime_ = tailCoreDataNum_ % tailCoreCopyTime;
    tailCoreTailCopyTime_ = tailCoreCopyTime - tailCoreFormerCopyTime_;
    tailCoreFormerDataNum_ = (tailCoreDataNum_ + tailCoreCopyTime - 1UL) / tailCoreCopyTime;
    tailCoreTailDataNum_ = tailCoreDataNum_ / tailCoreCopyTime;
    OP_LOGD(tilingContext_->GetNodeName(), "tiling for cast copy out end");
}

void AvgPool3dGradTiling::Tiling4CParam(
    const uint64_t ubSizePlatform, const int64_t cShape, const int64_t ndhwShape, const ge::DataType& dtype)
{
    OP_LOGD(tilingContext_->GetNodeName(), "tiling for c param start");
    auto ubSizeLeft = ubSizePlatform - static_cast<uint64_t>(SCALR_UB_SIZE);
    auto ubPart = (dtype == ge::DT_FLOAT) ? FP32_UB_PART : ((isOverlap_ != 0) ? BF16_UB_PART : BF16_UB_PART_NO_OVERLAP);
    auto byteSize = GetSizeByDataType(dtype);
    auto ubSize4C = ubSizeLeft / ubPart;
    auto ubSize4CAlign = AlignDown(ubSize4C, ALIGN_BYTES);
    int64_t ubMaxCNum = ubSize4CAlign / byteSize;
    cTotal_ = static_cast<uint64_t>(cShape);
    if (cShape > ubMaxCNum) {
        cAlign_ = static_cast<uint64_t>(ubMaxCNum);
        cCount_ = static_cast<uint64_t>(ubMaxCNum);
        cNum_ = (static_cast<uint64_t>(cShape) + cCount_ - 1UL) / cCount_;
        cTail_ = static_cast<uint64_t>(cShape) - (cNum_ - 1UL) * cCount_;
        cLine_ = 1UL;
    } else {
        auto alignBlocks = ALIGN_BYTES / byteSize;
        cAlign_ = AlignUp(cShape, alignBlocks);
        if (cAlign_ == 0UL) {
            return;
        }
        cCount_ = static_cast<uint64_t>(cShape);
        cNum_ = 1UL;
        cTail_ = cCount_;
        cLine_ = static_cast<uint64_t>(ubMaxCNum) / cAlign_;
    }
    auto dividShape = static_cast<uint64_t>(ndhwShape) * cNum_;
    normalCoreCNum_ = dividShape / coreNum_;
    lastCoreCNum_ = dividShape - normalCoreCNum_ * (coreNum_ - 1UL);
    OP_LOGD(tilingContext_->GetNodeName(), "tiling for c param end");
}

void AvgPool3dGradTiling::Tiling4HWParam(const uint64_t ubSizePlatform, const ge::DataType& dtype)
{
    OP_LOGD(tilingContext_->GetNodeName(), "tiling for hw param start");
    auto ubSizeLeft = ubSizePlatform - static_cast<uint64_t>(SCALR_UB_SIZE);
    auto ubPart = (dtype == ge::DT_FLOAT) ? FP32_UB_PART : ((isOverlap_ != 0) ? BF16_UB_PART : BF16_UB_PART_NO_OVERLAP);
    auto byteSize = GetSizeByDataType(dtype);
    auto ubSize4HW = ubSizeLeft / ubPart;
    auto ubSize4HWAlign = AlignDown(ubSize4HW, ALIGN_BYTES);
    int64_t ubMaxHWNum = ubSize4HWAlign / byteSize;
    auto hwShape = outDHW_.h * outDHW_.w;
    hwTotal_ = static_cast<uint64_t>(hwShape);
    if (hwShape > ubMaxHWNum) {
        hwAlign_ = static_cast<uint64_t>(ubMaxHWNum);
        hwCount_ = static_cast<uint64_t>(ubMaxHWNum);
        hwNum_ = (static_cast<uint64_t>(hwShape) + hwCount_ - 1UL) / hwCount_;
        hwTail_ = static_cast<uint64_t>(hwShape) - (hwNum_ - 1UL) * hwCount_;
        hwLine_ = 1UL;
    } else {
        auto alignBlocks = ALIGN_BYTES / byteSize;
        hwAlign_ = AlignUp(hwShape, alignBlocks);
        if (hwAlign_ == 0UL) {
            return;
        }
        hwCount_ = static_cast<uint64_t>(hwShape);
        hwNum_ = 1UL;
        hwTail_ = hwCount_;
        hwLine_ = static_cast<uint64_t>(ubMaxHWNum) / hwAlign_;
    }
    auto dividShape = N_ * C_ * static_cast<uint64_t>(outDHW_.d) * hwNum_;
    normalCoreHWNum_ = dividShape / coreNum_;
    lastCoreHWNum_ = dividShape - normalCoreHWNum_ * (coreNum_ - 1UL);
    OP_LOGD(tilingContext_->GetNodeName(), "tiling for hw param end");
}

void AvgPool3dGradTiling::Tiling4Block(const uint64_t ubSizePlatform, const ge::DataType& dtype)
{
    OP_LOGD(tilingContext_->GetNodeName(), "tiling for block param start");
    BasicBlockSizeGet(ubSizePlatform, dtype); 
    BasicBlockCore();
    BasicBlockUb(ubSizePlatform, dtype); 
    OP_LOGD(tilingContext_->GetNodeName(), "tiling for block param end");
}

void AvgPool3dGradTiling::BasicBlockSizeGet(const uint64_t ubSizePlatform, const ge::DataType& dtype)
{
    const uint64_t ncDim = N_ * C_;
    const uint64_t doDim = static_cast<uint64_t>(outDHW_.d);
    const uint64_t hoDim = static_cast<uint64_t>(outDHW_.h);
    const uint64_t woDim = static_cast<uint64_t>(outDHW_.w);
    auto ubSizeLeft = static_cast<uint64_t>(ubSizePlatform) - static_cast<uint64_t>(SCALR_UB_SIZE);

    if (BasicBlockComputerUbSize(1, 1, 1, 1, 1, std::min(kDHW_.w, inDHW_.w), dtype) > ubSizeLeft) {
        isScatter = static_cast<uint64_t>(1);
    }
    uint64_t step = 1;
    auto tmpNc = singleCoreNc;
    while (BasicBlockComputerUbSize(1, 1, woDim, 1, 1, inDHW_.w, dtype) <= ubSizeLeft) {
        step++;
        tmpNc = singleCoreNc;
        singleCoreNc *= step;
    }
    singleCoreNc = tmpNc;

    ncCnt = (ncDim + singleCoreNc - 1UL) / singleCoreNc;
    uint64_t curCnt = ncCnt;
    if (curCnt >= coreNum_ || isDetermine_ == 1UL) { 
        singleCoreDo = doDim;
        singleCoreHo = hoDim;
        singleCoreWo = woDim;
        return;
    }

    uint64_t doCntNeed = (coreNum_ + ncCnt - 1UL) / ncCnt;
    singleCoreDo = (doDim + (doCntNeed - 1UL)) / doCntNeed;
    doCnt = (doDim + singleCoreDo - 1UL) / singleCoreDo;
    curCnt *= doCnt;
    if (curCnt >= coreNum_) {
        singleCoreHo = hoDim;
        singleCoreWo = woDim;
        return;
    }

    uint64_t hoCntNeed = (coreNum_ + curCnt - 1UL) / curCnt;
    singleCoreHo = (hoDim + (hoCntNeed - 1UL)) / hoCntNeed;
    hoCnt = (hoDim + singleCoreHo - 1UL) / singleCoreHo;
    curCnt *= hoCnt;

    if (curCnt >= coreNum_) {
        singleCoreWo = woDim;
        return;
    }
    uint64_t woCntNeed = (coreNum_ + curCnt - 1UL) / curCnt;
    singleCoreWo = (woDim + (woCntNeed - 1UL)) / woCntNeed;
    woCnt = (woDim + singleCoreWo - 1UL) / singleCoreWo;
    return;
}

void AvgPool3dGradTiling::BasicBlockCore()
{
    const uint64_t ncDim = N_ * C_;
    const uint64_t doDim = static_cast<uint64_t>(outDHW_.d);
    const uint64_t hoDim = static_cast<uint64_t>(outDHW_.h);
    const uint64_t woDim = static_cast<uint64_t>(outDHW_.w);
    cTotal_ = ncDim;
    ncCnt = (ncDim + singleCoreNc - 1UL) / singleCoreNc;
    doCnt = (doDim + singleCoreDo - 1UL) / singleCoreDo;
    hoCnt = (hoDim + singleCoreHo - 1UL) / singleCoreHo;
    woCnt = (woDim + singleCoreWo - 1UL) / singleCoreWo;

    ncTailData = ncDim - (ncCnt - 1UL) * singleCoreNc;
    doTailData = doDim - (doCnt - 1UL) * singleCoreDo;
    hoTailData = hoDim - (hoCnt - 1UL) * singleCoreHo;
    woTailData = woDim - (woCnt - 1UL) * singleCoreWo;

    totalCnt = ncCnt * doCnt * hoCnt * woCnt;
}

uint64_t AvgPool3dGradTiling::BasicBlockComputerUbSize(
    const uint64_t baseDo, const uint64_t baseHo, const uint64_t baseWo, const uint64_t baseDi, const uint64_t baseHi,
    const uint64_t baseWi, const ge::DataType& dtype)
{
    uint64_t baseDoHoWo = baseDo * baseHo * baseWo;
    uint64_t baseDiHiWi = baseDi * baseHi * baseWi;
    uint64_t baseDoHoWoAlign8 = static_cast<uint64_t>(AlignUp(static_cast<int64_t>(baseDoHoWo), 8));
    uint64_t baseDoHoWoAlign16 = static_cast<uint64_t>(AlignUp(static_cast<int64_t>(baseDoHoWo), 16));
    uint64_t baseDiHiWiAlign8 = static_cast<uint64_t>(AlignUp(static_cast<int64_t>(baseDiHiWi), 8));
    uint64_t baseDiHiWiAlign16 = static_cast<uint64_t>(AlignUp(static_cast<int64_t>(baseDiHiWi), 16));
    uint64_t outputGradUbSize = 0;
    uint64_t inputGradUbSize = 0;
    uint64_t TranUbSize = 0;
    uint64_t castUbSize = 0;

    if (dtype == ge::DT_FLOAT) {
        outputGradUbSize = static_cast<uint64_t>(
            max(uint64_t(baseDoHoWoAlign8), uint64_t(baseDiHiWiAlign8)) * singleCoreNc *
            DTYPE_LEN_B32); 
        inputGradUbSize =
            static_cast<uint64_t>(baseDiHiWiAlign8 * singleCoreNc * DTYPE_LEN_B32); 
        TranUbSize = static_cast<uint64_t>(baseDoHoWoAlign8 * singleCoreNc * DTYPE_LEN_B32); 
    } else if (dtype != ge::DT_FLOAT && isOverlap_) {
        outputGradUbSize = static_cast<uint64_t>(baseDoHoWoAlign16 * singleCoreNc * DTYPE_LEN_B16);
        TranUbSize = static_cast<uint64_t>(baseDoHoWoAlign16 * singleCoreNc * DTYPE_LEN_B16);
        castUbSize = static_cast<uint64_t>(FRACTOR_TWO * singleCoreNc * DTYPE_LEN_B32); 
        outputGradUbSize = static_cast<uint64_t>(
            max(uint64_t(outputGradUbSize), uint64_t(baseDiHiWiAlign8 * singleCoreNc * DTYPE_LEN_B32)));
        inputGradUbSize = static_cast<uint64_t>(baseDiHiWiAlign8 * singleCoreNc * DTYPE_LEN_B32);
    } else {
        outputGradUbSize = static_cast<uint64_t>(
            max(uint64_t(baseDoHoWoAlign16), uint64_t(baseDiHiWiAlign16)) * singleCoreNc * DTYPE_LEN_B16);
        TranUbSize = static_cast<uint64_t>(baseDoHoWoAlign16 * singleCoreNc * DTYPE_LEN_B16);
        castUbSize = static_cast<uint64_t>(FRACTOR_TWO * singleCoreNc * DTYPE_LEN_B32);
        inputGradUbSize = static_cast<uint64_t>(baseDiHiWiAlign16 * singleCoreNc * DTYPE_LEN_B16);
    }
    return outputGradUbSize + inputGradUbSize + TranUbSize + castUbSize;
}

void AvgPool3dGradTiling::BasicBlockUb(const uint64_t ubSizePlatform, const ge::DataType& dtype)
{
    auto ubSizeLeft = static_cast<uint64_t>(ubSizePlatform) - static_cast<uint64_t>(SCALR_UB_SIZE);
    ubSize = ubSizeLeft;
    baseDo_ = static_cast<uint64_t>(1);
    baseHo_ = static_cast<uint64_t>(1);
    baseWo_ = static_cast<uint64_t>(1);
    baseDi_ = static_cast<uint64_t>(1); 
    baseHi_ = static_cast<uint64_t>(1); 
    baseWi_ = static_cast<uint64_t>(1); 

    if (BasicBlockComputerUbSize(1, 1, 1, 1, 1, std::min(kDHW_.w, inDHW_.w), dtype) > ubSizeLeft) {
        isScatter = static_cast<uint64_t>(1);
        return;
    }
    
    for (uint64_t i = singleCoreWo; i > 0UL; i--) {
        if (BasicBlockComputerUbSize(
                baseDo_, baseHo_, i, baseDi_, baseHi_,
                std::min(static_cast<uint64_t>((i - 1) * dDHW_.w + kDHW_.w), static_cast<uint64_t>(inDHW_.w)),
                dtype) <= ubSizeLeft) {
            baseWo_ = i;
            baseWi_ = std::min(
                (i - 1UL) * static_cast<uint64_t>(dDHW_.w) + static_cast<uint64_t>(kDHW_.w),
                static_cast<uint64_t>(inDHW_.w));
            break;
        }
    }
}

void AvgPool3dGradTiling::SetTilingKey(const ge::DataType& dtype)
{
    uint64_t tilingKey = static_cast<uint64_t>(TILINGKEY_SCATTER);
    tilingContext_->SetTilingKey(tilingKey);
}

ge::graphStatus AvgPool3dGradTiling::Init()
{
    OP_LOGD(tilingContext_->GetNodeName(), "Tiling initing");
    auto platformInfo = tilingContext_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();

    gert::Shape gradShape;
    ge::DataType dtype;
    OP_CHECK_IF(InitBaseParams(gradShape, dtype) != ge::GRAPH_SUCCESS, 
                OP_LOGE(tilingContext_->GetNodeName(), "InitBaseParams failed"),
                return ge::GRAPH_FAILED);

    int64_t ndhwShape = 1;
    int64_t ncShape = 1;
    int64_t cShape = 1;
    CalcShapeVars(gradShape, ncShape, ndhwShape, cShape);

    auto ret = InitDHW();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    SetWorkspace(dtype, ncShape);
    CalcCoreNumAndDetermineFlag(coreNum, ncShape, ndhwShape, gradShape.GetDim(0));

    OP_CHECK_IF(coreNum_ == 0UL, 
                OP_LOGE(tilingContext_->GetNodeName(), "CoreNum is zero, error."),
                return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    uint64_t ubSizePlatform = static_cast<int64_t>(ubSizePlatForm);
    DispatchTilingStrategy(ubSizePlatform, dtype, cShape, ndhwShape);
    if (dtype == ge::DT_BF16 || dtype == ge::DT_FLOAT16) {
        int64_t inDhwShape = static_cast<int64_t>(inDHW_.d * inDHW_.h * inDHW_.w);
        Tiling4CastCopyOut(static_cast<int64_t>(ubSizePlatform), ncShape, inDhwShape);
    }

    SetTilingKey(dtype);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AvgPool3dGradTiling::InitBaseParams(gert::Shape& gradShape, ge::DataType& dtype)
{
    gradShape = tilingContext_->GetInputShape(1)->GetStorageShape();
    dtype = tilingContext_->GetInputDesc(1)->GetDataType();
    auto attrs = tilingContext_->GetAttrs();
    
    countIncludePad_ = static_cast<uint64_t>(*attrs->GetAttrPointer<bool>(COUNT_IDX));
    divisorOverride_ = static_cast<int64_t>(*attrs->GetAttrPointer<int>(DIVISOR_IDX));
    dataFormat_ = attrs->GetStr(FORMAT_IDX);

    if (dataFormat_ != "NDHWC" && dataFormat_ != "NCDHW") {
        OP_LOGE(tilingContext_->GetNodeName(), "invalid data_format, should be NCDHW or NDHWC");
        return ge::GRAPH_FAILED;
    }

    if (gradShape.GetDimNum() != GRAD_SHAPE) {
        OP_LOGE(tilingContext_->GetNodeName(), "gradShape dim num is not 5");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void AvgPool3dGradTiling::CalcShapeVars(const gert::Shape& gradShape, int64_t& ncShape, int64_t& ndhwShape, int64_t& cShape)
{
    N_ = gradShape.GetDim(0);
    C_ = gradShape.GetDim(1);
    ncShape = static_cast<int64_t>(N_ * C_);
}

void AvgPool3dGradTiling::SetWorkspace(ge::DataType dtype, int64_t ncShape)
{
    size_t sysWorkspaceSize = 16UL * 1024UL * 1024UL;
    size_t castWorkspaceSize = 0;
    
    if (dtype == ge::DT_BF16 || dtype == ge::DT_FLOAT16) {
        size_t totalElements = static_cast<size_t>(inDHW_.d) * static_cast<size_t>(inDHW_.h) *
                               static_cast<size_t>(inDHW_.w) * static_cast<size_t>(ncShape);
        castWorkspaceSize = totalElements * sizeof(float);
    }

    size_t* currentWorkSpace = tilingContext_->GetWorkspaceSizes(1);
    currentWorkSpace[0] = sysWorkspaceSize + castWorkspaceSize;
}

void AvgPool3dGradTiling::CalcCoreNumAndDetermineFlag(
    uint32_t coreNum, int64_t ncShape, int64_t ndhwShape, int64_t gradDim0)
{
    int64_t taskCount = ncShape * outDHW_.d * outDHW_.h * outDHW_.w;
    coreNum_ = std::min(coreNum, static_cast<uint32_t>(taskCount));

    bool isDeterministic = (tilingContext_->GetDeterministic() == 1);
    if (!isDeterministic || isOverlap_ != 1) {
        return;
    }

    isDetermine_ = 1UL;
}

void AvgPool3dGradTiling::DispatchTilingStrategy(uint64_t ubSizePlatform, ge::DataType dtype, 
                                                 int64_t cShape, int64_t ndhwShape)
{
    Tiling4Block(ubSizePlatform, dtype);
}


ge::graphStatus AvgPool3dGradTiling::SetKernelTiling()
{
    tilingContext_->SetBlockDim(coreNum_);
    tilingData_.attrParam.set_N(N_);
    tilingData_.attrParam.set_C(C_);
    tilingData_.attrParam.set_inN(inDHW_.n);
    tilingData_.attrParam.set_inD(inDHW_.d);
    tilingData_.attrParam.set_inH(inDHW_.h);
    tilingData_.attrParam.set_inW(inDHW_.w);
    tilingData_.attrParam.set_outN(outDHW_.n);
    tilingData_.attrParam.set_outD(outDHW_.d);
    tilingData_.attrParam.set_outH(outDHW_.h);
    tilingData_.attrParam.set_outW(outDHW_.w);
    tilingData_.attrParam.set_kD(kDHW_.d);
    tilingData_.attrParam.set_kH(kDHW_.h);
    tilingData_.attrParam.set_kW(kDHW_.w);
    tilingData_.attrParam.set_dD(dDHW_.d);
    tilingData_.attrParam.set_dH(dDHW_.h);
    tilingData_.attrParam.set_dW(dDHW_.w);
    tilingData_.attrParam.set_padD(padDHW_.d);
    tilingData_.attrParam.set_padH(padDHW_.h);
    tilingData_.attrParam.set_padW(padDHW_.w);
    tilingData_.attrParam.set_countIncludePad(countIncludePad_);
    tilingData_.attrParam.set_divisorOverride(divisorOverride_);
    tilingData_.attrParam.set_isOverLap(isOverlap_);
    tilingData_.attrParam.set_isDetermine(isDetermine_);

    tilingData_.cParam.set_normalCoreCNum(normalCoreCNum_);
    tilingData_.cParam.set_lastCoreCNum(lastCoreCNum_);
    tilingData_.cParam.set_cAlign(cAlign_);
    tilingData_.cParam.set_cTotal(cTotal_);
    tilingData_.cParam.set_cCount(cCount_);
    tilingData_.cParam.set_cNum(cNum_);
    tilingData_.cParam.set_cLine(cLine_);
    tilingData_.cParam.set_cTail(cTail_);

    tilingData_.castCopyParam.set_maxDataNumInUb(maxDataNumInUb_);
    tilingData_.castCopyParam.set_normalCoreNum(normalCoreNum_);
    tilingData_.castCopyParam.set_tailCoreNum(tailCoreNum_);
    tilingData_.castCopyParam.set_normalCoreDataNum(normalCoreDataNum_);
    tilingData_.castCopyParam.set_tailCoreDataNum(tailCoreDataNum_);
    tilingData_.castCopyParam.set_normalCoreFormerCopyTime(normalCoreFormerCopyTime_);
    tilingData_.castCopyParam.set_normalCoreTailCopyTime(normalCoreTailCopyTime_);
    tilingData_.castCopyParam.set_normalCoreFormerDataNum(normalCoreFormerDataNum_);
    tilingData_.castCopyParam.set_normalCoreTailDataNum(normalCoreTailDataNum_);
    tilingData_.castCopyParam.set_tailCoreFormerCopyTime(tailCoreFormerCopyTime_);
    tilingData_.castCopyParam.set_tailCoreTailCopyTime(tailCoreTailCopyTime_);
    tilingData_.castCopyParam.set_tailCoreFormerDataNum(tailCoreFormerDataNum_);
    tilingData_.castCopyParam.set_tailCoreTailDataNum(tailCoreTailDataNum_);

    tilingData_.hwParam.set_normalCoreHWNum(normalCoreHWNum_);
    tilingData_.hwParam.set_lastCoreHWNum(lastCoreHWNum_);
    tilingData_.hwParam.set_hwAlign(hwAlign_);
    tilingData_.hwParam.set_hwTotal(hwTotal_);
    tilingData_.hwParam.set_hwCount(hwCount_);
    tilingData_.hwParam.set_hwNum(hwNum_);
    tilingData_.hwParam.set_nLine(hwLine_);
    tilingData_.hwParam.set_hwTail(hwTail_);

    tilingData_.blockParam.set_singleCoreNc(singleCoreNc);
    tilingData_.blockParam.set_singleCoreDo(singleCoreDo);
    tilingData_.blockParam.set_singleCoreHo(singleCoreHo);
    tilingData_.blockParam.set_singleCoreWo(singleCoreWo);
    tilingData_.blockParam.set_ncCnt(ncCnt);
    tilingData_.blockParam.set_doCnt(doCnt);
    tilingData_.blockParam.set_hoCnt(hoCnt);
    tilingData_.blockParam.set_woCnt(woCnt);
    tilingData_.blockParam.set_totalCnt(totalCnt);
    tilingData_.blockParam.set_ncTailData(ncTailData);
    tilingData_.blockParam.set_doTailData(doTailData);
    tilingData_.blockParam.set_hoTailData(hoTailData);
    tilingData_.blockParam.set_woTailData(woTailData);
    tilingData_.blockParam.set_baseDo(baseDo_);
    tilingData_.blockParam.set_baseHo(baseHo_);
    tilingData_.blockParam.set_baseWo(baseWo_);
    tilingData_.blockParam.set_baseDi(baseDo_);
    tilingData_.blockParam.set_baseHi(baseHi_);
    tilingData_.blockParam.set_baseWi(baseWi_);
    tilingData_.blockParam.set_ubSize(ubSize);
    tilingData_.blockParam.set_isScatter(isScatter);
    tilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    tilingContext_->SetScheduleMode(BATCH_MODE);
    TilingDataPrint();
    std::cout << "*******************START*******************" << std::endl;
    std::cout << "N = " << tilingData_.attrParam.get_N() << std::endl;
    std::cout << "C = " << tilingData_.attrParam.get_C() << std::endl;
    std::cout << "inN = " << tilingData_.attrParam.get_inN() << std::endl;
    std::cout << "inD = " << tilingData_.attrParam.get_inD() << std::endl;
    std::cout << "inH = " << tilingData_.attrParam.get_inH() << std::endl;
    std::cout << "inW = " << tilingData_.attrParam.get_inW() << std::endl;
    std::cout << "outN = " << tilingData_.attrParam.get_outN() << std::endl;
    std::cout << "outD = " << tilingData_.attrParam.get_outD() << std::endl;
    std::cout << "outH = " << tilingData_.attrParam.get_outH() << std::endl;
    std::cout << "outW = " << tilingData_.attrParam.get_outW() << std::endl;
    std::cout << "kD = " << tilingData_.attrParam.get_kD() << std::endl;
    std::cout << "kH = " << tilingData_.attrParam.get_kH() << std::endl;
    std::cout << "kW = " << tilingData_.attrParam.get_kW() << std::endl;
    std::cout << "dD = " << tilingData_.attrParam.get_dD() << std::endl;
    std::cout << "dH = " << tilingData_.attrParam.get_dH() << std::endl;
    std::cout << "dW = " << tilingData_.attrParam.get_dW() << std::endl;
    std::cout << "padD = " << tilingData_.attrParam.get_padD() << std::endl;
    std::cout << "padH = " << tilingData_.attrParam.get_padH() << std::endl;
    std::cout << "padW = " << tilingData_.attrParam.get_padW() << std::endl;
    std::cout << "countIncludePad = " << tilingData_.attrParam.get_countIncludePad() << std::endl;
    std::cout << "divisorOverride = " << tilingData_.attrParam.get_divisorOverride() << std::endl;
    std::cout << "isOverLap = " << tilingData_.attrParam.get_isOverLap() << std::endl;
    std::cout << "isDetermine = " << tilingData_.attrParam.get_isDetermine() << std::endl;
    std::cout << "normalCoreCNum = " << tilingData_.cParam.get_normalCoreCNum() << std::endl;
    std::cout << "lastCoreCNum = " << tilingData_.cParam.get_lastCoreCNum() << std::endl;
    std::cout << "cAlign = " << tilingData_.cParam.get_cAlign() << std::endl;
    std::cout << "cTotal = " << tilingData_.cParam.get_cTotal() << std::endl;
    std::cout << "cCount = " << tilingData_.cParam.get_cCount() << std::endl;
    std::cout << "cNum = " << tilingData_.cParam.get_cNum() << std::endl;
    std::cout << "cLine = " << tilingData_.cParam.get_cLine() << std::endl;
    std::cout << "cTail = " << tilingData_.cParam.get_cTail() << std::endl;
    std::cout << "maxDataNumInUb = " << tilingData_.castCopyParam.get_maxDataNumInUb() << std::endl;
    std::cout << "normalCoreNum = " << tilingData_.castCopyParam.get_normalCoreNum() << std::endl;
    std::cout << "tailCoreNum = " << tilingData_.castCopyParam.get_tailCoreNum() << std::endl;
    std::cout << "normalCoreDataNum = " << tilingData_.castCopyParam.get_normalCoreDataNum() << std::endl;
    std::cout << "tailCoreDataNum = " << tilingData_.castCopyParam.get_tailCoreDataNum() << std::endl;
    std::cout << "normalCoreFormerCopyTime = " << tilingData_.castCopyParam.get_normalCoreFormerCopyTime() << std::endl;
    std::cout << "normalCoreTailCopyTime = " << tilingData_.castCopyParam.get_normalCoreTailCopyTime() << std::endl;
    std::cout << "normalCoreFormerDataNum = " << tilingData_.castCopyParam.get_normalCoreFormerDataNum() << std::endl;
    std::cout << "normalCoreTailDataNum = " << tilingData_.castCopyParam.get_normalCoreTailDataNum() << std::endl;
    std::cout << "tailCoreFormerCopyTime = " << tilingData_.castCopyParam.get_tailCoreFormerCopyTime() << std::endl;
    std::cout << "tailCoreTailCopyTime = " << tilingData_.castCopyParam.get_tailCoreTailCopyTime() << std::endl;
    std::cout << "tailCoreFormerDataNum = " << tilingData_.castCopyParam.get_tailCoreFormerDataNum() << std::endl;
    std::cout << "tailCoreTailDataNum = " << tilingData_.castCopyParam.get_tailCoreTailDataNum() << std::endl;
    std::cout << "normalCoreHWNum = " << tilingData_.hwParam.get_normalCoreHWNum() << std::endl;
    std::cout << "lastCoreHWNum = " << tilingData_.hwParam.get_lastCoreHWNum() << std::endl;
    std::cout << "hwAlign = " << tilingData_.hwParam.get_hwAlign() << std::endl;
    std::cout << "hwTotal = " << tilingData_.hwParam.get_hwTotal() << std::endl;
    std::cout << "hwCount = " << tilingData_.hwParam.get_hwCount() << std::endl;
    std::cout << "hwNum = " << tilingData_.hwParam.get_hwNum() << std::endl;
    std::cout << "nLine = " << tilingData_.hwParam.get_nLine() << std::endl;
    std::cout << "hwTail = " << tilingData_.hwParam.get_hwTail() << std::endl;
    std::cout << "singleCoreNc = " << tilingData_.blockParam.get_singleCoreNc() << std::endl;
    std::cout << "singleCoreDo = " << tilingData_.blockParam.get_singleCoreDo() << std::endl;
    std::cout << "singleCoreHo = " << tilingData_.blockParam.get_singleCoreHo() << std::endl;
    std::cout << "singleCoreWo = " << tilingData_.blockParam.get_singleCoreWo() << std::endl;
    std::cout << "ncCnt = " << tilingData_.blockParam.get_ncCnt() << std::endl;
    std::cout << "doCnt = " << tilingData_.blockParam.get_doCnt() << std::endl;
    std::cout << "hoCnt = " << tilingData_.blockParam.get_hoCnt() << std::endl;
    std::cout << "woCnt = " << tilingData_.blockParam.get_woCnt() << std::endl;
    std::cout << "totalCnt = " << tilingData_.blockParam.get_totalCnt() << std::endl;
    std::cout << "ncTailData = " << tilingData_.blockParam.get_ncTailData() << std::endl;
    std::cout << "doTailData = " << tilingData_.blockParam.get_doTailData() << std::endl;
    std::cout << "hoTailData = " << tilingData_.blockParam.get_hoTailData() << std::endl;
    std::cout << "woTailData = " << tilingData_.blockParam.get_woTailData() << std::endl;
    std::cout << "baseDo = " << tilingData_.blockParam.get_baseDo() << std::endl;
    std::cout << "baseHo = " << tilingData_.blockParam.get_baseHo() << std::endl;
    std::cout << "baseWo = " << tilingData_.blockParam.get_baseWo() << std::endl;
    std::cout << "baseDi = " << tilingData_.blockParam.get_baseDi() << std::endl;
    std::cout << "baseHi = " << tilingData_.blockParam.get_baseHi() << std::endl;
    std::cout << "baseWi = " << tilingData_.blockParam.get_baseWi() << std::endl;
    std::cout << "ubSize = " << tilingData_.blockParam.get_ubSize() << std::endl;
    std::cout << "isScatter = " << tilingData_.blockParam.get_isScatter() << std::endl;
    std::cout << "*******************END*******************" << std::endl;
    return ge::GRAPH_SUCCESS;
}

void AvgPool3dGradTiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext_->GetNodeName(), "coreNum:                    %lu", coreNum_);
    OP_LOGD(tilingContext_->GetNodeName(), "outn:                       %ld", outDHW_.n);
    OP_LOGD(tilingContext_->GetNodeName(), "outd:                       %ld", outDHW_.d);
    OP_LOGD(tilingContext_->GetNodeName(), "outh:                       %ld", outDHW_.h);
    OP_LOGD(tilingContext_->GetNodeName(), "outw:                       %ld", outDHW_.w);
    OP_LOGD(tilingContext_->GetNodeName(), "inn:                        %ld", inDHW_.n);
    OP_LOGD(tilingContext_->GetNodeName(), "ind:                        %ld", inDHW_.d);
    OP_LOGD(tilingContext_->GetNodeName(), "inh:                        %ld", inDHW_.h);
    OP_LOGD(tilingContext_->GetNodeName(), "inw:                        %ld", inDHW_.w);
    OP_LOGD(tilingContext_->GetNodeName(), "kd:                         %ld", kDHW_.d);
    OP_LOGD(tilingContext_->GetNodeName(), "kh:                         %ld", kDHW_.h);
    OP_LOGD(tilingContext_->GetNodeName(), "kw:                         %ld", kDHW_.w);
    OP_LOGD(tilingContext_->GetNodeName(), "sd:                         %ld", dDHW_.d);
    OP_LOGD(tilingContext_->GetNodeName(), "sh:                         %ld", dDHW_.h);
    OP_LOGD(tilingContext_->GetNodeName(), "sw:                         %ld", dDHW_.w);
    OP_LOGD(tilingContext_->GetNodeName(), "padd:                       %ld", padDHW_.d);
    OP_LOGD(tilingContext_->GetNodeName(), "padh:                       %ld", padDHW_.h);
    OP_LOGD(tilingContext_->GetNodeName(), "padw:                       %ld", padDHW_.w);
    OP_LOGD(tilingContext_->GetNodeName(), "countIncludePad:            %lu", countIncludePad_);
    OP_LOGD(tilingContext_->GetNodeName(), "isOverlap:                  %lu", isOverlap_);
    OP_LOGD(tilingContext_->GetNodeName(), "normalCoreCNum:            %lu", normalCoreCNum_);
    OP_LOGD(tilingContext_->GetNodeName(), "lastCoreCNum:              %lu", lastCoreCNum_);
    OP_LOGD(tilingContext_->GetNodeName(), "cTotal:                    %lu", cTotal_);
    OP_LOGD(tilingContext_->GetNodeName(), "cAlign:                    %lu", cAlign_);
    OP_LOGD(tilingContext_->GetNodeName(), "cCount:                    %lu", cCount_);
    OP_LOGD(tilingContext_->GetNodeName(), "cNum:                      %lu", cNum_);
    OP_LOGD(tilingContext_->GetNodeName(), "cLine:                     %lu", cLine_);
    OP_LOGD(tilingContext_->GetNodeName(), "cTail:                     %lu", cTail_);
    OP_LOGD(tilingContext_->GetNodeName(), "maxDataNumInUb:             %lu", maxDataNumInUb_);
    OP_LOGD(tilingContext_->GetNodeName(), "normalCoreNum:              %lu", normalCoreNum_);
    OP_LOGD(tilingContext_->GetNodeName(), "tailCoreNum:                %lu", tailCoreNum_);
    OP_LOGD(tilingContext_->GetNodeName(), "normalCoreDataNum:          %lu", normalCoreDataNum_);
    OP_LOGD(tilingContext_->GetNodeName(), "tailCoreDataNum:            %lu", tailCoreDataNum_);
    OP_LOGD(tilingContext_->GetNodeName(), "normalCoreFormerCopyTime:   %lu", normalCoreFormerCopyTime_);
    OP_LOGD(tilingContext_->GetNodeName(), "normalCoreTailCopyTime:     %lu", normalCoreTailCopyTime_);
    OP_LOGD(tilingContext_->GetNodeName(), "normalCoreFormerDataNum:    %lu", normalCoreFormerDataNum_);
    OP_LOGD(tilingContext_->GetNodeName(), "normalCoreTailDataNum:      %lu", normalCoreTailDataNum_);
    OP_LOGD(tilingContext_->GetNodeName(), "tailCoreFormerCopyTime:     %lu", tailCoreFormerCopyTime_);
    OP_LOGD(tilingContext_->GetNodeName(), "tailCoreTailCopyTime:       %lu", tailCoreTailCopyTime_);
    OP_LOGD(tilingContext_->GetNodeName(), "tailCoreFormerDataNum:      %lu", tailCoreFormerDataNum_);
    OP_LOGD(tilingContext_->GetNodeName(), "tailCoreTailDataNum:        %lu", tailCoreTailDataNum_);
    OP_LOGD(tilingContext_->GetNodeName(), "normalCoreHWNum:            %lu", normalCoreHWNum_);
    OP_LOGD(tilingContext_->GetNodeName(), "lastCoreHWNum:              %lu", lastCoreHWNum_);
    OP_LOGD(tilingContext_->GetNodeName(), "hwTotal:                    %lu", hwTotal_);
    OP_LOGD(tilingContext_->GetNodeName(), "hwAlign:                    %lu", hwAlign_);
    OP_LOGD(tilingContext_->GetNodeName(), "hwCount:                    %lu", hwCount_);
    OP_LOGD(tilingContext_->GetNodeName(), "hwNum:                      %lu", hwNum_);
    OP_LOGD(tilingContext_->GetNodeName(), "hwLine:                     %lu", hwLine_);
    OP_LOGD(tilingContext_->GetNodeName(), "hwTail:                     %lu", hwTail_);
    OP_LOGD(tilingContext_->GetNodeName(), "singleCoreNc:                     %lu", singleCoreNc);
    OP_LOGD(tilingContext_->GetNodeName(), "singleCoreDo:                     %lu", singleCoreDo);
    OP_LOGD(tilingContext_->GetNodeName(), "singleCoreHo:                     %lu", singleCoreHo);
    OP_LOGD(tilingContext_->GetNodeName(), "singleCoreWo:                     %lu", singleCoreWo);
    OP_LOGD(tilingContext_->GetNodeName(), "ncCnt:                      %lu", ncCnt);
    OP_LOGD(tilingContext_->GetNodeName(), "doCnt:                      %lu", doCnt);
    OP_LOGD(tilingContext_->GetNodeName(), "hoCnt:                      %lu", hoCnt);
    OP_LOGD(tilingContext_->GetNodeName(), "woCnt:                      %lu", woCnt);
    OP_LOGD(tilingContext_->GetNodeName(), "totalCnt:                      %lu", totalCnt);
    OP_LOGD(tilingContext_->GetNodeName(), "ncTail:                      %lu", ncTailData);
    OP_LOGD(tilingContext_->GetNodeName(), "doTail:                      %lu", doTailData);
    OP_LOGD(tilingContext_->GetNodeName(), "hoTail:                      %lu", hoTailData);
    OP_LOGD(tilingContext_->GetNodeName(), "woTail:                      %lu", woTailData);
    OP_LOGD(tilingContext_->GetNodeName(), "baseDo:                      %lu", baseDo_);
    OP_LOGD(tilingContext_->GetNodeName(), "baseHo:                      %lu", baseHo_);
    OP_LOGD(tilingContext_->GetNodeName(), "baseWo:                      %lu", baseWo_);
    OP_LOGD(tilingContext_->GetNodeName(), "baseDi:                      %lu", baseDi_);
    OP_LOGD(tilingContext_->GetNodeName(), "baseHi:                      %lu", baseHi_);
    OP_LOGD(tilingContext_->GetNodeName(), "baseWi:                      %lu", baseWi_);
    OP_LOGD(tilingContext_->GetNodeName(), "ubSize:                      %lu", ubSize);
    OP_LOGD(tilingContext_->GetNodeName(), "isScatter:                      %lu", isScatter);
}

ge::graphStatus TilingAvgPool3dGradVec(gert::TilingContext* context)
{
    AvgPool3dGradTiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "tiling init fail");
        return ge::GRAPH_FAILED;
    }
    return tilingObject.SetKernelTiling();
}

ge::graphStatus TilingPrepareForAvgPool3dGradVec(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling Prepare For AvgPool3dGrad start");
    OP_LOGD(context->GetNodeName(), "Tiling prepare for AvgPool3dGrad end");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForAvgPool3DGrad(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("AvgPool3DGrad", "context is null"), return ge::GRAPH_FAILED);
    return TilingAvgPool3dGradVec(context);
}

static ge::graphStatus TilingPrepareForAvgPool3DGrad(gert::TilingParseContext* context)
{
    return TilingPrepareForAvgPool3dGradVec(context);
}

IMPL_OP_OPTILING(AvgPool3DGrad)
    .InputsDataDependency({ORIG_INPUT_SHAPE_INDEX})
    .Tiling(TilingForAvgPool3DGrad)
    .TilingParse<AvgPool3dGradCompileInfo>(TilingPrepareForAvgPool3DGrad);
}