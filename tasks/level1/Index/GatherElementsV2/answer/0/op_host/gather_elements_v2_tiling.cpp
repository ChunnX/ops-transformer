#include "gather_elements_v2_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGI(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGW(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) { \
        std::printf("nullptr error!"); \
        return ge::GRAPH_FAILED; \
    }
#define unlikely(x) __builtin_expect((x), 0)
#define OP_CHECK_IF(condition, log, return_expr) \
    do { \
        if (unlikely(condition)) { \
            log; \
            return_expr; \
        } \
    } while (0)

namespace optiling {
constexpr uint64_t RESERVED_UB_SIZE = 2048;
constexpr uint64_t CACHELINE = 512;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t TRANSPOSE_WS_LEN = 128;
constexpr uint64_t X_IDX = 0;
constexpr uint64_t INDEX_IDX = 1;
constexpr uint64_t NUM_TWO = 2;

// ========== LastDim tiling helper functions ==========

constexpr int64_t LAST_DIM_RESERVED_UB = 10 * 1024;
constexpr int64_t NUM_TEN = 10;
constexpr int64_t NUM_ONE = 1;
constexpr int64_t INT8_DSIZE = 1;
constexpr int64_t INT6_DSIZE = 2;
constexpr int64_t INT32_DSIZE = 4;
constexpr int64_t INT64_DSIZE = 8;
constexpr int64_t MAX_X_SIZE = 1024;
constexpr int64_t BLOCK_SIZE_T = 32;
constexpr int64_t TILING_KEY_2 = 2;
constexpr int64_t MAX_SLICE_NUM = 5;
constexpr int64_t MAX_SIZE_RATIO = 256;
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t INDEX_INDEX = 1;
constexpr uint32_t DOUBLE_TIME = 2;
const int32_t DIVISION_BY_ZERO_LOG = -100000;
constexpr float MASK_SIZE_RATE = 1.0f / 4;

// CeilAlign/CeilDiv helpers for tiling (host-side, not __aicore__)
inline int64_t HostCeilAlign(int64_t a, int64_t b) { return b == 0 ? 0 : (a + b - 1) / b * b; }
inline int64_t HostCeilDiv(int64_t a, int64_t b) { return b == 0 ? 0 : (a + b - 1) / b; }

// For the GatherElementsV2 tiling class, we use the same CeilAlign as in Ops::Base
namespace {
inline uint64_t CeilAlign(uint64_t a, uint64_t b) { return b == 0 ? 0 : (a + b - 1) / b * b; }
}

class GatherElementsV2Tiling
{
public:
    explicit GatherElementsV2Tiling(gert::TilingContext* context) : tilingContext_(context)
    {}
    ge::graphStatus Init();
    ge::graphStatus SetKernelTiling();
    void TilingDataPrint() const;

private:
    inline void SetTilingKeyMode();
    inline void Tiling4GatherElementsV2();
    inline void Tiling4Scalar();
    inline bool CalcMaxBufferSize();
    inline void CalcMaxIdxDataAlign();

    GatherElementsV2TilingData tilingData_;
    gert::TilingContext* tilingContext_ = nullptr;
    uint64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;

private:
    uint64_t usedCoreNum_ = 0;
    uint64_t tilingKey_ = 0;
    ge::DataType xDtype_;
    ge::DataType idxDtype_;
    uint64_t xDtypeSize_;

    uint64_t xPreDim_ = 1;
    uint64_t xGatherDim_ = 1;
    uint64_t xPostDim_ = 1;
    uint64_t idxPreDim_ = 1;
    uint64_t idxGatherDim_ = 1;
    uint64_t idxPostDim_ = 1;
    uint64_t coreGroupNum_ = 0;
    uint64_t formerGroupNum_ = 0;
    uint64_t tailGroupNum_ = 0;
    uint64_t formerGroupPreDim_ = 0;
    uint64_t tailGroupPreDim_ = 0;
    uint64_t formerGroupCoreNum_ = 0;
    uint64_t tailGroupCoreNum_ = 0;

    uint64_t formerGroupFormerNum_ = 0;
    uint64_t formerGroupTailNum_ = 0;
    uint64_t formerGroupFormerPostDim_ = 0;
    uint64_t formerGroupTailPostDim_ = 0;

    uint64_t tailGroupFormerNum_ = 0;
    uint64_t tailGroupTailNum_ = 0;
    uint64_t tailGroupFormerPostDim_ = 0;
    uint64_t tailGroupTailPostDim_ = 0;

    // transpose
    uint64_t carryNumAlign_ = 0;
    uint64_t xCarryNumAlign_ = 0;
    uint64_t idxCarryNumAlign_ = 0;
    uint64_t inBufferSize_ = 0;
    uint64_t outBufferSize_ = 0;
    uint64_t transGatherDimSlice_ = TRANSPOSE_WS_LEN;
    uint64_t idxGatherDimSlice_ = 0;
    uint64_t workspacePerBlock_ = 0;
    bool needTrans_ = false;

    // scalar
    uint64_t formerGroupFormerData_ = 0;
    uint64_t formerGroupTailData_ = 0;
    uint64_t tailGroupFormerData_ = 0;
    uint64_t tailGroupTailData_ = 0;
    uint64_t maxIdxDataAlign_ = 0;
};

inline void GatherElementsV2Tiling::SetTilingKeyMode()
{
    tilingKey_ = needTrans_;
    tilingContext_->SetTilingKey(tilingKey_);
}

inline bool GatherElementsV2Tiling::CalcMaxBufferSize()
{
    carryNumAlign_ = CACHELINE / xDtypeSize_;
    xCarryNumAlign_ = CACHELINE / xDtypeSize_;
    idxCarryNumAlign_ = CACHELINE / sizeof(int32_t);
    uint64_t xAlign = BLOCK_SIZE / xDtypeSize_;
    uint64_t idxAlign = BLOCK_SIZE / sizeof(int32_t);
    uint64_t availableUb = ubSize_ - RESERVED_UB_SIZE;
    uint64_t minIdxGatherDimSlice = CACHELINE;
    uint64_t gatherInBufferSize = CeilAlign(xGatherDim_, xAlign) * xDtypeSize_ +
                                  CeilAlign(minIdxGatherDimSlice, idxAlign) * sizeof(int32_t) * NUM_TWO;
    uint64_t gatherOutBufferSize = CeilAlign(minIdxGatherDimSlice, xAlign) * xDtypeSize_;
    uint64_t transInBufferSize = TRANSPOSE_WS_LEN * CACHELINE;
    uint64_t transOutBufferSize = TRANSPOSE_WS_LEN * CACHELINE;
    inBufferSize_ = std::max(gatherInBufferSize, transInBufferSize);
    outBufferSize_ = std::max(gatherOutBufferSize, transOutBufferSize);
    if (availableUb < (inBufferSize_ + outBufferSize_)) {
        return false;
    }
    uint64_t ubLeft =
        availableUb -
        (std::max(transInBufferSize, CeilAlign(xGatherDim_, xAlign) * xDtypeSize_) + transOutBufferSize);
    uint64_t maxIdxGatherDimSlice = ubLeft / (BLOCK_SIZE * NUM_TWO) * idxAlign;
    uint64_t idxGatherDimAlign = CeilAlign(idxGatherDim_, idxAlign);
    idxGatherDimSlice_ = std::min(maxIdxGatherDimSlice, idxGatherDimAlign);
    gatherInBufferSize = CeilAlign(xGatherDim_, xAlign) * xDtypeSize_ +
                         CeilAlign(idxGatherDimSlice_, idxAlign) * sizeof(int32_t);
    gatherOutBufferSize = CeilAlign(idxGatherDimSlice_, xAlign) * xDtypeSize_;
    inBufferSize_ = std::max(gatherInBufferSize, transInBufferSize);
    outBufferSize_ = std::max(gatherOutBufferSize, transOutBufferSize);
    return true;
}

inline void GatherElementsV2Tiling::CalcMaxIdxDataAlign()
{
    uint64_t availableUb = ubSize_ - RESERVED_UB_SIZE;
    maxIdxDataAlign_ = availableUb / BLOCK_SIZE * BLOCK_SIZE / (sizeof(int32_t) + xDtypeSize_);
}

inline void GatherElementsV2Tiling::Tiling4Scalar()
{
    OP_LOGD(tilingContext_, "Tiling4ScalarMode start");
    uint64_t idxDataPerPre = idxGatherDim_ * idxPostDim_;
    uint64_t idxAllData = idxPreDim_ * idxDataPerPre;
    usedCoreNum_ = std::min(idxAllData, coreNum_);
    if (idxPreDim_ >= usedCoreNum_) {
        formerGroupFormerData_ = idxDataPerPre;
        formerGroupTailData_ = idxDataPerPre;
        tailGroupFormerData_ = idxDataPerPre;
        tailGroupTailData_ = idxDataPerPre;
    } else {
        coreGroupNum_ = idxPreDim_;
        tailGroupNum_ = (coreGroupNum_ - usedCoreNum_ % coreGroupNum_) % coreGroupNum_;
        formerGroupNum_ = coreGroupNum_ - tailGroupNum_;

        formerGroupPreDim_ = 1;
        tailGroupPreDim_ = 1;
        formerGroupCoreNum_ = (usedCoreNum_ + coreGroupNum_ - 1) / coreGroupNum_;
        tailGroupCoreNum_ = usedCoreNum_ / coreGroupNum_;

        formerGroupTailNum_ = (formerGroupCoreNum_ - idxDataPerPre % formerGroupCoreNum_) % formerGroupCoreNum_;
        formerGroupFormerNum_ = formerGroupCoreNum_ - formerGroupTailNum_;
        formerGroupFormerData_ = (idxDataPerPre + formerGroupCoreNum_ - 1) / formerGroupCoreNum_;
        formerGroupTailData_ = idxDataPerPre / formerGroupCoreNum_;

        tailGroupTailNum_ = (tailGroupCoreNum_ - idxDataPerPre % tailGroupCoreNum_) % tailGroupCoreNum_;
        tailGroupFormerNum_ = tailGroupCoreNum_ - tailGroupTailNum_;
        tailGroupFormerData_ = (idxDataPerPre + tailGroupCoreNum_ - 1) / tailGroupCoreNum_;
        tailGroupTailData_ = idxDataPerPre / tailGroupCoreNum_;
    }
    CalcMaxIdxDataAlign();
    OP_LOGD(tilingContext_, "Tiling4ScalarMode finished");
}

inline void GatherElementsV2Tiling::Tiling4GatherElementsV2()
{
    OP_LOGD(tilingContext_, "Tiling4GatherElementsV2 start");

    usedCoreNum_ = std::min(idxPreDim_ * idxPostDim_, coreNum_);
    if (idxPreDim_ > usedCoreNum_) {
        coreGroupNum_ = usedCoreNum_;
        tailGroupNum_ = (coreGroupNum_ - idxPreDim_ % coreGroupNum_) % coreGroupNum_;
        formerGroupNum_ = coreGroupNum_ - tailGroupNum_;
        if (usedCoreNum_ == 0UL) {
            usedCoreNum_ = 1UL;
        }
        formerGroupPreDim_ = (idxPreDim_ + usedCoreNum_ - 1) / usedCoreNum_;
        tailGroupPreDim_ = idxPreDim_ / usedCoreNum_;
        formerGroupCoreNum_ = 1;
        tailGroupCoreNum_ = 1;

        formerGroupTailNum_ = 0;
        formerGroupFormerNum_ = 1;
        tailGroupTailNum_ = 0;
        tailGroupFormerNum_ = 1;

        formerGroupFormerPostDim_ = idxPostDim_;
        formerGroupTailPostDim_ = idxPostDim_;
        tailGroupFormerPostDim_ = idxPostDim_;
        tailGroupTailPostDim_ = idxPostDim_;
    } else {
        coreGroupNum_ = idxPreDim_;
        tailGroupNum_ = (coreGroupNum_ - usedCoreNum_ % coreGroupNum_) % coreGroupNum_;
        formerGroupNum_ = coreGroupNum_ - tailGroupNum_;

        formerGroupPreDim_ = 1;
        tailGroupPreDim_ = 1;
        formerGroupCoreNum_ = (usedCoreNum_ + coreGroupNum_ - 1) / coreGroupNum_;
        tailGroupCoreNum_ = usedCoreNum_ / coreGroupNum_;

        formerGroupTailNum_ = (formerGroupCoreNum_ - idxPostDim_ % formerGroupCoreNum_) % formerGroupCoreNum_;
        formerGroupFormerNum_ = formerGroupCoreNum_ - formerGroupTailNum_;
        formerGroupFormerPostDim_ = (idxPostDim_ + formerGroupCoreNum_ - 1) / formerGroupCoreNum_;
        formerGroupTailPostDim_ = idxPostDim_ / formerGroupCoreNum_;

        tailGroupTailNum_ = (tailGroupCoreNum_ - idxPostDim_ % tailGroupCoreNum_) % tailGroupCoreNum_;
        tailGroupFormerNum_ = tailGroupCoreNum_ - tailGroupTailNum_;
        tailGroupFormerPostDim_ = (idxPostDim_ + tailGroupCoreNum_ - 1) / tailGroupCoreNum_;
        tailGroupTailPostDim_ = idxPostDim_ / tailGroupCoreNum_;
    }

    OP_LOGD(tilingContext_, "Tiling4GatherElementsV2 finished");
}

ge::graphStatus GatherElementsV2Tiling::Init()
{
    OP_LOGD(tilingContext_, "GatherElementsV2Tiling initing");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext_->GetPlatformInfo());
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize_ = ubSizePlatForm;

    auto attrs = tilingContext_->GetAttrs();
    auto xShape = tilingContext_->GetInputShape(X_IDX)->GetStorageShape();
    auto idxShape = tilingContext_->GetInputShape(INDEX_IDX)->GetStorageShape();
    auto dim = *(attrs->GetAttrPointer<uint64_t>)(0);
    auto dimNum = xShape.GetDimNum();
    idxGatherDim_ = idxShape.GetDim(dim);
    xGatherDim_ = xShape.GetDim(dim);
    for (size_t i = 0; i < dim; i++) {
        idxPreDim_ *= idxShape.GetDim(i);
        xPreDim_ *= xShape.GetDim(i);
    }
    for (size_t i = dim + 1; i < dimNum; i++) {
        idxPostDim_ *= idxShape.GetDim(i);
        xPostDim_ *= xShape.GetDim(i);
    }
    xDtype_ = tilingContext_->GetInputDesc(X_IDX)->GetDataType();
    idxDtype_ = tilingContext_->GetInputDesc(INDEX_IDX)->GetDataType();
    xDtypeSize_ = GetSizeByDataType(xDtype_);

    Tiling4GatherElementsV2();
    size_t transWorkspaceSize = 0;
    if (CalcMaxBufferSize()) {
        needTrans_ = true;
        if (xDtypeSize_ == 0UL) {
            OP_LOGE("GatherElementsV2", "xDtypeSize is 0.");
            return ge::GRAPH_FAILED;
        }
        uint64_t xGatherDimAlign = CeilAlign(xGatherDim_ * xDtypeSize_, sizeof(int32_t)) / xDtypeSize_;
        uint64_t usedWorkspaceLen =
            std::min(carryNumAlign_, std::max(formerGroupFormerPostDim_, tailGroupFormerPostDim_));
        workspacePerBlock_ = usedWorkspaceLen * (xGatherDimAlign * xDtypeSize_ + idxGatherDim_ * sizeof(int32_t));
        transWorkspaceSize = usedCoreNum_ * workspacePerBlock_;
    } else {
        needTrans_ = false;
        Tiling4Scalar();
    }
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    sysWorkspaceSize = sysWorkspaceSize + transWorkspaceSize;
    size_t* currentWorkSpace = tilingContext_->GetWorkspaceSizes(1);
    currentWorkSpace[0] = sysWorkspaceSize;
    SetTilingKeyMode();

    OP_LOGD(tilingContext_, "GatherElementsV2Tiling inited");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherElementsV2Tiling::SetKernelTiling()
{
    tilingContext_->SetBlockDim(usedCoreNum_);
    tilingData_.params.set_xPreDim(xPreDim_);
    tilingData_.params.set_xGatherDim(xGatherDim_);
    tilingData_.params.set_xPostDim(xPostDim_);
    tilingData_.params.set_idxPreDim(idxPreDim_);
    tilingData_.params.set_idxGatherDim(idxGatherDim_);
    tilingData_.params.set_idxPostDim(idxPostDim_);

    tilingData_.params.set_coreGroupNum(coreGroupNum_);
    tilingData_.params.set_formerGroupNum(formerGroupNum_);
    tilingData_.params.set_tailGroupNum(tailGroupNum_);

    tilingData_.params.set_formerGroupPreDim(formerGroupPreDim_);
    tilingData_.params.set_tailGroupPreDim(tailGroupPreDim_);
    tilingData_.params.set_formerGroupCoreNum(formerGroupCoreNum_);
    tilingData_.params.set_tailGroupCoreNum(tailGroupCoreNum_);

    tilingData_.params.set_formerGroupFormerNum(formerGroupFormerNum_);
    tilingData_.params.set_formerGroupTailNum(formerGroupTailNum_);
    tilingData_.params.set_formerGroupFormerPostDim(formerGroupFormerPostDim_);
    tilingData_.params.set_formerGroupTailPostDim(formerGroupTailPostDim_);

    tilingData_.params.set_tailGroupFormerNum(tailGroupFormerNum_);
    tilingData_.params.set_tailGroupTailNum(tailGroupTailNum_);
    tilingData_.params.set_tailGroupFormerPostDim(tailGroupFormerPostDim_);
    tilingData_.params.set_tailGroupTailPostDim(tailGroupTailPostDim_);

    tilingData_.transTiling.set_carryNumAlign(carryNumAlign_);
    tilingData_.transTiling.set_xCarryNumAlign(xCarryNumAlign_);
    tilingData_.transTiling.set_idxCarryNumAlign(idxCarryNumAlign_);
    tilingData_.transTiling.set_inBufferSize(inBufferSize_);
    tilingData_.transTiling.set_outBufferSize(outBufferSize_);
    tilingData_.transTiling.set_transGatherDimSlice(transGatherDimSlice_);
    tilingData_.transTiling.set_idxGatherDimSlice(idxGatherDimSlice_);
    tilingData_.transTiling.set_workspacePerBlock(workspacePerBlock_);

    tilingData_.scalarTiling.set_formerGroupFormerData(formerGroupFormerData_);
    tilingData_.scalarTiling.set_formerGroupTailData(formerGroupTailData_);
    tilingData_.scalarTiling.set_tailGroupFormerData(tailGroupFormerData_);
    tilingData_.scalarTiling.set_tailGroupTailData(tailGroupTailData_);
    tilingData_.scalarTiling.set_maxIdxDataAlign(maxIdxDataAlign_);

    tilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    TilingDataPrint();
    return ge::GRAPH_SUCCESS;
}

void GatherElementsV2Tiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext_, "tilingKey:                  %lu", tilingKey_);
    OP_LOGD(tilingContext_, "usedCoreNum:                %lu", usedCoreNum_);
    OP_LOGD(tilingContext_, "xPreDim:                    %lu", xPreDim_);
    OP_LOGD(tilingContext_, "xGatherDim:                 %lu", xGatherDim_);
    OP_LOGD(tilingContext_, "xPostDim:                   %lu", xPostDim_);
    OP_LOGD(tilingContext_, "idxPreDim:                  %lu", idxPreDim_);
    OP_LOGD(tilingContext_, "idxGatherDim:               %lu", idxGatherDim_);
    OP_LOGD(tilingContext_, "idxPostDim_:                %lu", idxPostDim_);

    OP_LOGD(tilingContext_, "coreGroupNum_:              %lu", coreGroupNum_);
    OP_LOGD(tilingContext_, "formerGroupNum:             %lu", formerGroupNum_);
    OP_LOGD(tilingContext_, "tailGroupNum:               %lu", tailGroupNum_);

    OP_LOGD(tilingContext_, "formerGroupPreDim:          %lu", formerGroupPreDim_);
    OP_LOGD(tilingContext_, "tailGroupPreDim:            %lu", tailGroupPreDim_);
    OP_LOGD(tilingContext_, "formerGroupCoreNum:         %lu", formerGroupCoreNum_);
    OP_LOGD(tilingContext_, "tailGroupCoreNum:           %lu", tailGroupCoreNum_);

    OP_LOGD(tilingContext_, "formerGroupFormerNum:       %lu", formerGroupFormerNum_);
    OP_LOGD(tilingContext_, "formerGroupTailNum:         %lu", formerGroupTailNum_);
    OP_LOGD(tilingContext_, "formerGroupFormerPostDim:   %lu", formerGroupFormerPostDim_);
    OP_LOGD(tilingContext_, "formerGroupTailPostDim:     %lu", formerGroupTailPostDim_);

    OP_LOGD(tilingContext_, "tailGroupFormerNum:         %lu", tailGroupFormerNum_);
    OP_LOGD(tilingContext_, "tailGroupTailNum:           %lu", tailGroupTailNum_);
    OP_LOGD(tilingContext_, "tailGroupFormerPostDim:     %lu", tailGroupFormerPostDim_);
    OP_LOGD(tilingContext_, "tailGroupTailPostDim:       %lu", tailGroupTailPostDim_);

    OP_LOGD(tilingContext_, "carryNumAlign:              %lu", carryNumAlign_);
    OP_LOGD(tilingContext_, "xCarryNumAlign:             %lu", xCarryNumAlign_);
    OP_LOGD(tilingContext_, "idxCarryNumAlign:           %lu", idxCarryNumAlign_);

    OP_LOGD(tilingContext_, "inBufferSize:               %lu", inBufferSize_);
    OP_LOGD(tilingContext_, "outBufferSize:              %lu", outBufferSize_);
    OP_LOGD(tilingContext_, "transGatherDimSlice:        %lu", transGatherDimSlice_);
    OP_LOGD(tilingContext_, "idxGatherDimSlice:          %lu", idxGatherDimSlice_);
    OP_LOGD(tilingContext_, "workspacePerBlock:          %lu", workspacePerBlock_);

    OP_LOGD(tilingContext_, "formerGroupFormerData:      %lu", formerGroupFormerData_);
    OP_LOGD(tilingContext_, "formerGroupTailData:        %lu", formerGroupTailData_);
    OP_LOGD(tilingContext_, "tailGroupFormerData:        %lu", tailGroupFormerData_);
    OP_LOGD(tilingContext_, "tailGroupTailData:          %lu", tailGroupTailData_);
    OP_LOGD(tilingContext_, "maxIdxDataAlign:            %lu", maxIdxDataAlign_);
}

// ========== LastDim Tiling ==========

class GatherElementsV2LastDimTiling {
public:
    explicit GatherElementsV2LastDimTiling(gert::TilingContext* context) : context_(context)
    {}
    ge::graphStatus Init();
    ge::graphStatus MergeAxis(const gert::StorageShape* xShape, const gert::StorageShape* indexShape);
    void SetShape();
    void GetDSize();
    void IfEnableBatch();
    void DoUBSlice();
    void DoNeedUseCore();
    void SetTilingData();
    void DoScalarMode();
    void PrintTilingData() const;
    ge::graphStatus DoOpTiling();

private:
    GatherElementsV2TilingData tilingData_;
    gert::TilingContext* context_ = nullptr;
    gert::Shape xShape_;
    gert::Shape indexShape_;

    int64_t totalCoreNum_ = 0;
    int64_t ubSizePlatForm_ = 0;

    int64_t dimNum_ = 0;
    int64_t nonCollectingAxisSzie_ = 1;
    int64_t xDSize_ = 0;
    int64_t xRealDsize_ = 0;
    int64_t indexDSize_ = 0;
    int64_t indexRealDsize_ = 0;
    int64_t tilingKey_ = 0;
    int64_t xAxisSize_ = 0;
    int64_t indexAxisSize_ = 0;
    int64_t eachCalculationLines_ = 0;
    int64_t xSliceNum_ = 0;
    int64_t indexSliceNum_ = 0;
    int64_t xAlignSize_ = 0;
    int64_t indexAlignSize_ = 0;
    int64_t yAlignSize_ = 0;
    int64_t reservedXSize_ = 0;
    int64_t reservedIndexSize_ = 0;
    int64_t needUsedCore_ = 0;
    int64_t formerCoreRowNum_ = 0;
    int64_t formerCoreNum_ = 0;
    int64_t xBufferSize_ = 0;
    int64_t indexBufferSize_ = 0;
    int64_t yBufferSize_ = 0;
    int64_t maskBufferSize_ = 0;
    int64_t specialDataMove_ = 0;
    int64_t xDsizeRatio_ = 1;
    int64_t scalarModeLength_ = 0;
    bool batchProcess_ = false;
};

ge::graphStatus GatherElementsV2LastDimTiling::Init()
{
    const gert::StorageShape* xShape = context_->GetInputShape(X_INDEX);
    const gert::StorageShape* indexShape = context_->GetInputShape(INDEX_INDEX);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    totalCoreNum_ = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSizePlatForm_ = static_cast<int64_t>(ubSizePlatForm) - LAST_DIM_RESERVED_UB;

    MergeAxis(xShape, indexShape);
    DoOpTiling();

    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkSpace = context_->GetWorkspaceSizes(1);
    currentWorkSpace[0] = sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherElementsV2LastDimTiling::MergeAxis(
    const gert::StorageShape* xShape, const gert::StorageShape* indexShape)
{
    gert::Shape xOriginShape = xShape->GetStorageShape();
    gert::Shape indexOriginShape = indexShape->GetStorageShape();
    auto xDimNum = static_cast<int>(xOriginShape.GetDimNum());
    auto indexDimNum = static_cast<int>(indexOriginShape.GetDimNum());

    OP_CHECK_IF(
        xDimNum != indexDimNum,
        OP_LOGE(context_, "The dim %d of x and the dim %d of index are not equal.", xDimNum, indexDimNum),
        return ge::GRAPH_FAILED);

    int i = 0;
    while (i < xDimNum) {
        if (xOriginShape[i] != indexOriginShape[i] || i == xDimNum - 1) {
            xShape_.AppendDim(xOriginShape[i]);
            indexShape_.AppendDim(indexOriginShape[i]);
        } else {
            int j = i;
            while (j < xDimNum && xOriginShape[j] == indexOriginShape[j] && j != xDimNum - 1) {
                j++;
            }
            if (j - i > 1) {
                int64_t val = 1;
                for (int k = i; k < j; k++) {
                    val *= xOriginShape[k];
                }
                xShape_.AppendDim(val);
                indexShape_.AppendDim(val);
            } else {
                xShape_.AppendDim(xOriginShape[i]);
                indexShape_.AppendDim(indexOriginShape[i]);
            }
            i = j - 1;
        }
        i++;
    }
    dimNum_ = static_cast<int64_t>(xShape_.GetDimNum());
    tilingData_.lastDimTiling.set_dimNum(dimNum_);
    SetShape();
    return ge::GRAPH_SUCCESS;
}

void GatherElementsV2LastDimTiling::SetShape()
{
    int64_t xShapeArray[TILING_ARRAY_LEN_EIGHT] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t indexShapeArray[TILING_ARRAY_LEN_EIGHT] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t xStrideArray[TILING_ARRAY_LEN_EIGHT] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t indexStrideArray[TILING_ARRAY_LEN_EIGHT] = {1, 1, 1, 1, 1, 1, 1, 1};

    for (int i = 0; i < dimNum_; i++) {
        xShapeArray[i] = xShape_.GetDim(i);
        indexShapeArray[i] = indexShape_.GetDim(i);
        if (i < dimNum_ - 1) {
            nonCollectingAxisSzie_ *= xShapeArray[i];
            if (i != 0 && xShapeArray[i] != indexShapeArray[i]) {
                specialDataMove_ = 1;
            }
        } else {
            xAxisSize_ = xShapeArray[i];
            indexAxisSize_ = indexShapeArray[i];
        }
    }
    for (int i = dimNum_ - 1; i > 0; i--) {
        xStrideArray[i - 1] = xStrideArray[i] * xShapeArray[i];
        indexStrideArray[i - 1] = indexStrideArray[i] * indexShapeArray[i];
    }
    tilingData_.lastDimTiling.set_xShape(xShapeArray);
    tilingData_.lastDimTiling.set_indexShape(indexShapeArray);
    tilingData_.lastDimTiling.set_xStrideArray(xStrideArray);
    tilingData_.lastDimTiling.set_indexStrideArray(indexStrideArray);
}

void GatherElementsV2LastDimTiling::GetDSize()
{
    ge::DataType xDataType = context_->GetInputDesc(X_INDEX)->GetDataType();
    ge::DataType indexDataType = context_->GetInputDesc(INDEX_INDEX)->GetDataType();
    xDSize_ = ge::GetSizeByDataType(xDataType);
    xRealDsize_ = xDSize_;
    indexDSize_ = ge::GetSizeByDataType(indexDataType);
    indexRealDsize_ = indexDSize_;
    context_->SetTilingKey(TILING_KEY_2);
    if (xDSize_ == INT8_DSIZE) {
        xDSize_ = INT6_DSIZE;
        xDsizeRatio_ = DOUBLE_TIME;
    }
    if (indexDSize_ == INT32_DSIZE) {
        indexDSize_ = INT64_DSIZE;
    }
}

void GatherElementsV2LastDimTiling::IfEnableBatch()
{
    xAlignSize_ = HostCeilAlign(xAxisSize_ * xDSize_, BLOCK_SIZE_T * xDsizeRatio_);
    yAlignSize_ = HostCeilAlign(indexAxisSize_ * xDSize_, BLOCK_SIZE_T * xDsizeRatio_);
    indexAlignSize_ = HostCeilAlign(indexAxisSize_ * indexDSize_, BLOCK_SIZE_T * DOUBLE_TIME);
    batchProcess_ = xAlignSize_ + yAlignSize_ + indexAlignSize_ <= ubSizePlatForm_ / DOUBLE_TIME;
}

void GatherElementsV2LastDimTiling::DoUBSlice()
{
    if (indexShape_[dimNum_ - 1] == 1 && batchProcess_ && specialDataMove_ == 0) {
        tilingData_.lastDimTiling.set_indexAxisSizeEqualOne(NUM_ONE);
        int64_t perGroupSize = xAxisSize_ * xDSize_ + indexAxisSize_ * indexDSize_ + indexAxisSize_ * indexDSize_;
        eachCalculationLines_ = ubSizePlatForm_ / perGroupSize;
        xBufferSize_ = HostCeilAlign(xAxisSize_ * xDSize_ * eachCalculationLines_, BLOCK_SIZE_T * xDsizeRatio_);
        indexBufferSize_ =
            HostCeilAlign(indexAxisSize_ * indexDSize_ * eachCalculationLines_, BLOCK_SIZE_T * DOUBLE_TIME);
        yBufferSize_ =
            HostCeilAlign(indexAxisSize_ * indexDSize_ * eachCalculationLines_, BLOCK_SIZE_T * xDsizeRatio_);
    } else if (batchProcess_) {
        int64_t perGroupSize = xAlignSize_ + indexAlignSize_ + yAlignSize_;
        eachCalculationLines_ = ubSizePlatForm_ / perGroupSize;
        xBufferSize_ = xAlignSize_ * eachCalculationLines_;
        yBufferSize_ = yAlignSize_ * eachCalculationLines_;
        indexBufferSize_ = indexAlignSize_ * eachCalculationLines_;
        if (HostCeilDiv(indexAxisSize_ * indexDSize_, BLOCK_SIZE_T) % DOUBLE_TIME == 1 &&
            indexRealDsize_ == INT64_DSIZE) {
            tilingData_.lastDimTiling.set_dataMoveUBStride(NUM_ONE);
        }
    } else if (xAlignSize_ <= ubSizePlatForm_ / DOUBLE_TIME) {
        xSliceNum_ = NUM_ONE;
        eachCalculationLines_ = NUM_ONE;
        xBufferSize_ = xAlignSize_;
        indexBufferSize_ = (ubSizePlatForm_ - xBufferSize_) / (indexDSize_ + xDSize_) * indexDSize_;
        indexBufferSize_ = HostCeilAlign(indexBufferSize_, BLOCK_SIZE_T * DOUBLE_TIME);
        yBufferSize_ = indexBufferSize_ / indexDSize_ * xDSize_;
        yBufferSize_ = HostCeilAlign(yBufferSize_, BLOCK_SIZE_T);
        indexSliceNum_ = HostCeilDiv(indexAlignSize_, indexBufferSize_);
        reservedIndexSize_ = indexAxisSize_ - (indexSliceNum_ - 1) * indexBufferSize_ / indexDSize_;
    } else {
        eachCalculationLines_ = NUM_ONE;
        xBufferSize_ = ubSizePlatForm_ / DOUBLE_TIME;
        xSliceNum_ = HostCeilDiv(xAlignSize_, xBufferSize_);
        reservedXSize_ = xAxisSize_ - (xSliceNum_ - 1) * xBufferSize_ / xDSize_;
        indexBufferSize_ =
            static_cast<float>(xBufferSize_) / (indexDSize_ + xDSize_ * DOUBLE_TIME + MASK_SIZE_RATE) * indexDSize_;
        indexBufferSize_ = HostCeilAlign(indexBufferSize_, BLOCK_SIZE_T * INT64_DSIZE);
        indexSliceNum_ = HostCeilDiv(indexAlignSize_, indexBufferSize_);
        yBufferSize_ = indexBufferSize_ / indexDSize_ * xDSize_;
        yBufferSize_ = HostCeilAlign(yBufferSize_, BLOCK_SIZE_T);
        reservedIndexSize_ = indexAxisSize_ - (indexSliceNum_ - 1) * indexBufferSize_ / indexDSize_;
        maskBufferSize_ = indexBufferSize_ / indexDSize_ / INT64_DSIZE;
        maskBufferSize_ = HostCeilAlign(maskBufferSize_, BLOCK_SIZE_T);
    }

    DoScalarMode();
}

void GatherElementsV2LastDimTiling::DoScalarMode()
{
    if (xSliceNum_ > MAX_SLICE_NUM || xAxisSize_ / indexAxisSize_ > MAX_SIZE_RATIO) {
        tilingData_.lastDimTiling.set_scalarMode(NUM_ONE);
        int64_t idxBLockSize = HostCeilDiv(indexAlignSize_, BLOCK_SIZE_T * DOUBLE_TIME) * DOUBLE_TIME;
        int64_t yBlockSize = HostCeilDiv(indexAxisSize_ * xRealDsize_, BLOCK_SIZE_T);
        eachCalculationLines_ = ubSizePlatForm_ / BLOCK_SIZE_T / (idxBLockSize + yBlockSize);
        bool multRowProcessFlag =
            eachCalculationLines_ > 1 &&
            (nonCollectingAxisSzie_ > totalCoreNum_ ||
             (nonCollectingAxisSzie_ <= totalCoreNum_ && indexAxisSize_ < totalCoreNum_ * BLOCK_SIZE_T));
        if (multRowProcessFlag) {
            indexBufferSize_ = eachCalculationLines_ * idxBLockSize * BLOCK_SIZE_T;
            yBufferSize_ = eachCalculationLines_ * yBlockSize * BLOCK_SIZE_T;
        } else {
            indexSliceNum_ = HostCeilDiv(indexAlignSize_, indexBufferSize_);
            scalarModeLength_ = indexSliceNum_ * nonCollectingAxisSzie_;
            if (scalarModeLength_ < totalCoreNum_) {
                indexSliceNum_ = totalCoreNum_ / nonCollectingAxisSzie_;
                scalarModeLength_ = indexSliceNum_ * nonCollectingAxisSzie_;
            }
            indexBufferSize_ =
                HostCeilAlign(indexAlignSize_ / indexSliceNum_ + INT64_DSIZE, BLOCK_SIZE_T * DOUBLE_TIME);
            yBufferSize_ = indexBufferSize_ / indexDSize_ * xRealDsize_;
            yBufferSize_ = HostCeilAlign(yBufferSize_, BLOCK_SIZE_T);
            reservedIndexSize_ = indexAxisSize_ - (indexSliceNum_ - 1) * indexBufferSize_ / indexDSize_;
        }
    }
}

void GatherElementsV2LastDimTiling::DoNeedUseCore()
{
    if (scalarModeLength_ == 0) {
        needUsedCore_ = nonCollectingAxisSzie_ > totalCoreNum_ ? totalCoreNum_ : nonCollectingAxisSzie_;
        formerCoreRowNum_ = nonCollectingAxisSzie_ / needUsedCore_;
        formerCoreNum_ = nonCollectingAxisSzie_ % needUsedCore_;
        tilingData_.lastDimTiling.set_formerCoreRowNum(formerCoreRowNum_);
        tilingData_.lastDimTiling.set_formerCoreNum(formerCoreNum_);
    } else {
        needUsedCore_ = scalarModeLength_ > totalCoreNum_ ? totalCoreNum_ : scalarModeLength_;
    }

    context_->SetBlockDim(needUsedCore_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
}

void GatherElementsV2LastDimTiling::SetTilingData()
{
    tilingData_.lastDimTiling.set_xBufferSize(xBufferSize_);
    tilingData_.lastDimTiling.set_indexBufferSize(indexBufferSize_);
    tilingData_.lastDimTiling.set_yBufferSize(yBufferSize_);
    tilingData_.lastDimTiling.set_maskBufferSize(maskBufferSize_);
    tilingData_.lastDimTiling.set_eachCalculationLines(eachCalculationLines_);
    tilingData_.lastDimTiling.set_xSliceNum(xSliceNum_);
    tilingData_.lastDimTiling.set_indexSliceNum(indexSliceNum_);
    tilingData_.lastDimTiling.set_reservedXSize(reservedXSize_);
    tilingData_.lastDimTiling.set_reservedIndexSize(reservedIndexSize_);
    tilingData_.lastDimTiling.set_scalarModeLength(scalarModeLength_);
    tilingData_.lastDimTiling.set_specialDataMove(specialDataMove_);
}

void GatherElementsV2LastDimTiling::PrintTilingData() const
{
    OP_LOGD(context_, "dimNum_:                    %ld", dimNum_);
    OP_LOGD(context_, "nonCollectingAxisSzie_:     %ld", nonCollectingAxisSzie_);
    OP_LOGD(context_, "xDSize_:                    %ld", xDSize_);
    OP_LOGD(context_, "indexDSize_:                %ld", indexDSize_);
    OP_LOGD(context_, "eachCalculationLines_:      %ld", eachCalculationLines_);
    OP_LOGD(context_, "xAxisSize_:                 %ld", xAxisSize_);
    OP_LOGD(context_, "indexAxisSize_:             %ld", indexAxisSize_);
    OP_LOGD(context_, "tilingKey:                  %ld", tilingKey_);
    OP_LOGD(context_, "xSliceNum_:                 %ld", xSliceNum_);
    OP_LOGD(context_, "indexSliceNum_:             %ld", indexSliceNum_);
    OP_LOGD(context_, "xAlignSize_:                %ld", xAlignSize_);
    OP_LOGD(context_, "indexAlignSize_:            %ld", indexAlignSize_);
    OP_LOGD(context_, "yAlignSize_:                %ld", yAlignSize_);
    OP_LOGD(context_, "reservedXSize_:             %ld", reservedXSize_);
    OP_LOGD(context_, "reservedIndexSize_:         %ld", reservedIndexSize_);
    OP_LOGD(context_, "needUsedCore_:              %ld", needUsedCore_);
    OP_LOGD(context_, "formerCoreRowNum_:          %ld", formerCoreRowNum_);
    OP_LOGD(context_, "indexBufferSize_:           %ld", indexBufferSize_);
    OP_LOGD(context_, "yBufferSize_:               %ld", yBufferSize_);
    OP_LOGD(context_, "xBufferSize_:               %ld", xBufferSize_);
    OP_LOGD(context_, "maskBufferSize_:            %ld", maskBufferSize_);
    OP_LOGD(context_, "specialDataMove_:           %ld", specialDataMove_);
    OP_LOGD(context_, "xDsizeRatio_:               %ld", xDsizeRatio_);
    OP_LOGD(context_, "scalarModeLength_:          %ld", scalarModeLength_);
    OP_LOGD(context_, "batchProcess_:               %d", batchProcess_);
}

ge::graphStatus GatherElementsV2LastDimTiling::DoOpTiling()
{
    GetDSize();
    IfEnableBatch();
    DoUBSlice();
    SetTilingData();
    DoNeedUseCore();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

// ========== Main tiling entry ==========

ge::graphStatus TilingGatherElementsV2(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingGatherElementsV2 enter.");
    if (context == nullptr) {
        OP_LOGE("GatherElementsV2", "The context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto attr = context->GetAttrs();
    OP_CHECK_IF(attr == nullptr, OP_LOGE(context->GetNodeName(), "attr is nullptr"), return ge::GRAPH_FAILED);
    auto dim = *(attr->GetAttrPointer<uint64_t>)(0);
    auto dimNum = context->GetInputShape(X_IDX)->GetStorageShape().GetDimNum();
    if (dimNum == 0UL) {
        OP_LOGE("GatherElementsV2", "x's dim num is 0.");
        return ge::GRAPH_FAILED;
    }
    if ((dim + dimNum) % dimNum != dimNum - 1) {
        GatherElementsV2Tiling tilingObject(context);
        OP_CHECK_IF(
            tilingObject.Init() != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "tiling init fail"),
            return ge::GRAPH_FAILED);
        return tilingObject.SetKernelTiling();
    } else {
        OP_LOGD(context->GetNodeName(), "TilingGatherElementsV2 last dim enter.");
        GatherElementsV2LastDimTiling tilingObject(context);
        OP_CHECK_IF(
            tilingObject.Init() != ge::GRAPH_SUCCESS, OP_LOGE(context->GetNodeName(), "tiling init fail"),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GatherElementsV2)
    .Tiling(TilingGatherElementsV2);
} // namespace optiling
