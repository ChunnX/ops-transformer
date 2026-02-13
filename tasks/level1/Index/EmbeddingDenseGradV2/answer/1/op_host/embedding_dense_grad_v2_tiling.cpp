#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "embedding_dense_grad_v2_tiling.h"

#define OP_LOGD(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGE(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) {                      \
        std::printf("nullptr error!");           \
        return ge::GRAPH_FAILED;                 \
    }

namespace optiling {
constexpr uint64_t VEC_PROCESS_SIZE = 256;
constexpr uint64_t SIZE_OF_FP32 = 4;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t BLOCK_SIZE_GRAD = 64;
constexpr uint64_t RESERVED_UB_SIZE = 20480;
constexpr uint64_t USE_IDX_NUM_IN_UB = 3;
constexpr uint64_t USE_GRAD_NUM_IN_UB = 3;
constexpr uint64_t USE_GRAD_NUM_IN_UB_WITH_CACHE = 4;
constexpr uint64_t DOUBLE_PRE_CORE = 2;
constexpr uint64_t SMALL_DIM_THRESHOLD = 512;
constexpr uint64_t CAST_MAX_NUM = 16777216;
constexpr uint64_t ALIGN_32_NUM_SPCAE = 300;
constexpr uint64_t UB_SORT_PART = 10;
constexpr uint64_t TILING_KEY_NORNAL = 0;
constexpr uint64_t TILING_KEY_SMALL_DIM = 100;
constexpr size_t EMBEDDING_DENSE_GRAD_ATTR_NUM_WEIGHTS = 0;
constexpr size_t EMBEDDING_DENSE_GRAD_ATTR_PADDING_IDX = 1;
constexpr size_t EMBEDDING_DENSE_GRAD_ATTR_SCALE_GRAD_BY_FREQ = 2;

class EmbeddingDenseGradV2Tiling {
public:
    explicit EmbeddingDenseGradV2Tiling(gert::TilingContext* context) : tilingContext_(context)
    {}
    ge::graphStatus Init();
    ge::graphStatus SetKernelTiling();
    void TilingDataPrint() const;

private:
    inline void SetTilingKeyMode();
    inline void SetDtypeSize();
    inline void BaseTiling(const int64_t gradRow);
    inline void Tiling4Scale();
    inline void CalMaxFormerNum(uint64_t ubSizeLeft);
    inline bool CheckIsSmallDim(const uint64_t gradLastDim) const;
    inline size_t CalcWorkSpaceSize();

    EmbeddingDenseGradV2TilingData tilingData_;
    gert::TilingContext* tilingContext_ = nullptr;
    uint64_t coreNum_ = 0;
    uint64_t numWeights_ = 0;
    uint64_t embeddingDim_ = 0;
    uint64_t paddingIdx_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t dataTypeSize_ = 0;
    bool scaleGrad_ = false;
    uint64_t tilingKey_ = 0;

private:
    uint64_t formerRow_ = 0;
    uint64_t tailRow_ = 0;
    uint64_t formerRowRepTime_ = 0;
    uint64_t computeMask_ = 0;
    uint64_t formerComputeRepTime_ = 0;
    uint64_t formerComputeFormerNum_ = 0;
    uint64_t formerComputeTailNum_ = 0;
    uint64_t tailComputeRepTime_ = 0;
    uint64_t tailComputeFormerNum_ = 0;
    uint64_t tailComputeTailNum_ = 0;
    uint64_t scaleWorkspaceLength_ = 0;
    uint64_t outStageWorkspaceLength_ = 0;
    uint64_t outIndexWorkspaceLength_ = 0;
    uint64_t outCastedWorkspaceLength_ = 0;

private:
    // small dim
    uint64_t partNum_ = 0;
    uint64_t formerCopyRow_ = 0;
    uint64_t tailCopyRow_ = 0;
    uint64_t formerCopyTime_ = 0;
    uint64_t tailCopyTime_ = 0;
    uint64_t maxRowInUb_ = 0;
    uint64_t formerLastRow_ = 0;
    uint64_t tailLastRow_ = 0;

private:
    // scale
    uint64_t tailCoreRowNum_ = 0;
    uint64_t formerCoreRowNum_ = 0;
    uint64_t scaleMask_ = 0;
    uint64_t scaleRepTime_ = 0;
    uint64_t scaleFormerNum_ = 0;
    uint64_t scaleTailNum_ = 0;
    uint64_t formerCoreRowRepTime_ = 0;

private:
    // determin
    bool isDeterministMode_ = false;
    uint64_t tailRowNum_ = 0;
    uint64_t formerRowNum_ = 0;
    uint64_t formerRowNumRepTime_ = 0;
    uint64_t determinMask_ = 0;
    uint64_t formerDeterminRepTime_ = 0;
    uint64_t formerDeterminFormerNum_ = 0;
    uint64_t formerDeterminTailNum_ = 0;
    uint64_t tailDeterminRepTime_ = 0;
    uint64_t tailDeterminFormerNum_ = 0;
    uint64_t tailDeterminTailNum_ = 0;
    uint64_t gradRow_ = 0;

private:
    // big shape
    uint64_t tailEmbeddingDim_ = 0;
    uint64_t formerEmbeddingDim_ = 0;
    uint64_t formerDimRepTime_ = 0;
    uint64_t maxFormerNum = 0; // max formerEmbeddingDim_ that ub can calculate
};

constexpr uint64_t RecursiveSum()
{
    return 0;
}

template <typename T, typename... Args> constexpr uint64_t RecursiveSum(T templateId, Args... templateIds)
{
    return static_cast<uint64_t>(templateId) + 10 * RecursiveSum(templateIds...);
}

inline void EmbeddingDenseGradV2Tiling::SetTilingKeyMode()
{
    bool isSmallDim = CheckIsSmallDim(embeddingDim_);
    tilingKey_ = RecursiveSum(scaleGrad_, isDeterministMode_, isSmallDim);
    tilingContext_->SetTilingKey(tilingKey_);
}

inline void EmbeddingDenseGradV2Tiling::Tiling4Scale()
{
    OP_LOGD(tilingContext_, "scaleTiling start");
    tailCoreRowNum_ = numWeights_ / coreNum_;
    formerCoreRowNum_ = tailCoreRowNum_ + 1UL;
    formerCoreRowRepTime_ = numWeights_ % coreNum_;
    scaleMask_ = VEC_PROCESS_SIZE / SIZE_OF_FP32;

    // formerDim compute params
    formerComputeRepTime_ = formerEmbeddingDim_ / scaleMask_;
    formerComputeFormerNum_ = formerComputeRepTime_ * scaleMask_;
    formerComputeTailNum_ = formerEmbeddingDim_ - formerComputeFormerNum_;

    // tailDim compute params
    tailComputeRepTime_ = tailEmbeddingDim_ / scaleMask_;
    tailComputeFormerNum_ = tailComputeRepTime_ * scaleMask_;
    tailComputeTailNum_ = tailEmbeddingDim_ - tailComputeFormerNum_;
    OP_LOGD(tilingContext_, "scaleTiling finish");
}

inline void EmbeddingDenseGradV2Tiling::BaseTiling(const int64_t gradRow)
{
    OP_LOGD(tilingContext_, "BaseTiling start");
    gradRow_ = static_cast<uint64_t>(gradRow);
    tailRow_ = static_cast<uint64_t>(gradRow) / coreNum_;
    formerRow_ = tailRow_ + 1UL;
    formerRowRepTime_ = static_cast<uint64_t>(gradRow) % coreNum_;

    formerEmbeddingDim_ = embeddingDim_ <= maxFormerNum ? embeddingDim_ : maxFormerNum;
    formerDimRepTime_ = embeddingDim_ / formerEmbeddingDim_;
    tailEmbeddingDim_ = embeddingDim_ - formerEmbeddingDim_ * formerDimRepTime_;

    // formerDim compute params
    computeMask_ = VEC_PROCESS_SIZE / SIZE_OF_FP32;
    formerComputeRepTime_ = formerEmbeddingDim_ / computeMask_;
    formerComputeFormerNum_ = formerComputeRepTime_ * computeMask_;
    formerComputeTailNum_ = formerEmbeddingDim_ - formerComputeFormerNum_;
    // tailDim compute params
    tailComputeRepTime_ = tailEmbeddingDim_ / computeMask_;
    tailComputeFormerNum_ = tailComputeRepTime_ * computeMask_;
    tailComputeTailNum_ = tailEmbeddingDim_ - tailComputeFormerNum_;

    OP_LOGD(tilingContext_, "BaseTiling finish");
    if (scaleGrad_) {
        Tiling4Scale();
    }
}

inline void EmbeddingDenseGradV2Tiling::CalMaxFormerNum(uint64_t ubSizeLeft)
{
    uint64_t idxAlignNum = BLOCK_SIZE / sizeof(int);
    uint64_t gradAlignNum = BLOCK_SIZE_GRAD / sizeof(float);
    ubSizeLeft -= RESERVED_UB_SIZE + idxAlignNum * sizeof(int) * USE_IDX_NUM_IN_UB;
    uint64_t availableUbForGrad = ubSizeLeft > 0UL ? ubSizeLeft : 0UL;
    uint64_t useGradNum = dataTypeSize_ == sizeof(float) ? USE_GRAD_NUM_IN_UB : USE_GRAD_NUM_IN_UB_WITH_CACHE;
    maxFormerNum = (availableUbForGrad / (gradAlignNum * sizeof(float) * useGradNum)) * gradAlignNum;
}

inline size_t EmbeddingDenseGradV2Tiling::CalcWorkSpaceSize()
{
    size_t alignNum = BLOCK_SIZE / sizeof(int32_t);
    uint64_t gradAlignNum = BLOCK_SIZE / sizeof(float);
    scaleWorkspaceLength_ = scaleGrad_ ? (((numWeights_ + alignNum - 1UL) / alignNum) * alignNum) : 0;
    outStageWorkspaceLength_ = ((formerEmbeddingDim_ * coreNum_ * DOUBLE_PRE_CORE + gradAlignNum -1UL) / gradAlignNum) * gradAlignNum;
    outIndexWorkspaceLength_ = coreNum_ * DOUBLE_PRE_CORE;
    outCastedWorkspaceLength_ = numWeights_ * embeddingDim_;
    size_t scaleWorkspaceSize = scaleWorkspaceLength_ * sizeof(int32_t);
    size_t outStageWorkspaceSize = outStageWorkspaceLength_ * sizeof(float);
    size_t outIndexWorkspaceSize = outIndexWorkspaceLength_ * sizeof(int32_t);
    size_t outCastedWorkspaceSize = outCastedWorkspaceLength_ * sizeof(float);
    size_t sysWorkspaceSize = 16UL * 1024UL * 1024UL;
    size_t baseWorkspaceSize = sysWorkspaceSize + scaleWorkspaceSize;
    if (dataTypeSize_ == sizeof(float)) {
        return baseWorkspaceSize;
    } else {
        return baseWorkspaceSize + outStageWorkspaceSize + outIndexWorkspaceSize;
    }
}

ge::graphStatus EmbeddingDenseGradV2Tiling::Init()
{
    OP_LOGD(tilingContext_, "Tiling initing");
    SetDtypeSize();
    isDeterministMode_ = (tilingContext_->GetDeterministic() == 1) && dataTypeSize_ == sizeof(float);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext_->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNum();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    auto attrs = tilingContext_->GetAttrs();
    auto selfShape = tilingContext_->GetInputShape(0)->GetStorageShape();
    int64_t gradRow = 1;
    for (size_t i = 0; i < selfShape.GetDimNum() - 1; i++) {
        gradRow *= selfShape.GetDim(i);
    }
    embeddingDim_ = selfShape.GetDim(selfShape.GetDimNum() - 1);
    numWeights_ = *(attrs->GetAttrPointer<uint64_t>)(EMBEDDING_DENSE_GRAD_ATTR_NUM_WEIGHTS);
    paddingIdx_ = *(attrs->GetAttrPointer<uint64_t>)(EMBEDDING_DENSE_GRAD_ATTR_PADDING_IDX);
    scaleGrad_ = *(attrs->GetAttrPointer<bool>)(EMBEDDING_DENSE_GRAD_ATTR_SCALE_GRAD_BY_FREQ);

    coreNum_ = scaleGrad_ ? std::min(
                                static_cast<uint64_t>(coreNum),
                                std::min(static_cast<uint64_t>(numWeights_), static_cast<uint64_t>(gradRow))) :
                            std::min(static_cast<uint64_t>(coreNum), static_cast<uint64_t>(gradRow));
    if (coreNum_ == 0UL || embeddingDim_ == 0UL) {
        OP_LOGE(tilingContext_, "coreNum %lu, embeddingDim %lu", coreNum_, embeddingDim_);
        return ge::GRAPH_FAILED;
    }
    CalMaxFormerNum(ubSize_);
    if (maxFormerNum == 0) {
        OP_LOGE(tilingContext_, "Do not have enough ub size.");
        return ge::GRAPH_FAILED;
    }
    SetTilingKeyMode();
    tilingContext_->SetNeedAtomic(true);
    BaseTiling(gradRow);

    size_t* currentWorkSpace = tilingContext_->GetWorkspaceSizes(1);
    currentWorkSpace[0] = CalcWorkSpaceSize();
    OP_LOGD(tilingContext_, "Tiling inited");
    return ge::GRAPH_SUCCESS;
}

void EmbeddingDenseGradV2Tiling::SetDtypeSize()
{
    auto const gradDtype = tilingContext_->GetInputDesc(0)->GetDataType();
    switch (gradDtype) {
        case ge::DT_FLOAT:
            dataTypeSize_ = 4; // 4 bytes for DT_FLOAT
            break;
        case ge::DT_FLOAT16:
            dataTypeSize_ = 2; // 2 bytes for DT_FLOAT16
            break;
        case ge::DT_BF16:
            dataTypeSize_ = 2; // 2 bytes for DT_BF16
            break;
        default:
            break;
    }
}

bool EmbeddingDenseGradV2Tiling::CheckIsSmallDim(const uint64_t gradLastDim) const
{
    // 确定性场景下bf16、fp16不走小尾轴分支
    if (tilingContext_->GetDeterministic() == 1 && dataTypeSize_ != sizeof(float)) {
        return false;
    }
    return !isDeterministMode_ && gradLastDim <= SMALL_DIM_THRESHOLD && numWeights_ <= CAST_MAX_NUM;
}

ge::graphStatus EmbeddingDenseGradV2Tiling::SetKernelTiling()
{
    tilingContext_->SetBlockDim(coreNum_);
    if (tilingKey_ == TILING_KEY_NORNAL) {
        tilingContext_->SetScheduleMode(1);
    }
    tilingData_.params.set_coreNum(coreNum_);
    tilingData_.params.set_tailRowNum(tailRow_);
    tilingData_.params.set_formerRowNum(formerRow_);
    tilingData_.params.set_formerRowRepTime(formerRowRepTime_);
    tilingData_.params.set_computeMask(computeMask_);
    tilingData_.params.set_formerComputeRepTime(formerComputeRepTime_);
    tilingData_.params.set_formerComputeFormerNum(formerComputeFormerNum_);
    tilingData_.params.set_formerComputeTailNum(formerComputeTailNum_);
    tilingData_.params.set_tailComputeRepTime(tailComputeRepTime_);
    tilingData_.params.set_tailComputeFormerNum(tailComputeFormerNum_);
    tilingData_.params.set_tailComputeTailNum(tailComputeTailNum_);
    tilingData_.params.set_embeddingDim(embeddingDim_);
    tilingData_.params.set_numWeights(numWeights_);
    tilingData_.params.set_paddingIdx(paddingIdx_);
    tilingData_.params.set_scaleGradByFreq(scaleGrad_);
    tilingData_.params.set_formerDimRepTime(formerDimRepTime_);
    tilingData_.params.set_formerEmbeddingDim(formerEmbeddingDim_);
    tilingData_.params.set_tailEmbeddingDim(tailEmbeddingDim_);
    tilingData_.params.set_scaleWorkspaceLength(scaleWorkspaceLength_);
    tilingData_.params.set_outStageWorkspaceLength(outStageWorkspaceLength_);
    tilingData_.params.set_outIndexWorkspaceLength(outIndexWorkspaceLength_);
    tilingData_.params.set_outCastedWorkspaceLength(outCastedWorkspaceLength_);

    tilingData_.scaleTiling.set_tailCoreRowNum(tailCoreRowNum_);
    tilingData_.scaleTiling.set_formerCoreRowNum(formerCoreRowNum_);
    tilingData_.scaleTiling.set_formerCoreRowRepTime(formerCoreRowRepTime_);
    tilingData_.scaleTiling.set_mask(scaleMask_);
    tilingData_.scaleTiling.set_formerComputeRepTime(formerComputeRepTime_);
    tilingData_.scaleTiling.set_formerComputeFormerNum(formerComputeFormerNum_);
    tilingData_.scaleTiling.set_formerComputeTailNum(formerComputeTailNum_);
    tilingData_.scaleTiling.set_tailComputeRepTime(tailComputeRepTime_);
    tilingData_.scaleTiling.set_tailComputeFormerNum(tailComputeFormerNum_);
    tilingData_.scaleTiling.set_tailComputeTailNum(tailComputeTailNum_);

    tilingData_.SaveToBuffer(
        tilingContext_->GetRawTilingData()->GetData(), tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    TilingDataPrint();
    return ge::GRAPH_SUCCESS;
}

void EmbeddingDenseGradV2Tiling::TilingDataPrint() const
{
    OP_LOGD(tilingContext_, "maxFormerNum:            %lu", maxFormerNum);
    OP_LOGD(tilingContext_, "coreNum:                 %lu", coreNum_);
    OP_LOGD(tilingContext_, "numWeights:              %lu", numWeights_);
    OP_LOGD(tilingContext_, "embeddingDim:            %lu", embeddingDim_);
    OP_LOGD(tilingContext_, "paddingIdx:              %lu", paddingIdx_);
    OP_LOGD(tilingContext_, "scaleGrad:               %d", scaleGrad_);
    OP_LOGD(tilingContext_, "gradRow:                 %lu", gradRow_);
    OP_LOGD(tilingContext_, "tailRow:                 %lu", tailRow_);
    OP_LOGD(tilingContext_, "formerRow:               %lu", formerRow_);
    OP_LOGD(tilingContext_, "formerRowRepTime:        %lu", formerRowRepTime_);
    OP_LOGD(tilingContext_, "scaleWorkspaceLength:    %lu", scaleWorkspaceLength_);
    OP_LOGD(tilingContext_, "outStageWorkspaceLength: %lu", outStageWorkspaceLength_);
    OP_LOGD(tilingContext_, "outIndexWorkspaceLength: %lu", outIndexWorkspaceLength_);
    OP_LOGD(tilingContext_, "outCastedWorkspaceLength: %lu", outCastedWorkspaceLength_);

    OP_LOGD(tilingContext_, "formerDimRepTime:        %lu", formerDimRepTime_);
    OP_LOGD(tilingContext_, "formerEmbeddingDim:      %lu", formerEmbeddingDim_);
    OP_LOGD(tilingContext_, "tailEmbeddingDim:        %lu", tailEmbeddingDim_);

    OP_LOGD(tilingContext_, "computeMask:             %lu", computeMask_);
    OP_LOGD(tilingContext_, "formerComputeRepTime:    %lu", formerComputeRepTime_);
    OP_LOGD(tilingContext_, "formerComputeFormerNum:  %lu", formerComputeFormerNum_);
    OP_LOGD(tilingContext_, "formerComputeTailNum:    %lu", formerComputeTailNum_);

    OP_LOGD(tilingContext_, "tailComputeRepTime:      %lu", tailComputeRepTime_);
    OP_LOGD(tilingContext_, "tailComputeFormerNum:    %lu", tailComputeFormerNum_);
    OP_LOGD(tilingContext_, "tailComputeTailNum:      %lu", tailComputeTailNum_);

    OP_LOGD(tilingContext_, "tailCoreRow:             %lu", tailCoreRowNum_);
    OP_LOGD(tilingContext_, "formerCoreRow:           %lu", formerCoreRowNum_);
    OP_LOGD(tilingContext_, "formerCoreRowRepTime:    %lu", formerCoreRowRepTime_);
    OP_LOGD(tilingContext_, "scaleMask:               %lu", scaleMask_);
    OP_LOGD(tilingContext_, "scaleRepTime:            %lu", scaleRepTime_);
    OP_LOGD(tilingContext_, "scaleFormerNum:          %lu", scaleFormerNum_);
    OP_LOGD(tilingContext_, "scaleTailNum:            %lu", scaleTailNum_);
}

ge::graphStatus TilingEmbeddingDenseGradV2(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE("EmbeddingDenseGradV2", "The context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Tiling4EmbeddingDenseGradV2 enter.");

    EmbeddingDenseGradV2Tiling tilingObject(context);
    if (tilingObject.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "tiling init fail");
        return ge::GRAPH_FAILED;
    }
    return tilingObject.SetKernelTiling();
}
}

namespace ge {
constexpr size_t EMBEDDING_DENSE_GRAD_IN_GRAD = 0;
constexpr size_t EMBEDDING_DENSE_GRAD_IN_INDICES = 1;
constexpr size_t EMBEDDING_DENSE_GRAD_IN_POSIDX = 2;
constexpr size_t EMBEDDING_DENSE_GRAD_OUT_Y = 0;
constexpr size_t EMBEDDING_DENSE_GRAD_ATTR_NUM_WEIGHTS = 0;
constexpr size_t EMBEDDING_DENSE_GRAD_ATTR_PADDING_IDX = 1;
constexpr size_t EMBEDDING_DENSE_GRAD_ATTR_SCALE_GRAD_BY_FREQ = 2;
static ge::graphStatus InferShapeForEmbeddingDenseGradV2(gert::InferShapeContext* context)
{
    auto grad_shape = context->GetInputShape(EMBEDDING_DENSE_GRAD_IN_GRAD);
    OP_CHECK_NULL_WITH_CONTEXT(context, grad_shape);
    auto indices_shape = context->GetInputShape(EMBEDDING_DENSE_GRAD_IN_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context, indices_shape);
    auto pos_shape = context->GetInputShape(EMBEDDING_DENSE_GRAD_IN_POSIDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, pos_shape);
    auto out_shape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    auto num_weights = attrs->GetAttrPointer<int64_t>(EMBEDDING_DENSE_GRAD_ATTR_NUM_WEIGHTS);
    OP_CHECK_NULL_WITH_CONTEXT(context, num_weights);
    auto padding_idx = attrs->GetAttrPointer<int64_t>(EMBEDDING_DENSE_GRAD_ATTR_PADDING_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, padding_idx);
    auto scale_grad_by_freq = attrs->GetAttrPointer<bool>(EMBEDDING_DENSE_GRAD_ATTR_SCALE_GRAD_BY_FREQ);
    OP_CHECK_NULL_WITH_CONTEXT(context, scale_grad_by_freq);

    int64_t output_shape_len = 2;
    int64_t input_shape_len = grad_shape->GetDimNum();
    out_shape->SetDimNum(output_shape_len);
    out_shape->SetDim(0, *num_weights);
    out_shape->SetDim(1, grad_shape->GetDim(input_shape_len - 1));

    return GRAPH_SUCCESS;
}
}

namespace ops {
class EmbeddingDenseGradV2 : public OpDef {
public:
    explicit EmbeddingDenseGradV2(const char* name) : OpDef(name)
    {
        this->Input("grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("sort_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("pos_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .InitValue(0);
        this->Attr("num_weights").AttrType(REQUIRED).Int();
        this->Attr("padding_idx").AttrType(OPTIONAL).Int(-1);
        this->Attr("scale_grad_by_freq").AttrType(OPTIONAL).Bool(false);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true);
        this->SetInferShape(ge::InferShapeForEmbeddingDenseGradV2);
        this->AICore()
            .SetTiling(optiling::TilingEmbeddingDenseGradV2);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(EmbeddingDenseGradV2);
}