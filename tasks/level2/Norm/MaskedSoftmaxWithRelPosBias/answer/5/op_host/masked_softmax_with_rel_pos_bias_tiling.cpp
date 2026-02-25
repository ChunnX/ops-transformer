/**
 * Tiling for MaskedSoftmaxWithRelPosBias - BWN branch only (answer_next 6).
 */
#include <map>
#include <cmath>
#include <cfloat>
#include <sstream>
#include <algorithm>
#include <numeric>
#include "register/op_impl_registry.h"
#include "common/log.h"
#include "register/tilingdata_base.h"
#include "exe_graph/runtime/shape.h"
#include "common/tiling_base.h"
#include "common/tiling_templates_registry.h"
#include "exe_graph/runtime/tiling_context.h"
#include "graph/utils/type_utils.h"
#include "tiling/tiling_api.h"
#include "masked_softmax_with_rel_pos_bias_tiling.h"

namespace optiling {
using namespace OpTiling;
static const uint32_t BYTE_BLOCK = 32;
static const int32_t DIM_0 = 0;
static const int32_t DIM_1 = 1;
static const int32_t DIM_2 = 2;
static const int32_t DIM_3 = 3;
static const int32_t DIM_4 = 4;
static const int32_t DIM_5 = 5;
static const int32_t MIN_WORKSPACE_SIZE = 16 * 1024 * 1024;
static const int32_t BUFF_NUM = 2;

static const int32_t X_INPUT_INDEX = 0;
static const int32_t ATTEN_INPUT_INDEX = 1;
static const int32_t BIAS_INPUT_INDEX = 2;
static const int32_t Y_OUTPUT_INDEX = 0;
static constexpr int32_t MASKED_SOFTMAX_INVALED_KEY = 0;

static constexpr int32_t BWN_FLOAT_ATTEN_MULS = 301;
static constexpr int32_t BWN_FLOAT_ATTEN = 302;
static constexpr int32_t BWN_FLOAT_MULS = 303;
static constexpr int32_t BWN_FLOAT_NULL = 304;
static constexpr int32_t BWN_FLOAT16_ATTEN_MULS = 311;
static constexpr int32_t BWN_FLOATA16_ATTEN = 312;
static constexpr int32_t BWN_FLOAT16_MULS = 313;
static constexpr int32_t BWN_FLOAT16_NULL = 314;
static constexpr int32_t BWN_BF16_ATTEN_MULS = 321;
static constexpr int32_t BWN_BF16_ATTEN = 322;
static constexpr int32_t BWN_BF16_MULS = 323;
static constexpr int32_t BWN_BF16_NULL = 324;

class MaskedSoftmaxWithRelPosBiasBaseTiling : public TilingBaseClass {
 public:
  explicit MaskedSoftmaxWithRelPosBiasBaseTiling(gert::TilingContext* context) : TilingBaseClass(context) {}
  virtual ~MaskedSoftmaxWithRelPosBiasBaseTiling() override = default;

 protected:
  ge::graphStatus GetPlatformInfo() override;
  ge::graphStatus GetShapeAttrsInfo() override;
  ge::graphStatus DoLibApiTiling() override { return ge::GRAPH_SUCCESS; };
  ge::graphStatus GetWorkspaceSize() override;

 private:
  ge::graphStatus CheckOutShape();
  ge::graphStatus GetXTensorDims(const gert::StorageShape *x_shape);
  ge::graphStatus GetAttenMaskTensorDims(const gert::StorageShape *attention_shape);
  ge::graphStatus GetTensorDims();
  ge::graphStatus GetBiasTensorDims();

 protected:
  uint32_t aivNum{0};
  ge::DataType dataType{ge::DT_FLOAT};
  uint32_t b_{0};
  uint32_t w_{0};
  uint32_t n_{0};
  uint32_t s1_{0};
  uint32_t s2_{0};
  uint32_t att_b_{0};
  uint32_t att_w_{0};
  uint32_t att_n_{0};
  uint32_t att_s1_{0};
  uint32_t att_s2_{0};
  uint32_t bias_b_{0};
  uint32_t bias_w_{0};
  uint32_t bias_n_{0};
  uint32_t bias_s1_{0};
  uint32_t bias_s2_{0};
  uint32_t dtypeSize{0};
  int32_t alignNum{0};
  bool isAligned{false};
  uint32_t s2AlignedSize{0};
  uint32_t wns1s2{0};
  uint32_t totalSizeAligned{0};
  bool existAtten{false};
  bool existMuls{false};
  float scaleValue{1.0f};
  uint32_t softMaskMinTmpSize{0};
  uint64_t bf16AttenAndBiasCastTempSize{0};
  uint64_t bf16XCastTempSize{0};
  platform_ascendc::SocVersion socVersion{platform_ascendc::SocVersion::RESERVED_VERSION};
  MaskedSoftmaxWithRelPosBiasTilingData tilingData;
};

static const uint32_t DEFAULT_AIV_NUM_910B = 20;
static const uint64_t DEFAULT_UB_SIZE_910B = 512 * 1024;

ge::graphStatus MaskedSoftmaxWithRelPosBiasBaseTiling::GetPlatformInfo() {
  auto platformInfo = context_->GetPlatformInfo();
  if (platformInfo != nullptr) {
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    aivNum = ascendcPlatform.GetCoreNumAiv();
    aicoreParams_.blockDim = ascendcPlatform.GetCoreNum();
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = (ubSizePlatForm > 0) ? ubSizePlatForm : DEFAULT_UB_SIZE_910B;
    socVersion = ascendcPlatform.GetSocVersion();
    if (aivNum == 0) aivNum = DEFAULT_AIV_NUM_910B;
    if (aicoreParams_.ubSize == 0) aicoreParams_.ubSize = DEFAULT_UB_SIZE_910B;
  } else {
    auto compileInfoPtr = reinterpret_cast<const CompileInfoCommon*>(context_->GetCompileInfo());
    if (compileInfoPtr != nullptr && compileInfoPtr->aivNum > 0 && compileInfoPtr->ubSize > 0) {
      aivNum = compileInfoPtr->aivNum;
      aicoreParams_.ubSize = compileInfoPtr->ubSize;
      aicoreParams_.blockDim = compileInfoPtr->coreNum > 0 ?
          static_cast<uint64_t>(compileInfoPtr->coreNum) : compileInfoPtr->aivNum;
      socVersion = static_cast<platform_ascendc::SocVersion>(compileInfoPtr->socVersion);
    } else {
      aivNum = DEFAULT_AIV_NUM_910B;
      aicoreParams_.ubSize = DEFAULT_UB_SIZE_910B;
      aicoreParams_.blockDim = DEFAULT_AIV_NUM_910B;
      socVersion = platform_ascendc::SocVersion::ASCEND910B;
    }
  }
  if ((!isAligned) && (socVersion == platform_ascendc::SocVersion::ASCEND310P)) {
    OP_LOGE(context_, "Do tiling failed, s2 is not aligned.");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBaseTiling::CheckOutShape() {
  auto y_shape = context_->GetOutputShape(Y_OUTPUT_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, y_shape);
  if (y_shape->GetStorageShape().GetDimNum() == 5) {
    if (y_shape->GetStorageShape().GetDim(DIM_0) != b_ || y_shape->GetStorageShape().GetDim(DIM_1) != w_ ||
        y_shape->GetStorageShape().GetDim(DIM_2) != n_ || y_shape->GetStorageShape().GetDim(DIM_3) != s1_ ||
        y_shape->GetStorageShape().GetDim(DIM_4) != s2_) {
      OP_LOGE(context_, "Do tiling failed, output shape is not legal, please check it.");
      return ge::GRAPH_FAILED;
    }
  } else if (y_shape->GetStorageShape().GetDimNum() == 4) {
    if (y_shape->GetStorageShape().GetDim(DIM_0) != b_ * w_ || y_shape->GetStorageShape().GetDim(DIM_1) != n_ ||
        y_shape->GetStorageShape().GetDim(DIM_2) != s1_ || y_shape->GetStorageShape().GetDim(DIM_3) != s2_) {
      OP_LOGE(context_, "Do tiling failed, output shape is not legal, please check it.");
      return ge::GRAPH_FAILED;
    }
  } else {
    OP_LOGE(context_, "Do tiling failed, output shape is not legal, please check it.");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBaseTiling::GetXTensorDims(const gert::StorageShape *x_shape) {
  if (x_shape->GetStorageShape().GetDimNum() == 5) {
    b_ = static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(DIM_0));
    w_ = static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(DIM_1));
    n_ = static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(DIM_2));
    s1_ = static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(DIM_3));
    s2_ = static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(DIM_4));
  } else if (x_shape->GetStorageShape().GetDimNum() == 4) {
    b_ = static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(DIM_0));
    w_ = 1;
    n_ = static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(DIM_1));
    s1_ = static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(DIM_2));
    s2_ = static_cast<uint32_t>(x_shape->GetStorageShape().GetDim(DIM_3));
  } else {
    OP_LOGE(context_, "Do tiling failed, x dim num only support 4 or 5, but got dim num is %zu",
            x_shape->GetStorageShape().GetDimNum());
    return ge::GRAPH_FAILED;
  }
  if (b_ == 0 || w_ == 0 || n_ == 0 || s1_ == 0 || s2_ == 0) {
    OP_LOGE(context_, "Do tiling failed, b_[%u], w_[%u], n_[%u], s1[%u], s2_[%u]", b_, w_, n_, s1_, s2_);
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBaseTiling::GetAttenMaskTensorDims(
    const gert::StorageShape* attention_shape) {
  auto attenDimNum = attention_shape->GetStorageShape().GetDimNum();
  if (attenDimNum == DIM_3) {
    att_b_ = 1;
    att_w_ = static_cast<uint32_t>(attention_shape->GetStorageShape().GetDim(DIM_0));
    att_n_ = 1;
    att_s1_ = static_cast<uint32_t>(attention_shape->GetStorageShape().GetDim(DIM_1));
    att_s2_ = static_cast<uint32_t>(attention_shape->GetStorageShape().GetDim(DIM_2));
  } else if (attenDimNum == DIM_4) {
    att_b_ = 1;
    att_w_ = static_cast<uint32_t>(attention_shape->GetStorageShape().GetDim(DIM_0));
    att_n_ = static_cast<uint32_t>(attention_shape->GetStorageShape().GetDim(DIM_1));
    att_s1_ = static_cast<uint32_t>(attention_shape->GetStorageShape().GetDim(DIM_2));
    att_s2_ = static_cast<uint32_t>(attention_shape->GetStorageShape().GetDim(DIM_3));
  } else if (attenDimNum == DIM_5) {
    att_b_ = static_cast<uint32_t>(attention_shape->GetStorageShape().GetDim(DIM_0));
    att_w_ = static_cast<uint32_t>(attention_shape->GetStorageShape().GetDim(DIM_1));
    att_n_ = static_cast<uint32_t>(attention_shape->GetStorageShape().GetDim(DIM_2));
    att_s1_ = static_cast<uint32_t>(attention_shape->GetStorageShape().GetDim(DIM_3));
    att_s2_ = static_cast<uint32_t>(attention_shape->GetStorageShape().GetDim(DIM_4));
  }
  if (att_b_ != 1 || att_n_ != 1) {
    OP_LOGE(context_, "Do tiling failed, att_b_[%u], att_n_[%u]", att_b_, att_n_);
    return ge::GRAPH_FAILED;
  }
  if (att_b_ == 0 || att_w_ == 0 || att_n_ == 0 || att_s1_ == 0 || att_s2_ == 0) {
    OP_LOGE(context_, "Do tiling failed, att_b_[%u], att_w_[%u], att_n_[%u], att_s1_[%u], att_s2_[%u]", att_b_,
            att_w_, att_n_, att_s1_, att_s2_);
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBaseTiling::GetBiasTensorDims() {
  auto bias_shape = context_->GetInputShape(BIAS_INPUT_INDEX);
  if (bias_shape == nullptr) {
    bias_shape = context_->GetInputShape(ATTEN_INPUT_INDEX);
    existAtten = false;
    OP_LOGD(context_, "Do tiling, the bias shape is null");
  }
  OP_CHECK_NULL_WITH_CONTEXT(context_, bias_shape);
  auto biasDimNum = bias_shape->GetStorageShape().GetDimNum();
  if (biasDimNum == DIM_3) {
    bias_b_ = 1;
    bias_w_ = 1;
    bias_n_ = static_cast<uint32_t>(bias_shape->GetStorageShape().GetDim(DIM_0));
    bias_s1_ = static_cast<uint32_t>(bias_shape->GetStorageShape().GetDim(DIM_1));
    bias_s2_ = static_cast<uint32_t>(bias_shape->GetStorageShape().GetDim(DIM_2));
  } else if (biasDimNum == DIM_4) {
    bias_b_ = 1;
    bias_w_ = static_cast<uint32_t>(bias_shape->GetStorageShape().GetDim(DIM_0));
    bias_n_ = static_cast<uint32_t>(bias_shape->GetStorageShape().GetDim(DIM_1));
    bias_s1_ = static_cast<uint32_t>(bias_shape->GetStorageShape().GetDim(DIM_2));
    bias_s2_ = static_cast<uint32_t>(bias_shape->GetStorageShape().GetDim(DIM_3));
  } else if (biasDimNum == DIM_5) {
    bias_b_ = static_cast<uint32_t>(bias_shape->GetStorageShape().GetDim(DIM_0));
    bias_w_ = static_cast<uint32_t>(bias_shape->GetStorageShape().GetDim(DIM_1));
    bias_n_ = static_cast<uint32_t>(bias_shape->GetStorageShape().GetDim(DIM_2));
    bias_s1_ = static_cast<uint32_t>(bias_shape->GetStorageShape().GetDim(DIM_3));
    bias_s2_ = static_cast<uint32_t>(bias_shape->GetStorageShape().GetDim(DIM_4));
  }
  if (bias_b_ != 1 || bias_w_ != 1) {
    OP_LOGE(context_, "Do tiling failed, bias_b_[%u], bias_w_[%u]", bias_b_, bias_w_);
    return ge::GRAPH_FAILED;
  }
  if (bias_b_ == 0 || bias_w_ == 0 || bias_n_ == 0 || bias_s1_ == 0 || bias_s2_ == 0) {
    OP_LOGE(context_, "Do tiling failed, bias_b_[%u], bias_w_[%u], bias_n_[%u], bias_s1_[%u], bias_s2_[%u]", bias_b_,
            bias_w_, bias_n_, bias_s1_, bias_s2_);
    return ge::GRAPH_FAILED;
  }
  if (!(n_ == bias_n_ && s1_ == bias_s1_ && s2_ == bias_s2_)) {
    OP_LOGE(context_, "Do tiling failed, n_[%u], bias_n_[%u], s1_[%u], bias_s1_[%u], s2_[%u], bias_s2_[%u]", n_, bias_n_,
            s1_, bias_s1_, s2_, bias_s2_);
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBaseTiling::GetTensorDims() {
  auto x_shape = context_->GetInputShape(X_INPUT_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context_, x_shape);
  if (GetXTensorDims(x_shape) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  auto xDesc = context_->GetInputDesc(0);
  OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
  dataType = xDesc->GetDataType();
  dtypeSize = static_cast<uint32_t>(ge::GetSizeByDataType(dataType));
  alignNum = BYTE_BLOCK / dtypeSize;
  s2AlignedSize = (s2_ + alignNum - 1) / alignNum * alignNum;
  if (s2AlignedSize == s2_) {
    isAligned = true;
  }
  wns1s2 = w_ * n_ * s1_ * s2_;
  auto attenMaskTensor = context_->GetOptionalInputTensor(1);
  existAtten = false;
  if (attenMaskTensor != nullptr) {
    existAtten = true;
    auto attention_shape = context_->GetOptionalInputShape(ATTEN_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, attention_shape);
    if (GetAttenMaskTensorDims(attention_shape) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
    if (x_shape->GetStorageShape().GetDimNum() == 4) {
      if ((b_ / att_w_) == 0) {
        OP_LOGE(context_, "Do tiling failed, x and attenMask shape not compatible.");
        return ge::GRAPH_FAILED;
      }
      b_ = b_ / att_w_;
      w_ = att_w_;
    }
    if (!(w_ == att_w_ && s1_ == att_s1_ && s2_ == att_s2_)) {
      OP_LOGE(context_, "Do tiling failed, w_[%u], att_w_[%u], s1_[%u], att_s1_[%u], s2_[%u], att_s2_[%u]", w_, att_w_,
              s1_, att_s1_, s2_, att_s2_);
      return ge::GRAPH_FAILED;
    }
  }
  if (GetBiasTensorDims() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBaseTiling::GetShapeAttrsInfo() {
  if (GetTensorDims() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  if (CheckOutShape() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  auto attrs = context_->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
  const float* scaleValueAttr = attrs->GetAttrPointer<float>(0);
  OP_CHECK_NULL_WITH_CONTEXT(context_, scaleValueAttr);
  if (std::fabs(*scaleValueAttr - 1) > FLT_EPSILON) {
    existMuls = true;
    scaleValue = *scaleValueAttr;
  } else {
    existMuls = false;
  }
  tilingData.set_w(w_);
  tilingData.set_n(n_);
  tilingData.set_s1(s1_);
  tilingData.set_s2(s2_);
  tilingData.set_wns1s2(w_ * n_ * s1_ * s2_);
  tilingData.set_wns1(w_ * n_ * s1_);
  tilingData.set_ws1s2(w_ * s1_ * s2_);
  tilingData.set_ns1s2(n_ * s1_ * s2_);
  tilingData.set_ws1(w_ * s1_);
  tilingData.set_ns1(n_ * s1_);
  tilingData.set_s1s2(s1_ * s2_);
  tilingData.set_wns1s2Aligned(w_ * n_ * s1_ * s2AlignedSize);
  tilingData.set_s1s2Aligned(s1_ * s2AlignedSize);
  tilingData.set_ns1s2Aligned(n_ * s1_ * s2AlignedSize);
  tilingData.set_s2Aligned(s2AlignedSize);
  tilingData.set_s2DtypeSize(s2_ * dtypeSize);
  tilingData.set_scaleValue(scaleValue);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBaseTiling::GetWorkspaceSize() {
  size_t* workspaces = context_->GetWorkspaceSizes(1);
  OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
  if (socVersion == platform_ascendc::SocVersion::ASCEND910B) {
    workspaces[0] = MIN_WORKSPACE_SIZE;
  } else {
    workspaces[0] = 0;
  }
  return ge::GRAPH_SUCCESS;
}

class MaskedSoftmaxWithRelPosBiasBWNTiling : public MaskedSoftmaxWithRelPosBiasBaseTiling {
 public:
  explicit MaskedSoftmaxWithRelPosBiasBWNTiling(gert::TilingContext* context)
      : MaskedSoftmaxWithRelPosBiasBaseTiling(context) {}
  ~MaskedSoftmaxWithRelPosBiasBWNTiling() override = default;

  void Reset(gert::TilingContext* context) override {
    MaskedSoftmaxWithRelPosBiasBaseTiling::Reset(context);
    SetDate();
  }

 protected:
  bool IsCapable() override;
  ge::graphStatus DoOpTiling() override;
  uint64_t GetTilingKey() const override;
  ge::graphStatus PostTiling() override;
  void SetDate();
  uint64_t minUsedSize{0};
};

bool MaskedSoftmaxWithRelPosBiasBWNTiling::IsCapable() {
  if (b_ * w_ * n_ < aivNum) {
    OP_LOGD(context_, "Do BWN tiling: b*w*n (%u) < aivNum (%u), still allow for small batch.", b_ * w_ * n_, aivNum);
  }
  if (b_ == 0 || w_ == 0 || n_ == 0) {
    OP_LOGD(context_, "Do BWN tiling failed, b*w*n must be > 0.");
    return false;
  }
  minUsedSize = 0;
  if (dataType != ge::DT_FLOAT) {
    minUsedSize += static_cast<uint64_t>(s1_) * s2AlignedSize * sizeof(float) * 3;
    if (existAtten) {
      minUsedSize += static_cast<uint64_t>(s1_) * s2AlignedSize * sizeof(float);
    }
  }
  minUsedSize += static_cast<uint64_t>(s1_) * s2AlignedSize * dtypeSize * 3 * BUFF_NUM;
  if (existAtten) {
    minUsedSize += static_cast<uint64_t>(s1_) * s2AlignedSize * dtypeSize * BUFF_NUM;
  }
  ge::Shape softMaxSrcShape({1, s2AlignedSize});
  if (dataType != ge::DT_FLOAT) {
    softMaskMinTmpSize = s1_ * AscendC::GetSoftMaxFlashV2MinTmpSize(softMaxSrcShape, dtypeSize * 2, dtypeSize * 2, false, false);
  } else {
    softMaskMinTmpSize = s1_ * AscendC::GetSoftMaxFlashV2MinTmpSize(softMaxSrcShape, dtypeSize, dtypeSize, false, false);
  }
  minUsedSize += static_cast<uint64_t>(softMaskMinTmpSize);
  if (minUsedSize <= aicoreParams_.ubSize) {
    OP_LOGD(context_, "Do BWN tiling success, minUsedSize is %lu.", minUsedSize);
    return true;
  }
  OP_LOGD(context_, "Do BWN tiling failed, minUsedSize is %lu, ubSize is %lu.", minUsedSize, aicoreParams_.ubSize);
  return false;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBWNTiling::DoOpTiling() {
  int64_t totalBWN = static_cast<int64_t>(b_) * w_ * n_;
  if (totalBWN == 0) {
    return ge::GRAPH_FAILED;
  }
  int64_t singleCoreSize = totalBWN / aivNum;
  int64_t tailSize = totalBWN - singleCoreSize * aivNum;
  int64_t tailStartCoreIdx = aivNum;
  if (tailSize != 0) {
    tailStartCoreIdx = tailSize;
    singleCoreSize += 1;
  }
  uint32_t stackNum = aicoreParams_.ubSize / static_cast<uint32_t>(minUsedSize);
  if (stackNum > n_) {
    stackNum = n_;
  }
  tilingData.set_stackNum(stackNum);
  tilingData.set_tailStartCoreIdx(static_cast<uint32_t>(tailStartCoreIdx));
  tilingData.set_singleCoreSize(static_cast<uint32_t>(singleCoreSize));
  tilingData.set_w(w_);
  tilingData.set_n(n_);
  tilingData.set_s1(s1_);
  tilingData.set_s2(s2_);

  auto softMaxShape = ge::Shape({stackNum, s1_, s2AlignedSize});
  uint32_t localWorkSpaceSize = softMaskMinTmpSize * stackNum;
  AscendC::SoftMaxFlashV2TilingFunc(softMaxShape, sizeof(float), sizeof(float), localWorkSpaceSize,
                                    tilingData.softmaxTilingData, false, false);
  uint32_t tail = singleCoreSize % stackNum;
  if (tail != 0) {
    auto softMaxShapeTail = ge::Shape({tail, s1_, s2AlignedSize});
    AscendC::SoftMaxFlashV2TilingFunc(softMaxShapeTail, sizeof(float), sizeof(float), localWorkSpaceSize,
                                      tilingData.tailSoftmaxTilingData, false, false);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBWNTiling::PostTiling() {
  tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  context_->SetBlockDim(std::min(static_cast<uint32_t>(b_ * w_ * n_), aivNum));
  return ge::GRAPH_SUCCESS;
}

uint64_t MaskedSoftmaxWithRelPosBiasBWNTiling::GetTilingKey() const {
  if (dataType == ge::DT_FLOAT) {
    if (existAtten && existMuls) return BWN_FLOAT_ATTEN_MULS;
    if (existAtten) return BWN_FLOAT_ATTEN;
    if (existMuls) return BWN_FLOAT_MULS;
    return BWN_FLOAT_NULL;
  }
  if (dataType == ge::DT_FLOAT16) {
    if (existAtten && existMuls) return BWN_FLOAT16_ATTEN_MULS;
    if (existAtten) return BWN_FLOATA16_ATTEN;
    if (existMuls) return BWN_FLOAT16_MULS;
    return BWN_FLOAT16_NULL;
  }
  if (dataType == ge::DT_BF16) {
    if (existAtten && existMuls) return BWN_BF16_ATTEN_MULS;
    if (existAtten) return BWN_BF16_ATTEN;
    if (existMuls) return BWN_BF16_MULS;
    return BWN_BF16_NULL;
  }
  return BWN_FLOAT_NULL;
}

void MaskedSoftmaxWithRelPosBiasBWNTiling::SetDate() {
  tilingData.SetDataPtr(context_->GetRawTilingData()->GetData());
}

REGISTER_TILING_TEMPLATE("MaskedSoftmaxWithRelPosBias", MaskedSoftmaxWithRelPosBiasBWNTiling, 102);
static OpTiling::Register VAR_UNUSED Norm_MaskedSoftmaxWithRelPosBias_BWN_102_register =
    OpTiling::Register("Norm/MaskedSoftmaxWithRelPosBias").tiling<MaskedSoftmaxWithRelPosBiasBWNTiling>(102);

static ge::graphStatus TilingMaskedSoftmaxWithRelPosBias(gert::TilingContext* context) {
  return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForMaskedSoftmaxWithRelPosBias(gert::TilingParseContext* context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

struct MaskedSoftmaxWithRelPosBiasCompileInfo {};

IMPL_OP_OPTILING(MaskedSoftmaxWithRelPosBias)
    .Tiling(TilingMaskedSoftmaxWithRelPosBias)
    .TilingParse<MaskedSoftmaxWithRelPosBiasCompileInfo>(TilingPrepareForMaskedSoftmaxWithRelPosBias);
}  // namespace optiling
