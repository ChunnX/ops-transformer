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

static constexpr int32_t BWNS1_FLOAT_MULS = 203;
static constexpr int32_t BWNS1_FLOAT_NULL = 204;
static constexpr int32_t BWNS1_FLOAT16_MULS = 213;
static constexpr int32_t BWNS1_FLOAT16_NULL = 214;
static constexpr int32_t BWNS1_BF16_MULS = 223;
static constexpr int32_t BWNS1_BF16_NULL = 224;

static constexpr int32_t BWNS1_FLOAT_MULS_NOT_ALIGEND = 2103;
static constexpr int32_t BWNS1_FLOAT_NULL_NOT_ALIGEND = 2104;
static constexpr int32_t BWNS1_FLOAT16_MULS_NOT_ALIGEND = 2113;
static constexpr int32_t BWNS1_FLOAT16_NULL_NOT_ALIGEND = 2114;
static constexpr int32_t BWNS1_BF16_MULS_NOT_ALIGEND = 2123;
static constexpr int32_t BWNS1_BF16_NULL_NOT_ALIGEND = 2124;

class MaskedSoftmaxWithRelPosBiasBaseTiling : public TilingBaseClass {
 public:
  explicit MaskedSoftmaxWithRelPosBiasBaseTiling(gert::TilingContext* context) : TilingBaseClass(context) {}
  virtual ~MaskedSoftmaxWithRelPosBiasBaseTiling() override = default;

 protected:
  ge::graphStatus GetPlatformInfo() override;
  ge::graphStatus GetShapeAttrsInfo() override;
  ge::graphStatus DoLibApiTiling() override {
    return ge::GRAPH_SUCCESS;
  };
  ge::graphStatus GetWorkspaceSize() override;

 private:
  ge::graphStatus CheckOutShape();
  ge::graphStatus GetXTensorDims(const gert::StorageShape *x_shape);
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
    OP_LOGE(context_, "Do tiling failed, b_[%u]), w_[%u], n_[%u], s1[%u], s2_[%u]", b_, w_, n_, s1_, s2_);
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
    OP_LOGE(context_, "Do tiling failed, bias_b_[%u]), bias_b_[%u]", bias_b_, bias_w_);
    return ge::GRAPH_FAILED;
  }
  if (bias_b_ == 0 || bias_w_ == 0 || bias_n_ == 0 || bias_s1_ == 0 || bias_s2_ == 0) {
    OP_LOGE(context_, "Do tiling failed,  bias_b_[%u],  bias_w_[%u], bias_n_[%u], bias_s1_[%u], bias_s2_[%u]", bias_b_,
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
  existAtten = false;

  if (GetBiasTensorDims() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasBaseTiling::GetShapeAttrsInfo() {
  if(GetTensorDims() != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }

  if (CheckOutShape() != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
  }

  auto attrs = context_->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
  const float* scaleValueAttr = attrs->GetAttrPointer<float>(0);
  OP_CHECK_NULL_WITH_CONTEXT(context_, scaleValueAttr);
  OP_LOGD(context_, "scaleValue %f.", *scaleValueAttr);
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

class MaskedSoftmaxWithRelPosBiasS1BiasTiling : public MaskedSoftmaxWithRelPosBiasBaseTiling {
 public:
  explicit MaskedSoftmaxWithRelPosBiasS1BiasTiling(gert::TilingContext* context)
      : MaskedSoftmaxWithRelPosBiasBaseTiling(context) {
  }
  ~MaskedSoftmaxWithRelPosBiasS1BiasTiling() override = default;

 protected:
  bool IsCapable() override;
  ge::graphStatus DoSoftMaxFlashV2Tiling(uint32_t loopTailNum, uint32_t stackNum, uint32_t loopNum);
  ge::graphStatus DoOpTiling() override;
  uint64_t GetTilingKey() const override;
  ge::graphStatus PostTiling() override;

 private:
  uint64_t xSize{0};
  uint64_t biasSize{0};
  uint64_t bf16ABiasCastTempSize{0};
  uint64_t bf16YCastTempSize{0};
};

bool MaskedSoftmaxWithRelPosBiasS1BiasTiling::IsCapable() {
  if (existAtten) {
    OP_LOGD(context_, "Do tiling failed, existAtten is false.");
    return false;
  }
  xSize = static_cast<uint64_t>(s2AlignedSize) * dtypeSize;
  uint64_t xySize = 2 * xSize;
  biasSize = static_cast<uint64_t>(s2AlignedSize) * dtypeSize;
  ge::Shape softMaxSrcShape({1, s2AlignedSize});
  if (dataType != ge::DT_FLOAT) {
    softMaskMinTmpSize =
        AscendC::GetSoftMaxFlashV2MinTmpSize(softMaxSrcShape, dtypeSize * 2, dtypeSize * 2, false, false);
    bf16XCastTempSize = xSize * 2;
    bf16ABiasCastTempSize = biasSize * 2;
    bf16YCastTempSize = xSize * 2;
  } else {
    softMaskMinTmpSize = AscendC::GetSoftMaxFlashV2MinTmpSize(softMaxSrcShape, dtypeSize, dtypeSize, false, false);
  }
  auto totalSize = (xySize + biasSize) * 2 + softMaskMinTmpSize + bf16XCastTempSize + bf16ABiasCastTempSize +
                   bf16YCastTempSize;
  if (totalSize <= aicoreParams_.ubSize) {
    OP_LOGD(context_, "Do BWNS1 tiling success with no atten, totalSize is %lu, dtypeSize is %u.",
                totalSize, dtypeSize);
    return true;
  } else {
    OP_LOGE(
        context_,
        "Do BWNS1 tiling failed with no atten, min used size is %lu which has execeed UB space %lu, dtypeSize is "
        "%u, please check your shape size.",
        totalSize, aicoreParams_.ubSize, dtypeSize);
    return false;
  }
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasS1BiasTiling::DoSoftMaxFlashV2Tiling(uint32_t loopTailNum, uint32_t stackNum, uint32_t loopNum) {
  uint32_t localWorkSpaceSize = loopTailNum * softMaskMinTmpSize;
  if (stackNum != 0) {
    localWorkSpaceSize = stackNum * softMaskMinTmpSize;
    auto srcShape = ge::Shape({stackNum, s2AlignedSize});
    if (dataType != ge::DT_FLOAT) {
      AscendC::SoftMaxFlashV2TilingFunc(srcShape, dtypeSize * 2, dtypeSize * 2, localWorkSpaceSize,
                                        tilingData.softmaxTilingData,
                                        false, false);
    } else {
      AscendC::SoftMaxFlashV2TilingFunc(srcShape, dtypeSize, dtypeSize, localWorkSpaceSize,
                                        tilingData.softmaxTilingData, false, false);
    }
  }

  if (loopTailNum != 0) {
    auto srcShape = ge::Shape({loopTailNum, s2AlignedSize});
    if (dataType != ge::DT_FLOAT) {
      AscendC::SoftMaxFlashV2TilingFunc(srcShape, dtypeSize * 2, dtypeSize * 2, localWorkSpaceSize,
                                        tilingData.tailSoftmaxTilingData, false, false);
    } else {
      AscendC::SoftMaxFlashV2TilingFunc(srcShape, dtypeSize, dtypeSize, localWorkSpaceSize,
                                        tilingData.tailSoftmaxTilingData, false, false);
    }
  }
  tilingData.set_tmpXubSize(localWorkSpaceSize);
  tilingData.set_tmpMaskUbSize(0);
  OP_LOGD(context_, "MaskedSoftmaxWithRelPosBiasS1BiasTiling stackNum=%u, loopNum=%u, loopTailNum=%u.",
              stackNum, loopNum, loopTailNum);
  if (stackNum == 0 && loopTailNum == 0) {
    OP_LOGD(context_, "stackNum and LoopTailNum both all are zero.");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasS1BiasTiling::DoOpTiling() {
  uint32_t totalSize = b_ * w_ * n_ * s1_;
  uint32_t singleCoreSize = totalSize / aivNum;
  int64_t tailSize = b_ * w_ * n_ * s1_ - singleCoreSize * aivNum;
  int64_t tailStartCoreIdx = aivNum;
  if (tailSize != 0) {
    tailStartCoreIdx = tailSize;
    singleCoreSize += 1;
  }
  tilingData.set_tailStartCoreIdx(tailStartCoreIdx);
  tilingData.set_singleCoreSize(singleCoreSize);

  auto xYUbSize = 2 * xSize;
  uint32_t stackNum = aicoreParams_.ubSize / ((xYUbSize + static_cast<uint32_t>(biasSize)) * 2 + softMaskMinTmpSize + static_cast<uint32_t>(bf16XCastTempSize +
                                              bf16ABiasCastTempSize + bf16YCastTempSize));
  uint32_t loopNum = 0;
  if (tailSize != 0) {
    stackNum = std::min(stackNum, (singleCoreSize - 1) / BUFF_NUM);
    if (stackNum != 0) {
      loopNum = (singleCoreSize - 1) / stackNum;
    }
  } else {
    stackNum = std::min(stackNum, singleCoreSize / BUFF_NUM);
    if (stackNum != 0) {
      loopNum = singleCoreSize / stackNum;
    }
  }
  auto loopTailNum = singleCoreSize - stackNum * loopNum;
  if (stackNum != 0) {
    auto inQueueLen = stackNum * xSize;
    tilingData.set_inQueueLen(inQueueLen);
    tilingData.set_maskQueueLen(stackNum * static_cast<uint32_t>(biasSize));
    tilingData.set_castTempSize(stackNum * (static_cast<uint32_t>(bf16XCastTempSize + bf16ABiasCastTempSize + bf16YCastTempSize)));
  } else {
    auto inQueueLen = xSize;
    tilingData.set_inQueueLen(inQueueLen);
    tilingData.set_maskQueueLen(static_cast<uint32_t>(biasSize));
    tilingData.set_castTempSize(static_cast<uint32_t>(bf16XCastTempSize + bf16ABiasCastTempSize + bf16YCastTempSize));
  }

  tilingData.set_stackNum(stackNum);
  tilingData.set_loopNum(loopNum);
  tilingData.set_loopTailNum(loopTailNum);

  auto xCopyEleNum = stackNum * s2AlignedSize;
  tilingData.set_xCopyEleNum(xCopyEleNum);
  auto tailXCopyEleNum = loopTailNum * s2AlignedSize;
  tilingData.set_tailXCopyEleNum(tailXCopyEleNum);

  return DoSoftMaxFlashV2Tiling(loopTailNum, stackNum, loopNum);
}

uint64_t MaskedSoftmaxWithRelPosBiasS1BiasTiling::GetTilingKey() const {
  if (isAligned) {
    if (dataType == ge::DT_FLOAT) {
      if (existMuls) {
        return BWNS1_FLOAT_MULS;
      }
      return BWNS1_FLOAT_NULL;
    }

    if (dataType == ge::DT_FLOAT16) {
      if (existMuls) {
        return BWNS1_FLOAT16_MULS;
      }
      return BWNS1_FLOAT16_NULL;
    }

    if (dataType == ge::DT_BF16) {
      if (existMuls) {
        return BWNS1_BF16_MULS;
      }
      return BWNS1_BF16_NULL;
    }
    return MASKED_SOFTMAX_INVALED_KEY;
  } else {
    if (dataType == ge::DT_FLOAT) {
      if (existMuls) {
        return BWNS1_FLOAT_MULS_NOT_ALIGEND;
      }
      return BWNS1_FLOAT_NULL_NOT_ALIGEND;
    }

    if (dataType == ge::DT_FLOAT16) {
      if (existMuls) {
        return BWNS1_FLOAT16_MULS_NOT_ALIGEND;
      }
      return BWNS1_FLOAT16_NULL_NOT_ALIGEND;
    }

    if (dataType == ge::DT_BF16) {
      if (existMuls) {
        return BWNS1_BF16_MULS_NOT_ALIGEND;
      }
      return BWNS1_BF16_NULL_NOT_ALIGEND;
    }
    return MASKED_SOFTMAX_INVALED_KEY;
  }
}

ge::graphStatus MaskedSoftmaxWithRelPosBiasS1BiasTiling::PostTiling() {
  tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
  context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
  context_->SetBlockDim(std::min(aivNum, b_ * w_ * n_ * s1_));
  return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("MaskedSoftmaxWithRelPosBias", MaskedSoftmaxWithRelPosBiasS1BiasTiling, 103);
static OpTiling::Register VAR_UNUSED Norm_MaskedSoftmaxWithRelPosBias_S1Bias_103_register =
    OpTiling::Register("Norm/MaskedSoftmaxWithRelPosBias").tiling<MaskedSoftmaxWithRelPosBiasS1BiasTiling>(103);

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
    .TilingParse<MaskedSoftmaxWithRelPosBiasCompileInfo>(
        TilingPrepareForMaskedSoftmaxWithRelPosBias);
}  // namespace optiling