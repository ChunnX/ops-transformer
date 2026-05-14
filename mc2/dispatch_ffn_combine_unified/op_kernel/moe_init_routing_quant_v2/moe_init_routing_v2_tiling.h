#pragma once
#ifndef ASCENDC_DISPATCH_FFN_COMBINE_INIT_ROUTING_TILING_H
#define ASCENDC_DISPATCH_FFN_COMBINE_INIT_ROUTING_TILING_H

#include "tiling_base.h"
#include <cmath>

namespace optiling {
const static int64_t TILING_KEY_DROPLESS_SORT_ONE_CORE = 10001;
const static int64_t TILING_KEY_DROPLESS_SORT_MULTI_CORE = 10002;
const static int64_t TILING_KEY_DROP_PAD_MODE_SORT_ONE_CORE = 10011;
const static int64_t TILING_KEY_DROP_PAD_MODE_SORT_MULTI_CORE = 10012;
const static int64_t TILING_KEY_HIGH_PERFORMANCE = 20000;
const static int64_t NUM_TWO = 2;
const static int64_t NUM_THREE = 3;
const static int64_t NUM_FOUR = 4;
const static int64_t MRG_LIST_NUM = 4;
const static int64_t SORT32_ALIGN_ELEMENT = 32;
const static int64_t ONE_BLOCK_BYTE = 32;
const static size_t DIM_ONE = 1;
const static size_t DIM_TWO = 2;
const static size_t DIM_THREE = 3;
const static int32_t SIZE_16 = 16;
const static int32_t LENGTH_1024 = 1024;
const static int64_t MAX_COLS_ONE_LOOP = 16376;
const static int64_t ASSIST_NUM = 256;
const static int64_t INDEX_INPUT_X = 0;
const static int64_t INDEX_INPUT_EXPERT_IDX = 1;
const static int64_t ATTR_ACTIVE_ROWS = 0;
const static int64_t ATTR_EXPERT_CAPACITY = 1;
const static int64_t ATTR_EXPERT_NUM = 2;
const static int64_t ATTR_DROP_PAD_MODE = 3;
const static int64_t ATTR_EXPERT_TOKENS_COUNT_OR_CUMSUM_FLAG = 4;
const static int64_t ATTR_EXPERT_TOKENS_BEFORE_CAPACITY_FLAG = 5;
const static int64_t OUTOUT_EXPANDED_X = 0;
const static int64_t OUTOUT_EXPANDED_ROW_IDX = 1;
const static int64_t OUTOUT_EXPERT_TOKENS_COUNT_OR_CUMSUM = 2;
const static int64_t OUTOUT_EXPERT_TOKENS_BEFORE_CAPACITY = 3;
const static int64_t KV_FACTOR = 2;
const static int64_t ONE_CORE_SORT_BUFFER = 6;
const static int64_t EXPERT_TOKENS_COUNT = 2;


inline static int64_t CeilLog4(int64_t x) {
  return static_cast<int64_t>(std::ceil(std::log(x) / std::log(NUM_FOUR)));
}

inline static int64_t GetPerOrLastValue(int64_t x, int64_t y) {
  if (y == 0) {
    return 0;
  }
  return x <= y ? x : x % y;
}

template <class T>
constexpr T CeilDiv(const T dividend, const T divisor)
{
    return (dividend + divisor - 1) / divisor;
}


struct InnerMoeV2VBSComputeTilingData {
    int64_t needCoreNum = 0;
    int64_t perCoreElements = 0;
    int64_t perCoreLoops = 0;
    int64_t perCorePerLoopElements = 0;
    int64_t perCoreLastLoopElements = 0;
    int64_t lastCoreElements = 0;
    int64_t lastCoreLoops = 0;
    int64_t lastCorePerLoopElements = 0;
    int64_t lastCoreLastLoopElements = 0;
    int64_t oneLoopMaxElements = 0;
};

struct InnerMoeV2VMSMiddleComputeTilingData {
    int64_t needCoreNum = 0;
};

struct InnerMoeV2SortOutComputeTilingData {
    int64_t oneLoopMaxElements = 0;
};

struct InnerMoeV2GatherOutComputeTilingData {
    int64_t needCoreNum = 0;
    int64_t activateRows = 0;
    int64_t perCoreRows = 0;
    int64_t perCorePerLoopRows = 0;
    int64_t perCoreLastLoopRows = 0;
    int64_t lastCoreRows = 0;
    int64_t lastCorePerLoopRows = 0;
    int64_t lastCoreLastLoopRows = 0;
    int64_t perCoreLoops = 0;
    int64_t lastCoreLoops = 0;
    int64_t perLoopCols = 0;
    int64_t lastLoopCols = 0;
    int64_t colLoops = 0;
};

struct InnerMoeInitRoutingV2TilingData {
    int64_t coreNum;
    int64_t n;
    int64_t cols;
    int64_t k;
    int64_t expertCapacity;
    int64_t expertNum;
    int64_t dropPadMode;
    int64_t expertTokensCountOrCumsumFlag;
    int64_t expertTokensBeforeCapacityFlag;
    InnerMoeV2VBSComputeTilingData vbsComputeParamsOp;
    InnerMoeV2VMSMiddleComputeTilingData vmsMiddleComputeParamsOp;
    InnerMoeV2SortOutComputeTilingData sortOutComputeParamsOp;
    InnerMoeV2GatherOutComputeTilingData srcToDstComputeParamsOp;
    InnerMoeV2GatherOutComputeTilingData srcToDstCapacityComputeParamsOp;
    InnerMoeV2GatherOutComputeTilingData gatherOutComputeParamsOp;
};


class InnerMoeInitRoutingV2TilingBase : public TilingBaseClass {

protected:
  inline bool GetPlatformInfo(int64_t aivCoreNum, int64_t ubSizePlatForm) override;
  inline bool GetShapeAttrsInfo(int64_t m, int64_t cols, int64_t topK, int64_t expertCapacity, 
  int64_t expertNum, int64_t activeNum, int64_t dropPadMode, int64_t expertTokensCountOrCumsumFlag,
  bool expertTokensBeforeCapacityFlag, int64_t inuptXDtypeSize, int64_t quantMode, int64_t scaleDim0) override;

  inline bool DoOpTiling() override;
  inline uint64_t GetTilingKey() const override;
  inline bool GetWorkspaceSize() override;


protected:
  bool CheckTokenCount(int64_t num, const char* tag);

  virtual void Tiling4GatherOutCompute() = 0;
  inline void Tiling4SrcToDstCompute();
  virtual void Tiling4SrcToDstCapacityCompute();
  inline void Tiling4SortOutCompute();
  inline void Tiling4VMSMiddleCompute();
  inline void Tiling4VBSCompute();
  void ShowTilingData();
  inline void Tiling4VBSMultiCoreCompute(InnerMoeV2VBSComputeTilingData* tilingData);
  inline void Tiling4VBSOneCoreCompute(InnerMoeV2VBSComputeTilingData* tilingData);
  virtual bool IsFullLoad() = 0;

  int64_t aivNum = 0;
  int64_t sortLoopMaxElement = 0;
  int64_t mrgSortListMaxElement = 2040;
  int64_t totalLength = 0;
  int64_t activateNum = 0;
  int64_t expertCapacity = 0;
  int64_t expertNum = 0;
  int64_t dropPadMode = 0;
  int64_t expertTokensCountOrCumsumFlag = 0;
  bool expertTokensBeforeCapacityFlag = false;
  int64_t inuptXDtypeSize_ = 0;
  bool isFullLoad = false;

  InnerMoeInitRoutingV2TilingData moeInitRoutingTilingData;
};


inline bool InnerMoeInitRoutingV2TilingBase::DoOpTiling() {
  sortLoopMaxElement =
      (aicoreParams_.ubSize) / (sizeof(int32_t) * NUM_TWO * NUM_FOUR) / SORT32_ALIGN_ELEMENT * SORT32_ALIGN_ELEMENT;
  isFullLoad = IsFullLoad();
  Tiling4VBSCompute();
  Tiling4VMSMiddleCompute();
  Tiling4SortOutCompute();
  Tiling4SrcToDstCompute();
  Tiling4SrcToDstCapacityCompute();
  Tiling4GatherOutCompute();
  return true;
};

inline uint64_t InnerMoeInitRoutingV2TilingBase::GetTilingKey() const {
  if (isFullLoad) {
    return TILING_KEY_HIGH_PERFORMANCE;
  }
  if (dropPadMode == 0) {
    if (totalLength <= sortLoopMaxElement) {  // Sorting uses only one core
      return TILING_KEY_DROPLESS_SORT_ONE_CORE;
    } else {
      return TILING_KEY_DROPLESS_SORT_MULTI_CORE;
    }
  } else {
    if (totalLength <= sortLoopMaxElement) {
      return TILING_KEY_DROP_PAD_MODE_SORT_ONE_CORE;
    } else {
      return TILING_KEY_DROP_PAD_MODE_SORT_MULTI_CORE;
    }
  }
  return tilingKey_;
}

inline bool InnerMoeInitRoutingV2TilingBase::GetShapeAttrsInfo(int64_t m, int64_t cols, int64_t topK, int64_t expertCapacity, 
  int64_t expertNum, int64_t activateNum, int64_t dropPadMode, int64_t expertTokensCountOrCumsumFlag,
  bool expertTokensBeforeCapacityFlag, int64_t inuptXDtypeSize, int64_t quantMode, int64_t scaleDim0) {
  
  this->activateNum = activateNum;
  this->expertCapacity = expertCapacity;
  this->expertNum = expertNum;
  this->dropPadMode = dropPadMode;
  this->expertTokensCountOrCumsumFlag = expertTokensCountOrCumsumFlag;
  this->expertTokensBeforeCapacityFlag = expertTokensBeforeCapacityFlag;
  if (dropPadMode == 1) {
    // Do not output expertTokensCountOrCumsum in drop-pad mode
    expertTokensCountOrCumsumFlag = 0;
  } else {
    // Do not output expertTokensBeforeCapacity in dropless mode
    expertTokensBeforeCapacityFlag = false;
  }
  moeInitRoutingTilingData.cols = cols;
  moeInitRoutingTilingData.n = m;
  moeInitRoutingTilingData.k = topK;
  moeInitRoutingTilingData.expertCapacity = expertCapacity;
  moeInitRoutingTilingData.expertNum = expertNum;
  moeInitRoutingTilingData.dropPadMode = dropPadMode;
  moeInitRoutingTilingData.expertTokensCountOrCumsumFlag = expertTokensCountOrCumsumFlag;
  moeInitRoutingTilingData.expertTokensBeforeCapacityFlag = expertTokensBeforeCapacityFlag;
  totalLength = moeInitRoutingTilingData.n * moeInitRoutingTilingData.k;
  inuptXDtypeSize_ = inuptXDtypeSize;
  return true;
}

inline bool InnerMoeInitRoutingV2TilingBase::GetPlatformInfo(int64_t aivCoreNum, int64_t ubSizePlatForm) {
  aivNum = aivCoreNum;
  aicoreParams_.blockDim = aivCoreNum;
  aicoreParams_.ubSize = ubSizePlatForm;
  moeInitRoutingTilingData.coreNum = aivCoreNum;
  return true;
}


inline bool InnerMoeInitRoutingV2TilingBase::GetWorkspaceSize() {
  // Calculate workspace size
  size_t sortWorkspaceSize = totalLength * sizeof(float) * NUM_TWO * NUM_THREE;  // Space needed for sorting
  size_t scatterWorkspaceSize = totalLength * sizeof(int32_t) * NUM_TWO;
  size_t expertTokenFlagSize = aivNum * 2 * sizeof(int32_t);
  workspaceSize_ = sortWorkspaceSize + scatterWorkspaceSize + expertTokenFlagSize + SIZE_16 * LENGTH_1024 * LENGTH_1024;
  return true;
}

inline void InnerMoeInitRoutingV2TilingBase::Tiling4VBSOneCoreCompute(InnerMoeV2VBSComputeTilingData* tilingData) {
    tilingData->needCoreNum = 1;
    tilingData->perCoreElements = totalLength;
    tilingData->perCoreLoops = 1;
    tilingData->perCorePerLoopElements = tilingData->perCoreElements;
    tilingData->perCoreLastLoopElements = tilingData->perCoreElements;
    tilingData->lastCoreElements = tilingData->perCoreElements;
    tilingData->lastCoreLoops = 1;
    tilingData->lastCorePerLoopElements = tilingData->perCoreElements;
    tilingData->lastCoreLastLoopElements = tilingData->perCoreElements;
}

inline void InnerMoeInitRoutingV2TilingBase::Tiling4VBSMultiCoreCompute(InnerMoeV2VBSComputeTilingData* tilingData) {
      //Tiling4VBSMultiCoreCompute
      int64_t needCoreNum = CeilDiv(totalLength, sortLoopMaxElement);  // Round up
      needCoreNum = static_cast<int64_t>(std::pow(4, CeilLog4(needCoreNum)));
      needCoreNum = std::min(needCoreNum, aivNum);                     // Cannot exceed physical core count
      if (needCoreNum > 0) {
          int64_t perCoreElements = totalLength / needCoreNum;  // Elements handled per core
          int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
          int64_t lastCoreElement = totalLength - (needCoreNum - 1) * alineFloorPerCoreElements;
          int64_t alineCeilPerCoreElements = perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
          if (lastCoreElement > alineCeilPerCoreElements) {
            perCoreElements = alineCeilPerCoreElements;
            needCoreNum = CeilDiv(totalLength, perCoreElements);
          } else {
            perCoreElements = alineFloorPerCoreElements;
          }
          tilingData->needCoreNum = needCoreNum;
          do {
              tilingData->perCoreElements = perCoreElements;
              tilingData->perCoreLoops = CeilDiv(tilingData->perCoreElements, sortLoopMaxElement);  // Loops handled per core
              tilingData->perCorePerLoopElements = std::min(tilingData->perCoreElements, sortLoopMaxElement);
              tilingData->perCoreLastLoopElements = tilingData->perCoreElements - (tilingData->perCoreLoops - 1) * tilingData->perCorePerLoopElements;
              tilingData->lastCoreElements = totalLength - (tilingData->needCoreNum - 1) * tilingData->perCoreElements;
              tilingData->lastCoreLoops = tilingData->perCoreLoops;
              int64_t tmp = CeilDiv(tilingData->lastCoreElements, tilingData->lastCoreLoops);
              int64_t lastCorePerLoopElements =
                  CeilDiv(CeilDiv(tilingData->lastCoreElements, tilingData->lastCoreLoops), SORT32_ALIGN_ELEMENT) *
                  SORT32_ALIGN_ELEMENT;
              tilingData->lastCorePerLoopElements = lastCorePerLoopElements;
              tilingData->lastCoreLastLoopElements = tilingData-> lastCoreElements - (tilingData->lastCoreLoops - 1) * tilingData->lastCorePerLoopElements;
              perCoreElements -= SORT32_ALIGN_ELEMENT;
          } while (tilingData->lastCoreLastLoopElements <= 0 && perCoreElements > 0);
      }
}


inline void InnerMoeInitRoutingV2TilingBase::Tiling4VBSCompute() {
  auto tilingData = &moeInitRoutingTilingData.vbsComputeParamsOp;
  tilingData->oneLoopMaxElements = sortLoopMaxElement;
  if (totalLength <= sortLoopMaxElement) {  // Only one core is used
    Tiling4VBSOneCoreCompute(tilingData);
    return;
  }
  Tiling4VBSMultiCoreCompute(tilingData);
}

inline void InnerMoeInitRoutingV2TilingBase::Tiling4VMSMiddleCompute() {
  auto vbsComputeTilingData = &moeInitRoutingTilingData.vbsComputeParamsOp;
  auto tilingData = &moeInitRoutingTilingData.vmsMiddleComputeParamsOp;
  if (vbsComputeTilingData->needCoreNum <= MRG_LIST_NUM) {  // No intermediate merge if queue count fits one VMS
      tilingData->needCoreNum = 0;                               // Required core count
  } else {
      int64_t needCoreNum = CeilDiv(vbsComputeTilingData->needCoreNum, MRG_LIST_NUM);
      tilingData->needCoreNum = needCoreNum;  // Required core count
  }
}

inline void InnerMoeInitRoutingV2TilingBase::Tiling4SortOutCompute() {
  auto tilingData = &moeInitRoutingTilingData.sortOutComputeParamsOp;
  tilingData->oneLoopMaxElements = mrgSortListMaxElement;
}


inline void InnerMoeInitRoutingV2TilingBase::Tiling4SrcToDstCompute() {
  auto tilingData = &moeInitRoutingTilingData.srcToDstComputeParamsOp;

  int64_t perLoopMaxRows = (aicoreParams_.ubSize - ASSIST_NUM * sizeof(float) - aivNum * SORT32_ALIGN_ELEMENT) /
                           (SORT32_ALIGN_ELEMENT * NUM_TWO) / NUM_TWO;
  int64_t perCoreRows = CeilDiv(totalLength, aivNum);
  if (perCoreRows <= 0) {
    tilingData->needCoreNum = 0;
    return;
  }

  int64_t needCoreNum = CeilDiv(totalLength, perCoreRows);
  tilingData->needCoreNum = needCoreNum;
  int64_t lastCoreNum = totalLength - perCoreRows * (tilingData->needCoreNum - 1);
  tilingData->perCoreRows = perCoreRows;
  if (perLoopMaxRows >= tilingData->perCoreRows) {  // One loop completes
      tilingData->perCorePerLoopRows = tilingData->perCoreRows;
      tilingData->perCoreLastLoopRows = tilingData->perCoreRows;
  } else {
      tilingData->perCorePerLoopRows = perLoopMaxRows;
      tilingData->perCoreLastLoopRows = tilingData->perCoreRows - (CeilDiv(tilingData->perCoreRows, perLoopMaxRows) - 1) * perLoopMaxRows;
  }
  tilingData->lastCoreRows = lastCoreNum;
  if (perLoopMaxRows >= tilingData->lastCoreRows) {
      tilingData->lastCorePerLoopRows = tilingData->lastCoreRows;
      tilingData->lastCoreLastLoopRows = tilingData->lastCoreRows;
  } else {
      tilingData->lastCorePerLoopRows = perLoopMaxRows;
      tilingData->lastCoreLastLoopRows = tilingData->lastCoreRows - (CeilDiv(tilingData->lastCoreRows, perLoopMaxRows) - 1) * perLoopMaxRows;
  }
}


inline void InnerMoeInitRoutingV2TilingBase::Tiling4SrcToDstCapacityCompute() {
  auto tilingData = &moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp;
  int64_t perCoreRows = CeilDiv(totalLength, aivNum);

  if (perCoreRows <= 0) {
    tilingData->needCoreNum = 0;
    return;
  }

  int64_t needCoreNum = CeilDiv(totalLength, perCoreRows);
  tilingData->needCoreNum = needCoreNum;
  int64_t cols = moeInitRoutingTilingData.cols;
  tilingData->perCoreRows = perCoreRows;
  int64_t lastCoreRows = totalLength - perCoreRows * (needCoreNum - 1);
  tilingData->lastCoreRows = lastCoreRows;


  int64_t rowSize =
      (perCoreRows * sizeof(int32_t) * 2 + ONE_BLOCK_BYTE + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
  int64_t colSize = (cols * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;

  if (rowSize + colSize < static_cast<int64_t>(aicoreParams_.ubSize)) {
    tilingData->perCorePerLoopRows = perCoreRows;
    tilingData->perCoreLastLoopRows = perCoreRows;
    tilingData->lastCorePerLoopRows = lastCoreRows;
    tilingData->lastCoreLastLoopRows = lastCoreRows;
    tilingData->perCoreLoops = 1;
    tilingData->lastCoreLoops = 1;
    tilingData->perLoopCols = cols;
    tilingData->lastLoopCols = cols;
    tilingData->colLoops = 1;

  } else {
    int64_t baseMaxCols = MAX_COLS_ONE_LOOP;
    int64_t baseMaxColsSize = (baseMaxCols * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    int64_t basePerLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) - baseMaxColsSize - ONE_BLOCK_BYTE) /
                                 static_cast<int64_t>(sizeof(int32_t)) / NUM_TWO / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    if (cols < MAX_COLS_ONE_LOOP) {
      basePerLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) - colSize - ONE_BLOCK_BYTE) /
                           static_cast<int64_t>(sizeof(int32_t)) / NUM_TWO / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    } else if (perCoreRows < basePerLoopMaxRows) {
      baseMaxCols =
          (static_cast<int64_t>(aicoreParams_.ubSize) - rowSize) / inuptXDtypeSize_ / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    }
    tilingData->perLoopCols = (std::min(baseMaxCols, cols));
    tilingData->lastLoopCols = (GetPerOrLastValue(cols, baseMaxCols));
    tilingData->colLoops = ((cols + baseMaxCols - 1) / baseMaxCols);
    tilingData->perCorePerLoopRows = (std::min(perCoreRows, basePerLoopMaxRows));
    tilingData->perCoreLastLoopRows = (GetPerOrLastValue(perCoreRows, basePerLoopMaxRows));
    tilingData->perCoreLoops = ((perCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);
    tilingData->lastCorePerLoopRows = (std::min(lastCoreRows, basePerLoopMaxRows));
    tilingData->lastCoreLastLoopRows = (GetPerOrLastValue(lastCoreRows, basePerLoopMaxRows));
    tilingData->lastCoreLoops = ((lastCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);
  }
}


// =====================================================================
// MoeInitRoutingV2TilingBase
//   Standalone non-quant tiling class. Mirrors the structure of the
//   equivalent class in `mc2/dispatch_ffn_combine_bf16/op_kernel/moe_init_routing_v2/`:
//   it inherits directly from `TilingBaseClass` and re-implements the full
//   tiling pipeline inline, INDEPENDENTLY from `InnerMoeInitRoutingV2TilingBase`
//   (which remains the parent of the quant variant only).
//
//   Used by the unified DispatchFFNCombine tiling when `isQuantRouting == 0`
//   (BF16 / FP16 weight path: X is forwarded as-is, no per-token int8 quant).
//
//   To avoid disturbing the `cocTiling` / kernel data flow, the class still
//   exposes its result via `InnerMoeInitRoutingV2TilingData` (layout-identical
//   to the BF16 `MoeInitRoutingV2TilingData`).
//
//   The tilingKey scheme matches `MoeInitRoutingQuantV2TilingBase::GetTilingKey()`
//   with `quantMode == 0` so the kernel dispatcher in `moe_init_routing_v2.cpp`
//   keeps recognising 20000 / 10000 / 10010 / 10100 / 10110.
// =====================================================================
class MoeInitRoutingV2TilingBase : public TilingBaseClass {
protected:
    inline bool GetPlatformInfo(int64_t aivCoreNum, int64_t ubSizePlatForm) override;
    inline bool GetShapeAttrsInfo(int64_t m, int64_t cols, int64_t topK, int64_t expertCapacity,
        int64_t expertNum, int64_t activeNum, int64_t dropPadMode, int64_t expertTokensCountOrCumsumFlag,
        bool expertTokensBeforeCapacityFlag, int64_t inuptXDtypeSize, int64_t quantMode, int64_t scaleDim0) override;

    inline bool DoOpTiling() override;
    inline uint64_t GetTilingKey() const override;
    inline bool GetWorkspaceSize() override;
    inline bool PostTiling() override { return true; }

protected:
    inline void Tiling4GatherOutCompute();
    inline void Tiling4SrcToDstCompute();
    inline void Tiling4SrcToDstCapacityCompute();
    inline void Tiling4SortOutCompute();
    inline void Tiling4VMSMiddleCompute();
    inline void Tiling4VBSCompute();
    inline void Tiling4VBSMultiCoreCompute(InnerMoeV2VBSComputeTilingData* tilingData);
    inline void Tiling4VBSOneCoreCompute(InnerMoeV2VBSComputeTilingData* tilingData);
    inline bool IsFullLoad();

    int64_t aivNum = 0;
    int64_t sortLoopMaxElement = 0;
    int64_t mrgSortListMaxElement = 2040;
    int64_t totalLength = 0;
    int64_t activateNum = 0;
    int64_t expertCapacity = 0;
    int64_t expertNum = 0;
    int64_t dropPadMode = 0;
    int64_t expertTokensCountOrCumsumFlag = 0;
    bool expertTokensBeforeCapacityFlag = false;
    int64_t inuptXDtypeSize_ = 0;
    bool isFullLoad = false;

public:
    InnerMoeInitRoutingV2TilingData moeInitRoutingTilingData;
};


inline bool MoeInitRoutingV2TilingBase::GetPlatformInfo(int64_t aivCoreNum, int64_t ubSizePlatForm) {
    aivNum = aivCoreNum;
    aicoreParams_.blockDim = aivCoreNum;
    aicoreParams_.ubSize = ubSizePlatForm;
    moeInitRoutingTilingData.coreNum = aivCoreNum;
    return true;
}


inline bool MoeInitRoutingV2TilingBase::GetShapeAttrsInfo(int64_t m, int64_t cols, int64_t topK,
    int64_t expertCapacity, int64_t expertNum, int64_t activateNum, int64_t dropPadMode,
    int64_t expertTokensCountOrCumsumFlag, bool expertTokensBeforeCapacityFlag,
    int64_t inuptXDtypeSize, int64_t /*quantMode*/, int64_t /*scaleDim0*/) {
    this->activateNum = activateNum;
    this->expertCapacity = expertCapacity;
    this->expertNum = expertNum;
    this->dropPadMode = dropPadMode;
    this->expertTokensCountOrCumsumFlag = expertTokensCountOrCumsumFlag;
    this->expertTokensBeforeCapacityFlag = expertTokensBeforeCapacityFlag;
    if (dropPadMode == 1) {
        // Do not output expertTokensCountOrCumsum in drop-pad mode.
        expertTokensCountOrCumsumFlag = 0;
    } else {
        // Do not output expertTokensBeforeCapacity in dropless mode.
        expertTokensBeforeCapacityFlag = false;
    }
    moeInitRoutingTilingData.cols = cols;
    moeInitRoutingTilingData.n = m;
    moeInitRoutingTilingData.k = topK;
    moeInitRoutingTilingData.expertCapacity = expertCapacity;
    moeInitRoutingTilingData.expertNum = expertNum;
    moeInitRoutingTilingData.dropPadMode = dropPadMode;
    moeInitRoutingTilingData.expertTokensCountOrCumsumFlag = expertTokensCountOrCumsumFlag;
    moeInitRoutingTilingData.expertTokensBeforeCapacityFlag = expertTokensBeforeCapacityFlag;
    totalLength = moeInitRoutingTilingData.n * moeInitRoutingTilingData.k;
    inuptXDtypeSize_ = inuptXDtypeSize;
    return true;
}


inline bool MoeInitRoutingV2TilingBase::DoOpTiling() {
    sortLoopMaxElement =
        (aicoreParams_.ubSize) / (sizeof(int32_t) * NUM_TWO * NUM_FOUR) / SORT32_ALIGN_ELEMENT * SORT32_ALIGN_ELEMENT;
    isFullLoad = IsFullLoad();
    Tiling4VBSCompute();
    Tiling4VMSMiddleCompute();
    Tiling4SortOutCompute();
    Tiling4SrcToDstCompute();
    Tiling4SrcToDstCapacityCompute();
    Tiling4GatherOutCompute();
    return true;
}


inline uint64_t MoeInitRoutingV2TilingBase::GetTilingKey() const {
    // Matches `MoeInitRoutingQuantV2TilingBase::GetTilingKey()` with `quantMode == 0`
    // so the kernel-side dispatcher in `moe_init_routing_v2.cpp` keeps matching:
    //   20000 -> fullload
    //   10000 -> dropless,  one-core sort     (drop=0, sort=0)
    //   10010 -> dropless,  multi-core sort   (drop=0, sort=1)
    //   10100 -> drop-pad,  one-core sort     (drop=1, sort=0)
    //   10110 -> drop-pad,  multi-core sort   (drop=1, sort=1)
    constexpr uint64_t PERF_BASE = 20000;
    constexpr uint64_t BASE = 10000;
    constexpr uint64_t DROP_MODE_BASE = 100;
    constexpr uint64_t SORT_BASE = 10;
    if (isFullLoad) {
        return PERF_BASE;
    }
    return BASE + static_cast<uint64_t>(dropPadMode) * DROP_MODE_BASE +
           (totalLength > sortLoopMaxElement ? 1ULL : 0ULL) * SORT_BASE;
}


inline bool MoeInitRoutingV2TilingBase::GetWorkspaceSize() {
    size_t sortWorkspaceSize = totalLength * sizeof(float) * NUM_TWO * NUM_THREE;
    size_t scatterWorkspaceSize = totalLength * sizeof(int32_t) * NUM_TWO;
    size_t expertTokenFlagSize = aivNum * 2 * sizeof(int32_t);
    workspaceSize_ = sortWorkspaceSize + scatterWorkspaceSize + expertTokenFlagSize +
                     SIZE_16 * LENGTH_1024 * LENGTH_1024;
    return true;
}


inline void MoeInitRoutingV2TilingBase::Tiling4VBSOneCoreCompute(InnerMoeV2VBSComputeTilingData* tilingData) {
    tilingData->needCoreNum = 1;
    tilingData->perCoreElements = totalLength;
    tilingData->perCoreLoops = 1;
    tilingData->perCorePerLoopElements = tilingData->perCoreElements;
    tilingData->perCoreLastLoopElements = tilingData->perCoreElements;
    tilingData->lastCoreElements = tilingData->perCoreElements;
    tilingData->lastCoreLoops = 1;
    tilingData->lastCorePerLoopElements = tilingData->perCoreElements;
    tilingData->lastCoreLastLoopElements = tilingData->perCoreElements;
}


inline void MoeInitRoutingV2TilingBase::Tiling4VBSMultiCoreCompute(InnerMoeV2VBSComputeTilingData* tilingData) {
    int64_t needCoreNum = CeilDiv(totalLength, sortLoopMaxElement);
    needCoreNum = static_cast<int64_t>(std::pow(4, CeilLog4(needCoreNum)));
    needCoreNum = std::min(needCoreNum, aivNum);
    if (needCoreNum > 0) {
        int64_t perCoreElements = totalLength / needCoreNum;
        int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
        int64_t lastCoreElement = totalLength - (needCoreNum - 1) * alineFloorPerCoreElements;
        int64_t alineCeilPerCoreElements =
            perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
        if (lastCoreElement > alineCeilPerCoreElements) {
            perCoreElements = alineCeilPerCoreElements;
            needCoreNum = CeilDiv(totalLength, perCoreElements);
        } else {
            perCoreElements = alineFloorPerCoreElements;
        }
        tilingData->needCoreNum = needCoreNum;
        do {
            tilingData->perCoreElements = perCoreElements;
            tilingData->perCoreLoops = CeilDiv(tilingData->perCoreElements, sortLoopMaxElement);
            tilingData->perCorePerLoopElements = std::min(tilingData->perCoreElements, sortLoopMaxElement);
            tilingData->perCoreLastLoopElements =
                tilingData->perCoreElements - (tilingData->perCoreLoops - 1) * tilingData->perCorePerLoopElements;
            tilingData->lastCoreElements = totalLength - (tilingData->needCoreNum - 1) * tilingData->perCoreElements;
            tilingData->lastCoreLoops = tilingData->perCoreLoops;
            int64_t lastCorePerLoopElements =
                CeilDiv(CeilDiv(tilingData->lastCoreElements, tilingData->lastCoreLoops), SORT32_ALIGN_ELEMENT) *
                SORT32_ALIGN_ELEMENT;
            tilingData->lastCorePerLoopElements = lastCorePerLoopElements;
            tilingData->lastCoreLastLoopElements =
                tilingData->lastCoreElements - (tilingData->lastCoreLoops - 1) * tilingData->lastCorePerLoopElements;
            perCoreElements -= SORT32_ALIGN_ELEMENT;
        } while (tilingData->lastCoreLastLoopElements <= 0 && perCoreElements > 0);
    }
}


inline void MoeInitRoutingV2TilingBase::Tiling4VBSCompute() {
    auto tilingData = &moeInitRoutingTilingData.vbsComputeParamsOp;
    tilingData->oneLoopMaxElements = sortLoopMaxElement;
    if (totalLength <= sortLoopMaxElement) {
        Tiling4VBSOneCoreCompute(tilingData);
        return;
    }
    Tiling4VBSMultiCoreCompute(tilingData);
}


inline void MoeInitRoutingV2TilingBase::Tiling4VMSMiddleCompute() {
    auto vbsComputeTilingData = &moeInitRoutingTilingData.vbsComputeParamsOp;
    auto tilingData = &moeInitRoutingTilingData.vmsMiddleComputeParamsOp;
    if (vbsComputeTilingData->needCoreNum <= MRG_LIST_NUM) {
        tilingData->needCoreNum = 0;
    } else {
        tilingData->needCoreNum = CeilDiv(vbsComputeTilingData->needCoreNum, MRG_LIST_NUM);
    }
}


inline void MoeInitRoutingV2TilingBase::Tiling4SortOutCompute() {
    auto tilingData = &moeInitRoutingTilingData.sortOutComputeParamsOp;
    tilingData->oneLoopMaxElements = mrgSortListMaxElement;
}


inline void MoeInitRoutingV2TilingBase::Tiling4SrcToDstCompute() {
    auto tilingData = &moeInitRoutingTilingData.srcToDstComputeParamsOp;

    int64_t perLoopMaxRows = (aicoreParams_.ubSize - ASSIST_NUM * sizeof(float) - aivNum * SORT32_ALIGN_ELEMENT) /
                             (SORT32_ALIGN_ELEMENT * NUM_TWO) / NUM_TWO;
    int64_t perCoreRows = CeilDiv(totalLength, aivNum);
    if (perCoreRows <= 0) {
        tilingData->needCoreNum = 0;
        return;
    }

    int64_t needCoreNum = CeilDiv(totalLength, perCoreRows);
    tilingData->needCoreNum = needCoreNum;
    int64_t lastCoreNum = totalLength - perCoreRows * (tilingData->needCoreNum - 1);
    tilingData->perCoreRows = perCoreRows;
    if (perLoopMaxRows >= tilingData->perCoreRows) {
        tilingData->perCorePerLoopRows = tilingData->perCoreRows;
        tilingData->perCoreLastLoopRows = tilingData->perCoreRows;
    } else {
        tilingData->perCorePerLoopRows = perLoopMaxRows;
        tilingData->perCoreLastLoopRows =
            tilingData->perCoreRows - (CeilDiv(tilingData->perCoreRows, perLoopMaxRows) - 1) * perLoopMaxRows;
    }
    tilingData->lastCoreRows = lastCoreNum;
    if (perLoopMaxRows >= tilingData->lastCoreRows) {
        tilingData->lastCorePerLoopRows = tilingData->lastCoreRows;
        tilingData->lastCoreLastLoopRows = tilingData->lastCoreRows;
    } else {
        tilingData->lastCorePerLoopRows = perLoopMaxRows;
        tilingData->lastCoreLastLoopRows =
            tilingData->lastCoreRows - (CeilDiv(tilingData->lastCoreRows, perLoopMaxRows) - 1) * perLoopMaxRows;
    }
}


inline void MoeInitRoutingV2TilingBase::Tiling4SrcToDstCapacityCompute() {
    auto tilingData = &moeInitRoutingTilingData.srcToDstCapacityComputeParamsOp;
    int64_t perCoreRows = CeilDiv(totalLength, aivNum);
    if (perCoreRows <= 0) {
        tilingData->needCoreNum = 0;
        return;
    }

    int64_t needCoreNum = CeilDiv(totalLength, perCoreRows);
    tilingData->needCoreNum = needCoreNum;
    int64_t cols = moeInitRoutingTilingData.cols;
    tilingData->perCoreRows = perCoreRows;
    int64_t lastCoreRows = totalLength - perCoreRows * (needCoreNum - 1);
    tilingData->lastCoreRows = lastCoreRows;

    int64_t rowSize =
        (perCoreRows * sizeof(int32_t) * 2 + ONE_BLOCK_BYTE + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    int64_t colSize = (cols * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;

    if (rowSize + colSize < static_cast<int64_t>(aicoreParams_.ubSize)) {
        tilingData->perCorePerLoopRows = perCoreRows;
        tilingData->perCoreLastLoopRows = perCoreRows;
        tilingData->lastCorePerLoopRows = lastCoreRows;
        tilingData->lastCoreLastLoopRows = lastCoreRows;
        tilingData->perCoreLoops = 1;
        tilingData->lastCoreLoops = 1;
        tilingData->perLoopCols = cols;
        tilingData->lastLoopCols = cols;
        tilingData->colLoops = 1;
    } else {
        int64_t baseMaxCols = MAX_COLS_ONE_LOOP;
        int64_t baseMaxColsSize =
            (baseMaxCols * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        int64_t basePerLoopMaxRows =
            (static_cast<int64_t>(aicoreParams_.ubSize) - baseMaxColsSize - ONE_BLOCK_BYTE) /
            static_cast<int64_t>(sizeof(int32_t)) / NUM_TWO / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        if (cols < MAX_COLS_ONE_LOOP) {
            basePerLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) - colSize - ONE_BLOCK_BYTE) /
                                 static_cast<int64_t>(sizeof(int32_t)) / NUM_TWO / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        } else if (perCoreRows < basePerLoopMaxRows) {
            baseMaxCols = (static_cast<int64_t>(aicoreParams_.ubSize) - rowSize) / inuptXDtypeSize_ /
                          ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        }
        tilingData->perLoopCols = (std::min(baseMaxCols, cols));
        tilingData->lastLoopCols = (GetPerOrLastValue(cols, baseMaxCols));
        tilingData->colLoops = ((cols + baseMaxCols - 1) / baseMaxCols);
        tilingData->perCorePerLoopRows = (std::min(perCoreRows, basePerLoopMaxRows));
        tilingData->perCoreLastLoopRows = (GetPerOrLastValue(perCoreRows, basePerLoopMaxRows));
        tilingData->perCoreLoops = ((perCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);
        tilingData->lastCorePerLoopRows = (std::min(lastCoreRows, basePerLoopMaxRows));
        tilingData->lastCoreLastLoopRows = (GetPerOrLastValue(lastCoreRows, basePerLoopMaxRows));
        tilingData->lastCoreLoops = ((lastCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);
    }
}


inline void MoeInitRoutingV2TilingBase::Tiling4GatherOutCompute() {
    auto tilingData = &moeInitRoutingTilingData.gatherOutComputeParamsOp;
    tilingData->activateRows = totalLength;
    if (dropPadMode == 0) {
        tilingData->activateRows = activateNum;
    }
    int64_t perCoreRows = CeilDiv(totalLength, aivNum);
    if (perCoreRows <= 0 || moeInitRoutingTilingData.cols <= 0) {
        tilingData->needCoreNum = 0;
        return;
    }
    tilingData->needCoreNum = CeilDiv(totalLength, perCoreRows);
    int64_t cols = moeInitRoutingTilingData.cols;
    tilingData->perCoreRows = perCoreRows;
    int64_t lastCoreRows = totalLength - perCoreRows * (tilingData->needCoreNum - 1);
    tilingData->lastCoreRows = lastCoreRows;

    int64_t rowSize = (perCoreRows * sizeof(int32_t) + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
    int64_t colSize = (cols * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;

    if (rowSize + colSize < static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO) {
        tilingData->perCorePerLoopRows = perCoreRows;
        tilingData->perCoreLastLoopRows = perCoreRows;
        tilingData->lastCorePerLoopRows = lastCoreRows;
        tilingData->lastCoreLastLoopRows = lastCoreRows;
        tilingData->perCoreLoops = 1;
        tilingData->lastCoreLoops = 1;
        tilingData->perLoopCols = cols;
        tilingData->lastLoopCols = cols;
        tilingData->colLoops = 1;
    } else {
        int64_t baseMaxCols = MAX_COLS_ONE_LOOP;
        int64_t baseMaxColsSize =
            (baseMaxCols * inuptXDtypeSize_ + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        int64_t basePerLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO - baseMaxColsSize) /
                                     static_cast<int64_t>(sizeof(int32_t)) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        if (cols < MAX_COLS_ONE_LOOP) {
            basePerLoopMaxRows = (static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO - colSize) /
                                 static_cast<int64_t>(sizeof(int32_t)) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        } else if (perCoreRows < basePerLoopMaxRows) {
            baseMaxCols = (static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO - rowSize) / inuptXDtypeSize_ /
                          ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        }
        tilingData->perLoopCols = (std::min(baseMaxCols, cols));
        tilingData->lastLoopCols = (GetPerOrLastValue(cols, baseMaxCols));
        tilingData->colLoops = ((cols + baseMaxCols - 1) / baseMaxCols);

        tilingData->perCorePerLoopRows = (std::min(perCoreRows, basePerLoopMaxRows));
        tilingData->perCoreLastLoopRows = (GetPerOrLastValue(perCoreRows, basePerLoopMaxRows));
        tilingData->perCoreLoops = ((perCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);

        tilingData->lastCorePerLoopRows = (std::min(lastCoreRows, basePerLoopMaxRows));
        tilingData->lastCoreLastLoopRows = (GetPerOrLastValue(lastCoreRows, basePerLoopMaxRows));
        tilingData->lastCoreLoops = ((lastCoreRows + basePerLoopMaxRows - 1) / basePerLoopMaxRows);
    }
}


inline bool MoeInitRoutingV2TilingBase::IsFullLoad() {
    if (totalLength > sortLoopMaxElement || moeInitRoutingTilingData.cols > MAX_COLS_ONE_LOOP ||
        this->dropPadMode == 1) {
        return false;
    }
    int64_t sortBufferNum = ONE_CORE_SORT_BUFFER;
    int64_t sortSpace =
        CeilDiv(this->totalLength, SORT32_ALIGN_ELEMENT) * SORT32_ALIGN_ELEMENT * sizeof(int32_t) * sortBufferNum;
    int64_t otherSpace =
        CeilDiv(this->totalLength, SORT32_ALIGN_ELEMENT) * SORT32_ALIGN_ELEMENT * sizeof(int32_t) * NUM_THREE;
    int64_t expertSpace = CeilDiv(this->expertNum * int64_t(sizeof(int32_t)), ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE;
    int64_t perCoreXRows = moeInitRoutingTilingData.n / aivNum;
    int64_t remainder = moeInitRoutingTilingData.n % aivNum;
    // NUM_TWO is Max xRows need add 2 because the left and right row may be another row.
    perCoreXRows = remainder <= 1 ? perCoreXRows + 1 : perCoreXRows + NUM_TWO;
    int64_t gatherSpace =
        CeilDiv(moeInitRoutingTilingData.cols * inuptXDtypeSize_, ONE_BLOCK_BYTE) * ONE_BLOCK_BYTE * perCoreXRows;
    int64_t remainUbAfterSort = aicoreParams_.ubSize - sortSpace - otherSpace - expertSpace - gatherSpace;
    return remainUbAfterSort > 0;
}

}
#endif