#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <limits>
#include "../../op_kernel/mega_moe_tiling_def.h"
namespace optiling {

// ===== 常量定义（从原文件复制） =====
const static int64_t SIMT_DCACHE_SIZE = 64 * 1024LL;
const static int64_t SORT_API_MAX_ELEM = 32 * 255LL;
const static int64_t MRG_SORT_API_MAX_ELEM = 1024LL;
const static int64_t MX_QUANT_BLOCK_SIZE = 32LL;

const static int64_t NUM_TWO = 2LL;
const static int64_t NUM_THREE = 3LL;
const static int64_t NUM_FOUR = 4LL;
const static int64_t MRG_LIST_NUM = 4LL;
const static int64_t SORT32_ALIGN_ELEMENT = 32LL;
const static int64_t UB_BLOCK_SIZE = 32LL;
const static size_t DIM_ONE = 1ULL;
const static size_t DIM_TWO = 2ULL;
const static int64_t KV_FACTOR = 2LL;
const static int64_t EXPERT_IDX_MAX = 10240LL;
const static int64_t KV_MODE_EXPERT_IDX_MAX = EXPERT_IDX_MAX / KV_FACTOR;
const static int64_t RANK_ONE = 1LL;
const static int64_t RANK_TWO = 2LL;

const static int64_t DYNAMIC_QUANT_COLS_BUFFER = 21LL;

const static int64_t ROW_IDX_GATHER = 0LL;
const static int64_t ROW_IDX_SCATTER = 1LL;
const static int64_t QUANT_MODE_UNQUANT = -1LL;
const static int64_t QUANT_MODE_STATIC = 0LL;
const static int64_t QUANT_MODE_DYNAMIC = 1LL;
const static int64_t QUANT_MODE_MXFP8_E5M2 = 2LL;
const static int64_t QUANT_MODE_MXFP8_E4M3FN = 3LL;
const static int64_t EXPERT_TOKENS_TYPE_COUNT = 1LL;
const static int64_t EXPERT_TOKENS_TYPE_KEY_VALUE = 2LL;
const static int64_t DROP_PAD_MODE_DROPLESS = 0LL;
const static int64_t SORT_CORE_TILINGKEY_BASE = 100000LL;
// ===== 工具函数 =====

template<typename T>
inline static T CeilDiv(T x, T y)
{
    return (x + y - 1) / y;
}

template<typename T>
inline static T CeilAlign(T x, T align)
{
    return ((x + align - 1) / align) * align;
}

inline static int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE * UB_BLOCK_SIZE / bytes;
}

inline static int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + UB_BLOCK_SIZE - 1) / UB_BLOCK_SIZE * UB_BLOCK_SIZE;
}

inline static int64_t CeilLog4(int64_t x)
{
    if (x <= 1) return 0;
    return static_cast<int64_t>(std::ceil(std::log(x) / std::log(NUM_FOUR)));
}

inline static int64_t GetSizeByDataType(int32_t dtype)
{
    // ge::DataType枚举值映射到字节数
    switch (dtype) {
        case 1: return 64;    // DT_DOUBLE
        case 2: return 4;     // DT_FLOAT
        case 3: return 4;     // DT_FLOAT32 (alias)
        case 4: return 2;     // DT_FLOAT16
        case 5: return 1;     // DT_INT8
        case 6: return 2;     // DT_INT16
        case 7: return 4;     // DT_INT32
        case 8: return 8;     // DT_INT64
        case 9: return 1;     // DT_UINT8
        case 10: return 2;    // DT_UINT16
        case 11: return 4;    // DT_UINT32
        case 12: return 8;    // DT_UINT64
        case 13: return 1;    // DT_BOOL
        case 14: return 2;    // DT_INT4
        case 16: return 2;    // DT_BF16
        default: return -1;   // 未知类型
    }
}

// ===== Tiling计算辅助结构 =====

struct TilingContext {
    // 平台信息
    int64_t aivCoreNum;
    int64_t totalUbSize;
    int64_t availUbSize;

    // 重要变量
    int64_t sortLoopMaxElement;
    int64_t totalLength;
    int64_t n;
    int64_t k;
    int64_t cols;
    int64_t inputXDtypeSize;
    int64_t isInputScale;
    int64_t isInputOffset;
    int64_t sortMode;

    // 算子属性
    int64_t activeNum;
    int64_t expertCapacity;
    int64_t expertNum;
    int64_t dropPadMode;
    int64_t expertTokensNumType;
    bool expertTokensNumFlag;
    int64_t quantMode;
    int64_t expertStart;
    int64_t expertEnd;
    int64_t rowIdxType;
};

// ===== Tiling计算阶段函数 =====

inline static void Tiling4VBSOneCore(MoeV3Arch35VBSComputeTilingData *vbsTiling,
                                     const TilingContext& ctx) {
    vbsTiling->needCoreNum = 1;
    vbsTiling->perCoreElements = ctx.totalLength;
    vbsTiling->perCoreLoops = CeilDiv(ctx.totalLength, ctx.sortLoopMaxElement);
    vbsTiling->perCorePerLoopElements = std::min(ctx.totalLength, ctx.sortLoopMaxElement);
    vbsTiling->perCoreLastLoopElements = vbsTiling->perCoreElements;
    vbsTiling->lastCoreElements = vbsTiling->perCoreElements;
    vbsTiling->lastCoreLoops = 1;
    vbsTiling->lastCorePerLoopElements = vbsTiling->perCoreElements;
    vbsTiling->lastCoreLastLoopElements = vbsTiling->perCoreElements;
}

inline static void Tiling4VBSMultiCore(MoeV3Arch35VBSComputeTilingData *vbsTiling,
                                        TilingContext& ctx) {
    int64_t needCoreNum = CeilDiv(ctx.totalLength, ctx.sortLoopMaxElement);
    needCoreNum = static_cast<int64_t>(std::pow(4, CeilLog4(needCoreNum)));
    needCoreNum = std::min(needCoreNum, ctx.aivCoreNum);

    if (needCoreNum == 0) {
        needCoreNum = 1;
    }
    int64_t perCoreElements = (needCoreNum == 0) ? 0 : (ctx.totalLength / needCoreNum);
    int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
    int64_t lastCoreElement = ctx.totalLength - (needCoreNum - 1) * alineFloorPerCoreElements;
    int64_t alineCeilPerCoreElements = perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
    if (lastCoreElement > alineCeilPerCoreElements) {
        perCoreElements = alineCeilPerCoreElements;
        needCoreNum = CeilDiv(ctx.totalLength, perCoreElements);
    } else {
        perCoreElements = alineFloorPerCoreElements;
    }

    vbsTiling->needCoreNum = needCoreNum;
    do {
        vbsTiling->perCoreElements = perCoreElements;
        vbsTiling->perCoreLoops = CeilDiv(vbsTiling->perCoreElements, ctx.sortLoopMaxElement);
        vbsTiling->perCorePerLoopElements = std::min(vbsTiling->perCoreElements, ctx.sortLoopMaxElement);

        vbsTiling->perCoreLastLoopElements =
            vbsTiling->perCoreElements - (vbsTiling->perCoreLoops - 1) * vbsTiling->perCorePerLoopElements;

        vbsTiling->lastCoreElements = ctx.totalLength - (vbsTiling->needCoreNum - 1) * vbsTiling->perCoreElements;
        vbsTiling->lastCoreLoops = vbsTiling->perCoreLoops;
        int64_t lastCorePerLoopElements =
            CeilAlign(CeilDiv(vbsTiling->lastCoreElements, vbsTiling->lastCoreLoops), SORT32_ALIGN_ELEMENT);
        vbsTiling->lastCorePerLoopElements = lastCorePerLoopElements;
        vbsTiling->lastCoreLastLoopElements =
            vbsTiling->lastCoreElements - (vbsTiling->lastCoreLoops - 1) * vbsTiling->lastCorePerLoopElements;
        perCoreElements -= SORT32_ALIGN_ELEMENT;
    } while (vbsTiling->lastCoreLastLoopElements <= 0 && perCoreElements > 0);
}

inline static void Tiling4VBSCompute(MoeInitRoutingV3Arch35TilingData *tilingData,
                                     TilingContext& ctx) {
    auto *vbsTiling = &(tilingData->vbsComputeParamsOp);
    vbsTiling->oneLoopMaxElements = ctx.sortLoopMaxElement;

    if (ctx.totalLength <= ctx.sortLoopMaxElement) {
        ctx.sortMode = 0;
        Tiling4VBSOneCore(vbsTiling, ctx);
    } else {
        ctx.sortMode = 1;
        Tiling4VBSMultiCore(vbsTiling, ctx);
    }
    tilingData->tilingKey = ctx.sortMode * SORT_CORE_TILINGKEY_BASE + 1031000;
}

inline static void Tiling4VMSMiddleCompute(MoeInitRoutingV3Arch35TilingData *tilingData) {
    auto *vbsTiling = &(tilingData->vbsComputeParamsOp);
    auto *vmsMiddleTiling = &(tilingData->vmsMiddleComputeParamsOp);

    if (vbsTiling->needCoreNum <= MRG_LIST_NUM) {
        vmsMiddleTiling->needCoreNum = 0;
        return;
    }
    int64_t needCoreNum = CeilDiv(vbsTiling->needCoreNum, MRG_LIST_NUM);
    vmsMiddleTiling->needCoreNum = needCoreNum;
}

inline static void Tiling4SortOutCompute(MoeInitRoutingV3Arch35TilingData *tilingData) {
    auto *sortOutTiling = &(tilingData->sortOutComputeParamsOp);
    sortOutTiling->oneLoopMaxElements = MRG_SORT_API_MAX_ELEM;
}

inline static void Tiling4ExpertTokensCountCompute(MoeInitRoutingV3Arch35TilingData *tilingData,
                                                    const TilingContext& ctx) {
    auto *tokensCountTiling = &(tilingData->expertTokensCountTilingDataOp);
    int64_t totalElements = tilingData->n * tilingData->k;
    int64_t perCoreElements = CeilDiv(totalElements, ctx.aivCoreNum);
    int64_t needCoreNum = CeilDiv(totalElements, perCoreElements);
    int64_t lastCoreElements = totalElements - (needCoreNum - 1) * perCoreElements;

    tokensCountTiling->needCoreNum = needCoreNum;
    tokensCountTiling->perCoreElements = perCoreElements;
    tokensCountTiling->lastCoreElements = lastCoreElements;

    int64_t expertNumElement = (tilingData->expertTokensNumType != EXPERT_TOKENS_TYPE_KEY_VALUE) ?
                                   tilingData->actualExpertNum :
                                   (tilingData->actualExpertNum + 1) * DIM_TWO;
    int64_t maxElementsPerLoop =
        (ctx.availUbSize -
         CeilAlign(expertNumElement, UB_BLOCK_SIZE) *
             (static_cast<int64_t>(sizeof(int32_t)) * NUM_TWO + static_cast<int64_t>(sizeof(int64_t))) -
         UB_BLOCK_SIZE) /
        static_cast<int64_t>(sizeof(int32_t));

    int64_t perCoreLoops = CeilDiv(perCoreElements, maxElementsPerLoop);
    int64_t perCorePerLoopElements = CeilDiv(perCoreElements, perCoreLoops);
    int64_t perCoreLastLoopElements = perCoreElements - (perCoreLoops - 1) * perCorePerLoopElements;
    tokensCountTiling->perCoreLoops = perCoreLoops;
    tokensCountTiling->perCorePerLoopElements = perCorePerLoopElements;
    tokensCountTiling->perCoreLastLoopElements = perCoreLastLoopElements;

    int64_t lastCoreLoops = CeilDiv(lastCoreElements, maxElementsPerLoop);
    int64_t lastCorePerLoopElements = CeilDiv(lastCoreElements, lastCoreLoops);
    int64_t lastCoreLastLoopElements = lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoopElements;
    tokensCountTiling->lastCoreLoops = lastCoreLoops;
    tokensCountTiling->lastCorePerLoopElements = lastCorePerLoopElements;
    tokensCountTiling->lastCoreLastLoopElements = lastCoreLastLoopElements;
}

inline static void Tiling4GatherOutCompute(MoeInitRoutingV3Arch35TilingData *tilingData,
                                           const TilingContext& ctx) {
    auto *gatherOutTiling = &(tilingData->gatherOutComputeParamsOp);
    int64_t perCoreIndicesElements = CeilDiv(ctx.totalLength, ctx.aivCoreNum);
    if (perCoreIndicesElements <= 0) {
        gatherOutTiling->needCoreNum = 0;
        return;
    }
    int64_t needCoreNum = CeilDiv(ctx.totalLength, perCoreIndicesElements);
    int64_t lastCoreIndicesElements = ctx.totalLength - (needCoreNum - 1) * perCoreIndicesElements;

    int64_t perLoopCols = tilingData->cols;
    int64_t colMultiple = NUM_TWO * ctx.inputXDtypeSize;
    int64_t rowMultiple = NUM_TWO;
    if (ctx.quantMode == QUANT_MODE_DYNAMIC) {
        colMultiple = DYNAMIC_QUANT_COLS_BUFFER;
        rowMultiple = NUM_FOUR;
    }
    int64_t perLoopMaxIndicesElements =
        (ctx.availUbSize - Align(perLoopCols, ctx.inputXDtypeSize) * colMultiple - UB_BLOCK_SIZE * NUM_TWO) / rowMultiple /
        static_cast<int64_t>(sizeof(int32_t));
    while (perLoopMaxIndicesElements <= 0) {
        perLoopCols = CeilDiv(perLoopCols, NUM_TWO);
        perLoopMaxIndicesElements =
            (ctx.availUbSize - Align(perLoopCols, ctx.inputXDtypeSize) * colMultiple - UB_BLOCK_SIZE * NUM_TWO) /
            rowMultiple / static_cast<int64_t>(sizeof(int32_t));
    }
    int64_t colsLoops = CeilDiv(tilingData->cols, perLoopCols);
    int64_t lastLoopCols = tilingData->cols - (colsLoops - 1) * perLoopCols;
    gatherOutTiling->needCoreNum = needCoreNum;
    gatherOutTiling->perCoreIndicesElements = perCoreIndicesElements;
    gatherOutTiling->lastCoreIndicesElements = lastCoreIndicesElements;
    gatherOutTiling->colsLoops = colsLoops;
    gatherOutTiling->perLoopCols = perLoopCols;
    gatherOutTiling->lastLoopCols = lastLoopCols;

    int64_t perCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, perCoreIndicesElements);
    int64_t perCoreIndicesLoops = CeilDiv(perCoreIndicesElements, perCorePerLoopIndicesElements);
    int64_t perCoreLastLoopIndicesElements =
        perCoreIndicesElements - (perCoreIndicesLoops - 1) * perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreIndicesLoops = perCoreIndicesLoops;
    gatherOutTiling->perCorePerLoopIndicesElements = perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreLastLoopIndicesElements = perCoreLastLoopIndicesElements;

    int64_t lastCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, lastCoreIndicesElements);
    int64_t lastCoreIndicesLoops = CeilDiv(lastCoreIndicesElements, lastCorePerLoopIndicesElements);
    int64_t lastCoreLastLoopIndicesElements =
        lastCoreIndicesElements - (lastCoreIndicesLoops - 1) * lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreIndicesLoops = lastCoreIndicesLoops;
    gatherOutTiling->lastCorePerLoopIndicesElements = lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreLastLoopIndicesElements = lastCoreLastLoopIndicesElements;
}

inline static int64_t CalcMaxRowIdxPerLoopMxQuant(int64_t perLoopCols, int64_t inputXDtypeSize,
                                                    int64_t availUbSize) {
    int64_t xInSize = AlignBytes(perLoopCols, inputXDtypeSize) + AlignBytes(perLoopCols, sizeof(int8_t));
    int64_t scaleSize = 2 * AlignBytes(perLoopCols / MX_QUANT_BLOCK_SIZE, inputXDtypeSize) +
                        AlignBytes(perLoopCols / MX_QUANT_BLOCK_SIZE, sizeof(int8_t));
    int64_t xOutSize = Align(perLoopCols / 4, sizeof(int8_t)) * 4;
    return (availUbSize - (xInSize + scaleSize + xOutSize)) / static_cast<int64_t>(sizeof(int32_t));
}

inline static void Tiling4GatherOutMxQuant(MoeInitRoutingV3Arch35TilingData *tilingData,
                                           const TilingContext& ctx) {
    auto *gatherOutTiling = &(tilingData->gatherOutComputeParamsOp);
    int64_t perCoreIndicesElements = CeilDiv(ctx.totalLength, ctx.aivCoreNum);
    if (perCoreIndicesElements <= 0) {
        gatherOutTiling->needCoreNum = 0;
        return;
    }
    int64_t needCoreNum = CeilDiv(ctx.totalLength, perCoreIndicesElements);
    int64_t lastCoreIndicesElements = ctx.totalLength - (needCoreNum - 1) * perCoreIndicesElements;

    int64_t perLoopCols = CeilAlign(tilingData->cols, MX_QUANT_BLOCK_SIZE);
    int64_t perLoopMaxIndicesElements = CalcMaxRowIdxPerLoopMxQuant(perLoopCols, ctx.inputXDtypeSize, ctx.availUbSize);
    while (perLoopMaxIndicesElements <= 0) {
        perLoopCols = CeilAlign(CeilDiv(perLoopCols, NUM_TWO), MX_QUANT_BLOCK_SIZE);
        perLoopMaxIndicesElements = CalcMaxRowIdxPerLoopMxQuant(perLoopCols, ctx.inputXDtypeSize, ctx.availUbSize);
    }
    int64_t colsLoops = CeilDiv(tilingData->cols, perLoopCols);
    int64_t lastLoopCols = tilingData->cols - (colsLoops - 1) * perLoopCols;
    gatherOutTiling->needCoreNum = needCoreNum;
    gatherOutTiling->perCoreIndicesElements = perCoreIndicesElements;
    gatherOutTiling->lastCoreIndicesElements = lastCoreIndicesElements;
    gatherOutTiling->colsLoops = colsLoops;
    gatherOutTiling->perLoopCols = perLoopCols;
    gatherOutTiling->lastLoopCols = lastLoopCols;

    int64_t perCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, perCoreIndicesElements);
    int64_t perCoreIndicesLoops = CeilDiv(perCoreIndicesElements, perCorePerLoopIndicesElements);
    int64_t perCoreLastLoopIndicesElements =
        perCoreIndicesElements - (perCoreIndicesLoops - 1) * perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreIndicesLoops = perCoreIndicesLoops;
    gatherOutTiling->perCorePerLoopIndicesElements = perCorePerLoopIndicesElements;
    gatherOutTiling->perCoreLastLoopIndicesElements = perCoreLastLoopIndicesElements;

    int64_t lastCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, lastCoreIndicesElements);
    int64_t lastCoreIndicesLoops = CeilDiv(lastCoreIndicesElements, lastCorePerLoopIndicesElements);
    int64_t lastCoreLastLoopIndicesElements =
        lastCoreIndicesElements - (lastCoreIndicesLoops - 1) * lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreIndicesLoops = lastCoreIndicesLoops;
    gatherOutTiling->lastCorePerLoopIndicesElements = lastCorePerLoopIndicesElements;
    gatherOutTiling->lastCoreLastLoopIndicesElements = lastCoreLastLoopIndicesElements;
}

// ===== 主函数 =====

MoeInitRoutingV3Arch35TilingData ComputeMoeInitRoutingV3Tiling( // 参数构造，走进这个函数
    // 输入形状
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& expert_idx_shape,

    // // 输出形状（用于验证）
    // const std::vector<int64_t>& expanded_x_shape,
    // const std::vector<int64_t>& expanded_row_idx_shape,
    // const std::vector<int64_t>& expert_tokens_count_shape,
    // const std::vector<int64_t>& expanded_scale_shape,

    // 输入数据类型 (int32类型标识，参考ge::DataType枚举值: 4=FLOAT16, 16=BF16, 2=FLOAT, 5=INT8)
    int32_t x_dtype,
    int32_t expert_idx_dtype,

    // 可选输入标志和类型
    bool is_input_scale,
    bool is_input_offset,
    int32_t scale_dtype, // 参数没用到？
    int32_t offset_dtype, // 参数没用到？

    // 输出类型
    int32_t expanded_x_dtype,
    int32_t expanded_row_idx_dtype,
    int32_t expert_tokens_count_dtype,
    int32_t expanded_scale_dtype,

    // 算子属性
    int64_t active_num,
    int64_t expert_capacity,
    int64_t expert_num,
    int64_t drop_pad_mode,
    int64_t expert_tokens_num_type,
    bool expert_tokens_num_flag,
    int64_t quant_mode,
    const std::vector<int64_t>& expert_range,
    int64_t row_idx_type,

    // 平台信息
    int64_t aivNum,
    int64_t ubSize
) {
    MoeInitRoutingV3Arch35TilingData tilingData;
    TilingContext ctx;

    // ===== 1. 设置平台信息 =====
    ctx.aivCoreNum = aivNum;
    ctx.totalUbSize = ubSize;
    ctx.availUbSize = ubSize - SIMT_DCACHE_SIZE;

    tilingData.coreNum = aivNum;

    // ===== 2. 设置算子属性到tilingData =====
    // expertTokensNumType
    if (expert_tokens_num_type != EXPERT_TOKENS_TYPE_COUNT &&
        expert_tokens_num_type != EXPERT_TOKENS_TYPE_KEY_VALUE) {
        printf("错误:：不支持的expert_tokens_num_type\n");
        expert_tokens_num_type = EXPERT_TOKENS_TYPE_COUNT;
    }
    tilingData.expertTokensNumType = expert_tokens_num_type;

    // expertNum
    int64_t maxExpertNum = (expert_tokens_num_type == EXPERT_TOKENS_TYPE_COUNT) ?
                            EXPERT_IDX_MAX : KV_MODE_EXPERT_IDX_MAX;
    if (expert_num <= 0 || expert_num > maxExpertNum) {
        printf("错误:：expert_num超出范围\n");
    }
    tilingData.expertNum = expert_num;

    // dropPadMode
    if (drop_pad_mode != DROP_PAD_MODE_DROPLESS) {
        printf("错误:：不支持的drop_pad_mode\n");
    }
    tilingData.dropPadMode = drop_pad_mode;

    // expertTokensNumFlag
    if (!expert_tokens_num_flag) {
        printf("错误:：expert_tokens_num_flag必须为true\n");
    }
    tilingData.expertTokensNumFlag = expert_tokens_num_flag ? 1 : 0;

    // quantMode
    if (quant_mode != QUANT_MODE_UNQUANT && quant_mode != QUANT_MODE_DYNAMIC &&
        quant_mode != QUANT_MODE_MXFP8_E5M2 && quant_mode != QUANT_MODE_MXFP8_E4M3FN) {
        printf("错误:：不支持的quant_mode\n");
    }
    tilingData.quantMode = quant_mode;

    // rowIdxType
    if (row_idx_type != ROW_IDX_SCATTER && row_idx_type != ROW_IDX_GATHER) {
        printf("错误:：不支持的row_idx_type\n");
    }
    tilingData.rowIdxType = row_idx_type;

    // expertStart, expertEnd
    if (expert_range.size() != 2) {
        printf("错误:：expert_range长度必须为2\n");
    }
    int64_t expertStart = expert_range[0];
    int64_t expertEnd = expert_range[1];
    if (expertStart < 0 || expertStart >= expertEnd || expertEnd > expert_num) {
        printf("错误:：expert_range值非法\n");
    }
    tilingData.expertStart = expertStart;
    tilingData.expertEnd = expertEnd;
    tilingData.actualExpertNum = expertEnd - expertStart;

    // ===== 3. 从shapes提取关键信息 =====
    // 输入X的shape
    if (x_shape.size() != 2) {
        printf("错误:：x_shape.rank必须为2\n");
    }
    int64_t n = x_shape[0];
    int64_t cols = x_shape[1];

    // 输入ExpertIdx的shape
    if (expert_idx_shape.size() != 2) {
        printf("错误:：expert_idx_shape.rank必须为2\n");
    }
    if (expert_idx_shape[0] != n) {
        printf("错误:：expert_idx_shape[0]必须等于x_shape[0]\n");
    }
    int64_t k = expert_idx_shape[1];

    int64_t inputXDtypeSize = GetSizeByDataType(x_dtype);
    if (inputXDtypeSize < 0) {
        printf("错误:：不支持的数据类型\n");
    }

    ctx.n = n;
    ctx.k = k;
    ctx.cols = cols;
    ctx.inputXDtypeSize = inputXDtypeSize;
    ctx.isInputScale = is_input_scale ? 1 : 0;
    ctx.isInputOffset = is_input_offset ? 1 : 0;
    ctx.activeNum = active_num;
    ctx.expertCapacity = expert_capacity;
    ctx.expertNum = expert_num;
    ctx.dropPadMode = drop_pad_mode;
    ctx.expertTokensNumType = expert_tokens_num_type;
    ctx.expertTokensNumFlag = expert_tokens_num_flag;
    ctx.quantMode = quant_mode;
    ctx.expertStart = expertStart;
    ctx.expertEnd = expertEnd;
    ctx.rowIdxType = row_idx_type;
    ctx.totalLength = n * k;

    tilingData.n = n;
    tilingData.k = k;
    tilingData.cols = cols;
    tilingData.isInputScale = ctx.isInputScale;
    tilingData.isInputOffset = ctx.isInputOffset;

    // activeNum验证（保留以保持兼容性，虽然不使用）
    const static int64_t ACTIVE_NUM_MIN_VALUE = -1LL;
    if (active_num != ACTIVE_NUM_MIN_VALUE) {
        if (active_num != ctx.totalLength) {
            printf("错误:：active_num必须等于n*k\n");
        }
    }

    // ===== 4. 计算sortLoopMaxElement =====
    ctx.sortLoopMaxElement = ctx.availUbSize / (NUM_FOUR * NUM_TWO * NUM_FOUR) / SORT32_ALIGN_ELEMENT * SORT32_ALIGN_ELEMENT;
    ctx.sortLoopMaxElement = std::min(ctx.sortLoopMaxElement, SORT_API_MAX_ELEM);

    // ===== 5. 调用5个阶段的Tiling计算 =====
    Tiling4VBSCompute(&tilingData, ctx);
    Tiling4VMSMiddleCompute(&tilingData);
    Tiling4SortOutCompute(&tilingData);
    Tiling4ExpertTokensCountCompute(&tilingData, ctx);

    // 根据quantMode选择不同的拷出阶段逻辑
    if (quant_mode == QUANT_MODE_MXFP8_E5M2 || quant_mode == QUANT_MODE_MXFP8_E4M3FN) {
        Tiling4GatherOutMxQuant(&tilingData, ctx);
    } else {
        Tiling4GatherOutCompute(&tilingData, ctx);
    }

    // 其他用于向后兼容的字段（原代码中有，但注释为未使用）
    tilingData.gatherFirstFullload = 0;
    tilingData.epFullload = 0;  // 未使用
    tilingData.activeNum = ctx.totalLength;
    tilingData.smoothType = 0;  // 暂不支持

    return tilingData;
}

void GetMoeInitRoutingV3Tiling(MegaMoeTilingData &tilingData, uint32_t aivNum, uint64_t ubSize)
{
    auto m = tilingData.m;
    auto k = tilingData.k;
    auto topK = tilingData.topK;
    auto expertPerRank = tilingData.expertPerRank;
    auto EP = tilingData.EP;
    std::vector<int64_t> x_shape = {m, k};
    std::vector<int64_t> expert_idx_shape = {m, topK};
    int32_t x_dtype = 16; // DT_BF16
    int32_t expert_idx_dtype = 7; // DT_INT32
    bool is_input_scale = false;
    bool is_input_offset = false;
    int32_t scale_dtype = 0; // 没用到
    int32_t offset_dtype = 0; // 没用到

    // 输出类型
    int32_t expanded_x_dtype = 0; // 没用到
    int32_t expanded_row_idx_dtype = 0; // 没用到
    int32_t expert_tokens_count_dtype = 0; // 没用到
    int32_t expanded_scale_dtype = 0; // 没用到

    // 算子属性
    int64_t active_num = m * k;
    int64_t expert_capacity = k;
    int64_t expert_num = EP * expertPerRank;
    int64_t drop_pad_mode = 0;
    int64_t expert_tokens_num_type = 1;
    bool expert_tokens_num_flag = true;
    int64_t quant_mode = 3;
    const std::vector<int64_t> expert_range = {0, expert_num};
    int64_t row_idx_type = 1;

    // --- 基础变量打印 ---
    printf("Basic Tiling Data:\n");
    printf("m: %ld, k: %ld, topK: %ld, expertPerRank: %ld, EP: %ld\n", m, k, topK, expertPerRank, EP);

    // --- Shape 向量打印 ---
    printf("x_shape: [%ld, %ld]\n", x_shape[0], x_shape[1]);
    printf("expert_idx_shape: [%ld, %ld]\n", expert_idx_shape[0], expert_idx_shape[1]);

    // --- Dtype 与 Boolean 打印 ---
    printf("Dtypes - x: %d, expert_idx: %d\n", x_dtype, expert_idx_dtype);
    printf("Flags - is_input_scale: %s, is_input_offset: %s\n", 
            is_input_scale ? "true" : "false", is_input_offset ? "true" : "false");

    // --- 算子属性打印 ---
    printf("--- Operator Attributes ---\n");
    printf("active_num: %ld\n", active_num);
    printf("expert_capacity: %ld\n", expert_capacity);
    printf("expert_num: %ld\n", expert_num);
    printf("drop_pad_mode: %ld, expert_tokens_num_type: %ld\n", drop_pad_mode, expert_tokens_num_type);
    printf("expert_tokens_num_flag: %s, quant_mode: %ld\n", 
            expert_tokens_num_flag ? "true" : "false", quant_mode);
    printf("expert_range: [%ld, %ld]\n", expert_range[0], expert_range[1]);
    printf("row_idx_type: %ld\n", row_idx_type);

    // 平台信息
    auto ss_tiling = ComputeMoeInitRoutingV3Tiling(
        x_shape, expert_idx_shape, x_dtype, expert_idx_dtype, is_input_scale, is_input_offset, scale_dtype, offset_dtype,
        expanded_x_dtype, expanded_row_idx_dtype, expert_tokens_count_dtype, expanded_scale_dtype,
        active_num, expert_capacity, expert_num, drop_pad_mode, expert_tokens_num_type, expert_tokens_num_flag, quant_mode, expert_range, row_idx_type,
        aivNum, ubSize);
    tilingData.moeInitRoutingTilingData = ss_tiling;
}

} // namespace standalone_tiling