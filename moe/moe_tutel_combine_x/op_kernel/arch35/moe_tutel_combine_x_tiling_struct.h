/**
 * =============================================================================
 * op_kernel/arch35/moe_tutel_combine_x_tiling_struct.h
 * =============================================================================
 * Role: Shared TilingData structure for MoeTutelCombineX on arch35 (Ascend 950).
 *        Included by BOTH the host tiling code (op_host/
 *        moe_tutel_combine_x_tiling_arch35.cpp) and the device kernel entry
 *        (op_kernel/moe_tutel_combine_x_apt.cpp via REGISTER_TILING_DEFAULT /
 *        GET_TILING_DATA_WITH_STRUCT).
 *
 * Tiling keys (set by the host tiling, dispatched by the kernel entry):
 *   0 - normal, no batch (y_grad is 2D, metadata 1D)
 *   1 - empty tensor shortcut (output x_grad is zeroed)
 *   2 - normal, with batch (y_grad is 3D, metadata 2D)
 *
 * Field units: every split field counts ELEMENTS (coreNum-family fields count
 *   CORES); "寻址全 int64" - B*EC*H / B*S*H offsets can exceed int32, so the
 *   shape/split fields are int64; bigCoreCnt is a core count / token remainder
 *   that fits int32.
 *
 * Memory layout note: trivially copyable only (no pointers, no virtual
 *   methods); the struct is serialised into the raw tiling blob read by the
 *   kernel via GET_TILING_DATA.
 *
 * dtype does NOT enter TilingData: dtype is dispatched at compile time via the
 *   OpDef .DataType columns (DTYPE_Y_GRAD macro binds float / half /
 *   bfloat16_t).
 *
 * To create a new operator (e.g. FooBar), replace:
 *   MOE_TUTEL_COMBINE_X_TILING_STRUCT_H -> FOO_BAR_TILING_STRUCT_H
 *   MoetutelcombinexTilingData -> FooBarTilingData
 *   moe_tutel_combine_x -> foo_bar
 * =============================================================================
 */

#ifndef MOE_TUTEL_COMBINE_X_TILING_STRUCT_H
#define MOE_TUTEL_COMBINE_X_TILING_STRUCT_H

#include <cstdint>

// ---------------------------------------------------------------------------
// Tiling keys (set by the host tiling, dispatched by the kernel entry):
//   0 - normal, no batch (y_grad is 2D)
//   1 - empty tensor shortcut (output x_grad is zeroed)
//   2 - normal, with batch (y_grad is 3D)
// ---------------------------------------------------------------------------
enum MoetutelcombinexTilingKey {
    TILING_KEY_NORMAL_NO_BATCH = 0,
    TILING_KEY_EMPTY = 1,
    TILING_KEY_NORMAL_BATCH = 2
};

// IO slot counts: y_grad / gates / indices / locations in, x_grad out.
constexpr int64_t kMaxInputSlots = 4;
constexpr int64_t kMaxOutputSlots = 1;

// MoetutelcombinexTilingData — 三分支（key=0/1/2）共用单一 struct。
// 所有 split 字段单位为元素数（coreNum 族字段单位为核数）。
struct MoetutelcombinexTilingData {
    // ─── shape / 路由元数据（host 由输入 shape 与 indices 值派生）───
    int64_t batchSize; // B：微批数（2D 形态恒 1；3D 形态 = y_grad.dim0 = gates.dim0）
    int64_t samples;   // S：单 batch 内 token 副本数（= gates.shape[-1]）
    int64_t hidden;    // H：隐藏维（= y_grad.shape[-1]，行内连续元素数）
    int64_t capacity;  // C：每专家容量 = floor(EC / E)，E = max(indices)+1；
                       //   host 不可读 indices 时置 EC（E=1 哨兵），kernel 侧扫描校正
    int64_t eC;        // EC：y_grad 展平行数（= y_grad.shape[-2]）

    // ─── 多核切分（token 维扁平化：总 token 数 = B*S）───
    int64_t tokenTotal;    // B*S：全部 token 副本总数（条目数）
    int64_t coreTokenBase; // 每核主 token 数 = floor(tokenTotal / usedCoreNum)（条目数）
    int32_t bigCoreCnt;    // 多处理一个 token 的核数 = tokenTotal % usedCoreNum（核数）

    // ─── UB 切分（H 维 chunk）───
    int64_t hChunk;     // UB 内 H 维 chunk 大小（元素数，按 dtype UB 预算反推）
    int64_t hChunkCnt;  // H 维 chunk 数 = CeilDiv(hidden, hChunk)
    int64_t hChunkTail; // 末 chunk 实际元素数 = hidden - (hChunkCnt - 1) * hChunk
};

#endif // MOE_TUTEL_COMBINE_X_TILING_STRUCT_H
