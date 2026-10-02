// sa2_sim_kv_prep.h — SageAttention2 宿主仿真 KV 预处理（纯标准 C++17，零硬件依赖）
//
// 与设备侧（op_kernel/sa2_kv_prep.h）的镜像关系（2026-10-02 三级流水收口）：
//   1. ProcessKMean：K 均值 oracle——128 行块 → 64 行子块 → 全头折叠两级 FP32 加法树。
//      设备已落地「MTE2 行块双缓冲 + 64-lane 纵向 Cast/Add 归约」，列 lane 内仍按行序逐次
//      相加 ⇒ 加法树未换，本 oracle 语句序继续逐位镜像（v31 迭代 1 义务注记延续成立）。
//   2. ProcessHeadChunkPipeline：K/V 混流三级切片流水镜像——与设备共用
//      ExecuteQuantBatchPipeline 骨架与 Sa2QuantSliceDesc 槽位调度（Load i+1 / Compute i /
//      Store i-1，输入/输出槽独立双缓冲），Stage2 公式与设备逐位同源（Sa2QuantScaleReciprocal
//      一次倒数 + aicNearbyInt（≡ 设备 CAST_RINT）钳制量化；V 组 E8M0 位技巧 + 2^-e 精确
//      二次幂倒数行 + FloatToFp8E4M3 铸码）。
//   3. ProcessKQuantScalarRef / ProcessVQuantScalarRef：流水线引入前的标量 oracle（v31 前
//      形态逐语句保留），供 conformance 证明流水线路径与标量路径逐位一致（三方对账的基准腿）。
#ifndef SA2_SIM_KV_PREP_H_
#define SA2_SIM_KV_PREP_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "sa2_spec.h"
#include "sa2_planner.h"
#include "sa2_ub_plan.h"

namespace sa2 {
namespace sim {

template <typename Dtype, Gran G>
class Sa2KvPreprocessorSim {
public:
    inline Sa2KvPreprocessorSim() = default;

    /**
     * @brief 宿主仿真：对指定 KV head 执行完整的 K/V 预处理（均值统计与量化写入）。
     */
    inline void ProcessHead(
        uint8_t *kGmBase, uint8_t *vGmBase, uint8_t *userWs,
        const Sa2TilingData &td, int64_t b, int64_t kh)
    {
        const Sa2Params &p = td.params;
        const Sa2KvHeadView view = KvHeadView(p, b, kh);

        const Dtype *gmK = reinterpret_cast<const Dtype *>(kGmBase) + view.gmK;
        const Dtype *gmV = reinterpret_cast<const Dtype *>(vGmBase) + view.gmV;

        float *wsKMean = reinterpret_cast<float *>(userWs + td.kMean + view.wsKMean);
        float *wsVMean = reinterpret_cast<float *>(userWs + td.vMean + view.wsVMean);
        int8_t *wsK8 = reinterpret_cast<int8_t *>(userWs + td.k8 + view.wsK8);
        half *wsKScale = reinterpret_cast<half *>(userWs + td.kScale + view.wsKScale);
        int8_t *wsV8 = reinterpret_cast<int8_t *>(userWs + td.v8 + view.wsV8);
        int16_t *wsVScale = reinterpret_cast<int16_t *>(userWs + td.vScale + view.wsVScale);

        // 1. K 均值统计（smooth_k=True 时；与设备 ProcessKMeanDevice 同树）
        ProcessKMean(gmK, wsKMean, p);

        // 2. V 均值统计（smooth_v=True 且 fp32 累加时；与设备 ProcessVMeanDevice 同序）
        ProcessVMean(gmV, wsVMean, p);

        // 3. K/V 混流三级切片流水量化（与设备 ProcessHeadChunk 同骨架同公式）
        ProcessHeadChunkPipeline(gmK, gmV, wsKMean, wsVMean, wsK8, wsKScale, wsV8, wsVScale, p,
                                 0, p.kp);
    }

    // ---- K 均值 oracle（设备 64-lane 纵向归约的逐位镜像：列内行序 FP32 加法树）----
    inline void ProcessKMean(
        const Dtype *gmK, float *wsKMean, const Sa2Params &p)
    {
        if (!p.smoothK || p.sk <= 0) {
            for (int64_t c = 0; c < p.dp; ++c) {
                wsKMean[c] = 0.0f;
            }
            return;
        }
        float totalSum[kMaxHeadDim] = {0.0f};
        float chunkBuf[kKMeanDoubleBufSlots][kMaxHeadDim];
        int64_t currentSlot = 0;
        for (int64_t sBlock = 0; sBlock < p.sk; sBlock += kKMeanBlockRows) {
            const int64_t blockEnd = std::min(sBlock + kKMeanBlockRows, p.sk);
            for (int64_t c = 0; c < p.d; ++c) {
                chunkBuf[currentSlot][c] = 0.0f;
            }
            for (int64_t subBlock = sBlock; subBlock < blockEnd; subBlock += kKMeanDataBlockLanes) {
                const int64_t subEnd = std::min(subBlock + kKMeanDataBlockLanes, blockEnd);
                for (int64_t s = subBlock; s < subEnd; ++s) {
                    const Dtype *row = gmK + s * p.kStride[2];
                    for (int64_t c = 0; c < p.d; ++c) {
                        chunkBuf[currentSlot][c] += static_cast<float>(row[c]);
                    }
                }
            }
            for (int64_t c = 0; c < p.d; ++c) {
                totalSum[c] += chunkBuf[currentSlot][c];
            }
            currentSlot = 1 - currentSlot;
        }
        // 全头独写（设备侧唯一可达分支；跨界汇聚 = Stream-K 预留，oracle 不镜像不可达分支）
        const float invSk = 1.0f / static_cast<float>(p.sk);
        for (int64_t c = 0; c < p.d; ++c) {
            wsKMean[c] = totalSum[c] * invSk;
        }
        for (int64_t c = p.d; c < p.dp; ++c) {
            wsKMean[c] = 0.0f;
        }
    }

    // ---- V 均值镜像（smooth_v 且 fp32 档；行主序与设备 ProcessVMeanDevice 同序）----
    inline void ProcessVMean(
        const Dtype *gmV, float *wsVMean, const Sa2Params &p)
    {
        const bool smoothVActive = p.smoothV && (p.accum == static_cast<int64_t>(PvAccum::kFp32));
        if (smoothVActive && p.sk > 0) {
            float sumVBuf[kMaxHeadDim] = {0.0f};
            for (int64_t s = 0; s < p.sk; ++s) {
                const Dtype *row = gmV + s * p.vStride[2];
                for (int64_t c = 0; c < p.d; ++c) {
                    sumVBuf[c] += static_cast<float>(row[c]);
                }
            }
            float invSk = 1.0f / static_cast<float>(p.sk);
            for (int64_t c = 0; c < p.d; ++c) {
                wsVMean[c] = sumVBuf[c] * invSk;
            }
        } else {
            for (int64_t c = 0; c < p.d; ++c) {
                wsVMean[c] = 0.0f;
            }
        }
        for (int64_t c = p.d; c < p.dp; ++c) {
            wsVMean[c] = 0.0f;
        }
    }

    // ---- K/V 混流三级切片流水镜像（与设备共用 ExecuteQuantBatchPipeline 骨架）----
    // 槽位语义与设备一致：Load 写入槽 (i+1)%2、Compute 读输入槽 i%2 写输出槽 i%2、
    // Store 读输出槽 (i-1)%2；输入/输出独立双缓冲 ⇒ 三级两两不同槽（无竞争调度的
    // 形式化断言见 conformance kv_prep_three_stage_pipeline_guard）。
    inline void ProcessHeadChunkPipeline(
        const Dtype *gmK, const Dtype *gmV, const float *wsKMean, const float *wsVMean,
        int8_t *wsK8, half *wsKScale, int8_t *wsV8, int16_t *wsVScale,
        const Sa2Params &p, int64_t chunkStart, int64_t chunkLen)
    {
        struct SimChunkCtx {
            const Dtype *gmK;
            const Dtype *gmV;
            const float *wsKMean;
            const float *wsVMean;
            int8_t *wsK8;
            half *wsKScale;
            int8_t *wsV8;
            int16_t *wsVScale;
            const Sa2Params *p;
            int64_t chunkStart;
            int64_t chunkLen;
            // Stage1/2/3 切片暂存（输入/输出独立双缓冲，槽 = desc.bufferSlot）
            float kIn[2][kKvPrepSliceRows * kMaxHeadDim];
            float vIn[2][kKvPrepSliceRows * kMaxHeadDim];
            uint8_t kOut[2][kKvPrepSliceRows * kMaxHeadDim];
            uint8_t vOut[2][kKvPrepSliceRows * kMaxHeadDim];
            half kScaleOut[2][kKvPrepSliceRows * kSegCols];  // 槽内行距 = segWidth 上界（与 GM 同构）
            uint8_t expSub[2][kMxScalePack][kMaxHeadDim];
        } ctx;

        ctx.gmK = gmK;
        ctx.gmV = gmV;
        ctx.wsKMean = wsKMean;
        ctx.wsVMean = wsVMean;
        ctx.wsK8 = wsK8;
        ctx.wsKScale = wsKScale;
        ctx.wsV8 = wsV8;
        ctx.wsVScale = wsVScale;
        ctx.p = &p;
        ctx.chunkStart = chunkStart;
        ctx.chunkLen = chunkLen;

        auto loadFn = [&](const Sa2QuantSliceDesc &desc) {
            const int64_t slot = desc.bufferSlot;
            const int64_t rows = std::min<int64_t>(kKvPrepSliceRows, ctx.chunkLen - desc.rowOffset);
            const int64_t tokenStart = ctx.chunkStart + desc.rowOffset;
            // Stage1 镜像：无效 token 行与 pad 列确定性清零后回填 [0,d)（与设备 LoadRowsToUb 一致）
            for (int64_t i = 0; i < rows * p.dp; ++i) {
                ctx.kIn[slot][i] = 0.0f;
                ctx.vIn[slot][i] = 0.0f;
            }
            for (int64_t r = 0; r < rows; ++r) {
                if (tokenStart + r >= p.sk) {
                    continue;
                }
                const Dtype *kRow = ctx.gmK + (tokenStart + r) * p.kStride[2];
                const Dtype *vRow = ctx.gmV + (tokenStart + r) * p.vStride[2];
                for (int64_t c = 0; c < p.d; ++c) {
                    ctx.kIn[slot][r * p.dp + c] = static_cast<float>(kRow[c]);
                    ctx.vIn[slot][r * p.dp + c] = static_cast<float>(vRow[c]);
                }
            }
        };
        auto computeFn = [&](const Sa2QuantSliceDesc &desc) {
            const int64_t slot = desc.bufferSlot;
            const int64_t rows = std::min<int64_t>(kKvPrepSliceRows, ctx.chunkLen - desc.rowOffset);
            const int64_t tokenStart = ctx.chunkStart + desc.rowOffset;
            SimKQuantCompute(ctx, slot, rows, tokenStart);
            SimVQuantCompute(ctx, slot, rows, tokenStart);
        };
        auto storeFn = [&](const Sa2QuantSliceDesc &desc) {
            const int64_t slot = desc.bufferSlot;
            const int64_t rows = std::min<int64_t>(kKvPrepSliceRows, ctx.chunkLen - desc.rowOffset);
            const int64_t tokenStart = ctx.chunkStart + desc.rowOffset;
            for (int64_t i = 0; i < rows * p.dp; ++i) {
                ctx.wsK8[tokenStart * p.dp + i] = static_cast<int8_t>(ctx.kOut[slot][i]);
                ctx.wsV8[tokenStart * p.dp + i] = static_cast<int8_t>(ctx.vOut[slot][i]);
            }
            for (int64_t i = 0; i < rows * p.segWidth; ++i) {
                ctx.wsKScale[tokenStart * p.segWidth + i] = ctx.kScaleOut[slot][i];
            }
            // vScale：逐步 nN 窗序列化（VScalePairWordOff 唯一算式，与设备 Stage3 同式）
            const int64_t pairIdx = tokenStart / (kMxGroupTokens * kMxScalePack);
            for (int64_t c = 0; c < p.dp; ++c) {
                const uint16_t packedU = static_cast<uint16_t>(ctx.expSub[slot][0][c]) |
                                         static_cast<uint16_t>(static_cast<uint16_t>(ctx.expSub[slot][1][c]) << 8);
                ctx.wsVScale[VScalePairWordOff(pairIdx, c, p.dp)] = static_cast<int16_t>(packedU);
            }
        };

        ExecuteQuantBatchPipeline(chunkLen, p.d, p.dp, loadFn, computeFn, storeFn);
    }

    // ---- 标量 oracle（v31 前形态逐语句保留；conformance 三方对账基准腿）----
    inline void ProcessKQuantScalarRef(
        const Dtype *gmK, const float *wsKMean,
        int8_t *wsK8, half *wsKScale, const Sa2Params &p)
    {
        constexpr bool isThreadGran = (G == Gran::kPerThread);
        const float kFactor = p.kFactor;
        float rowDiff[kMaxHeadDim] = {};

        for (int64_t s = 0; s < p.kp; ++s) {
            bool validToken = (s < p.sk);
            if (validToken) {
                const Dtype *rowPtr = gmK + s * p.kStride[2];
                for (int64_t c = 0; c < p.d; ++c) {
                    rowDiff[c] = static_cast<float>(rowPtr[c]) - wsKMean[c];
                }
            }

            if constexpr (!isThreadGran) {
                // per_warp: 每 token 对应单个 scale（行内 amax/127）
                float amax = 0.0f;
                if (validToken) {
                    for (int64_t c = 0; c < p.d; ++c) {
                        float absVal = std::abs(rowDiff[c]);
                        if (absVal > amax) {
                            amax = absVal;
                        }
                    }
                }
                float scale = amax / 127.0f;
                if (scale < 1e-7f) {
                    scale = 1.0f;
                }
                float invScale = 1.0f / scale;

                for (int64_t c = 0; c < p.dp; ++c) {
                    if (validToken && c < p.d) {
                        float scaled = aicNearbyInt(rowDiff[c] * invScale);
                        float clamped = aicMax(-127.0f, aicMin(127.0f, scaled));
                        wsK8[s * p.dp + c] = static_cast<int8_t>(clamped);
                    } else {
                        wsK8[s * p.dp + c] = 0;
                    }
                }
                wsKScale[s] = static_cast<half>(scale * kFactor);
            } else {
                // per_thread: D 维按 kSegCols=16 分段，每段一个 scale
                const int64_t numSegs = p.dp / kSegCols;
                for (int64_t seg = 0; seg < numSegs; ++seg) {
                    float amax = 0.0f;
                    int64_t segStart = seg * kSegCols;
                    int64_t segEnd = aicMin(segStart + kSegCols, p.d);

                    if (validToken && segStart < p.d) {
                        for (int64_t c = segStart; c < segEnd; ++c) {
                            float absVal = std::abs(rowDiff[c]);
                            if (absVal > amax) {
                                amax = absVal;
                            }
                        }
                    }
                    float scale = amax / 127.0f;
                    if (scale < 1e-7f) {
                        scale = 1.0f;
                    }
                    float invScale = 1.0f / scale;

                    for (int64_t c = segStart; c < segStart + kSegCols; ++c) {
                        if (validToken && c < p.d) {
                            float scaled = aicNearbyInt(rowDiff[c] * invScale);
                            float clamped = aicMax(-127.0f, aicMin(127.0f, scaled));
                            wsK8[s * p.dp + c] = static_cast<int8_t>(clamped);
                        } else {
                            wsK8[s * p.dp + c] = 0;
                        }
                    }
                    wsKScale[s * p.segWidth + seg] = static_cast<half>(scale * kFactor);
                }
            }
        }
    }

    // ---- V 量化标量 oracle（v31 前形态逐语句保留，含内联 V 均值）----
    inline void ProcessVQuantScalarRef(
        const Dtype *gmV, float *wsVMean,
        int8_t *wsV8, int16_t *wsVScale, const Sa2Params &p)
    {
        const bool smoothVActive = p.smoothV && (p.accum == static_cast<int64_t>(PvAccum::kFp32));
        float sumVBuf[kMaxHeadDim] = {0.0f};
        if (smoothVActive && p.sk > 0) {
            for (int64_t s = 0; s < p.sk; ++s) {
                const Dtype *row = gmV + s * p.vStride[2];
                for (int64_t c = 0; c < p.d; ++c) {
                    sumVBuf[c] += static_cast<float>(row[c]);
                }
            }
            float invSk = 1.0f / static_cast<float>(p.sk);
            for (int64_t c = 0; c < p.d; ++c) {
                wsVMean[c] = sumVBuf[c] * invSk;
            }
        } else {
            for (int64_t c = 0; c < p.d; ++c) {
                wsVMean[c] = 0.0f;
            }
        }
        for (int64_t c = p.d; c < p.dp; ++c) {
            wsVMean[c] = 0.0f;
        }

        const int64_t numGroups = p.kp / kMxGroupTokens;
        const int64_t numPairs = numGroups / kMxScalePack;

        struct VQuantScratch {
            float vChunk[kMxGroupTokens][kMaxHeadDim];
            uint8_t expSub[kMxScalePack][kMaxHeadDim];
        };
        static VQuantScratch s_scratch;
        VQuantScratch *scratch = &s_scratch;

        auto &vChunk = scratch->vChunk;
        auto &expSub = scratch->expSub;

        for (int64_t pairIdx = 0; pairIdx < numPairs; ++pairIdx) {
            for (int64_t sub = 0; sub < kMxScalePack; ++sub) {
                int64_t g = pairIdx * kMxScalePack + sub;
                int64_t tStart = g * kMxGroupTokens;

                for (int64_t i = 0; i < kMxGroupTokens; ++i) {
                    int64_t t = tStart + i;
                    if (t < p.sk) {
                        const Dtype *row = gmV + t * p.vStride[2];
                        for (int64_t c = 0; c < p.d; ++c) {
                            vChunk[i][c] = static_cast<float>(row[c]) - wsVMean[c];
                        }
                    } else {
                        for (int64_t c = 0; c < p.d; ++c) {
                            vChunk[i][c] = 0.0f;
                        }
                    }
                }

                for (int64_t c = 0; c < p.dp; ++c) {
                    float amax = 0.0f;
                    if (c < p.d) {
                        for (int64_t i = 0; i < kMxGroupTokens; ++i) {
                            float absVal = std::abs(vChunk[i][c]);
                            if (absVal > amax) amax = absVal;
                        }
                    }

                    uint8_t uexp = ComputeE8m0Exponent(amax);
                    expSub[sub][c] = uexp;

                    const int32_t expVal = static_cast<int32_t>(uexp) - 127;
                    for (int64_t i = 0; i < kMxGroupTokens; ++i) {
                        int64_t t = tStart + i;
                        if (t < p.sk && c < p.d) {
                            uint8_t fp8Code = FloatToFp8E4M3(AmlaScaleMultiply(vChunk[i][c], -expVal));
                            wsV8[t * p.dp + c] = static_cast<int8_t>(fp8Code);
                        } else {
                            wsV8[t * p.dp + c] = 0;
                        }
                    }
                }
            }

            // v31 迭代 2 布局契约闭合：与设备写像同构（逐步 nN 窗序列化，VScalePairWordOff 唯一算式）。
            for (int64_t c = 0; c < p.dp; ++c) {
                int16_t packed = static_cast<int16_t>(expSub[0][c] | (static_cast<uint16_t>(expSub[1][c]) << 8));
                wsVScale[VScalePairWordOff(pairIdx, c, p.dp)] = packed;
            }
        }
    }

private:
    // Stage2 镜像（K）：中心化 → 行/段 amax → Sa2QuantScaleReciprocal 一次倒数 →
    // 倒数向量乘（标量镜像为逐元素乘同一 invScale）→ aicNearbyInt（≡ 设备 CAST_RINT）→ 钳制。
    template <typename Ctx>
    inline void SimKQuantCompute(Ctx &ctx, int64_t slot, int64_t rows, int64_t tokenStart)
    {
        const Sa2Params &p = *ctx.p;
        constexpr bool isThreadGran = (G == Gran::kPerThread);
        const float kFactor = p.kFactor;

        for (int64_t r = 0; r < rows; ++r) {
            const int64_t token = tokenStart + r;
            const bool validToken = (token < p.sk);
            const float *rowIn = ctx.kIn[slot] + r * p.dp;
            uint8_t *rowOut = ctx.kOut[slot] + r * p.dp;
            float rowDiff[kMaxHeadDim] = {0.0f};
            for (int64_t c = 0; c < p.d; ++c) {
                rowDiff[c] = rowIn[c] - ctx.wsKMean[c];
            }
            if constexpr (!isThreadGran) {
                float amax = 0.0f;
                for (int64_t c = 0; c < p.d; ++c) {
                    amax = aicMax(amax, aicAbs(rowDiff[c]));
                }
                const Sa2QuantScaleReciprocal kRecip(amax, kFactor);
                for (int64_t c = 0; c < p.dp; ++c) {
                    rowOut[c] = (validToken && c < p.d)
                                    ? static_cast<uint8_t>(kRecip.Quantize(rowDiff[c]))
                                    : 0;
                }
                ctx.kScaleOut[slot][r * p.segWidth] = Sa2ScalarFromFloat<half>(kRecip.scaledFactor);
            } else {
                const int64_t numSegs = p.dp / kSegCols;
                for (int64_t seg = 0; seg < numSegs; ++seg) {
                    const int64_t segBase = seg * kSegCols;
                    const int64_t segEnd = aicMin(segBase + kSegCols, p.d);
                    float amax = 0.0f;
                    for (int64_t c = segBase; c < segEnd; ++c) {
                        amax = aicMax(amax, aicAbs(rowDiff[c]));
                    }
                    const Sa2QuantScaleReciprocal kRecip(amax, kFactor);
                    for (int64_t c = segBase; c < segBase + kSegCols; ++c) {
                        rowOut[c] = (validToken && c < p.d)
                                        ? static_cast<uint8_t>(kRecip.Quantize(rowDiff[c]))
                                        : 0;
                    }
                    ctx.kScaleOut[slot][r * p.segWidth + seg] = Sa2ScalarFromFloat<half>(kRecip.scaledFactor);
                }
            }
        }
    }

    // Stage2 镜像（V）：组 per-channel amax → E8M0 位技巧 → 2^-e 精确二次幂倒数行 →
    // 逐元素倒数乘 → FloatToFp8E4M3 铸码（与设备 vapi 序列逐位同源）。
    template <typename Ctx>
    inline void SimVQuantCompute(Ctx &ctx, int64_t slot, int64_t rows, int64_t tokenStart)
    {
        const Sa2Params &p = *ctx.p;
        const int64_t pairIdx = tokenStart / (kMxGroupTokens * kMxScalePack);
        const int64_t chunkEnd = ctx.chunkStart + ctx.chunkLen;

        for (int64_t sub = 0; sub < kMxScalePack; ++sub) {
            const int64_t g = pairIdx * kMxScalePack + sub;
            const int64_t tStart = g * kMxGroupTokens;
            if (tStart >= chunkEnd) {
                break;  // 组越出分块（与设备同款防御分支；现网网格恒 128 不可达）
            }
            float amaxRow[kMaxHeadDim] = {0.0f};
            float scaleRow[kMaxHeadDim] = {0.0f};
            for (int64_t i = 0; i < kMxGroupTokens; ++i) {
                const int64_t token = tStart + i;
                if (token >= p.sk) {
                    continue;  // 无效行零贡献（与设备纵向 Max 阵列 skip 一致）
                }
                const float *row = ctx.vIn[slot] + (tStart - tokenStart + i) * p.dp;
                for (int64_t c = 0; c < p.d; ++c) {
                    amaxRow[c] = aicMax(amaxRow[c], aicAbs(row[c] - ctx.wsVMean[c]));
                }
            }
            for (int64_t c = 0; c < p.dp; ++c) {
                const uint8_t uexp = ComputeE8m0Exponent(amaxRow[c]);
                ctx.expSub[slot][sub][c] = uexp;
                scaleRow[c] = AmlaScaleMultiply(1.0f, -(static_cast<int32_t>(uexp) - 127));
            }
            for (int64_t i = 0; i < kMxGroupTokens; ++i) {
                const int64_t token = tStart + i;
                const int64_t rowLocal = tStart - tokenStart + i;
                if (rowLocal >= rows) {
                    break;
                }
                uint8_t *rowOut = ctx.vOut[slot] + rowLocal * p.dp;
                if (token >= p.sk) {
                    for (int64_t c = 0; c < p.dp; ++c) {
                        rowOut[c] = 0;
                    }
                    continue;
                }
                const float *row = ctx.vIn[slot] + rowLocal * p.dp;
                for (int64_t c = 0; c < p.dp; ++c) {
                    const float centered = row[c] - ctx.wsVMean[c];
                    rowOut[c] = FloatToFp8E4M3(centered * scaleRow[c]);
                }
            }
        }
    }

};

} // namespace sim
} // namespace sa2

#endif // SA2_SIM_KV_PREP_H_
