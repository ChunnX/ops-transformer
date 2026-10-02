// sa2_sim_kv_prep.h — SageAttention2 宿主仿真 KV 预处理（纯标准 C++17，零硬件依赖）
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
     * @brief 宿主机纯标量仿真：对指定 KV head 执行完整的 K/V 预处理（均值统计与量化写入）。
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

        // 1. K 均值统计（smooth_k=True 时）
        ProcessKMean(gmK, wsKMean, p);

        // 2. K 中心化与 INT8 对称量化
        ProcessKQuant(gmK, wsKMean, wsK8, wsKScale, p);

        // 3. V MX 分组量化与 E8M0 双打包（D-R4 口径）
        ProcessVQuant(gmV, wsVMean, wsV8, wsVScale, p);
    }

private:
    inline void ProcessKMean(
        const Dtype *gmK, float *wsKMean, const Sa2Params &p)
    {
        float sumBuf[kMaxHeadDim] = {0.0f};
        if (p.smoothK && p.sk > 0) {
            for (int64_t s = 0; s < p.sk; ++s) {
                const Dtype *row = gmK + s * p.kStride[2];
                for (int64_t c = 0; c < p.d; ++c) {
                    sumBuf[c] += static_cast<float>(row[c]);
                }
            }
            float invSk = 1.0f / static_cast<float>(p.sk);
            for (int64_t c = 0; c < p.d; ++c) {
                wsKMean[c] = sumBuf[c] * invSk;
            }
        } else {
            for (int64_t c = 0; c < p.d; ++c) {
                wsKMean[c] = 0.0f;
            }
        }
        for (int64_t c = p.d; c < p.dp; ++c) {
            wsKMean[c] = 0.0f;
        }
    }

    inline void ProcessKQuant(
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

    inline void ProcessVQuant(
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

            for (int64_t c = 0; c < p.dp; ++c) {
                int16_t packed = static_cast<int16_t>(expSub[0][c] | (static_cast<uint16_t>(expSub[1][c]) << 8));
                wsVScale[c * numPairs + pairIdx] = packed;
            }
        }
    }
};

} // namespace sim
} // namespace sa2

#endif // SA2_SIM_KV_PREP_H_
