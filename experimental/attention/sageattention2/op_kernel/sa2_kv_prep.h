// sa2_kv_prep.h — SageAttention2 KV 预处理（K 均值 L2-replay + INT8 K 量化 + MX FP8 V 量化）。
#if !defined(__AICORE__) && !defined(__CCE_KT_TEST__)
#error "op_kernel/ is strictly reserved for Ascend NPU device compilation (__AICORE__ or __CCE_KT_TEST__)! Host simulation must use sim/sa2_sim_engine.h instead."
#endif

#ifndef SA2_KV_PREP_H_
#define SA2_KV_PREP_H_

#include <cstdint>
#include <cstring>

#include "sa2_spec.h"
#include "sa2_planner.h"
#include "sa2_ub_plan.h"

#if __has_include("kernel_operator.h")
#include "kernel_operator.h"
#elif __has_include("sa2_ascendc_hoststub.h")
#include "sa2_ascendc_hoststub.h"
#endif

#ifndef __aicore__
#define __aicore__
#endif
#ifndef __gm__
#define __gm__
#endif
#ifndef GM_ADDR
using GM_ADDR = uint8_t*;
#endif

namespace sa2 {

// M2c-A5 Stream-K K 均值统计分块常量与 Sa2StreamKMeanTile 描述符定义于 sa2_spec.h

template <typename Dtype, Gran G>
class Sa2KvPreprocessor {
public:
    __aicore__ inline Sa2KvPreprocessor() = default;

    /**
     * @brief 对指定 KV head 执行完整的 K/V 预处理（均值统计与量化写入）。
     */
    __aicore__ inline void ProcessHead(
        GM_ADDR kGmBase, GM_ADDR vGmBase, GM_ADDR userWs,
        const Sa2TilingData &td, int64_t b, int64_t kh)
    {
        ProcessHeadDevice(kGmBase, vGmBase, userWs, td, b, kh);
    }

    __aicore__ inline void ProcessHeadDevice(
        GM_ADDR kGmBase, GM_ADDR vGmBase, GM_ADDR userWs,
        const Sa2TilingData &td, int64_t b, int64_t kh)
    {
        ProcessHeadMeans(kGmBase, vGmBase, userWs, td, b, kh);
        ProcessHeadChunk(kGmBase, vGmBase, userWs, td, b, kh, 0, td.params.kp);
    }

    /**
     * @brief 对指定 KV head 执行 K/V 均值预统计（Stage 1 Pass 1）。
     */
    __aicore__ inline void ProcessHeadMeans(
        GM_ADDR kGmBase, GM_ADDR vGmBase, GM_ADDR userWs,
        const Sa2TilingData &td, int64_t b, int64_t kh)
    {
        const Sa2Params &p = td.params;
        const Sa2KvHeadView view = KvHeadView(p, b, kh);

        __gm__ const Dtype *gmK = reinterpret_cast<__gm__ const Dtype *>(kGmBase) + view.gmK;
        __gm__ const Dtype *gmV = reinterpret_cast<__gm__ const Dtype *>(vGmBase) + view.gmV;

        __gm__ float *wsKMean = reinterpret_cast<__gm__ float *>(userWs + td.kMean + view.wsKMean);
        __gm__ float *wsVMean = reinterpret_cast<__gm__ float *>(userWs + td.vMean + view.wsVMean);

        // 1. K 均值统计（smooth_k=True 时）
        ProcessKMeanDevice(gmK, wsKMean, p);

        // 2. V 均值统计（smooth_v=True 且 fp32 累加时）
        ProcessVMeanDevice(gmV, wsVMean, p);
    }

    /**
     * @brief 对指定 KV head 的特定 Token 分块执行 K/V 量化（Stage 1 Pass 2，支持全 AIV 核平铺）。
     */
    __aicore__ inline void ProcessHeadChunk(
        GM_ADDR kGmBase, GM_ADDR vGmBase, GM_ADDR userWs,
        const Sa2TilingData &td, int64_t b, int64_t kh,
        int64_t chunkStart, int64_t chunkLen)
    {
        const Sa2Params &p = td.params;
        const Sa2KvHeadView view = KvHeadView(p, b, kh);

        __gm__ const Dtype *gmK = reinterpret_cast<__gm__ const Dtype *>(kGmBase) + view.gmK;
        __gm__ const Dtype *gmV = reinterpret_cast<__gm__ const Dtype *>(vGmBase) + view.gmV;

        __gm__ const float *wsKMean = reinterpret_cast<__gm__ const float *>(userWs + td.kMean + view.wsKMean);
        __gm__ const float *wsVMean = reinterpret_cast<__gm__ const float *>(userWs + td.vMean + view.wsVMean);
        __gm__ int8_t *wsK8 = reinterpret_cast<__gm__ int8_t *>(userWs + td.k8 + view.wsK8);
        __gm__ half *wsKScale = reinterpret_cast<__gm__ half *>(userWs + td.kScale + view.wsKScale);
        __gm__ int8_t *wsV8 = reinterpret_cast<__gm__ int8_t *>(userWs + td.v8 + view.wsV8);
        __gm__ int16_t *wsVScale = reinterpret_cast<__gm__ int16_t *>(userWs + td.vScale + view.wsVScale);

        // K 中心化与 INT8 对称量化（当前分块）
        ProcessKQuantChunkDevice(gmK, wsKMean, wsK8, wsKScale, p, chunkStart, chunkLen);

        // V MX 分组量化与 E8M0 双打包（当前分块）
        ProcessVQuantChunkDevice(gmV, wsVMean, wsV8, wsVScale, p, chunkStart, chunkLen);
    }

private:
    // 【迭代 9 算法甄别：K 均值向量化实现路径（D-R8 对照）】向量化层（M2b/M2.6）采用
    // 「AIV 内 datablock 归约 + head 单写者」，否决「跨核单字归约协议」。依据：
    //   1) 无原子：本 head 累加只发生在一个 AIV 内（sa2_kernel.cpp kvIdx % totalAivs 单写者
    //      分配），零原子操作；跨核单字归约需 GM 原子加（浮点原子加序不定，破坏 §4.7
    //      「与全量两遍逐位一致」契约）或额外部分和交换旗号（MODE4 预算 12/16 已冻结，无余量）；
    //   2) 确定性加法序：datablock 归约树（block 内 lane 序 + block 间行序）完全静态可复现，
    //      可与 conformance 位一致断言兼容；跨核合并序随核间到达序波动，不可冻结；
    //   3) GM 读 1×：pass1(本函数)与 pass2(ProcessKQuantChunkDevice) 同 AIV 背靠背（L2-replay 融合，
    //      分块网格结构不动），pass2 命中 L2 ⇒ GM 有效读 1×；跨核归约需部分和 GM 往返，
    //      与 D-R2/D-R13 零 GM 驻留原则冲突。
    // 义务注记（v31 迭代 1 收口）：设备标量载体现为 128 行分块 × 64 行子块的两级树
    //（ProcessKMeanStreamKDevice），FP32 加法非结合 ⇒ sim oracle（sa2_sim_kv_prep.h
    // ProcessKMean）已在本轮改为语句序同构的分块树镜像，「宿主 oracle 逐位镜像」契约恢复；
    // 向量化落 64-lane datablock 树时，同构义务延续：oracle 与设备须同一提交内换树。

    /**
     * @brief M2c-A5 K 均值 128 行分块统计（D-R8 形态；跨界汇聚分支 = Stream-K 预留，不可达）
     * 1. 完整 head 独写：单写者持有完整 head，直接执行分块行累加与倒数乘写回，零原子操作；
     * 2. 跨界 head 汇聚：为 Stream-K 跨核分块预留的接口形态（当前唯一调用 = 全量区间独写）；
     * 3. 加法树 = 128 行块 → 64 行子块 → 全头折叠，与 sim oracle 同构（见义务注记）。
     */
    __aicore__ inline void ProcessKMeanStreamKDevice(
        __gm__ const Dtype *gmK, __gm__ float *wsKMean, const Sa2Params &p,
        int64_t rangeStart = 0, int64_t rangeEnd = -1, bool isExclusiveHead = true)
    {
        if (!p.smoothK || p.sk <= 0) {
            for (int64_t c = 0; c < p.dp; ++c) {
                wsKMean[c] = 0.0f;
            }
            return;
        }

        const int64_t actualStart = rangeStart;
        const int64_t actualEnd = (rangeEnd >= 0) ? aicMin(rangeEnd, p.sk) : p.sk;
        if (actualStart >= actualEnd) {
            return;
        }

        // 总累加和缓冲区（D 维）
        float totalSum[kMaxHeadDim] = {0.0f};

        // 双缓冲槽位：ping (0) 与 pong (1)，每个承载 128 行分块内的累加中间和
        float chunkBuf[kKMeanDoubleBufSlots][kMaxHeadDim] = {};

        int64_t currentSlot = 0;
        for (int64_t sBlock = actualStart; sBlock < actualEnd; sBlock += kKMeanBlockRows) {
            const int64_t blockEnd = aicMin(sBlock + kKMeanBlockRows, actualEnd);

            // 清零当前双缓冲槽位
            for (int64_t c = 0; c < p.d; ++c) {
                chunkBuf[currentSlot][c] = 0.0f;
            }

            // D-R8：在 128 行分块内，按 64 行子块（datablock）分段累加，利于向量化归约树展开
            for (int64_t subBlock = sBlock; subBlock < blockEnd; subBlock += kKMeanDataBlockLanes) {
                const int64_t subEnd = aicMin(subBlock + kKMeanDataBlockLanes, blockEnd);
                for (int64_t s = subBlock; s < subEnd; ++s) {
                    __gm__ const Dtype *row = gmK + s * p.kStride[2];
                    for (int64_t c = 0; c < p.d; ++c) {
                        chunkBuf[currentSlot][c] += Sa2ScalarToFloat(row[c]);
                    }
                }
            }

            // 将当前双缓冲槽位的块和累入总累加和（流水化汇聚）
            for (int64_t c = 0; c < p.d; ++c) {
                totalSum[c] += chunkBuf[currentSlot][c];
            }

            // 切换双缓冲槽位 (ping <-> pong)
            currentSlot = 1 - currentSlot;
        }

        // 完整 head 独写 vs 跨界 head 汇聚
        if (isExclusiveHead && actualStart == 0 && actualEnd == p.sk) {
            // 完整 head 独写模式（零原子操作，符合 D-R8 甄别结论）
            const float invSk = 1.0f / static_cast<float>(p.sk);
            for (int64_t c = 0; c < p.d; ++c) {
                wsKMean[c] = totalSum[c] * invSk;
            }
            for (int64_t c = p.d; c < p.dp; ++c) {
                wsKMean[c] = 0.0f;
            }
        } else {
            // 跨界 head 汇聚模式：针对切分片段进行分块结果写入或累积
            if (actualStart == 0) {
                for (int64_t c = 0; c < p.d; ++c) {
                    wsKMean[c] = totalSum[c];
                }
                for (int64_t c = p.d; c < p.dp; ++c) {
                    wsKMean[c] = 0.0f;
                }
            } else {
                for (int64_t c = 0; c < p.d; ++c) {
                    wsKMean[c] += totalSum[c];
                }
            }
            if (actualEnd == p.sk) {
                const float invSk = 1.0f / static_cast<float>(p.sk);
                for (int64_t c = 0; c < p.d; ++c) {
                    wsKMean[c] *= invSk;
                }
            }
        }
    }

    __aicore__ inline void ProcessKMeanDevice(
        __gm__ const Dtype *gmK, __gm__ float *wsKMean, const Sa2Params &p)
    {
        ProcessKMeanStreamKDevice(gmK, wsKMean, p, 0, p.sk, true);
    }

    __aicore__ inline void ProcessVMeanDevice(
        __gm__ const Dtype *gmV, __gm__ float *wsVMean, const Sa2Params &p)
    {
        const bool smoothVActive = p.smoothV && (p.accum == static_cast<int64_t>(PvAccum::kFp32));
        float sumVBuf[kMaxHeadDim] = {0.0f};
        if (smoothVActive && p.sk > 0) {
            for (int64_t s = 0; s < p.sk; ++s) {
                __gm__ const Dtype *row = gmV + s * p.vStride[2];
                for (int64_t c = 0; c < p.d; ++c) {
                    sumVBuf[c] += Sa2ScalarToFloat(row[c]);
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

    __aicore__ inline void ProcessKQuantChunkDevice(
        __gm__ const Dtype *gmK, __gm__ const float *wsKMean,
        __gm__ int8_t *wsK8, __gm__ half *wsKScale, const Sa2Params &p,
        int64_t chunkStart, int64_t chunkLen)
    {
        constexpr bool isThreadGran = (G == Gran::kPerThread);
        const float kFactor = p.kFactor;
        float rowDiff[kMaxHeadDim] = {};

        const int64_t chunkEnd = aicMin(chunkStart + chunkLen, p.kp);
        for (int64_t s = chunkStart; s < chunkEnd; ++s) {
            bool validToken = (s < p.sk);
            if (validToken) {
                __gm__ const Dtype *rowPtr = gmK + s * p.kStride[2];
                for (int64_t c = 0; c < p.d; ++c) {
                    rowDiff[c] = Sa2ScalarToFloat(rowPtr[c]) - wsKMean[c];
                }
            }

            if constexpr (!isThreadGran) {
                // per_warp: 每 token 对应单个 scale（行内 amax/127）
                float amax = 0.0f;
                if (validToken) {
                    for (int64_t c = 0; c < p.d; ++c) {
                        float absVal = aicAbs(rowDiff[c]);
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
                wsKScale[s] = Sa2ScalarFromFloat<half>(scale * kFactor);
            } else {
                // per_thread: D 维按 kSegCols=16 分段，每段一个 scale
                const int64_t numSegs = p.dp / kSegCols;
                for (int64_t seg = 0; seg < numSegs; ++seg) {
                    float amax = 0.0f;
                    int64_t segStart = seg * kSegCols;
                    int64_t segEnd = aicMin(segStart + kSegCols, p.d);

                    if (validToken && segStart < p.d) {
                        for (int64_t c = segStart; c < segEnd; ++c) {
                            float absVal = aicAbs(rowDiff[c]);
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
                    wsKScale[s * p.segWidth + seg] = Sa2ScalarFromFloat<half>(scale * kFactor);
                }
            }
        }
    }

    // v31 迭代 1 裁决（P2-1 同款纪律）：零调用点整头包装 ProcessKQuantDevice /
    // ProcessVQuantDevice 删除——Stage 1 入口恒经 ProcessHeadDevice（Pass1 均值 +
    // Pass2 分块量化，分块粒度由 sa2_kernel.cpp 网格给定）；重引入条件 = 量化阶段
    // 需要绕过分块网格的整头单核路径（如单核诊断模式）时按需恢复。

    __aicore__ inline void ProcessVQuantChunkDevice(
        __gm__ const Dtype *gmV, __gm__ const float *wsVMean,
        __gm__ int8_t *wsV8, __gm__ int16_t *wsVScale, const Sa2Params &p,
        int64_t chunkStart, int64_t chunkLen)
    {
        const int64_t numGroups = p.kp / kMxGroupTokens;
        const int64_t numPairs = numGroups / kMxScalePack;

        const int64_t pairStart = (chunkStart / kMxGroupTokens) / kMxScalePack;
        const int64_t numPairsInChunk = (chunkLen / kMxGroupTokens) / kMxScalePack;
        const int64_t pairEnd = aicMin(pairStart + numPairsInChunk, numPairs);

        uint8_t expSub[kMxScalePack][kMaxHeadDim] = {};
        for (int64_t pairIdx = pairStart; pairIdx < pairEnd; ++pairIdx) {
            for (int64_t sub = 0; sub < kMxScalePack; ++sub) {
                int64_t g = pairIdx * kMxScalePack + sub;
                int64_t tStart = g * kMxGroupTokens;

                // pass1：流式 amax（per-channel）
                for (int64_t c = 0; c < p.dp; ++c) {
                    float amax = 0.0f;
                    if (c < p.d) {
                        for (int64_t i = 0; i < kMxGroupTokens; ++i) {
                            int64_t t = tStart + i;
                            if (t < p.sk) {
                                __gm__ const Dtype *row = gmV + t * p.vStride[2];
                                float absVal = aicAbs(Sa2ScalarToFloat(row[c]) - wsVMean[c]);
                                if (absVal > amax) amax = absVal;
                            }
                        }
                    }

                    // P1-6：E8M0 向上取幂位技巧 + 幂边界饱和兜底
                    uint8_t uexp = ComputeE8m0Exponent(amax);
                    expSub[sub][c] = uexp;

                    // pass2：重读定码字
                    const int32_t expVal = static_cast<int32_t>(uexp) - 127;
                    for (int64_t i = 0; i < kMxGroupTokens; ++i) {
                        int64_t t = tStart + i;
                        if (t < p.sk && c < p.d) {
                            __gm__ const Dtype *row = gmV + t * p.vStride[2];
                            float centered = Sa2ScalarToFloat(row[c]) - wsVMean[c];
                            uint8_t fp8Code = FloatToFp8E4M3(AmlaScaleMultiply(centered, -expVal));
                            wsV8[t * p.dp + c] = static_cast<int8_t>(fp8Code);
                        } else {
                            wsV8[t * p.dp + c] = 0;
                        }
                    }
                }
            }

            // v31 迭代 2 布局契约闭合：写像 = 逐步 nN 窗序列化（VScalePairWordOff 唯一算式），
            // 与 attention_loop 消费窗（tla nN MX 布局平面化字节像）逐字节一致；
            // 旧「通道主序 [c][pair]」写像在多步窗下与消费窗错位，已废弃。
            for (int64_t c = 0; c < p.dp; ++c) {
                uint16_t packedU = static_cast<uint16_t>(expSub[0][c]) | (static_cast<uint16_t>(expSub[1][c]) << 8);
                wsVScale[VScalePairWordOff(pairIdx, c, p.dp)] = static_cast<int16_t>(packedU);
            }
        }
    }

};

} // namespace sa2

#endif // SA2_KV_PREP_H_
