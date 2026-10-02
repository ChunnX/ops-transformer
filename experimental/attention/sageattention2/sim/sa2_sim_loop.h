// sa2_sim_loop.h — SageAttention2 宿主仿真注意力流水线（纯标准 C++17，零硬件依赖）
#ifndef SA2_SIM_LOOP_H_
#define SA2_SIM_LOOP_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "sa2_spec.h"
#include "sa2_planner.h"
#include "sa2_ub_plan.h"
#include "sa2_events.h"

namespace sa2 {
namespace sim {

struct TileFusedReductionsSim {
    template <int64_t Cols>
    static inline float TileRowMax(const float *rowPtr) {
        float maxVal = -INFINITY;
        for (int64_t c = 0; c < Cols; ++c) {
            if (rowPtr[c] > maxVal) {
                maxVal = rowPtr[c];
            }
        }
        return maxVal;
    }

    template <int64_t Cols>
    static inline void TileRowExpandSubExpFp8(
        const float *rowS, float mNew,
        uint8_t *pSlotUbRow, uint8_t *pSlotL1Row, float &sumP8Out)
    {
        double wideSum = 0.0;
        double compensation = 0.0;

        for (int64_t col = 0; col < Cols; ++col) {
            float pVal = std::exp(rowS[col] - mNew);
            float pClamped = std::max(0.0f, std::min(448.0f, pVal));
            uint8_t fp8Code = FloatToFp8E4M3(pClamped);
            float pDiscretized = Fp8E4M3ToFloat(fp8Code);
            pSlotUbRow[col] = fp8Code;
            pSlotL1Row[col] = fp8Code;

            double y = static_cast<double>(pDiscretized) - compensation;
            double t = wideSum + y;
            compensation = (t - wideSum) - y;
            wideSum = t;
        }
        sumP8Out = static_cast<float>(wideSum);
    }
};

template <typename Dtype, Gran G, bool IsCausal>
class Sa2AttentionLoopSim {
public:
    inline Sa2AttentionLoopSim()
    {
        ub_.SetBase(ubMem_.data());
        l1_.SetBase(l1Mem_.data());
        l0c_.SetBase(l0cMem_.data());
    }

    inline void Init(
        uint8_t *qGm, uint8_t *outGm, uint8_t *lseGm,
        uint8_t *userWs, const Sa2TilingData &td)
    {
        qGmBase_ = qGm;
        outGmBase_ = outGm;
        lseGmBase_ = lseGm;
        userWs_ = userWs;
        td_ = &td;
    }

    inline void ProcessTile(const Sa2Task &task, int64_t streamId = 0)
    {
        if (!task.valid || task.steps <= 0) {
            return;
        }
        StepVectorQQuant(task, streamId);
        // P1-2（迭代 2）：与设备 ProcessTileVector（sa2_attention_loop.h）同构的配对 PV
        // 折叠调度——step+1 < steps 时相邻两步成对（softmax×2 → 配对折叠）、奇数尾单步。
        // 配对序 o·a1 → o += a1·f16(pv0) → o += f16(pv1) 与单步序 (o += f16(pv0))·a1
        // → o += f16(pv1) 的 FP32 舍入点不同（设备侧 Sa2VfFoldRun paired 分支为准），
        // 本镜像保证宿主 oracle 与设备舍入序逐位一致。
        int64_t step = 0;
        for (; step + 1 < task.steps; step += 2) {
            StepCubeQk(task, step, streamId);
            StepVectorSoftmax(task, step, streamId);
            StepCubePv(task, step, streamId);
            StepCubeQk(task, step + 1, streamId);
            StepVectorSoftmax(task, step + 1, streamId);
            StepCubePv(task, step + 1, streamId);
            StepVectorPvFoldPair(task, step, step + 1, streamId);
        }
        for (; step < task.steps; ++step) {
            StepCubeQk(task, step, streamId);
            StepVectorSoftmax(task, step, streamId);
            StepCubePv(task, step, streamId);
            StepVectorPvFold(task, step, streamId);
        }
        FinishTileVector(task, streamId);
        FinishTileCube(task, streamId);
    }

private:
    inline void StepVectorQQuant(const Sa2Task &task, int64_t streamId)
    {
        const Sa2Params &p = td_->params;
        constexpr bool isThreadGran = (G == Gran::kPerThread);
        const int64_t validRows = QTileValidRows(p, task.m0);

        const Sa2KvHeadView kvView = KvHeadView(p, task.b, task.kh);
        const float *wsKMean = reinterpret_cast<const float *>(
            userWs_ + td_->kMean + kvView.wsKMean);
        const int64_t qElemOffset = QTileOffset(p, task.b, task.qh, task.m0);
        const Dtype *qPtr = reinterpret_cast<const Dtype *>(qGmBase_) + qElemOffset;

        float *qFloatBuf = ub_.GetPvSlot(0);
        float *corrSlot = ub_.GetStateCorr(streamId);

        for (int64_t r = 0; r < kStreamRows; ++r) {
            float rowCorr = 0.0f;
            if (r < validRows) {
                for (int64_t c = 0; c < p.dp; ++c) {
                    float qVal = (c < p.d) ? static_cast<float>(qPtr[r * p.qStride[2] + c]) : 0.0f;
                    qFloatBuf[r * p.dp + c] = qVal;
                    rowCorr += qVal * wsKMean[c];
                }
            } else {
                for (int64_t c = 0; c < p.dp; ++c) {
                    qFloatBuf[r * p.dp + c] = 0.0f;
                }
            }
            corrSlot[r] = rowCorr;
        }

        int8_t *q8L1 = l1_.GetQ8Slot(streamId);
        half *qScaleL1 = l1_.GetQScaleSlot(streamId);

        if constexpr (!isThreadGran) {
            for (int64_t r = 0; r < kStreamRows; ++r) {
                float amax = 0.0f;
                if (r < validRows) {
                    for (int64_t c = 0; c < p.d; ++c) {
                        float absVal = std::abs(qFloatBuf[r * p.dp + c]);
                        if (absVal > amax) amax = absVal;
                    }
                }
                float scale = amax / 127.0f;
                if (scale < 1e-7f) scale = 1.0f;
                float invScale = 1.0f / scale;

                for (int64_t c = 0; c < p.dp; ++c) {
                    if (r < validRows && c < p.d) {
                        float scaled = std::nearbyint(qFloatBuf[r * p.dp + c] * invScale);
                        q8L1[r * p.dp + c] = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, scaled)));
                    } else {
                        q8L1[r * p.dp + c] = 0;
                    }
                }
                qScaleL1[r] = half(scale * p.qFactor);
            }
        } else {
            const int64_t numSegs = p.dp / kSegCols;
            for (int64_t r = 0; r < kStreamRows; ++r) {
                for (int64_t seg = 0; seg < numSegs; ++seg) {
                    float amax = 0.0f;
                    int64_t segStart = seg * kSegCols;
                    int64_t segEnd = std::min(segStart + kSegCols, p.d);

                    if (r < validRows && segStart < p.d) {
                        for (int64_t c = segStart; c < segEnd; ++c) {
                            float absVal = std::abs(qFloatBuf[r * p.dp + c]);
                            if (absVal > amax) amax = absVal;
                        }
                    }
                    float scale = amax / 127.0f;
                    if (scale < 1e-7f) scale = 1.0f;
                    float invScale = 1.0f / scale;

                    for (int64_t c = segStart; c < segStart + kSegCols; ++c) {
                        if (r < validRows && c < p.d) {
                            float scaled = std::nearbyint(qFloatBuf[r * p.dp + c] * invScale);
                            q8L1[r * p.dp + c] = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, scaled)));
                        } else {
                            q8L1[r * p.dp + c] = 0;
                        }
                    }
                    qScaleL1[r * numSegs + seg] = half(scale * p.qFactor);
                }
            }
        }

        float *rowStat = ub_.GetStateRowStat(streamId);
        float *denomSlot = ub_.GetStateDenom(streamId);
        for (int64_t r = 0; r < kStreamRows; ++r) {
            rowStat[r * 2 + 0] = kFp16NegMax;
            rowStat[r * 2 + 1] = 0.0f;
            denomSlot[r] = 0.0f;
        }

        float *oAcc = ub_.GetOAcc();
        for (int64_t i = 0; i < kStreamRows * p.dp; ++i) {
            oAcc[i] = 0.0f;
        }
    }

    inline void StepCubeQk(const Sa2Task &task, int64_t step, int64_t streamId)
    {
        int64_t parity = step % 2;
        const Sa2Params &p = td_->params;
        constexpr bool isThreadGran = (G == Gran::kPerThread);

        const Sa2KvHeadView kvView = KvHeadView(p, task.b, task.kh);
        const int8_t *wsK8Base = reinterpret_cast<const int8_t *>(
            userWs_ + td_->k8 + kvView.wsK8);
        const half *wsKScaleBase = reinterpret_cast<const half *>(
            userWs_ + td_->kScale + kvView.wsKScale);

        const int8_t *q8L1 = l1_.GetQ8Slot(streamId);
        const half *qScaleL1 = l1_.GetQScaleSlot(streamId);
        const int8_t *k8Step = wsK8Base + step * kKvStep * p.dp;
        const half *kScaleStep = wsKScaleBase + step * kKvStep * p.segWidth;

        int32_t *l0cQuad = static_cast<int32_t *>(l0c_.GetQuadrant(streamId, parity));
        const int64_t validRows = QTileValidRows(p, task.m0);

        for (int64_t r = 0; r < kStreamRows; ++r) {
            for (int64_t col = 0; col < kKvStep; ++col) {
                l0cQuad[r * kKvStep + col] = 0;
            }
        }

        for (int64_t r = 0; r < validRows; ++r) {
            for (int64_t col = 0; col < kKvStep; ++col) {
                int32_t dot = 0;
                for (int64_t c = 0; c < p.dp; ++c) {
                    dot += static_cast<int32_t>(q8L1[r * p.dp + c]) * static_cast<int32_t>(k8Step[col * p.dp + c]);
                }
                l0cQuad[r * kKvStep + col] = dot;
            }
        }

        if constexpr (!isThreadGran) {
            half *scoreSlotWarp = ub_.GetScoreSlotWarp(parity);
            for (int64_t r = 0; r < validRows; ++r) {
                for (int64_t col = 0; col < kKvStep; ++col) {
                    int32_t dot = l0cQuad[r * kKvStep + col];
                    float deqScore = static_cast<float>(dot) * (1.0f / 16384.0f);
                    scoreSlotWarp[r * kKvStep + col] = half(deqScore);
                }
            }
        } else {
            float *scoreSlotThread = ub_.GetScoreSlotThread(parity);
            const int64_t numSegs = p.dp / kSegCols;
            for (int64_t r = 0; r < validRows; ++r) {
                for (int64_t col = 0; col < kKvStep; ++col) {
                    float sumScore = 0.0f;
                    for (int64_t seg = 0; seg < numSegs; ++seg) {
                        int32_t segDot = 0;
                        int64_t segStart = seg * kSegCols;
                        for (int64_t c = segStart; c < segStart + kSegCols; ++c) {
                            segDot += static_cast<int32_t>(q8L1[r * p.dp + c]) * static_cast<int32_t>(k8Step[col * p.dp + c]);
                        }
                        float deq = static_cast<float>(segDot) * (1.0f / 16384.0f);
                        float qh = static_cast<float>(qScaleL1[r * numSegs + seg]);
                        float kh = static_cast<float>(kScaleStep[col * p.segWidth + seg]);
                        sumScore += deq * qh * kh;
                    }
                    scoreSlotThread[r * kKvStep + col] = sumScore;
                }
            }
        }
    }

    inline void StepVectorSoftmax(const Sa2Task &task, int64_t step, int64_t streamId)
    {
        int64_t parity = step % 2;
        const Sa2Params &p = td_->params;
        constexpr bool isThreadGran = (G == Gran::kPerThread);
        const int64_t validRows = QTileValidRows(p, task.m0);
        const int64_t kvTokenStart = step * kKvStep;
        const float lnPDomain = p.lnPDomain;

        const Sa2KvHeadView kvView = KvHeadView(p, task.b, task.kh);
        const half *wsKScaleBase = reinterpret_cast<const half *>(
            userWs_ + td_->kScale + kvView.wsKScale);
        const half *kScaleStep = wsKScaleBase + step * kKvStep * p.segWidth;
        const half *qScalePtr = l1_.GetQScaleSlot(streamId);

        float *rowStat = ub_.GetStateRowStat(streamId);
        float *denomSlot = ub_.GetStateDenom(streamId);
        float *oAcc = ub_.GetOAcc();
        float *alphaSlot = ub_.GetStateAlpha(streamId, parity);  // P1-2：配对折叠消费 step1 alpha
        uint8_t *pSlotUb = ub_.GetPSlot(parity);
        uint8_t *pSlotL1 = l1_.GetPNzSlot(streamId);

        for (int64_t r = 0; r < validRows; ++r) {
            int64_t qTokenIdx = task.m0 + r;

            float rowS[kKvStep];
            for (int64_t col = 0; col < kKvStep; ++col) {
                int64_t kTokenIdx = kvTokenStart + col;
                if (kTokenIdx >= p.sk || (IsCausal && kTokenIdx > qTokenIdx)) {
                    rowS[col] = -INFINITY;
                    continue;
                }

                if constexpr (!isThreadGran) {
                    float deq = static_cast<float>(ub_.GetScoreSlotWarp(parity)[r * kKvStep + col]);
                    float qh = static_cast<float>(qScalePtr[r]);
                    float kh = static_cast<float>(kScaleStep[col]);
                    rowS[col] = deq * qh * kh;
                } else {
                    rowS[col] = ub_.GetScoreSlotThread(parity)[r * kKvStep + col];
                }
            }

            float mBlk = TileFusedReductionsSim::TileRowMax<kKvStep>(rowS);
            float mOld = rowStat[r * 2 + 0];
            float mNew = std::max(mOld, mBlk - lnPDomain);
            float alpha = std::exp(mOld - mNew);

            bool skipRescale = (mBlk - mOld <= lnPDomain);
            if (skipRescale) {
                alpha = 1.0f;
                mNew = mOld;
            }

            float sumP8 = 0.0f;
            TileFusedReductionsSim::TileRowExpandSubExpFp8<kKvStep>(
                rowS, mNew,
                &pSlotUb[r * kKvStep],
                &pSlotL1[r * kKvStep],
                sumP8);

            denomSlot[r] = alpha * denomSlot[r] + sumP8;
            rowStat[r * 2 + 0] = mNew;
            alphaSlot[r] = alpha;  // 设备 Sa2VfSoftmaxRun alphaSlot 回写同构（配对折叠源）

            if (!skipRescale && alpha != 1.0f) {
                for (int64_t c = 0; c < p.dp; ++c) {
                    oAcc[r * p.dp + c] *= alpha;
                }
            }
        }
    }

    inline void StepCubePv(const Sa2Task &task, int64_t step, int64_t streamId)
    {
        int64_t parity = step % 2;
        const Sa2Params &p = td_->params;
        const Sa2KvHeadView kvView = KvHeadView(p, task.b, task.kh);
        const int8_t *wsV8Base = reinterpret_cast<const int8_t *>(
            userWs_ + td_->v8 + kvView.wsV8);
        const int16_t *wsVScaleBase = reinterpret_cast<const int16_t *>(
            userWs_ + td_->vScale + kvView.wsVScale);
        const int8_t *v8Step = wsV8Base + step * kKvStep * p.dp;

        const uint8_t *p8Ptr = l1_.GetPNzSlot(streamId);
        float *pvSlot = ub_.GetPvSlot(parity);
        const int64_t validRows = QTileValidRows(p, task.m0);
        const int64_t kvTokenStart = step * kKvStep;

        for (int64_t r = 0; r < kStreamRows; ++r) {
            for (int64_t c = 0; c < p.dp; ++c) {
                pvSlot[r * p.dp + c] = 0.0f;
            }
        }

        for (int64_t col = 0; col < kKvStep; ++col) {
            int64_t kTokenIdx = kvTokenStart + col;
            if (kTokenIdx >= p.sk) continue;

            int64_t groupIdx = col / kMxGroupTokens;
            int64_t pairIdx = (step * (kKvStep / kMxGroupTokens) + groupIdx) / kMxScalePack;
            int64_t subIdx = (step * (kKvStep / kMxGroupTokens) + groupIdx) % kMxScalePack;

            for (int64_t c = 0; c < p.dp; ++c) {
                // 消费与设备 StepCubePvDevice 同构：逐步 nN 窗序列化（VScalePairWordOff 唯一算式）。
                int16_t packed = wsVScaleBase[VScalePairWordOff(pairIdx, c, p.dp)];
                uint8_t uexp = (subIdx == 0) ? (packed & 0xFF) : ((packed >> 8) & 0xFF);
                int32_t expVal = static_cast<int32_t>(uexp) - 127;
                float vRaw = Fp8E4M3ToFloat(static_cast<uint8_t>(v8Step[col * p.dp + c]));
                float vDequant = AmlaScaleMultiply(vRaw, expVal);

                for (int64_t r = 0; r < validRows; ++r) {
                    float pDeq = Fp8E4M3ToFloat(p8Ptr[r * kKvStep + col]);
                    pvSlot[r * p.dp + c] += pDeq * vDequant;
                }
            }
        }
    }

    inline void StepVectorPvFold(const Sa2Task &task, int64_t step, int64_t /*streamId*/)
    {
        int64_t parity = step % 2;
        const Sa2Params &p = td_->params;
        const int64_t validRows = QTileValidRows(p, task.m0);
        const float *pvSlot = ub_.GetPvSlot(parity);
        float *oAcc = ub_.GetOAcc();
        const bool isFp16Accum = (p.accum == static_cast<int64_t>(PvAccum::kFp32Fp16));

        for (int64_t r = 0; r < validRows; ++r) {
            for (int64_t c = 0; c < p.dp; ++c) {
                float blockVal = pvSlot[r * p.dp + c];
                if (isFp16Accum) {
                    blockVal = static_cast<float>(half(blockVal));
                }
                oAcc[r * p.dp + c] += blockVal;
            }
        }
    }

    // P1-2（迭代 2）：设备 Sa2VfFoldRun paired 分支的宿主镜像（M2c-A2 配对 PV 聚合）。
    // 语句序与设备逐条对应：f16 往返（fp32+fp16 档）→ step1 alpha 条件乘（位型 ≠ 1.0f
    // 语义 = a1 != 1.0f，全位型等价）→ o += a1·f16(pv0) → o += f16(pv1)。
    inline void StepVectorPvFoldPair(const Sa2Task &task, int64_t step0, int64_t /*step1*/, int64_t streamId)
    {
        const int64_t parity0 = step0 % 2;
        const int64_t parity1 = (step0 + 1) % 2;
        const Sa2Params &p = td_->params;
        const int64_t validRows = QTileValidRows(p, task.m0);
        const float *pvSlot0 = ub_.GetPvSlot(parity0);
        const float *pvSlot1 = ub_.GetPvSlot(parity1);
        const float *alpha1 = ub_.GetStateAlpha(streamId, parity1);
        float *oAcc = ub_.GetOAcc();
        const bool isFp16Accum = (p.accum == static_cast<int64_t>(PvAccum::kFp32Fp16));

        for (int64_t r = 0; r < validRows; ++r) {
            const float a1 = alpha1[r];
            const bool scalePv0 = (a1 != 1.0f);
            for (int64_t c = 0; c < p.dp; ++c) {
                const int64_t k = r * p.dp + c;
                float p0 = pvSlot0[k];
                float p1 = pvSlot1[k];
                if (isFp16Accum) {
                    p0 = static_cast<float>(half(p0));
                    p1 = static_cast<float>(half(p1));
                }
                if (scalePv0) {
                    p0 = p0 * a1;
                }
                oAcc[k] += p0;
                oAcc[k] += p1;
            }
        }
    }

    inline void FinishTileVector(const Sa2Task &task, int64_t streamId)
    {
        const Sa2Params &p = td_->params;
        const Sa2KvHeadView kvView = KvHeadView(p, task.b, task.kh);
        const float *wsVMean = reinterpret_cast<const float *>(
            userWs_ + td_->vMean + kvView.wsVMean);
        const int64_t validRows = QTileValidRows(p, task.m0);
        const bool smoothVActive = p.smoothV && (p.accum == static_cast<int64_t>(PvAccum::kFp32));

        const int64_t outElemOffset = OutTileOffset(p, task.b, task.qh, task.m0);
        Dtype *outPtr = reinterpret_cast<Dtype *>(outGmBase_) + outElemOffset;

        const float *denom = ub_.GetStateDenom(streamId);
        const float *oAcc = ub_.GetOAcc();

        for (int64_t r = 0; r < validRows; ++r) {
            float invDenom = (denom[r] > 0.0f) ? (1.0f / denom[r]) : 0.0f;
            for (int64_t c = 0; c < p.d; ++c) {
                float oNorm = oAcc[r * p.dp + c] * invDenom;
                if (smoothVActive && wsVMean != nullptr) {
                    oNorm += wsVMean[c];
                }
                outPtr[r * p.qStride[2] + c] = static_cast<Dtype>(oNorm);
            }
        }

        if (p.returnLse && lseGmBase_ != nullptr) {
            const int64_t lseOffset = LseOffset(p, task.b, task.qh, task.m0);
            float *lsePtr = reinterpret_cast<float *>(lseGmBase_) + lseOffset;
            const float *rowStat = ub_.GetStateRowStat(streamId);
            const float *corr = ub_.GetStateCorr(streamId);

            for (int64_t r = 0; r < validRows; ++r) {
                float dVal = denom[r];
                float mVal = rowStat[r * 2 + 0];
                float lseVal = (dVal > 0.0f) ? (std::log(dVal) + mVal + p.smScale * corr[r])
                                             : -INFINITY;
                lsePtr[r] = lseVal;
            }
        }
    }

    inline void FinishTileCube(const Sa2Task &/*task*/, int64_t /*streamId*/)
    {
        // 宿主仿真无需事件平账
    }

    uint8_t *qGmBase_{nullptr};
    uint8_t *outGmBase_{nullptr};
    uint8_t *lseGmBase_{nullptr};
    uint8_t *userWs_{nullptr};
    const Sa2TilingData *td_{nullptr};

    std::vector<uint8_t> ubMem_ = std::vector<uint8_t>(Sa2DeviceRes<G>::kUbEnd, 0);
    std::vector<uint8_t> l1Mem_ = std::vector<uint8_t>(kL1End, 0);
    std::vector<uint8_t> l0cMem_ = std::vector<uint8_t>(kL0CSize, 0);

    Sa2UbBuffer<G> ub_{};
    Sa2L1Buffer l1_{};
    Sa2L0cBuffer l0c_{};
};

} // namespace sim
} // namespace sa2

#endif // SA2_SIM_LOOP_H_
