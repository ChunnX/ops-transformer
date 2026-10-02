// sa2_kv_prep.h — SageAttention2 KV 预处理（K 均值 UB 分块宽向量归约 + INT8 K 量化 + MX FP8 V 量化，
// K/V 混流三级切片流水）。
//
// 结构（2026-10-02 全向量化收口；接线状态唯一权威注记见 sa2_spec.h M2c-A3 段）：
//   K1 ProcessKMeanStreamKDevice：GM K → UB 行块 ping-pong（MTE2 拉块）→ 64-lane 纵向归约
//      （Cast<float> + Add 阵列）→ 全头单写者 GM 写回。128 行块 → 64 行子块 → 全头折叠的
//      确定性加法树与树序不变（FP32 非结合下与 sim oracle 逐位同构），零跨核原子加。
//   K2 ProcessHeadChunk：K/V 混流量化经 ExecuteQuantBatchPipeline 三级流水——
//      Stage1 LoadQuantSliceDevice    MTE2 预取切片 i+1：K/V 行块入输入双缓冲槽（kKvPrepKIn/VIn），
//                                     无效 token 行与 pad 列确定性清零（量化内层免掩码 + 码字 pad=0）；
//      Stage2 ComputeQuantSliceDevice 切片 i：K 中心化 → 行/段 amax（ReduceMax）→
//            Sa2QuantScaleReciprocal 同式一次倒数（amax/127 → 1e-7 下限 → 1/scale）→ 倒数向量乘
//            + CAST_RINT 取整 + ±127 钳制 → 码字槽；V MX 组 per-channel amax（纵向 Max 阵列）→
//            E8M0 阶码位技巧（+0x007FFFFF>>23 + P1-6 饱和兜底，不动）→ 2^-e 精确二次幂倒数行
//            → 向量乘 + F8 RNE 铸码；
//      Stage3 StoreQuantSliceDevice   写回切片 i-1：k8/kScale/v8 槽 MTE3 猝发写回，vScale 按
//            VScalePairWordOff nN 唯一算式标量写像（D-R4 打包布局契约不动）。
//   输入/输出槽独立双缓冲：Load(i+1) 写入槽 (i+1)%2、Compute(i) 读输入槽 i%2 写输出槽 i%2、
//   Store(i-1) 读输出槽 (i-1)%2——程序序 + 本地 HardEvent 对（票 kKvPrepEvSlot0/1，域
//   MTE2_V/V_MTE2/V_MTE3/MTE3_V）收口下槽复用距离 = 2 切片，三级两两不同槽；
//   无竞争调度证明 = conformance kv_prep_three_stage_pipeline_guard。
//
// 双轨口径：向量计算序列（vapi）为 __NPU_ARCH__ arch pass 与 __CCE_KT_TEST__ 宿主桩轨共用——
// 宿主语义 = hoststub MicroAPI 逐位仿真，conformance 直接执行本序列与 sim oracle 对账；
// MIX 第三 pass（kernel_operator 在而 vapi 别名不在）走标量解析轨（仅编译期通过；运行期调用点
// sa2_kernel.cpp ProcessKvPreprocessing 由 __NPU_ARCH__ 门控，MIX pass 不可达）。
#if !defined(__AICORE__) && !defined(__CCE_KT_TEST__)
#error "op_kernel/ is strictly reserved for Ascend NPU device compilation (__AICORE__ or __CCE_KT_TEST__)! Host simulation must use sim/sa2_sim_engine.h instead."
#endif

#ifndef SA2_KV_PREP_H_
#define SA2_KV_PREP_H_

#include <cstdint>
#include <cstring>
#include <cstdio>

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
#ifndef __ubuf__
#define __ubuf__
#endif
#ifndef GM_ADDR
using GM_ADDR = uint8_t*;
#endif

namespace sa2 {

inline int c_trace = 0;

// 阶段一本地 HardEvent 票（票号 0/1，逐槽携带 MTE2_V/V_MTE2/V_MTE3/MTE3_V 四域事件对）。
// 票号分域分相不相交：MTE1_MTE2 域 0..3 = 注意力阶段 K/V 预取、MTE2_V/V_MTE2 域 4/5 =
// scale 预取（注意力阶段）；阶段一与注意力阶段经 sa2_kernel.cpp SyncAll 收口，票号复用安全。
constexpr int64_t kKvPrepEvSlot0 = 0;
constexpr int64_t kKvPrepEvSlot1 = 1;

// 向量计算轨门控：arch pass（真机）与宿主桩轨共用 vapi 序列；MIX 第三 pass 走标量解析轨。
#if defined(__NPU_ARCH__) || defined(__CCE_KT_TEST__)
#define SA2_KV_PREP_VAPI 1
#else
#define SA2_KV_PREP_VAPI 0
#endif

#if SA2_KV_PREP_VAPI
// CastTrait 口径同 sa2_attention_loop.h：宽化 UNKNOWN（ZERO/ONE = 源寄存器低/高半区选择）、
// 窄化一律 CAST_RINT（RNE，与宿主 aicNearbyInt / FloatToFp8E4M3 舍入一致）、F8 域 SAT。
using vapi::CastTrait;
using vapi::MaskMergeMode;
using vapi::RegLayout;
using vapi::SatMode;
constexpr int64_t kKvPrepVlF32 = 64;  // RegTensor<float> lane 数（256B 寄存器）
constexpr vapi::CastTrait kKvCastWide0 = {
    RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
constexpr vapi::CastTrait kKvCastWide1 = {
    RegLayout::ONE, SatMode::UNKNOWN, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
constexpr vapi::CastTrait kKvCastRint = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
constexpr vapi::CastTrait kKvCastF322F8 = {
    RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
#endif

template <typename Dtype, Gran G>
class Sa2KvPreprocessor {
public:
    __aicore__ inline Sa2KvPreprocessor()
    {
        pipe_.InitBuffer(ubBuf_, static_cast<uint32_t>(kKvPrepUbBytes));
        // M2b-0 同款：真实 LocalTensor 无 GetData()，UB 基址经 GetPhyAddr() 转 __ubuf__ 限定指针
        //（hoststub 下展开为普通指针，宿主执行轨零改动）。
        ub_ = reinterpret_cast<__ubuf__ uint8_t *>(ubBuf_.template Get<uint8_t>().GetPhyAddr());
    }

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
     * @brief 对指定 KV head 的特定 Token 分块执行 K/V 量化（Stage 1 Pass 2，三级切片流水）。
     */
    __aicore__ inline void ProcessHeadChunk(
        GM_ADDR kGmBase, GM_ADDR vGmBase, GM_ADDR userWs,
        const Sa2TilingData &td, int64_t b, int64_t kh,
        int64_t chunkStart, int64_t chunkLen)
    {
        const Sa2Params &p = td.params;
        const Sa2KvHeadView view = KvHeadView(p, b, kh);

        Sa2ChunkCtx ctx;
        ctx.gmK = reinterpret_cast<__gm__ const Dtype *>(kGmBase) + view.gmK;
        ctx.gmV = reinterpret_cast<__gm__ const Dtype *>(vGmBase) + view.gmV;
        ctx.wsKMean = reinterpret_cast<__gm__ const float *>(userWs + td.kMean + view.wsKMean);
        ctx.wsVMean = reinterpret_cast<__gm__ const float *>(userWs + td.vMean + view.wsVMean);
        ctx.wsK8 = reinterpret_cast<__gm__ int8_t *>(userWs + td.k8 + view.wsK8);
        ctx.wsKScale = reinterpret_cast<__gm__ half *>(userWs + td.kScale + view.wsKScale);
        ctx.wsV8 = reinterpret_cast<__gm__ int8_t *>(userWs + td.v8 + view.wsV8);
        ctx.wsVScale = reinterpret_cast<__gm__ int16_t *>(userWs + td.vScale + view.wsVScale);
        ctx.p = &p;
        ctx.chunkStart = chunkStart;
        ctx.chunkLen = chunkLen;

        // K/V 混流三级切片流水（Stage1 MTE2 预取 i+1 / Stage2 Vector 量化 i / Stage3 MTE3 写回 i-1）
        Sa2LoadFn loadFn{this, &ctx};
        Sa2ComputeFn computeFn{this, &ctx};
        Sa2StoreFn storeFn{this, &ctx};
        ExecuteQuantBatchPipeline(chunkLen, p.d, p.dp, loadFn, computeFn, storeFn);
    }

private:
    // 【迭代 9 算法甄别（D-R8 对照）结论不变】K 均值归约只发生在单 AIV 内（sa2_kernel.cpp
    // 单写者 head 分配），零跨核原子加；64 行子块 → 128 行块 → 全头折叠的加法树静态可复现，
    // 与 conformance 位一致断言兼容。本轮落地：子块行数据经 MTE2 拉入 UB 双缓冲槽后由
    // 64-lane 纵向 Cast/Add 阵列归约——每列 lane 内仍按行序逐次 FP32 相加，加法树与
    // sim oracle（sa2_sim_kv_prep.h ProcessKMean）逐位同构（树未换，义务注记继续成立）。

    // ---- 切片上下文与三级流水函数子（经 ExecuteQuantBatchPipeline 驱动）----
    struct Sa2ChunkCtx {
        __gm__ const Dtype *gmK;
        __gm__ const Dtype *gmV;
        __gm__ const float *wsKMean;
        __gm__ const float *wsVMean;
        __gm__ int8_t *wsK8;
        __gm__ half *wsKScale;
        __gm__ int8_t *wsV8;
        __gm__ int16_t *wsVScale;
        const Sa2Params *p;
        int64_t chunkStart;
        int64_t chunkLen;
    };

    struct Sa2LoadFn {
        Sa2KvPreprocessor *self;
        const Sa2ChunkCtx *ctx;
        __aicore__ inline void operator()(const Sa2QuantSliceDesc &desc) const
        {
            self->LoadQuantSlice(*ctx, desc);
        }
    };
    struct Sa2ComputeFn {
        Sa2KvPreprocessor *self;
        const Sa2ChunkCtx *ctx;
        __aicore__ inline void operator()(const Sa2QuantSliceDesc &desc) const
        {
            self->ComputeQuantSlice(*ctx, desc);
        }
    };
    struct Sa2StoreFn {
        Sa2KvPreprocessor *self;
        const Sa2ChunkCtx *ctx;
        __aicore__ inline void operator()(const Sa2QuantSliceDesc &desc) const
        {
            self->StoreQuantSlice(*ctx, desc);
        }
    };

    template <typename T>
    __aicore__ inline __ubuf__ T *UbAt(int64_t byteOffset)
    {
        return reinterpret_cast<__ubuf__ T *>(ub_ + byteOffset);
    }

    __aicore__ inline __ubuf__ Dtype *MeanSliceOf(int64_t slot)
    {
        return UbAt<Dtype>(slot == 0 ? kKvPrepMeanSlice0 : kKvPrepMeanSlice1);
    }
    __aicore__ inline __ubuf__ Dtype *KInOf(int64_t slot)
    {
        return UbAt<Dtype>(slot == 0 ? kKvPrepKIn0 : kKvPrepKIn1);
    }
    __aicore__ inline __ubuf__ Dtype *VInOf(int64_t slot)
    {
        return UbAt<Dtype>(slot == 0 ? kKvPrepVIn0 : kKvPrepVIn1);
    }
    __aicore__ inline __ubuf__ uint8_t *KOutOf(int64_t slot)
    {
        return UbAt<uint8_t>(slot == 0 ? kKvPrepKOut0 : kKvPrepKOut1);
    }
    __aicore__ inline __ubuf__ uint8_t *VOutOf(int64_t slot)
    {
        return UbAt<uint8_t>(slot == 0 ? kKvPrepVOut0 : kKvPrepVOut1);
    }
    __aicore__ inline __ubuf__ half *KScaleOf(int64_t slot)
    {
        return UbAt<half>(slot == 0 ? kKvPrepKScale0 : kKvPrepKScale1);
    }
    __aicore__ inline __ubuf__ uint8_t *ExpSubOf(int64_t slot, int64_t sub)
    {
        // 槽局部（防跨切片覆写竞态：Store(i-1) 读的一定是本切片槽写入的阶码）
        const int64_t base = (slot == 0) ? kKvPrepExpSub0 : kKvPrepExpSub1;
        return UbAt<uint8_t>(base + sub * kKvPrepExpSubBytes);
    }
    static __aicore__ inline uint16_t EvOf(int64_t slot)
    {
        return static_cast<uint16_t>(slot == 0 ? kKvPrepEvSlot0 : kKvPrepEvSlot1);
    }

    // ---- Stage1 共用：GM 行块 → UB 槽（MTE2 拉块；stub 轨为真实逐元素搬运）----
    // 确定性零填充整槽后仅回填有效 token 行的 [0,d) 列：无效 token 行与 pad 列恒 0，
    // 使量化内层全宽免掩码（pad 列贡献 0 到 amax 且产 0 码字，与 oracle 写 0 逐位一致）。
    // 真机 DataCopy 行内块宽 = d16 = d & ~15（32B 块粒度），尾列 d%16 由标量回填，
    // 全程零 GM 越读；LoadAlign 整寄存器读的越行字节只入寄存器高位 lane，被 CastTrait
    // ZERO 半区选择与全槽零填充双重屏蔽，且不越出 TBuf 分配（槽间邻接见 spec 偏移链）。
    __aicore__ inline void LoadRowsToUb(
        __ubuf__ Dtype *ubRows, __gm__ const Dtype *gmRows,
        int64_t rows, int64_t d, int64_t dp, int64_t gmRowStride,
        int64_t tokenStart, int64_t sk)
    {
        const int64_t total = rows * dp;
        for (int64_t i = 0; i < total; ++i) {
            ubRows[i] = static_cast<Dtype>(0);
        }
        int64_t cVec = 0;
#if __has_include("kernel_operator.h")
        cVec = d & ~static_cast<int64_t>(15);
        for (int64_t r = 0; r < rows; ++r) {
            if (tokenStart + r >= sk || cVec <= 0) {
                continue;
            }
            AscendC::GlobalTensor<Dtype> gmSrc;
            gmSrc.SetGlobalBuffer(const_cast<__gm__ Dtype *>(gmRows + r * gmRowStride));
            const int64_t rowByteOff =
                reinterpret_cast<__ubuf__ uint8_t *>(ubRows + r * dp) - ub_;
            AscendC::LocalTensor<Dtype> ubDst =
                ubBuf_.template Get<uint8_t>()[static_cast<uint32_t>(rowByteOff)]
                    .template ReinterpretCast<Dtype>();
            AscendC::DataCopy(ubDst, gmSrc, static_cast<uint32_t>(cVec));
        }
        AscendC::PipeBarrier<PIPE_MTE2>();
#endif
        for (int64_t r = 0; r < rows; ++r) {
            if (tokenStart + r >= sk) {
                continue;
            }
            __gm__ const Dtype *gmRow = gmRows + r * gmRowStride;
            __ubuf__ Dtype *ubRow = ubRows + r * dp;
            for (int64_t c = cVec; c < d; ++c) {
                ubRow[c] = gmRow[c];
            }
        }
    }

    // ---- K 均值：64 行子块 UB 纵向归约（列 lane 内按行序 FP32 逐次相加）----
    // 子块和累入块级累加器 kKvPrepMeanAcc；每 128 行块由 MergeKMeanAcc 汇入全头和一次——
    // 加法树 = 128 行块 → 64 行子块 → 全头折叠，与 sim oracle（ProcessKMean chunkBuf 树）
    // 逐位同构（FP32 非结合：块内先折子块、块间再折叠的次序不可换）。
    __aicore__ inline void ReduceKMeanSub(
        __ubuf__ const Dtype *slice, int64_t rows, int64_t d, int64_t dp)
    {
        __ubuf__ float *accUb = UbAt<float>(kKvPrepMeanAcc);
        (void)d;
#if SA2_KV_PREP_VAPI
        vapi::MaskReg m64 = vapi::CreateMask<float, vapi::MaskPattern::ALL>();
        vapi::RegTensor<float> acc0;
        vapi::LoadAlign(acc0, accUb);
        vapi::RegTensor<float> acc1;
        const bool wide = (dp > kKvPrepVlF32);
        if (wide) {
            vapi::LoadAlign(acc1, accUb + kKvPrepVlF32);
        }
        for (int64_t r = 0; r < rows; ++r) {
            __ubuf__ const Dtype *row = slice + r * dp;
            vapi::RegTensor<Dtype> h0;
            vapi::LoadAlign(h0, row);
            vapi::RegTensor<float> f0;
            vapi::Cast<float, Dtype, kKvCastWide0>(f0, h0, m64);
            vapi::Add(acc0, acc0, f0, m64);
            if (wide) {
                vapi::RegTensor<Dtype> h1;
                vapi::LoadAlign(h1, row + kKvPrepVlF32);
                vapi::RegTensor<float> f1;
                vapi::Cast<float, Dtype, kKvCastWide1>(f1, h1, m64);
                vapi::Add(acc1, acc1, f1, m64);
            }
        }
        vapi::StoreAlign(accUb, acc0, m64);
        if (wide) {
            vapi::StoreAlign(accUb + kKvPrepVlF32, acc1, m64);
        }
#else
        // MIX 第三 pass 标量解析轨（运行期不可达）：语句序与 vapi 轨逐 lane 同构
        for (int64_t r = 0; r < rows; ++r) {
            __ubuf__ const Dtype *row = slice + r * dp;
            for (int64_t c = 0; c < d; ++c) {
                accUb[c] += static_cast<float>(row[c]);
            }
        }
#endif
    }

    // 块汇聚：全头累加和 += 块级累加器（每 128 行块恰一次；树序与 oracle 一致）
    __aicore__ inline void MergeKMeanAcc(int64_t dp)
    {
        __ubuf__ float *accUb = UbAt<float>(kKvPrepMeanAcc);
        __ubuf__ float *totalUb = UbAt<float>(kKvPrepMeanTotal);
#if SA2_KV_PREP_VAPI
        vapi::MaskReg m64 = vapi::CreateMask<float, vapi::MaskPattern::ALL>();
        vapi::RegTensor<float> tot0;
        vapi::LoadAlign(tot0, totalUb);
        vapi::RegTensor<float> acc0;
        vapi::LoadAlign(acc0, accUb);
        vapi::Add(tot0, tot0, acc0, m64);
        vapi::StoreAlign(totalUb, tot0, m64);
        if (dp > kKvPrepVlF32) {
            vapi::RegTensor<float> tot1;
            vapi::LoadAlign(tot1, totalUb + kKvPrepVlF32);
            vapi::RegTensor<float> acc1;
            vapi::LoadAlign(acc1, accUb + kKvPrepVlF32);
            vapi::Add(tot1, tot1, acc1, m64);
            vapi::StoreAlign(totalUb + kKvPrepVlF32, tot1, m64);
        }
#else
        for (int64_t c = 0; c < dp; ++c) {
            totalUb[c] += accUb[c];
        }
#endif
    }

    /**
     * @brief M2c-A5 K 均值 128 行分块统计（D-R8 形态；跨界汇聚分支 = Stream-K 预留，不可达）
     * 1. 完整 head 独写：单写者持有完整 head，零原子操作；
     * 2. 跨界 head 汇聚：为 Stream-K 跨核分块预留的接口形态（当前唯一调用 = 全量区间独写）；
     * 3. 加法树 = 128 行块 → 64 行子块 → 全头折叠：子块行块经 MTE2 双缓冲拉入 UB，64-lane
     *    纵向归约（列内行序 FP32 逐次相加），树序与 sim oracle 逐位同构。
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

        // 全头累加和（UB）：head 起点清零
        __ubuf__ float *totalUb = UbAt<float>(kKvPrepMeanTotal);
        for (int64_t c = 0; c < p.dp; ++c) {
            totalUb[c] = 0.0f;
        }

        const int64_t numSubs =
            (actualEnd - actualStart + kKMeanDataBlockLanes - 1) / kKMeanDataBlockLanes;
        const int64_t numBlocks =
            (actualEnd - actualStart + kKMeanBlockRows - 1) / kKMeanBlockRows;
        __ubuf__ float *accUb = UbAt<float>(kKvPrepMeanAcc);
        // MTE2 双缓冲：预取子块 0 后进入主循环——先发子块 i+1 拉块（槽 (i+1)%2），再归约子块 i。
        // 槽复用（i+2）前以 V_MTE2 事件等待 Vector 读毕（WAR 收口），装载毕以 MTE2_V 放行归约。
        {
            const int64_t s0 = actualStart;
            const int64_t rows0 = aicMin(kKMeanDataBlockLanes, actualEnd - s0);
            LoadRowsToUb(MeanSliceOf(0), gmK + s0 * p.kStride[2], rows0, p.d, p.dp,
                         p.kStride[2], s0, p.sk);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EvOf(0));
        }
        int64_t subGlobal = 0;
        for (int64_t block = 0; block < numBlocks; ++block) {
            const int64_t blockStart = actualStart + block * kKMeanBlockRows;
            const int64_t blockEnd = aicMin(blockStart + kKMeanBlockRows, actualEnd);
            const int64_t subsInBlock =
                (blockEnd - blockStart + kKMeanDataBlockLanes - 1) / kKMeanDataBlockLanes;
            // 块级累加器清零（树的第一层：128 行块内先折 64 行子块）
            for (int64_t c = 0; c < p.dp; ++c) {
                accUb[c] = 0.0f;
            }
            for (int64_t s = 0; s < subsInBlock; ++s, ++subGlobal) {
                const int64_t slot = subGlobal & 1;
                const int64_t next = subGlobal + 1;
                if (next < numSubs) {
                    const int64_t nextSlot = slot ^ 1;
                    if (next >= 2) {
                        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EvOf(nextSlot));
                    }
                    const int64_t s1 = actualStart + next * kKMeanDataBlockLanes;
                    const int64_t rows1 = aicMin(kKMeanDataBlockLanes, actualEnd - s1);
                    LoadRowsToUb(MeanSliceOf(nextSlot), gmK + s1 * p.kStride[2], rows1, p.d, p.dp,
                                 p.kStride[2], s1, p.sk);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EvOf(nextSlot));
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EvOf(slot));
                ReduceKMeanSub(MeanSliceOf(slot),
                               aicMin(kKMeanDataBlockLanes, actualEnd -
                                                                          (actualStart + subGlobal * kKMeanDataBlockLanes)),
                               p.d, p.dp);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EvOf(slot));
            }
            MergeKMeanAcc(p.dp);  // 树的第二层：块和汇入全头和（每 128 行块恰一次）
        }

        // 完整 head 独写 vs 跨界 head 汇聚（单写者 GM 写回；语义与分块树序不变）
        if (isExclusiveHead && actualStart == 0 && actualEnd == p.sk) {
            const float invSk = 1.0f / static_cast<float>(p.sk);
            for (int64_t c = 0; c < p.d; ++c) {
                wsKMean[c] = totalUb[c] * invSk;
            }
            for (int64_t c = p.d; c < p.dp; ++c) {
                wsKMean[c] = 0.0f;
            }
        } else {
            if (actualStart == 0) {
                for (int64_t c = 0; c < p.d; ++c) {
                    wsKMean[c] = totalUb[c];
                }
                for (int64_t c = p.d; c < p.dp; ++c) {
                    wsKMean[c] = 0.0f;
                }
            } else {
                for (int64_t c = 0; c < p.d; ++c) {
                    wsKMean[c] += totalUb[c];
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

    // V 均值：smooth_v 且 fp32 档语义（行主序逐行累加，与 sim oracle 同序）；smooth 关闭只落零。
    __aicore__ inline void ProcessVMeanDevice(
        __gm__ const Dtype *gmV, __gm__ float *wsVMean, const Sa2Params &p)
    {
        const bool smoothVActive = p.smoothV && (p.accum == static_cast<int64_t>(PvAccum::kFp32));
        if (smoothVActive && p.sk > 0) {
            float sumVBuf[kMaxHeadDim] = {0.0f};
            for (int64_t s = 0; s < p.sk; ++s) {
                __gm__ const Dtype *row = gmV + s * p.vStride[2];
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

    // ---- 三级流水 Stage1：切片 K/V 行块 MTE2 预取 ----
    __aicore__ inline void LoadQuantSlice(const Sa2ChunkCtx &ctx, const Sa2QuantSliceDesc &desc)
    {
        const Sa2Params &p = *ctx.p;
        const int64_t slot = desc.bufferSlot;
        const int64_t rows = aicMin(kKvPrepSliceRows, ctx.chunkLen - desc.rowOffset);
        const int64_t tokenStart = ctx.chunkStart + desc.rowOffset;
        // 槽复用（sliceIndex ≥ 2）前等 Vector 读毕同槽（V_MTE2 WAR 收口）
        if (desc.sliceIndex >= 2) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EvOf(slot));
        }
        LoadRowsToUb(KInOf(slot), ctx.gmK + tokenStart * p.kStride[2], rows,
                     p.d, p.dp, p.kStride[2], tokenStart, p.sk);
        LoadRowsToUb(VInOf(slot), ctx.gmV + tokenStart * p.vStride[2], rows,
                     p.d, p.dp, p.vStride[2], tokenStart, p.sk);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EvOf(slot));
    }

    // μ 行 GM → UB 暂存（标量装载；vapi Sub 的向量侧操作数来源）
    __aicore__ inline void LoadMeanRow(__gm__ const float *wsMean, int64_t dp)
    {
        __ubuf__ float *meanRow = UbAt<float>(kKvPrepMeanRow);
        for (int64_t c = 0; c < dp; ++c) {
            meanRow[c] = wsMean[c];
        }
    }

    // ---- Stage2（K）：中心化 → amax → 一次倒数 → 倒数向量乘 + CAST_RINT + 钳制 ----
    __aicore__ inline void KQuantCompute(
        const Sa2ChunkCtx &ctx, const Sa2QuantSliceDesc &desc, int64_t slot, int64_t rows)
    {
        const Sa2Params &p = *ctx.p;
        constexpr bool isThreadGran = (G == Gran::kPerThread);
        const float kFactor = p.kFactor;
        const int64_t tokenStart = ctx.chunkStart + desc.rowOffset;
        __ubuf__ const Dtype *kIn = KInOf(slot);
        __ubuf__ uint8_t *kOut = KOutOf(slot);
        __ubuf__ half *kScaleOut = KScaleOf(slot);
        __ubuf__ float *meanRow = UbAt<float>(kKvPrepMeanRow);
        __ubuf__ float *scalarSlot = UbAt<float>(kKvPrepScalarSlot);
#if SA2_KV_PREP_VAPI
        vapi::MaskReg m64 = vapi::CreateMask<float, vapi::MaskPattern::ALL>();
        uint32_t oneLane = 1u;
        vapi::MaskReg m1 = vapi::UpdateMask<uint32_t>(oneLane);
        vapi::RegTensor<int16_t> pos127;
        vapi::Duplicate(pos127, 127);
        vapi::RegTensor<int16_t> neg127;
        vapi::Duplicate(neg127, -127);
        for (int64_t r = 0; r < rows; ++r) {
            const int64_t token = tokenStart + r;
            const bool validToken = (token < p.sk);
            __ubuf__ const Dtype *rowIn = kIn + r * p.dp;
            __ubuf__ uint8_t *rowOut = kOut + r * p.dp;
            if (!validToken) {
                // 无效 token：码字行清零；scale = amax=0 → 下限 1.0 的既有契约（half(kFactor)）
                vapi::RegTensor<uint8_t> zero8;
                vapi::Duplicate(zero8, 0);
                vapi::StoreAlign(rowOut, zero8, m64);
                if (p.dp > kKvPrepVlF32) {
                    vapi::StoreAlign(rowOut + kKvPrepVlF32, zero8, m64);
                }
                if constexpr (!isThreadGran) {
                    kScaleOut[r * p.segWidth] = Sa2ScalarFromFloat<half>(kFactor);
                } else {
                    const int64_t numSegs = p.dp / kSegCols;
                    for (int64_t seg = 0; seg < numSegs; ++seg) {
                        kScaleOut[r * p.segWidth + seg] = Sa2ScalarFromFloat<half>(kFactor);
                    }
                }
                continue;
            }
            float amax = 0.0f;
            vapi::RegTensor<float> invBrc;
            if constexpr (!isThreadGran) {
                // per_warp：行 amax（全宽，pad 列经槽零填充贡献 0）
                vapi::RegTensor<float> amaxV;
                vapi::Duplicate(amaxV, 0.0f);
                if (c_trace && desc.sliceIndex == 1 && r >= 62) std::fprintf(stderr, "[KB] r=%lld pre-loop ub=%p vOut00=%02x\n", (long long)r, (void*)VOutOf(0), VOutOf(0)[0]);
                for (int64_t seg = 0; seg < p.dp; seg += kKvPrepVlF32) {
                    vapi::RegTensor<Dtype> h;
                    vapi::LoadAlign(h, rowIn + seg);
                    if (c_trace && desc.sliceIndex == 1 && r >= 62) std::fprintf(stderr, "[KB] r=%lld seg=%lld after-loadh vOut00=%02x\n", (long long)r, (long long)seg, VOutOf(0)[0]);
                    vapi::RegTensor<float> f;
                    vapi::Cast<float, Dtype, kKvCastWide0>(f, h, m64);
                    vapi::RegTensor<float> meanV;
                    vapi::LoadAlign(meanV, meanRow + seg);
                    if (c_trace && desc.sliceIndex == 1 && r >= 62) std::fprintf(stderr, "[KB] r=%lld seg=%lld after-loadmean vOut00=%02x\n", (long long)r, (long long)seg, VOutOf(0)[0]);
                    vapi::Sub(f, f, meanV, m64);
                    vapi::RegTensor<float> neg;
                    vapi::Muls(neg, f, -1.0f, m64);
                    vapi::Max(f, f, neg, m64);  // |x| = max(x, −x)
                    vapi::Max(amaxV, amaxV, f, m64);
                }
                if (c_trace && desc.sliceIndex == 1 && r >= 62) std::fprintf(stderr, "[KB] r=%lld after-amaxloop vOut00=%02x\n", (long long)r, VOutOf(0)[0]);
                vapi::RegTensor<float> amaxRed;
                vapi::ReduceMax(amaxRed, amaxV, m64);
                vapi::StoreAlign(scalarSlot, amaxRed, m1);
                amax = scalarSlot[0];  // 直接标量装载（p33e 合法形态）
                // 一次倒数（Sa2QuantScaleReciprocal 同式：amax/127 → 1e-7 下限 → 1/scale）
                const Sa2QuantScaleReciprocal kRecip(amax, kFactor);
                scalarSlot[0] = kRecip.invScale;
                vapi::LoadAlign<float, vapi::LoadDist::DIST_BRC_B32>(invBrc, scalarSlot);
                for (int64_t seg = 0; seg < p.dp; seg += kKvPrepVlF32) {
                    QuantRowSeg(rowIn + seg, meanRow + seg, rowOut + seg, invBrc, pos127, neg127);
                }
                kScaleOut[r * p.segWidth] = Sa2ScalarFromFloat<half>(kRecip.scaledFactor);
            } else {
                // per_thread：kSegCols=16 分段，每段一次倒数 + 16-lane 掩码向量量化
                const int64_t numSegs = p.dp / kSegCols;
                for (int64_t seg = 0; seg < numSegs; ++seg) {
                    const int64_t segBase = seg * kSegCols;
                    uint32_t cnt16 = static_cast<uint32_t>(kSegCols);
                    vapi::MaskReg act16 = vapi::UpdateMask<uint32_t>(cnt16);
                    vapi::RegTensor<Dtype> h;
                    vapi::LoadAlign(h, rowIn + segBase);
                    vapi::RegTensor<float> f;
                    vapi::Cast<float, Dtype, kKvCastWide0>(f, h, act16);
                    vapi::RegTensor<float> meanV;
                    vapi::LoadAlign(meanV, meanRow + segBase);
                    vapi::Sub(f, f, meanV, act16);
                    vapi::RegTensor<float> neg;
                    vapi::Muls(neg, f, -1.0f, act16);
                    vapi::Max(f, f, neg, act16);
                    vapi::RegTensor<float> amaxRed;
                    vapi::ReduceMax(amaxRed, f, act16);
                    vapi::StoreAlign(scalarSlot, amaxRed, m1);
                    amax = scalarSlot[0];
                    const Sa2QuantScaleReciprocal kRecip(amax, kFactor);
                    scalarSlot[0] = kRecip.invScale;
                    vapi::LoadAlign<float, vapi::LoadDist::DIST_BRC_B32>(invBrc, scalarSlot);
                    QuantRowSegMasked(rowIn + segBase, meanRow + segBase, rowOut + segBase,
                                      invBrc, pos127, neg127);
                    kScaleOut[r * p.segWidth + seg] = Sa2ScalarFromFloat<half>(kRecip.scaledFactor);
                }
            }
        }
#else
        // MIX 第三 pass 标量解析轨（运行期不可达）：公式与 vapi 轨逐位同源
        for (int64_t r = 0; r < rows; ++r) {
            const int64_t token = tokenStart + r;
            const bool validToken = (token < p.sk);
            __ubuf__ const Dtype *rowIn = kIn + r * p.dp;
            __ubuf__ uint8_t *rowOut = kOut + r * p.dp;
            float rowDiff[kMaxHeadDim] = {0.0f};
            if (validToken) {
                for (int64_t c = 0; c < p.d; ++c) {
                    rowDiff[c] = static_cast<float>(rowIn[c]) - meanRow[c];
                }
            }
            if constexpr (!isThreadGran) {
                float amax = 0.0f;
                for (int64_t c = 0; c < p.d; ++c) {
                    amax = aicMax(amax, aicAbs(rowDiff[c]));
                }
                const Sa2QuantScaleReciprocal kRecip(amax, kFactor);
                for (int64_t c = 0; c < p.dp; ++c) {
                    rowOut[c] = (validToken && c < p.d) ? static_cast<uint8_t>(kRecip.Quantize(rowDiff[c])) : 0;
                }
                kScaleOut[r * p.segWidth] = Sa2ScalarFromFloat<half>(kRecip.scaledFactor);
            } else {
                const int64_t numSegs = p.dp / kSegCols;
                for (int64_t seg = 0; seg < numSegs; ++seg) {
                    const int64_t segBase = seg * kSegCols;
                    const int64_t segEnd = aicMin(segBase + kSegCols, p.d);
                    float amax = 0.0f;
                    if (validToken && segBase < p.d) {
                        for (int64_t c = segBase; c < segEnd; ++c) {
                            amax = aicMax(amax, aicAbs(rowDiff[c]));
                        }
                    }
                    const Sa2QuantScaleReciprocal kRecip(amax, kFactor);
                    for (int64_t c = segBase; c < segBase + kSegCols; ++c) {
                        rowOut[c] = (validToken && c < p.d) ? static_cast<uint8_t>(kRecip.Quantize(rowDiff[c])) : 0;
                    }
                    kScaleOut[r * p.segWidth + seg] = Sa2ScalarFromFloat<half>(kRecip.scaledFactor);
                }
            }
        }
#endif
    }

#if SA2_KV_PREP_VAPI
    // 单行 64-lane 段量化：q = diff × invScale（向量乘）→ CAST_RINT 取整 → ±127 钳制 → int8。
    // 与 Sa2QuantScaleReciprocal::Quantize 逐位同源（aicNearbyInt ≡ CAST_RINT；浮点钳制与
    // 整型钳制在 RNE 域 {…,-128→-127, +128→+127} 等价，见 conformance kv_quant_reciprocal_e8m0_guard）。
    // pos127/neg127 由调用方以循环不变寄存器传入（vloop 标量参指令外提纪律，p33d）。
    __aicore__ inline void QuantRowSeg(
        __ubuf__ const Dtype *rowSeg, __ubuf__ const float *meanSeg, __ubuf__ uint8_t *outSeg,
        vapi::RegTensor<float> &invBrc,
        vapi::RegTensor<int16_t> &pos127, vapi::RegTensor<int16_t> &neg127)
    {
        vapi::MaskReg m64 = vapi::CreateMask<float, vapi::MaskPattern::ALL>();
        vapi::RegTensor<Dtype> h;
        vapi::LoadAlign(h, rowSeg);
        vapi::RegTensor<float> f;
        vapi::Cast<float, Dtype, kKvCastWide0>(f, h, m64);
        vapi::RegTensor<float> meanV;
        vapi::LoadAlign(meanV, meanSeg);
        vapi::Sub(f, f, meanV, m64);
        vapi::Mul(f, f, invBrc, m64);
        vapi::RegTensor<int16_t> r16;
        vapi::Cast<int16_t, float, kKvCastRint>(r16, f, m64);
        vapi::RegTensor<int16_t> clamped;
        vapi::Max(clamped, r16, neg127, m64);
        vapi::Min(clamped, clamped, pos127, m64);
        vapi::RegTensor<uint8_t> c8;
        vapi::Cast<uint8_t, int16_t>(c8, clamped, m64);
        vapi::StoreAlign(outSeg, c8, m64);
    }

    __aicore__ inline void QuantRowSegMasked(
        __ubuf__ const Dtype *rowSeg, __ubuf__ const float *meanSeg, __ubuf__ uint8_t *outSeg,
        vapi::RegTensor<float> &invBrc,
        vapi::RegTensor<int16_t> &pos127, vapi::RegTensor<int16_t> &neg127)
    {
        uint32_t cnt16 = static_cast<uint32_t>(kSegCols);
        vapi::MaskReg act16 = vapi::UpdateMask<uint32_t>(cnt16);
        vapi::RegTensor<Dtype> h;
        vapi::LoadAlign(h, rowSeg);
        vapi::RegTensor<float> f;
        vapi::Cast<float, Dtype, kKvCastWide0>(f, h, act16);
        vapi::RegTensor<float> meanV;
        vapi::LoadAlign(meanV, meanSeg);
        vapi::Sub(f, f, meanV, act16);
        vapi::Mul(f, f, invBrc, act16);
        vapi::RegTensor<int16_t> r16;
        vapi::Cast<int16_t, float, kKvCastRint>(r16, f, act16);
        vapi::RegTensor<int16_t> clamped;
        vapi::Max(clamped, r16, neg127, act16);
        vapi::Min(clamped, clamped, pos127, act16);
        vapi::RegTensor<uint8_t> c8;
        vapi::Cast<uint8_t, int16_t>(c8, clamped, act16);
        vapi::StoreAlign(outSeg, c8, act16);
    }
#endif

    // ---- Stage2（V）：组 per-channel amax → E8M0 → 二次幂倒数行 → 向量乘 + F8 铸码 ----
    __aicore__ inline void VQuantCompute(
        const Sa2ChunkCtx &ctx, const Sa2QuantSliceDesc &desc, int64_t slot, int64_t rows)
    {
        const Sa2Params &p = *ctx.p;
        const int64_t tokenStart = ctx.chunkStart + desc.rowOffset;
        const int64_t pairIdx = tokenStart / (kMxGroupTokens * kMxScalePack);
        const int64_t chunkEnd = ctx.chunkStart + ctx.chunkLen;
        __ubuf__ const Dtype *vIn = VInOf(slot);
        __ubuf__ uint8_t *vOut = VOutOf(slot);
        __ubuf__ float *meanRow = UbAt<float>(kKvPrepMeanRow);
        __ubuf__ float *amaxRow = UbAt<float>(kKvPrepAmaxRow);
        __ubuf__ float *scaleRow = UbAt<float>(kKvPrepScaleRow);
        for (int64_t sub = 0; sub < kMxScalePack; ++sub) {
            const int64_t g = pairIdx * kMxScalePack + sub;
            const int64_t tStart = g * kMxGroupTokens;
            if (tStart >= chunkEnd) {
                break;  // 组越出分块（chunkLen 非 128 对齐的防御分支；现网网格恒 128 不可达）
            }
            __ubuf__ uint8_t *expSub = ExpSubOf(slot, sub);
#if SA2_KV_PREP_VAPI
            vapi::MaskReg m64 = vapi::CreateMask<float, vapi::MaskPattern::ALL>();
            // per-channel amax：32 token 行纵向 Max 阵列（64 lane 并行）；无效行零贡献
            for (int64_t seg = 0; seg < p.dp; seg += kKvPrepVlF32) {
                vapi::RegTensor<float> amaxV;
                vapi::Duplicate(amaxV, 0.0f);
                for (int64_t i = 0; i < kMxGroupTokens; ++i) {
                    const int64_t token = tStart + i;
                    if (token >= p.sk) {
                        continue;
                    }
                    __ubuf__ const Dtype *row = vIn + (tStart - tokenStart + i) * p.dp + seg;
                    vapi::RegTensor<Dtype> h;
                    vapi::LoadAlign(h, row);
                    vapi::RegTensor<float> f;
                    vapi::Cast<float, Dtype, kKvCastWide0>(f, h, m64);
                    vapi::RegTensor<float> meanV;
                    vapi::LoadAlign(meanV, meanRow + seg);
                    vapi::Sub(f, f, meanV, m64);
                    vapi::RegTensor<float> neg;
                    vapi::Muls(neg, f, -1.0f, m64);
                    vapi::Max(f, f, neg, m64);
                    vapi::Max(amaxV, amaxV, f, m64);
                }
                vapi::StoreAlign(amaxRow + seg, amaxV, m64);
            }
#else
            // MIX 第三 pass 标量解析轨（运行期不可达）
            for (int64_t c = 0; c < p.dp; ++c) {
                amaxRow[c] = 0.0f;
            }
            for (int64_t i = 0; i < kMxGroupTokens; ++i) {
                const int64_t token = tStart + i;
                if (token >= p.sk) {
                    continue;
                }
                __ubuf__ const Dtype *row = vIn + (tStart - tokenStart + i) * p.dp;
                for (int64_t c = 0; c < p.d; ++c) {
                    amaxRow[c] = aicMax(amaxRow[c], aicAbs(static_cast<float>(row[c]) - meanRow[c]));
                }
            }
#endif
            // E8M0 阶码位技巧（+0x007FFFFF>>23 + P1-6 兜底，不动）+ 2^-e 精确二次幂倒数行
            //（AmlaScaleMultiply 对 1.0 的指数平移，乘幂精确 ⇒ 倒数乘 ≡ 除以 2^e）
            for (int64_t c = 0; c < p.dp; ++c) {
                const uint8_t uexp = ComputeE8m0Exponent(amaxRow[c]);
                expSub[c] = uexp;
                scaleRow[c] = AmlaScaleMultiply(1.0f, -(static_cast<int32_t>(uexp) - 127));
            }
            // per-row：中心化 → 倒数向量乘 → F8 RNE 铸码（SAT 域与 FloatToFp8E4M3 语义一致）
            for (int64_t i = 0; i < kMxGroupTokens; ++i) {
                const int64_t token = tStart + i;
                const int64_t rowLocal = tStart - tokenStart + i;
                if (rowLocal >= rows) {
                    break;
                }
                __ubuf__ uint8_t *rowOut = vOut + rowLocal * p.dp;
                if (token >= p.sk) {
                    for (int64_t c = 0; c < p.dp; ++c) {
                        rowOut[c] = 0;
                    }
                    continue;
                }
#if SA2_KV_PREP_VAPI
                __ubuf__ const Dtype *row = vIn + rowLocal * p.dp;
                for (int64_t seg = 0; seg < p.dp; seg += kKvPrepVlF32) {
                    vapi::MaskReg m64 = vapi::CreateMask<float, vapi::MaskPattern::ALL>();
                    vapi::RegTensor<Dtype> h;
                    vapi::LoadAlign(h, row + seg);
                    vapi::RegTensor<float> f;
                    vapi::Cast<float, Dtype, kKvCastWide0>(f, h, m64);
                    vapi::RegTensor<float> meanV;
                    vapi::LoadAlign(meanV, meanRow + seg);
                    vapi::Sub(f, f, meanV, m64);
                    vapi::RegTensor<float> scale2V;
                    vapi::LoadAlign(scale2V, scaleRow + seg);
                    vapi::Mul(f, f, scale2V, m64);
                    vapi::RegTensor<uint8_t> c8;
                    vapi::Cast<uint8_t, float, kKvCastF322F8>(c8, f, m64);
                    vapi::StoreAlign(rowOut + seg, c8, m64);
                }
#else
                // MIX 第三 pass 标量解析轨（运行期不可达）：Amla 指数平移与倒数行乘逐位同源
                __ubuf__ const Dtype *row = vIn + rowLocal * p.dp;
                for (int64_t c = 0; c < p.dp; ++c) {
                    const float centered = static_cast<float>(row[c]) - meanRow[c];
                    const int32_t expVal = static_cast<int32_t>(expSub[c]) - 127;
                    rowOut[c] = FloatToFp8E4M3(AmlaScaleMultiply(centered, -expVal));
                }
#endif
            }
        }
    }

    // ---- Stage3：输出槽 → GM 写回（MTE3 猝发 + vScale nN 标量写像）----
    __aicore__ inline void StoreQuantSlice(const Sa2ChunkCtx &ctx, const Sa2QuantSliceDesc &desc)
    {
        const Sa2Params &p = *ctx.p;
        const int64_t slot = desc.bufferSlot;
        const int64_t rows = aicMin(kKvPrepSliceRows, ctx.chunkLen - desc.rowOffset);
        const int64_t tokenStart = ctx.chunkStart + desc.rowOffset;
        __ubuf__ uint8_t *kOut = KOutOf(slot);
        __ubuf__ uint8_t *vOut = VOutOf(slot);
        __ubuf__ half *kScaleOut = KScaleOf(slot);
        __ubuf__ uint8_t *expSub0 = ExpSubOf(slot, 0);
        __ubuf__ uint8_t *expSub1 = ExpSubOf(slot, 1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EvOf(slot));
        const int64_t k8Bytes = rows * p.dp;
#if __has_include("kernel_operator.h")
        if ((k8Bytes & 31) == 0) {
            CopyUbToGm(ctx.wsK8 + tokenStart * p.dp, kOut, k8Bytes);
            CopyUbToGm(reinterpret_cast<__gm__ uint8_t *>(ctx.wsV8 + tokenStart * p.dp), vOut, k8Bytes);
        } else {
            CopyBytesGm(reinterpret_cast<__gm__ uint8_t *>(ctx.wsK8 + tokenStart * p.dp), kOut, k8Bytes);
            CopyBytesGm(reinterpret_cast<__gm__ uint8_t *>(ctx.wsV8 + tokenStart * p.dp), vOut, k8Bytes);
        }
        const int64_t scaleBytes = rows * p.segWidth * 2;
        if ((scaleBytes & 31) == 0) {
            CopyUbToGm(reinterpret_cast<__gm__ uint8_t *>(ctx.wsKScale + tokenStart * p.segWidth),
                       reinterpret_cast<__ubuf__ uint8_t *>(kScaleOut), scaleBytes);
        } else {
            CopyBytesGm(reinterpret_cast<__gm__ uint8_t *>(ctx.wsKScale + tokenStart * p.segWidth),
                        reinterpret_cast<__ubuf__ uint8_t *>(kScaleOut), scaleBytes);
        }
        AscendC::PipeBarrier<PIPE_MTE3>();
#else
        CopyBytesGm(reinterpret_cast<__gm__ uint8_t *>(ctx.wsK8 + tokenStart * p.dp), kOut, k8Bytes);
        CopyBytesGm(reinterpret_cast<__gm__ uint8_t *>(ctx.wsV8 + tokenStart * p.dp), vOut, k8Bytes);
        CopyBytesGm(reinterpret_cast<__gm__ uint8_t *>(ctx.wsKScale + tokenStart * p.segWidth),
                    reinterpret_cast<__ubuf__ uint8_t *>(kScaleOut), rows * p.segWidth * 2);
#endif
        // vScale：逐步 nN 窗序列化（VScalePairWordOff 唯一算式；D-R4 双组打包布局契约不动）
        const int64_t pairIdx = tokenStart / (kMxGroupTokens * kMxScalePack);
        for (int64_t c = 0; c < p.dp; ++c) {
            const uint16_t packedU =
                static_cast<uint16_t>(expSub0[c]) | static_cast<uint16_t>(
                    static_cast<uint16_t>(expSub1[c]) << 8);
            ctx.wsVScale[VScalePairWordOff(pairIdx, c, p.dp)] = static_cast<int16_t>(packedU);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EvOf(slot));
    }

    __aicore__ inline void CopyUbToGm(__gm__ uint8_t *gmDst, __ubuf__ const uint8_t *ubSrc, int64_t bytes)
    {
#if __has_include("kernel_operator.h")
        AscendC::GlobalTensor<uint8_t> gmDstT;
        gmDstT.SetGlobalBuffer(gmDst);
        AscendC::LocalTensor<uint8_t> ubSrcT = ubBuf_.template Get<uint8_t>()[
            static_cast<uint32_t>((ubSrc - reinterpret_cast<__ubuf__ const uint8_t *>(ub_)))];
        AscendC::DataCopy(gmDstT, ubSrcT, static_cast<uint32_t>(bytes));
#else
        CopyBytesGm(gmDst, ubSrc, bytes);
#endif
    }

    __aicore__ inline void CopyBytesGm(__gm__ uint8_t *gmDst, __ubuf__ const uint8_t *ubSrc, int64_t bytes)
    {
        for (int64_t i = 0; i < bytes; ++i) {
            gmDst[i] = ubSrc[i];
        }
    }

    __aicore__ inline void ComputeQuantSlice(const Sa2ChunkCtx &ctx, const Sa2QuantSliceDesc &desc)
    {
        const Sa2Params &p = *ctx.p;
        const int64_t slot = desc.bufferSlot;
        const int64_t rows = aicMin(kKvPrepSliceRows, ctx.chunkLen - desc.rowOffset);
        // 输出槽复用（sliceIndex ≥ 2）前等 MTE3 读毕（MTE3_V WAR 收口）
        if (desc.sliceIndex >= 2) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EvOf(slot));
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EvOf(slot));
        LoadMeanRow(ctx.wsKMean, p.dp);
        KQuantCompute(ctx, desc, slot, rows);
        LoadMeanRow(ctx.wsVMean, p.dp);
        VQuantCompute(ctx, desc, slot, rows);
        // 出口双向放行：V 写毕输出槽 → MTE3；V 读毕输入槽 → MTE2 重填
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EvOf(slot));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EvOf(slot));
    }

    AscendC::TPipe pipe_{};
    AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf_{};
    __ubuf__ uint8_t *ub_{nullptr};
};

} // namespace sa2

#endif // SA2_KV_PREP_H_
