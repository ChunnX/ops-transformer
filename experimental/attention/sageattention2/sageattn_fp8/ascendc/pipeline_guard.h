/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SAGE_ATTN_FP8_ASCENDC_PIPELINE_GUARD_H
#define SAGE_ATTN_FP8_ASCENDC_PIPELINE_GUARD_H

#include <cstdint>
#include <cstddef>
#include <type_traits>

/**
 * 类型状态流水守卫（RES-05，task/03_review.md 资源与同步粒度类）
 *
 * 吸收自 ascend-rs 案例研究 §10/§11 的 async 类型状态协议：把"缓冲区处于哪个
 * 流水域"编码进 C++ 类型，让漏插同步从运行时死锁/audit 报警变成编译错误。
 * 状态机（单 owner，move-only，禁止拷贝）：
 *   LoadUb()  -> UbTile<float, DmaPending>    仅可 Sync()
 *   .Sync()   -> UbTile<float, ComputeReady>  可参与 Vector 计算与写回
 *   .BeginStore()                              V -> MTE3 定向事件对 + DataCopy 出
 * 纪律对应（A4/A5 向量化填实的执行约束）：
 *   - 同引擎连续指令零同步：ComputeReady 上连续调用 Exp/Adds/Muls/Reduce 不插任何
 *     屏障（同流水硬件保序）；
 *   - 跨引擎边界定向事件对：MTE2->V 与 V->MTE3 各一对 SetFlag/WaitFlag，禁止
 *     pipe_barrier(PIPE_ALL) 全停顿；
 *   - event_id 由调用方按槽位分配（双缓冲 ping/pong 各用独立 id，SYNC-04 配对纪律）。
 *
 * 宿主/KT 分支不触硬件，按序记录 OpCode 流水日志（tests/test_pipeline_guard.cpp
 * 消费），用于协议时序回归；设备分支映射 CANN 官方接口（形态依据 ascendc-sync-audit
 * references/sync-mechanisms.md 的 HardEvent 通路）。
 */

#ifndef SAGE_ATTN_DEVICE_COMPILE
#if defined(__CCE__) && !defined(__CCE_KT_TEST__)
#define SAGE_ATTN_DEVICE_COMPILE 1
#else
#define SAGE_ATTN_DEVICE_COMPILE 0
#endif
#endif

#ifndef __aicore__
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_VEC__) || defined(__DAV_C220_CUBE__)
#define __aicore__ [[]]
#else
#define __aicore__
#endif
#endif

// 宿主/KT 分支的地址空间限定回退（对齐 op_kernel/sageattention2.cpp 宿主桩约定）；
// bisheng 设备编译由编译器内建定义，不受此处影响
#ifndef __gm__
#define __gm__
#endif

#if SAGE_ATTN_DEVICE_COMPILE
#include "kernel_operator.h"
#else
#include <vector>
#endif

namespace sage_attn {
namespace pipeline {

#if !SAGE_ATTN_DEVICE_COMPILE
/**
 * 宿主流水日志：设备分支零开销（不编译），单测经 PopAll 读取时序。
 */
enum OpCode : uint32_t {
    OP_LOAD = 0,       // GM -> UB DataCopy 下发
    OP_SYNC,           // MTE2 -> V 等待（DmaPending 归位）
    OP_EXP,            // 向量 Exp
    OP_ADDS,           // 向量广播加
    OP_MULS,           // 向量广播乘
    OP_REDUCE_MAX,     // 向量归约 max
    OP_REDUCE_SUM,     // 向量归约 sum
    OP_STORE,          // V -> MTE3 事件对 + DataCopy 出
    OP_SCRATCH         // 引擎本地 scratch 分配（无 DMA，生而 ComputeReady）
};

namespace detail {
inline std::vector<uint32_t> &OpLog()
{
    static std::vector<uint32_t> log;
    return log;
}

inline void ClearOpLog()
{
    OpLog().clear();
}
} // namespace detail
#endif

/**
 * 状态标签（空类型，仅参与类型推导，零存储零开销）
 */
struct DmaPending {};    // GM -> UB 搬运在途：仅 Sync() 合法
struct ComputeReady {};  // V 引擎可见：计算/归约/写回合法

// 主模板不定义：状态即 API。使用未特化的 (T, State) 组合（如 UbTile<int8_t, ...>）
// 直接得到不完整类型编译错误；当前特化覆盖 fp32 softmax lane，A5 量化 staging
// 的 int8/fp16 通道按需增补特化。
template <typename T, typename State>
class UbTile;

/**
 * 在途态：唯一合法动作是 Sync()。
 * 忘记 Sync 直接在后续计算调用点报 "no member named 'Exp' in
 * 'UbTile<float, DmaPending>'"——编译期拦截，替代运行时同步审计单点防线。
 */
template <>
class UbTile<float, DmaPending> {
public:
#if SAGE_ATTN_DEVICE_COMPILE
    UbTile() = delete;
    UbTile(AscendC::TBuf<AscendC::TPosition::VECCALC> *buf, uint8_t event_id)
        : buf_(buf), event_id_(event_id) {}
#else
    // 宿主/KT 分支：瓦片为纯日志句柄，默认构造仅限经 LoadUb/MakeScratchTile 产生
    UbTile() {}
#endif

    UbTile(const UbTile &) = delete;
    UbTile &operator=(const UbTile &) = delete;
    UbTile(UbTile &&) noexcept = default;
    UbTile &operator=(UbTile &&) noexcept = default;

    /**
     * MTE2 -> V 定向事件等待：消费在途态，返回引擎可见态。
     * 设备分支 WaitFlag 与 LoadUb 中的 SetFlag 1:1 配对（同一 event_id）。
     * 定义置于 ComputeReady 特化之后（返回值需其完整类型，out-of-line inline）。
     */
    __aicore__ UbTile<float, ComputeReady> Sync();

private:
#if SAGE_ATTN_DEVICE_COMPILE
    AscendC::TBuf<AscendC::TPosition::VECCALC> *buf_;
    uint8_t event_id_;
#endif
};

/**
 * 引擎可见态：同引擎连续计算零同步，写回时由 BeginStore 插 V -> MTE3 事件对。
 * move-only：单 owner 语义杜绝同一缓冲被两段流水并发持有。
 */
template <>
class UbTile<float, ComputeReady> {
public:
#if SAGE_ATTN_DEVICE_COMPILE
    UbTile() = delete;
    UbTile(AscendC::TBuf<AscendC::TPosition::VECCALC> *buf, uint8_t event_id)
        : buf_(buf), event_id_(event_id) {}
#else
    UbTile() {}
#endif

    UbTile(const UbTile &) = delete;
    UbTile &operator=(const UbTile &) = delete;
    UbTile(UbTile &&) noexcept = default;
    UbTile &operator=(UbTile &&) noexcept = default;

    // 以下计算原语均为同引擎（PIPE_V）操作：硬件保序，零屏障（RES-02 纪律）。

    __aicore__ void Exp(uint32_t count)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::LocalTensor<float> t = buf_->Get<float>();
        AscendC::Exp(t, t, count);
#else
        (void)count;
        detail::OpLog().push_back(OP_EXP);
#endif
    }

    __aicore__ void Adds(float scalar, uint32_t count)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::LocalTensor<float> t = buf_->Get<float>();
        AscendC::Adds(t, t, scalar, count);
#else
        (void)scalar;
        (void)count;
        detail::OpLog().push_back(OP_ADDS);
#endif
    }

    __aicore__ void Muls(float scalar, uint32_t count)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::LocalTensor<float> t = buf_->Get<float>();
        AscendC::Muls(t, t, scalar, count);
#else
        (void)scalar;
        (void)count;
        detail::OpLog().push_back(OP_MULS);
#endif
    }

    /**
     * 向量归约 max（RES-04 纪律：归约必须走 Vector 指令，禁止标量循环）。
     * scratch_dst 与 scratch_rwork 为两个独立引擎本地缓冲（归约工作区与结果区
     * 不得复用同一缓冲）；结果经 scratch_dst[0] 标量读回。
     */
    __aicore__ float ReduceMax(UbTile<float, ComputeReady> &scratch_dst,
                               UbTile<float, ComputeReady> &scratch_rwork,
                               uint32_t count)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::LocalTensor<float> src = buf_->Get<float>();
        AscendC::LocalTensor<float> dst = scratch_dst.buf_->Get<float>();
        AscendC::LocalTensor<float> rwork = scratch_rwork.buf_->Get<float>();
        AscendC::ReduceMax(dst, src, rwork, count);
        return dst.GetValue(0);
#else
        (void)scratch_dst;
        (void)scratch_rwork;
        (void)count;
        detail::OpLog().push_back(OP_REDUCE_MAX);
        return 0.0f;
#endif
    }

    __aicore__ float ReduceSum(UbTile<float, ComputeReady> &scratch_dst,
                               UbTile<float, ComputeReady> &scratch_rwork,
                               uint32_t count)
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::LocalTensor<float> src = buf_->Get<float>();
        AscendC::LocalTensor<float> dst = scratch_dst.buf_->Get<float>();
        AscendC::LocalTensor<float> rwork = scratch_rwork.buf_->Get<float>();
        AscendC::ReduceSum(dst, src, rwork, count);
        return dst.GetValue(0);
#else
        (void)scratch_dst;
        (void)scratch_rwork;
        (void)count;
        detail::OpLog().push_back(OP_REDUCE_SUM);
        return 0.0f;
#endif
    }

    /**
     * 写回 GM：V -> MTE3 定向事件对 + DataCopy。消费 *this（右值限定），
     * 写回后的缓冲不可再被计算引用，状态机自然终结。
     */
    __aicore__ void BeginStore(__gm__ float *gm_dst, uint32_t count) &&
    {
#if SAGE_ATTN_DEVICE_COMPILE
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id_);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id_);
        AscendC::GlobalTensor<float> g;
        g.SetGlobalBuffer(gm_dst, count);
        AscendC::DataCopy(g, buf_->Get<float>(), count);
#else
        (void)gm_dst;
        (void)count;
        detail::OpLog().push_back(OP_STORE);
#endif
    }

private:
#if SAGE_ATTN_DEVICE_COMPILE
    AscendC::TBuf<AscendC::TPosition::VECCALC> *buf_;
    uint8_t event_id_;
#endif
};

/**
 * DmaPending::Sync 的 out-of-line inline 定义（两特化均已完整）。
 */
__aicore__ inline UbTile<float, ComputeReady> UbTile<float, DmaPending>::Sync()
{
#if SAGE_ATTN_DEVICE_COMPILE
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id_);
    return UbTile<float, ComputeReady>(buf_, event_id_);
#else
    detail::OpLog().push_back(OP_SYNC);
    return UbTile<float, ComputeReady>();
#endif
}

/**
 * GM -> UB 搬运入口：下发 DataCopy（MTE2 引擎）后置 MTE2 -> V 事件，
 * 返回在途态。event_id 须按缓冲槽位分配（ping/pong 分离，SYNC-04）。
 */
#if SAGE_ATTN_DEVICE_COMPILE
template <typename T>
__aicore__ inline UbTile<float, DmaPending> LoadUb(
    AscendC::TPipe &pipe,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &buf,
    const __gm__ T *gm_src,
    uint32_t count,
    uint8_t event_id)
{
    static_assert(std::is_same<T, float>::value, "pipeline_guard 当前仅特化 fp32 softmax lane");
    pipe.InitBuffer(buf, count * sizeof(T));
    AscendC::GlobalTensor<T> g;
    g.SetGlobalBuffer(gm_src, count);
    AscendC::DataCopy(buf.Get<T>(), g, count);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);
    return UbTile<float, DmaPending>(&buf, event_id);
}

/**
 * 引擎本地 scratch：仅 InitBuffer，无 DMA，生而 ComputeReady（无需 Sync）。
 */
__aicore__ inline UbTile<float, ComputeReady> MakeScratchTile(
    AscendC::TPipe &pipe,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &buf,
    uint32_t count)
{
    pipe.InitBuffer(buf, count * sizeof(float));
    return UbTile<float, ComputeReady>(&buf, 0);
}
#else
inline UbTile<float, DmaPending> LoadUb(uint32_t tile_id, uint32_t count)
{
    (void)tile_id;
    (void)count;
    detail::OpLog().push_back(OP_LOAD);
    return UbTile<float, DmaPending>();
}

inline UbTile<float, ComputeReady> MakeScratchTile(uint32_t tile_id, uint32_t count)
{
    (void)tile_id;
    (void)count;
    detail::OpLog().push_back(OP_SCRATCH);
    return UbTile<float, ComputeReady>();
}
#endif

} // namespace pipeline
} // namespace sage_attn

#endif // SAGE_ATTN_FP8_ASCENDC_PIPELINE_GUARD_H
