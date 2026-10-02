// sa2_events.h — SageAttention2 v2 跨核旗号与本地事件契约（编号即契约，engine 文件禁止散落魔数）。
//
// 规格来源：DESIGN_SPEC §6（主循环 5 旗号 ×2 流 + 本地事件 6 ≈ 16 ≤ MODE4 预算）；
// 通知侧管道语义按附录 C.4 口径落地（见各 helper 注释；Notify* = 发送方生产管道，Await* = 接收方消费管道）。
#ifndef SA2_EVENTS_H_
#define SA2_EVENTS_H_

#include <cstdint>

#include "sa2_spec.h"

#if __has_include("kernel_operator.h")
#include "kernel_operator.h"
#elif __has_include("sa2_ascendc_hoststub.h")
#include "sa2_ascendc_hoststub.h"
#endif

#ifndef __aicore__
#define __aicore__
#endif

namespace sa2 {

// ---- 跨核旗号（MODE4 计数 flag：CrossCoreSetFlag/WaitFlag，AIC↔AIV 双向可用）----
// 方向与语义（DESIGN_SPEC §6；槽奇偶 = j%2）：
//   Q8_OK     AIV→AIC  本任务 q8/qScale 已落 L1（QQuant 完成），AIC 才可发起该任务首步 QK
//                      （P1-2 任务切换竞态闭环；计数 1 set/1 wait per task，FinishTile 不需 drain）
//   SCORE_OK  AIC→AIV  score 槽(j%2) fixpipe 完成可消费               NotifyFix / AwaitFix
//   QK_FREE   AIV→AIC  score 槽(j%2) softmax 消费毕、fixpipe 可复用  NotifyV   / AwaitV
//   PV_OK     AIC→AIV  PV 槽(j%2) fixpipe 完成可折叠                 NotifyFix / AwaitFix
//   PV_FREE   AIV→AIC  PV 槽(j%2) O 折叠毕、fixpipe 可复用           NotifyV   / AwaitV
//   P_TAKEN   AIV→AIC  P 已落 L1（UB 暂存槽可复用），AIC 可发起 L1→L0A  NotifyMte3 / AwaitMte1
// 流 1 = 基号 + kStreamFlagStride（双流隔离，AIC 侧按流分别置/等）。
enum class Sa2CrossFlag : int64_t {
  kQ8Ok = 5,
  kScoreOk = 0,
  kQkFree = 1,
  kPvOk = 2,
  kPvFree = 3,
  kPTaken = 4,
};

constexpr int64_t kStreamFlagStride = 8;  // 流 1 基号偏移（Stream 0 为 0..5，Stream 1 为 8..13；MODE4 共 16 槽）

// __aicore__ 标注：设备构建（-xcce 混编）下被 Notify*/Await* 的 __aicore__ 函数运行期调用；
// 静态断言走常量求值，不受影响。
__aicore__ constexpr uint8_t Sa2FlagId(int64_t stream, Sa2CrossFlag flag) {
  return static_cast<uint8_t>(stream * kStreamFlagStride + static_cast<int64_t>(flag));
}

// 旗号预算（v27 P1-2 修订）：跨核 MODE4 旗号 6×2 流 = 12 ≤ 16 槽；本地 HardEvent 票 6 张
//（K/V 预取 4 为 M2b MTE 预取预留、scale 2）。MODE4 与 HardEvent 是两个独立资源域，分开记账。
constexpr int64_t kCrossFlagBudget = 2 * 6;  // 12
constexpr int64_t kLocalEventBudget = 6;     // K/V 预取 4（M2b 预留）+ scale 2
static_assert(kCrossFlagBudget <= 16, "MODE4 跨核旗号 12 ≤ 16 槽（真机容量核实挂 M2b 950PR 语义核实项）");
static_assert(static_cast<int64_t>(Sa2CrossFlag::kPTaken) == 4 &&
                  static_cast<int64_t>(Sa2CrossFlag::kQ8Ok) == 5,
              "旗号编号冻结 0..5");
static_assert(Sa2FlagId(1, Sa2CrossFlag::kQ8Ok) == 13, "流 1 基号偏移 = 8（双流隔离）");

// ---- 本地 HardEvent 票号（同一编号在不同 HardEvent 域内各自独立成组）----
//   K/V 预取 ∈ MTE1_MTE2 域：MTE1 取空 L1 槽后 Set，MTE2 预取下一 tile 进同槽前 Wait；
//   scale ∈ V_MTE2 域：AIV 读毕 scale 槽后 Set，MTE2 重取前 Wait。
constexpr int64_t kEvKPrefetch0 = 0;  // K 步 tile 预取 ping（L1 槽 kL1KStep0）
constexpr int64_t kEvKPrefetch1 = 1;  // K 步 tile 预取 pong（L1 槽 kL1KStep1）
constexpr int64_t kEvVPrefetch0 = 2;  // V 步 tile 预取 ping（L1 槽 kL1VStep0）
constexpr int64_t kEvVPrefetch1 = 3;  // V 步 tile 预取 pong（L1 槽 kL1VStep1）
constexpr int64_t kEvScale0 = 4;      // scale 槽 ping（vScale / qScale，随步预取侧）
constexpr int64_t kEvScale1 = 5;      // scale 槽 pong

// ---- 计数守恒纪律（engine 文件必须遵守；§6 + R.3.2）----
// 1. MODE4 为计数旗号：每流每旗号的 Set/Wait 次数逐任务 1:1 对账（生产/消费守恒）；
// 2. 结束路径：全流任务完成后 AIC 归还全部未消费旗号（与申请对称），再 PipeBarrier(PIPE_ALL) 收口；
// 3. 两流事件对独立（流 1 = 基号 + kStreamFlagStride = +8 偏移隔离），任何一侧禁止等待对侧的旗号。

#if __has_include("kernel_operator.h")

// Notify*：发送方在“生产该数据的管道”上全部操作完成后再置旗号（管道序保证数据可见）。
// __aicore__ 标注为设备构建必需（-xcce 混编模式下无标注函数视为 __host__，无法调用 AscendC
// 设备 API；CANN 9.2.0-beta.1 dav-c310 实测）。
__aicore__ inline void NotifyFix(int64_t stream, Sa2CrossFlag flag) {
  AscendC::CrossCoreSetFlag<4, PIPE_FIX>(Sa2FlagId(stream, flag));  // AIC：fixpipe L0C→UB 退休后（SCORE_OK/PV_OK）
}

__aicore__ inline void NotifyV(int64_t stream, Sa2CrossFlag flag) {
  AscendC::CrossCoreSetFlag<4, PIPE_V>(Sa2FlagId(stream, flag));  // AIV：Vector 消费完成后（QK_FREE/PV_FREE）
}

__aicore__ inline void NotifyMte3(int64_t stream, Sa2CrossFlag flag) {
  AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(Sa2FlagId(stream, flag));  // AIV：UB→L1 P 搬运退休后（P_TAKEN）
}

// Await*：接收方在“即将消费该数据的管道”上先等旗号再操作。
__aicore__ inline void AwaitV(int64_t stream, Sa2CrossFlag flag) {
  AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(Sa2FlagId(stream, flag));  // AIC：QK_FREE/PV_FREE → 随后 FIX 重填槽位
}

__aicore__ inline void AwaitMte1(int64_t stream, Sa2CrossFlag flag) {
  AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(Sa2FlagId(stream, flag));  // AIC：P_TAKEN → 随后 MTE1 取 P 入 L0A
}

__aicore__ inline void AwaitFix(int64_t stream, Sa2CrossFlag flag) {
  AscendC::CrossCoreWaitFlag<4, PIPE_V>(Sa2FlagId(stream, flag));  // AIV：SCORE_OK/PV_OK → 随后 V 侧消费
}

// ---- M2c-B3 K-scale 双缓冲预取与事件映射（Scale 预取与本地事件协同）----
// 将预留的本地 HardEvent 票号 kEvScale0 (ping) 与 kEvScale1 (pong)
// 正式建立 MTE2 异步加载与 Vector 核心消费的流水化协同映射：
// 1. MTE2 预取加载 Scale 到 slot(parity) 前，若 step>=2 先等待 Vector 消费完毕 (V_MTE2)；
// 2. MTE2 完成异步 DataCopy 后，通过 NotifyScaleLoaded(parity) 放行 Vector 核心 (MTE2_V)；
// 3. Vector 核心在计算 Softmax 前通过 AwaitScaleLoaded(parity) 等待 Scale 就绪 (MTE2_V)；
// 4. Vector 核心消费完毕后通过 NotifyScaleConsumed(parity) 释放该 slot (V_MTE2)。
__aicore__ inline void NotifyScaleLoaded(int64_t parity) {
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(static_cast<uint16_t>(parity == 0 ? kEvScale0 : kEvScale1));
}

__aicore__ inline void AwaitScaleLoaded(int64_t parity) {
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<uint16_t>(parity == 0 ? kEvScale0 : kEvScale1));
}

__aicore__ inline void NotifyScaleConsumed(int64_t parity) {
  AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(static_cast<uint16_t>(parity == 0 ? kEvScale0 : kEvScale1));
}

__aicore__ inline void AwaitScaleConsumed(int64_t parity) {
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(static_cast<uint16_t>(parity == 0 ? kEvScale0 : kEvScale1));
}

#else

inline void NotifyFix(int64_t, Sa2CrossFlag) {}
inline void NotifyV(int64_t, Sa2CrossFlag) {}
inline void NotifyMte3(int64_t, Sa2CrossFlag) {}
inline void AwaitV(int64_t, Sa2CrossFlag) {}
inline void AwaitMte1(int64_t, Sa2CrossFlag) {}
inline void AwaitFix(int64_t, Sa2CrossFlag) {}
inline void NotifyScaleLoaded(int64_t) {}
inline void AwaitScaleLoaded(int64_t) {}
inline void NotifyScaleConsumed(int64_t) {}
inline void AwaitScaleConsumed(int64_t) {}

#endif  // __has_include("kernel_operator.h")

// =============================================================================
// M2.7.4 类型状态（Typestate）安全封装（RAII Async Guards）
// 吸收 Rust/tile-rs 的 Typestate 设计思想：
// 将硬件异构流水线阶段（DMA Pending / Cross-Engine Pending）编码进 C++ 类型系统。
// 处于 Pending 状态的句柄没有任何数据读取/计算方法（operator[]/Data() 均不可用）；
// 必须通过具有右值引用限定的 Sync() && 或 Await() && 移动析构，才能产出可计算的 ReadyTensor，
// 从编译期物理杜绝 RAW（Read-After-Write）数据冒险与竞态隐患！
// =============================================================================

template <typename T>
class ReadyTensor {
public:
    __aicore__ inline ReadyTensor() : ptr_(nullptr), size_(0) {}
    __aicore__ inline explicit ReadyTensor(T* ptr, int64_t size = 0) : ptr_(ptr), size_(size) {}

    // Move-only 语义：严禁拷贝，确保就绪张量单一所有权与线性类型状态
    ReadyTensor(const ReadyTensor&) = delete;
    ReadyTensor& operator=(const ReadyTensor&) = delete;
    __aicore__ inline ReadyTensor(ReadyTensor&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    __aicore__ inline ReadyTensor& operator=(ReadyTensor&& other) noexcept {
        if (this != &other) {
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    __aicore__ inline T* Data() const { return ptr_; }
    __aicore__ inline T& operator[](int64_t idx) const { return ptr_[idx]; }
    __aicore__ inline int64_t Size() const { return size_; }
    __aicore__ inline bool IsValid() const { return ptr_ != nullptr; }

private:
    T* ptr_{nullptr};
    int64_t size_{0};
};

template <>
class ReadyTensor<void> {
public:
    __aicore__ inline ReadyTensor() = default;
    __aicore__ inline explicit ReadyTensor(void*, int64_t = 0) {}
    ReadyTensor(const ReadyTensor&) = delete;
    ReadyTensor& operator=(const ReadyTensor&) = delete;
    ReadyTensor(ReadyTensor&&) noexcept = default;
    ReadyTensor& operator=(ReadyTensor&&) noexcept = default;
    __aicore__ inline bool IsValid() const { return true; }
};

template <typename T>
class DmaPending {
public:
    // Move-only 语义：严禁拷贝，防止未同步句柄被复制逃逸
    DmaPending(const DmaPending&) = delete;
    DmaPending& operator=(const DmaPending&) = delete;

    __aicore__ inline explicit DmaPending(T* ptr = nullptr, int64_t size = 0)
        : ptr_(ptr), size_(size) {}

    __aicore__ inline DmaPending(DmaPending&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    __aicore__ inline DmaPending& operator=(DmaPending&& other) noexcept {
        if (this != &other) {
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    __aicore__ inline ~DmaPending() = default;

    // 类型状态跃迁：必须通过右值引用限定符 (&&) 显式调用 Sync() 消费自身
    // 插入对应管道的流水线同步，产出可计算的 ReadyTensor
    template <int32_t PipeType = 0>
    __aicore__ inline ReadyTensor<T> Sync() && {
        AscendC::PipeBarrier<PipeType>();
        T* p = ptr_;
        int64_t s = size_;
        ptr_ = nullptr;
        size_ = 0;
        return ReadyTensor<T>(p, s);
    }

private:
    T* ptr_{nullptr};
    int64_t size_{0};
};

template <typename T, Sa2CrossFlag Flag>
class CrossEnginePending {
public:
    CrossEnginePending(const CrossEnginePending&) = delete;
    CrossEnginePending& operator=(const CrossEnginePending&) = delete;

    __aicore__ inline explicit CrossEnginePending(T* ptr = nullptr, int64_t size = 0, int64_t streamId = 0)
        : ptr_(ptr), size_(size), streamId_(streamId) {}

    __aicore__ inline CrossEnginePending(CrossEnginePending&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_), streamId_(other.streamId_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    __aicore__ inline CrossEnginePending& operator=(CrossEnginePending&& other) noexcept {
        if (this != &other) {
            ptr_ = other.ptr_;
            size_ = other.size_;
            streamId_ = other.streamId_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // 跨核状态跃迁：消费自身，等待对应跨核硬件事件完成，产出 ReadyTensor
    __aicore__ inline ReadyTensor<T> Await() && {
        if constexpr (Flag == Sa2CrossFlag::kScoreOk || Flag == Sa2CrossFlag::kPvOk) {
            AwaitFix(streamId_, Flag);
        } else if constexpr (Flag == Sa2CrossFlag::kQkFree || Flag == Sa2CrossFlag::kPvFree) {
            AwaitV(streamId_, Flag);
        } else if constexpr (Flag == Sa2CrossFlag::kPTaken) {
            AwaitMte1(streamId_, Flag);
        }
        T* p = ptr_;
        int64_t s = size_;
        ptr_ = nullptr;
        size_ = 0;
        return ReadyTensor<T>(p, s);
    }

private:
    T* ptr_{nullptr};
    int64_t size_{0};
    int64_t streamId_{0};
};

// =============================================================================
// M2.7.4 状态类型化槽位（StateTypedSlot，迭代 9 落地）：把「未同步先读写」封死在编译期。
//
// 状态环：Empty --AcquireSlot<Flag>（内部 Await 事件）--> Full --ReleaseSlot<Flag>（内部
// Notify 事件）--> Empty。与上方 CrossEnginePending 的分工：后者只覆盖消费侧半程（Pending →
// Ready），释放侧 Notify 仍是散置自由函数；本封装把生产/消费两个方向都纳入线性类型。
//
// 编译期约束（负向能力表；真编译失败形态 = 对 Empty 调 SlotData / 默认或裸指针构造
// SlotRef<Full>，分别为 no matching function / private ctor 硬错误，SFINAE 可检测形态
// 见 conformance 第 15 用例 state_typed_slot_typestate_guard）：
//   1. 数据访问原语 SlotData<T> 只对 SlotRef<Full> 声明（Empty 态无对应重载）——持未同步槽
//      拿不到任何数据访问函数，收敛旧式「裸指针访问器整体绕过」访问面（todo §一.3 D-R12/P1-5）；
//   2. SlotRef<Full> 构造全部私有：默认/拷贝/裸指针构造不可达，Full 只能由 AcquireSlot 产出；
//   3. 两态均 Move-only（拷贝 = delete）；ReleaseSlot 以右值引用消费 Full，被释放句柄指针
//      擦除（IsValid()==false），杜绝 moved-from 状态混乱；
//   4. 状态迁移仅 AcquireSlot / ReleaseSlot 两个函数：内部事件 = 既有 Await*/Notify* 原样复用
//      （SlotAwaitPipeOf / SlotNotifyPipeOf 双映射表 static_assert 冻结），旗号预算与 1:1 守恒
//      不变（kCrossFlagBudget = 12、本地票 6 张均不动；每槽周期仍 1 Await + 1 Notify）。
//
// 槽周期旗号配对（ping-pong 双槽对，SlotReleaseFlagOf 冻结）：kScoreOk↔kQkFree（score 槽）、
// kPvOk↔kPvFree（PV 槽）；kQ8Ok / kPTaken 为任务级单向信用旗（生产者即通知方，无接收方释放
// 对），保持散置写法待 M2b 槽化。访问器约定：槽位使用处一律经 SlotData<T>(ref) 取指针（可在
// 使用处再包具名访问器），禁止绕过状态态自取裸地址。
// =============================================================================
struct Empty {};
struct Full {};

namespace detail {
// 共同存储核：仅承载槽身份（地址/字节数/流号），不提供任何数据访问。
struct SlotCore {
    SlotCore() = default;
    __aicore__ inline SlotCore(uint8_t *addr, int64_t bytes, int64_t stream)
        : addr(addr), bytes(bytes), stream(stream) {}
    SlotCore(const SlotCore &) = delete;
    SlotCore &operator=(const SlotCore &) = delete;
    __aicore__ inline SlotCore(SlotCore &&other) noexcept
        : addr(other.addr), bytes(other.bytes), stream(other.stream)
    {
        other.addr = nullptr;
        other.bytes = 0;
    }
    __aicore__ inline SlotCore &operator=(SlotCore &&other) noexcept
    {
        if (this != &other) {
            addr = other.addr;
            bytes = other.bytes;
            stream = other.stream;
            other.addr = nullptr;
            other.bytes = 0;
        }
        return *this;
    }
    uint8_t *addr{nullptr};
    int64_t bytes{0};
    int64_t stream{0};

    // 内省（非数据访问）：全状态可用，供持有方与测试核对槽身份；数据访问仍仅 SlotData@Full。
    __aicore__ inline bool IsValid() const { return addr != nullptr; }
    __aicore__ inline int64_t SizeBytes() const { return bytes; }
    __aicore__ inline int64_t StreamId() const { return stream; }
};
}  // namespace detail

// 主模板 = Empty（未同步）态：可默认/显式构造（空槽不授予任何能力），无任何数据访问成员。
template <typename StateTag>
class SlotRef : public detail::SlotCore {
public:
    SlotRef() = default;
    __aicore__ inline explicit SlotRef(uint8_t *addr, int64_t bytes, int64_t stream)
        : detail::SlotCore(addr, bytes, stream) {}
    SlotRef(const SlotRef &) = delete;
    SlotRef &operator=(const SlotRef &) = delete;
    __aicore__ inline SlotRef(SlotRef &&) noexcept = default;
    __aicore__ inline SlotRef &operator=(SlotRef &&) noexcept = default;
};

// Full 特化 = 已同步态：构造全部私有（仅 AcquireSlot 可产出），数据访问仅经 friend SlotData。
template <>
class SlotRef<Full> : public detail::SlotCore {
public:
    SlotRef(const SlotRef &) = delete;
    SlotRef &operator=(const SlotRef &) = delete;
    __aicore__ inline SlotRef(SlotRef &&) noexcept = default;
    __aicore__ inline SlotRef &operator=(SlotRef &&) noexcept = default;

private:
    SlotRef() = default;  // 无效 Full 不可外部构造（is_default_constructible == false）
    __aicore__ inline SlotRef(uint8_t *addr, int64_t bytes, int64_t stream)
        : detail::SlotCore(addr, bytes, stream) {}

    template <typename T>
    friend __aicore__ inline T *SlotData(const SlotRef<Full> &ref);
    template <Sa2CrossFlag kFlag>
    friend __aicore__ inline SlotRef<Full> AcquireSlot(void *addr, int64_t bytes, int64_t stream);
    template <Sa2CrossFlag kFlag>
    friend __aicore__ inline SlotRef<Empty> ReleaseSlot(SlotRef<Full> &&ref);
};

// 唯一数据访问原语：只对 SlotRef<Full> 声明；SlotRef<Empty> 无对应重载 ⇒ 「未同步先读写」
// 为编译期硬错误（no matching function）。
template <typename T>
__aicore__ inline T *SlotData(const SlotRef<Full> &ref)
{
    return reinterpret_cast<T *>(ref.addr);
}

// 旗号 → 接收方 Await 事件管道映射表（0 = AwaitFix，1 = AwaitV，2 = AwaitMte1）。
// 与既有散置调用点逐一等同（审计表 C.4 配对），无新增旗号。__aicore__ 标注同 Sa2FlagId：
// 仅设备构建需要，static_assert 走常量求值不受影响。
__aicore__ constexpr int SlotAwaitPipeOf(Sa2CrossFlag flag)
{
    return (flag == Sa2CrossFlag::kScoreOk || flag == Sa2CrossFlag::kPvOk) ? 0
           : (flag == Sa2CrossFlag::kQkFree || flag == Sa2CrossFlag::kPvFree) ? 1
           : 2;  // kPTaken / kQ8Ok：AIC 消费侧在 MTE1 域等
}

// 旗号 → 发送方 Notify 事件管道映射表（0 = NotifyFix，1 = NotifyV，2 = NotifyMte3）。
__aicore__ constexpr int SlotNotifyPipeOf(Sa2CrossFlag flag)
{
    return (flag == Sa2CrossFlag::kScoreOk || flag == Sa2CrossFlag::kPvOk) ? 0
           : (flag == Sa2CrossFlag::kPTaken) ? 2
           : 1;  // kQkFree / kPvFree / kQ8Ok：AIV 生产侧在 V 域放行
}

static_assert(SlotAwaitPipeOf(Sa2CrossFlag::kScoreOk) == 0 && SlotAwaitPipeOf(Sa2CrossFlag::kPvOk) == 0,
              "SCORE_OK/PV_OK 消费侧 = AwaitFix（C.4：FIX 产 V 收）");
static_assert(SlotAwaitPipeOf(Sa2CrossFlag::kQkFree) == 1 && SlotAwaitPipeOf(Sa2CrossFlag::kPvFree) == 1,
              "QK_FREE/PV_FREE 消费侧 = AwaitV（V 产 FIX 收）");
static_assert(SlotAwaitPipeOf(Sa2CrossFlag::kPTaken) == 2 && SlotAwaitPipeOf(Sa2CrossFlag::kQ8Ok) == 2,
              "P_TAKEN/Q8_OK 消费侧 = AwaitMte1（MTE3/V 产 MTE1 收）");
static_assert(SlotNotifyPipeOf(Sa2CrossFlag::kScoreOk) == 0 && SlotNotifyPipeOf(Sa2CrossFlag::kPvOk) == 0,
              "SCORE_OK/PV_OK 生产侧 = NotifyFix（FIX 产 FIX 旗）");
static_assert(SlotNotifyPipeOf(Sa2CrossFlag::kQkFree) == 1 && SlotNotifyPipeOf(Sa2CrossFlag::kPvFree) == 1 &&
                  SlotNotifyPipeOf(Sa2CrossFlag::kQ8Ok) == 1,
              "QK_FREE/PV_FREE/Q8_OK 生产侧 = NotifyV（V 产 V 旗）");
static_assert(SlotNotifyPipeOf(Sa2CrossFlag::kPTaken) == 2, "P_TAKEN 生产侧 = NotifyMte3（MTE3 产旗）");

// 槽周期旗号配对（唯一陈述；每周期 1 Await + 1 Notify，与散置写法 1:1 守恒完全一致）。
__aicore__ constexpr Sa2CrossFlag SlotReleaseFlagOf(Sa2CrossFlag acquire)
{
    return acquire == Sa2CrossFlag::kScoreOk ? Sa2CrossFlag::kQkFree
           : acquire == Sa2CrossFlag::kPvOk  ? Sa2CrossFlag::kPvFree
           : acquire;  // 单向信用旗（kQ8Ok/kPTaken）无接收方释放对，M2b 槽化时再定
}
static_assert(SlotReleaseFlagOf(Sa2CrossFlag::kScoreOk) == Sa2CrossFlag::kQkFree &&
                  SlotReleaseFlagOf(Sa2CrossFlag::kPvOk) == Sa2CrossFlag::kPvFree,
              "ping-pong 双槽配对冻结");

// AcquireSlot：Empty → Full 的唯一入口（内部执行接收方 Await 事件，与散置 Await* 全等）。
template <Sa2CrossFlag kFlag>
__aicore__ inline SlotRef<Full> AcquireSlot(void *addr, int64_t bytes, int64_t stream)
{
    constexpr int kPipe = SlotAwaitPipeOf(kFlag);
    if constexpr (kPipe == 0) {
        AwaitFix(stream, kFlag);
    } else if constexpr (kPipe == 1) {
        AwaitV(stream, kFlag);
    } else {
        AwaitMte1(stream, kFlag);
    }
    return SlotRef<Full>(static_cast<uint8_t *>(addr), bytes, stream);
}

// ReleaseSlot 内部事件分发（发送方 Notify，管道域由 SlotNotifyPipeOf 冻结表决定）。
template <Sa2CrossFlag kFlag>
__aicore__ inline void ReleaseSlotEvent(int64_t stream)
{
    constexpr int kPipe = SlotNotifyPipeOf(kFlag);
    if constexpr (kPipe == 0) {
        NotifyFix(stream, kFlag);
    } else if constexpr (kPipe == 2) {
        NotifyMte3(stream, kFlag);
    } else {
        NotifyV(stream, kFlag);
    }
}

// ReleaseSlot：Full → Empty 的唯一出口（内部执行发送方 Notify 事件；右值消费 + 句柄擦除）。
template <Sa2CrossFlag kFlag>
__aicore__ inline SlotRef<Empty> ReleaseSlot(SlotRef<Full> &&ref)
{
    ReleaseSlotEvent<kFlag>(ref.stream);
    SlotRef<Empty> done(ref.addr, ref.bytes, ref.stream);
    ref.addr = nullptr;  // 已释放句柄擦除（纵深防御；线性类型已杜绝再访问）
    ref.bytes = 0;
    return done;
}

// 类型状态层零新增旗号：上方双映射表覆盖且仅覆盖 Sa2CrossFlag 六枚举，预算断言不受影响。
static_assert(kCrossFlagBudget == 12 && kLocalEventBudget == 6, "StateTypedSlot 不改变旗号预算");

}  // namespace sa2

#endif  // SA2_EVENTS_H_
