/**
 * Verification test for pipeline_guard.h type-state protocol (RES-05).
 *
 * Host-only: the device branch maps to real AscendC primitives and is exercised
 * by the container bisheng gate; here we prove two things that cannot regress
 * silently:
 *  1. The protocol is enforced by the type system: methods of one pipeline state
 *     simply do not exist on the other state (SFINAE detection traits), so a
 *     forgotten Sync() is a compile error, not a runtime race.
 *  2. Tiles are move-only single-owner handles and the host op log records the
 *     expected operation ordering (load -> sync -> compute -> store).
 */

#include <cassert>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

#include "../sageattn_fp8/ascendc/pipeline_guard.h"

using sage_attn::pipeline::UbTile;
using sage_attn::pipeline::DmaPending;
using sage_attn::pipeline::ComputeReady;
using sage_attn::pipeline::LoadUb;
using sage_attn::pipeline::MakeScratchTile;
using sage_attn::pipeline::detail::OpLog;
using sage_attn::pipeline::detail::ClearOpLog;
using sage_attn::pipeline::OP_LOAD;
using sage_attn::pipeline::OP_SYNC;
using sage_attn::pipeline::OP_EXP;
using sage_attn::pipeline::OP_ADDS;
using sage_attn::pipeline::OP_MULS;
using sage_attn::pipeline::OP_REDUCE_MAX;
using sage_attn::pipeline::OP_REDUCE_SUM;
using sage_attn::pipeline::OP_STORE;
using sage_attn::pipeline::OP_SCRATCH;

using PendingTile = UbTile<float, DmaPending>;
using ReadyTile = UbTile<float, ComputeReady>;

// C++17 SFINAE 检测 trait：成员在目标类型（按给定引用类别）上是否可调用。
// 只探测存在性，不实际调用；未特化通道/错误状态下的成员不可名状即 false。
template <typename T, typename = void>
struct HasSync : std::false_type {};
template <typename T>
struct HasSync<T, std::void_t<decltype(std::declval<T &>().Sync())>> : std::true_type {};

template <typename T, typename = void>
struct HasExp : std::false_type {};
template <typename T>
struct HasExp<T, std::void_t<decltype(std::declval<T &>().Exp(128u))>> : std::true_type {};

template <typename T, typename = void>
struct HasAdds : std::false_type {};
template <typename T>
struct HasAdds<T, std::void_t<decltype(std::declval<T &>().Adds(0.0f, 128u))>> : std::true_type {};

template <typename T, typename = void>
struct HasMuls : std::false_type {};
template <typename T>
struct HasMuls<T, std::void_t<decltype(std::declval<T &>().Muls(1.0f, 128u))>> : std::true_type {};

// BeginStore 为 && 限定：右值（所有权转移）合法，左值不可调用
template <typename T, typename = void>
struct HasBeginStoreRvalue : std::false_type {};
template <typename T>
struct HasBeginStoreRvalue<T, std::void_t<decltype(std::declval<T>().BeginStore(nullptr, 128u))>>
    : std::true_type {};

template <typename T, typename = void>
struct HasBeginStoreLvalue : std::false_type {};
template <typename T>
struct HasBeginStoreLvalue<T, std::void_t<decltype(std::declval<T &>().BeginStore(nullptr, 128u))>>
    : std::true_type {};

void TestTypeStateProtocol() {
    std::cout << "[Test 1] Type-state protocol: methods exist only on their state..." << std::endl;

    // Sync 只存在于在途态；ComputeReady 无 Sync（重复同步在编译期即不可能）
    static_assert(HasSync<PendingTile>::value, "DmaPending must expose Sync()");
    static_assert(!HasSync<ReadyTile>::value, "ComputeReady must NOT expose Sync()");

    // 计算原语只存在于引擎可见态；在途态忘 Sync 时调用即编译错误
    static_assert(HasExp<ReadyTile>::value, "ComputeReady must expose Exp()");
    static_assert(!HasExp<PendingTile>::value,
                  "DmaPending must NOT expose Exp() -- forgot Sync() is a compile error");
    static_assert(!HasAdds<PendingTile>::value, "DmaPending must NOT expose Adds()");
    static_assert(!HasMuls<PendingTile>::value, "DmaPending must NOT expose Muls()");
    static_assert(!HasBeginStoreRvalue<PendingTile>::value,
                  "DmaPending must NOT expose BeginStore()");

    // 写回是右值限定动作：只能在显式转移所有权时调用（防止写回后继续计算）
    static_assert(HasBeginStoreRvalue<ReadyTile>::value,
                  "BeginStore must be callable on an rvalue (ownership transfer)");
    static_assert(!HasBeginStoreLvalue<ReadyTile>::value,
                  "BeginStore must NOT be callable on an lvalue");

    // 单 owner：禁止拷贝，允许移动
    static_assert(!std::is_copy_constructible<PendingTile>::value, "tiles must be move-only");
    static_assert(!std::is_copy_constructible<ReadyTile>::value, "tiles must be move-only");
    static_assert(std::is_move_constructible<PendingTile>::value, "tiles must be movable");
    static_assert(std::is_move_constructible<ReadyTile>::value, "tiles must be movable");

    std::cout << "[Test 1 Passed] type-state protocol enforced at compile time!" << std::endl;
}

void TestHostOpOrdering() {
    std::cout << "[Test 2] Host op-log ordering of the guarded pipeline..." << std::endl;

    ClearOpLog();

    // 标准链路：load -> sync -> 同引擎连续计算（零同步）-> 归约 -> 写回
    auto pending = LoadUb(/*tile_id=*/0, /*count=*/4096);
    auto ready = pending.Sync();
    ready.Exp(4096);
    ready.Adds(-2.0f, 4096);
    ready.Muls(448.0f, 4096);
    auto scratch_dst = MakeScratchTile(/*tile_id=*/1, 8);
    auto scratch_rwork = MakeScratchTile(/*tile_id=*/2, 8);
    (void)ready.ReduceMax(scratch_dst, scratch_rwork, 4096);
    (void)ready.ReduceSum(scratch_dst, scratch_rwork, 4096);
    std::move(ready).BeginStore(nullptr, 4096);

    const std::vector<uint32_t> expected = {
        OP_LOAD, OP_SYNC, OP_EXP, OP_ADDS, OP_MULS,
        OP_SCRATCH, OP_SCRATCH, OP_REDUCE_MAX, OP_REDUCE_SUM, OP_STORE,
    };
    assert(OpLog() == expected);

    // 协议反例（运行时序层）：同引擎连续计算之间不得出现任何同步点。
    // SYNC 记录只允许出现在第一次计算之前。
    bool sync_after_first_compute = false;
    bool seen_compute = false;
    for (uint32_t op : OpLog()) {
        if (op == OP_EXP || op == OP_ADDS || op == OP_MULS ||
            op == OP_REDUCE_MAX || op == OP_REDUCE_SUM) {
            seen_compute = true;
        }
        if (op == OP_SYNC && seen_compute) {
            sync_after_first_compute = true;
        }
    }
    assert(!sync_after_first_compute);

    std::cout << "[Test 2 Passed] guarded pipeline ordering verified!" << std::endl;
}

int main() {
    std::cout << "Starting pipeline_guard type-state protocol verification..." << std::endl;
    TestTypeStateProtocol();
    TestHostOpOrdering();
    std::cout << "All pipeline_guard tests PASSED successfully!" << std::endl;
    return 0;
}
