// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.

#include <array>
#include <barrier>
#include <future>
#include <iostream>
#include <vector>
#include "../../../../op_kernel/arch35/common/mega_moe_input_sync.h"

using namespace MegaMoeImpl;
using namespace std::chrono_literals;

namespace {
constexpr uint32_t CORES = 4;
constexpr uint32_t RANKS = 2;
constexpr int32_t CANARY = 0x34567812;
struct alignas(64) Window {
    std::array<int32_t, 60 * 1024 / sizeof(int32_t)> words;
    Window()
    {
        words.fill(CANARY);
    }
    GM_ADDR Base()
    {
        return reinterpret_cast<GM_ADDR>(words.data());
    }
    void Seed(uint32_t value)
    {
        for (uint32_t core = 0; core < CORES; ++core) {
            *GetMteSyncCountAddress(Base(), core) = static_cast<int32_t>(value);
        }
    }
    void CheckCanaries()
    {
        for (uint32_t i = 0; i < words.size(); ++i) {
            uint32_t offset = i * sizeof(int32_t);
            bool counter = offset >= MTE_SYNC_COUNT_OFFSET &&
                           offset < MTE_SYNC_COUNT_OFFSET + CORES * MTE_SYNC_COUNT_STRIDE &&
                           (offset - MTE_SYNC_COUNT_OFFSET) % MTE_SYNC_COUNT_STRIDE == 0;
            assert(counter || words[i] == CANARY);
        }
    }
};

void SelectOwner(Window &window, uint32_t core)
{
    AscendC::ownedSequence = GetMteSyncCountAddress(window.Base(), core);
}

void TestDelayedProducer()
{
    Window window;
    window.Seed(0);
    for (uint32_t core = 0; core < CORES - 1; ++core) {
        SelectOwner(window, core);
        PublishMteQuantReady(window.Base(), core);
    }
    AscendC::watchedRead = false;
    AscendC::watchedAddress = GetMteSyncCountAddress(window.Base(), CORES - 1);
    std::atomic<bool> returned{false};
    auto sender = std::async(std::launch::async, [&] {
        WaitForMteQuantReady(window.Base(), 0, CORES, 1);
        returned = true;
    });
    auto deadline = std::chrono::steady_clock::now() + 2s;
    while (!AscendC::watchedRead && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    assert(AscendC::watchedRead && !returned);
    // Even an empty-token / no-route owner must publish to release the sender.
    SelectOwner(window, CORES - 1);
    PublishMteQuantReady(window.Base(), CORES - 1);
    assert(sender.wait_for(2s) == std::future_status::ready);
    sender.get();
    AscendC::watchedAddress = nullptr;
    window.CheckCanaries();
}

void TestRepeatedLaunches(uint32_t seed, uint32_t senders)
{
    std::array<Window, RANKS> windows;
    std::array<std::array<std::atomic<uint32_t>, CORES>, RANKS> quantVersions{};
    std::array<std::array<std::array<std::atomic<uint32_t>, CORES>, RANKS>, RANKS> receivedCounts{};
    std::array<std::barrier<>, RANKS> localBarrier{std::barrier<>(CORES), std::barrier<>(CORES)};
    std::barrier<> outputBarrier(RANKS * CORES);
    constexpr uint32_t LAUNCHES = 260; // several 8-bit epoch wraps
    for (auto &window : windows) {
        window.Seed(seed);
    }
    std::vector<std::thread> workers;
    for (uint32_t rank = 0; rank < RANKS; ++rank) {
        for (uint32_t core = 0; core < CORES; ++core) {
            workers.emplace_back([&, rank, core] {
                auto &window = windows[rank];
                SelectOwner(window, core);
                for (uint32_t launch = 1; launch <= LAUNCHES; ++launch) {
                    // Skew quant completion; core 3 owns no routes when senders=1.
                    for (uint32_t delay = 0; delay < (launch + core * 7 + rank * 11) % 23; ++delay) {
                        std::this_thread::yield();
                    }
                    // Empty producers still advance their sequence; nonempty writes
                    // become visible only when the production helper drains MTE3.
                    bool hasTokens = (launch + core) % 3 != 0;
                    if (hasTokens) {
                        AscendC::pendingPayload = &quantVersions[rank][core];
                        AscendC::pendingVersion = launch;
                    }
                    PublishMteQuantReady(window.Base(), core);
                    int32_t inputSequence = AscendC::ReadGmByPassDCache(AscendC::ownedSequence);
                    assert(static_cast<uint32_t>(inputSequence) == seed + 2U * launch - 1U);
                    uint32_t epoch = GetMteCountEpoch(inputSequence);
                    if (core < senders) {
                        WaitForMteQuantReady(window.Base(), core, CORES, inputSequence);
                        for (uint32_t peer = 0; peer < CORES; ++peer) {
                            if ((launch + peer) % 3 != 0) {
                                assert(quantVersions[rank][peer] == launch);
                            }
                        }
                        uint32_t count = (launch + core) % 5 == 0 ? 0 : launch;
                        for (uint32_t destination = 0; destination < RANKS; ++destination) {
                            receivedCounts[destination][rank][core] = (epoch << 24U) | count;
                        }
                    }
                    // Existing input SyncAll: all local readers finish before a
                    // counter may be reused by the output phase.
                    localBarrier[rank].arrive_and_wait();
                    assert(GetMteCountEpoch(AscendC::ReadGmByPassDCache(GetMteSyncCountAddress(window.Base(), 0))) ==
                           epoch);
                    if (core % 2 == 1) {
                        for (uint32_t source = 0; source < RANKS; ++source) {
                            for (uint32_t sender = 0; sender < senders; ++sender) {
                                uint32_t word;
                                do {
                                    word = receivedCounts[rank][source][sender];
                                    std::this_thread::yield();
                                } while (word >> 24U != epoch);
                                uint32_t expected = (launch + sender) % 5 == 0 ? 0 : launch;
                                assert((word & 0xFFFFFFU) == expected);
                            }
                        }
                    }
                    localBarrier[rank].arrive_and_wait(); // end of MoE AIV work
                    outputBarrier.arrive_and_wait();      // preserved cross-rank output rendezvous
                    uint32_t outputSequence = static_cast<uint32_t>(inputSequence) + 1U;
                    AscendC::WriteGmByPassDCache(AscendC::ownedSequence, static_cast<int32_t>(outputSequence));
                    localBarrier[rank].arrive_and_wait(); // existing output AIV SyncAll
                }
            });
        }
    }
    for (auto &worker : workers) {
        worker.join();
    }
    for (auto &window : windows) {
        window.CheckCanaries();
    }
}
} // namespace

int main()
{
    if constexpr (g_coreType == AscendC::AIC) {
        Window window;
        PublishMteQuantReady(window.Base(), 0);
        WaitForMteQuantReady(window.Base(), 0, CORES, 1);
        for (int32_t word : window.words) {
            assert(word == CANARY);
        }
        std::cout << "AIC does not access quant-ready state\n";
    } else {
        TestDelayedProducer();
        for (uint32_t seed : {0U, 0x7FFFFFFEU, 0xFFFFFFFEU}) {
            TestRepeatedLaunches(seed, 1);
            TestRepeatedLaunches(seed, CORES);
        }
        std::cout << "PASS: delayed/empty producers, zero counts, 1560 repeated launches, epoch/signed/full wrap, "
                     "single-writer slots and canaries\n";
    }
}
