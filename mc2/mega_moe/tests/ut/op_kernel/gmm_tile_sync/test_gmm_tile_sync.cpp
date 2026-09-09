/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include "mega_moe_gmm_tile_sync.h"

using Sync = MegaMoeImpl::GmmTileSync;
namespace {
struct Counts {
    int ready = 0, ack = 0, sets = 0, waits = 0;
};
std::array<Counts, 2> counts;
std::mutex mu;
std::condition_variable cv;
thread_local bool producer = false;
thread_local int channel = 0;
bool prefilled = false;
Sync::Consumer gmm1Peer = Sync::Consumer::AIV1;
} // namespace
namespace AscendC {
void TestSet(uint8_t mode, pipe_t pipe, uint16_t id)
{
    assert(mode == 4);
    std::lock_guard<std::mutex> lock(mu);
    auto &c = counts[channel];
    if (producer) {
        assert(pipe == PIPE_FIX && id == 8 + 2 * channel + (channel == 0 ? static_cast<int>(gmm1Peer) : 16));
        ++c.ready;
    } else {
        assert(pipe == PIPE_MTE3 && id == 9 + 2 * channel);
        ++c.ack;
    }
    assert(c.ready <= Sync::MAX_INFLIGHT && c.ack <= Sync::MAX_INFLIGHT);
    ++c.sets;
    cv.notify_all();
}
void TestWait(uint8_t mode, pipe_t pipe, uint16_t id)
{
    assert(mode == 4);
    std::unique_lock<std::mutex> lock(mu);
    auto &c = counts[channel];
    if (producer) {
        assert(pipe == PIPE_S && id == 9 + 2 * channel + (channel == 0 ? static_cast<int>(gmm1Peer) : 16));
        assert(cv.wait_for(lock, std::chrono::seconds(5), [&] { return c.ack > 0; }));
        --c.ack;
    } else {
        assert(pipe == PIPE_MTE2 && id == 8 + 2 * channel);
        assert(cv.wait_for(lock, std::chrono::seconds(5), [&] { return c.ready > 0; }));
        --c.ready;
    }
    ++c.waits;
}
} // namespace AscendC

void Run(const std::vector<int> &schedule, bool delayProducer, bool fillBothFirst)
{
    counts = {};
    prefilled = false;
    std::thread p([&] {
        producer = true;
        Sync g1(Sync::Stage::GMM1, gmm1Peer), g2(Sync::Stage::GMM2, Sync::Consumer::AIV1);
        Sync *s[] = {&g1, &g2};
        for (int which : schedule) {
            channel = which;
            s[which]->Acquire();
            s[which]->PublishReady();
            if (delayProducer)
                std::this_thread::sleep_for(std::chrono::microseconds(5));
        }
        if (fillBothFirst) {
            std::lock_guard<std::mutex> lock(mu);
            assert(counts[0].ready == 15 && counts[1].ready == 15);
            prefilled = true;
            cv.notify_all();
        }
        channel = 0;
        g1.Drain();
        channel = 1;
        g2.Drain();
    });
    std::thread c([&] {
        producer = false;
        Sync g1(Sync::Stage::GMM1, gmm1Peer), g2(Sync::Stage::GMM2, Sync::Consumer::AIV1);
        Sync *s[] = {&g1, &g2};
        if (fillBothFirst) {
            std::unique_lock<std::mutex> lock(mu);
            assert(cv.wait_for(lock, std::chrono::seconds(5), [] { return prefilled; }));
        }
        auto consumeSchedule = schedule;
        if (fillBothFirst)
            std::reverse(consumeSchedule.begin(), consumeSchedule.end());
        for (int which : consumeSchedule) {
            channel = which;
            s[which]->WaitReady();
            if (!delayProducer)
                std::this_thread::sleep_for(std::chrono::microseconds(5));
            s[which]->Release();
        }
    });
    p.join();
    c.join();
    for (int ch = 0; ch < 2; ++ch) {
        const auto &c = counts[ch];
        assert(c.ready == 0 && c.ack == 0);
        assert(c.sets == c.waits && c.sets == 2 * std::count(schedule.begin(), schedule.end(), ch));
    }
}
int main()
{
    static_assert(Sync::MAX_INFLIGHT == 15);
    int runs = 0;
    for (auto peer : {Sync::Consumer::AIV0, Sync::Consumer::AIV1}) {
        gmm1Peer = peer;
        for (int n : {0, 1, 14, 15, 16, 31, 275, 1000}) {
            for (int pattern = 0; pattern < 3; ++pattern) {
                std::vector<int> schedule;
                for (int i = 0; i < n; ++i) {
                    if (pattern == 0)
                        schedule.push_back(0);
                    else if (pattern == 1)
                        schedule.push_back(1);
                    else {
                        schedule.push_back(0);
                        schedule.push_back(1);
                        schedule.push_back(1);
                    }
                }
                for (bool slow : {false, true}) {
                    Run(schedule, slow, false);
                    ++runs;
                }
            }
        }
        std::vector<int> full(15, 0);
        full.insert(full.end(), 15, 1);
        Run(full, false, true);
        ++runs;
    }
    std::cout << "Actual split helper: " << runs
              << " schedules passed; independent 15+15 ready notifications, exact IDs/pipes,"
                 " separate ACK accounting and final drains passed\n";
}
