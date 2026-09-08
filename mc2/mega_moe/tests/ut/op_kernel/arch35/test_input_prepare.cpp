/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include <algorithm>
#include <iostream>

namespace InputPrepareAic {
void Notify();
void Wait();
void ReleaseUb();
void Store(GM_ADDR, GM_ADDR, uint32_t, uint32_t, uint32_t, uint32_t, uint8_t *);
} // namespace InputPrepareAic
namespace InputPrepareAiv {
void Notify();
void Wait();
void ReleaseUb();
void Store(GM_ADDR, GM_ADDR, uint32_t, uint32_t, uint32_t, uint32_t, uint8_t *);
} // namespace InputPrepareAiv
using namespace InputPrepareMock;

void TestHandshake(uint32_t blocks, uint32_t round, bool shared)
{
    Machine sim(blocks);
    machine = &sim;
    std::vector<bool> ready(blocks * 2), released(blocks * 2), routeAllowed(blocks * 2);
    std::vector<bool> computed(blocks), written(blocks), vectorStarted(blocks * 2);
    bool inputAllowed = false;
    uint32_t slow = round % (blocks * 2);
    for (uint32_t cube = 0; cube < blocks; ++cube) {
        sim.current = cube;
        InputPrepareAic::Notify(); // Must not publish a vector notification.
        InputPrepareAic::Wait();
        InputPrepareAic::ReleaseUb();
        if (shared) {
            sim.Enqueue(PIPE_M, [&, cube]() {
                assert(std::all_of(ready.begin(), ready.end(), [](bool value) { return value; }));
                computed[cube] = true;
                return true;
            });
            sim.Enqueue(PIPE_FIX, [&, cube]() {
                if (!computed[cube]) {
                    return false;
                }
                assert(released[cube * 2] && released[cube * 2 + 1]);
                written[cube] = true;
                return true;
            });
        }
    }
    for (uint32_t vector = 0; vector < blocks * 2; ++vector) {
        sim.current = blocks + vector;
        // Empty-token workers still reset their workspace segment and publish exactly once.
        sim.Enqueue(PIPE_MTE3, [&, vector]() {
            if (vector == slow && !inputAllowed) {
                return false;
            }
            ready[vector] = true;
            return true;
        });
        InputPrepareAiv::Notify();
        sim.Enqueue(PIPE_MTE3, [&, vector]() {
            if (!routeAllowed[vector]) {
                return false;
            }
            released[vector] = true;
            return true;
        });
        // The retained AIV PIPE_ALL drains route/count DMA before UB release and scalar continuation.
        sim.Enqueue(PIPE_S, [&, vector]() { return released[vector]; });
        InputPrepareAiv::ReleaseUb();
        InputPrepareAiv::Wait();
        sim.Enqueue(PIPE_S, [&, vector]() {
            assert(std::all_of(ready.begin(), ready.end(), [](bool value) { return value; }));
            assert(released[vector]);
            vectorStarted[vector] = true;
            return true;
        });
    }
    sim.Run();
    assert(std::none_of(computed.begin(), computed.end(), [](bool value) { return value; }));
    inputAllowed = true;
    sim.Run();
    for (bool value : computed) {
        assert(value == shared); // Cube compute progresses while all route producers are blocked.
    }
    assert(std::none_of(written.begin(), written.end(), [](bool value) { return value; }));
    for (uint32_t index = 0; index < blocks * 2; ++index) {
        uint32_t vector = (index + slow) % (blocks * 2);
        routeAllowed[vector] = true;
        sim.Run();
        for (uint32_t cube = 0; cube < blocks; ++cube) {
            assert(written[cube] == (shared && released[cube * 2] && released[cube * 2 + 1]));
        }
    }
    assert(std::all_of(vectorStarted.begin(), vectorStarted.end(), [](bool value) { return value; }));
    sim.AssertDrained(); // Includes unused-tile/no-shared cases: no hardware event survives the launch.
}

void TestSharedStore(uint32_t hidden, uint32_t packed, bool prefetch)
{
    Machine sim(1);
    machine = &sim;
    sim.current = 1;
    uint32_t bytes = hidden / packed;
    uint32_t scaleOffset = (bytes + 255) / 256 * 256;
    uint32_t scaleCount = (hidden + 63) / 64 * 2;
    uint32_t recordBytes = scaleOffset + (scaleCount + 31) / 32 * 32 + (prefetch ? 32 : 0);
    constexpr uint32_t tokens = 7;
    constexpr uint32_t guard = 32;
    std::vector<uint8_t> data(guard * 2 + bytes * tokens, 0xa5);
    std::vector<uint8_t> scales(guard * 2 + scaleCount * tokens, 0xa5);
    std::vector<uint8_t> communication(recordBytes * tokens, 0);
    std::vector<uint8_t> expectedData = data, expectedScales = scales, expectedCommunication = communication;
    std::array<std::vector<uint8_t>, 2> ub{std::vector<uint8_t>(recordBytes), std::vector<uint8_t>(recordBytes)};
    for (uint32_t token = 0; token < tokens; ++token) {
        auto &buffer = ub[token % 2];
        // Complete both destinations before reusing a double-buffer slot.
        sim.Run();
        for (uint32_t index = 0; index < recordBytes; ++index) {
            buffer[index] = static_cast<uint8_t>(token * 37 + index);
        }
        if (hidden % 64 != 0) {
            buffer[scaleOffset + scaleCount - 1] = 0; // Preserve padded scale group.
        }
        std::copy_n(buffer.data(), bytes, expectedData.data() + guard + token * bytes);
        std::copy_n(buffer.data() + scaleOffset, scaleCount, expectedScales.data() + guard + token * scaleCount);
        std::copy(buffer.begin(), buffer.end(), expectedCommunication.begin() + token * recordBytes);
        AscendC::DataCopyPad<uint8_t>({communication.data() + token * recordBytes}, {buffer.data()},
                                      {1, recordBytes, 0, 0, 0});
        InputPrepareAiv::Store(data.data() + guard, scales.data() + guard, token, bytes, scaleCount, scaleOffset,
                               buffer.data());
    }
    sim.Run();
    assert(data == expectedData && scales == expectedScales && communication == expectedCommunication);
    sim.AssertDrained();
    sim.current = 0;
    InputPrepareAic::Store(nullptr, nullptr, 0, bytes, scaleCount, scaleOffset, nullptr);
    sim.AssertDrained();
}

int main()
{
    uint32_t handshakes = 0;
    for (uint32_t blocks : {1U, 2U, 8U}) {
        for (uint32_t round = 0; round < 256; ++round) {
            for (bool shared : {false, true}) {
                TestHandshake(blocks, round, shared);
                ++handshakes;
            }
        }
    }
    uint32_t layouts = 0;
    for (uint32_t hidden = 32; hidden <= 8192; hidden += 32) {
        for (uint32_t packed : {1U, 2U}) {
            for (bool prefetch : {false, true}) {
                TestSharedStore(hidden, packed, prefetch);
                ++layouts;
            }
        }
    }
    std::cout << handshakes << " hardware handshakes and " << layouts << " shared layouts passed (UBSan)\n";
}
