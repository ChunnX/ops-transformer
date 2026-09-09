// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
// Host-only instruction recorder. Never include this directory in device builds.
#pragma once
#include <cstdint>
#include <type_traits>
#include <vector>
#define __aicore__
namespace AscendC {
struct half { uint16_t bits; };
struct bfloat16_t { uint16_t bits; };
template<class A, class B> using IsSameType = std::is_same<A, B>;
enum class TPosition { A1, B1, A2, B2, CO1 };
enum pipe_t { PIPE_MTE1, PIPE_M };
template<class T> struct LocalTensor {
    TPosition position;
    uint64_t address;
    uint32_t size;
    LocalTensor(TPosition p, uint64_t a, uint32_t s) : position(p), address(a), size(s) {}
    LocalTensor operator[](uint64_t elements) const { return {position, address + elements * sizeof(T), size}; }
};
struct LoadData2DParamsV2 {
    uint32_t mStartPosition = 0, kStartPosition = 0, mStep = 0, kStep = 0, srcStride = 0, dstStride = 0;
    bool ifTranspose = false;
};
struct MmadParams {
    uint32_t m = 0, n = 0, k = 0;
    bool cmatrixInitVal = false, cmatrixSource = false;
    uint32_t unitFlag = 0;
};
struct Event {
    char kind;
    uint32_t pipe = 0, id = 0;
    TPosition position = TPosition::A1;
    uint64_t dst = 0, src = 0, b = 0;
    LoadData2DParamsV2 load;
    MmadParams mm;
};
inline std::vector<Event> events;
struct Mutex {
    template<pipe_t P> static void Lock(uint32_t id) { events.push_back({'L', P, id}); }
    template<pipe_t P> static void Unlock(uint32_t id) { events.push_back({'U', P, id}); }
};
template<class T> void LoadData(LocalTensor<T> dst, LocalTensor<T> src, LoadData2DParamsV2 p)
{
    Event e{'D'}; e.position = dst.position; e.dst = dst.address; e.src = src.address; e.load = p;
    events.push_back(e);
}
template<class C, class A, class B>
void Mmad(LocalTensor<C> c, LocalTensor<A> a, LocalTensor<B> b, MmadParams p)
{
    Event e{'M'}; e.dst = c.address; e.src = a.address; e.b = b.address; e.mm = p;
    events.push_back(e);
}
}
