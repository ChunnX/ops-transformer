// Host simulation of timer bookkeeping only; AscendC instruction semantics need device validation.
#include <cassert>
#include <cstdint>
#include <vector>

#define __aicore__
#define __gm__
#define GM_ADDR uint8_t *
#define MOE_DISPATCH_FULL_MESH_TIMER 1
namespace AscendC {
enum class CacheLine {
    SINGLE_CACHE_LINE
};
enum class DcciDst {
    CACHELINE_OUT
};
uint64_t now = 1000;
uint64_t GetSystemCycle()
{
    return now;
}
template <typename T>
struct GlobalTensor {
    T *data = nullptr;
    void SetGlobalBuffer(T *ptr)
    {
        data = ptr;
    }
    void SetValue(uint64_t index, T value)
    {
        data[index] = value;
    }
    GlobalTensor operator[](uint64_t index)
    {
        return {data + index};
    }
};
template <typename T, CacheLine, DcciDst>
void DataCacheCleanAndInvalid(GlobalTensor<T>)
{}
} // namespace AscendC
#include "../../../common/op_kernel/moe_full_mesh_timer.hpp"
using Timer = Mc2Kernel::MoeFullMeshCocTimer;

int main()
{
    // ABI v5: 512 B per core; 48 AIVs need A >= 50 (offset 640 + 128 + 48 * 512 = 25344 <= 50 * 512).
    std::vector<uint64_t> data(50 * 64, 0);
    Timer timer;
    auto init = [&]() { timer.Init(reinterpret_cast<uint8_t *>(data.data()), 50, 0, 48, 3, 40, 4608, 4096); };
    init();
    timer.Tik(Timer::INIT); // 1000
    AscendC::now = 1050;
    timer.Tok(Timer::INIT);
    timer.Tik(Timer::STAGE); // 1050
    timer.Tik(Timer::SEND);
    timer.Tik(Timer::COMM_TOTAL); // 1050, same send-phase entry
    timer.Tik(Timer::CNT_SEND);   // 1050
    timer.Tik(Timer::EXPID_COPY); // 1050
    AscendC::now = 1100;
    timer.Tok(Timer::EXPID_COPY);
    AscendC::now = 1150;
    timer.Tik(Timer::CNT_NOTIFY); // 1150, merged cnt segment
    AscendC::now = 1200;
    timer.Tok(Timer::CNT_NOTIFY);
    AscendC::now = 1250;
    timer.Tok(Timer::CNT_SEND);
    timer.Tik(Timer::PAYLOAD); // 1250
    AscendC::now = 1300;
    timer.Tok(Timer::PAYLOAD);
    AscendC::now = 1350;
    timer.Tok(Timer::SEND);      // Must not include the receive stage.
    timer.Tik(Timer::RECV_COPY); // 1350, wraps the receive loop.
    timer.Tik(Timer::RECV_WAIT); // 1350, wraps the cumsum flag polling.
    AscendC::now = 1400;
    timer.Tok(Timer::RECV_WAIT);
    AscendC::now = 1500;
    timer.Tok(Timer::RECV_COPY);
    timer.Tik(Timer::TAIL); // 1500, output drain span
    AscendC::now = 1550;
    timer.Tok(Timer::TAIL);
    AscendC::now = 1600;
    timer.Tok(Timer::COMM_TOTAL); // 1050-1600
    AscendC::now = 1650;
    timer.Tok(Timer::STAGE);
    timer.Flush();
    auto *record = data.data() + (640 + 128) / 8;
    assert(record[2] == 0 && record[3] == 0 && record[4] == 0); // wait counters retired
    assert(record[8] == 1050 && record[9] == 1650);
    assert(record[10] == 1050 && record[11] == 1350);
    assert(record[12] == 1050 && record[13] == 1600);
    assert(record[14] == 1500 && record[15] == 1550);
    assert(record[16] == 1000 && record[17] == 1050); // INIT
    assert(record[18] == 1050 && record[19] == 1250); // CNT_SEND
    assert(record[20] == 1250 && record[21] == 1300); // PAYLOAD
    assert(record[22] == 0 && record[23] == 0);       // CNT_CALC unused on send cores
    assert(record[24] == 1350 && record[25] == 1500); // RECV_COPY
    assert(record[26] == 1050 && record[27] == 1100); // EXPID_COPY
    assert(record[28] == 1150 && record[29] == 1200); // CNT_NOTIFY
    assert(record[30] == 0 && record[31] == 0);       // WAIT_DISPATCH unused on send cores
    assert(record[32] == 0 && record[33] == 0);       // CUMSUM_CAL unused on send cores
    assert(record[34] == 1350 && record[35] == 1400); // RECV_WAIT
    assert(record[36] == 0 && record[63] == 0);       // trailing slots zeroed

    // Reuse the object/output: no send or receive must not retain earlier events.
    init();
    AscendC::now = 3100;
    timer.Tik(Timer::STAGE);
    timer.Tik(Timer::RECV_COPY);
    AscendC::now = 3200;
    timer.Tok(Timer::RECV_COPY);
    timer.Tik(Timer::TAIL);
    AscendC::now = 3250;
    timer.Tok(Timer::TAIL);
    timer.Tok(Timer::STAGE);
    timer.Flush();
    assert(record[2] == 0 && record[3] == 0 && record[4] == 0);
    assert(record[8] == 3100 && record[9] == 3250);
    assert(record[10] == 0 && record[11] == 0 && record[12] == 0 && record[13] == 0);
    assert(record[14] == 3200 && record[15] == 3250);
    assert(record[16] == 0 && record[17] == 0 && record[18] == 0 && record[19] == 0);
    assert(record[20] == 0 && record[21] == 0);
    assert(record[24] == 3100 && record[25] == 3200);
    assert(record[26] == 0 && record[27] == 0 && record[34] == 0 && record[35] == 0);

    // Capacity boundaries and per-core isolation: preserve both output guards and index prefix.
    constexpr uint64_t sentinel = 0xDEADBEEFULL;
    for (uint64_t a = 0; a <= 100; ++a) {
        for (uint32_t cores : {1U, 24U, 48U, 64U}) {
            std::vector<uint64_t> memory(a * 64 + 32, sentinel);
            auto *out = reinterpret_cast<uint8_t *>(memory.data() + 16);
            uint64_t offset = (a * 12 + 127) / 128 * 128;
            for (uint32_t core = 0; core < cores; ++core) {
                Timer t;
                t.Init(out, a, core, cores, 0, cores - 1, 4608, 4096);
                t.Tik(Timer::STAGE);
                t.Tik(Timer::TAIL);
                t.Tok(Timer::TAIL);
                t.Tok(Timer::STAGE);
                t.Flush();
            }
            for (uint64_t i = 0; i < 16; ++i)
                assert(memory[i] == sentinel);
            for (uint64_t i = 16 + a * 64; i < memory.size(); ++i)
                assert(memory[i] == sentinel);
            for (uint64_t i = 16; i < 16 + offset / 8 && i < 16 + a * 64; ++i)
                assert(memory[i] == sentinel);
            if (offset + 128 <= a * 512) {
                bool fits = offset + 128 + cores * 512 <= a * 512;
                assert(memory[16 + offset / 8 + 2] == (fits ? 1U : 2U));
                if (fits) {
                    for (uint32_t core = 0; core < cores; ++core) {
                        auto *r = memory.data() + 16 + offset / 8 + 16 + core * 64;
                        assert(r[0] == Timer::MAGIC && r[1] == core);
                        assert(r[14] == AscendC::now && r[15] == AscendC::now);
                        assert(r[24] == 0 && r[25] == 0);
                    }
                }
            }
        }
    }
}
