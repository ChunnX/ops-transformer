/*!
 * \file local_cke_dcpad_kernel.h
 * \brief DataCopyPad 方式读写 CCU Event(CKE 16bit) Kernel 核心逻辑
 *        走 DMA 搬运通路（DataCopyPad），与 Scalar 方式（WriteGmByPassDCache）对比。
 *        单个 CKE 仅 2B，不满足 DataCopy 32B 对齐要求，使用 DataCopyPad 支持非对齐搬运。
 */

#ifndef LOCAL_CKE_DCPAD_KERNEL_H
#define LOCAL_CKE_DCPAD_KERNEL_H

#include "kernel_operator.h"

namespace LocalCkeDcPadImpl {

using namespace AscendC;

enum RwMode : uint32_t {
    MODE_WRITE = 0,
    MODE_READ = 1,
    MODE_WRITE_THEN_READ = 2,
};

class LocalCkeDcPad {
public:
    __aicore__ inline LocalCkeDcPad(){};
    __aicore__ inline void Init(uint32_t mode, uint32_t enableWarmup, uint32_t loopCount,
                                uint64_t pattern, GM_ADDR ckeAddr, GM_ADDR timeAddr, TPipe *pipe)
    {
        mode_ = mode;
        warmupCount_ = enableWarmup;
        loopCount_ = loopCount == 0 ? 1 : loopCount;
        pattern_ = static_cast<uint16_t>(pattern & 0xFFFFU);
        aivId_ = GetBlockIdx();
        if (aivId_ > 0) { return; }
        ckeGm_.SetGlobalBuffer((__gm__ uint16_t *)ckeAddr);
        timeGm_.SetGlobalBuffer((__gm__ uint64_t *)timeAddr);
        pipe->InitBuffer(dataBuf_, 32);
        pipe->InitBuffer(readBuf_, 32);
        dataLocal_ = dataBuf_.Get<uint16_t>();
        readLocal_ = readBuf_.Get<uint16_t>();
    }
    __aicore__ inline void Process()
    {
        if (aivId_ > 0) { return; }

        dataLocal_.SetValue(0, pattern_);
        PipeBarrier<PIPE_ALL>();

        verifyResult_ = 0;
        uint64_t warmupCycles = 0;
        if (warmupCount_ != 0) {
            uint32_t savedLoop = loopCount_;
            loopCount_ = warmupCount_;
            uint64_t ws = GetSystemCycle();
            switch (mode_) {
                case MODE_WRITE:            WriteMode(); break;
                case MODE_READ:             ReadMode(); break;
                default:                    WriteThenReadMode(); break;
            }
            warmupCycles = GetSystemCycle() - ws;
            loopCount_ = savedLoop;
        }

        uint64_t startCycle = GetSystemCycle();
        switch (mode_) {
            case MODE_WRITE:            WriteMode(); break;
            case MODE_READ:             ReadMode(); break;
            default:                    WriteThenReadMode(); break;
        }
        uint64_t endCycle = GetSystemCycle();

        if (mode_ == MODE_WRITE_THEN_READ) {
            readBack_ = readLocal_.GetValue(0);
            if (readBack_ != pattern_) { verifyResult_ = 1; }
        }

        timeGm_.SetValue(0, startCycle);
        timeGm_.SetValue(1, endCycle);
        timeGm_.SetValue(2, static_cast<uint64_t>(verifyResult_));
        timeGm_.SetValue(4, static_cast<uint64_t>(readBack_));
        timeGm_.SetValue(5, warmupCycles);
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE,
            DcciDst::CACHELINE_OUT>(timeGm_);
    }
private:
    __aicore__ inline void WriteMode()
    {
        for (uint32_t i = 0; i < loopCount_; i++) {
            DataCopyPad(ckeGm_, dataLocal_,
                        DataCopyParams{1, static_cast<uint32_t>(sizeof(uint16_t)), 0, 0});
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void ReadMode()
    {
        for (uint32_t i = 0; i < loopCount_; i++) {
            DataCopyPad(readLocal_, ckeGm_,
                        DataCopyParams{1, static_cast<uint32_t>(sizeof(uint16_t)), 0, 0},
                        DataCopyPadParams{false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void WriteThenReadMode()
    {
        for (uint32_t i = 0; i < loopCount_; i++) {
            DataCopyPad(ckeGm_, dataLocal_,
                        DataCopyParams{1, static_cast<uint32_t>(sizeof(uint16_t)), 0, 0});
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(readLocal_, ckeGm_,
                        DataCopyParams{1, static_cast<uint32_t>(sizeof(uint16_t)), 0, 0},
                        DataCopyPadParams{false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
        }
    }
    uint32_t mode_{0};
    uint32_t warmupCount_{0};
    uint32_t loopCount_{1};
    uint32_t aivId_{0};
    uint32_t verifyResult_{0};
    uint16_t pattern_{0};
    uint16_t readBack_{0};
    GlobalTensor<uint16_t> ckeGm_;
    GlobalTensor<uint64_t> timeGm_;
    TBuf<TPosition::VECCALC> dataBuf_;
    TBuf<TPosition::VECCALC> readBuf_;
    LocalTensor<uint16_t> dataLocal_;
    LocalTensor<uint16_t> readLocal_;
};

} // namespace LocalCkeDcPadImpl

#endif // LOCAL_CKE_DCPAD_KERNEL_H
