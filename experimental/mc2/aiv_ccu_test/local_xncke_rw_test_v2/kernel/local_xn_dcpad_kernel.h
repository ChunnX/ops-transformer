/*!
 * \file local_xn_dcpad_kernel.h
 * \brief DataCopyPad 方式读写 CCU Variable(XN 64bit) Kernel 核心逻辑
 *        DataCopy 要求 32B 对齐，单个 XN 仅 8B，改用 DataCopyPad 支持非对齐搬运。
 *        仅测 XN（64bit），CKE 为 16bit 不适合 DataCopy。
 */

#ifndef LOCAL_XN_DCPAD_KERNEL_H
#define LOCAL_XN_DCPAD_KERNEL_H

#include "kernel_operator.h"

namespace LocalXnDcPadImpl {

using namespace AscendC;

enum RwMode : uint32_t {
    MODE_WRITE = 0,
    MODE_READ = 1,
    MODE_WRITE_THEN_READ = 2,
};

class LocalXnDcPad {
public:
    __aicore__ inline LocalXnDcPad(){};
    __aicore__ inline void Init(uint32_t mode, uint32_t enableWarmup, uint32_t loopCount,
                                uint64_t pattern, GM_ADDR xnAddr, GM_ADDR timeAddr, TPipe *pipe)
    {
        mode_ = mode;
        warmupCount_ = enableWarmup;
        loopCount_ = loopCount == 0 ? 1 : loopCount;
        pattern_ = pattern;
        aivId_ = GetBlockIdx();
        if (aivId_ > 0) { return; }
        xnGm_.SetGlobalBuffer((__gm__ uint64_t *)xnAddr);
        timeGm_.SetGlobalBuffer((__gm__ uint64_t *)timeAddr);
        pipe->InitBuffer(dataBuf_, 32);
        pipe->InitBuffer(readBuf_, 32);
        dataLocal_ = dataBuf_.Get<uint64_t>();
        readLocal_ = readBuf_.Get<uint64_t>();
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
        timeGm_.SetValue(3, readBack_);
        timeGm_.SetValue(5, warmupCycles);
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE,
            DcciDst::CACHELINE_OUT>(timeGm_);
    }
private:
    __aicore__ inline void WriteMode()
    {
        for (uint32_t i = 0; i < loopCount_; i++) {
            DataCopyPad(xnGm_, dataLocal_,
                        DataCopyParams{1, static_cast<uint32_t>(sizeof(uint64_t)), 0, 0});
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void ReadMode()
    {
        for (uint32_t i = 0; i < loopCount_; i++) {
            DataCopyPad(readLocal_, xnGm_,
                        DataCopyParams{1, static_cast<uint32_t>(sizeof(uint64_t)), 0, 0},
                        DataCopyPadParams{false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void WriteThenReadMode()
    {
        for (uint32_t i = 0; i < loopCount_; i++) {
            DataCopyPad(xnGm_, dataLocal_,
                        DataCopyParams{1, static_cast<uint32_t>(sizeof(uint64_t)), 0, 0});
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(readLocal_, xnGm_,
                        DataCopyParams{1, static_cast<uint32_t>(sizeof(uint64_t)), 0, 0},
                        DataCopyPadParams{false, 0, 0, 0});
            PipeBarrier<PIPE_ALL>();
        }
    }
    uint32_t mode_{0};
    uint32_t warmupCount_{0};
    uint32_t loopCount_{1};
    uint32_t aivId_{0};
    uint32_t verifyResult_{0};
    uint64_t pattern_{0};
    uint64_t readBack_{0};
    GlobalTensor<uint64_t> xnGm_;
    GlobalTensor<uint64_t> timeGm_;
    TBuf<TPosition::VECCALC> dataBuf_;
    TBuf<TPosition::VECCALC> readBuf_;
    LocalTensor<uint64_t> dataLocal_;
    LocalTensor<uint64_t> readLocal_;
};

} // namespace LocalXnDcPadImpl

#endif // LOCAL_XN_DCPAD_KERNEL_H
