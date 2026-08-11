/*!
 * \file local_xn_dc_multi_kernel.h
 * \brief DataCopy 方式聚合读写多个 CCU Variable(XN 64bit) Kernel 核心逻辑
 *        单个 XN 仅 8B，不满足 DataCopy 32B 最小要求。
 *        申请 xnNum(>=4) 个 XN， xnNum*8B >= 32B，满足 DataCopy 对齐要求。
 *        假设通过 HcommCcuVariableAlloc 申请的多个 XN 地址连续。
 */

#ifndef LOCAL_XN_DC_MULTI_KERNEL_H
#define LOCAL_XN_DC_MULTI_KERNEL_H

#include "kernel_operator.h"

namespace LocalXnDcMultiImpl {

using namespace AscendC;

enum RwMode : uint32_t {
    MODE_WRITE = 0,
    MODE_READ = 1,
    MODE_WRITE_THEN_READ = 2,
};

class LocalXnDcMulti {
public:
    __aicore__ inline LocalXnDcMulti(){};
    __aicore__ inline void Init(uint32_t mode, uint32_t enableWarmup, uint32_t loopCount,
                                uint64_t pattern, uint32_t xnNum,
                                GM_ADDR xnAddr, GM_ADDR timeAddr, TPipe *pipe)
    {
        mode_ = mode;
        warmupCount_ = enableWarmup;
        loopCount_ = loopCount == 0 ? 1 : loopCount;
        pattern_ = pattern;
        xnNum_ = xnNum;
        aivId_ = GetBlockIdx();
        if (aivId_ > 0) { return; }
        xnGm_.SetGlobalBuffer((__gm__ uint64_t *)xnAddr);
        timeGm_.SetGlobalBuffer((__gm__ uint64_t *)timeAddr);
        uint32_t bufBytes = xnNum * sizeof(uint64_t);
        if (bufBytes < 32) { bufBytes = 32; }
        pipe->InitBuffer(dataBuf_, bufBytes);
        pipe->InitBuffer(readBuf_, bufBytes);
        dataLocal_ = dataBuf_.Get<uint64_t>();
        readLocal_ = readBuf_.Get<uint64_t>();
    }
    __aicore__ inline void Process()
    {
        if (aivId_ > 0) { return; }

        for (uint32_t i = 0; i < xnNum_; i++) {
            dataLocal_.SetValue(i, pattern_);
        }
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
            for (uint32_t i = 0; i < xnNum_; i++) {
                if (readLocal_.GetValue(i) != pattern_) {
                    verifyResult_ = 1;
                    break;
                }
            }
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
            DataCopy(xnGm_, dataLocal_, xnNum_);
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void ReadMode()
    {
        for (uint32_t i = 0; i < loopCount_; i++) {
            DataCopy(readLocal_, xnGm_, xnNum_);
            PipeBarrier<PIPE_ALL>();
        }
    }
    __aicore__ inline void WriteThenReadMode()
    {
        for (uint32_t i = 0; i < loopCount_; i++) {
            DataCopy(xnGm_, dataLocal_, xnNum_);
            PipeBarrier<PIPE_ALL>();
            DataCopy(readLocal_, xnGm_, xnNum_);
            PipeBarrier<PIPE_ALL>();
        }
    }
    uint32_t mode_{0};
    uint32_t warmupCount_{0};
    uint32_t loopCount_{1};
    uint32_t aivId_{0};
    uint32_t verifyResult_{0};
    uint32_t xnNum_{4};
    uint64_t pattern_{0};
    uint64_t readBack_{0};
    GlobalTensor<uint64_t> xnGm_;
    GlobalTensor<uint64_t> timeGm_;
    TBuf<TPosition::VECCALC> dataBuf_;
    TBuf<TPosition::VECCALC> readBuf_;
    LocalTensor<uint64_t> dataLocal_;
    LocalTensor<uint64_t> readLocal_;
};

} // namespace LocalXnDcMultiImpl

#endif // LOCAL_XN_DC_MULTI_KERNEL_H
