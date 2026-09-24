
#ifndef DATA_COPY_EXAMPLE_H
#define DATA_COPY_EXAMPLE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
using namespace AscendC;

template <typename T>
class DataCopyExample {
public:
    __aicore__ inline DataCopyExample() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, const DataCopyExampleTilingData* tiling_data, TPipe* tmpPipe)
    {
        pipe = tmpPipe;
        curBlockIdx = GetBlockIdx();
        x_ = x;
        y_ = y;
        shapeT = tiling_data->shapeT;
        shapeN = tiling_data->shapeN;
        shapeD = tiling_data->shapeD;
        B = tiling_data->B;
        shapeTAlign = tiling_data->shapeTAlign;
        shapeNDAlign = tiling_data->shapeNDAlign;
        useCoreNum = tiling_data->useCoreNum;
        shapeNDLoop = (shapeN * shapeD + shapeNDAlign - 1) / shapeNDAlign;
        // 计算ND维的尾块大小
        lastShapeND = shapeN * shapeD - (shapeNDLoop - 1) * shapeNDAlign;

        inputXGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x), shapeT * shapeN * shapeD);
        inputYGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(y), B+1);

        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(z), shapeT * shapeN * shapeD);

        useUB1 = true;

        // 初始化 Ping-Pong Buffer 的事件 ID
        for (int i = 0; i < 2; i++) {
            eventIdMTE2_MTE3[i] = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_MTE3>());
            eventIdMTE3_MTE2[i] = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_MTE2>());
        }

        eventIdV_MTE3 = static_cast<event_t>(pipe->AllocEventID<HardEvent::V_MTE3>());
        eventIdMTE3_V = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_V>());
        eventIdS_V = static_cast<event_t>(pipe->AllocEventID<HardEvent::S_V>());

        // 打印验证
        // if(curBlockIdx == 1) {
        //     AscendC::printf("<<<<<<< 验证 kernel Init >>>>>>>\n");
        //     AscendC::printf("验证 shapeT %d\n", shapeT);
        //     AscendC::printf("验证 shapeN %d\n", shapeN);
        //     AscendC::printf("验证 shapeD %d\n", shapeD);
        //     AscendC::printf("验证 B %d\n", B);
        //     AscendC::printf("验证 GetBlockNum() %d\n", GetBlockNum());
        //     AscendC::printf("验证 curBlockIdx %d \n", curBlockIdx);
        //     AscendC::printf("验证 useCoreNum %d\n", useCoreNum);
        //     AscendC::printf("验证 shapeNDLoop %d\n", shapeNDLoop);
        //     AscendC::printf("验证 lastShapeND %d\n", lastShapeND);
        // }

    }

    __aicore__ inline void InitBuffer()
    {
        pipe->InitBuffer(inputXUb[0], shapeTAlign * shapeNDAlign * sizeof(T));
        pipe->InitBuffer(inputXUb[1], shapeTAlign * shapeNDAlign * sizeof(T));
    }

    __aicore__ inline void GetLocalTensor()
    {
        inputXLocal[0] = inputXUb[0].Get<T>();
        inputXLocal[1] = inputXUb[1].Get<T>();
    }

    __aicore__ inline void ReleaseEventID()
    {
        for (int i = 0; i < 2; i++) {
            pipe->ReleaseEventID<HardEvent::MTE2_MTE3>(eventIdMTE2_MTE3[i]); // mte3搬出等待mte2搬入完成
            pipe->ReleaseEventID<HardEvent::MTE3_MTE2>(eventIdMTE3_MTE2[i]); // mte2搬入等待mte3搬出完成
        }
        
        pipe->ReleaseEventID<HardEvent::V_MTE3>(eventIdV_MTE3); // mte3搬出等待vec计算完成
        pipe->ReleaseEventID<HardEvent::MTE3_V>(eventIdMTE3_V);
        pipe->ReleaseEventID<HardEvent::S_V>(eventIdS_V);
        
    }

    __aicore__ inline void Process()
    {
        // 记录两个 Buffer 分别被 Set 了多少次，用于收尾清理
        uint32_t countUB[2] = {0, 0};
        int currentBuffer = 0;  // 当前使用的 buffer 索引 (0 或 1)
        uint64_t globalBlockIdx = 0;
        // 先处理Bseq范围内的，搬运任务，遍历B seq
        for(uint64_t BseqLoop = 0; BseqLoop < B; BseqLoop++) {
            // if(curBlockIdx == 1) {AscendC::printf("验证 当前 BseqLoop %d\n", BseqLoop);}
            const int32_t BseqLen = inputYGm.GetValue(BseqLoop + 1) - inputYGm.GetValue(BseqLoop);
            // 当前BseqLoop任务在T维的起始偏移
            uint64_t offsetT = inputYGm.GetValue(BseqLoop); 
            // 每个seq在T维切24，ND合轴后切1024，按一个基本块的力度搬运
            const uint32_t thisBLoopNum = (BseqLen + shapeTAlign - 1) / shapeTAlign;
            // 计算T维的尾块大小
            uint64_t lastShapeT = BseqLen - (thisBLoopNum - 1) * shapeTAlign;
            // 初始化double buffer执行相关参数
            
            for (uint64_t TLoop = 0; TLoop < thisBLoopNum; TLoop++) {
                // if(curBlockIdx == 1) {AscendC::printf("验证 当前 TLoop %d\n", TLoop);}
                uint64_t curShapeT = TLoop == (thisBLoopNum - 1) ? lastShapeT : shapeTAlign;
                uint64_t offsetB = TLoop * shapeTAlign; // 在T维的偏移（按tiling 24）

                for(uint64_t NDLoop = 0; NDLoop < shapeNDLoop; NDLoop++) {
                    // if(curBlockIdx == 1) {AscendC::printf("验证 当前 NDLoop %d\n", NDLoop);}
                    uint64_t shapeND = NDLoop == (shapeNDLoop - 1) ? lastShapeND : shapeNDAlign;
                    // 当前为第多少个block，若轮到此block则搬运
                    if (globalBlockIdx % useCoreNum == curBlockIdx) {                        
                        // if(curBlockIdx == 1) {AscendC::printf("验证 使用了当前core\n");}
                        uint64_t offsetND = NDLoop * shapeNDAlign; // 在ND维的偏移（按tiling 1024）
                        copyInParams = {static_cast<uint16_t>(curShapeT), (uint32_t)(shapeND * sizeof(T)), (uint32_t)((shapeN * shapeD - shapeND) * sizeof(T)), 0, 0};
                        copyOutParams = {static_cast<uint16_t>(curShapeT), (uint32_t)(shapeND * sizeof(T)), 0, (uint32_t)((shapeN * shapeD - shapeND) * sizeof(T)), 0};
                        // DataCopyExtParams：blockCount，blockLen，srcStride，dstStride
                        
                        if (countUB[currentBuffer] > 0) {
                            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3_MTE2[currentBuffer]); // 搬入等待搬出完成
                        }
                        DataCopyPad(inputXLocal[currentBuffer], inputXGm[(offsetT + offsetB) * shapeN * shapeD + offsetND], copyInParams, padParamsFloat);
                        SetFlag<HardEvent::MTE2_MTE3>(eventIdMTE2_MTE3[currentBuffer]); // 设置上一行搬入任务的flag
                        
                        // 从ub到gm搬运 mte3 搬出ub1
                        WaitFlag<HardEvent::MTE2_MTE3>(eventIdMTE2_MTE3[currentBuffer]); // 搬出等待搬入完成
                        DataCopyPad(outputGm[(offsetT + offsetB) * shapeN * shapeD + offsetND], inputXLocal[currentBuffer], copyOutParams);
                        SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3_MTE2[currentBuffer]); // 设置上一行搬出任务的flag
                        countUB[currentBuffer]++;
                        currentBuffer = currentBuffer ? 0 : 1;
                    }
                    globalBlockIdx ++;
                }
            }
        }
        for (int i = 0; i < 2; i++) {
            if (countUB[i] > 0) {
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3_MTE2[i]);
            }
        }

        SetFlag<HardEvent::MTE3_V>(eventIdMTE3_V);


        // 处理超出Bseq范围部分，改为-1
        const uint64_t BseqTotalLen = static_cast<uint64_t>(inputYGm.GetValue(B));
        SetFlag<HardEvent::S_V>(eventIdS_V);

        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3_V);
        WaitFlag<HardEvent::S_V>(eventIdS_V);
        if (shapeT > BseqTotalLen) {
            // 有超出Bseq范围部分
            uint64_t shapeOverBseq = shapeT - BseqTotalLen;
            uint64_t OverElements = shapeOverBseq * shapeN * shapeD;
            uint64_t OverLen = shapeOverBseq * shapeN * shapeD * sizeof(T);
            uint64_t UBElements = shapeTAlign * shapeNDAlign;
            uint64_t UBLen = shapeTAlign * shapeNDAlign * sizeof(T);

            int64_t mask = 64;
            uint64_t OverLoops = (OverLen + UBLen - 1) / UBLen;
            uint64_t lastOverLen = OverLen - (OverLoops - 1) * UBLen;
            // 剩余元素数小于UB，则只复制剩余大小
            uint64_t MaxElementNums = (OverElements > UBElements) ?  UBElements : OverElements;
            
            DuplicateWithMask(inputXLocal[0], (T)(-1), MaxElementNums);
            // Duplicate(inputXLocal, (T)(-1), mask, MaxElementNums / mask, 1, 8);

            SetFlag<HardEvent::V_MTE3>(eventIdV_MTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIdV_MTE3);

            for (uint64_t OverLoop = 0; OverLoop < OverLoops; OverLoop++) {
                uint64_t thisOverLen = (OverLoop == OverLoops - 1) ? lastOverLen : UBLen;
                if (OverLoop % useCoreNum == curBlockIdx) {
                    // 按T维逐行复制（约束是N<100，D<=256，空间足够）
                    // 从ub到gm搬运 mte3
                    copyOutDupParams = {static_cast<uint16_t>(1), (uint32_t)(thisOverLen), 0, 0, 0};
                    DataCopyPad(outputGm[BseqTotalLen * shapeN * shapeD + OverLoop * MaxElementNums], inputXLocal[0], copyOutDupParams);
                }
            }
        }
    }

private:
    // LocalTensor使用
    __aicore__ inline void DuplicateWithMask(const LocalTensor<T>& dstTensor, const T& scalarValue, const uint64_t elementNums) {
        uint64_t mask = 64;
        uint8_t maxRepeat = 255;
        uint32_t maxElements = mask * maxRepeat;
        if (elementNums > maxElements) {
            // 需要循环执行复制指令
            uint64_t dupLoops = elementNums / maxElements;
            uint64_t leftElements = elementNums - dupLoops * maxElements;
            for (uint64_t dupLoop=0; dupLoop < dupLoops; dupLoop++) {
                Duplicate(dstTensor[dupLoop * maxRepeat * mask], scalarValue, mask, maxRepeat, 1, 8);
            }
            // 搬运元素个数不一定是mask整数倍，处理余数部分
            uint8_t repeat = leftElements / mask;
            uint32_t remainder = leftElements % mask;
            if (repeat > 0) {
                Duplicate(dstTensor[dupLoops * maxRepeat * mask], scalarValue, mask, repeat, 1, 8);
            }
            if (remainder > 0) {
                Duplicate(dstTensor[dupLoops * maxRepeat * mask + repeat * mask], scalarValue, remainder, 1, 1, 8);
            }
        } else {
            uint8_t repeat = elementNums / mask;
            uint32_t remainder = elementNums % mask;
            // 元素数量在上限内，执行一次复制指令
            if (repeat > 0) {
                Duplicate(dstTensor, scalarValue, mask, repeat, 1, 8);
            }
            if (remainder > 0) {
                Duplicate(dstTensor[repeat * mask], scalarValue, remainder, 1, 1, 8);
            }
        }
    }

private:
    TPipe *pipe;
    GM_ADDR x_;
    GM_ADDR y_;
    GlobalTensor<T> inputXGm, outputGm;
    GlobalTensor<int32_t> inputYGm;
    TBuf<TPosition::VECCALC> inputXUb[2]; // Ping-Pong Buffer 数组
    LocalTensor<T> inputXLocal[2]; // LocalTensor 数组
    LocalTensor<uint32_t> inputYLocal;
    uint64_t curBlockIdx;
    uint64_t shapeT, shapeN, shapeD, B, shapeTAlign, shapeNDAlign, useCoreNum;
    uint64_t shapeNDLoop, lastShapeND;
    DataCopyExtParams copyInParams, copyOutParams, copyOutDupParams;
    DataCopyPadExtParams<T> padParamsFloat {false, 0, 0, 0}; // 不指定填充值
    event_t eventIdV_MTE3, eventIdMTE3_V, eventIdS_V;
    event_t eventIdMTE2_MTE3[2], eventIdMTE3_MTE2[2]; // Ping-Pong 事件 ID 数组
    bool useUB1;
};

#endif // DATA_COPY_EXAMPLE
