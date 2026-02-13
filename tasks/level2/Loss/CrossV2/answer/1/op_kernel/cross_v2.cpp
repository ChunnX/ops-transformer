#include "kernel_operator.h"
namespace CrossV2NS {
using namespace AscendC;
const int32_t BUFFER_NUM = 2;
const int64_t SPECIFIED_DIM_SIZE = 3;
const uint32_t UINT8_ALIGN_SIZE = 256;
const uint32_t BLOCK_SIZE = 32;
const uint32_t NUM_PER_BLK_32 = 8;
const uint32_t NUM_PER_BLK_16 = 16;
const uint32_t IDX_0 = 0;
const uint32_t IDX_1 = 1;
const uint32_t IDX_2 = 2;
const half HALF_ZERO = 0;
const half HALF_256 = 256;
const uint16_t GATHER_MASK_UINT16_0 = 0b1001'0010'0100'1001;
const uint16_t GATHER_MASK_UINT16_1 = 0b0010'0100'1001'0010;
const uint16_t GATHER_MASK_UINT16_2 = 0b0100'1001'0010'0100;
const uint32_t GATHER_MASK_UINT32_0 = 0b0100'1001'0010'0100'1001'0010'0100'1001;
const uint32_t GATHER_MASK_UINT32_1 = 0b1001'0010'0100'1001'0010'0100'1001'0010;
const uint32_t GATHER_MASK_UINT32_2 = 0b0010'0100'1001'0010'0100'1001'0010'0100;

__aicore__ inline uint32_t RoundUp(uint32_t x, uint32_t y)
{
    if (y == 0) {
        return x;
    }
    return (x + y - 1) / y * y;
}

template <typename T>
class KernelCrossOneStep
{
public:
    __aicore__ inline KernelCrossOneStep()
    {}

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y, int64_t tileNum, int64_t tileDataNum, int64_t tailDataNum)
    {
        this->coreIdx = GetBlockIdx();
        this->coreNum = GetBlockNum();
        x1Gm.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2);
        yGm.SetGlobalBuffer((__gm__ T*)y);
        this->tileNum = tileNum;
        this->tileDataNum = tileDataNum;
        this->tailDataNum = tailDataNum;

        pipe.InitBuffer(inQueueSelf, BUFFER_NUM, SPECIFIED_DIM_SIZE * this->tileDataNum * sizeof(T));
        pipe.InitBuffer(inQueueOther, BUFFER_NUM, SPECIFIED_DIM_SIZE * this->tileDataNum * sizeof(T));
        pipe.InitBuffer(outQueueOut, BUFFER_NUM, SPECIFIED_DIM_SIZE * this->tileDataNum * sizeof(T));
        pipe.InitBuffer(srcOffsetBuf, SPECIFIED_DIM_SIZE * BLOCK_SIZE);
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, int16_t> || std::is_same_v<T, bfloat16_t>) {
            pipe.InitBuffer(bufXaYb, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(bufXbYa, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(x1Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(y1Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(z1Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(x2Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(y2Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(z2Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(x3Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(y3Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(z3Buf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(selfBuf, SPECIFIED_DIM_SIZE * this->tileDataNum * sizeof(float));
            pipe.InitBuffer(otherBuf, SPECIFIED_DIM_SIZE * this->tileDataNum * sizeof(float));
            pipe.InitBuffer(outBuf, SPECIFIED_DIM_SIZE * this->tileDataNum * sizeof(float));
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            pipe.InitBuffer(bufXaYb, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(bufXbYa, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(x1Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(y1Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(z1Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(x2Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(y2Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(z2Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(x3Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(y3Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(z3Buf, this->tileDataNum * sizeof(half));
            pipe.InitBuffer(selfBuf, SPECIFIED_DIM_SIZE * this->tileDataNum * sizeof(half));
            pipe.InitBuffer(otherBuf, SPECIFIED_DIM_SIZE * this->tileDataNum * sizeof(half));
            pipe.InitBuffer(outBuf, SPECIFIED_DIM_SIZE * this->tileDataNum * sizeof(half));
        } else {
            pipe.InitBuffer(x1Buf, this->tileDataNum * sizeof(T));
            pipe.InitBuffer(y1Buf, this->tileDataNum * sizeof(T));
            pipe.InitBuffer(z1Buf, this->tileDataNum * sizeof(T));
            pipe.InitBuffer(x2Buf, this->tileDataNum * sizeof(T));
            pipe.InitBuffer(y2Buf, this->tileDataNum * sizeof(T));
            pipe.InitBuffer(z2Buf, this->tileDataNum * sizeof(T));
            pipe.InitBuffer(x3Buf, this->tileDataNum * sizeof(T));
            pipe.InitBuffer(y3Buf, this->tileDataNum * sizeof(T));
            pipe.InitBuffer(z3Buf, this->tileDataNum * sizeof(T));
            pipe.InitBuffer(bufXaYb, this->tileDataNum * sizeof(T));
            pipe.InitBuffer(bufXbYa, this->tileDataNum * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int16_t>) {
            LocalTensor<uint16_t> xOffset = srcOffsetBuf.GetWithOffset<uint16_t>(NUM_PER_BLK_16, 0);
            LocalTensor<uint16_t> yOffset = srcOffsetBuf.GetWithOffset<uint16_t>(NUM_PER_BLK_16, BLOCK_SIZE);
            LocalTensor<uint16_t> zOffset = srcOffsetBuf.GetWithOffset<uint16_t>(NUM_PER_BLK_16, IDX_2 * BLOCK_SIZE);
            xOffset.SetValue(IDX_0, GATHER_MASK_UINT16_0);
            xOffset.SetValue(IDX_1, GATHER_MASK_UINT16_2);
            xOffset.SetValue(IDX_2, GATHER_MASK_UINT16_1);
            yOffset.SetValue(IDX_0, GATHER_MASK_UINT16_1);
            yOffset.SetValue(IDX_1, GATHER_MASK_UINT16_0);
            yOffset.SetValue(IDX_2, GATHER_MASK_UINT16_2);
            zOffset.SetValue(IDX_0, GATHER_MASK_UINT16_2);
            zOffset.SetValue(IDX_1, GATHER_MASK_UINT16_1);
            zOffset.SetValue(IDX_2, GATHER_MASK_UINT16_0);
        } else {
            LocalTensor<uint32_t> xOffset = srcOffsetBuf.GetWithOffset<uint32_t>(NUM_PER_BLK_32, 0);
            LocalTensor<uint32_t> yOffset = srcOffsetBuf.GetWithOffset<uint32_t>(NUM_PER_BLK_32, BLOCK_SIZE);
            LocalTensor<uint32_t> zOffset = srcOffsetBuf.GetWithOffset<uint32_t>(NUM_PER_BLK_32, IDX_2 * BLOCK_SIZE);
            xOffset.SetValue(IDX_0, GATHER_MASK_UINT32_0);
            yOffset.SetValue(IDX_0, GATHER_MASK_UINT32_1);
            zOffset.SetValue(IDX_0, GATHER_MASK_UINT32_2);
        }

        int64_t loopCount = this->tileNum;
        for (int64_t i = coreIdx; i < loopCount; i += this->coreNum) {
            this->processDataNum = this->tileDataNum;
            int64_t batchIdx = i * this->tileDataNum;
            if (i == loopCount - 1) {
                this->processDataNum = this->tailDataNum;
            }
            int64_t offset = batchIdx * SPECIFIED_DIM_SIZE;
            CopyInOneStep(offset);
            ComputeOneStep();
            CopyOutOneStep(offset);
        }
    }

private:
    __aicore__ inline void CopyInOneStep(int64_t offset)
    {
        LocalTensor<T> selfLocal = inQueueSelf.AllocTensor<T>();
        LocalTensor<T> otherLocal = inQueueOther.AllocTensor<T>();

        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(SPECIFIED_DIM_SIZE * this->processDataNum * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(selfLocal, x1Gm[offset], copyParams, padParams);
        DataCopyPad(otherLocal, x2Gm[offset], copyParams, padParams);

        inQueueSelf.EnQue(selfLocal);
        inQueueOther.EnQue(otherLocal);
    }

    __aicore__ inline void ComputeBit16(
        const LocalTensor<T>& selfLocal, const LocalTensor<T>& otherLocal, const LocalTensor<T>& outLocal)
    {
        using U = float;
        LocalTensor<uint32_t> xOffset = srcOffsetBuf.GetWithOffset<uint32_t>(NUM_PER_BLK_32, 0);
        LocalTensor<uint32_t> yOffset = srcOffsetBuf.GetWithOffset<uint32_t>(NUM_PER_BLK_32, BLOCK_SIZE);
        LocalTensor<uint32_t> zOffset = srcOffsetBuf.GetWithOffset<uint32_t>(NUM_PER_BLK_32, IDX_2 * BLOCK_SIZE);
        LocalTensor<U> self = selfBuf.Get<U>();
        LocalTensor<U> other = otherBuf.Get<U>();
        LocalTensor<U> out = outBuf.Get<U>();
        Cast(self, selfLocal, RoundMode::CAST_NONE, SPECIFIED_DIM_SIZE * this->processDataNum);
        Cast(other, otherLocal, RoundMode::CAST_NONE, SPECIFIED_DIM_SIZE * this->processDataNum);
        LocalTensor<U> x1 = x1Buf.Get<U>();
        LocalTensor<U> y1 = y1Buf.Get<U>();
        LocalTensor<U> z1 = z1Buf.Get<U>();
        LocalTensor<U> x2 = x2Buf.Get<U>();
        LocalTensor<U> y2 = y2Buf.Get<U>();
        LocalTensor<U> z2 = z2Buf.Get<U>();
        LocalTensor<U> x3 = x3Buf.Get<U>();
        LocalTensor<U> y3 = y3Buf.Get<U>();
        LocalTensor<U> z3 = z3Buf.Get<U>();
        LocalTensor<U> xayb = bufXaYb.Get<U>();
        LocalTensor<U> xbya = bufXbYa.Get<U>();
        uint8_t repeatTimes = this->tileDataNum / NUM_PER_BLK_32;
        uint32_t mask = SPECIFIED_DIM_SIZE * NUM_PER_BLK_32;
        uint64_t rsvdCnt = 0;
        GatherMask(x1, self, xOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(y1, self, yOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(z1, self, zOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(x2, other, xOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(y2, other, yOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(z2, other, zOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        Mul(xayb, y1, z2, this->processDataNum);
        Mul(xbya, y2, z1, this->processDataNum);
        Sub(x3, xayb, xbya, this->processDataNum);
        Mul(xayb, x2, z1, this->processDataNum);
        Mul(xbya, x1, z2, this->processDataNum);
        Sub(y3, xayb, xbya, this->processDataNum);
        Mul(xayb, x1, y2, this->processDataNum);
        Mul(xbya, x2, y1, this->processDataNum);
        Sub(z3, xayb, xbya, this->processDataNum);
        for (uint32_t i = 0; i < this->processDataNum; i++) {
            out.SetValue(i * SPECIFIED_DIM_SIZE + IDX_0, x3.GetValue(i));
            out.SetValue(i * SPECIFIED_DIM_SIZE + IDX_1, y3.GetValue(i));
            out.SetValue(i * SPECIFIED_DIM_SIZE + IDX_2, z3.GetValue(i));
        }
        Cast(outLocal, out, RoundMode::CAST_RINT, SPECIFIED_DIM_SIZE * this->processDataNum);
    }

    __aicore__ inline void ComputeBit8(
        const LocalTensor<T>& selfLocal, const LocalTensor<T>& otherLocal, const LocalTensor<T>& outLocal)
    {
        using U = half;
        LocalTensor<uint16_t> xOffset = srcOffsetBuf.GetWithOffset<uint16_t>(NUM_PER_BLK_16, 0);
        LocalTensor<uint16_t> yOffset = srcOffsetBuf.GetWithOffset<uint16_t>(NUM_PER_BLK_16, BLOCK_SIZE);
        LocalTensor<uint16_t> zOffset = srcOffsetBuf.GetWithOffset<uint16_t>(NUM_PER_BLK_16, IDX_2 * BLOCK_SIZE);
        LocalTensor<U> self = selfBuf.Get<U>();
        LocalTensor<U> other = otherBuf.Get<U>();
        LocalTensor<U> out = outBuf.Get<U>();
        Cast(self, selfLocal, RoundMode::CAST_NONE, SPECIFIED_DIM_SIZE * this->processDataNum);
        Cast(other, otherLocal, RoundMode::CAST_NONE, SPECIFIED_DIM_SIZE * this->processDataNum);
        LocalTensor<U> x1 = x1Buf.Get<U>();
        LocalTensor<U> y1 = y1Buf.Get<U>();
        LocalTensor<U> z1 = z1Buf.Get<U>();
        LocalTensor<U> x3 = x3Buf.Get<U>();
        LocalTensor<U> y3 = y3Buf.Get<U>();
        LocalTensor<U> z3 = z3Buf.Get<U>();
        LocalTensor<U> x2 = x2Buf.Get<U>();
        LocalTensor<U> y2 = y2Buf.Get<U>();
        LocalTensor<U> z2 = z2Buf.Get<U>();
        LocalTensor<U> xayb = bufXaYb.Get<U>();
        LocalTensor<U> xbya = bufXbYa.Get<U>();
        uint8_t repeatTimes = this->tileDataNum / NUM_PER_BLK_16;
        uint32_t mask = SPECIFIED_DIM_SIZE * NUM_PER_BLK_16;
        uint64_t rsvdCnt = 0;
        GatherMask(x2, other, xOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(y2, other, yOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(z2, other, zOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(x1, self, xOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(y1, self, yOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(z1, self, zOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        Mul(xayb, y1, z2, this->processDataNum);
        Mul(xbya, y2, z1, this->processDataNum);
        Sub(x3, xayb, xbya, this->processDataNum);
        Mul(xayb, x1, y2, this->processDataNum);
        Mul(xbya, x2, y1, this->processDataNum);
        Sub(z3, xayb, xbya, this->processDataNum);
        Mul(xayb, x2, z1, this->processDataNum);
        Mul(xbya, x1, z2, this->processDataNum);
        Sub(y3, xayb, xbya, this->processDataNum);
        if constexpr (std::is_same_v<T, uint8_t>) {
            LocalTensor<uint8_t> maskX = x1Buf.Get<uint8_t>();
            LocalTensor<uint8_t> maskY = y1Buf.Get<uint8_t>();
            LocalTensor<uint8_t> maskZ = z1Buf.Get<uint8_t>();
            LocalTensor<U> x3Tmp = x2Buf.Get<U>();
            LocalTensor<U> y3Tmp = y2Buf.Get<U>();
            LocalTensor<U> z3Tmp = z2Buf.Get<U>();
            uint32_t calCount =
                RoundUp(static_cast<uint32_t>(this->processDataNum * sizeof(U)), UINT8_ALIGN_SIZE) / sizeof(U);
            CompareScalar(maskX, x3, HALF_ZERO, CMPMODE::LT, calCount);
            CompareScalar(maskY, y3, HALF_ZERO, CMPMODE::LT, calCount);
            CompareScalar(maskZ, z3, HALF_ZERO, CMPMODE::LT, calCount);
            Adds(x3Tmp, x3, HALF_256, this->processDataNum);
            Adds(z3Tmp, z3, HALF_256, this->processDataNum);
            Adds(y3Tmp, y3, HALF_256, this->processDataNum);
            Select(x3, maskX, x3Tmp, x3, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
            Select(y3, maskY, y3Tmp, y3, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
            Select(z3, maskZ, z3Tmp, z3, SELMODE::VSEL_TENSOR_TENSOR_MODE, this->processDataNum);
        }
        for (uint32_t i = 0; i < this->processDataNum; i++) {
            out.SetValue(i * SPECIFIED_DIM_SIZE + IDX_0, x3.GetValue(i));
            out.SetValue(i * SPECIFIED_DIM_SIZE + IDX_1, y3.GetValue(i));
            out.SetValue(i * SPECIFIED_DIM_SIZE + IDX_2, z3.GetValue(i));
        }
        Cast(outLocal, out, RoundMode::CAST_NONE, SPECIFIED_DIM_SIZE * this->processDataNum);
    }

    template <typename U>
    __aicore__ inline void ComputeNotCast(
        const LocalTensor<T>& selfLocal, const LocalTensor<T>& otherLocal, const LocalTensor<T>& outLocal)
    {
        uint32_t numPerBlock = BLOCK_SIZE / sizeof(U);
        LocalTensor<U> xOffset = srcOffsetBuf.GetWithOffset<U>(numPerBlock, 0);
        LocalTensor<U> yOffset = srcOffsetBuf.GetWithOffset<U>(numPerBlock, BLOCK_SIZE);
        LocalTensor<U> zOffset = srcOffsetBuf.GetWithOffset<U>(numPerBlock, IDX_2 * BLOCK_SIZE);
        LocalTensor<T> x1 = x1Buf.Get<T>();
        LocalTensor<T> y1 = y1Buf.Get<T>();
        LocalTensor<T> z1 = z1Buf.Get<T>();
        LocalTensor<T> x2 = x2Buf.Get<T>();
        LocalTensor<T> y2 = y2Buf.Get<T>();
        LocalTensor<T> z2 = z2Buf.Get<T>();
        LocalTensor<T> x3 = x3Buf.Get<T>();
        LocalTensor<T> y3 = y3Buf.Get<T>();
        LocalTensor<T> z3 = z3Buf.Get<T>();
        LocalTensor<T> xayb = bufXaYb.Get<T>();
        LocalTensor<T> xbya = bufXbYa.Get<T>();
        uint8_t repeatTimes = this->tileDataNum / numPerBlock;
        uint32_t mask = SPECIFIED_DIM_SIZE * numPerBlock;
        uint64_t rsvdCnt = 0;
        GatherMask(x1, selfLocal, xOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(y1, selfLocal, yOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(z1, selfLocal, zOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(x2, otherLocal, xOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(y2, otherLocal, yOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        GatherMask(z2, otherLocal, zOffset, true, mask, {1, repeatTimes, SPECIFIED_DIM_SIZE, 0}, rsvdCnt);
        Mul(xayb, x2, z1, this->processDataNum);
        Mul(xbya, x1, z2, this->processDataNum);
        Sub(y3, xayb, xbya, this->processDataNum);
        Mul(xayb, y1, z2, this->processDataNum);
        Mul(xbya, y2, z1, this->processDataNum);
        Sub(x3, xayb, xbya, this->processDataNum);
        Mul(xayb, x1, y2, this->processDataNum);
        Mul(xbya, x2, y1, this->processDataNum);
        Sub(z3, xayb, xbya, this->processDataNum);
        for (uint32_t i = 0; i < this->processDataNum; i++) {
            outLocal.SetValue(i * SPECIFIED_DIM_SIZE + IDX_0, x3.GetValue(i));
            outLocal.SetValue(i * SPECIFIED_DIM_SIZE + IDX_1, y3.GetValue(i));
            outLocal.SetValue(i * SPECIFIED_DIM_SIZE + IDX_2, z3.GetValue(i));
        }
    }

    __aicore__ inline void ComputeOneStep()
    {
        LocalTensor<T> selfLocal = inQueueSelf.DeQue<T>();
        LocalTensor<T> otherLocal = inQueueOther.DeQue<T>();
        LocalTensor<T> outLocal = outQueueOut.AllocTensor<T>();
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, bfloat16_t>) {
            ComputeBit16(selfLocal, otherLocal, outLocal);
        } else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
            ComputeBit8(selfLocal, otherLocal, outLocal);
        } else if constexpr (std::is_same_v<T, int16_t>) {
            ComputeNotCast<uint16_t>(selfLocal, otherLocal, outLocal);
        } else {
            ComputeNotCast<uint32_t>(selfLocal, otherLocal, outLocal);
        }
        outQueueOut.EnQue<T>(outLocal);
        inQueueSelf.FreeTensor(selfLocal);
        inQueueOther.FreeTensor(otherLocal);
    }

    __aicore__ inline void CopyOutOneStep(int64_t offset)
    {
        LocalTensor<T> outLocal = outQueueOut.DeQue<T>();
        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(SPECIFIED_DIM_SIZE * this->processDataNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm[offset], outLocal, copyParams);
        outQueueOut.FreeTensor(outLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueSelf;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueOther;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOut;
    TBuf<QuePosition::VECCALC> bufXaYb;
    TBuf<QuePosition::VECCALC> bufXbYa;
    TBuf<QuePosition::VECCALC> selfBuf;
    TBuf<QuePosition::VECCALC> otherBuf;
    TBuf<QuePosition::VECCALC> outBuf;
    TBuf<QuePosition::VECCALC> x1Buf;
    TBuf<QuePosition::VECCALC> y1Buf;
    TBuf<QuePosition::VECCALC> z1Buf;
    TBuf<QuePosition::VECCALC> x2Buf;
    TBuf<QuePosition::VECCALC> y2Buf;
    TBuf<QuePosition::VECCALC> z2Buf;
    TBuf<QuePosition::VECCALC> x3Buf;
    TBuf<QuePosition::VECCALC> y3Buf;
    TBuf<QuePosition::VECCALC> z3Buf;
    TBuf<QuePosition::VECCALC> srcOffsetBuf;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> yGm;
    int64_t tileNum;
    int64_t tileDataNum;
    int64_t tailDataNum;
    int64_t processDataNum;
    uint32_t coreIdx;
    uint32_t coreNum;
};
} // namespace CrossV2NS
extern "C" __global__ __aicore__ void cross_v2(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    using namespace CrossV2NS;
    GET_TILING_DATA(tiling_data, tiling);
    KernelCrossOneStep<DTYPE_X1> op;
    op.Init(x1, x2, y, tiling_data.tileNum, tiling_data.tileDataNum, tiling_data.tailDataNum);
    op.Process();
}