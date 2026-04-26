template<typename TYPE_X, typename T>
__aicore__ inline void MhcPreGradKernel<TYPE_X, T>::ComputeGradPre(const int32_t taskOffset, const int32_t tileTaskCount, const int32_t innerId)
{
    WaitFlag<HardEvent::V_MTE2>(0);
    Duplicate(tmpLocal_, 0.f, tileTaskCount * n_ * ELEMENTS_SIZE_PER_REPEAT);
    for (int32_t loopIdC = 0; loopIdC < c1Align_; loopIdC+=1) {
        int64_t copyLen = c0_;
        bool isPad = false;
        uint8_t padLen = 0;
        int64_t ubAlignC = c0_;
        if (loopIdC == c1_) {
            isPad = true;
            copyLen = cTail_;
            ubAlignC = cTailAlign_;
            padLen = static_cast<uint8_t>(cTailAlign_ - cTail_);
            // Duplicate(gradYLocal_, 0.f, c0_ - cTailAlign_, tileTaskCount, 1, c0_ / ELEMENTS_SIZE_PER_BLOCK);
            // Duplicate(xLocal_, 0.f, c0_ - cTailAlign_, tileTaskCount * n_, 1, c0_ / ELEMENTS_SIZE_PER_BLOCK);
        }
        auto xLocal_ = inputXInQueue.AllocTensor<TYPE_X>();

        gradYLocal_ =  xLocal_[tileTaskCount * n_ * c0_];

        DataCopyPad(gradYLocal_, gradYGlobal_[taskOffset * c_ + c0_ * loopIdC], {static_cast<uint16_t>(tileTaskCount),
            static_cast<uint32_t>(copyLen * sizeof(TYPE_X)), static_cast<uint32_t>((c_ - copyLen) * sizeof(TYPE_X)), 0, 0}, {isPad, 0, padLen, 0});
        DataCopyPad(xLocal_, xGlobal_[taskOffset * n_ * c_ + c0_ * loopIdC], {static_cast<uint16_t>(n_ * tileTaskCount),
            static_cast<uint32_t>(copyLen * sizeof(TYPE_X)), static_cast<uint32_t>((c_ - copyLen) * sizeof(TYPE_X)), 0, 0}, {isPad, 0, padLen, 0});
        inputXInQueue.EnQue(xLocal_);
        xLocal_ = inputXInQueue.DeQue();
        PipeBarrier<PIPE_V>();

        auto xCastOutLocal = OutQueue.AllocTensor<float>();

        Cast(xCastOutLocal, xLocal_, RoundMode::CAST_NONE,ubAlignC * n_ * tileTaskCount);
        OutQueue.EnQue(xCastOutLocal);
        xCastOutLocal = OutQueue.DeQue();
        
        DataCopyPad(xWorkspaceGlobal_[taskOffset * n_ * c_ + c0_ * loopIdC], xCastOutLocal, {static_cast<uint16_t>(n_ * tileTaskCount),
            static_cast<uint32_t>(copyLen * sizeof(float)), static_cast<uint32_t>((c_ - copyLen) * sizeof(float)), 0, 0});

        Cast(gradYCastLocal_, gradYLocal_, RoundMode::CAST_NONE,ubAlignC * tileTaskCount);
        inputXInQueue.FreeTensor(xLocal_);


        PipeBarrier<PIPE_V>();
        uint8_t blkStride3 = static_cast<uint8_t>(n_ * ubAlignC / ELEMENTS_SIZE_PER_BLOCK);
        for (int32_t loopIdN = 0; loopIdN < n_; loopIdN+=1) {
            for (int32_t loopOffsetC0 = 0; loopOffsetC0 < copyLen; loopOffsetC0+=ELEMENTS_SIZE_PER_REPEAT) {
                uint64_t mask = min(static_cast<uint64_t>(ELEMENTS_SIZE_PER_REPEAT), static_cast<uint64_t>(copyLen - loopOffsetC0));
                mul(xCastOutLocal[loopIdN * ubAlignC + loopOffsetC0], xCastOutLocal[loopIdN * ubAlignC + loopOffsetC0],
                    gradYCastLocal_[loopOffsetC0],
                    mask, tileTaskCount,
                    { 1, 1, 1, blkStride3, blkStride3, blkStride3 });
            }
        }
        PipeBarrier<PIPE_V>();

        int64_t reduceLen = ubAlignC;
        uint8_t blkStride = static_cast<uint8_t>(ubAlignC / ELEMENTS_SIZE_PER_BLOCK);
        if (loopIdC != c1_) {
            Add(xCastOutLocal[64], xCastOutLocal[64], xCastOutLocal[128 + 64], ELEMENTS_SIZE_PER_REPEAT, tileTaskCount * n_,
                {1, 1, 1, blkStride, blkStride, blkStride});
            Add(xCastOutLocal, xCastOutLocal, xCastOutLocal[128], ELEMENTS_SIZE_PER_REPEAT, tileTaskCount * n_,
                {1, 1, 1, blkStride, blkStride, blkStride});
            PipeBarrier<PIPE_V>();

            Add(xCastOutLocal, xCastOutLocal, xCastOutLocal[64], ELEMENTS_SIZE_PER_REPEAT, tileTaskCount * n_,
                {1, 1, 1, blkStride, blkStride, blkStride});
        } else {
            uint64_t mask = max(static_cast<uint64_t>(cTail_ - (128 + 64)), static_cast<uint64_t>(0));
            if (mask > 0) {
                Add(xCastOutLocal[64], xCastOutLocal[64], xCastOutLocal[128 + 64], mask, tileTaskCount * n_,
                    {1, 1, 1, blkStride, blkStride, blkStride});
            }
            mask = max(static_cast<uint64_t>(cTail_ - (128)), static_cast<uint64_t>(0));
            if (mask > 0) {
                Add(xCastOutLocal, xCastOutLocal, xCastOutLocal[128], mask, tileTaskCount * n_,
                    {1, 1, 1, blkStride, blkStride, blkStride});
            }
            PipeBarrier<PIPE_V>();
            mask = max(static_cast<uint64_t>(cTail_ - (64)), static_cast<uint64_t>(0));
            if (mask > 0) {
                Add(xCastOutLocal, xCastOutLocal, xCastOutLocal[64], mask, tileTaskCount * n_,
                    {1, 1, 1, blkStride, blkStride, blkStride});  
            }
        }
        uint64_t mask = min(static_cast<uint64_t>(cTail_) , static_cast<uint64_t>(REPEAT_LENTH));
        
        PipeBarrier<PIPE_V>();
        Add(tmpLocal_, tmpLocal_, xCastOutLocal, mask, tileTaskCount * n_,
            {1, 1, 1, 8, 8, blkStride});
        OutQueue.FreeTensor(xCastOutLocal);
    }
    PipeBarrier<PIPE_V>();
    Duplicate(dPrePostTempLocal_, 0.f, tileCoreBS_ * n_ * 2);
    PipeBarrier<PIPE_V>();
    for (int32_t loopIdN = 0; loopIdN < n_; loopIdN+=1) {
        WholeReduceSum(dPrePostTempLocal_[loopIdN + innerId * n_ * 2], tmpLocal_[loopIdN * REPEAT_LENTH], REPEAT_LENTH,
            tileTaskCount, n_ * 2, 1, n_ * 8);// dst repeat stride  srcblkstride srcrepstride
        // dst 间隔 8 2 *n
    }
}
