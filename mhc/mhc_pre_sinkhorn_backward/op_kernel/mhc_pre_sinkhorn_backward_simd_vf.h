template<typename TYPE_X, typename T>
__aicore__ inline void MhcPreGradKernel<TYPE_X, T>::ComputeGradX1(const int32_t taskOffset, const int32_t tileTaskCount, const int32_t innerId)
{


    PipeBarrier<PIPE_V>();



    // AscendC::Broadcast<float, 2, 1>(dRsqrtBrcbLocal_, gradRsqrtLocal_, dstRsqrtShape, srcRsqrtShape)
    PipeBarrier<PIPE_V>();

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

        }
        xLocal_ = inputXInQueue.AllocTensor<TYPE_X>();
        gradXCubeLocal_ = xLocal_.template ReinterpretCast<float>()[tileTaskCount * n_ * c0_ / 2];
        gradYLocal_ =  xLocal_[tileTaskCount * n_ * c0_ * 3];
        
        DataCopyPad(gradYLocal_, gradYGlobal_[taskOffset * c_ + c0_ * loopIdC], {static_cast<uint16_t>(tileTaskCount),
            static_cast<uint32_t>(copyLen * sizeof(TYPE_X)), static_cast<uint32_t>((c_ - copyLen) * sizeof(TYPE_X)), 0, 0}, {isPad, 0, padLen, 0});
        DataCopyPad(xLocal_, xGlobal_[taskOffset * n_ * c_ + c0_ * loopIdC], {static_cast<uint16_t>(n_ * tileTaskCount),
            static_cast<uint32_t>(copyLen * sizeof(TYPE_X)), static_cast<uint32_t>((c_ - copyLen) * sizeof(TYPE_X)), 0, 0}, {isPad, 0, padLen, 0});
        DataCopyPad(gradXCubeLocal_, gradXCubeGlobal_[taskOffset * n_ * c_ + c0_ * loopIdC], {static_cast<uint16_t>(n_ * tileTaskCount),
            static_cast<uint32_t>(copyLen * sizeof(TYPE_X)), static_cast<uint32_t>((c_ - copyLen) * sizeof(TYPE_X)), 0, 0}, {isPad, 0, padLen, 0});

        inputXInQueue.EnQue(xLocal_);
        xLocal_ = inputXInQueue.DeQue();
        PipeBarrier<PIPE_V>();
        Cast(gradYCastLocal_, gradYLocal_, RoundMode::CAST_NONE,ubAlignC * tileTaskCount);

        Cast(xCastLocal_, xLocal_, RoundMode::CAST_NONE,ubAlignC * n_ * tileTaskCount);

        PipeBarrier<PIPE_V>();
        uint8_t blkStride1 = static_cast<uint8_t>(ubAlignC / ELEMENTS_SIZE_PER_BLOCK);
        uint8_t blkStride2 = static_cast<uint8_t>(n_ * ubAlignC / ELEMENTS_SIZE_PER_BLOCK);
        for (int32_t loopIdN = 0; loopIdN < n_; loopIdN+=1) {
            for (int32_t loopOffsetC0 = 0; loopOffsetC0 < copyLen; loopOffsetC0+=ELEMENTS_SIZE_PER_REPEAT) {
                uint64_t mask = min(static_cast<uint64_t>(ELEMENTS_SIZE_PER_REPEAT), static_cast<uint64_t>(copyLen - loopOffsetC0));
                mul(gradXCastLocal_[loopIdN * ubAlignC + loopOffsetC0 ], gradYCastLocal_[loopOffsetC0 ],
                    preBrcbLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK + innerId * n_ * ELEMENTS_SIZE_PER_BLOCK], mask, tileTaskCount,
                    { 1, 1, 1, blkStride1, blkStride1, 0 });
            }
        }
        PipeBarrier<PIPE_V>();
        for (int32_t loopIdN = 0; loopIdN < n_; loopIdN+=1) {
            for (int32_t loopOffsetC0 = 0; loopOffsetC0 < copyLen; loopOffsetC0+=ELEMENTS_SIZE_PER_REPEAT) {
                uint64_t mask = min(static_cast<uint64_t>(ELEMENTS_SIZE_PER_REPEAT), static_cast<uint64_t>(copyLen - loopOffsetC0));
                MulAddDst(gradXCastLocal_[loopIdN * ubAlignC + loopOffsetC0], xCastLocal_[loopIdN * ubAlignC + loopOffsetC0], dRsqrtBrcbLocal_[innerId * ELEMENTS_SIZE_PER_BLOCK],
                    mask, tileTaskCount, { 1, 1, 0, blkStride2, blkStride2, 1 });
            }
        }


        Add(gradXCastLocal_, gradXCubeLocal_, gradXCastLocal_, ubAlignC * n_ * tileTaskCount);
        PipeBarrier<PIPE_V>();
        inputXInQueue.FreeTensor(xLocal_);

        auto gradXLocalOut = OutQueue.AllocTensor<TYPE_X>();

        Cast(gradXLocalOut, gradXCastLocal_, RoundMode::CAST_RINT,ubAlignC * n_ * tileTaskCount);
        OutQueue.EnQue(gradXLocalOut);
        gradXLocalOut = OutQueue.DeQue();
    
        DataCopyPad(gradXGlobal_[taskOffset * n_ * c_ + c0_ * loopIdC], gradXLocalOut, {static_cast<uint16_t>(n_ * tileTaskCount),
            static_cast<uint32_t>(copyLen * sizeof(TYPE_X)), static_cast<uint32_t>((c_ - copyLen) * sizeof(TYPE_X)), 0, 0});
        OutQueue.FreeTensor(gradXLocalOut);

        PipeBarrier<PIPE_V>();
    }
}
