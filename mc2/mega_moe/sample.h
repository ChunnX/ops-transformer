template <typename T, typename U>
__aicore__ inline void MoeV3GatherMxfp4Quant<T, U>::CopyOut(int64_t dstIdx, int64_t colIdx, int64_t loopCols,
                                                             int64_t loopScaleCols)
{
    LocalTensor<uint8_t> mxScaleLocal = mxScaleOutQueue_.DeQue<uint8_t>();
    LocalTensor<uint8_t> outLocal = xQuantOutQueue_.DeQue<uint8_t>();
    // ============ DEBUG 打印开始 ============
    int64_t xOutOffset = (dstIdx * cols_ + colIdx * perLoopCols_) / 2;
    int64_t scaleOutOffset = dstIdx * scaleCols_ + colIdx * perLoopScaleCols_;
    
    printf("=== CopyOut DEBUG ===\n");
    printf("dstIdx=%lld, colIdx=%lld, loopCols=%lld, loopScaleCols=%lld\n", 
           dstIdx, colIdx, loopCols, loopScaleCols);
    printf("xOutOffset=%lld, xOutBytes=%u\n", xOutOffset, 
           static_cast<uint32_t>(loopCols * sizeof(uint8_t) / 2));
    printf("scaleOutOffset=%lld, scaleOutBytes=%u\n", scaleOutOffset,
           static_cast<uint32_t>(loopScaleCols * sizeof(uint8_t)));
    printf("cols_=%lld, scaleCols_=%lld, perLoopCols_=%lld, perLoopScaleCols_=%lld\n",
           cols_, scaleCols_, perLoopCols_, perLoopScaleCols_);
    printf("validScaleCols_=%lld, lastLoopScaleCols_=%lld, lastLoopValidScaleCols_=%lld, colLoops_=%lld\n",
           validScaleCols_, lastLoopScaleCols_, lastLoopValidScaleCols_, colLoops_);
    
    // 打印UB中前几个scale值
    printf("mxScaleLocal data (first 8 bytes): ");
    for (int i = 0; i < 8 && i < loopScaleCols; i++) {
        printf("0x%02x ", mxScaleLocal.GetValue(i));
    }
    printf("\n");
    // ============ DEBUG 打印结束 ============
    DataCopyExtParams copyOutParams = {1, static_cast<uint32_t>(loopCols * sizeof(uint8_t) / 2), 0, 0, 0};
    DataCopyPad<uint8_t>(expandedXOutGm_[(dstIdx * cols_ + colIdx * perLoopCols_) / 2], outLocal, copyOutParams);
    DataCopyExtParams copyScaleParams = {1, static_cast<uint32_t>(loopScaleCols * sizeof(uint8_t)), 0, 0, 0};
    DataCopyPad<uint8_t>(expandedScaleOutGm_[dstIdx * scaleCols_ + colIdx * perLoopScaleCols_], mxScaleLocal,
                         copyScaleParams);
    xQuantOutQueue_.FreeTensor(outLocal);
    mxScaleOutQueue_.FreeTensor(mxScaleLocal);
}
同时在 Process 函数中也添加打印：
template <typename T, typename U>
__aicore__ inline void MoeV3GatherMxfp4Quant<T, U>::CopyExpandedXandMXQuant(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = sortedRowIdxInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    for (int64_t index = 0; index < currentLoopRows_; index++) {
        int32_t srcIdx = indicesLocal.GetValue(index);
        int64_t dstIdx = perCoreRow_ * blockIdx_ + perLoopRows_ * progress + index;
        
        // ============ DEBUG 打印 ============
        printf("=== CopyExpandedXandMXQuant: index=%lld, srcIdx=%d, dstIdx=%lld, progress=%lld ===\n",
               index, srcIdx, dstIdx, progress);
        // ============ DEBUG 打印结束 ============
        
        for (int64_t j = 0; j < colLoops_; j++) {
            int64_t loopCols = (j == colLoops_ - 1) ? lastLoopCols_ : perLoopCols_;
            uint32_t loopScaleCols = (j == colLoops_ - 1) ? lastLoopScaleCols_ : perLoopScaleCols_;
            uint32_t loopValidScaleCols = (j == colLoops_ - 1) ? lastLoopValidScaleCols_ : perLoopScaleCols_;
            
            // ============ DEBUG 打印 ============
            printf("  inner loop: j=%lld, loopCols=%lld, loopScaleCols=%u, loopValidScaleCols=%u\n",
                   j, loopCols, loopScaleCols, loopValidScaleCols);
            // ============ DEBUG 打印结束 ============
            
            CopyIn(srcIdx / k_, j, loopCols);
            Compute(loopCols, loopScaleCols, loopValidScaleCols);
            CopyOut(dstIdx, j, loopCols, loopScaleCols);
        }
    }
}