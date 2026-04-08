ge::graphStatus MaskedCausalConv1dBackwardTiling::ComputeIntraCoreUbTiling()
{
    // UB 预留系统空间
    int64_t availableUbSize = static_cast<int64_t>(ubSize_) - SYSTEM_RESERVED_UB_SIZE;
    OP_LOGI(context_->GetNodeName(), "ubSize is %ld", ubSize_);
    OP_LOGI(context_->GetNodeName(), "availableUbSize is %ld", availableUbSize);
    OP_CHECK_IF(availableUbSize <= 0, OP_LOGE(context_->GetNodeName(), "available UB size <= 0, ubSize=%lu", ubSize_),
                return ge::GRAPH_FAILED);

    // UB先按H=64(128B)，全载核内BS
    int64_t minH = DIM_ALIGN_ELEMENT;  // 64
    int64_t perCoreH = (hMainCoreCnt_ > 0) ? hMainSize_ : hTailSize_;

    // 核内仅切 S（bUB 固定为 1）
    // 注意：sUB_ 存储的是有效输出行数（不包含重叠部分）
    hUB_ = minH;
    bUB_ = 1;

    // Buffer占用估算函数
    // 参数 s_effective 是有效输出行数（不包含重叠部分）
    // 修复：内部动态计算 sLoopCnt，不再依赖外部未赋值的成员变量
    auto calculateBufferSize = [&](int64_t h, int64_t b, int64_t s_effective) -> int64_t {
        // 【关键修复】根据当前传入的s切分大小，动态计算循环次数，不再使用未初始化的sLoopCnt_
        int64_t currentSLoopCnt = (S_ + s_effective - 1) / s_effective;
        
        int64_t s_with_overlap = s_effective + (W_ - 1);
        int64_t gradOutputSize = h * b * s_with_overlap * static_cast<int64_t>(dtypeSize_);
        int64_t inputSize = h * b * s_effective * static_cast<int64_t>(dtypeSize_);
        int64_t weightSize = h * W_ * static_cast<int64_t>(dtypeSize_);
        int64_t maskSize = 0;
        if (hasMask_) {
            int64_t maskBytes = b * s_with_overlap;
            maskSize = ((maskBytes + ALIGN_BYTES - 1) / ALIGN_BYTES) * ALIGN_BYTES;
        }
        int64_t gradInputSize = h * b * s_effective * static_cast<int64_t>(dtypeSize_);
        int64_t gradWeightSize = h * W_ * static_cast<int64_t>(dtypeSize_);
        int64_t tempBuffSize = h * W_ * B_ * sizeof(float);
        int64_t prodBufSize = W_ * s_effective * h * sizeof(float);
        int64_t sumBufSize = h * sizeof(float);

        int64_t wsSize = 3LL * currentSLoopCnt * h * sizeof(float);
        int64_t totalSize = BUFFER_NUM * (gradOutputSize + inputSize + weightSize + maskSize +
                                          gradInputSize + gradWeightSize) + tempBuffSize
                                          + prodBufSize + sumBufSize + wsSize;
        OP_LOGI(context_->GetNodeName(),"gradOutputSize = %ld", gradOutputSize);
        OP_LOGI(context_->GetNodeName(),"inputSize = %ld", inputSize);
        OP_LOGI(context_->GetNodeName(),"weightSize = %ld", weightSize);
        OP_LOGI(context_->GetNodeName(),"maskSize = %ld", maskSize);
        OP_LOGI(context_->GetNodeName(),"gradInputSize = %ld", gradInputSize);
        OP_LOGI(context_->GetNodeName(),"gradWeightSize = %ld", gradWeightSize);
        OP_LOGI(context_->GetNodeName(),"tempBuffSize = %ld", tempBuffSize);
        OP_LOGI(context_->GetNodeName(),"prodBufSize = %ld", prodBufSize);
        OP_LOGI(context_->GetNodeName(),"sumBufSize = %ld", sumBufSize);
        OP_LOGI(context_->GetNodeName(),"wsSize = %ld", wsSize);
        OP_LOGI(context_->GetNodeName(),"totalSize = %ld", totalSize);
        return totalSize;
    };

    // 在 bUB_=1 下，按 UB 容量求最大的 sUB_（考虑 mask 的32字节对齐）
    // 用 calculateBufferSize 作为真实成本函数，执行二分搜索以找到最大 sUB_，避免线性近似误差
    int64_t lowS = 1;
    int64_t highS = std::max<int64_t>(1, S_);
    while (lowS <= highS) {
        int64_t midS = (lowS + highS) / 2;
        int64_t need = calculateBufferSize(hUB_, bUB_, midS);
        if (need <= availableUbSize) {
            sUB_ = midS;        // 可行，尝试更大
            lowS = midS + 1;
        } else {
            highS = midS - 1;  // 不可行，缩小范围
        }
    }
    // 保护：不超过 S_
    if (sUB_ > S_) sUB_ = S_;

    // 循环次数（B 不切分，尾块同理）
    hLoopCnt_ = (perCoreH + hUB_ - 1) / hUB_;
    sLoopCnt_ = (S_ + sUB_ - 1) / sUB_;

    // 主块与尾块
    ubMainFactorH_ = hUB_;
    ubTailFactorH_ = (hLoopCnt_ == 1) ? perCoreH : (perCoreH % hUB_ == 0 ? hUB_ : perCoreH % hUB_);

    ubMainFactorB_ = 1;
    ubTailFactorB_ = 1;

    // 有效输出（不含重叠）
    ubMainFactorS_ = sUB_;
    int64_t s_effective_remain = S_ % sUB_;
    ubTailFactorS_ = (s_effective_remain == 0 && sLoopCnt_ > 0) ? sUB_ : (s_effective_remain > 0 ? s_effective_remain : sUB_);

    // 尾核（保持 B 参数为 1）
    if (hTailCoreCnt_ > 0 && hTailSize_ > 0) {
        // 尾核可能需要不同的H循环次数
        int64_t tailCoreH = hTailSize_;
        tailHLoopCnt_ = (tailCoreH + hUB_ - 1) / hUB_;
        tailSLoopCnt_ = sLoopCnt_;

        tailCoreUbMainFactorH_ = hUB_;
        tailCoreUbTailFactorH_ = (tailHLoopCnt_ == 1) ? hUB_ :
                                 (tailCoreH % hUB_ == 0 ? hUB_ : tailCoreH % hUB_);

        tailCoreUbMainFactorB_ = ubMainFactorB_;
        tailCoreUbTailFactorB_ = ubTailFactorB_;

        tailCoreUbMainFactorS_ = ubMainFactorS_;  // 有效输出大小，不包含重叠
        tailCoreUbTailFactorS_ = ubTailFactorS_;  // 有效输出大小，不包含重叠
    } else {
        // 没有尾核，或尾核参数与主核相同
        tailHLoopCnt_ = hLoopCnt_;
        tailSLoopCnt_ = sLoopCnt_;

        tailCoreUbMainFactorH_ = ubMainFactorH_;
        tailCoreUbTailFactorH_ = ubTailFactorH_;
        tailCoreUbMainFactorB_ = ubMainFactorB_;
        tailCoreUbTailFactorB_ = ubTailFactorB_;
        tailCoreUbMainFactorS_ = ubMainFactorS_;
        tailCoreUbTailFactorS_ = ubTailFactorS_;
    }

    allUb_ = calculateBufferSize(ubMainFactorH_, ubMainFactorB_, ubMainFactorS_);
    OP_LOGI(context_->GetNodeName(), "allUb_ = %ld", allUb_);
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus MaskedCausalConv1dBackwardTiling::DoOpTiling()
{
    OP_CHECK_IF(ComputeInterCoreSplit() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "ComputeInterCoreSplit failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(ComputeIntraCoreUbTiling() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "ComputeIntraCoreUbTiling failed"), return ge::GRAPH_FAILED);

    // 核间切分参数
    tilingData_.hMainCoreCnt = hMainCoreCnt_;
    tilingData_.hTailCoreCnt = hTailCoreCnt_;
    tilingData_.hMainSize = hMainSize_;
    tilingData_.hTailSize = hTailSize_;

    // 主核循环参数
    tilingData_.hLoopCnt = hLoopCnt_;
    tilingData_.sLoopCnt = sLoopCnt_;

    // 主核UB切块参数
    tilingData_.ubMainFactorH = ubMainFactorH_;
    tilingData_.ubTailFactorH = ubTailFactorH_;
    tilingData_.ubMainFactorB = ubMainFactorB_;
    tilingData_.ubTailFactorB = ubTailFactorB_;
    tilingData_.ubMainFactorS = ubMainFactorS_;
    tilingData_.ubTailFactorS = ubTailFactorS_;

    tilingData_.tailHLoopCnt = tailHLoopCnt_;
    tilingData_.tailSLoopCnt = tailSLoopCnt_;

    tilingData_.tailCoreUbMainFactorH = tailCoreUbMainFactorH_;
    tilingData_.tailCoreUbTailFactorH = tailCoreUbTailFactorH_;
    tilingData_.tailCoreUbMainFactorB = tailCoreUbMainFactorB_;
    tilingData_.tailCoreUbTailFactorB = tailCoreUbTailFactorB_;
    tilingData_.tailCoreUbMainFactorS = tailCoreUbMainFactorS_;
    tilingData_.tailCoreUbTailFactorS = tailCoreUbTailFactorS_;

    // 全局参数
    tilingData_.hasMask = hasMask_;
    tilingData_.S = S_;
    tilingData_.B = B_;
    tilingData_.H = H_;
    tilingData_.W = W_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MaskedCausalConv1dBackwardTiling::GetTilingKey() const
{
    // 可按 dtype 区分，当前返回 0
    if (dataType_ == ge::DataType::DT_BF16) {
        return TILING_KEY_MASK_EDCAUSAL_CONV1D_BACKWARD_BF16;
    } else if (dataType_ == ge::DataType::DT_FLOAT16) {
        return TILING_KEY_MASK_EDCAUSAL_CONV1D_BACKWARD_FP16;
    }
    
}

ge::graphStatus MaskedCausalConv1dBackwardTiling::GetWorkspaceSize()
{
    auto platformInfo = context_->GetPlatformInfo();
    uint32_t sysWorkspaceSize = 0;
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    }
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(0UL + sysWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaskedCausalConv1dBackwardTiling::PostTiling()
{
    // 设置核数
    context_->SetBlockDim(static_cast<uint32_t>(usedCoreNum_));

    // 保存 tiling 数据
    auto *raw = context_->GetRawTilingData();
    auto tilingDataSize = sizeof(MaskedCausalConv1dBackwardTilingDataV35);
    errno_t ret = memcpy_s(raw->GetData(), raw->GetCapacity(), reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    raw->SetDataSize(tilingDataSize);

    context_->SetTilingKey(GetTilingKey());
    return ge::GRAPH_SUCCESS;
}

void MaskedCausalConv1dBackwardTiling::DumpTilingInfo()
{
    OP_LOGI(context_->GetNodeName(), "=== MaskedCausalConv1dBackward DumpTilingInfo ===");

    OP_LOGI(context_->GetNodeName(), "hMainCoreCnt = %ld", hMainCoreCnt_);
    OP_LOGI(context_->GetNodeName(), "hTailCoreCnt = %ld", hTailCoreCnt_);
    OP_LOGI(context_->GetNodeName(), "hMainSize = %ld", hMainSize_);
    OP_LOGI(context_->GetNodeName(), "hTailSize = %ld", hTailSize_);
    OP_LOGI(context_->GetNodeName(), "hLoopCnt = %ld", hLoopCnt_);
    OP_LOGI(context_->GetNodeName(), "bLoopCnt = %ld", bLoopCnt_);
    OP_LOGI(context_->GetNodeName(), "sLoopCnt = %ld", sLoopCnt_);
    OP_LOGI(context_->GetNodeName(), "ubMainFactorH = %ld", ubMainFactorH_);
    OP_LOGI(context_->GetNodeName(), "ubTailFactorH = %ld", ubTailFactorH_);
    OP_LOGI(context_->GetNodeName(), "ubMainFactorB = %ld", ubMainFactorB_);
    OP_LOGI(context_->GetNodeName(), "ubTailFactorB = %ld", ubTailFactorB_);
    OP_LOGI(context_->GetNodeName(), "ubMainFactorS = %ld", ubMainFactorS_);
    OP_LOGI(context_->GetNodeName(), "ubTailFactorS = %ld", ubTailFactorS_);
    OP_LOGI(context_->GetNodeName(), "tailHLoopCnt = %ld", tailHLoopCnt_);
    OP_LOGI(context_->GetNodeName(), "tailBLoopCnt = %ld", tailBLoopCnt_);
    OP_LOGI(context_->GetNodeName(), "tailSLoopCnt = %ld", tailSLoopCnt_);
    OP_LOGI(context_->GetNodeName(), "tailCoreUbMainFactorH = %ld", tailCoreUbMainFactorH_);
    OP_LOGI(context_->GetNodeName(), "tailCoreUbTailFactorH = %ld", tailCoreUbTailFactorH_);
    OP_LOGI(context_->GetNodeName(), "tailCoreUbMainFactorB = %ld", tailCoreUbMainFactorB_);
    OP_LOGI(context_->GetNodeName(), "tailCoreUbTailFactorB = %ld", tailCoreUbTailFactorB_);
    OP_LOGI(context_->GetNodeName(), "tailCoreUbMainFactorS = %ld", tailCoreUbMainFactorS_);
    OP_LOGI(context_->GetNodeName(), "tailCoreUbTailFactorS = %ld", tailCoreUbTailFactorS_);
    OP_LOGI(context_->GetNodeName(), "hasMask = %ld", hasMask_);
    OP_LOGI(context_->GetNodeName(), "S = %ld", S_);
    OP_LOGI(context_->GetNodeName(), "B = %ld", B_);
    OP_LOGI(context_->GetNodeName(), "H = %ld", H_);
    OP_LOGI(context_->GetNodeName(), "W = %ld", W_);

}