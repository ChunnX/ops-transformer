# MegaMoe Kernel 实现详解

## Kernel 入口

**文件**: `kernel_gmm_swiglu_mxquant.h`

```cpp
template <class BlockMmad_, class BlockMmad2_, class BlockEpilogue_, class BlockScheduler_>
class KernelGmmSwiGluMixOnlineDynamic {
public:
    CATLASS_DEVICE void operator()(const Params& params);
};
```

### 模板参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `BlockMmad_` | `BlockMmadTla` | 第一层 GEMM |
| `BlockMmad2_` | `BlockMmadTla` | 第二层 GEMM |
| `BlockEpilogue_` | `BlockEpilogueSwigluQuant` | SwiGLU + 量化 |
| `BlockScheduler_` | `GemmIdentityBlockSwizzle` | 分块调度器 |

### Params 结构体

```cpp
struct Params {
    GM_ADDR aGmAddr;           // 输入 token x
    GM_ADDR expertIdxGmAddr;   // 专家索引
    GM_ADDR bGmAddr;           // Weight1
    GM_ADDR bScaleGmAddr;      // Weight1 Scale
    GM_ADDR b2GmAddr;          // Weight2
    GM_ADDR b2ScaleGmAddr;     // Weight2 Scale
    GM_ADDR y2GmAddr;          // 输出 y
    GM_ADDR probsGmAddr;       // gating 概率
    WorkspaceInfo workspaceInfo;  // 工作区内存
    PeermemInfo peermemInfo;      // 远端共享内存
    HcclShmem shmem;              // HCCL 通信
    const MegaMoeTilingData *tilingData;  // Tiling 参数
};
```

## operator() 主流程

```cpp
CATLASS_DEVICE void operator()(const Params& params)
{
    // 1. 初始化参数和张量
    InitParamsAndTensor(params);
    
    // 2. InitRouting
    moe_init_routing_v3<bfloat16_t, fp8_e4m3fn_t>(...);
    
    // 3. 跨 Rank 同步
    CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2(params, ...);
    AivSyncAllForAic();
    
    // 4. 对每个专家组循环: GMM1 + SwiGLU + Quant + Dispatch
    for (uint32_t groupIdx = 0; groupIdx < expertPerRank; groupIdx++) {
        if (!UpdateGroupParams(params, groupIdx)) continue;
        UpdateGlobalBuffer(params);
        
        // AIV subBlockIdx=1: Dispatch
        if constexpr(g_coreType == AscendC::AIV) {
            if(subBlockIdx == 1) {
                Dispatch(params, groupIdx);
            }
        }
        
        // subBlockIdx=0: GMM1 + SwiGLU + Quant
        if (subBlockIdx == 0) {
            GroupMatmulSwigluQuant(params);
        }
    }
    
    // 5. 对每个专家组循环: GMM2 + Combine
    for (uint32_t groupIdx = 0; groupIdx < expertPerRank; groupIdx++) {
        if (!UpdateGroupParams(params, groupIdx)) continue;
        UpdateGlobalBuffer(params);
        
        if (subBlockIdx == 0) {
            GroupMatmul2(params, groupIdx);
        }
    }
    
    // 6. 同步和 Finalize
    AscendC::SyncAll<true>();
    moe_finalize_routing_v2<bfloat16_t>(...);
}
```

## InitParamsAndTensor

**功能**: 初始化参数和全局张量

```cpp
CATLASS_DEVICE void InitParamsAndTensor(const Params& params)
{
    // 初始化 baseOffset
    baseOffset = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    
    // 设置问题形状
    Get<N_VALUE>(problemShape) = tilingData->n;
    Get<K_VALUE>(problemShape) = tilingData->k;
    
    // 设置 tokenPerExpert 布局
    tokenPerExpertLayout = Layout3D(CeilAlign(tilingData->EP * tilingData->expertPerRank, ALIGN_128),
                                    tilingData->expertPerRank);
    
    // 设置全局张量
    preSumBeforeRank.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.ptrSumBeforeRank);
    tokenPerExpert.SetGlobalBuffer((__gm__ int32_t*)params.peermemInfo.ptrTokenPerExpert);
    cumsumMM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params.workspaceInfo.ptrcumsumMM));
    groupListGm = cumsumMM[(tilingData->EP - 1) * tilingData->expertPerRank];
    
    // 初始化 groupFlagList (清零)
    if constexpr(g_coreType == AscendC::AIV) {
        // ... 清零操作
    }
    
    // 初始化 Epilogue
    epilogueOp.Init({params.workspaceInfo.ptrA2, params.workspaceInfo.ptrA2Scale, ...});
    
    // 设置 dispatch 块数
    dispatchBlockNumPerEP = tilingData->blockNumPerEP;
}
```

## UpdateGroupParams

**功能**: 更新当前专家组的参数

```cpp
CATLASS_DEVICE bool UpdateGroupParams(const Params& params, uint32_t groupIdx)
{
    // 更新 baseOffset
    UpdateOffset(groupIdx);
    
    // 从 groupList 获取当前组的 m 值
    int32_t splitValue = GetSplitValueFromGroupList(groupIdx, params.tilingData->groupListType);
    Get<M_VALUE>(problemShape) = splitValue;
    
    // 如果 m=0，跳过该组
    if (Get<M_VALUE>(problemShape) == 0) {
        return false;
    }
    return true;
}
```

## UpdateOffset

**功能**: 更新 baseOffset 到下一个专家组

```cpp
CATLASS_DEVICE void UpdateOffset(uint32_t groupIdx)
{
    if (groupIdx == 0) return;
    
    uint64_t m = Get<M_VALUE>(problemShape);
    uint64_t n = Get<N_VALUE>(problemShape);
    uint64_t k = Get<K_VALUE>(problemShape);
    
    // 更新各个偏移量
    Get<IDX_A_OFFSET>(baseOffset) += m * k;
    Get<IDX_B_OFFSET>(baseOffset) += n * k;
    auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    Get<IDX_A_SCALE_OFFSET>(baseOffset) += m * scaleK;
    Get<IDX_B_SCALE_OFFSET>(baseOffset) += n * scaleK;
    Get<IDX_C_OFFSET>(baseOffset) += m * n / SWIGLU_N_HALF;
    Get<IDX_C_SCALE_OFFSET>(baseOffset) += m * CeilDiv(n / SWIGLU_N_HALF, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    Get<IDX_FLAG_OFFSET>(baseOffset) += 1;
    Get<IDX_B2_OFFSET>(baseOffset) += k * n / SWIGLU_N_HALF;
    Get<IDX_B2_SCALE_OFFSET>(baseOffset) += k * CeilDiv(n / SWIGLU_N_HALF, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    Get<IDX_Y2_OFFSET>(baseOffset) += m * k;
    Get<IDX_M_OFFSET>(baseOffset) += m;
}
```

## UpdateGlobalBuffer

**功能**: 根据 baseOffset 更新全局张量地址

```cpp
CATLASS_DEVICE void UpdateGlobalBuffer(const Params& params)
{
    // AIC: 更新 GEMM 相关张量
    if constexpr(g_coreType == AscendC::AIC) {
        aGlobal.SetGlobalBuffer((__gm__ ElementA*)params.workspaceInfo.ptrA + Get<IDX_A_OFFSET>(baseOffset));
        aScaleGlobal.SetGlobalBuffer((__gm__ ElementMxScaleA*)params.workspaceInfo.ptrAScale + Get<IDX_A_SCALE_OFFSET>(baseOffset));
        bGlobal.SetGlobalBuffer((__gm__ ElementB*)params.bGmAddr + Get<IDX_B_OFFSET>(baseOffset));
        bScaleGlobal.SetGlobalBuffer((__gm__ ElementMxScaleB*)params.bScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(baseOffset));
        a2Global.SetGlobalBuffer((__gm__ ElementA2*)params.workspaceInfo.ptrA2 + Get<IDX_C_OFFSET>(baseOffset));
        a2ScaleGlobal.SetGlobalBuffer((__gm__ ElementMxScaleA2*)params.workspaceInfo.ptrA2Scale + Get<IDX_C_SCALE_OFFSET>(baseOffset));
        b2Global.SetGlobalBuffer((__gm__ ElementB2*)params.b2GmAddr + Get<IDX_B2_OFFSET>(baseOffset));
        b2ScaleGlobal.SetGlobalBuffer((__gm__ ElementMxScaleB2*)params.b2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(baseOffset));
    }
    
    // AIV: 更新 Epilogue 相关张量
    if constexpr(g_coreType == AscendC::AIV) {
        AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
            Get<IDX_C_OFFSET>(baseOffset), Get<IDX_C_SCALE_OFFSET>(baseOffset),
            Get<IDX_FLAG_OFFSET>(baseOffset), 0L, 0L, 0L};
        epilogueOp.UpdateGlobalAddr(vecBaseOffset);
        outputGm.SetGlobalBuffer((__gm__ ElementMMadOut2*)params.y2GmAddr + Get<IDX_Y2_OFFSET>(baseOffset));
    }
    
    groupFlagListGm.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.ptrFlagList + Get<IDX_FLAG_OFFSET>(baseOffset) * INT_CACHELINE * MAX_AICORE_NUM);
    groupFlagListGm2.SetGlobalBuffer((__gm__ int32_t*)params.workspaceInfo.ptrFlagList2 + Get<IDX_FLAG_OFFSET>(baseOffset) * INT_CACHELINE * MAX_AICORE_NUM);
}
```

## Dispatch

**功能**: 跨卡拉取分配给本地专家的 token

```cpp
CATLASS_DEVICE void Dispatch(Params const &params, uint32_t groupIdx)
{
    int64_t widthA = Get<K_VALUE>(problemShape);
    int64_t widthAScale = CeilDiv(widthA, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    
    // 遍历所有 EP 的 dispatch 块
    for(int32_t idx = blockIdx; idx < tilingData->EP * dispatchBlockNumPerEP; idx += blockNum) {
        uint32_t dstEpIdx = idx / dispatchBlockNumPerEP;
        uint32_t idxInEp = idx % dispatchBlockNumPerEP;
        
        // 计算目标行范围
        uint32_t rowStart = (dstEpIdx == 0 ? 0 : cumsumMM((dstEpIdx - 1) * tilingData->expertPerRank + groupIdx));
        uint32_t rows = tokenPerExpert(tokenPerExpertLayout(dstEpIdx, tilingData->rank, groupIdx));
        
        // 计算每核的行范围
        uint32_t rowsPerCore = CeilDiv(rows, dispatchBlockNumPerEP);
        uint32_t rowStartPerCore = rowStart + idxInEp * rowsPerCore;
        uint32_t rowSrcPerCore = prevSum + idxInEp * rowsPerCore;
        
        if (rowsPerCore > 0) {
            // 设置远端地址
            AscendC::GlobalTensor<ElementA> gmRemoteA;
            gmRemoteA.SetGlobalBuffer(
                reinterpret_cast<__gm__ ElementA*>(params.shmem(params.peermemInfo.ptrA0, dstEpIdx)));
            
            // 计算偏移
            int64_t gmOffsetA = rowStartPerCore * widthA;
            int64_t gmOffsetAScale = rowStartPerCore * widthAScale;
            int64_t gmOffsetPeer = rowSrcPerCore * (widthA + widthAScale);
            
            // 从远端拷贝到本地
            CopyGMToGMPerTokenV2(aGlobal[gmOffsetA],
                                 aScaleGlobal[gmOffsetAScale],
                                 gmRemoteA[gmOffsetPeer],
                                 rowsPerCore, Get<K_VALUE>(problemShape),
                                 tilingData->dispatchRows, pingpongId);
        }
        
        // 原子增加 flag，通知 AIC
        AtomicAddFlag((__gm__ int32_t*)groupFlagListGm2.GetPhyAddr(), MAX_AICORE_NUM * INT_CACHELINE);
    }
}
```

## CopyGMToGMPerTokenV2

**功能**: 从远端拷贝 token 到本地 (支持 MXFP 格式)

```cpp
template<typename T, typename ElementMxScaleA>
CATLASS_DEVICE void CopyGMToGMPerTokenV2(
    AscendC::GlobalTensor<T> dst,
    AscendC::GlobalTensor<ElementMxScaleA> dstScale,
    AscendC::GlobalTensor<T> src,
    int32_t rows,
    int32_t hiddenSize,
    int32_t ubMoveNum,
    int32_t& pingpongId)
{
    // 双缓冲
    constexpr int32_t BufferNum = 2;
    AscendC::LocalTensor<T> tmpBuffer1{AscendC::TPosition::VECIN, 0, 0};
    AscendC::LocalTensor<T> tmpBuffer2{AscendC::TPosition::VECIN, 96 * 1024, 0};
    
    uint32_t scaleLen = CeilDiv(hiddenSize, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    uint32_t copyInNum = hiddenSize + scaleLen;
    auto processCount = CeilDiv(rows, ubMoveNum);
    
    for (uint32_t processIndex = 0; processIndex < processCount; ++processIndex) {
        pingpongId = (pingpongId + 1) % BufferNum;
        AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
        AscendC::LocalTensor<T> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
        AscendC::LocalTensor<ElementMxScaleA> bufScale = buf[hiddenSize].template ReinterpretCast<ElementMxScaleA>();
        
        // 计算当前批次的行数
        int32_t rowNum = ubMoveNum;
        if (processIndex == processCount - 1) {
            rowNum = rows - processIndex * ubMoveNum;
        }
        
        // 等待上一批完成
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        
        // 从远端读取 (data + scale)
        int64_t dataLen = rowNum * copyInNum;
        AscendC::DataCopy(buf, src[inputOffset], dataLen);
        
        // 等待读取完成
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        
        // 写入本地 (data 和 scale 分开)
        AscendC::DataCopyPad(dst[outputOffset], buf, {U16(rowNum), U16(hiddenSize), U16(scaleLen / 32), 0, 0});
        AscendC::DataCopyPad(dstScale[processIndex * ubMoveNum * scaleLen], bufScale, {U16(rowNum), U16(scaleLen), U16(hiddenSize / 32), 0, 0});
        
        // 设置完成标志
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
    }
}
```

## GroupMatmulSwigluQuant

**功能**: 第一层 GEMM + SwiGLU + 量化

```cpp
CATLASS_DEVICE void GroupMatmulSwigluQuant(const Params& params)
{
    uint32_t m = Get<M_VALUE>(problemShape);
    uint32_t n = Get<N_VALUE>(problemShape);
    uint32_t k = Get<K_VALUE>(problemShape);
    uint32_t outputN = n / SWIGLU_N_HALF;
    
    // 更新 Epilogue 的问题形状
    epilogueOp.UpdateNextProblem({m, outputN, k, 0});
    
    // 创建 BlockMmad
    BlockMmad blockMmad(resource);
    
    // 创建布局
    auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
    auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
    auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA, LayoutTagA, false>(m, scaleK);
    auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagB, true>(scaleK, n);
    auto layoutC = tla::MakeLayout<ElementMMadOut, LayoutTagC>(L1_TILE_M, L1_TILE_N);
    
    // 创建张量
    auto tensorA = tla::MakeTensor(aGlobal, layoutA, Arch::PositionGM{});
    auto tensorB = tla::MakeTensor(bGlobal, layoutB, Arch::PositionGM{});
    auto tensorMxScaleA = tla::MakeTensor(aScaleGlobal, layoutMxScaleA, Arch::PositionGM{});
    auto tensorMxScaleB = tla::MakeTensor(bScaleGlobal, layoutMxScaleB, Arch::PositionGM{});
    auto tensorUbFirst = tla::MakeTensor(l0cOutUbFirst, layoutC, Arch::PositionUB{});
    auto tensorUbSecond = tla::MakeTensor(l0cOutUbSecond, layoutC, Arch::PositionUB{});
    
    // 创建 BlockScheduler
    BlockScheduler matmulBlockScheduler({m, outputN, k}, MakeCoord(L1_TILE_M, L1_TILE_N));
    uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
    
    // 遍历所有 tile
    for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += blockNum) {
        GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
        GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
        uint32_t mLoc = blockCoord.m() * L1_TILE_M;
        uint32_t nLoc = blockCoord.n() * L1_TILE_N;
        uint32_t kLoc = blockCoord.k() * L1_TILE_K;
        
        // AIC: 执行 GEMM
        if constexpr(g_coreType == AscendC::AIC) {
            // 等待 Dispatch 完成
            if (loopIdx == startLoopIdx) {
                uint32_t targetValue = params.tilingData->EP * dispatchBlockNumPerEP;
                __gm__ int32_t* flagValueAddr = (__gm__ int32_t*)groupFlagListGm2.GetPhyAddr();
                while(targetValue != AscendC::ReadGmByPassDCache(flagValueAddr)) {
                    // 忙等
                }
            }
            
            // 等待 AIV 通知
            if (isVecSetSyncCom) {
                WaitForVector();
            }
            
            // 获取 tile
            auto tensorBlockA = GetTile(tensorA, tla::MakeCoord(mLoc, kLoc), ...);
            auto tensorBlockMxScaleA = GetTile(tensorMxScaleA, ...);
            auto tensorBlockUbFirst = GetTile(tensorUbFirst, ...);
            auto tensorBlockUbSecond = GetTile(tensorUbSecond, ...);
            
            // 对 SWIGLU_N_HALF=2 个权重块执行 GEMM
            for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                auto tensorBlockB = GetTile(tensorB, ...);
                auto tensorBlockMxScaleB = GetTile(tensorMxScaleB, ...);
                blockMmad(tensorBlockA, tensorBlockB, 
                          weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond,
                          actualBlockShape, tensorBlockMxScaleA, tensorBlockMxScaleB);
            }
            
            // 通知 AIV: GEMM 完成
            NotifyVector();
        }
        
        isVecSetSyncCom = true;
        
        // AIV: 执行 SwiGLU + 量化
        if constexpr(g_coreType == AscendC::AIV) {
            if (subBlockIdx == 0) {
                // 等待 AIC: GEMM 完成
                WaitForCube();
                
                // 调用 Epilogue
                AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{...};
                AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{...};
                epilogueOp(epilogueShape, epilogueOffset);
                
                // 通知 AIC: SwiGLU+Quant 完成
                NotifyCube();
            }
        }
    }
    
    startBlockIdx = (startBlockIdx + coreLoops) % blockNum;
}
```

## GroupMatmul2

**功能**: 第二层 GEMM + Combine (远端写回)

```cpp
CATLASS_DEVICE void GroupMatmul2(const Params& params, uint32_t groupIdx)
{
    uint32_t m = Get<M_VALUE>(problemShape);
    uint32_t n = Get<K_VALUE>(problemShape);
    uint32_t k = Get<N_VALUE>(problemShape) / SWIGLU_N_HALF;
    
    // 创建 BlockMmad2
    BlockMmad2 blockMmad(resource);
    
    // 创建布局和张量 (类似 GMM1)
    // ...
    
    // 创建 BlockScheduler
    BlockScheduler matmulBlockScheduler({m, n, k}, MakeCoord(L1_TILE_M2, L1_TILE_N2));
    uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
    
    // 遍历所有 tile
    for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += blockNum) {
        GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
        GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
        uint32_t mLoc = blockCoord.m() * L1_TILE_M2;
        uint32_t nLoc = blockCoord.n() * L1_TILE_N2;
        uint32_t kLoc = blockCoord.k() * L1_TILE_K2;
        
        // 选择 ping-pong 缓冲
        auto tensorUb = GMM2_PINGPONG_IDX == 0 ? tensorUbFirst : tensorUbSecond;
        auto l0cOutUbGMM2 = GMM2_PINGPONG_IDX == 0 ? l0cOutUbGMM2First : l0cOutUbGMM2Second;
        
        // AIC: 执行 GEMM
        if constexpr(g_coreType == AscendC::AIC) {
            // 等待 GMM1 完成 (通过 groupFlagListGm)
            if (loopIdx == startLoopIdx) {
                BlockScheduler gmmBlockScheduler({m, k, n}, MakeCoord(L1_TILE_M, L1_TILE_N));
                uint32_t targetLoops = gmmBlockScheduler.GetCoreLoops();
                __gm__ int32_t* flagValueAddr = (__gm__ int32_t*)groupFlagListGm.GetPhyAddr();
                while(targetLoops != AscendC::ReadGmByPassDCache(flagValueAddr)) {
                    // 忙等
                }
            }
            
            // 等待 AIV: 上一批 GEM2 Combine 完成
            if (VecSetSyncCom2 >= GMM2_PINGPONG_COUNT) {
                WaitForVector(GMM2_PINGPONG_IDX);
            }
            
            // 获取 tile
            auto tensorBlockA = GetTile(tensorA, ...);
            auto tensorBlockMxScaleA = GetTile(tensorMxScaleA, ...);
            auto tensorBlockUb = GetTile(tensorUb, ...);
            auto tensorBlockB = GetTile(tensorB, ...);
            auto tensorBlockMxScaleB = GetTile(tensorMxScaleB, ...);
            
            // 执行 GEMM
            blockMmad(tensorBlockA, tensorBlockB, tensorBlockUb,
                      actualBlockShape, tensorBlockMxScaleA, tensorBlockMxScaleB);
            
            // 通知 AIV: GEMM2 完成
            NotifyVector(GMM2_PINGPONG_IDX);
        }
        
        VecSetSyncCom2++;
        
        // AIV: 执行 Combine (远端写回)
        if constexpr(g_coreType == AscendC::AIV) {
            if (subBlockIdx == 0) {
                // 等待 AIC: GEMM2 完成
                WaitForCube(GMM2_PINGPONG_IDX);
                
                // 遍历所有 EP，将结果写回对应的卡
                int32_t preSumRankInExpert = 0;
                int32_t tileOffset = 0;
                
                for (int32_t dstEpIdx = 0; dstEpIdx < params.tilingData->EP; dstEpIdx++) {
                    // 获取当前 EP 在本 expert 中的 token 数
                    uint32_t lenRankInExpert = tokenPerExpert(tokenPerExpertLayout(dstEpIdx, params.tilingData->rank, groupIdx));
                    uint32_t dstExpertOffset = preSumBeforeRank(dstEpIdx * params.tilingData->expertPerRank + groupIdx);
                    uint32_t stRankInExpert = preSumRankInExpert;
                    uint32_t edRankInExpert = stRankInExpert + lenRankInExpert;
                    preSumRankInExpert += lenRankInExpert;
                    
                    // 计算当前 tile 与该 EP 的交集
                    if (stRankInExpert >= edTile) break;
                    else if (edRankInExpert <= stTile) continue;
                    
                    uint32_t stData = tla::max(stRankInExpert, stTile);
                    uint32_t edData = tla::min(edRankInExpert, edTile);
                    uint32_t lenData = edData - stData;
                    if (lenData <= 0) continue;
                    
                    // 计算目标偏移
                    uint32_t dstOffsetInExpert = 0;
                    if (stTile > stRankInExpert) {
                        dstOffsetInExpert = stTile - stRankInExpert;
                    }
                    
                    // 设置远端地址
                    AscendC::GlobalTensor<ElementMMadOut2> gmRemoteD;
                    __gm__ void* dstPeermemPtr = params.shmem(params.peermemInfo.ptrD, dstEpIdx);
                    gmRemoteD.SetGlobalBuffer(reinterpret_cast<__gm__ ElementMMadOut2*>(dstPeermemPtr));
                    
                    // 计算远端偏移
                    uint64_t gmDstOffset = (dstOffsetInExpert + dstExpertOffset) * n + nLoc;
                    auto gmTileD = gmRemoteD[gmDstOffset];
                    
                    // 写入远端
                    AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
                    ub2GmParams.blockCount = lenData;
                    ub2GmParams.blockLen = actualBlockShape.n() * sizeof(ElementMMadOut2);
                    ub2GmParams.dstStride = (n - actualBlockShape.n()) * sizeof(ElementMMadOut2);
                    AscendC::DataCopyPad(gmTileD, l0cOutUbGMM2[tileOffset * actualBlockShape.n()], ub2GmParams);
                    tileOffset += lenData;
                }
                
                // 通知 AIC: Combine 完成
                NotifyCube(GMM2_PINGPONG_IDX);
            }
        }
        
        // 切换 ping-pong
        GMM2_PINGPONG_IDX = 1 - GMM2_PINGPONG_IDX;
    }
    
    preSrcExpertSum += currentExpertM;
    startBlockIdx = (startBlockIdx + coreLoops) % blockNum;
}
```

## 同步机制

### AIC/AIV 同步

```cpp
// AIC 通知 AIV
CATLASS_DEVICE void NotifyVector(uint16_t value = 0) {
    AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + value);
}

// AIV 等待 AIC
CATLASS_DEVICE void WaitForCube(uint16_t value = 0) {
    AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG + value);
}

// AIV 通知 AIC
CATLASS_DEVICE void NotifyCube(uint16_t value = 0) {
    AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG + value);
}

// AIC 等待 AIV
CATLASS_DEVICE void WaitForVector(uint16_t value = 0) {
    AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + value);
}
```

### 跨 Rank 同步

```cpp
CATLASS_DEVICE void CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2(Params const &params, GM_ADDR tokenPerExpertCurRank)
{
    // 1. AllGather: 广播 tokenPerExpert 到所有卡
    for(int32_t dstEpIdx = aivCoreIdx; dstEpIdx < tilingData->EP; dstEpIdx += blockNum) {
        if (dstEpIdx == tilingData->rank) continue;
        
        // 读取本地 tokenPerExpert
        AscendC::DataCopy(tmpBuffer, srcAddress, numPerCore);
        
        // 添加标记 (0x800000)
        AscendC::Adds(tmpBuffer, tmpBuffer, 0x800000, numPerCore);
        
        // 写入远端
        AscendC::DataCopy(dstAddress, tmpBuffer, numPerCore);
    }
    
    // 2. 等待所有卡完成
    for(int32_t dstEpIdx = aivCoreIdx; dstEpIdx < tilingData->EP; dstEpIdx += blockNum) {
        if (dstEpIdx != tilingData->rank) {
            // 等待远端数据就绪
            gm_signal_wait_until_ne(sync_check, 0);
            
            // 读取远端数据
            AscendC::DataCopy(tmpBuffer, tokenPerExpert[tokenPerExpertLayout(dstEpIdx, 0, 0)], numPerCore);
            
            // 移除标记
            AscendC::Adds(tmpBuffer, tmpBuffer, -0x800000, numPerCore);
        }
        
        // 3. 计算前缀和
        int32_t tmpSum = 0;
        for (int32_t i = 0; i < (tilingData->rank + 1) * tilingData->expertPerRank; i++) {
            if (i >= tilingData->rank * tilingData->expertPerRank) {
                prevSumBuf(j) = tmpSum;
                j++;
            }
            tmpSum += tmpBuffer(i);
        }
        
        // 写入 preSumBeforeRank
        AscendC::DataCopyPad(preSumBeforeRank[dstEpIdx * tilingData->expertPerRank], prevSumBuf, ...);
    }
    
    // 4. 全局同步
    AscendC::SyncAll<true>();
    
    // 5. 计算 cumsumMM
    if (aivCoreIdx == 0) {
        GetCumsumForMMAIV(tokenPerExpert, cumsumMM, tilingData->expertPerRank, tilingData->rank, tilingData->EP);
    }
    
    // 6. 设置 prevSum
    if (blockIdx < tilingData->EP * dispatchBlockNumPerEP) {
        prevSum = preSumBeforeRank(blockIdx / dispatchBlockNumPerEP * tilingData->expertPerRank);
    }
}
```


让我用更详细的例子和图来解释：
场景设定
假设：
- blockNum = 8 (8个核心同时工作)
- coreLoops = 20 (共20个矩阵块需要计算)
- startBlockIdx = 6 (本轮从核心6开始，核心0-5在上一轮已经处理过了)
标准 Catlass 模式 (startBlockIdx = 0)
核心0: loopIdx = 0, 8, 16
核心1: loopIdx = 1, 9, 17
核心2: loopIdx = 2, 10, 18
核心3: loopIdx = 3, 11, 19
核心4: loopIdx = 4, 12  (结束)
核心5: loopIdx = 5, 13  (结束)
核心6: loopIdx = 6, 14  (结束)
核心7: loopIdx = 7, 15  (结束)
每个核心处理的block分布：
核心0: [0] [8] [16]
核心1: [1] [9] [17]
核心2: [2] [10] [18]
核心3: [3] [11] [19]
核心4: [4] [12]
核心5: [5] [13]
核心6: [6] [14]
核心7: [7] [15]
MegaMoe 模式 (startBlockIdx = 6)
现在 startBlockIdx = 6，意味着核心6,7,0,1,2,3,4,5 按这个顺序参与本轮计算。
公式计算 startLoopIdx
核心0: (0 < 6) → 0 + 8 - 6 = 2
核心1: (1 < 6) → 1 + 8 - 6 = 3
核心2: (2 < 6) → 2 + 8 - 6 = 4
核心3: (3 < 6) → 3 + 8 - 6 = 5
核心4: (4 < 6) → 4 + 8 - 6 = 6
核心5: (5 < 6) → 5 + 8 - 6 = 7
核心6: (6 < 6) false → 6 - 6 = 0
核心7: (7 < 6) false → 7 - 6 = 1
核心分配的 loopIdx
核心6 (startLoopIdx=0):  loopIdx = 0, 8, 16
核心7 (startLoopIdx=1):  loopIdx = 1, 9, 17
核心0 (startLoopIdx=2):  loopIdx = 2, 10, 18
核心1 (startLoopIdx=3):  loopIdx = 3, 11, 19
核心2 (startLoopIdx=4):  loopIdx = 4, 12
核心3 (startLoopIdx=5):  loopIdx = 5, 13
核心4 (startLoopIdx=6):  loopIdx = 6, 14
核心5 (startLoopIdx=7):  loopIdx = 7, 15
图示对比
标准模式 (startBlockIdx = 0)
核心编号:  0   1   2   3   4   5   6   7
         ┌───┬───┬───┬───┬───┬───┬───┐
第1轮(0-7):│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │
         ├───┼───┼───┼───┼───┼───┼───┼───┤
第2轮(8-15):│ 8 │ 9 │10 │11 │12 │13 │14 │15 │
         ├───┼───┼───┼───┼───┼───┼───┼───┤
第3轮(16-19):│16 │17 │18 │19 │   │   │   │   │
         └───┴───┴───┴───┴───┴───┴───┘
MegaMoe 模式 (startBlockIdx = 6)
核心顺序:  6   7   0   1   2   3   4   5  (环形)
         ┌───┬───┬───┬───┬───┬───┬───┐
第1轮:   │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │
         ├───┼───┼───┼───┼───┼───┼───┼───┤
第2轮:   │ 8 │ 9 │10 │11 │12 │13 │14 │15 │
         ├───┼───┼───┼───┼───┼───┼───┼───┤
第3轮:   │16 │17 │18 │19 │   │   │   │   │
         └───┴───┴───┴───┴───┴───┴───┘
注意: 虽然起始核心是6，但通过 startLoopIdx 的调整，每个核心最终处理的 block 集合是相同的！
核心公式理解
startLoopIdx = (blockIdx < startBlockIdx ? blockIdx + blockNum : blockIdx) - startBlockIdx
本质: 把"环形索引"转换成"线性索引"
原始索引空间:    0   1   2   3   4   5   6   7
                ↑                   ↑
                │                   │
                └──── 环形 wrap ────┘
当 startBlockIdx = 6:
                6   7   0   1   2   3   4   5
                ├───┼───┼───┼───┼───┼───┼───┤
线性索引:        0   1   2   3   4   5   6   7
通过公式，把原本的核心索引映射到新的起点，让每个核心从"自己的位置"开始遍历。