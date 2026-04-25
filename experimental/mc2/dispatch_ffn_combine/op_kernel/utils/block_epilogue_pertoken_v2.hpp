#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_V2_ONLY_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_V2_ONLY_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/callback.hpp"

#include "hccl_shmem.hpp"
#include "layout3d.hpp"

namespace Catlass::Epilogue::Block {

template<bool IS_A2_>
struct IS_A2_T {
    static constexpr bool Value = IS_A2_;
};

template <
    uint32_t UB_STAGES_,
    class CType_,
    class LayoutPerTokenScale_,
    class DType_,
    class TileCopy_,
    bool IS_A2_
>
class BlockEpilogue <
    EpilogueAtlasA2PerTokenDequantV2<UB_STAGES_>,
    CType_,
    Gemm::GemmType<float, LayoutPerTokenScale_>,
    DType_,
    TileCopy_,
    IS_A2_T<IS_A2_>
> {
public:
    using DispatchPolicy = EpilogueAtlasA2PerTokenDequantV2<UB_STAGES_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
    static constexpr bool IS_A2 = IS_A2_T<IS_A2_>::Value;

    // Data infos
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = LayoutPerTokenScale_;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    using CopyScaleGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, Gemm::GemmType<float, layout::VectorLayout>>;
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;

    struct Params {
        __gm__ int32_t *ptrTokenPerExpert{nullptr};
        int32_t EP;
        int32_t expertPerRank;
        int32_t n2;
        LayoutC layoutC;
        int32_t n0;
        int32_t rank;
        HcclShmem<IS_A2> shmem;
        int32_t offsetD;

        CATLASS_DEVICE
        Params() {};
        CATLASS_DEVICE
        Params(int32_t EP_, int32_t expertPerRank_, int32_t rank_, __gm__ int32_t *ptrTokenPerExpert_, 
        LayoutC layoutC_, int32_t n2_, int32_t n0_, HcclShmem<IS_A2>& shmem_, int32_t offsetD_) : 
        ptrTokenPerExpert(ptrTokenPerExpert_), EP(EP_), 
        expertPerRank(expertPerRank_),rank(rank_), layoutC(layoutC_), n2(n2_), n0(n0_),
        shmem(shmem_), offsetD(offsetD_)
         {}
    };


    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource, Params const &params = Params{}) : params(params)
    {
        //ub:192KB
        n0 = params.n0;
        size_t ubOffset = 0;
        for(int32_t i = 0; i < 2; i++) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += max_len * sizeof(ElementC);
            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += max_len * sizeof(ElementD);
            ubFp32List[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += max_len * sizeof(float);
            scaleUbList[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += (max_len / n0) * sizeof(float);
            source_scale_offset[i] = -1;
        }
        tokenPerExpert.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.ptrTokenPerExpert));
        tokenPerExpertLayout = Layout3D(AlignUp(params.EP * params.expertPerRank, 128), params.expertPerRank);
        is_ping = true;
    }

    CATLASS_DEVICE
    void Finalize()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);

    }
    CATLASS_DEVICE
    ~BlockEpilogue()
    {
        
    }
    CATLASS_DEVICE
    void operator() (
        AscendC::GlobalTensor<ElementC> const &gmC,
        AscendC::GlobalTensor<ElementPerTokenScale> const &gmPerTokenScale,
        GemmCoord& blockCoord,
        GemmCoord& actualBlockShape,
        int32_t groupIdx,
        int32_t preSrcExpertSum,
        AscendC::GlobalTensor<int32_t> preSumBeforeRank
    ){
        is_ping = !is_ping;
        auto event_id = is_ping ? EVENT_ID0 : EVENT_ID1;
        auto event_id_2 = is_ping ? EVENT_ID2 : EVENT_ID3;

        auto &ubC = ubCList[is_ping];
        auto &ubD = ubDList[is_ping];
        int32_t gmCOffset = preSrcExpertSum * params.n2 + blockCoord.m() * params.n2 + blockCoord.n();
        auto gmTileC = gmC[gmCOffset];
        auto &ubCFp32 = ubFp32List[is_ping];
        auto &scaleUb = scaleUbList[is_ping];

        LayoutC layoutGM{actualBlockShape.m(), actualBlockShape.n(), params.n2};
        LayoutC layoutUB{actualBlockShape.m(), actualBlockShape.n(), n0};


        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id);
        copyGmToUbC(ubC, gmTileC, layoutUB, layoutGM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
        AscendC::Cast<float, ElementC, false>(ubCFp32, ubC, AscendC::RoundMode::CAST_NONE, -1, repeat, {1, 1, 8, 4});
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id);


        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id_2);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(event_id_2);

        int32_t gmScaleOffset = preSrcExpertSum + blockCoord.m();
        layout::VectorLayout scaleLauout{actualBlockShape.m()};
        if (source_scale_offset[event_id] != gmScaleOffset) {
                source_scale_offset[event_id] = gmScaleOffset;
                copyScaleGmToUb(scaleUb, gmPerTokenScale[gmScaleOffset],  scaleLauout, scaleLauout);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(event_id_2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id_2);

        

        
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id_2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(event_id_2); // Note that the value must be MTE2_S instead of MTE2_V.
                                                                   // Otherwise, 0 will be read, causing garbled characters.
        AscendC::PipeBarrier<PIPE_V>();
        for (int32_t row = 0; row < actualBlockShape.m(); ++row) {
                float scale = scaleUb(row);
                Muls<float, false>(ubCFp32[n0* row], ubCFp32[n0 * row] , scale, -1, (actualBlockShape.n() + 127) / 128 * 2, {1, 1, 8, 8});
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event_id);
        AscendC::Cast<ElementD, float, false>(ubD, ubCFp32, AscendC::RoundMode::CAST_RINT, -1, repeat, {1, 1, 4, 8});
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(event_id_2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id_2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id);

        int32_t lenTile = actualBlockShape.m();
        int32_t stTile = blockCoord.m();
        int32_t edTile = stTile + lenTile;
        int32_t preSumRankInExpert = 0;
        int32_t tileOffset = 0;

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id);
        for (int32_t dstEpIdx = 0; dstEpIdx < params.EP; dstEpIdx ++) {
            int32_t lenRankInExpert = tokenPerExpert(tokenPerExpertLayout(dstEpIdx, params.rank, groupIdx));
            int32_t dstExpertOffset = preSumBeforeRank(dstEpIdx * params.expertPerRank + groupIdx);
            int32_t stRankInExpert = preSumRankInExpert;
            int32_t edRankInExpert = stRankInExpert + lenRankInExpert;
            preSumRankInExpert += lenRankInExpert;
            if (stRankInExpert >= edTile) {
                break;
            }
            else if (edRankInExpert <= stTile) {
                continue;
            }
            int32_t stData = max(stRankInExpert, stTile);
            int32_t edData = min(edRankInExpert, edTile);
            uint32_t lenData = edData - stData;
            if (lenData <= 0){
                continue;
            }
            
            uint32_t dstOffsetInExpert = 0;
            if (stTile > stRankInExpert) {
                dstOffsetInExpert = stTile - stRankInExpert;
            }
            AscendC::GlobalTensor<ElementD> gmRemotePeer;
            __gm__ void* dstPeermemPtr = params.shmem(params.offsetD, dstEpIdx);
            gmRemotePeer.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD*>(dstPeermemPtr));
            MatrixCoord dstOffset{dstOffsetInExpert + dstExpertOffset, blockCoord.n()};
            int64_t gmDstOffset = params.layoutC.GetOffset(dstOffset);
            auto gmTileD = gmRemotePeer[gmDstOffset];
            LayoutC layoutGM2{lenData, actualBlockShape.n(), params.n2};
            LayoutC layoutUB2{lenData, actualBlockShape.n(), n0};
            copyUbToGmD(gmTileD, ubD[tileOffset *  n0], layoutGM2, layoutUB2);
            tileOffset += lenData;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event_id);

    }

    // ==================== operatorA2: A2 专用版本 ====================
    // 功能：处理任意大小的 actualBlockShape，按 16 行分块处理
    // 用于 CombineV3 调用，支持 A2 架构的特定优化
    // 
    // 处理流程（每个 16 行子块）：
    //   1. 从 gmC 读取量化数据到 UB
    //   2. 类型转换：量化格式 → float
    //   3. 读取 scale 并执行反量化
    //   4. 类型转换：float → 输出格式
    //   5. 数据写入 windowOut
    CATLASS_DEVICE
    void operatorA2 (
        AscendC::GlobalTensor<ElementC> const &gmC,              // GMM2 输出（量化数据）
        AscendC::GlobalTensor<ElementPerTokenScale> const &gmPerTokenScale,  // 反量化 scale
        GemmCoord& blockCoord,                                   // 当前块坐标
        GemmCoord& actualBlockShape,                             // 当前块实际大小
        int32_t groupIdx,                                        // 当前 expert 组索引
        int32_t preSrcExpertSum,                                 // 已处理的源 expert token 总数
        AscendC::GlobalTensor<int32_t> preSumBeforeRank,          // 各 rank 在各 expert 的起始偏移
        AscendC::GlobalTensor<ElementD> gmSrcD                    // 源 expert 数据（反量化后）
    ){
        // ==================== 0. 分块参数设置 ====================
        // m0: 每个子块的行数（固定为 16）
        // 由于 UB 缓冲区大小限制，每次最多处理 16 行数据
        constexpr int32_t m0 = 16;
        
        // 获取当前块的总行数
        int32_t totalRows = actualBlockShape.m();
        
        // 计算需要多少次迭代（向上取整）
        int32_t numSubTiles = (totalRows + m0 - 1) / m0;

        // 将所有计算完毕的tile写入本地 windowsOut区（RDMA-accessible
        
        // ==================== 遍历每个 16 行子块 ====================
        for (int32_t subTileIdx = 0; subTileIdx < numSubTiles; ++subTileIdx) {
            // ==================== 1. 计算当前子块的参数 ====================
            // 当前子块的起始行（在当前块内）
            int32_t subTileStartRow = subTileIdx * m0;
            
            // 当前子块的行数（最后一个子块可能不足 16 行）
            int32_t subTileRows = min(m0, totalRows - subTileStartRow);
            
            // 当前子块在全局 token 中的起始行
            int32_t globalStartRow = blockCoord.m() + subTileStartRow;
            
            // ==================== 2. Ping-Pong 缓冲区选择 ====================
            is_ping = !is_ping;
            auto event_id = is_ping ? EVENT_ID0 : EVENT_ID1;
            auto event_id_2 = is_ping ? EVENT_ID2 : EVENT_ID3;

            auto &ubC = ubCList[is_ping];
            auto &ubD = ubDList[is_ping];
            auto &ubCFp32 = ubFp32List[is_ping];
            auto &scaleUb = scaleUbList[is_ping];

            // ==================== 3. 计算全局内存偏移 ====================
            // gmCOffset = 已处理 token 总数 × n2 + (块起始行 + 子块起始行) × n2 + 块起始列
            int32_t gmCOffset = preSrcExpertSum * params.n2 + globalStartRow * params.n2 + blockCoord.n();
            auto gmTileC = gmC[gmCOffset];

            // ==================== 4. 设置内存布局 ====================
            // 子块的内存布局
            LayoutC layoutGM{subTileRows, actualBlockShape.n(), params.n2};
            LayoutC layoutUB{subTileRows, actualBlockShape.n(), n0};

            // ==================== 5. 从 GM 读取量化数据到 UB ====================
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id);
            copyGmToUbC(ubC, gmTileC, layoutUB, layoutGM);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);

            // ==================== 6. 类型转换：量化格式 → float ====================
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
            AscendC::Cast<float, ElementC, false>(ubCFp32, ubC, AscendC::RoundMode::CAST_NONE, -1, repeat, {1, 1, 8, 4});
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id);

            // ==================== 7. 读取 scale 数据 ====================
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id_2);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(event_id_2);

            // 计算 scale 在全局内存中的偏移
            int32_t gmScaleOffset = preSrcExpertSum + globalStartRow;
            layout::VectorLayout scaleLayout{subTileRows};
            
            // Scale 缓存优化：如果偏移未变，复用之前读取的 scale
            if (source_scale_offset[event_id] != gmScaleOffset) {
                source_scale_offset[event_id] = gmScaleOffset;
                copyScaleGmToUb(scaleUb, gmPerTokenScale[gmScaleOffset], scaleLayout, scaleLayout);
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(event_id_2);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id_2);

            // ==================== 8. 执行反量化计算 ====================
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id_2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(event_id_2);
            
            AscendC::PipeBarrier<PIPE_V>();
            
            // 逐行执行反量化
            for (int32_t row = 0; row < subTileRows; ++row) {
                float scale = scaleUb(row);
                Muls<float, false>(ubCFp32[n0 * row], ubCFp32[n0 * row], scale, -1, (actualBlockShape.n() + 127) / 128 * 2, {1, 1, 8, 8});
            }
            
            AscendC::PipeBarrier<PIPE_V>();

            // ==================== 9. 类型转换：float → 输出格式 ====================
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event_id);
            AscendC::Cast<ElementD, float, false>(ubD, ubCFp32, AscendC::RoundMode::CAST_RINT, -1, repeat, {1, 1, 4, 8});
            
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(event_id_2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id_2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id);

            // ==================== 10. 数据写入 windowOut ====================
            // 当前子块的 token 范围
            int32_t lenTile = subTileRows;
            int32_t stTile = globalStartRow;
            int32_t edTile = stTile + lenTile;
            int32_t preSumRankInExpert = 0;
            int32_t tileOffset = 0;

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id);

            // 遍历每个 rank，写入 windowOut
            for (int32_t dstEpIdx = 0; dstEpIdx < params.EP; dstEpIdx++) {
                int32_t lenRankInExpert = tokenPerExpert(tokenPerExpertLayout(dstEpIdx, params.rank, groupIdx));
                int32_t dstExpertOffset = preSumBeforeRank(dstEpIdx * params.expertPerRank + groupIdx);
                
                int32_t stRankInExpert = preSumRankInExpert;
                int32_t edRankInExpert = stRankInExpert + lenRankInExpert;
                preSumRankInExpert += lenRankInExpert;
                
                // 判断当前子块与当前 rank 的范围是否有交集
                if (stRankInExpert >= edTile) {
                    break;
                }
                else if (edRankInExpert <= stTile) {
                    continue;
                }
                
                // 计算交集范围
                int32_t stData = max(stRankInExpert, stTile);
                int32_t edData = min(edRankInExpert, edTile);
                uint32_t lenData = edData - stData;
                
                if (lenData <= 0){
                    continue;
                }
                
                // 计算目标偏移
                uint32_t dstOffsetInExpert = 0;
                if (stTile > stRankInExpert) {
                    dstOffsetInExpert = stTile - stRankInExpert;
                }

                // 写入gmSrcD
                // 计算目标写入位置
                // dstOffsetInExpert: 当前 rank 在 expert 内的偏移
                // dstExpertOffset: 当前 rank 在所有 expert 中的全局偏移
                MatrixCoord dstOffset{dstOffsetInExpert + dstExpertOffset, blockCoord.n()};
                int64_t gmDstOffset = params.layoutC.GetOffset(dstOffset);
                auto gmTileD = gmSrcD[gmDstOffset];
                
                // 设置内存布局
                LayoutC layoutGM2{lenData, actualBlockShape.n(), params.n2};
                LayoutC layoutUB2{lenData, actualBlockShape.n(), n0};
                
                // 从 UB 拷贝数据到目标 rank 的共享内存
                copyUbToGmD(gmTileD, ubD[tileOffset * n0], layoutGM2, layoutUB2);
                
                tileOffset += lenData;
            }
            
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event_id);
        }
    }

private:
    
    Params params;
    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];
    AscendC::LocalTensor<float> ubFp32List[UB_STAGES];
    AscendC::LocalTensor<float> scaleUbList[UB_STAGES];
    int32_t source_scale_offset[UB_STAGES];

    int32_t max_len = 8 * 32 / 4 * 128;
    int32_t n0;
    bool is_ping = false;

    
    int32_t repeat = 128;

    CopyGmToUbC copyGmToUbC;
    CopyUbToGmD copyUbToGmD;
    
    CopyScaleGmToUb copyScaleGmToUb;
    AscendC::GlobalTensor<int32_t> tokenPerExpert;
    Layout3D tokenPerExpertLayout;
};
}
#endif