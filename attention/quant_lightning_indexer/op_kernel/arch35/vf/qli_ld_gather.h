#ifndef QLI_ARCH35_VF_QLI_LD_GATHER_H
#define QLI_ARCH35_VF_QLI_LD_GATHER_H

// LD index pass-through gather (ported from quant_lightning_indexer_v2).
// QLI-local copy: keeps attention/lightning_indexer/ byte-identical to upstream.

namespace topkb16gather {

__simd_vf__ void FindLDRealIndexVFImpl(__ubuf__ uint32_t* outputIdxBuf, __ubuf__ uint16_t* tmpIdxBuf,
    __ubuf__ uint32_t* hisIdxBuf, uint16_t vfLoop)
{
    MicroAPI::MaskReg pregB32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<uint16_t> tmpIdx;
    MicroAPI::RegTensor<uint32_t> outputIdx;
    for (uint16_t i = 0; i < (uint16_t)(vfLoop); ++i) {
        MicroAPI::LoadAlign<uint16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(tmpIdx, tmpIdxBuf + i * 64);
        MicroAPI::Gather(outputIdx, hisIdxBuf, (MicroAPI::RegTensor<uint32_t>&)tmpIdx, pregB32);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_NORM>(outputIdxBuf + i * 64, outputIdx, pregB32);
    }
}

__aicore__ inline void LiTopKLDGatherVF(const LocalTensor<uint32_t>& outputIdxLocal,
                                        const LocalTensor<uint16_t>& tmpIdxLocal,
                                        const LocalTensor<uint32_t>& hisIdxLocal,
                                        uint32_t topK)
{
    __ubuf__ uint32_t* outputIdxBuf = (__ubuf__ uint32_t*)outputIdxLocal.GetPhyAddr();
    __ubuf__ uint16_t* tmpIdxBuf = (__ubuf__ uint16_t*)tmpIdxLocal.GetPhyAddr();
    __ubuf__ uint32_t* hisIdxBuf = (__ubuf__ uint32_t*)hisIdxLocal.GetPhyAddr();
    const uint16_t repeatSize32 = 64;
    uint16_t topkLoopNum32 = (topK + repeatSize32 - 1) / repeatSize32;
    FindLDRealIndexVFImpl(outputIdxBuf, tmpIdxBuf, hisIdxBuf, topkLoopNum32);
}

}  // namespace topkb16gather

#endif
