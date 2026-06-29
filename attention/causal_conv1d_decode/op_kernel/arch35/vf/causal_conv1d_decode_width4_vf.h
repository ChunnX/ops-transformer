#ifndef CAUSAL_CONV1D_DECODE_WIDTH4_VF_H
#define CAUSAL_CONV1D_DECODE_WIDTH4_VF_H

#include "kernel_operator.h"

using namespace AscendC;

constexpr MicroAPI::CastTrait CCDC_CAST_B162B32 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

constexpr MicroAPI::CastTrait CCDC_CAST_B322B16 = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

constexpr uint32_t CCDC_REGSIZE = 256;
constexpr uint32_t CCDC_B32_REP_SIZE = CCDC_REGSIZE / sizeof(float);

#define CCDC_B16_TO_F32(dst, src, mask) MicroAPI::Cast<float, bfloat16_t, CCDC_CAST_B162B32>(dst, src, mask)
#define CCDC_F32_TO_B16(dst, src, mask) MicroAPI::Cast<bfloat16_t, float, CCDC_CAST_B322B16>(dst, src, mask)

__simd_vf__ void Conv1dDecodeW4VF(__ubuf__ bfloat16_t *src0, __ubuf__ bfloat16_t *src1,
                                  __ubuf__ bfloat16_t *src2, __ubuf__ bfloat16_t *src3,
                                  __ubuf__ bfloat16_t *weight, __ubuf__ bfloat16_t *bias, __ubuf__ bfloat16_t *y,
                                  uint32_t dimLen, int32_t hasBias)
{
    MicroAPI::RegTensor<bfloat16_t> x0B16, x1B16, x2B16, x3B16, w0B16, w1B16, w2B16, w3B16, biasB16, yB16;
    MicroAPI::RegTensor<float> x0B32, x1B32, x2B32, x3B32, w0B32, w1B32, w2B32, w3B32, tmpB32, biasB32, yB32;
    uint32_t dimLoopNum = dimLen / CCDC_B32_REP_SIZE;
    uint32_t dimRem = dimLen - dimLoopNum * CCDC_B32_REP_SIZE;
    uint32_t offset = 0;

#define CCDC_W4_STEP(mask)                                                                                         \
    do {                                                                                                           \
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(x0B16, src0 + offset);                \
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(x1B16, src1 + offset);                \
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(x2B16, src2 + offset);                \
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(x3B16, src3 + offset);                \
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(w0B16, weight + offset);              \
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(w1B16, weight + dimLen + offset);     \
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(w2B16, weight + 2 * dimLen + offset); \
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(w3B16, weight + 3 * dimLen + offset); \
        CCDC_B16_TO_F32(x0B32, x0B16, mask);                                                                       \
        CCDC_B16_TO_F32(x1B32, x1B16, mask);                                                                       \
        CCDC_B16_TO_F32(x2B32, x2B16, mask);                                                                       \
        CCDC_B16_TO_F32(x3B32, x3B16, mask);                                                                       \
        CCDC_B16_TO_F32(w0B32, w0B16, mask);                                                                       \
        CCDC_B16_TO_F32(w1B32, w1B16, mask);                                                                       \
        CCDC_B16_TO_F32(w2B32, w2B16, mask);                                                                       \
        CCDC_B16_TO_F32(w3B32, w3B16, mask);                                                                       \
        MicroAPI::Mul(yB32, x0B32, w0B32, mask);                                                                   \
        MicroAPI::Mul(tmpB32, x1B32, w1B32, mask);                                                                 \
        MicroAPI::Add(yB32, yB32, tmpB32, mask);                                                                   \
        MicroAPI::Mul(tmpB32, x2B32, w2B32, mask);                                                                 \
        MicroAPI::Add(yB32, yB32, tmpB32, mask);                                                                   \
        MicroAPI::Mul(tmpB32, x3B32, w3B32, mask);                                                                 \
        MicroAPI::Add(yB32, yB32, tmpB32, mask);                                                                   \
        if (hasBias != 0) {                                                                                        \
            MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(biasB16, bias + offset);          \
            CCDC_B16_TO_F32(biasB32, biasB16, mask);                                                              \
            MicroAPI::Add(yB32, yB32, biasB32, mask);                                                             \
        }                                                                                                          \
        CCDC_F32_TO_B16(yB16, yB32, mask);                                                                        \
        MicroAPI::StoreAlign<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(y + offset, yB16, mask);              \
    } while (0)

    for (uint32_t dimLoop = 0; dimLoop < dimLoopNum; ++dimLoop) {
        MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        CCDC_W4_STEP(maskB32);
        offset += CCDC_B32_REP_SIZE;
    }
    if (dimRem != 0) {
        MicroAPI::MaskReg maskRem = MicroAPI::UpdateMask<float>(dimRem);
        CCDC_W4_STEP(maskRem);
    }
#undef CCDC_W4_STEP
}

__aicore__ inline void Conv1dDecodeW4(LocalTensor<bfloat16_t> &src0, LocalTensor<bfloat16_t> &src1,
                                      LocalTensor<bfloat16_t> &src2, LocalTensor<bfloat16_t> &src3,
                                      LocalTensor<bfloat16_t> &weight, LocalTensor<bfloat16_t> &bias,
                                      LocalTensor<bfloat16_t> &y, uint32_t dimLen, int32_t hasBias)
{
    Conv1dDecodeW4VF((__ubuf__ bfloat16_t *)src0.GetPhyAddr(), (__ubuf__ bfloat16_t *)src1.GetPhyAddr(),
                     (__ubuf__ bfloat16_t *)src2.GetPhyAddr(), (__ubuf__ bfloat16_t *)src3.GetPhyAddr(),
                     (__ubuf__ bfloat16_t *)weight.GetPhyAddr(), (__ubuf__ bfloat16_t *)bias.GetPhyAddr(),
                     (__ubuf__ bfloat16_t *)y.GetPhyAddr(), dimLen, hasBias);
}

__simd_vf__ void Conv1dDecodeW4MultiSVF(__ubuf__ bfloat16_t *t0, __ubuf__ bfloat16_t *t1,
                                        __ubuf__ bfloat16_t *t2, __ubuf__ bfloat16_t *t3,
                                        __ubuf__ bfloat16_t *t4, __ubuf__ bfloat16_t *t5,
                                        __ubuf__ bfloat16_t *t6, __ubuf__ bfloat16_t *weight,
                                        __ubuf__ bfloat16_t *bias, __ubuf__ bfloat16_t *y0,
                                        __ubuf__ bfloat16_t *y1, __ubuf__ bfloat16_t *y2,
                                        __ubuf__ bfloat16_t *y3, uint32_t dimLen, int32_t hasBias, int32_t tokenCount)
{
    MicroAPI::RegTensor<bfloat16_t> t0B16, t1B16, t2B16, t3B16, t4B16, t5B16, t6B16;
    MicroAPI::RegTensor<bfloat16_t> w0B16, w1B16, w2B16, w3B16, biasB16, yB16;
    MicroAPI::RegTensor<float> t0B32, t1B32, t2B32, t3B32, t4B32, t5B32, t6B32;
    MicroAPI::RegTensor<float> w0B32, w1B32, w2B32, w3B32, biasB32;
    MicroAPI::RegTensor<float> acc0B32, acc1B32, acc2B32, acc3B32, tmpB32;
    uint32_t dimLoopNum = dimLen / CCDC_B32_REP_SIZE;
    uint32_t dimRem = dimLen - dimLoopNum * CCDC_B32_REP_SIZE;
    uint32_t offset = 0;

#define CCDC_LOAD_MULTI(mask)                                                                                       \
    do {                                                                                                            \
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(t0B16, t0 + offset);                   \
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(t1B16, t1 + offset);                   \
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(t2B16, t2 + offset);                   \
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(t3B16, t3 + offset);                   \
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(t4B16, t4 + offset);                   \
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(t5B16, t5 + offset);                   \
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(t6B16, t6 + offset);                   \
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(w0B16, weight + offset);               \
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(w1B16, weight + dimLen + offset);      \
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(w2B16, weight + 2 * dimLen + offset);  \
        MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(w3B16, weight + 3 * dimLen + offset);  \
        CCDC_B16_TO_F32(t0B32, t0B16, mask);                                                                       \
        CCDC_B16_TO_F32(t1B32, t1B16, mask);                                                                       \
        CCDC_B16_TO_F32(t2B32, t2B16, mask);                                                                       \
        CCDC_B16_TO_F32(t3B32, t3B16, mask);                                                                       \
        CCDC_B16_TO_F32(t4B32, t4B16, mask);                                                                       \
        CCDC_B16_TO_F32(t5B32, t5B16, mask);                                                                       \
        CCDC_B16_TO_F32(t6B32, t6B16, mask);                                                                       \
        CCDC_B16_TO_F32(w0B32, w0B16, mask);                                                                       \
        CCDC_B16_TO_F32(w1B32, w1B16, mask);                                                                       \
        CCDC_B16_TO_F32(w2B32, w2B16, mask);                                                                       \
        CCDC_B16_TO_F32(w3B32, w3B16, mask);                                                                       \
        if (hasBias != 0) {                                                                                        \
            MicroAPI::LoadAlign<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(biasB16, bias + offset);          \
            CCDC_B16_TO_F32(biasB32, biasB16, mask);                                                              \
        }                                                                                                          \
    } while (0)

#define CCDC_COMPUTE_ONE(acc, a, b, c, d, mask)                                                                     \
    do {                                                                                                            \
        MicroAPI::Mul(acc, a, w0B32, mask);                                                                         \
        MicroAPI::Mul(tmpB32, b, w1B32, mask);                                                                      \
        MicroAPI::Add(acc, acc, tmpB32, mask);                                                                      \
        MicroAPI::Mul(tmpB32, c, w2B32, mask);                                                                      \
        MicroAPI::Add(acc, acc, tmpB32, mask);                                                                      \
        MicroAPI::Mul(tmpB32, d, w3B32, mask);                                                                      \
        MicroAPI::Add(acc, acc, tmpB32, mask);                                                                      \
        if (hasBias != 0) {                                                                                         \
            MicroAPI::Add(acc, acc, biasB32, mask);                                                                 \
        }                                                                                                           \
    } while (0)

#define CCDC_STORE_ONE(acc, out, mask)                                                                              \
    do {                                                                                                            \
        CCDC_F32_TO_B16(yB16, acc, mask);                                                                           \
        MicroAPI::StoreAlign<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(out + offset, yB16, mask);             \
    } while (0)

    for (uint32_t dimLoop = 0; dimLoop < dimLoopNum; ++dimLoop) {
        MicroAPI::MaskReg maskB32 = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        CCDC_LOAD_MULTI(maskB32);
        CCDC_COMPUTE_ONE(acc0B32, t0B32, t1B32, t2B32, t3B32, maskB32);
        CCDC_STORE_ONE(acc0B32, y0, maskB32);
        if (tokenCount >= 2) {
            CCDC_COMPUTE_ONE(acc1B32, t1B32, t2B32, t3B32, t4B32, maskB32);
            CCDC_STORE_ONE(acc1B32, y1, maskB32);
        }
        if (tokenCount >= 3) {
            CCDC_COMPUTE_ONE(acc2B32, t2B32, t3B32, t4B32, t5B32, maskB32);
            CCDC_STORE_ONE(acc2B32, y2, maskB32);
        }
        if (tokenCount >= 4) {
            CCDC_COMPUTE_ONE(acc3B32, t3B32, t4B32, t5B32, t6B32, maskB32);
            CCDC_STORE_ONE(acc3B32, y3, maskB32);
        }
        offset += CCDC_B32_REP_SIZE;
    }

    if (dimRem != 0) {
        MicroAPI::MaskReg maskRem = MicroAPI::UpdateMask<float>(dimRem);
        CCDC_LOAD_MULTI(maskRem);
        CCDC_COMPUTE_ONE(acc0B32, t0B32, t1B32, t2B32, t3B32, maskRem);
        CCDC_STORE_ONE(acc0B32, y0, maskRem);
        if (tokenCount >= 2) {
            CCDC_COMPUTE_ONE(acc1B32, t1B32, t2B32, t3B32, t4B32, maskRem);
            CCDC_STORE_ONE(acc1B32, y1, maskRem);
        }
        if (tokenCount >= 3) {
            CCDC_COMPUTE_ONE(acc2B32, t2B32, t3B32, t4B32, t5B32, maskRem);
            CCDC_STORE_ONE(acc2B32, y2, maskRem);
        }
        if (tokenCount >= 4) {
            CCDC_COMPUTE_ONE(acc3B32, t3B32, t4B32, t5B32, t6B32, maskRem);
            CCDC_STORE_ONE(acc3B32, y3, maskRem);
        }
    }

#undef CCDC_STORE_ONE
#undef CCDC_COMPUTE_ONE
#undef CCDC_LOAD_MULTI
}

__aicore__ inline void Conv1dDecodeMultiS(LocalTensor<bfloat16_t> &t0, LocalTensor<bfloat16_t> &t1,
                                          LocalTensor<bfloat16_t> &t2, LocalTensor<bfloat16_t> &t3,
                                          LocalTensor<bfloat16_t> &t4, LocalTensor<bfloat16_t> &t5,
                                          LocalTensor<bfloat16_t> &t6, LocalTensor<bfloat16_t> &weight,
                                          LocalTensor<bfloat16_t> &bias, LocalTensor<bfloat16_t> &y0,
                                          LocalTensor<bfloat16_t> &y1, LocalTensor<bfloat16_t> &y2,
                                          LocalTensor<bfloat16_t> &y3, uint32_t dimLen, int32_t hasBias,
                                          int32_t tokenCount)
{
    auto t0Addr = (__ubuf__ bfloat16_t *)t0.GetPhyAddr();
    auto t1Addr = (__ubuf__ bfloat16_t *)t1.GetPhyAddr();
    auto t2Addr = (__ubuf__ bfloat16_t *)t2.GetPhyAddr();
    auto t3Addr = (__ubuf__ bfloat16_t *)t3.GetPhyAddr();
    auto t4Addr = (__ubuf__ bfloat16_t *)t4.GetPhyAddr();
    auto t5Addr = (__ubuf__ bfloat16_t *)t5.GetPhyAddr();
    auto t6Addr = (__ubuf__ bfloat16_t *)t6.GetPhyAddr();
    auto weightAddr = (__ubuf__ bfloat16_t *)weight.GetPhyAddr();
    auto biasAddr = (__ubuf__ bfloat16_t *)bias.GetPhyAddr();
    auto y0Addr = (__ubuf__ bfloat16_t *)y0.GetPhyAddr();
    auto y1Addr = (__ubuf__ bfloat16_t *)y1.GetPhyAddr();
    auto y2Addr = (__ubuf__ bfloat16_t *)y2.GetPhyAddr();
    auto y3Addr = (__ubuf__ bfloat16_t *)y3.GetPhyAddr();
    Conv1dDecodeW4MultiSVF(t0Addr, t1Addr, t2Addr, t3Addr, t4Addr, t5Addr, t6Addr, weightAddr, biasAddr, y0Addr,
                           y1Addr, y2Addr, y3Addr, dimLen, hasBias, tokenCount);
}

#undef CCDC_STORE_MULTI_ONE
#undef CCDC_F32_TO_B16
#undef CCDC_B16_TO_F32

#endif
