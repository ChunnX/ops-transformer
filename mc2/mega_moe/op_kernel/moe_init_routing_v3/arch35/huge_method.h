template <typename T, typename U>
__simd_vf__ inline void ComputeFP4FromFP32(RegTensor<float>& vdExpFP32,
                                            MaskReg& pregAll32)
{
    MaskReg zeroMask;
    MaskReg specialMask;
    MaskReg negInfMask;
    RegTensor<int32_t> negZero;
    RegTensor<int32_t> maxExpFP32;
    RegTensor<int32_t> exp0FP32;
    RegTensor<int32_t> exp1FP32;
    Duplicate(negZero, NEG_ZERO);
    Compare<int32_t, CMPMODE::EQ>(negInfMask, (RegTensor<int32_t>&)vdExpFP32, negZero, pregAll32);
    Duplicate(maxExpFP32, FP32_EMASK_AND_INF_VAL);
    And(exp0FP32, (RegTensor<int32_t>&)vdExpFP32, maxExpFP32, pregAll32);
    ShiftRights(exp0FP32, exp0FP32, FP32_EXP_SHR_BITS, pregAll32);
    Adds(exp0FP32, exp0FP32, FP32_BIAS_NEG, pregAll32);
    Maxs(exp0FP32, exp0FP32, 0, pregAll32);
    Adds(exp0FP32, exp0FP32, NEG_ONE, pregAll32);
    Muls(exp1FP32, exp0FP32, NEG_ONE, pregAll32);
    Adds(exp1FP32, exp1FP32, FP32_BIAS, pregAll32);
    ShiftLefts(exp1FP32, exp1FP32, FP32_EXP_SHR_BITS, pregAll32);
    Mul(vdExpFP32, vdExpFP32, (RegTensor<float>&)exp1FP32, pregAll32);
    Adds(exp0FP32, exp0FP32, FP32_BIAS, pregAll32);
    ShiftLefts(exp0FP32, exp0FP32, FP32_EXP_SHR_BITS, pregAll32);
    CompareScalar<float, CMPMODE::LT>(specialMask, vdExpFP32, 0, pregAll32);
    Truncate<float, RoundMode::CAST_RINT>(vdExpFP32, vdExpFP32, pregAll32);
    Mul(vdExpFP32, vdExpFP32, (RegTensor<float>&)exp0FP32, pregAll32);
    CompareScalar<float, CMPMODE::EQ>(zeroMask, vdExpFP32, 0, pregAll32);
    MaskAnd(zeroMask, specialMask, zeroMask, pregAll32);
    MaskOr(zeroMask, negInfMask, zeroMask, pregAll32);
    Select<int32_t>((RegTensor<int32_t>&)vdExpFP32, negZero,
                    (RegTensor<int32_t>&)vdExpFP32, zeroMask);
}
template <typename T, typename U>
__simd_vf__ inline void ProcessFP16Quantization(
    RegTensor<T>& x0, RegTensor<T>& x1,
    RegTensor<uint16_t>& invScale,
    RegTensor<U>& xQuant0, RegTensor<U>& xQuant1,
    MaskReg& maskAllB16, MaskReg& maskAllB32)
{
    using namespace AscendC::MicroAPI;
    
    static constexpr CastTrait castTraitF16toFp32Zero = {RegLayout::ZERO, SatMode::UNKNOWN, 
                                                         MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr CastTrait castTraitF16toFp32One = {RegLayout::ONE, SatMode::UNKNOWN, 
                                                        MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr CastTrait castTraitFp32toBF16 = {RegLayout::ZERO, SatMode::NO_SAT, 
                                                      MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait traitB16ToB4 = {RegLayout::ZERO, SatMode::UNKNOWN, 
                                               MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    RegTensor<float> invScaleFP32, halfScaleForMulFP32;
    RegTensor<float> vdExp0ZeroFP32, vdExp0OneFP32;
    RegTensor<float> vdExp1ZeroFP32, vdExp1OneFP32;
    RegTensor<bfloat16_t> vdExp0ZeroBF16, vdExp0OneBF16;
    RegTensor<bfloat16_t> vdExp1ZeroBF16, vdExp1OneBF16;
    Cast<float, bfloat16_t, castTraitF16toFp32Zero>(halfScaleForMulFP32,
                                                    (RegTensor<bfloat16_t>&)invScale, maskAllB16);
    Cast<float, T, castTraitF16toFp32Zero>(vdExp0ZeroFP32, x0, maskAllB16);
    Cast<float, T, castTraitF16toFp32One>(vdExp0OneFP32, x0, maskAllB16);
    Mul(vdExp0ZeroFP32, vdExp0ZeroFP32, halfScaleForMulFP32, maskAllB32);
    Mul(vdExp0OneFP32, vdExp0OneFP32, halfScaleForMulFP32, maskAllB32);
    MaskReg pregAll321 = CreateMask<uint32_t, MaskPattern::ALL>();
    ComputeFP4FromFP32<T, U>(vdExp0ZeroFP32, pregAll321);
    MaskReg pregAll322 = CreateMask<uint32_t, MaskPattern::ALL>();
    ComputeFP4FromFP32<T, U>(vdExp0OneFP32, pregAll322);
    Cast<bfloat16_t, float, castTraitFp32toBF16>(vdExp0ZeroBF16, vdExp0ZeroFP32, maskAllB32);
    Cast<bfloat16_t, float, castTraitFp32toBF16>(vdExp0OneBF16, vdExp0OneFP32, maskAllB32);
    Pack<uint16_t, uint32_t, HighLowPart::LOWEST>((RegTensor<uint16_t>&)vdExp0ZeroBF16,
                                                   (RegTensor<uint32_t>&)vdExp0ZeroBF16);
    Pack<uint16_t, uint32_t, HighLowPart::LOWEST>((RegTensor<uint16_t>&)vdExp0OneBF16,
                                                   (RegTensor<uint32_t>&)vdExp0OneBF16);
    Interleave(vdExp0ZeroBF16, vdExp0OneBF16, vdExp0ZeroBF16, vdExp0OneBF16);
    Cast<float, T, castTraitF16toFp32Zero>(vdExp1ZeroFP32, x1, maskAllB16);
    Cast<float, T, castTraitF16toFp32One>(vdExp1OneFP32, x1, maskAllB16);
    Mul(vdExp1ZeroFP32, vdExp1ZeroFP32, halfScaleForMulFP32, maskAllB32);
    Mul(vdExp1OneFP32, vdExp1OneFP32, halfScaleForMulFP32, maskAllB32);
    MaskReg pregAll323 = CreateMask<uint32_t, MaskPattern::ALL>();
    ComputeFP4FromFP32<T, U>(vdExp1ZeroFP32, pregAll323);
    MaskReg pregAll324 = CreateMask<uint32_t, MaskPattern::ALL>();
    ComputeFP4FromFP32<T, U>(vdExp1OneFP32, pregAll324);
    Cast<bfloat16_t, float, castTraitFp32toBF16>(vdExp1ZeroBF16, vdExp1ZeroFP32, maskAllB32);
    Cast<bfloat16_t, float, castTraitFp32toBF16>(vdExp1OneBF16, vdExp1OneFP32, maskAllB32);
    Pack<uint16_t, uint32_t, HighLowPart::LOWEST>((RegTensor<uint16_t>&)vdExp1ZeroBF16,
                                                   (RegTensor<uint32_t>&)vdExp1ZeroBF16);
    Pack<uint16_t, uint32_t, HighLowPart::LOWEST>((RegTensor<uint16_t>&)vdExp1OneBF16,
                                                   (RegTensor<uint32_t>&)vdExp1OneBF16);
    Interleave(vdExp1ZeroBF16, vdExp1OneBF16, vdExp1ZeroBF16, vdExp1OneBF16);
    Interleave(vdExp0ZeroBF16, vdExp1ZeroBF16, vdExp0ZeroBF16, vdExp1ZeroBF16);
    Cast<U, bfloat16_t, traitB16ToB4>(xQuant0, vdExp0ZeroBF16, maskAllB16);
    Cast<U, bfloat16_t, traitB16ToB4>(xQuant1, vdExp1ZeroBF16, maskAllB16);
}
template <typename T, typename U>
__simd_vf__ inline void ProcessBF16Quantization(
    RegTensor<T>& x0, RegTensor<T>& x1,
    RegTensor<uint16_t>& invScale,
    RegTensor<U>& xQuant0, RegTensor<U>& xQuant1,
    MaskReg& maskAllB16)
{
    using namespace AscendC::MicroAPI;
    static constexpr CastTrait traitB16ToB4 = {RegLayout::ZERO, SatMode::UNKNOWN, 
                                               MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    Mul(x0, x0, (RegTensor<T>&)invScale, maskAllB16);
    Mul(x1, x1, (RegTensor<T>&)invScale, maskAllB16);
    Interleave(x0, x1, x0, x1);
    Cast<U, bfloat16_t, traitB16ToB4>(xQuant0, x0, maskAllB16);
    Cast<U, bfloat16_t, traitB16ToB4>(xQuant1, x1, maskAllB16);
}
template <typename T, typename U>
__simd_vf__ inline void vfMxfp4ComputeData(__ubuf__ T *xAddr, __ubuf__ uint16_t *invScaleInAddr,
                                           __ubuf__ int8_t *xQuantOutAddr, uint32_t xElemNum, 
                                           uint16_t vfLoopNum, uint32_t vlForT, uint32_t numVRegBlocks)
{
    using namespace AscendC::MicroAPI;
    RegTensor<uint16_t> invScale;
    RegTensor<T> x0, x1;
    RegTensor<U> xQuant0, xQuant1;
    MaskReg maskXQuant;
    MaskReg maskAllB16 = CreateMask<half>();
    MaskReg maskAllB32 = CreateMask<float>();
    for (uint16_t i = 0; i < vfLoopNum; i++) {
        maskXQuant = UpdateMask<float>(xElemNum);
        DataCopy<T, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(x0, x1, xAddr, vlForT * 2);
        DataCopy<uint16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B16>(
            invScale, invScaleInAddr, numVRegBlocks);
        if constexpr (IsSameType<T, half>::value) {
            ProcessFP16Quantization<T, U>(x0, x1, invScale, xQuant0, xQuant1, 
                                          maskAllB16, maskAllB32);
        } else {
            ProcessBF16Quantization<T, U>(x0, x1, invScale, xQuant0, xQuant1, maskAllB16);
        }
        DataCopy<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
            xQuantOutAddr, (RegTensor<int8_t>&)xQuant0, OUT_ELE_NUM_ONE_BLK, maskXQuant);
        DataCopy<int8_t, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
            xQuantOutAddr, (RegTensor<int8_t>&)xQuant1, OUT_ELE_NUM_ONE_BLK, maskXQuant);
    }
}