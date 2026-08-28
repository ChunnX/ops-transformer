namespace Blaze::Epilogue {

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, uint16_t reduceSize, bool isUpdatePre, bool isMlaFullQuant>
__simd_vf__ inline void FlashUpdateBasicVF_MX(__ubuf__ float * dstUb, __ubuf__ float * curUb, __ubuf__ float * preUb,
    __ubuf__ float * expMaxUb, const uint16_t m, const uint16_t d,
    const float deScaleV, const float deScaleVPre)
{
    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t dLoops = srcD / floatRepSize;
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_row_max;
    RegTensor<float> vreg_input_pre;
    RegTensor<float> vreg_input_cur;
    RegTensor<float> vreg_mul;
    RegTensor<float> vreg_add;

    MaskReg preg_all = pset_b32(PAT_ALL);

    constexpr auto dist_brc_b32 = std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_BRC_B32)>();
    constexpr auto dist_norm = std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_NORM)>();
    constexpr auto dist_norm_b32 = std::integral_constant<::DistVST, static_cast<::DistVST>(StoreDist::DIST_NORM_B32)>();

    // dstTensor = preTensor * expMaxTensor + curTensor
    for (uint16_t i = 0; i < m; ++i) {
        vlds(vreg_exp_max, (__ubuf__ float*)expMaxUb + i * reduceSize, 0, dist_brc_b32);
        // if constexpr (isMlaFullQuant) {
        //     LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_row_max, rowMaxUb + i * reduceSize);
        // }


        for (uint16_t j = 0; j < dLoops; ++j) {
            vlds(vreg_input_pre, (__ubuf__ float*)preUb + i * d + j * floatRepSize, 0, dist_norm);
            vlds(vreg_input_cur, (__ubuf__ float*)curUb + i * d + j * floatRepSize, 0, dist_norm);//不对
            // if constexpr (isMlaFullQuant) {
            //     Mul(vreg_input_cur, vreg_input_cur, vreg_row_max, preg_all);
            // }
            vmul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_all, MODE_ZEROING);
            if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value ||
                            IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                            IsSameType<INPUT_T, hifloat8_t>::value ||
                            IsSameType<INPUT_T, int8_t>::value) {
                vmuls(vreg_input_cur, vreg_input_cur, deScaleV, preg_all, MODE_ZEROING);
                if constexpr (isUpdatePre) {
                    vmuls(vreg_mul, vreg_mul, deScaleVPre, preg_all, MODE_ZEROING);
                }
            }
            vadd(vreg_add, vreg_mul, vreg_input_cur, preg_all, MODE_ZEROING);
            vsts(vreg_add, (__ubuf__ float *&)dstUb + i * d + j * floatRepSize, 0, dist_norm_b32, preg_all);
        }
    }
}

template <typename T, typename INPUT_T, typename OUTPUT_T, uint16_t srcD, uint16_t reduceSize, bool isUpdatePre, bool isMlaFullQuant>
__simd_vf__ inline void FlashUpdateLastBasicVF_MX(__ubuf__ float * dstUb, __ubuf__ float * curUb, __ubuf__ float * preUb,
    __ubuf__ float * expMaxUb, __ubuf__ float * expSumUb, const uint16_t m, const uint16_t d,
    const float deScaleV, const float deScaleVPre)
{
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_row_max;
    RegTensor<float> vreg_input_pre;
    RegTensor<float> vreg_input_cur;
    RegTensor<float> vreg_mul;
    RegTensor<float> vreg_add;
    RegTensor<float> vreg_div;
    RegTensor<half> vreg_cast;
    RegTensor<float> vreg_exp_sum;

    MaskReg preg_all = pset_b32(PAT_ALL);
    constexpr uint16_t floatRepSize = 64;
    constexpr uint16_t dLoops = srcD / floatRepSize;
    constexpr float fp8e4m3MaxValueRec = 1 / 448.0f;
    constexpr float int8MaxValueRec = 1 / 127.0f;
    constexpr float hifp8MaxValueRec = 1 / 32768.0f;

    constexpr auto dist_brc_b32 = std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_BRC_B32)>();
    constexpr auto dist_norm = std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_NORM)>();
    constexpr auto dist_norm_b32 = std::integral_constant<::DistVST, static_cast<::DistVST>(StoreDist::DIST_NORM_B32)>();

    for (uint16_t i = 0; i < m; ++i) {
        vlds(vreg_exp_max, (__ubuf__ float*)expMaxUb + i * reduceSize, 0, dist_brc_b32);//有0
        vlds(vreg_exp_sum, (__ubuf__ float*)expSumUb + i * reduceSize, 0, dist_brc_b32);
        for (uint16_t j = 0; j < dLoops; ++j) {
            vlds(vreg_input_pre, (__ubuf__ float*)preUb + i * d + j * floatRepSize, 0, dist_norm);
            vlds(vreg_input_cur, (__ubuf__ float*)curUb + i * d + j * floatRepSize, 0, dist_norm);//不是0
            vmul(vreg_mul, vreg_exp_max, vreg_input_pre, preg_all, MODE_ZEROING);//有0

            vmuls(vreg_input_cur, vreg_input_cur, deScaleV, preg_all, MODE_ZEROING);
            if constexpr (isUpdatePre) {
                vmuls(vreg_mul, vreg_mul, deScaleVPre, preg_all, MODE_ZEROING);
            }
            
            vadd(vreg_add, vreg_mul, vreg_input_cur, preg_all, MODE_ZEROING);
            vdiv(vreg_div, vreg_add, vreg_exp_sum, preg_all, MODE_ZEROING);
            vsts(vreg_div, (__ubuf__ float *&)dstUb + i * d + j * floatRepSize, 0, dist_norm_b32, preg_all);
        }
    }
}

    template <typename T, typename T2, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 256, bool hasAtten = 0>
    __simd_vf__ void process_vec1_noupdate_subloop0(__ubuf__ fp8_e4m3fn_t *expUb1, __ubuf__ fp8_e4m3fn_t *expUb2, __ubuf__ float *maxUb, __ubuf__ float *maxUbStart,
        __ubuf__ float *srcUb, __ubuf__ float *expSumUb, __ubuf__ uint8_t *indexesUb, __ubuf__ uint32_t *maskUb1, __ubuf__ uint32_t *maskUb2, __ubuf__ uint32_t *maskUb3,
        __ubuf__ uint32_t *maskUb4, __ubuf__ fp8_e8m0_t *pScaleSubLoop0, const uint32_t nPadding, const uint32_t blockStride, const uint32_t repeatStride, uint32_t pltOriTailN1, uint32_t pltOriTailN2, uint32_t pltTailN1, uint32_t pltTailN2,
        const uint16_t m, const float scale, const float minValue, const float pScale = 1.0f)//m=16
    {
        RegTensor<float> vreg_min;
        RegTensor<float> vreg_sel1;
        RegTensor<float> vreg_sel2;
        RegTensor<float> vreg_sel3;
        RegTensor<float> vreg_sel4;
        RegTensor<float> vreg_sel3_new;
        RegTensor<float> vreg_sel4_new;
        RegTensor<float> vreg_input_x1;
        RegTensor<float> vreg_input_x2;
        RegTensor<float> vreg_input_x3;
        RegTensor<float> vreg_input_x4;
        RegTensor<float> vreg_input_x3_new;
        RegTensor<float> vreg_input_x4_new;
        RegTensor<float> vreg_max_tmp1;
        RegTensor<float> vreg_max_tmp2;
        RegTensor<float> vreg_max_tmp3;
        RegTensor<float> vreg_input_max;
        RegTensor<float> vreg_exp_max;
        RegTensor<float> vreg_max_brc;
        RegTensor<float> vreg_zero;
        RegTensor<float> vreg_exp_sum1;
        RegTensor<float> vreg_exp_sum2;
        RegTensor<float> vreg_exp_sum3;
        RegTensor<float> vreg_exp_sum_brc;
        RegTensor<float> vreg_exp_even1;
        RegTensor<float> vreg_exp_odd1;
        RegTensor<float> vreg_exp_even2;
        RegTensor<float> vreg_exp_odd2;
        RegTensor<float> vreg_pse1;
        RegTensor<float> vreg_pse2;
        RegTensor<float> vreg_pse3;
        RegTensor<float> vreg_pse4;
        RegTensor<float> vreg_alibi1;
        RegTensor<float> vreg_alibi2;
        RegTensor<float> vreg_alibi3;
        RegTensor<float> vreg_alibi4;
        RegTensor<float> vreg_sel_drop;
        RegTensor<float> vreg_sel_drop2;
        RegTensor<float> vreg_sink_input;
        // bfloat16_t
        RegTensor<bfloat16_t> vreg_exp_even1_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd1_bf16;
        RegTensor<bfloat16_t> vreg_exp_even2_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd2_bf16;
        RegTensor<bfloat16_t> vreg_exp1_bf16;
        RegTensor<bfloat16_t> vreg_exp2_bf16;
        RegTensor<bfloat16_t> vreg_pse_bf16_src1;
        RegTensor<bfloat16_t> vreg_pse_bf16_src2;
        RegTensor<bfloat16_t> vreg_pse1_bf16;
        RegTensor<bfloat16_t> vreg_pse2_bf16;
        RegTensor<bfloat16_t> vreg_pse3_bf16;
        RegTensor<bfloat16_t> vreg_pse4_bf16;
        // half
        RegTensor<half> vreg_exp_even1_f16;
        RegTensor<half> vreg_exp_odd1_f16;
        RegTensor<half> vreg_exp_even2_f16;
        RegTensor<half> vreg_exp_odd2_f16;
        RegTensor<half> vreg_exp1_f16;
        RegTensor<half> vreg_exp2_f16;
        RegTensor<half> vreg_pse_f16_src1;
        RegTensor<half> vreg_pse_f16_src2;
        RegTensor<half> vreg_pse1_f16;
        RegTensor<half> vreg_pse2_f16;
        RegTensor<half> vreg_pse3_f16;
        RegTensor<half> vreg_pse4_f16;
        UnalignRegForStore ureg_max;
        UnalignRegForStore ureg_exp_sum;

        // fp8_e8m0_t
 	    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0;

        MaskReg preg_all = pset_b32(PAT_ALL);
        MaskReg preg_all_b16 = pset_b16(PAT_ALL);
        MaskReg preg_tail_n1 = plt_b32(pltTailN1, POST_UPDATE);
        MaskReg preg_ori_tail_n1 = plt_b32(pltOriTailN1, POST_UPDATE);
        MaskReg preg_tail_n2 = plt_b32(pltTailN2, POST_UPDATE);
        MaskReg preg_ori_tail_n2 = plt_b32(pltOriTailN2, POST_UPDATE);
        MaskReg preg_reduce_n = pset_b32(PAT_VL8);
        MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();

        MaskReg preg_compare1;
        MaskReg preg_compare2;
        MaskReg preg_compare3;
        MaskReg preg_compare4;

        MaskReg preg1;
        MaskReg preg2 = pset_b8(PAT_ALLF);

        MaskReg preg3;
        MaskReg preg4;
        MaskReg preg5;
        MaskReg preg6;
        // pScale
        RegTensor<float> vreg_p_scale;
        RegTensor<float> vreg_ln_p_scale;
        vbr(vreg_p_scale, static_cast<float>(pScale));
        vln(vreg_ln_p_scale, vreg_p_scale, preg_all, MODE_ZEROING);

        vbr(vreg_min, minValue);

        constexpr auto dist_norm = std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_NORM)>();
        constexpr auto dist_brc_b32 = std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_BRC_B32)>();
        constexpr auto dist_dintlv_b32 = std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_DINTLV_B32)>();
        constexpr auto dist_norm_b8 = std::integral_constant<::DistVST, static_cast<::DistVST>(StoreDist::DIST_NORM_B8)>();
        constexpr auto dist_norm_b32 = std::integral_constant<::DistVST, static_cast<::DistVST>(StoreDist::DIST_NORM_B32)>();
        constexpr auto dist_ds = std::integral_constant<::Dist, static_cast<::Dist>(MaskDist::DIST_DS)>();
        // if constexpr (hasSink) {
        //     Duplicate(vreg_sink_input, sinkValue);
        // }
        // if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
        //               pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        //     Arange(vreg_alibi1, posShift);
        //     Arange(vreg_alibi2, posShift + floatRepSize);
        //     Arange(vreg_alibi3, posShift + floatRepSize * 2);
        //     Arange(vreg_alibi4, posShift + floatRepSize * 3);
        // }
        for (uint16_t i = 0; i < m; ++i) {
            vlds(vreg_input_x1, (__ubuf__ float*)srcUb + i * 256, 0, dist_norm);
            vlds(vreg_input_x2, (__ubuf__ float*)srcUb + floatRepSize + i * 256, 0, dist_norm);
            vlds(vreg_input_x3, (__ubuf__ float*)srcUb + floatRepSize * 2 + i * 256, 0, dist_norm);
            vlds(vreg_input_x4, (__ubuf__ float*)srcUb + floatRepSize * 3 + i * 256, 0, dist_norm);
            // if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                vmuls(vreg_input_x1, vreg_input_x1, scale, preg_all, MODE_ZEROING); // Muls(scale)
                vmuls(vreg_input_x2, vreg_input_x2, scale, preg_all, MODE_ZEROING);
                vmuls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1, MODE_ZEROING);
                vmuls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2, MODE_ZEROING);

            // if constexpr (hasAtten == 1) {
                plds(preg_compare1, (__ubuf__ uint32_t *&)maskUb1, nPadding, dist_ds, POST_UPDATE);
                plds(preg_compare2, (__ubuf__ uint32_t *&)maskUb2, nPadding, dist_ds, POST_UPDATE);
                plds(preg_compare3, (__ubuf__ uint32_t *&)maskUb3, nPadding, dist_ds, POST_UPDATE);
                plds(preg_compare4, (__ubuf__ uint32_t *&)maskUb4, nPadding, dist_ds, POST_UPDATE);
                vsel(vreg_sel1, vreg_min, vreg_input_x1, preg_compare1);
                vsel(vreg_sel2, vreg_min, vreg_input_x2, preg_compare2);
                vsel(vreg_sel3, vreg_min, vreg_input_x3, preg_compare3);
                vsel(vreg_sel4, vreg_min, vreg_input_x4, preg_compare4);
                // vsel(vreg_sel3_new, vreg_sel3, vreg_min, preg_ori_tail_n1);
                // vsel(vreg_sel4_new, vreg_sel4, vreg_min, preg_ori_tail_n2);
                vsts(vreg_sel1, (__ubuf__ float *&)srcUb + i * 256, 0, dist_norm_b32, preg_all);
                vsts(vreg_sel2, (__ubuf__ float *&)srcUb + floatRepSize + i * 256, 0, dist_norm_b32, preg_all);
                vsts(vreg_sel3, (__ubuf__ float *&)srcUb + floatRepSize * 2 + i * 256, 0, dist_norm_b32, preg_all);
                vsts(vreg_sel4, (__ubuf__ float *&)srcUb + floatRepSize * 3 + i * 256, 0, dist_norm_b32, preg_tail_n2);
                vmax(vreg_max_tmp1, vreg_sel1, vreg_sel2, preg_all, MODE_ZEROING);
                vmax(vreg_max_tmp2, vreg_sel3, vreg_sel4, preg_all, MODE_ZEROING);
                vmax(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all, MODE_ZEROING);
                vcmax(vreg_input_max, vreg_max_tmp3, preg_all, MODE_ZEROING);

            vmuls(vreg_input_max, vreg_input_max, INV_LN2, preg_all, MODE_ZEROING);
            vtrc(vreg_input_max, vreg_input_max, ROUND_C, preg_all, MODE_ZEROING);
            vmuls(vreg_input_max, vreg_input_max, LN2, preg_all, MODE_ZEROING);
            vstus(ureg_max, 1, vreg_input_max, (__ubuf__ float *&)maxUb, POST_UPDATE);
        }
        vstas(ureg_max, (__ubuf__ float *&)maxUb, 0, POST_UPDATE);
        // if constexpr (hasDrop == 1) {
        //     Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_zero, 0.0f, preg_all);
        // }
        mem_bar(VST_VLD);

        for (uint16_t i = 0; i < m; ++i) {
            vlds(vreg_max_brc, (__ubuf__ float*)maxUbStart + i, 0, dist_brc_b32);
            vsub(vreg_max_brc, vreg_max_brc, vreg_ln_p_scale, preg_all, MODE_ZEROING);
            vlds(vreg_input_x1, vreg_input_x2, (__ubuf__ float*)srcUb + i * 256, 0, dist_dintlv_b32);
            vlds(vreg_input_x3, vreg_input_x4, (__ubuf__ float*)srcUb + floatRepSize * 2 + i * 256, 0, dist_dintlv_b32);
            vexpdif(vreg_exp_even1, vreg_input_x1, vreg_max_brc, preg_all, PART_EVEN);
            vexpdif(vreg_exp_odd1, vreg_input_x2, vreg_max_brc, preg_all, PART_EVEN);
            vexpdif(vreg_exp_even2, vreg_input_x3, vreg_max_brc, preg_all, PART_EVEN);
            vexpdif(vreg_exp_odd2, vreg_input_x4, vreg_max_brc, preg_all, PART_EVEN);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            vadd(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all, MODE_ZEROING);
            vadd(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all, MODE_ZEROING);
            vadd(vreg_exp_sum3, vreg_exp_sum1, vreg_exp_sum2, preg_all, MODE_ZEROING);
            vcadd(vreg_exp_sum3, vreg_exp_sum3, preg_all, MODE_ZEROING);
            vstus(ureg_exp_sum, 1, vreg_exp_sum3, (__ubuf__ float *&)expSumUb, POST_UPDATE);

                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_2;
                RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
                // RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes_2;
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = plt_b8(maskLen, POST_UPDATE);
                vcvt(vreg_exp_even_f8e4m3_1, vreg_exp_even1, preg_all, ROUND_R, RS_DISABLE, PART_P0, MODE_ZEROING);
                vcvt(vreg_exp_even_f8e4m3_2, vreg_exp_even2, preg_all, ROUND_R, RS_DISABLE, PART_P0, MODE_ZEROING);
                vcvt(vreg_exp_odd_f8e4m3_1, vreg_exp_odd1, preg_all, ROUND_R, RS_DISABLE, PART_P2, MODE_ZEROING);
                vcvt(vreg_exp_odd_f8e4m3_2, vreg_exp_odd2, preg_all, ROUND_R, RS_DISABLE, PART_P2, MODE_ZEROING);
                vor((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_1, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3_1,
                   (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3_1, preg_all_b8, MODE_ZEROING);
                vor((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_2, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3_2,
                   (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3_2, preg_all_b8, MODE_ZEROING);
                vlds(vreg_exp_merge_f8e4m3_indexes, (__ubuf__ uint8_t*)indexesUb, 0, dist_norm);
                vselr((RegTensor<uint8_t> &)vreg_exp_merge_f8e4m3_1, (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_1, (RegTensor<uint8_t> &)vreg_exp_merge_f8e4m3_indexes);
                vselr((RegTensor<uint8_t> &)vreg_exp_merge_f8e4m3_2, (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_2, (RegTensor<uint8_t> &)vreg_exp_merge_f8e4m3_indexes);
                vsstb((RegTensor<uint8_t> &)vreg_exp_merge_f8e4m3_1, (__ubuf__ uint8_t *&)expUb1,
                      (blockStride << 16u) | (repeatStride & 0xFFFFU), preg_all_b8_128, POST_UPDATE);
                vsstb((RegTensor<uint8_t> &)vreg_exp_merge_f8e4m3_2, (__ubuf__ uint8_t *&)expUb2,
                      (blockStride << 16u) | (repeatStride & 0xFFFFU), preg_all_b8_128, POST_UPDATE);
            // }
        }
        vstas(ureg_exp_sum, (__ubuf__ float *&)expSumUb, 0, POST_UPDATE);
        // pScale update
        const int8_t val_value = 0x7f;
        vdup((RegTensor<int8_t>&)vreg_p_scale_f8e8m0, (int8_t&)val_value, preg_all_b8, MODE_ZEROING);
 	    // StoreAlign<fp8_e8m0_t, MicroAPI::StoreDist::DIST_NORM_B8>(((__ubuf__ fp8_e8m0_t *&)pScaleSubLoop0),
 	    //                                                    vreg_p_scale_f8e8m0, preg_all_b8);
        vsts((RegTensor<uint8_t>&)vreg_p_scale_f8e8m0, (__ubuf__ uint8_t*)pScaleSubLoop0, 0, dist_norm_b8, preg_all_b8);
    }

    template <typename T, typename T2, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 256, bool hasAtten = 0>
    __simd_vf__ void process_vec1_noupdate_subloop1(__ubuf__ fp8_e4m3fn_t *expUb1, __ubuf__ fp8_e4m3fn_t *expUb2, __ubuf__ float *maxUb, __ubuf__ float *maxUbStart,
        __ubuf__ float *srcUb, __ubuf__ float * expMaxUb, __ubuf__ float * inMaxUb, __ubuf__ float * expSumUb, __ubuf__ float * inExpSumUb, __ubuf__ float * tmpExpSumUb,
        __ubuf__ float * tmpExpSumUb2, __ubuf__ float * tmpMaxUb, __ubuf__ float * tmpMaxUb2,
        __ubuf__ uint8_t *indexesUb, __ubuf__ uint32_t *maskUb1, __ubuf__ uint32_t *maskUb2, __ubuf__ uint32_t *maskUb3,
        __ubuf__ uint32_t *maskUb4, __ubuf__ fp8_e8m0_t *pScaleSubLoop0, const uint32_t nPadding, const uint32_t blockStride, const uint32_t repeatStride, uint32_t pltOriTailN1, uint32_t pltOriTailN2, uint32_t pltTailN1, uint32_t pltTailN2,
        const uint16_t m, const float scale, const float minValue, const float pScale = 1.0f)//m=16
    {
        RegTensor<float> vreg_min;
        RegTensor<float> vreg_sel1;
        RegTensor<float> vreg_sel2;
        RegTensor<float> vreg_sel3;
        RegTensor<float> vreg_sel4;
        RegTensor<float> vreg_sel3_new;
        RegTensor<float> vreg_sel4_new;
        RegTensor<float> vreg_input_x1;
        RegTensor<float> vreg_input_x2;
        RegTensor<float> vreg_input_x3;
        RegTensor<float> vreg_input_x4;
        RegTensor<float> vreg_input_x3_new;
        RegTensor<float> vreg_input_x4_new;
        RegTensor<float> vreg_max_tmp1;
        RegTensor<float> vreg_max_tmp2;
        RegTensor<float> vreg_max_tmp3;
        RegTensor<float> vreg_input_max;
        RegTensor<float> vreg_exp_max;
        RegTensor<float> vreg_max_brc;
        RegTensor<float> vreg_zero;
        RegTensor<float> vreg_exp_sum1;
        RegTensor<float> vreg_exp_sum2;
        RegTensor<float> vreg_exp_sum3;
        RegTensor<float> vreg_exp_sum_brc;
        RegTensor<float> vreg_exp_even1;
        RegTensor<float> vreg_exp_odd1;
        RegTensor<float> vreg_exp_even2;
        RegTensor<float> vreg_exp_odd2;
        RegTensor<float> vreg_pse1;
        RegTensor<float> vreg_pse2;
        RegTensor<float> vreg_pse3;
        RegTensor<float> vreg_pse4;
        RegTensor<float> vreg_alibi1;
        RegTensor<float> vreg_alibi2;
        RegTensor<float> vreg_alibi3;
        RegTensor<float> vreg_alibi4;
        RegTensor<float> vreg_sel_drop;
        RegTensor<float> vreg_sel_drop2;
        RegTensor<float> vreg_sink_input;
        // bfloat16_t
        RegTensor<bfloat16_t> vreg_exp_even1_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd1_bf16;
        RegTensor<bfloat16_t> vreg_exp_even2_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd2_bf16;
        RegTensor<bfloat16_t> vreg_exp1_bf16;
        RegTensor<bfloat16_t> vreg_exp2_bf16;
        RegTensor<bfloat16_t> vreg_pse_bf16_src1;
        RegTensor<bfloat16_t> vreg_pse_bf16_src2;
        RegTensor<bfloat16_t> vreg_pse1_bf16;
        RegTensor<bfloat16_t> vreg_pse2_bf16;
        RegTensor<bfloat16_t> vreg_pse3_bf16;
        RegTensor<bfloat16_t> vreg_pse4_bf16;
        // half
        RegTensor<half> vreg_exp_even1_f16;
        RegTensor<half> vreg_exp_odd1_f16;
        RegTensor<half> vreg_exp_even2_f16;
        RegTensor<half> vreg_exp_odd2_f16;
        RegTensor<half> vreg_exp1_f16;
        RegTensor<half> vreg_exp2_f16;
        RegTensor<half> vreg_pse_f16_src1;
        RegTensor<half> vreg_pse_f16_src2;
        RegTensor<half> vreg_pse1_f16;
        RegTensor<half> vreg_pse2_f16;
        RegTensor<half> vreg_pse3_f16;
        RegTensor<half> vreg_pse4_f16;
        // mxfp8
 	    RegTensor<bfloat16_t> vreg_p_scale_bf16_0;
 	    RegTensor<bfloat16_t> vreg_p_scale_bf16_1;
 	    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_0;
 	    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_1;
 	    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_pad;
 	    RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
        UnalignRegForStore ureg_max;
        UnalignRegForStore ureg_exp_sum;

        MaskReg preg_all = pset_b32(PAT_ALL);
        MaskReg preg_all_b16 = pset_b16(PAT_ALL);
        MaskReg preg_tail_n1 = plt_b32(pltTailN1, POST_UPDATE);
        MaskReg preg_ori_tail_n1 = plt_b32(pltOriTailN1, POST_UPDATE);
        MaskReg preg_tail_n2 = plt_b32(pltTailN2, POST_UPDATE);
        MaskReg preg_ori_tail_n2 = plt_b32(pltOriTailN2, POST_UPDATE);
        MaskReg preg_reduce_n = pset_b32(PAT_VL8);
        MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
 	    uint32_t maskLen = 128;
 	    MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
 	    MaskReg preg_all_b8_half = CreateMask<int8_t, MaskPattern::ALL>();
        MaskReg preg_compare1;
        MaskReg preg_compare2;
        MaskReg preg_compare3;
        MaskReg preg_compare4;

        MaskReg preg1;
        MaskReg preg2 = pset_b8(PAT_ALLF);

        MaskReg preg3;
        MaskReg preg4;
        MaskReg preg5;
        MaskReg preg6;
        // pScale
        RegTensor<float> vreg_p_scale;
        RegTensor<float> vreg_ln_p_scale;
        vbr(vreg_p_scale, static_cast<float>(pScale));
        vln(vreg_ln_p_scale, vreg_p_scale, preg_all, MODE_ZEROING);

        vbr(vreg_min, minValue);

        constexpr auto dist_norm = std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_NORM)>();
        constexpr auto dist_brc_b32 = std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_BRC_B32)>();
        constexpr auto dist_dintlv_b32 = std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_DINTLV_B32)>();
        constexpr auto dist_norm_b32 = std::integral_constant<::DistVST, static_cast<::DistVST>(StoreDist::DIST_NORM_B32)>();
        constexpr auto dist_norm_b8 = std::integral_constant<::DistVST, static_cast<::DistVST>(StoreDist::DIST_NORM_B8)>();
        constexpr auto dist_ds = std::integral_constant<::Dist, static_cast<::Dist>(MaskDist::DIST_DS)>();
        // if constexpr (hasSink) {
        //     Duplicate(vreg_sink_input, sinkValue);
        // }
        // if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
        //               pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        //     Arange(vreg_alibi1, posShift);
        //     Arange(vreg_alibi2, posShift + floatRepSize);
        //     Arange(vreg_alibi3, posShift + floatRepSize * 2);
        //     Arange(vreg_alibi4, posShift + floatRepSize * 3);
        // }
        for (uint16_t i = 0; i < m; ++i) {
            vlds(vreg_input_x1, (__ubuf__ float*)srcUb + i * 256, 0, dist_norm);
            vlds(vreg_input_x2, (__ubuf__ float*)srcUb + floatRepSize + i * 256, 0, dist_norm);
            vlds(vreg_input_x3, (__ubuf__ float*)srcUb + floatRepSize * 2 + i * 256, 0, dist_norm);
            vlds(vreg_input_x4, (__ubuf__ float*)srcUb + floatRepSize * 3 + i * 256, 0, dist_norm);
            // if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                vmuls(vreg_input_x1, vreg_input_x1, scale, preg_all, MODE_ZEROING); // Muls(scale)
                vmuls(vreg_input_x2, vreg_input_x2, scale, preg_all, MODE_ZEROING);
                vmuls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1, MODE_ZEROING);
                vmuls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2, MODE_ZEROING);
            // } else {
            //     if constexpr (IsSameType<T2, fp8_e5m2_t>::value || IsSameType<T2, fp8_e4m3fn_t>::value ||
            //                   IsSameType<T2, hifloat8_t>::value) {
            //         Muls(vreg_input_x1, vreg_input_x1, dScaleQK, preg_all); // Muls(dScaleQK)
            //         Muls(vreg_input_x2, vreg_input_x2, dScaleQK, preg_all);
            //         Muls(vreg_input_x3, vreg_input_x3, dScaleQK, preg_all);
            //         Muls(vreg_input_x4, vreg_input_x4, dScaleQK, preg_all);
            //     }
            // }
            // if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            //     if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
            //                   pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            //         Abs(vreg_pse1, vreg_alibi1, preg_all);
            //         Abs(vreg_pse2, vreg_alibi2, preg_all);
            //         Abs(vreg_pse3, vreg_alibi3, preg_all);
            //         Abs(vreg_pse4, vreg_alibi4, preg_all);
            //         if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            //             Sqrt(vreg_pse1, vreg_pse1, preg_all);
            //             Sqrt(vreg_pse2, vreg_pse2, preg_all);
            //             Sqrt(vreg_pse3, vreg_pse3, preg_all);
            //             Sqrt(vreg_pse4, vreg_pse4, preg_all);
            //         }
            //         Muls(vreg_pse1, vreg_pse1, slopes, preg_all);
            //         Muls(vreg_pse2, vreg_pse2, slopes, preg_all);
            //         Muls(vreg_pse3, vreg_pse3, slopes, preg_all);
            //         Muls(vreg_pse4, vreg_pse4, slopes, preg_all);
            //         Adds(vreg_alibi1, vreg_alibi1, -1.0f, preg_all);
            //         Adds(vreg_alibi2, vreg_alibi2, -1.0f, preg_all);
            //         Adds(vreg_alibi3, vreg_alibi3, -1.0f, preg_all);
            //         Adds(vreg_alibi4, vreg_alibi4, -1.0f, preg_all);
            //     } else {
            //         if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
            //             LoadAlign(vreg_pse_bf16_src1, pseUb + i * pseStride);
            //             LoadAlign(vreg_pse_bf16_src2, pseUb + floatRepSize * 2 + i * pseStride);
            //             Interleave(vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse_bf16_src1, vreg_pse_bf16_src1);
            //             Interleave(vreg_pse3_bf16, vreg_pse4_bf16, vreg_pse_bf16_src2, vreg_pse_bf16_src2);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_bf16, preg_all_b16);
            //         } else if constexpr (IsSameType<pseShiftType, half>::value) {
            //             LoadAlign(vreg_pse_f16_src1, pseUb + i * pseStride);
            //             LoadAlign(vreg_pse_f16_src2, pseUb + floatRepSize * 2 + i * pseStride);
            //             Interleave(vreg_pse1_f16, vreg_pse2_f16, vreg_pse_f16_src1, vreg_pse_f16_src1);
            //             Interleave(vreg_pse3_f16, vreg_pse4_f16, vreg_pse_f16_src2, vreg_pse_f16_src2);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_f16, preg_all_b16);
            //         }
            //     }
            //     Add(vreg_input_x1, vreg_input_x1, vreg_pse1, preg_all);
            //     Add(vreg_input_x2, vreg_input_x2, vreg_pse2, preg_all);
            //     Add(vreg_input_x3, vreg_input_x3, vreg_pse3, preg_ori_tail_n1);
            //     Add(vreg_input_x4, vreg_input_x4, vreg_pse4, preg_ori_tail_n2);
            // }
            // if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            //     Muls(vreg_input_x1, vreg_input_x1, scale, preg_all); // Muls(scale)
            //     Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
            //     Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
            //     Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
            // }

            // if constexpr (hasAtten == 1) {
                plds(preg_compare1, (__ubuf__ uint32_t *&)maskUb1, nPadding, dist_ds, POST_UPDATE);
                plds(preg_compare2, (__ubuf__ uint32_t *&)maskUb2, nPadding, dist_ds, POST_UPDATE);
                plds(preg_compare3, (__ubuf__ uint32_t *&)maskUb3, nPadding, dist_ds, POST_UPDATE);
                plds(preg_compare4, (__ubuf__ uint32_t *&)maskUb4, nPadding, dist_ds, POST_UPDATE);
                vsel(vreg_sel1, vreg_min, vreg_input_x1, preg_compare1);
                vsel(vreg_sel2, vreg_min, vreg_input_x2, preg_compare2);
                vsel(vreg_sel3, vreg_min, vreg_input_x3, preg_compare3);
                vsel(vreg_sel4, vreg_min, vreg_input_x4, preg_compare4);
                // vsel(vreg_sel3_new, vreg_sel3, vreg_min, preg_ori_tail_n1);
                // vsel(vreg_sel4_new, vreg_sel4, vreg_min, preg_ori_tail_n2);
                vsts(vreg_sel1, (__ubuf__ float *&)srcUb + i * 256, 0, dist_norm_b32, preg_all);
                vsts(vreg_sel2, (__ubuf__ float *&)srcUb + floatRepSize + i * 256, 0, dist_norm_b32, preg_all);
                vsts(vreg_sel3, (__ubuf__ float *&)srcUb + floatRepSize * 2 + i * 256, 0, dist_norm_b32, preg_all);
                vsts(vreg_sel4, (__ubuf__ float *&)srcUb + floatRepSize * 3 + i * 256, 0, dist_norm_b32, preg_tail_n2);
                vmax(vreg_max_tmp1, vreg_sel1, vreg_sel2, preg_all, MODE_ZEROING);
                vmax(vreg_max_tmp2, vreg_sel3, vreg_sel4, preg_all, MODE_ZEROING);
                vmax(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all, MODE_ZEROING);
                vcmax(vreg_input_max, vreg_max_tmp3, preg_all, MODE_ZEROING);
            // }
            // else {
                // Select(vreg_input_x3_new, vreg_input_x3, vreg_min, preg_ori_tail_n1);
                // Select(vreg_input_x4_new, vreg_input_x4, vreg_min, preg_ori_tail_n2);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * 256, vreg_input_x1,
                //                                                   preg_all);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * 256,
                //                                                   vreg_input_x2, preg_all);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 2 + i * 256,
                //                                                   vreg_input_x3_new, preg_tail_n1);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 3 + i * 256,
                //                                                   vreg_input_x4_new, preg_tail_n2);

                // Max(vreg_max_tmp1, vreg_input_x1, vreg_input_x2, preg_all);
                // Max(vreg_max_tmp2, vreg_input_x3_new, vreg_input_x4_new, preg_all);
                // Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                // Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max,
                //                                                                                   vreg_max_tmp3, preg_all);
            // }
            // if constexpr (hasSink) {
            //     Max(vreg_input_max, vreg_input_max, vreg_sink_input, preg_all);
            // }
            vmuls(vreg_input_max, vreg_input_max, INV_LN2, preg_all, MODE_ZEROING);
            vtrc(vreg_input_max, vreg_input_max, ROUND_C, preg_all, MODE_ZEROING);
            vmuls(vreg_input_max, vreg_input_max, LN2, preg_all, MODE_ZEROING);
            vstus(ureg_max, 1, vreg_input_max, (__ubuf__ float *&)tmpMaxUb, POST_UPDATE);
        }
        vstas(ureg_max, (__ubuf__ float *&)tmpMaxUb, 0, POST_UPDATE);
        RegTensor<float> vreg_in_max;
        vlds(vreg_in_max, (__ubuf__ float*)inMaxUb, 0, dist_norm);
        mem_bar(VST_VLD);
        vlds(vreg_input_max, (__ubuf__ float*)tmpMaxUb2, 0, dist_norm);
        vmax(vreg_input_max, vreg_in_max, vreg_input_max,
            preg_all, MODE_ZEROING); // loop1的max需要和loop0的max取一个max，保证loop1的max是当前的全局最大max，不要被loop1的max覆盖掉
        vexpdif(vreg_exp_max, vreg_in_max, vreg_input_max, preg_all, PART_EVEN);
        vsts(vreg_exp_max, (__ubuf__ float *&)expMaxUb, 0, dist_norm_b32, preg_all);
        vsts(vreg_input_max, (__ubuf__ float *&)maxUb, 0, dist_norm_b32, preg_all);

        // if constexpr (hasDrop == 1) {
        //     Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, T>(vreg_zero, 0.0f, preg_all);
        // }
        mem_bar(VST_VLD);

        for (uint16_t i = 0; i < m; ++i) {
            vlds(vreg_max_brc, (__ubuf__ float*)maxUbStart + i, 0, dist_brc_b32);
            vsub(vreg_max_brc, vreg_max_brc, vreg_ln_p_scale, preg_all, MODE_ZEROING);
            vlds(vreg_input_x1, vreg_input_x2, (__ubuf__ float*)srcUb + i * 256, 0, dist_dintlv_b32);
            vlds(vreg_input_x3, vreg_input_x4,
                 (__ubuf__ float*)srcUb + floatRepSize * 2 + i * 256, 0, dist_dintlv_b32);
            vexpdif(vreg_exp_even1, vreg_input_x1, vreg_max_brc, preg_all, PART_EVEN);
            vexpdif(vreg_exp_odd1, vreg_input_x2, vreg_max_brc, preg_all, PART_EVEN);
            vexpdif(vreg_exp_even2, vreg_input_x3, vreg_max_brc, preg_all, PART_EVEN);
            vexpdif(vreg_exp_odd2, vreg_input_x4, vreg_max_brc, preg_all, PART_EVEN);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            vadd(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all, MODE_ZEROING);
            vadd(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all, MODE_ZEROING);
            vadd(vreg_exp_sum3, vreg_exp_sum1, vreg_exp_sum2, preg_all, MODE_ZEROING);
            vcadd(vreg_exp_sum3, vreg_exp_sum3, preg_all, MODE_ZEROING);
            vstus(ureg_exp_sum, 1, vreg_exp_sum3, (__ubuf__ float *&)tmpExpSumUb, POST_UPDATE);
            // dropmask compute
            // if constexpr (hasDrop == 1) {
            //     LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
            //         preg1, (__ubuf__ uint32_t *&)dropMaskUb1, 256 >> 3);
            //     // preg1: 0011223344556677 preg2: 0000000000000000
            //     MaskInterleave<half>(preg3, preg4, preg1, preg2);
            //     // preg3: 0000110022003300 preg4: 4400550066007700
            //     MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            //     // preg5(even-4bit): 0000220044006600 preg6(odd-4bit): 1100330055007700
            //     Select(vreg_sel_drop, vreg_exp_even1, vreg_zero, preg5);
            //     Muls(vreg_exp_even1, vreg_sel_drop, divValue, preg_all);
            //     Select(vreg_sel_drop2, vreg_exp_odd1, vreg_zero, preg6);
            //     Muls(vreg_exp_odd1, vreg_sel_drop2, divValue, preg_all);
            //     LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
            //         preg1, (__ubuf__ uint32_t *&)dropMaskUb2, 256 >> 3);
            //     // preg1: 0011223344556677 preg2: 0000000000000000
            //     MaskInterleave<half>(preg3, preg4, preg1, preg2);
            //     // preg3: 0000110022003300 preg4: 4400550066007700
            //     MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            //     // preg5(even-4bit): 0000220044006600 preg6(odd-4bit): 1100330055007700
            //     Select(vreg_sel_drop, vreg_exp_even2, vreg_zero, preg5);
            //     Muls(vreg_exp_even2, vreg_sel_drop, divValue, preg_all);
            //     Select(vreg_sel_drop2, vreg_exp_odd2, vreg_zero, preg6);
            //     Muls(vreg_exp_odd2, vreg_sel_drop2, divValue, preg_all);
            // }

            // if constexpr (IsSameType<T2, bfloat16_t>::value) {
            //     Cast<T2, T, castTraitZero>(vreg_exp_even1_bf16, vreg_exp_even1, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd1_bf16, vreg_exp_odd1, preg_all);
            //     Cast<T2, T, castTraitZero>(vreg_exp_even2_bf16, vreg_exp_even2, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd2_bf16, vreg_exp_odd2, preg_all);
            //     Or((RegTensor<uint16_t> &)vreg_exp1_bf16, (RegTensor<uint16_t> &)vreg_exp_even1_bf16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd1_bf16, preg_all_b16);
            //     Or((RegTensor<uint16_t> &)vreg_exp2_bf16, (RegTensor<uint16_t> &)vreg_exp_even2_bf16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd2_bf16, preg_all_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, preg_all_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, preg_all_b16);
            // } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_2;
                // // RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
                // // RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes_2;
                // MaskReg preg_all_b8 = pset_b8(PAT_ALL);
                // uint32_t maskLen = 128;
                // MaskReg preg_all_b8_128 = plt_b8(maskLen, POST_UPDATE);
                vcvt(vreg_exp_even_f8e4m3_1, vreg_exp_even1, preg_all, ROUND_R, RS_DISABLE, PART_P0, MODE_ZEROING);
                vcvt(vreg_exp_even_f8e4m3_2, vreg_exp_even2, preg_all, ROUND_R, RS_DISABLE, PART_P0, MODE_ZEROING);
                vcvt(vreg_exp_odd_f8e4m3_1, vreg_exp_odd1, preg_all, ROUND_R, RS_DISABLE, PART_P2, MODE_ZEROING);
                vcvt(vreg_exp_odd_f8e4m3_2, vreg_exp_odd2, preg_all, ROUND_R, RS_DISABLE, PART_P2, MODE_ZEROING);
                vor((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_1, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3_1,
                   (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3_1, preg_all_b8, MODE_ZEROING);
                vor((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_2, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3_2,
                   (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3_2, preg_all_b8, MODE_ZEROING);
                vlds(vreg_exp_merge_f8e4m3_indexes, (__ubuf__ uint8_t*)indexesUb, 0, dist_norm);
                vselr((RegTensor<uint8_t> &)vreg_exp_merge_f8e4m3_1, (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_1, (RegTensor<uint8_t> &)vreg_exp_merge_f8e4m3_indexes);
                vselr((RegTensor<uint8_t> &)vreg_exp_merge_f8e4m3_2, (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_2, (RegTensor<uint8_t> &)vreg_exp_merge_f8e4m3_indexes);
                vsstb((RegTensor<uint8_t> &)vreg_exp_merge_f8e4m3_1, (__ubuf__ uint8_t *&)expUb1,
                      (blockStride << 16u) | (repeatStride & 0xFFFFU), preg_all_b8_128, POST_UPDATE);
                vsstb((RegTensor<uint8_t> &)vreg_exp_merge_f8e4m3_2, (__ubuf__ uint8_t *&)expUb2,
                      (blockStride << 16u) | (repeatStride & 0xFFFFU), preg_all_b8_128, POST_UPDATE);
            // }
        }
        vstas(ureg_exp_sum, (__ubuf__ float *&)tmpExpSumUb, 0, POST_UPDATE);
        mem_bar(VST_VLD);
        RegTensor<float> first_loop_sum;
        vlds(first_loop_sum, (__ubuf__ float*)inExpSumUb, 0, dist_norm);
        vlds(vreg_exp_sum_brc, (__ubuf__ float*)tmpExpSumUb2, 0, dist_norm);
        vmul(first_loop_sum, vreg_exp_max, first_loop_sum, preg_all, MODE_ZEROING);
        vadd(vreg_exp_sum3, first_loop_sum, vreg_exp_sum_brc, preg_all, MODE_ZEROING);
        vsts(vreg_exp_sum3, (__ubuf__ float *&)expSumUb, 0, dist_norm_b32, preg_all);
        // pScale update
 	    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_dst0;
 	    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_dst1;
 	    vbr(vreg_p_scale, static_cast<float>(1.0f));
        vmul(vreg_p_scale, vreg_exp_max, vreg_p_scale, preg_all, MODE_ZEROING);
 	    vcvt(vreg_p_scale_bf16_0, vreg_p_scale, preg_all, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
 	    vshls((RegTensor<uint16_t> &)vreg_p_scale_bf16_0, (RegTensor<uint16_t> &)vreg_p_scale_bf16_0, SHIFT_ONE_BIT, preg_all_b16, MODE_ZEROING);
        vshrs((RegTensor<uint16_t> &)vreg_p_scale_bf16_0, (RegTensor<uint16_t> &)vreg_p_scale_bf16_0, SHIFT_EIGHT_BIT, preg_all_b16, MODE_ZEROING);
        vcvt((RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_0, (RegTensor<uint16_t> &)vreg_p_scale_bf16_0, preg_all_b16, RS_DISABLE, PART_EVEN,
            MODE_ZEROING);
 	    vcvt(vreg_p_scale_bf16_1, vreg_p_scale, preg_all, ROUND_R, RS_DISABLE, PART_ODD, MODE_ZEROING);
 	    vshls((RegTensor<uint16_t> &)vreg_p_scale_bf16_1, (RegTensor<uint16_t> &)vreg_p_scale_bf16_1, SHIFT_ONE_BIT, preg_all_b16, MODE_ZEROING);
        vshrs((RegTensor<uint16_t> &)vreg_p_scale_bf16_1, (RegTensor<uint16_t> &)vreg_p_scale_bf16_1, SHIFT_EIGHT_BIT, preg_all_b16, MODE_ZEROING);
        vcvt((RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_1, (RegTensor<uint16_t> &)vreg_p_scale_bf16_1, preg_all_b16, RS_DISABLE, PART_EVEN,
            MODE_ZEROING);
 	    vor((RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_0, (RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_0,
                   (RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_1, preg_all_b8, MODE_ZEROING);
        const int8_t val_value = 0x7f;
        vdup((RegTensor<int8_t>&)vreg_p_scale_f8e8m0_1, (int8_t&)val_value, preg_all_b8, MODE_ZEROING);
        vdintlv((RegTensor<int8_t>&)vreg_p_scale_f8e8m0_dst0, (RegTensor<int8_t>&)vreg_p_scale_f8e8m0_dst1, (RegTensor<int8_t>&)vreg_p_scale_f8e8m0_0, (RegTensor<int8_t>&)vreg_p_scale_f8e8m0_1);
 	    vsts((RegTensor<uint8_t>&)vreg_p_scale_f8e8m0_dst0, (__ubuf__ uint8_t*)pScaleSubLoop0, 0, dist_norm_b8, preg_all_b8);
    }

    template <typename T, typename T2, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 256, bool hasAtten = 0>
    __simd_vf__ void process_vec1_update_subloop0(__ubuf__ fp8_e4m3fn_t *expUb1, __ubuf__ fp8_e4m3fn_t *expUb2, __ubuf__ float *maxUb, __ubuf__ float *maxUbStart,
        __ubuf__ float *srcUb, __ubuf__ float * expMaxUb, __ubuf__ float * inMaxUb, __ubuf__ float * expSumUb, __ubuf__ float * inExpSumUb, __ubuf__ float * tmpExpSumUb,
        __ubuf__ float * tmpExpSumUb2, __ubuf__ float * tmpMaxUb, __ubuf__ float * tmpMaxUb2,
        __ubuf__ uint8_t *indexesUb, __ubuf__ uint32_t *maskUb1, __ubuf__ uint32_t *maskUb2, __ubuf__ uint32_t *maskUb3,
        __ubuf__ uint32_t *maskUb4, __ubuf__ fp8_e8m0_t *pScaleSubLoop0, __ubuf__ float *preLoopMaxUb, __ubuf__ float *preLoopSumUb, __ubuf__ float *firstLoopSumUb,
        const uint32_t nPadding, const uint32_t blockStride, const uint32_t repeatStride, uint32_t pltOriTailN1, uint32_t pltOriTailN2, uint32_t pltTailN1, uint32_t pltTailN2,
        const uint16_t m, const float scale, const float minValue, const float pScale = 1.0f)//m=16
    {
        RegTensor<float> vreg_min;
        RegTensor<float> vreg_sel1;
        RegTensor<float> vreg_sel2;
        RegTensor<float> vreg_sel3;
        RegTensor<float> vreg_sel4;
        RegTensor<float> vreg_sel3_new;
        RegTensor<float> vreg_sel4_new;
        RegTensor<float> vreg_input_x1;
        RegTensor<float> vreg_input_x2;
        RegTensor<float> vreg_input_x3;
        RegTensor<float> vreg_input_x4;
        RegTensor<float> vreg_input_x3_new;
        RegTensor<float> vreg_input_x4_new;
        RegTensor<float> vreg_max_tmp1;
        RegTensor<float> vreg_max_tmp2;
        RegTensor<float> vreg_max_tmp3;
        RegTensor<float> vreg_input_max;
        RegTensor<float> vreg_exp_max;
        RegTensor<float> vreg_subloop_update;
        RegTensor<float> vreg_pre_loop_max;
        RegTensor<float> vreg_max_new;
        RegTensor<float> vreg_exp_sum1;
        RegTensor<float> vreg_exp_sum2;
        RegTensor<float> vreg_exp_sum3;
        RegTensor<float> vreg_in_exp_sum;
        RegTensor<float> vreg_exp_sum_brc;
        RegTensor<float> vreg_pre_sum;
        RegTensor<float> vreg_in_max;
        RegTensor<float> vreg_max;
        RegTensor<float> vreg_exp_even1;
        RegTensor<float> vreg_exp_odd1;
        RegTensor<float> vreg_exp_even2;
        RegTensor<float> vreg_exp_odd2;
        RegTensor<float> vreg_pse1;
        RegTensor<float> vreg_pse2;
        RegTensor<float> vreg_pse3;
        RegTensor<float> vreg_pse4;
        RegTensor<float> vreg_alibi1;
        RegTensor<float> vreg_alibi2;
        RegTensor<float> vreg_alibi3;
        RegTensor<float> vreg_alibi4;
        RegTensor<float> vreg_sel_drop;
        RegTensor<float> vreg_sel_drop2;
        RegTensor<float> vreg_zero;
        RegTensor<float> vreg_sink_input;
        // bfloat16_t
        RegTensor<bfloat16_t> vreg_exp_even1_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd1_bf16;
        RegTensor<bfloat16_t> vreg_exp_even2_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd2_bf16;
        RegTensor<bfloat16_t> vreg_exp1_bf16;
        RegTensor<bfloat16_t> vreg_exp2_bf16;
        RegTensor<bfloat16_t> vreg_pse_bf16_src1;
        RegTensor<bfloat16_t> vreg_pse_bf16_src2;
        RegTensor<bfloat16_t> vreg_pse1_bf16;
        RegTensor<bfloat16_t> vreg_pse2_bf16;
        RegTensor<bfloat16_t> vreg_pse3_bf16;
        RegTensor<bfloat16_t> vreg_pse4_bf16;
        // half
        RegTensor<half> vreg_exp_even1_f16;
        RegTensor<half> vreg_exp_odd1_f16;
        RegTensor<half> vreg_exp_even2_f16;
        RegTensor<half> vreg_exp_odd2_f16;
        RegTensor<half> vreg_exp1_f16;
        RegTensor<half> vreg_exp2_f16;
        RegTensor<half> vreg_pse_f16_src1;
        RegTensor<half> vreg_pse_f16_src2;
        RegTensor<half> vreg_pse1_f16;
        RegTensor<half> vreg_pse2_f16;
        RegTensor<half> vreg_pse3_f16;
        RegTensor<half> vreg_pse4_f16;
        // mxfp8
 	    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0;

        UnalignRegForStore ureg_max;
        UnalignRegForStore ureg_exp_sum;

        MaskReg preg_all = pset_b32(PAT_ALL);
        MaskReg preg_all_b16 = pset_b16(PAT_ALL);
        // MaskReg preg_n_b16 = UpdateMask<uint16_t>(pltN);
        MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
        MaskReg preg_tail_n1 = plt_b32(pltTailN1, POST_UPDATE);
        MaskReg preg_ori_tail_n1 = plt_b32(pltOriTailN1, POST_UPDATE);
        MaskReg preg_tail_n2 = plt_b32(pltTailN2, POST_UPDATE);
        MaskReg preg_ori_tail_n2 = plt_b32(pltOriTailN2, POST_UPDATE);
        MaskReg preg_compare1;
        MaskReg preg_compare2;
        MaskReg preg_compare3;
        MaskReg preg_compare4;
        MaskReg preg1;
        MaskReg preg2 = pset_b8(PAT_ALLF);
        MaskReg preg3;
        MaskReg preg4;
        MaskReg preg5;
        MaskReg preg6;
        // pScale
        RegTensor<float> vreg_p_scale;
        RegTensor<float> vreg_ln_p_scale;
        vbr(vreg_p_scale, static_cast<float>(pScale));
        vln(vreg_ln_p_scale, vreg_p_scale, preg_all, MODE_ZEROING);

        vbr(vreg_min, minValue);

        constexpr auto dist_norm = std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_NORM)>();
        constexpr auto dist_brc_b32 = std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_BRC_B32)>();
        constexpr auto dist_dintlv_b32 = std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_DINTLV_B32)>();
        constexpr auto dist_norm_b32 = std::integral_constant<::DistVST, static_cast<::DistVST>(StoreDist::DIST_NORM_B32)>();
        constexpr auto dist_norm_b8 = std::integral_constant<::DistVST, static_cast<::DistVST>(StoreDist::DIST_NORM_B8)>();
        constexpr auto dist_ds = std::integral_constant<::Dist, static_cast<::Dist>(MaskDist::DIST_DS)>();
        // if constexpr (hasSink) {
        //     Duplicate(vreg_sink_input, sinkValue);
        // }
        // if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
        //               pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        //     Arange(vreg_alibi1, posShift);
        //     Arange(vreg_alibi2, posShift + floatRepSize);
        //     Arange(vreg_alibi3, posShift + floatRepSize * 2);
        //     Arange(vreg_alibi4, posShift + floatRepSize * 3);
        // }
        for (uint16_t i = 0; i < m; ++i) {
            vlds(vreg_input_x1, (__ubuf__ float*)srcUb + i * 256, 0, dist_norm);
            vlds(vreg_input_x2, (__ubuf__ float*)srcUb + floatRepSize + i * 256, 0, dist_norm);
            vlds(vreg_input_x3, (__ubuf__ float*)srcUb + floatRepSize * 2 + i * 256, 0, dist_norm);
            vlds(vreg_input_x4, (__ubuf__ float*)srcUb + floatRepSize * 3 + i * 256, 0, dist_norm);
            // if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                vmuls(vreg_input_x1, vreg_input_x1, scale, preg_all, MODE_ZEROING); // Muls(scale)
                vmuls(vreg_input_x2, vreg_input_x2, scale, preg_all, MODE_ZEROING);
                vmuls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1, MODE_ZEROING);
                vmuls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2, MODE_ZEROING);
            // }
            // if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            //     if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
            //                   pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            //         Abs(vreg_pse1, vreg_alibi1, preg_all);
            //         Abs(vreg_pse2, vreg_alibi2, preg_all);
            //         Abs(vreg_pse3, vreg_alibi3, preg_all);
            //         Abs(vreg_pse4, vreg_alibi4, preg_all);
            //         if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            //             Sqrt(vreg_pse1, vreg_pse1, preg_all);
            //             Sqrt(vreg_pse2, vreg_pse2, preg_all);
            //             Sqrt(vreg_pse3, vreg_pse3, preg_all);
            //             Sqrt(vreg_pse4, vreg_pse4, preg_all);
            //         }
            //         Muls(vreg_pse1, vreg_pse1, slopes, preg_all);
            //         Muls(vreg_pse2, vreg_pse2, slopes, preg_all);
            //         Muls(vreg_pse3, vreg_pse3, slopes, preg_all);
            //         Muls(vreg_pse4, vreg_pse4, slopes, preg_all);
            //         Adds(vreg_alibi1, vreg_alibi1, -1.0f, preg_all);
            //         Adds(vreg_alibi2, vreg_alibi2, -1.0f, preg_all);
            //         Adds(vreg_alibi3, vreg_alibi3, -1.0f, preg_all);
            //         Adds(vreg_alibi4, vreg_alibi4, -1.0f, preg_all);
            //     } else {
            //         if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
            //             LoadAlign(vreg_pse_bf16_src1, pseUb + i * pseStride);
            //             LoadAlign(vreg_pse_bf16_src2, pseUb + floatRepSize * 2 + i * pseStride);
            //             Interleave(vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse_bf16_src1, vreg_pse_bf16_src1);
            //             Interleave(vreg_pse3_bf16, vreg_pse4_bf16, vreg_pse_bf16_src2, vreg_pse_bf16_src2);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_bf16, preg_all_b16);
            //         } else if constexpr (IsSameType<pseShiftType, half>::value) {
            //             LoadAlign(vreg_pse_f16_src1, pseUb + i * pseStride);
            //             LoadAlign(vreg_pse_f16_src2, pseUb + floatRepSize * 2 + i * pseStride);
            //             Interleave(vreg_pse1_f16, vreg_pse2_f16, vreg_pse_f16_src1, vreg_pse_f16_src1);
            //             Interleave(vreg_pse3_f16, vreg_pse4_f16, vreg_pse_f16_src2, vreg_pse_f16_src2);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_f16, preg_all_b16);
            //         }
            //     }
            //     Add(vreg_input_x1, vreg_input_x1, vreg_pse1, preg_all);
            //     Add(vreg_input_x2, vreg_input_x2, vreg_pse2, preg_all);
            //     Add(vreg_input_x3, vreg_input_x3, vreg_pse3, preg_ori_tail_n1);
            //     Add(vreg_input_x4, vreg_input_x4, vreg_pse4, preg_ori_tail_n2);
            // }
            // if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            //     Muls(vreg_input_x1, vreg_input_x1, scale, preg_all); // Muls(scale)
            //     Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
            //     Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
            //     Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
            // }

            // if constexpr (hasAtten == 1) {
                // atten mask
                plds(preg_compare1, (__ubuf__ uint32_t *&)maskUb1, nPadding, dist_ds, POST_UPDATE);
                plds(preg_compare2, (__ubuf__ uint32_t *&)maskUb2, nPadding, dist_ds, POST_UPDATE);
                plds(preg_compare3, (__ubuf__ uint32_t *&)maskUb3, nPadding, dist_ds, POST_UPDATE);
                plds(preg_compare4, (__ubuf__ uint32_t *&)maskUb4, nPadding, dist_ds, POST_UPDATE);
                vsel(vreg_sel1, vreg_min, vreg_input_x1, preg_compare1);
                vsel(vreg_sel2, vreg_min, vreg_input_x2, preg_compare2);
                vsel(vreg_sel3, vreg_min, vreg_input_x3, preg_compare3);
                vsel(vreg_sel4, vreg_min, vreg_input_x4, preg_compare4);
                // vsel(vreg_sel3_new, vreg_sel3, vreg_min, preg_ori_tail_n1);
                // vsel(vreg_sel4_new, vreg_sel4, vreg_min, preg_ori_tail_n2);
                vsts(vreg_sel1, (__ubuf__ float *&)srcUb + i * 256, 0, dist_norm_b32, preg_all);
                vsts(vreg_sel2, (__ubuf__ float *&)srcUb + floatRepSize + i * 256, 0, dist_norm_b32, preg_all);
                vsts(vreg_sel3, (__ubuf__ float *&)srcUb + floatRepSize * 2 + i * 256, 0, dist_norm_b32, preg_all);
                vsts(vreg_sel4, (__ubuf__ float *&)srcUb + floatRepSize * 3 + i * 256, 0, dist_norm_b32, preg_all);
                vmax(vreg_max_tmp1, vreg_sel1, vreg_sel2, preg_all, MODE_ZEROING);
                vmax(vreg_max_tmp2, vreg_sel3, vreg_sel4, preg_all, MODE_ZEROING);
                vmax(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all, MODE_ZEROING);
                vcmax(vreg_input_max, vreg_max_tmp3, preg_all, MODE_ZEROING);
            // } else {
                // Select(vreg_input_x3_new, vreg_input_x3, vreg_min, preg_ori_tail_n1);
                // Select(vreg_input_x4_new, vreg_input_x4, vreg_min, preg_ori_tail_n2);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * 256, vreg_input_x1,
                //                                                   preg_all);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * 256,
                //                                                   vreg_input_x2, preg_all);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 2 + i * 256,
                //                                                   vreg_input_x3_new, preg_all);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 3 + i * 256,
                //                                                   vreg_input_x4_new, preg_all);
                // Max(vreg_max_tmp1, vreg_input_x1, vreg_input_x2, preg_all);
                // Max(vreg_max_tmp2, vreg_input_x3_new, vreg_input_x4_new, preg_all);
                // Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                // Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max,
                //                                                                                   vreg_max_tmp3, preg_all);
            // }
            // if constexpr (hasSink) {
            //     Max(vreg_input_max, vreg_input_max, vreg_sink_input, preg_all);
            // }
            vmuls(vreg_input_max, vreg_input_max, INV_LN2, preg_all, MODE_ZEROING);
            vtrc(vreg_input_max, vreg_input_max, ROUND_C, preg_all, MODE_ZEROING);
            vmuls(vreg_input_max, vreg_input_max, LN2, preg_all, MODE_ZEROING);
            vstus(ureg_max, 1, vreg_input_max, (__ubuf__ float *&)tmpMaxUb, POST_UPDATE);
        }
        vstas(ureg_max, (__ubuf__ float *&)tmpMaxUb, 0, POST_UPDATE);

        vlds(vreg_in_max, (__ubuf__ float*)inMaxUb, 0, dist_norm);
        mem_bar(VST_VLD);
        vlds(vreg_input_max, (__ubuf__ float*)tmpMaxUb2, 0, dist_norm);
        vsts(vreg_in_max, (__ubuf__ float *&)preLoopMaxUb, 0, dist_norm_b32, preg_all);
        vmax(vreg_max_new, vreg_in_max, vreg_input_max, preg_all, MODE_ZEROING);
        vexpdif(vreg_exp_max, vreg_in_max, vreg_max_new, preg_all, PART_EVEN);
        vsts(vreg_exp_max, (__ubuf__ float *&)expMaxUb, 0, dist_norm_b32, preg_all);
        vsts(vreg_max_new, (__ubuf__ float *&)maxUb, 0, dist_norm_b32, preg_all);


        // if constexpr (hasDrop == 1) {
        //     Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_zero, 0.0f, preg_all);
        // }
        mem_bar(VST_VLD);

        for (uint16_t i = 0; i < m; ++i) {
            vlds(vreg_max, (__ubuf__ float*)maxUbStart + i, 0, dist_brc_b32);
            vsub(vreg_max, vreg_max, vreg_ln_p_scale, preg_all, MODE_ZEROING);
            vlds(vreg_input_x1, vreg_input_x2, (__ubuf__ float*)srcUb + i * 256, 0, dist_dintlv_b32);
            vlds(vreg_input_x3, vreg_input_x4,
                 (__ubuf__ float*)srcUb + floatRepSize * 2 + i * 256, 0, dist_dintlv_b32);
            vexpdif(vreg_exp_even1, vreg_input_x1, vreg_max, preg_all, PART_EVEN);
            vexpdif(vreg_exp_odd1, vreg_input_x2, vreg_max, preg_all, PART_EVEN);
            vexpdif(vreg_exp_even2, vreg_input_x3, vreg_max, preg_all, PART_EVEN);
            vexpdif(vreg_exp_odd2, vreg_input_x4, vreg_max, preg_all, PART_EVEN);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            vadd(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all, MODE_ZEROING);
            vadd(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all, MODE_ZEROING);
            vadd(vreg_exp_sum3, vreg_exp_sum1, vreg_exp_sum2, preg_all, MODE_ZEROING);
            vcadd(vreg_exp_sum3, vreg_exp_sum3, preg_all, MODE_ZEROING);
            vstus(ureg_exp_sum, 1, vreg_exp_sum3, (__ubuf__ float *&)tmpExpSumUb, POST_UPDATE);

            // dropmask compute
            // if constexpr (hasDrop == 1) {
            //     LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
            //         preg1, (__ubuf__ uint32_t *&)dropMaskUb1, 256 >> 3);
            //     MaskInterleave<half>(preg3, preg4, preg1, preg2);
            //     MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            //     Select(vreg_sel_drop, vreg_exp_even1, vreg_zero, preg5);
            //     Muls(vreg_exp_even1, vreg_sel_drop, divValue, preg_all);
            //     Select(vreg_sel_drop2, vreg_exp_odd1, vreg_zero, preg6);
            //     Muls(vreg_exp_odd1, vreg_sel_drop2, divValue, preg_all);
            //     LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
            //         preg1, (__ubuf__ uint32_t *&)dropMaskUb2, 256 >> 3);
            //     MaskInterleave<half>(preg3, preg4, preg1, preg2);
            //     MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            //     Select(vreg_sel_drop, vreg_exp_even2, vreg_zero, preg5);
            //     Muls(vreg_exp_even2, vreg_sel_drop, divValue, preg_all);
            //     Select(vreg_sel_drop2, vreg_exp_odd2, vreg_zero, preg6);
            //     Muls(vreg_exp_odd2, vreg_sel_drop2, divValue, preg_all);
            // }

            // if constexpr (IsSameType<T2, bfloat16_t>::value) {
            //     Cast<T2, T, castTraitZero>(vreg_exp_even1_bf16, vreg_exp_even1, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd1_bf16, vreg_exp_odd1, preg_all);
            //     Cast<T2, T, castTraitZero>(vreg_exp_even2_bf16, vreg_exp_even2, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd2_bf16, vreg_exp_odd2, preg_all);
            //     Or((RegTensor<uint16_t> &)vreg_exp1_bf16, (RegTensor<uint16_t> &)vreg_exp_even1_bf16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd1_bf16, preg_all_b16);
            //     Or((RegTensor<uint16_t> &)vreg_exp2_bf16, (RegTensor<uint16_t> &)vreg_exp_even2_bf16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd2_bf16, preg_all_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, preg_n_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, preg_n_b16);
            // } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_2;
                RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
                // RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes_2;
                MaskReg preg_all_b8 = pset_b8(PAT_ALL);
                uint32_t maskLen = 128;
                MaskReg preg_all_b8_128 = plt_b8(maskLen, POST_UPDATE);
                vcvt(vreg_exp_even_f8e4m3_1, vreg_exp_even1, preg_all, ROUND_R, RS_DISABLE, PART_P0, MODE_ZEROING);
                vcvt(vreg_exp_even_f8e4m3_2, vreg_exp_even2, preg_all, ROUND_R, RS_DISABLE, PART_P0, MODE_ZEROING);
                vcvt(vreg_exp_odd_f8e4m3_1, vreg_exp_odd1, preg_all, ROUND_R, RS_DISABLE, PART_P2, MODE_ZEROING);
                vcvt(vreg_exp_odd_f8e4m3_2, vreg_exp_odd2, preg_all, ROUND_R, RS_DISABLE, PART_P2, MODE_ZEROING);
                vor((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_1, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3_1,
                   (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3_1, preg_all_b8, MODE_ZEROING);
                vor((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_2, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3_2,
                   (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3_2, preg_all_b8, MODE_ZEROING);
                vlds(vreg_exp_merge_f8e4m3_indexes, (__ubuf__ uint8_t*)indexesUb, 0, dist_norm);
                vselr((RegTensor<uint8_t> &)vreg_exp_merge_f8e4m3_1, (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_1, (RegTensor<uint8_t> &)vreg_exp_merge_f8e4m3_indexes);
                vselr((RegTensor<uint8_t> &)vreg_exp_merge_f8e4m3_2, (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_2, (RegTensor<uint8_t> &)vreg_exp_merge_f8e4m3_indexes);
                vsstb((RegTensor<uint8_t> &)vreg_exp_merge_f8e4m3_1, (__ubuf__ uint8_t *&)expUb1,
                      (blockStride << 16u) | (repeatStride & 0xFFFFU), preg_all_b8_128, POST_UPDATE);
                vsstb((RegTensor<uint8_t> &)vreg_exp_merge_f8e4m3_2, (__ubuf__ uint8_t *&)expUb2,
                      (blockStride << 16u) | (repeatStride & 0xFFFFU), preg_all_b8_128, POST_UPDATE);
            // } else if constexpr (IsSameType<T2, half>::value) {
            //     Cast<T2, T, castTraitZero>(vreg_exp_even1_f16, vreg_exp_even1, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd1_f16, vreg_exp_odd1, preg_all);
            //     Cast<T2, T, castTraitZero>(vreg_exp_even2_f16, vreg_exp_even2, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd2_f16, vreg_exp_odd2, preg_all);
            //     Or((RegTensor<uint16_t> &)vreg_exp1_f16, (RegTensor<uint16_t> &)vreg_exp_even1_f16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd1_f16, preg_all_b16);
            //     Or((RegTensor<uint16_t> &)vreg_exp2_f16, (RegTensor<uint16_t> &)vreg_exp_even2_f16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd2_f16, preg_all_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb1), vreg_exp1_f16, blockStride, repeatStride, preg_n_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb2), vreg_exp2_f16, blockStride, repeatStride, preg_n_b16);
            // }
        }
        vstas(ureg_exp_sum, (__ubuf__ float *&)tmpExpSumUb, 0, POST_UPDATE);

        mem_bar(VST_VLD);
        vlds(vreg_in_exp_sum, (__ubuf__ float*)inExpSumUb, 0, dist_norm);
        vlds(vreg_exp_sum_brc, (__ubuf__ float*)tmpExpSumUb2, 0, dist_norm);
        vsts(vreg_in_exp_sum, (__ubuf__ float *&)preLoopSumUb, 0, dist_norm_b32, preg_all);
        vmul(vreg_exp_max, vreg_exp_max, vreg_in_exp_sum, preg_all, MODE_ZEROING);
        vadd(vreg_exp_max, vreg_exp_max, vreg_exp_sum_brc,
            preg_all, MODE_ZEROING);
        vsts(vreg_exp_max, (__ubuf__ float *&)expSumUb, 0, dist_norm_b32, preg_all);
        vsts(vreg_exp_sum_brc, (__ubuf__ float *&)firstLoopSumUb, 0, dist_norm_b32, preg_all); // 传给loop 1用于更新rowSum
        const int8_t val_value = 0x7f;
        vdup((RegTensor<int8_t>&)vreg_p_scale_f8e8m0, (int8_t&)val_value, preg_all_b8, MODE_ZEROING);
        vsts((RegTensor<uint8_t>&)vreg_p_scale_f8e8m0, (__ubuf__ uint8_t*)pScaleSubLoop0, 0, dist_norm_b8, preg_all_b8);
    }

    template <typename T, typename T2, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 256, bool hasAtten = 0>
    __simd_vf__ void process_vec1_update_subloop1(__ubuf__ fp8_e4m3fn_t *expUb1, __ubuf__ fp8_e4m3fn_t *expUb2, __ubuf__ float *maxUb, __ubuf__ float *maxUbStart,
        __ubuf__ float *srcUb, __ubuf__ float * expMaxUb, __ubuf__ float * inMaxUb, __ubuf__ float * expSumUb, __ubuf__ float * inExpSumUb, __ubuf__ float * tmpExpSumUb,
        __ubuf__ float * tmpExpSumUb2, __ubuf__ float * tmpMaxUb, __ubuf__ float * tmpMaxUb2,
        __ubuf__ uint8_t *indexesUb, __ubuf__ uint32_t *maskUb1, __ubuf__ uint32_t *maskUb2, __ubuf__ uint32_t *maskUb3,
        __ubuf__ uint32_t *maskUb4, __ubuf__ fp8_e8m0_t *pScaleSubLoop0, __ubuf__ float *preLoopMaxUb, __ubuf__ float *preLoopSumUb, __ubuf__ float *firstLoopSumUb,
        const uint32_t nPadding, const uint32_t blockStride, const uint32_t repeatStride, uint32_t pltOriTailN1, uint32_t pltOriTailN2, uint32_t pltTailN1, uint32_t pltTailN2,
        const uint16_t m, const float scale, const float minValue, const float pScale = 1.0f)//m=16
    {
        RegTensor<float> vreg_min;
        RegTensor<float> vreg_sel1;
        RegTensor<float> vreg_sel2;
        RegTensor<float> vreg_sel3;
        RegTensor<float> vreg_sel4;
        RegTensor<float> vreg_sel3_new;
        RegTensor<float> vreg_sel4_new;
        RegTensor<float> vreg_input_x1;
        RegTensor<float> vreg_input_x2;
        RegTensor<float> vreg_input_x3;
        RegTensor<float> vreg_input_x4;
        RegTensor<float> vreg_input_x3_new;
        RegTensor<float> vreg_input_x4_new;
        RegTensor<float> vreg_max_tmp1;
        RegTensor<float> vreg_max_tmp2;
        RegTensor<float> vreg_max_tmp3;
        RegTensor<float> vreg_input_max;
        RegTensor<float> vreg_exp_max;
        RegTensor<float> vreg_subloop_update;
        RegTensor<float> vreg_pre_loop_max;
        RegTensor<float> vreg_max_new;
        RegTensor<float> vreg_exp_sum1;
        RegTensor<float> vreg_exp_sum2;
        RegTensor<float> vreg_exp_sum3;
        RegTensor<float> vreg_in_exp_sum;
        RegTensor<float> vreg_exp_sum_brc;
        RegTensor<float> vreg_pre_sum;
        RegTensor<float> vreg_in_max;
        RegTensor<float> vreg_max;
        RegTensor<float> vreg_exp_even1;
        RegTensor<float> vreg_exp_odd1;
        RegTensor<float> vreg_exp_even2;
        RegTensor<float> vreg_exp_odd2;
        RegTensor<float> vreg_pse1;
        RegTensor<float> vreg_pse2;
        RegTensor<float> vreg_pse3;
        RegTensor<float> vreg_pse4;
        RegTensor<float> vreg_alibi1;
        RegTensor<float> vreg_alibi2;
        RegTensor<float> vreg_alibi3;
        RegTensor<float> vreg_alibi4;
        RegTensor<float> vreg_sel_drop;
        RegTensor<float> vreg_sel_drop2;
        RegTensor<float> vreg_zero;
        RegTensor<float> vreg_sink_input;
        // bfloat16_t
        RegTensor<bfloat16_t> vreg_exp_even1_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd1_bf16;
        RegTensor<bfloat16_t> vreg_exp_even2_bf16;
        RegTensor<bfloat16_t> vreg_exp_odd2_bf16;
        RegTensor<bfloat16_t> vreg_exp1_bf16;
        RegTensor<bfloat16_t> vreg_exp2_bf16;
        RegTensor<bfloat16_t> vreg_pse_bf16_src1;
        RegTensor<bfloat16_t> vreg_pse_bf16_src2;
        RegTensor<bfloat16_t> vreg_pse1_bf16;
        RegTensor<bfloat16_t> vreg_pse2_bf16;
        RegTensor<bfloat16_t> vreg_pse3_bf16;
        RegTensor<bfloat16_t> vreg_pse4_bf16;
        // half
        RegTensor<half> vreg_exp_even1_f16;
        RegTensor<half> vreg_exp_odd1_f16;
        RegTensor<half> vreg_exp_even2_f16;
        RegTensor<half> vreg_exp_odd2_f16;
        RegTensor<half> vreg_exp1_f16;
        RegTensor<half> vreg_exp2_f16;
        RegTensor<half> vreg_pse_f16_src1;
        RegTensor<half> vreg_pse_f16_src2;
        RegTensor<half> vreg_pse1_f16;
        RegTensor<half> vreg_pse2_f16;
        RegTensor<half> vreg_pse3_f16;
        RegTensor<half> vreg_pse4_f16;
        // mxfp8
 	    RegTensor<bfloat16_t> vreg_p_scale_bf16_0;
 	    RegTensor<bfloat16_t> vreg_p_scale_bf16_1;
 	    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_0;
 	    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_1;
 	    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_pad;
 	    RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;

        UnalignRegForStore ureg_max;
        UnalignRegForStore ureg_exp_sum;

        MaskReg preg_all = pset_b32(PAT_ALL);
        MaskReg preg_all_b16 = pset_b16(PAT_ALL);
        // MaskReg preg_n_b16 = UpdateMask<uint16_t>(pltN);
        MaskReg preg_tail_n1 = plt_b32(pltTailN1, POST_UPDATE);
        MaskReg preg_ori_tail_n1 = plt_b32(pltOriTailN1, POST_UPDATE);
        MaskReg preg_tail_n2 = plt_b32(pltTailN2, POST_UPDATE);
        MaskReg preg_ori_tail_n2 = plt_b32(pltOriTailN2, POST_UPDATE);
        MaskReg preg_all_b8 = CreateMask<T2, MaskPattern::ALL>();
 	    uint32_t maskLen = 128;
 	    MaskReg preg_all_b8_128 = UpdateMask<T2>(maskLen);
 	    MaskReg preg_all_b8_half = CreateMask<int8_t, MaskPattern::ALL>();
        MaskReg preg_compare1;
        MaskReg preg_compare2;
        MaskReg preg_compare3;
        MaskReg preg_compare4;
        MaskReg preg1;
        MaskReg preg2 = pset_b8(PAT_ALLF);
        MaskReg preg3;
        MaskReg preg4;
        MaskReg preg5;
        MaskReg preg6;
        // pScale
        RegTensor<float> vreg_p_scale;
        RegTensor<float> vreg_ln_p_scale;
        vbr(vreg_p_scale, static_cast<float>(pScale));
        vln(vreg_ln_p_scale, vreg_p_scale, preg_all, MODE_ZEROING);

        vbr(vreg_min, minValue);

        constexpr auto dist_norm = std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_NORM)>();
        constexpr auto dist_brc_b32 = std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_BRC_B32)>();
        constexpr auto dist_dintlv_b32 = std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_DINTLV_B32)>();
        constexpr auto dist_norm_b32 = std::integral_constant<::DistVST, static_cast<::DistVST>(StoreDist::DIST_NORM_B32)>();
        constexpr auto dist_norm_b8 = std::integral_constant<::DistVST, static_cast<::DistVST>(StoreDist::DIST_NORM_B8)>();
        constexpr auto dist_ds = std::integral_constant<::Dist, static_cast<::Dist>(MaskDist::DIST_DS)>();
        // if constexpr (hasSink) {
        //     Duplicate(vreg_sink_input, sinkValue);
        // }
        // if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
        //               pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
        //     Arange(vreg_alibi1, posShift);
        //     Arange(vreg_alibi2, posShift + floatRepSize);
        //     Arange(vreg_alibi3, posShift + floatRepSize * 2);
        //     Arange(vreg_alibi4, posShift + floatRepSize * 3);
        // }
        // x_max = max(src, axis=-1, keepdims=True); x_max = Max(x_max, inMax)
        for (uint16_t i = 0; i < m; ++i) {
            vlds(vreg_input_x1, (__ubuf__ float*)srcUb + i * 256, 0, dist_norm);
            vlds(vreg_input_x2, (__ubuf__ float*)srcUb + floatRepSize + i * 256, 0, dist_norm);
            vlds(vreg_input_x3, (__ubuf__ float*)srcUb + floatRepSize * 2 + i * 256, 0, dist_norm);
            vlds(vreg_input_x4, (__ubuf__ float*)srcUb + floatRepSize * 3 + i * 256, 0, dist_norm);
            // if constexpr (pseMode != PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
                vmuls(vreg_input_x1, vreg_input_x1, scale, preg_all, MODE_ZEROING); // Muls(scale)
                vmuls(vreg_input_x2, vreg_input_x2, scale, preg_all, MODE_ZEROING);
                vmuls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1, MODE_ZEROING);
                vmuls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2, MODE_ZEROING);
            // }
            // if constexpr (pseMode != PseTypeEnum::PSE_NONE_TYPE) {
            //     if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_TYPE ||
            //                   pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            //         Abs(vreg_pse1, vreg_alibi1, preg_all);
            //         Abs(vreg_pse2, vreg_alibi2, preg_all);
            //         Abs(vreg_pse3, vreg_alibi3, preg_all);
            //         Abs(vreg_pse4, vreg_alibi4, preg_all);
            //         if constexpr (pseMode == PseTypeEnum::PSE_INNER_MUL_ADD_SQRT_TYPE) {
            //             Sqrt(vreg_pse1, vreg_pse1, preg_all);
            //             Sqrt(vreg_pse2, vreg_pse2, preg_all);
            //             Sqrt(vreg_pse3, vreg_pse3, preg_all);
            //             Sqrt(vreg_pse4, vreg_pse4, preg_all);
            //         }
            //         Muls(vreg_pse1, vreg_pse1, slopes, preg_all);
            //         Muls(vreg_pse2, vreg_pse2, slopes, preg_all);
            //         Muls(vreg_pse3, vreg_pse3, slopes, preg_all);
            //         Muls(vreg_pse4, vreg_pse4, slopes, preg_all);
            //         Adds(vreg_alibi1, vreg_alibi1, -1.0f, preg_all);
            //         Adds(vreg_alibi2, vreg_alibi2, -1.0f, preg_all);
            //         Adds(vreg_alibi3, vreg_alibi3, -1.0f, preg_all);
            //         Adds(vreg_alibi4, vreg_alibi4, -1.0f, preg_all);
            //     } else {
            //         if constexpr (IsSameType<pseShiftType, bfloat16_t>::value) {
            //             LoadAlign(vreg_pse_bf16_src1, pseUb + i * pseStride);
            //             LoadAlign(vreg_pse_bf16_src2, pseUb + floatRepSize * 2 + i * pseStride);
            //             Interleave(vreg_pse1_bf16, vreg_pse2_bf16, vreg_pse_bf16_src1, vreg_pse_bf16_src1);
            //             Interleave(vreg_pse3_bf16, vreg_pse4_bf16, vreg_pse_bf16_src2, vreg_pse_bf16_src2);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_bf16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_bf16, preg_all_b16);
            //         } else if constexpr (IsSameType<pseShiftType, half>::value) {
            //             LoadAlign(vreg_pse_f16_src1, pseUb + i * pseStride);
            //             LoadAlign(vreg_pse_f16_src2, pseUb + floatRepSize * 2 + i * pseStride);
            //             Interleave(vreg_pse1_f16, vreg_pse2_f16, vreg_pse_f16_src1, vreg_pse_f16_src1);
            //             Interleave(vreg_pse3_f16, vreg_pse4_f16, vreg_pse_f16_src2, vreg_pse_f16_src2);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse1, vreg_pse1_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse2, vreg_pse2_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse3, vreg_pse3_f16, preg_all_b16);
            //             Cast<T, pseShiftType, castTraitZero>(vreg_pse4, vreg_pse4_f16, preg_all_b16);
            //         }
            //     }
            //     Add(vreg_input_x1, vreg_input_x1, vreg_pse1, preg_all);
            //     Add(vreg_input_x2, vreg_input_x2, vreg_pse2, preg_all);
            //     Add(vreg_input_x3, vreg_input_x3, vreg_pse3, preg_ori_tail_n1);
            //     Add(vreg_input_x4, vreg_input_x4, vreg_pse4, preg_ori_tail_n2);
            // }
            // if constexpr (pseMode == PseTypeEnum::PSE_OUTER_ADD_MUL_TYPE) {
            //     Muls(vreg_input_x1, vreg_input_x1, scale, preg_all); // Muls(scale)
            //     Muls(vreg_input_x2, vreg_input_x2, scale, preg_all);
            //     Muls(vreg_input_x3, vreg_input_x3, scale, preg_ori_tail_n1);
            //     Muls(vreg_input_x4, vreg_input_x4, scale, preg_ori_tail_n2);
            // }

            // if constexpr (hasAtten == 1) {
                plds(preg_compare1, (__ubuf__ uint32_t *&)maskUb1, nPadding, dist_ds, POST_UPDATE);
                plds(preg_compare2, (__ubuf__ uint32_t *&)maskUb2, nPadding, dist_ds, POST_UPDATE);
                plds(preg_compare3, (__ubuf__ uint32_t *&)maskUb3, nPadding, dist_ds, POST_UPDATE);
                plds(preg_compare4, (__ubuf__ uint32_t *&)maskUb4, nPadding, dist_ds, POST_UPDATE);
                vsel(vreg_sel1, vreg_min, vreg_input_x1, preg_compare1);
                vsel(vreg_sel2, vreg_min, vreg_input_x2, preg_compare2);
                vsel(vreg_sel3, vreg_min, vreg_input_x3, preg_compare3);
                vsel(vreg_sel4, vreg_min, vreg_input_x4, preg_compare4);
                // vsel(vreg_sel3_new, vreg_sel3, vreg_min, preg_ori_tail_n1);
                // vsel(vreg_sel4_new, vreg_sel4, vreg_min, preg_ori_tail_n2);
                vsts(vreg_sel1, (__ubuf__ float *&)srcUb + i * 256, 0, dist_norm_b32, preg_all);
                vsts(vreg_sel2, (__ubuf__ float *&)srcUb + floatRepSize + i * 256, 0, dist_norm_b32, preg_all);
                vsts(vreg_sel3, (__ubuf__ float *&)srcUb + floatRepSize * 2 + i * 256, 0, dist_norm_b32, preg_all);
                vsts(vreg_sel4, (__ubuf__ float *&)srcUb + floatRepSize * 3 + i * 256, 0, dist_norm_b32, preg_all);
                vmax(vreg_max_tmp1, vreg_sel1, vreg_sel2, preg_all, MODE_ZEROING);
                vmax(vreg_max_tmp2, vreg_sel3, vreg_sel4, preg_all, MODE_ZEROING);
                vmax(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all, MODE_ZEROING);
                vcmax(vreg_input_max, vreg_max_tmp3, preg_all, MODE_ZEROING);
            // } else {
                // Select(vreg_input_x3_new, vreg_input_x3, vreg_min, preg_ori_tail_n1);
                // Select(vreg_input_x4_new, vreg_input_x4, vreg_min, preg_ori_tail_n2);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + i * 256, vreg_input_x1,
                //                                                   preg_all);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize + i * 256,
                //                                                   vreg_input_x2, preg_all);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 2 + i * 256,
                //                                                   vreg_input_x3_new, preg_all);
                // StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T *&)srcUb + floatRepSize * 3 + i * 256,
                //                                                   vreg_input_x4_new, preg_all);
                // Max(vreg_max_tmp1, vreg_input_x1, vreg_input_x2, preg_all);
                // Max(vreg_max_tmp2, vreg_input_x3_new, vreg_input_x4_new, preg_all);
                // Max(vreg_max_tmp3, vreg_max_tmp1, vreg_max_tmp2, preg_all);
                // Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(vreg_input_max,
                //                                                                                   vreg_max_tmp3, preg_all);
            // }
            // if constexpr (hasSink) {
            //     Max(vreg_input_max, vreg_input_max, vreg_sink_input, preg_all);
            // }
            vmuls(vreg_input_max, vreg_input_max, INV_LN2, preg_all, MODE_ZEROING);
            vtrc(vreg_input_max, vreg_input_max, ROUND_C, preg_all, MODE_ZEROING);
            vmuls(vreg_input_max, vreg_input_max, LN2, preg_all, MODE_ZEROING);
            vstus(ureg_max, 1, vreg_input_max, (__ubuf__ float *&)tmpMaxUb, POST_UPDATE);
        }
        vstas(ureg_max, (__ubuf__ float *&)tmpMaxUb, 0, POST_UPDATE);

        vlds(vreg_in_max, (__ubuf__ float*)inMaxUb, 0, dist_norm);
        mem_bar(VST_VLD);
        vlds(vreg_input_max, (__ubuf__ float*)tmpMaxUb2, 0, dist_norm);
        vmax(vreg_max_new, vreg_in_max, vreg_input_max, preg_all, MODE_ZEROING);
        vexpdif(vreg_subloop_update, vreg_in_max, vreg_max_new, preg_all, PART_EVEN);
        vlds(vreg_pre_loop_max, (__ubuf__ float*)preLoopMaxUb, 0, dist_norm);
        vexpdif(vreg_exp_max, vreg_pre_loop_max, vreg_max_new, preg_all, PART_EVEN);
        vsts(vreg_exp_max, (__ubuf__ float *&)expMaxUb, 0, dist_norm_b32, preg_all);
        vsts(vreg_max_new, (__ubuf__ float *&)maxUb, 0, dist_norm_b32, preg_all);

        // if constexpr (hasDrop == 1) {
        //     Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vreg_zero, 0.0f, preg_all);
        // }
        mem_bar(VST_VLD);

        for (uint16_t i = 0; i < m; ++i) {
            vlds(vreg_max, (__ubuf__ float*)maxUbStart + i, 0, dist_brc_b32);
            vsub(vreg_max, vreg_max, vreg_ln_p_scale, preg_all, MODE_ZEROING);
            vlds(vreg_input_x1, vreg_input_x2, (__ubuf__ float*)srcUb + i * 256, 0, dist_dintlv_b32);
            vlds(vreg_input_x3, vreg_input_x4,
                 (__ubuf__ float*)srcUb + floatRepSize * 2 + i * 256, 0, dist_dintlv_b32);
            vexpdif(vreg_exp_even1, vreg_input_x1, vreg_max, preg_all, PART_EVEN);
            vexpdif(vreg_exp_odd1, vreg_input_x2, vreg_max, preg_all, PART_EVEN);
            vexpdif(vreg_exp_even2, vreg_input_x3, vreg_max, preg_all, PART_EVEN);
            vexpdif(vreg_exp_odd2, vreg_input_x4, vreg_max, preg_all, PART_EVEN);

            // x_sum = sum(x_exp, axis=-1, keepdims=True)
            vadd(vreg_exp_sum1, vreg_exp_even1, vreg_exp_odd1, preg_all, MODE_ZEROING);
            vadd(vreg_exp_sum2, vreg_exp_even2, vreg_exp_odd2, preg_all, MODE_ZEROING);
            vadd(vreg_exp_sum3, vreg_exp_sum1, vreg_exp_sum2, preg_all, MODE_ZEROING);
            vcadd(vreg_exp_sum3, vreg_exp_sum3, preg_all, MODE_ZEROING);
            vstus(ureg_exp_sum, 1, vreg_exp_sum3, (__ubuf__ float *&)tmpExpSumUb, POST_UPDATE);

            // dropmask compute
            // if constexpr (hasDrop == 1) {
            //     LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
            //         preg1, (__ubuf__ uint32_t *&)dropMaskUb1, 256 >> 3);
            //     MaskInterleave<half>(preg3, preg4, preg1, preg2);
            //     MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            //     Select(vreg_sel_drop, vreg_exp_even1, vreg_zero, preg5);
            //     Muls(vreg_exp_even1, vreg_sel_drop, divValue, preg_all);
            //     Select(vreg_sel_drop2, vreg_exp_odd1, vreg_zero, preg6);
            //     Muls(vreg_exp_odd1, vreg_sel_drop2, divValue, preg_all);
            //     LoadAlign<uint32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::MaskDist::DIST_US>(
            //         preg1, (__ubuf__ uint32_t *&)dropMaskUb2, 256 >> 3);
            //     MaskInterleave<half>(preg3, preg4, preg1, preg2);
            //     MaskDeInterleave<T>(preg5, preg6, preg3, preg4);
            //     Select(vreg_sel_drop, vreg_exp_even2, vreg_zero, preg5);
            //     Muls(vreg_exp_even2, vreg_sel_drop, divValue, preg_all);
            //     Select(vreg_sel_drop2, vreg_exp_odd2, vreg_zero, preg6);
            //     Muls(vreg_exp_odd2, vreg_sel_drop2, divValue, preg_all);
            // }

            // if constexpr (IsSameType<T2, bfloat16_t>::value) {
            //     Cast<T2, T, castTraitZero>(vreg_exp_even1_bf16, vreg_exp_even1, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd1_bf16, vreg_exp_odd1, preg_all);
            //     Cast<T2, T, castTraitZero>(vreg_exp_even2_bf16, vreg_exp_even2, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd2_bf16, vreg_exp_odd2, preg_all);
            //     Or((RegTensor<uint16_t> &)vreg_exp1_bf16, (RegTensor<uint16_t> &)vreg_exp_even1_bf16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd1_bf16, preg_all_b16);
            //     Or((RegTensor<uint16_t> &)vreg_exp2_bf16, (RegTensor<uint16_t> &)vreg_exp_even2_bf16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd2_bf16, preg_all_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb1), vreg_exp1_bf16, blockStride, repeatStride, preg_n_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb2), vreg_exp2_bf16, blockStride, repeatStride, preg_n_b16);
            // } else if constexpr (IsSameType<T2, fp8_e4m3fn_t>::value) {
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_even_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_odd_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_tmp_f8e4m3_2;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_1;
                RegTensor<fp8_e4m3fn_t> vreg_exp_merge_f8e4m3_2;
                // RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes;
                // // RegTensor<uint8_t> vreg_exp_merge_f8e4m3_indexes_2;
                // MaskReg preg_all_b8 = pset_b8(PAT_ALL);
                // uint32_t maskLen = 128;
                // MaskReg preg_all_b8_128 = plt_b8(maskLen, POST_UPDATE);
                vcvt(vreg_exp_even_f8e4m3_1, vreg_exp_even1, preg_all, ROUND_R, RS_DISABLE, PART_P0, MODE_ZEROING);
                vcvt(vreg_exp_even_f8e4m3_2, vreg_exp_even2, preg_all, ROUND_R, RS_DISABLE, PART_P0, MODE_ZEROING);
                vcvt(vreg_exp_odd_f8e4m3_1, vreg_exp_odd1, preg_all, ROUND_R, RS_DISABLE, PART_P2, MODE_ZEROING);
                vcvt(vreg_exp_odd_f8e4m3_2, vreg_exp_odd2, preg_all, ROUND_R, RS_DISABLE, PART_P2, MODE_ZEROING);
                vor((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_1, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3_1,
                   (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3_1, preg_all_b8, MODE_ZEROING);
                vor((RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_2, (RegTensor<uint8_t> &)vreg_exp_even_f8e4m3_2,
                   (RegTensor<uint8_t> &)vreg_exp_odd_f8e4m3_2, preg_all_b8, MODE_ZEROING);
                vlds(vreg_exp_merge_f8e4m3_indexes, (__ubuf__ uint8_t*)indexesUb, 0, dist_norm);
                vselr((RegTensor<uint8_t> &)vreg_exp_merge_f8e4m3_1, (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_1, (RegTensor<uint8_t> &)vreg_exp_merge_f8e4m3_indexes);
                vselr((RegTensor<uint8_t> &)vreg_exp_merge_f8e4m3_2, (RegTensor<uint8_t> &)vreg_exp_merge_tmp_f8e4m3_2, (RegTensor<uint8_t> &)vreg_exp_merge_f8e4m3_indexes);
                vsstb((RegTensor<uint8_t> &)vreg_exp_merge_f8e4m3_1, (__ubuf__ uint8_t *&)expUb1,
                      (blockStride << 16u) | (repeatStride & 0xFFFFU), preg_all_b8_128, POST_UPDATE);
                vsstb((RegTensor<uint8_t> &)vreg_exp_merge_f8e4m3_2, (__ubuf__ uint8_t *&)expUb2,
                      (blockStride << 16u) | (repeatStride & 0xFFFFU), preg_all_b8_128, POST_UPDATE);
            // } else if constexpr (IsSameType<T2, half>::value) {
            //     Cast<T2, T, castTraitZero>(vreg_exp_even1_f16, vreg_exp_even1, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd1_f16, vreg_exp_odd1, preg_all);
            //     Cast<T2, T, castTraitZero>(vreg_exp_even2_f16, vreg_exp_even2, preg_all);
            //     Cast<T2, T, castTraitOne>(vreg_exp_odd2_f16, vreg_exp_odd2, preg_all);
            //     Or((RegTensor<uint16_t> &)vreg_exp1_f16, (RegTensor<uint16_t> &)vreg_exp_even1_f16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd1_f16, preg_all_b16);
            //     Or((RegTensor<uint16_t> &)vreg_exp2_f16, (RegTensor<uint16_t> &)vreg_exp_even2_f16,
            //        (RegTensor<uint16_t> &)vreg_exp_odd2_f16, preg_all_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb1), vreg_exp1_f16, blockStride, repeatStride, preg_n_b16);
            //     StoreAlign<T2, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            //         ((__ubuf__ T2 *&)expUb2), vreg_exp2_f16, blockStride, repeatStride, preg_n_b16);
            // }
        }
        vstas(ureg_exp_sum, (__ubuf__ float *&)tmpExpSumUb, 0, POST_UPDATE);

        mem_bar(VST_VLD);
        vlds(vreg_exp_sum_brc, (__ubuf__ float*)tmpExpSumUb2, 0, dist_norm);
        RegTensor<float> vreg_first_sum;
        vlds(vreg_first_sum, (__ubuf__ float*)firstLoopSumUb, 0, dist_norm);
        vlds(vreg_pre_sum, (__ubuf__ float*)preLoopSumUb, 0, dist_norm);
        vmul(vreg_first_sum, vreg_subloop_update, vreg_first_sum, preg_all, MODE_ZEROING);
        vadd(vreg_first_sum, vreg_first_sum, vreg_exp_sum_brc, preg_all, MODE_ZEROING);
        vmul(vreg_pre_sum, vreg_exp_max, vreg_pre_sum, preg_all, MODE_ZEROING);
        vadd(vreg_pre_sum, vreg_first_sum, vreg_pre_sum, preg_all, MODE_ZEROING);
        vsts(vreg_pre_sum, (__ubuf__ float *&)expSumUb, 0, dist_norm_b32, preg_all);
        // pScale
 	    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_dst0;
 	    RegTensor<fp8_e8m0_t> vreg_p_scale_f8e8m0_dst1;
 	    vbr(vreg_p_scale, static_cast<float>(1.0f));
        vmul(vreg_p_scale, vreg_subloop_update, vreg_p_scale, preg_all, MODE_ZEROING);
        vcvt(vreg_p_scale_bf16_0, vreg_p_scale, preg_all, ROUND_R, RS_DISABLE, PART_EVEN, MODE_ZEROING);
 	    vshls((RegTensor<uint16_t> &)vreg_p_scale_bf16_0, (RegTensor<uint16_t> &)vreg_p_scale_bf16_0, SHIFT_ONE_BIT, preg_all_b16, MODE_ZEROING);
        vshrs((RegTensor<uint16_t> &)vreg_p_scale_bf16_0, (RegTensor<uint16_t> &)vreg_p_scale_bf16_0, SHIFT_EIGHT_BIT, preg_all_b16, MODE_ZEROING);
        vcvt((RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_0, (RegTensor<uint16_t> &)vreg_p_scale_bf16_0, preg_all_b16, RS_DISABLE, PART_EVEN,
            MODE_ZEROING);
 	    vcvt(vreg_p_scale_bf16_1, vreg_p_scale, preg_all, ROUND_R, RS_DISABLE, PART_ODD, MODE_ZEROING);
        vshls((RegTensor<uint16_t> &)vreg_p_scale_bf16_1, (RegTensor<uint16_t> &)vreg_p_scale_bf16_1, SHIFT_ONE_BIT, preg_all_b16, MODE_ZEROING);
        vshrs((RegTensor<uint16_t> &)vreg_p_scale_bf16_1, (RegTensor<uint16_t> &)vreg_p_scale_bf16_1, SHIFT_EIGHT_BIT, preg_all_b16, MODE_ZEROING);
        vcvt((RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_1, (RegTensor<uint16_t> &)vreg_p_scale_bf16_1, preg_all_b16, RS_DISABLE, PART_EVEN,
            MODE_ZEROING);
 	    vor((RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_0, (RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_0,
                   (RegTensor<uint8_t> &)vreg_p_scale_f8e8m0_1, preg_all_b8, MODE_ZEROING);
        const int8_t val_value = 0x7f;
        vdup((RegTensor<int8_t>&)vreg_p_scale_f8e8m0_1, (int8_t&)val_value, preg_all_b8, MODE_ZEROING);
 	    vdintlv((RegTensor<int8_t>&)vreg_p_scale_f8e8m0_dst0, (RegTensor<int8_t>&)vreg_p_scale_f8e8m0_dst1, (RegTensor<int8_t>&)vreg_p_scale_f8e8m0_0, (RegTensor<int8_t>&)vreg_p_scale_f8e8m0_1);
        vsts((RegTensor<uint8_t>&)vreg_p_scale_f8e8m0_dst0, (__ubuf__ uint8_t*)pScaleSubLoop0, 0, dist_norm_b8, preg_all_b8);
    }

template <typename T, typename INPUT_T, typename OUTPUT_T, uint32_t srcD, bool isMlaFullQuant>
__simd_vf__ inline void LastDivNewVF_MX(__ubuf__ float * dstUb, __ubuf__ float * curUb, __ubuf__ float * expSumUb,
    const uint16_t m, const uint16_t d, const float deScaleV)
{
    RegTensor<float> vreg_input_cur;
    RegTensor<float> vreg_div;
    RegTensor<float> vreg_exp_sum;
    MaskReg preg_all = pset_b32(PAT_ALL);
    constexpr uint16_t floatRepSize = 64;
    const uint16_t dLoops = d >> 6;
    constexpr float fp8e4m3MaxValueRec = 1 / 448.0f;
    constexpr float int8MaxValueRec = 1 / 127.0f;
    constexpr float hifp8MaxValueRec = 1 / 32768.0f;

    constexpr auto dist_brc_b32 = std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_BRC_B32)>();
    constexpr auto dist_norm = std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_NORM)>();
    constexpr auto dist_norm_b32 = std::integral_constant<::DistVST, static_cast<::DistVST>(StoreDist::DIST_NORM_B32)>();

    for (uint16_t i = 0; i < m; ++i) {
        uint32_t sreg_init = d;
        vlds(vreg_exp_sum, (__ubuf__ float*)expSumUb + i * REDUCE_SIZE, 0, dist_brc_b32);
        for (uint16_t j = 0; j < dLoops; ++j) {
            MaskReg preg_update = plt_b32(sreg_init, POST_UPDATE);

            vlds(vreg_input_cur, (__ubuf__ float*)curUb + i * d + j * floatRepSize, 0, dist_norm);
            vmuls(vreg_input_cur, vreg_input_cur, deScaleV, preg_all, MODE_ZEROING);
            vdiv(vreg_div, vreg_input_cur, vreg_exp_sum, preg_update, MODE_ZEROING);
            vsts(vreg_div, (__ubuf__ float *&)dstUb + i * d + j * floatRepSize, 0, dist_norm_b32, preg_update);
        }
    }
}

    __simd_vf__ void MaskUbCopyS1GMX(__ubuf__ uint8_t * maskUb, int32_t headGLoop, int32_t midGLoop,
                                        int32_t tailGLoop, int32_t midS1Count)
    {
        if (midGLoop <= 1) {
            return;
        }
        constexpr uint32_t repeatStride = 8;//256 >> 5

        RegTensor<uint8_t> vreg_g_head;
        RegTensor<uint8_t> vreg_g_mid;
        RegTensor<uint8_t> vreg_g_tail;

        MaskReg preg_all;
        preg_all = CreateMask<bool, MaskPattern::ALL>();

        for (uint16_t x = headGLoop; x > 0; x = 0) {     // if (headGLoop > 0) {}
            constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(MicroAPI::PostLiteral::POST_MODE_UPDATE)>();
            vsldb((RegTensor<uint8_t>&)vreg_g_head, (__ubuf__ uint8_t*&)maskUb,
                (1 << 16u) | (repeatStride & 0xFFFFU), preg_all, postValue);
            for (uint16_t i = 1; i < headGLoop; ++i) {
                constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(MicroAPI::PostLiteral::POST_MODE_UPDATE)>();
                vsstb((RegTensor<uint8_t>&)vreg_g_head, (__ubuf__ uint8_t*&)maskUb,
                  (1 << 16u) | (repeatStride & 0xFFFFU), preg_all, postValue);
            }
        }

        for (uint16_t x = midGLoop; x > 0; x = 0) {//2     // if (midGLoop > 0) {}
            for (uint16_t i = 0; i < midS1Count; ++i) {//8
                constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(MicroAPI::PostLiteral::POST_MODE_UPDATE)>();
                vsldb((RegTensor<uint8_t>&)vreg_g_mid, (__ubuf__ uint8_t*&)maskUb,
                    (1 << 16u) | (repeatStride & 0xFFFFU), preg_all, postValue);
                for (uint16_t j = 1; j < midGLoop; ++j) {//2
                    constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(MicroAPI::PostLiteral::POST_MODE_UPDATE)>();
                    vsstb((RegTensor<uint8_t>&)vreg_g_mid, (__ubuf__ uint8_t*&)maskUb,
                      (1 << 16u) | (repeatStride & 0xFFFFU), preg_all, postValue);
                    }
                mem_bar(VST_VLD);
            }
        }

        for (uint16_t x = tailGLoop; x > 1; x = 0) {     // if (tailGLoop > 1) {}
            constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(MicroAPI::PostLiteral::POST_MODE_UPDATE)>();
            vsldb((RegTensor<uint8_t>&)vreg_g_tail, (__ubuf__ uint8_t*&)maskUb,
                (1 << 16u) | (repeatStride & 0xFFFFU), preg_all, postValue);
            for (uint16_t i = 1; i < tailGLoop; ++i) {
                constexpr auto postValue = std::integral_constant<::Post, static_cast<::Post>(MicroAPI::PostLiteral::POST_MODE_UPDATE)>();
                    vsstb((RegTensor<uint8_t>&)vreg_g_tail, (__ubuf__ uint8_t*&)maskUb,
                      (1 << 16u) | (repeatStride & 0xFFFFU), preg_all, postValue);
            }
        }
    }

template <typename T>
__simd_vf__ inline void RowInvalidUpdateVF_MX(__ubuf__ T *finalUb, __ubuf__ float *maxUb, const uint16_t m,
    const uint16_t d, int64_t dSize, const uint32_t pltTailD, const uint16_t hasTail, const uint32_t min = 0xFF7FFFFF)
{
    constexpr uint16_t floatRepSize = 64; // 64: 一个寄存器可以存储64个float类型数据
    const uint16_t dLoops = d / floatRepSize;


    constexpr uint32_t tmpZero = 0x00000000; // zero value of fp16 and fp32
    const T zeroValue = *((T*)&tmpZero);
    const float minValue = *((float*)&min);
    MicroAPI::RegTensor<float> vregMinValue;
    MicroAPI::RegTensor<T> vregZeroValue;
    MicroAPI::RegTensor<float> vregMax;
    MicroAPI::RegTensor<T> vregFinal;
    MicroAPI::RegTensor<T> vregFinalNew;

    MicroAPI::MaskReg pregAll = pset_b32(PAT_ALL);
    uint32_t tmpTailD = pltTailD;
    MicroAPI::MaskReg pregTailD = plt_b32(tmpTailD, POST_UPDATE);
    MicroAPI::MaskReg pregCompare;

    constexpr auto dist_brc_b32 = std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_BRC_B32)>();
    constexpr auto dist_norm = std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_NORM)>();
    constexpr auto dist_norm_b32 = std::integral_constant<::DistVST, static_cast<::DistVST>(StoreDist::DIST_NORM_B32)>();

    vbr(vregMinValue, minValue);
    vbr(vregZeroValue, zeroValue);
    for (uint16_t i = 0; i < m; ++i) {
        vlds(vregMax, (__ubuf__ float*)maxUb + i, 0, dist_brc_b32);
        vcmp_eq(pregCompare, vregMax, vregMinValue, pregAll);
        for (uint16_t j = 0; j < dLoops; ++j) {
            vlds(vregFinal, (__ubuf__ float*)finalUb + i * dSize + j * floatRepSize, 0, dist_norm);
            vsel(vregFinalNew, vregZeroValue, vregFinal, pregCompare);
            vsts(vregFinalNew, (__ubuf__ float *&)finalUb + i * dSize + j * floatRepSize, 0, dist_norm_b32, pregAll);
        }
        for (uint16_t t = 0; t < hasTail; ++t) {
            vlds(vregFinal, (__ubuf__ float*)finalUb + i * dSize + dLoops * floatRepSize, 0, dist_norm);
            vsel(vregFinalNew, vregZeroValue, vregFinal, pregCompare);
            vsts(vregFinalNew, (__ubuf__ float *&)finalUb + i * dSize + dLoops * floatRepSize, 0, dist_norm_b32, pregTailD);
        }
    }
}

template <typename T>
__aicore__ inline void RowInvalidUpdateVF_MX(__ubuf__ T *finalUb, __ubuf__ float *maxUb, 
    const uint16_t m, const uint16_t d, int64_t dSize, const uint32_t min = 0xFF7FFFFF)
{
    // __ubuf__ T * finalUb = (__ubuf__ T*)finalTensor.GetPhyAddr();
    // __ubuf__ float * maxUb = (__ubuf__ float*)maxTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64;
    const uint16_t tailD = d % floatRepSize;
    uint32_t pltTailD = static_cast<uint32_t>(tailD);
    uint16_t hasTail = 0;
    if (tailD > 0) {
        hasTail = 1;
    }

    RowInvalidUpdateVF<T>(finalUb, maxUb, m, d, dSize, pltTailD, hasTail, min);
}

template <typename DST_TYPE, typename SRC_TYPE, RoundMode roundMode>
__simd_vf__ inline void CastIntrinsicsImplVF_MX(__ubuf__ DST_TYPE *dst, __ubuf__ SRC_TYPE *src, const uint32_t calCount)
{
    uint16_t oneRepSize = GetVecLen() / sizeof(SRC_TYPE);
    if constexpr (sizeof(SRC_TYPE) < sizeof(DST_TYPE)) {
        oneRepSize = GetVecLen() / sizeof(DST_TYPE);
    }
    uint16_t repeatTime = (calCount + oneRepSize - 1) / oneRepSize;
    uint32_t sreg = static_cast<uint32_t>(calCount);
    static constexpr Reg::CastTrait castTrait = {
        Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, roundMode};
    Reg::MaskReg preg;
    Reg::RegTensor<SRC_TYPE> srcVreg;
    Reg::RegTensor<DST_TYPE> dstVreg;
    Reg::RegTensor<float> tmpVreg;
    for (uint16_t i = 0; i < repeatTime; ++i) {//有问题
        preg = plt_b32(sreg, POST_UPDATE);
        constexpr auto distValue = std::integral_constant<::Dist, static_cast<::Dist>(LoadDist::DIST_NORM)>();
        vlds(srcVreg, src + i * oneRepSize, 0, distValue);
        constexpr auto partModeValue = std::integral_constant<::Part, static_cast<::Part>(Reg::RegLayout::ZERO)>();
        constexpr auto modeValue = std::integral_constant<::Mode, static_cast<::Mode>(Reg::MaskMergeMode::ZEROING)>();
        constexpr auto roundModeValue = std::integral_constant<::ROUND, GetRound<roundMode>()>();
        constexpr auto satModeValue = std::integral_constant<::RoundingSaturation, static_cast<::RoundingSaturation>(Reg::SatMode::SAT)>();
        vcvt(dstVreg, srcVreg, preg, roundModeValue, satModeValue, partModeValue, modeValue);
        constexpr auto distValue2 = std::integral_constant<::DistVST, static_cast<::DistVST>(GetStoreDist<DST_TYPE, Reg::StoreDist::DIST_PACK_B32>())>();
        vsts(dstVreg, dst + i * oneRepSize, 0, distValue2, preg);
    }
}
}
