template <typename T>
__simd_vf__ inline void UpdateExpSumAndExpMaxImplVF(__ubuf__ T * maxUb, __ubuf__ T * inMaxUb, __ubuf__ T * expMaxUb,
    __ubuf__ T * expSumUb, __ubuf__ T * inExpSumUb, __ubuf__ T * tmpExpSumUb, __ubuf__ T * tmpMaxUb, const uint32_t m)
{
    RegTensor<float> vreg_input_x;
    RegTensor<float> vreg_input_x_unroll;
    RegTensor<float> vreg_max;
    RegTensor<float> vreg_in_max;
    RegTensor<float> vreg_exp_sum;
    RegTensor<float> vreg_in_exp_sum;
    RegTensor<float> vreg_exp_max;
    RegTensor<float> vreg_exp_sum_brc;
    RegTensor<float> vreg_exp_sum_update;
    MaskReg preg_all = CreateMask<float, MaskPattern::ALL>();
 
    // 注意：当m大于64的时候需要开启循环
    LoadAlign(vreg_max, tmpMaxUb);
    LoadAlign(vreg_in_max, inMaxUb);
    FusedExpSub(vreg_exp_max, vreg_in_max, vreg_max, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)expMaxUb, vreg_exp_max, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)maxUb, vreg_max, preg_all);
    LoadAlign(vreg_in_exp_sum, inExpSumUb);
 
    // x_sum = exp_max * insum + x_sum
    LoadAlign(vreg_exp_sum_brc, tmpExpSumUb);
    Mul(vreg_exp_sum_update, vreg_exp_max, vreg_in_exp_sum, preg_all);
    Add(vreg_exp_sum_update, vreg_exp_sum_update, vreg_exp_sum_brc, preg_all);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
        (__ubuf__ T *&)expSumUb, vreg_exp_sum_update, preg_all);
 
 
}