#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR SparseIndices_handle,  GM_ADDR Query_handle,  GM_ADDR QueryRope_handle,  GM_ADDR Key_handle,  GM_ADDR KeyRope_handle,  GM_ADDR QueryIndex_handle,  GM_ADDR KeyIndex_handle,  GM_ADDR Weights_handle,  GM_ADDR SoftmaxMax_handle,  GM_ADDR SoftmaxSum_handle,  GM_ADDR dQueryIndexAccum_handle,  GM_ADDR GatherKey_ws_handle,  GM_ADDR GatherKeyRope_ws_handle,  GM_ADDR GatherKeyIndex_ws_handle,  GM_ADDR bmm1Res_handle,  GM_ADDR bmm2Res_handle,  GM_ADDR PReduced_handle,  GM_ADDR ReduceSumY_handle,  GM_ADDR SoftmaxI_handle,  GM_ADDR dk_matmul_partial_handle,  GM_ADDR dQueryIndex_handle,  GM_ADDR dWeight_partial_handle,  GM_ADDR reluGradRes_handle,  GM_ADDR LossOut_handle,  GM_ADDR dKeyIndex_handle, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<int> SparseIndices;
  SparseIndices.SetGlobalBuffer((__gm__ int*)SparseIndices_handle);
  AscendC::GlobalTensor<half> Query;
  Query.SetGlobalBuffer((__gm__ half*)Query_handle);
  AscendC::GlobalTensor<half> QueryRope;
  QueryRope.SetGlobalBuffer((__gm__ half*)QueryRope_handle);
  AscendC::GlobalTensor<half> Key;
  Key.SetGlobalBuffer((__gm__ half*)Key_handle);
  AscendC::GlobalTensor<half> KeyRope;
  KeyRope.SetGlobalBuffer((__gm__ half*)KeyRope_handle);
  AscendC::GlobalTensor<half> QueryIndex;
  QueryIndex.SetGlobalBuffer((__gm__ half*)QueryIndex_handle);
  AscendC::GlobalTensor<half> KeyIndex;
  KeyIndex.SetGlobalBuffer((__gm__ half*)KeyIndex_handle);
  AscendC::GlobalTensor<half> Weights;
  Weights.SetGlobalBuffer((__gm__ half*)Weights_handle);
  AscendC::GlobalTensor<float> SoftmaxMax;
  SoftmaxMax.SetGlobalBuffer((__gm__ float*)SoftmaxMax_handle);
  AscendC::GlobalTensor<float> SoftmaxSum;
  SoftmaxSum.SetGlobalBuffer((__gm__ float*)SoftmaxSum_handle);
  AscendC::GlobalTensor<float> dQueryIndexAccum;
  dQueryIndexAccum.SetGlobalBuffer((__gm__ float*)dQueryIndexAccum_handle);
  AscendC::GlobalTensor<half> GatherKey_ws;
  GatherKey_ws.SetGlobalBuffer((__gm__ half*)GatherKey_ws_handle);
  AscendC::GlobalTensor<half> GatherKeyRope_ws;
  GatherKeyRope_ws.SetGlobalBuffer((__gm__ half*)GatherKeyRope_ws_handle);
  AscendC::GlobalTensor<half> GatherKeyIndex_ws;
  GatherKeyIndex_ws.SetGlobalBuffer((__gm__ half*)GatherKeyIndex_ws_handle);
  AscendC::GlobalTensor<float> bmm1Res;
  bmm1Res.SetGlobalBuffer((__gm__ float*)bmm1Res_handle);
  AscendC::GlobalTensor<float> bmm2Res;
  bmm2Res.SetGlobalBuffer((__gm__ float*)bmm2Res_handle);
  AscendC::GlobalTensor<float> PReduced;
  PReduced.SetGlobalBuffer((__gm__ float*)PReduced_handle);
  AscendC::GlobalTensor<float> ReduceSumY;
  ReduceSumY.SetGlobalBuffer((__gm__ float*)ReduceSumY_handle);
  AscendC::GlobalTensor<float> SoftmaxI;
  SoftmaxI.SetGlobalBuffer((__gm__ float*)SoftmaxI_handle);
  AscendC::GlobalTensor<float> dk_matmul_partial;
  dk_matmul_partial.SetGlobalBuffer((__gm__ float*)dk_matmul_partial_handle);
  AscendC::GlobalTensor<half> dQueryIndex;
  dQueryIndex.SetGlobalBuffer((__gm__ half*)dQueryIndex_handle);
  AscendC::GlobalTensor<half> dWeight_partial;
  dWeight_partial.SetGlobalBuffer((__gm__ half*)dWeight_partial_handle);
  AscendC::GlobalTensor<half> reluGradRes;
  reluGradRes.SetGlobalBuffer((__gm__ half*)reluGradRes_handle);
  AscendC::GlobalTensor<float> LossOut;
  LossOut.SetGlobalBuffer((__gm__ float*)LossOut_handle);
  AscendC::GlobalTensor<float> dKeyIndex;
  dKeyIndex.SetGlobalBuffer((__gm__ float*)dKeyIndex_handle);

  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 65536);
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1; pipe.InitBuffer(ascend_l1, 524032);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c; pipe.InitBuffer(ascend_l0c, 131072);
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  pipe.Destroy();
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    cid = cid / 2;
  }
  auto key_chunk_ub = ascend_ub.GetWithOffset<half>(16384, 0);
  auto ki_chunk_ub = ascend_ub.GetWithOffset<half>(4096, 32768);
  auto kr_chunk_ub = ascend_ub.GetWithOffset<half>(4096, 40960);
  auto C_L0C = ascend_l0c.GetWithOffset<float>(8192, 0);
  auto Q_L1 = ascend_l1.GetWithOffset<half>(16384, 0);
  auto GK_L1 = ascend_l1.GetWithOffset<half>(8192, 32768);
  auto QI_L1 = ascend_l1.GetWithOffset<half>(8192, 49152);
  auto GKI_L1 = ascend_l1.GetWithOffset<half>(8192, 65536);
  auto C2_L0C = ascend_l0c.GetWithOffset<float>(4096, 32768);
  auto sm_max_ub = ascend_ub.GetWithOffset<float>(128, 49152);
  auto sm_sum_ub = ascend_ub.GetWithOffset<float>(128, 49664);
  auto eps_sm_ub = ascend_ub.GetWithOffset<float>(128, 50176);
  auto bmm1_ub = ascend_ub.GetWithOffset<float>(8192, 50688);
  auto p_reduced_tmp_ub = ascend_ub.GetWithOffset<float>(64, 83456);
  auto re_tmp_ub = ascend_ub.GetWithOffset<float>(1024, 83712);
  auto tmp_ub = ascend_ub.GetWithOffset<uint8_t>(32768, 87808);
  auto weight_all_half_ub = ascend_ub.GetWithOffset<half>(64, 120576);
  auto weight_all_ub = ascend_ub.GetWithOffset<float>(64, 120704);
  auto weight_2d_ub = ascend_ub.GetWithOffset<float>(4096, 120960);
  auto rsy_max_ub = ascend_ub.GetWithOffset<float>(1, 137344);
  auto rsy_all_ub = ascend_ub.GetWithOffset<float>(2048, 154016);
  auto rsy_bc_ub = ascend_ub.GetWithOffset<float>(64, 162208);
  auto bmm2_ub = ascend_ub.GetWithOffset<float>(4096, 137376);
  auto rsy_reduce_ub = ascend_ub.GetWithOffset<float>(64, 153760);
  auto rsy_chunk_max_ub = ascend_ub.GetWithOffset<float>(1, 162464);
  auto softmax_i_chunk_ub = ascend_ub.GetWithOffset<float>(64, 162496);
  auto rsy_sum_ub = ascend_ub.GetWithOffset<float>(1, 162752);
  auto rsy_chunk_ub = ascend_ub.GetWithOffset<float>(64, 162784);
  auto rsy_chunk_sum_ub = ascend_ub.GetWithOffset<float>(1, 163040);
  auto eps_sum_ub = ascend_ub.GetWithOffset<float>(1, 163072);
  auto ones_2d_ub = ascend_ub.GetWithOffset<float>(2048, 163104);
  auto eps_ub = ascend_ub.GetWithOffset<float>(64, 171296);
  auto diff_ub = ascend_ub.GetWithOffset<float>(64, 171552);
  auto weight_ub = ascend_ub.GetWithOffset<float>(32, 171808);
  auto dw_chunk_ub = ascend_ub.GetWithOffset<float>(2048, 171936);
  auto rg_chunk_ub = ascend_ub.GetWithOffset<float>(2048, 0);
  auto rsy_tmp_ub = ascend_ub.GetWithOffset<float>(2048, 180128);
  auto dw_reduce_ub = ascend_ub.GetWithOffset<float>(32, 188320);
  auto rg_half_ub = ascend_ub.GetWithOffset<half>(2048, 188448);
  auto dw_half_ub = ascend_ub.GetWithOffset<half>(32, 192544);
  auto loss_chunk_ub = ascend_ub.GetWithOffset<float>(64, 192608);
  auto loss_log_p_ub = ascend_ub.GetWithOffset<float>(64, 192864);
  auto loss_log_si_ub = ascend_ub.GetWithOffset<float>(64, 193120);
  auto loss_val_ub = ascend_ub.GetWithOffset<float>(64, 193376);
  auto loss_sum_ub = ascend_ub.GetWithOffset<float>(1, 193632);
  auto RG_L1 = ascend_l1.GetWithOffset<half>(4096, 81920);
  auto GKI2_L1 = ascend_l1.GetWithOffset<half>(8192, 90112);
  auto DQ_L0C = ascend_l0c.GetWithOffset<float>(8192, 49152);
  auto QI_L1_dk = ascend_l1.GetWithOffset<half>(8192, 106496);
  auto DK_L0C = ascend_l0c.GetWithOffset<float>(8192, 81920);
  auto dq_fp32_ub = ascend_ub.GetWithOffset<float>(512, 193664);
  auto dq_fp16_ub = ascend_ub.GetWithOffset<half>(512, 0);
  auto dk_chunk_ub = ascend_ub.GetWithOffset<float>(128, 195712);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    for (int32_t i = 0; i < 16; ++i) {
      int32_t idx = SparseIndices.GetValue(((cid * 64) + i));
      if (0 <= idx) {
        tl::ascend::copy_gm_to_ub<half, 512, 1>(key_chunk_ub[(i * 512)], Key[(idx * 512)], 1048576, 1, 512, half(0.000000e+00f));
        tl::ascend::copy_gm_to_ub<half, 128, 1>(ki_chunk_ub[(i * 128)], KeyIndex[(idx * 128)], 262144, 1, 128, half(0.000000e+00f));
        tl::ascend::copy_gm_to_ub<half, 128, 1>(kr_chunk_ub[(i * 128)], KeyRope[(idx * 128)], 262144, 1, 128, half(0.000000e+00f));
      }
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
    for (int32_t chunk_id = 0; chunk_id < 4; ++chunk_id) {
      if (chunk_id < 3) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(((chunk_id + 1) % 2));
        for (int32_t i_1 = 0; i_1 < 16; ++i_1) {
          int32_t idx_1 = SparseIndices.GetValue(((((cid * 64) + (chunk_id * 16)) + i_1) + 16));
          if (0 <= idx_1) {
            tl::ascend::copy_gm_to_ub<half, 512, 1>(key_chunk_ub[((((chunk_id + 1) % 2) * 8192) + (i_1 * 512))], Key[(idx_1 * 512)], 1048576, 1, 512, half(0.000000e+00f));
            tl::ascend::copy_gm_to_ub<half, 128, 1>(ki_chunk_ub[((((chunk_id + 1) % 2) * 2048) + (i_1 * 128))], KeyIndex[(idx_1 * 128)], 262144, 1, 128, half(0.000000e+00f));
            tl::ascend::copy_gm_to_ub<half, 128, 1>(kr_chunk_ub[((((chunk_id + 1) % 2) * 2048) + (i_1 * 128))], KeyRope[(idx_1 * 128)], 262144, 1, 128, half(0.000000e+00f));
          }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(((chunk_id + 1) % 2));
      }
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>((chunk_id % 2));
      tl::ascend::copy_ub_to_gm<half, 512, 16>(GatherKey_ws[((cid * 32768) + (chunk_id * 8192))], key_chunk_ub[((chunk_id % 2) * 8192)], 512, 16, 512);
      tl::ascend::copy_ub_to_gm<half, 128, 16>(GatherKeyIndex_ws[((cid * 8192) + (chunk_id * 2048))], ki_chunk_ub[((chunk_id % 2) * 2048)], 128, 16, 128);
      tl::ascend::copy_ub_to_gm<half, 128, 16>(GatherKeyRope_ws[((cid * 8192) + (chunk_id * 2048))], kr_chunk_ub[((chunk_id % 2) * 2048)], 128, 16, 128);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((chunk_id % 2));
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    AscendC::CrossCoreSetFlag<0x2, PIPE_V>(10);
  }
  if ASCEND_IS_AIC {
    AscendC::CrossCoreWaitFlag(10);
    for (int32_t d_idx = 0; d_idx < 5; ++d_idx) {
      if (d_idx < 4) {
        tl::ascend::copy_gm_to_l1<half, 128, 128>(Q_L1[0], Query[(d_idx * 128)], 512, 128, 128, true);
      } else {
        tl::ascend::copy_gm_to_l1<half, 128, 128>(Q_L1[0], QueryRope[((d_idx * 128) - 512)], 128, 128, 128, true);
      }
      if (d_idx < 4) {
        tl::ascend::copy_gm_to_l1<half, 64, 128>(GK_L1[0], GatherKey_ws[((cid * 32768) + (d_idx * 128))], 512, 64, 128, true);
      } else {
        tl::ascend::copy_gm_to_l1<half, 64, 128>(GK_L1[0], GatherKeyRope_ws[(((cid * 8192) + (d_idx * 128)) - 512)], 128, 64, 128, true);
      }
      AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(0);
      tl::ascend::gemm_v0<half, float, 128, 64, 128, false, true>(Q_L1[0], GK_L1[0], C_L0C[0], ascend_l0a, ascend_l0b, (d_idx == 0));
      AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(0);
      AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(0);
    }
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(1);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(1);
    tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 64, 0>(bmm1Res[(cid * 64)], C_L0C[0], 2048, 128, 64);
    tl::ascend::copy_gm_to_l1<half, 64, 128>(QI_L1[0], QueryIndex[0], 128, 64, 128, true);
    tl::ascend::copy_gm_to_l1<half, 64, 128>(GKI_L1[0], GatherKeyIndex_ws[(cid * 8192)], 128, 64, 128, true);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(0);
    tl::ascend::gemm_v0<half, float, 64, 64, 128, false, true>(QI_L1[0], GKI_L1[0], C2_L0C[0], ascend_l0a, ascend_l0b, (bool)1);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(1);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(1);
    tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 64, 1>(bmm2Res[(cid * 64)], C2_L0C[0], 2048, 64, 64);
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(1);
  }
  if ASCEND_IS_AIV {
    AscendC::CrossCoreWaitFlag(1);
    tl::ascend::copy_gm_to_ub<float, 128>(sm_max_ub[0], SoftmaxMax[0], 128, 1, 128, 0.000000e+00f);
    tl::ascend::copy_gm_to_ub<float, 128>(sm_sum_ub[0], SoftmaxSum[0], 128, 1, 128, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(4);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(4);
    AscendC::Abs(sm_sum_ub[0], sm_sum_ub[0], 128);
    tl::ascend::Fill<float>(eps_sm_ub[0], 1.000000e-08f, 128);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Max(sm_sum_ub[0], sm_sum_ub[0], eps_sm_ub[0], 128);
    tl::ascend::copy_gm_to_ub<float, 64, 128>(bmm1_ub[0], bmm1Res[(cid * 64)], 2048, 128, 64, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(5);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(5);
    {
    AscendC::Muls(bmm1_ub[0], bmm1_ub[0], 4.419417e-02f, 8192);
    }
    tl::ascend::Fill<float>(p_reduced_tmp_ub[0], 1.000000e+00f, 64);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::brcb<float>(re_tmp_ub[0], sm_max_ub[0], 16, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::sub_mask<float>(bmm1_ub[0], bmm1_ub[0], re_tmp_ub[0], 0xFFFFFFFFFFFFFFFF, 0, 128, 1, 1, 0, 8, 8, 1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(bmm1_ub[0], bmm1_ub[0], 8192);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::brcb<float>(re_tmp_ub[0], sm_sum_ub[0], 16, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::div_mask<float>(bmm1_ub[0], bmm1_ub[0], re_tmp_ub[0], 0xFFFFFFFFFFFFFFFF, 0, 128, 1, 1, 0, 8, 8, 1);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::reduce_sum<float,  128,  64,  0>(p_reduced_tmp_ub[0], bmm1_ub[0], tmp_ub[0], true);
    AscendC::PipeBarrier<PIPE_V>();
    {
    AscendC::Muls(p_reduced_tmp_ub[0], p_reduced_tmp_ub[0], 7.812500e-03f, 64);
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(6);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(6);
    tl::ascend::copy_ub_to_gm<float, 64>(PReduced[(cid * 64)], p_reduced_tmp_ub[0], 2048, 1, 64);
    tl::ascend::copy_gm_to_ub<half, 64>(weight_all_half_ub[0], Weights[0], 64, 1, 64, half(0.000000e+00f));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(7);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(7);
    AscendC::Cast(weight_all_ub[0], weight_all_half_ub[0], AscendC::RoundMode::CAST_NONE, 64);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    tl::ascend::Fill<float>(p_reduced_tmp_ub[0], 1.000000e+00f, 64);
    for (int32_t k = 0; k < 64; ++k) {
      AscendC::PipeBarrier<PIPE_V>();
      {
      AscendC::PipeBarrier<PIPE_ALL>();
      auto weight_all_ub_scalar = weight_all_ub.GetValue(k);
      AscendC::Muls(weight_2d_ub[(k * 64)], p_reduced_tmp_ub[0], weight_all_ub_scalar, 64);
      }
    }
    tl::ascend::Fill<float>(rsy_max_ub[0], -1.000000e+30f, 1);
    for (int32_t n_rsy = 0; n_rsy < 32; ++n_rsy) {
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
      tl::ascend::copy_gm_to_ub<float, 64, 64>(bmm2_ub[0], bmm2Res[(n_rsy * 64)], 2048, 64, 64, 0.000000e+00f);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(bmm2_ub[0], bmm2_ub[0], weight_2d_ub[0], 4096);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::reduce_sum<float,  64,  64,  0>(rsy_reduce_ub[0], bmm2_ub[0], tmp_ub[0], true);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::copy_ub_to_ub<float, float, 64>(rsy_all_ub[(n_rsy * 64)], rsy_reduce_ub[0], 1, 64, 64, 1, 64, 64);
      AscendC::PipeBarrier<PIPE_MTE3>();
      tl::ascend::copy_ub_to_gm<float, 64>(ReduceSumY[(n_rsy * 64)], rsy_reduce_ub[0], 2048, 1, 64);
      {
      AscendC::Muls(rsy_bc_ub[0], rsy_reduce_ub[0], 1.000000e+00f, 64);
      }
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::reduce_max<float, 1, 64, -1>(rsy_chunk_max_ub[0], rsy_bc_ub[0], tmp_ub[0], true);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Max(rsy_max_ub[0], rsy_max_ub[0], rsy_chunk_max_ub[0], 1);
    }
    AscendC::PipeBarrier<PIPE_V>();
    {
    AscendC::PipeBarrier<PIPE_ALL>();
    auto rsy_max_ub_scalar = rsy_max_ub.GetValue(0);
    AscendC::Muls(softmax_i_chunk_ub[0], p_reduced_tmp_ub[0], rsy_max_ub_scalar, 64);
    }
    tl::ascend::Fill<float>(rsy_sum_ub[0], 0, 1);
    for (int32_t n_idx_iter = 0; n_idx_iter < 32; ++n_idx_iter) {
      tl::ascend::copy_ub_to_ub<float, float, 64>(rsy_chunk_ub[0], rsy_all_ub[(n_idx_iter * 64)], 1, 64, 64, 1, 64, 64);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub(rsy_chunk_ub[0], rsy_chunk_ub[0], softmax_i_chunk_ub[0], 64);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Exp(rsy_chunk_ub[0], rsy_chunk_ub[0], 64);
      AscendC::PipeBarrier<PIPE_V>();
      {
      AscendC::Muls(rsy_bc_ub[0], rsy_chunk_ub[0], 1.000000e+00f, 64);
      }
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::reduce_sum<float,  1,  64,  -1>(rsy_chunk_sum_ub[0], rsy_bc_ub[0], tmp_ub[0], true);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add(rsy_sum_ub[0], rsy_sum_ub[0], rsy_chunk_sum_ub[0], 1);
    }
    tl::ascend::Fill<float>(eps_sum_ub[0], 1.000000e-08f, 1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Max(rsy_sum_ub[0], rsy_sum_ub[0], eps_sum_ub[0], 1);
    tl::ascend::copy_ub_to_ub<float, float, 64>(rsy_chunk_ub[0], rsy_all_ub[(cid * 64)], 1, 64, 64, 1, 64, 64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(rsy_chunk_ub[0], rsy_chunk_ub[0], softmax_i_chunk_ub[0], 64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(rsy_chunk_ub[0], rsy_chunk_ub[0], 64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::PipeBarrier<PIPE_ALL>();
    auto rsy_sum_ub_scalar = 1.0f / (float)rsy_sum_ub.GetValue(0);
    AscendC::Muls(rsy_chunk_ub[0], rsy_chunk_ub[0], rsy_sum_ub_scalar, 64);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(4);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(4);
    tl::ascend::copy_ub_to_gm<float, 64>(SoftmaxI[(cid * 64)], rsy_chunk_ub[0], 2048, 1, 64);
    tl::ascend::Fill<float>(ones_2d_ub[0], 1.000000e+00f, 2048);
    tl::ascend::Fill<float>(eps_ub[0], 1.000000e-08f, 64);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(5);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(5);
    tl::ascend::copy_gm_to_ub<float, 64>(softmax_i_chunk_ub[0], SoftmaxI[(cid * 64)], 2048, 1, 64, 0.000000e+00f);
    tl::ascend::copy_gm_to_ub<float, 64>(p_reduced_tmp_ub[0], PReduced[(cid * 64)], 2048, 1, 64, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(6);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(6);
    AscendC::Sub(diff_ub[0], softmax_i_chunk_ub[0], p_reduced_tmp_ub[0], 64);
    for (int32_t g_id = 0; g_id < 2; ++g_id) {
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::Fill<float>(weight_ub[0], 0, 32);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(2);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(2);
      tl::ascend::copy_gm_to_ub<float, 64, 32>(dw_chunk_ub[0], bmm2Res[((g_id * 65536) + (cid * 64))], 2048, 32, 64, 0.000000e+00f);
      for (int32_t i_2 = 0; i_2 < 32; ++i_2) {
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(7);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(7);
        AscendC::PipeBarrier<PIPE_V>();
        {
        AscendC::Muls(rsy_tmp_ub[(i_2 * 64)], dw_chunk_ub[(i_2 * 64)], -1.000000e+04f, 64);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(rsy_tmp_ub[(i_2 * 64)], rsy_tmp_ub[(i_2 * 64)], 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(rsy_tmp_ub[(i_2 * 64)], ones_2d_ub[(i_2 * 64)], rsy_tmp_ub[(i_2 * 64)], 64);
        AscendC::Mul(rg_chunk_ub[(i_2 * 64)], weight_2d_ub[((g_id * 2048) + (i_2 * 64))], diff_ub[0], 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(rg_chunk_ub[(i_2 * 64)], rg_chunk_ub[(i_2 * 64)], rsy_tmp_ub[(i_2 * 64)], 64);
        AscendC::Mul(dw_chunk_ub[(i_2 * 64)], dw_chunk_ub[(i_2 * 64)], diff_ub[0], 64);
      }
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::reduce_sum<float,  32,  64,  -1>(dw_reduce_ub[0], dw_chunk_ub[0], tmp_ub[0], true);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add(weight_ub[0], weight_ub[0], dw_reduce_ub[0], 32);
      AscendC::Cast(rg_half_ub[0], rg_chunk_ub[0], AscendC::RoundMode::CAST_ROUND, 2048);
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
      AscendC::PipeBarrier<PIPE_MTE3>();
      tl::ascend::copy_ub_to_gm<half, 64, 32>(reluGradRes[((g_id * 65536) + (cid * 64))], rg_half_ub[0], 2048, 32, 64);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(dw_half_ub[0], weight_ub[0], AscendC::RoundMode::CAST_ROUND, 32);
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);
      tl::ascend::copy_ub_to_gm<half, 32>(dWeight_partial[((cid * 64) + (g_id * 32))], dw_half_ub[0], 2048, 1, 32);
    }
    tl::ascend::copy_gm_to_ub<float, 64>(loss_chunk_ub[0], PReduced[(cid * 64)], 2048, 1, 64, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    AscendC::Max(loss_chunk_ub[0], loss_chunk_ub[0], eps_ub[0], 64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Ln(loss_log_p_ub[0], loss_chunk_ub[0], 64);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
    tl::ascend::copy_gm_to_ub<float, 64>(loss_chunk_ub[0], SoftmaxI[(cid * 64)], 2048, 1, 64, 0.000000e+00f);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
    AscendC::Max(loss_chunk_ub[0], loss_chunk_ub[0], eps_ub[0], 64);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Ln(loss_log_si_ub[0], loss_chunk_ub[0], 64);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
    tl::ascend::copy_gm_to_ub<float, 64>(loss_chunk_ub[0], PReduced[(cid * 64)], 2048, 1, 64, 0.000000e+00f);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(loss_val_ub[0], loss_log_p_ub[0], loss_log_si_ub[0], 64);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(4);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(4);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(loss_val_ub[0], loss_chunk_ub[0], loss_val_ub[0], 64);
    AscendC::PipeBarrier<PIPE_V>();
    {
    AscendC::Muls(rsy_bc_ub[0], loss_val_ub[0], 1.000000e+00f, 64);
    }
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::reduce_sum<float,  1,  64,  -1>(loss_sum_ub[0], rsy_bc_ub[0], tmp_ub[0], true);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(5);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(5);
    tl::ascend::copy_ub_to_gm<float, 1>(LossOut[cid], loss_sum_ub[0], 32, 1, 1);
    AscendC::CrossCoreSetFlag<0x2, PIPE_V>(12);
  }
  if ASCEND_IS_AIC {
    AscendC::CrossCoreWaitFlag(12);
    tl::ascend::copy_gm_to_l1<half, 64, 64>(RG_L1[0], reluGradRes[(cid * 64)], 2048, 64, 64, true);
    tl::ascend::copy_gm_to_l1<half, 64, 128>(GKI2_L1[0], GatherKeyIndex_ws[(cid * 8192)], 128, 64, 128, true);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(6);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(6);
    tl::ascend::gemm_v0<half, float, 64, 128, 64, false, false>(RG_L1[0], GKI2_L1[0], DQ_L0C[0], ascend_l0a, ascend_l0b, (bool)1);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(7);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(7);
    tl::ascend::atomic_add_l0c_to_gm<float, float, layout::RowMajor, 64, 128>(dQueryIndexAccum[0], DQ_L0C[0], 128, 64, 128);
    tl::ascend::copy_gm_to_l1<half, 64, 128>(QI_L1_dk[0], QueryIndex[0], 128, 64, 128, true);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(8);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(8);
    tl::ascend::gemm_v0<half, float, 64, 128, 64, true, false>(RG_L1[0], QI_L1_dk[0], DK_L0C[0], ascend_l0a, ascend_l0b, (bool)1);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(9);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(9);
    tl::ascend::atomic_add_l0c_to_gm<float, float, layout::RowMajor, 64, 128>(dk_matmul_partial[(cid * 8192)], DK_L0C[0], 128, 64, 128);
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(3);
  }
  if ASCEND_IS_AIV {
    AscendC::CrossCoreWaitFlag(3);
    for (int32_t g_id_1 = 0; g_id_1 < 16; ++g_id_1) {
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
      tl::ascend::copy_gm_to_ub<float, 128, 4>(dq_fp32_ub[0], dQueryIndexAccum[(g_id_1 * 512)], 128, 4, 128, 0.000000e+00f);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(4);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(4);
      AscendC::Cast(dq_fp16_ub[0], dq_fp32_ub[0], AscendC::RoundMode::CAST_ROUND, 512);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);
      AscendC::PipeBarrier<PIPE_MTE3>();
      tl::ascend::copy_ub_to_gm<half, 128, 4>(dQueryIndex[(g_id_1 * 512)], dq_fp16_ub[0], 128, 4, 128);
    }
  }
  if ASCEND_IS_AIV {
    for (int32_t k_id = (cid * 64); k_id < ((cid * 64) + 64); ++k_id) {
      int32_t idx_2 = SparseIndices.GetValue(k_id);
      if (0 <= idx_2) {
        tl::ascend::copy_gm_to_ub<float, 128, 1>(dk_chunk_ub[0], dk_matmul_partial[(k_id * 128)], 262144, 1, 128, 0.000000e+00f);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
        tl::ascend::atomic_add_ub_to_gm<float, 128, 1>(dKeyIndex[(idx_2 * 128)], dk_chunk_ub[0], 262144, 1, 128);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
      }
    }
  }
}

void main_kernel_tiling() {
}

extern "C" void call(uint8_t* SparseIndices_handle, uint8_t* Query_handle, uint8_t* QueryRope_handle, uint8_t* Key_handle, uint8_t* KeyRope_handle, uint8_t* QueryIndex_handle, uint8_t* KeyIndex_handle, uint8_t* Weights_handle, uint8_t* SoftmaxMax_handle, uint8_t* SoftmaxSum_handle, uint8_t* dQueryIndexAccum_handle, uint8_t* GatherKey_ws_handle, uint8_t* GatherKeyRope_ws_handle, uint8_t* GatherKeyIndex_ws_handle, uint8_t* bmm1Res_handle, uint8_t* bmm2Res_handle, uint8_t* PReduced_handle, uint8_t* ReduceSumY_handle, uint8_t* SoftmaxI_handle, uint8_t* dk_matmul_partial_handle, uint8_t* dQueryIndex_handle, uint8_t* dWeight_partial_handle, uint8_t* reluGradRes_handle, uint8_t* LossOut_handle, uint8_t* dKeyIndex_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<32, nullptr, stream>>>(SparseIndices_handle, Query_handle, QueryRope_handle, Key_handle, KeyRope_handle, QueryIndex_handle, KeyIndex_handle, Weights_handle, SoftmaxMax_handle, SoftmaxSum_handle, dQueryIndexAccum_handle, GatherKey_ws_handle, GatherKeyRope_ws_handle, GatherKeyIndex_ws_handle, bmm1Res_handle, bmm2Res_handle, PReduced_handle, ReduceSumY_handle, SoftmaxI_handle, dk_matmul_partial_handle, dQueryIndex_handle, dWeight_partial_handle, reluGradRes_handle, LossOut_handle, dKeyIndex_handle, fftsAddr);
}
