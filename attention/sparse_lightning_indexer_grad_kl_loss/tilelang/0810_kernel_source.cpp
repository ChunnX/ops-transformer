#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR SparseIndices_handle,  GM_ADDR KRealSize_handle,  GM_ADDR ValidKMask_handle,  GM_ADDR V0ZeroRow_handle,  GM_ADDR Query_handle,  GM_ADDR QueryRope_handle,  GM_ADDR Key_handle,  GM_ADDR KeyRope_handle,  GM_ADDR QueryIndex_handle,  GM_ADDR KeyIndex_handle,  GM_ADDR Weights_handle,  GM_ADDR SoftmaxMax_handle,  GM_ADDR SoftmaxSum_handle,  GM_ADDR dQueryIndexAccum_handle,  GM_ADDR dKeyIndexAccum_handle,  GM_ADDR GatherKey_ws_handle,  GM_ADDR GatherKeyRope_ws_handle,  GM_ADDR GatherKeyIndex_ws_handle,  GM_ADDR bmm1Res_handle,  GM_ADDR bmm2Res_handle,  GM_ADDR PReduced_handle,  GM_ADDR SoftmaxI_handle,  GM_ADDR dk_matmul_partial_handle,  GM_ADDR reluGradRes_handle,  GM_ADDR LossPartial_handle,  GM_ADDR C2ZeroPad_handle,  GM_ADDR dQueryIndex_handle,  GM_ADDR dKeyIndexOut_handle,  GM_ADDR dWeightOut_handle,  GM_ADDR LossOut_handle, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<int> SparseIndices;
  SparseIndices.SetGlobalBuffer((__gm__ int*)SparseIndices_handle);
  AscendC::GlobalTensor<int> KRealSize;
  KRealSize.SetGlobalBuffer((__gm__ int*)KRealSize_handle);
  AscendC::GlobalTensor<float> ValidKMask;
  ValidKMask.SetGlobalBuffer((__gm__ float*)ValidKMask_handle);
  AscendC::GlobalTensor<bfloat16_t> V0ZeroRow;
  V0ZeroRow.SetGlobalBuffer((__gm__ bfloat16_t*)V0ZeroRow_handle);
  AscendC::GlobalTensor<bfloat16_t> Query;
  Query.SetGlobalBuffer((__gm__ bfloat16_t*)Query_handle);
  AscendC::GlobalTensor<bfloat16_t> QueryRope;
  QueryRope.SetGlobalBuffer((__gm__ bfloat16_t*)QueryRope_handle);
  AscendC::GlobalTensor<bfloat16_t> Key;
  Key.SetGlobalBuffer((__gm__ bfloat16_t*)Key_handle);
  AscendC::GlobalTensor<bfloat16_t> KeyRope;
  KeyRope.SetGlobalBuffer((__gm__ bfloat16_t*)KeyRope_handle);
  AscendC::GlobalTensor<bfloat16_t> QueryIndex;
  QueryIndex.SetGlobalBuffer((__gm__ bfloat16_t*)QueryIndex_handle);
  AscendC::GlobalTensor<bfloat16_t> KeyIndex;
  KeyIndex.SetGlobalBuffer((__gm__ bfloat16_t*)KeyIndex_handle);
  AscendC::GlobalTensor<bfloat16_t> Weights;
  Weights.SetGlobalBuffer((__gm__ bfloat16_t*)Weights_handle);
  AscendC::GlobalTensor<float> SoftmaxMax;
  SoftmaxMax.SetGlobalBuffer((__gm__ float*)SoftmaxMax_handle);
  AscendC::GlobalTensor<float> SoftmaxSum;
  SoftmaxSum.SetGlobalBuffer((__gm__ float*)SoftmaxSum_handle);
  AscendC::GlobalTensor<float> dQueryIndexAccum;
  dQueryIndexAccum.SetGlobalBuffer((__gm__ float*)dQueryIndexAccum_handle);
  AscendC::GlobalTensor<float> dKeyIndexAccum;
  dKeyIndexAccum.SetGlobalBuffer((__gm__ float*)dKeyIndexAccum_handle);
  AscendC::GlobalTensor<bfloat16_t> GatherKey_ws;
  GatherKey_ws.SetGlobalBuffer((__gm__ bfloat16_t*)GatherKey_ws_handle);
  AscendC::GlobalTensor<bfloat16_t> GatherKeyRope_ws;
  GatherKeyRope_ws.SetGlobalBuffer((__gm__ bfloat16_t*)GatherKeyRope_ws_handle);
  AscendC::GlobalTensor<bfloat16_t> GatherKeyIndex_ws;
  GatherKeyIndex_ws.SetGlobalBuffer((__gm__ bfloat16_t*)GatherKeyIndex_ws_handle);
  AscendC::GlobalTensor<float> bmm1Res;
  bmm1Res.SetGlobalBuffer((__gm__ float*)bmm1Res_handle);
  AscendC::GlobalTensor<float> bmm2Res;
  bmm2Res.SetGlobalBuffer((__gm__ float*)bmm2Res_handle);
  AscendC::GlobalTensor<float> PReduced;
  PReduced.SetGlobalBuffer((__gm__ float*)PReduced_handle);
  AscendC::GlobalTensor<float> SoftmaxI;
  SoftmaxI.SetGlobalBuffer((__gm__ float*)SoftmaxI_handle);
  AscendC::GlobalTensor<float> dk_matmul_partial;
  dk_matmul_partial.SetGlobalBuffer((__gm__ float*)dk_matmul_partial_handle);
  AscendC::GlobalTensor<bfloat16_t> reluGradRes;
  reluGradRes.SetGlobalBuffer((__gm__ bfloat16_t*)reluGradRes_handle);
  AscendC::GlobalTensor<float> LossPartial;
  LossPartial.SetGlobalBuffer((__gm__ float*)LossPartial_handle);
  AscendC::GlobalTensor<bfloat16_t> C2ZeroPad;
  C2ZeroPad.SetGlobalBuffer((__gm__ bfloat16_t*)C2ZeroPad_handle);
  AscendC::GlobalTensor<bfloat16_t> dQueryIndex;
  dQueryIndex.SetGlobalBuffer((__gm__ bfloat16_t*)dQueryIndex_handle);
  AscendC::GlobalTensor<bfloat16_t> dKeyIndexOut;
  dKeyIndexOut.SetGlobalBuffer((__gm__ bfloat16_t*)dKeyIndexOut_handle);
  AscendC::GlobalTensor<bfloat16_t> dWeightOut;
  dWeightOut.SetGlobalBuffer((__gm__ bfloat16_t*)dWeightOut_handle);
  AscendC::GlobalTensor<float> LossOut;
  LossOut.SetGlobalBuffer((__gm__ float*)LossOut_handle);

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
  auto tmp_ub = ascend_ub.GetWithOffset<uint8_t>(32768, 120832);
  auto v0_key_ub = ascend_ub.GetWithOffset<bfloat16_t>(16384, 0);
  auto v0_ki_ub = ascend_ub.GetWithOffset<bfloat16_t>(4096, 32768);
  auto v0_rope_ub = ascend_ub.GetWithOffset<bfloat16_t>(4096, 40960);
  auto v1_bw_eps = ascend_ub.GetWithOffset<float>(64, 190016);
  auto c2_qi_l1 = ascend_l1.GetWithOffset<bfloat16_t>(8192, 245760);
  auto c1_query_chunk_0_l1 = ascend_l1.GetWithOffset<bfloat16_t>(16384, 0);
  auto c1_query_chunk_1_l1 = ascend_l1.GetWithOffset<bfloat16_t>(16384, 32768);
  auto c1_query_chunk_2_l1 = ascend_l1.GetWithOffset<bfloat16_t>(16384, 65536);
  auto c1_query_chunk_3_l1 = ascend_l1.GetWithOffset<bfloat16_t>(16384, 98304);
  auto c1_query_rope_cache_l1 = ascend_l1.GetWithOffset<bfloat16_t>(16384, 131072);
  auto c1_qi_l1 = ascend_l1.GetWithOffset<bfloat16_t>(8192, 163840);
  auto c1_key_l1 = ascend_l1.GetWithOffset<bfloat16_t>(16384, 180224);
  auto c1_p_slot0_q_l0a = ascend_l0a.GetWithOffset<bfloat16_t>(16384, 0);
  auto c1_p_slot0_key_l0b = ascend_l0b.GetWithOffset<bfloat16_t>(16384, 0);
  auto c1_p_slot0_l0c = ascend_l0c.GetWithOffset<float>(16384, 0);
  auto c1_p_slot1_q_l0a = ascend_l0a.GetWithOffset<bfloat16_t>(16384, 32768);
  auto c1_p_slot1_key_l0b = ascend_l0b.GetWithOffset<bfloat16_t>(16384, 32768);
  auto c1_p_slot1_l0c = ascend_l0c.GetWithOffset<float>(16384, 65536);
  auto c1_sy_slot0_qi_l0a = ascend_l0a.GetWithOffset<bfloat16_t>(8192, 0);
  auto c1_sy_slot1_qi_l0a = ascend_l0a.GetWithOffset<bfloat16_t>(8192, 16384);
  auto c1_ki_l1 = ascend_l1.GetWithOffset<bfloat16_t>(16384, 212992);
  auto c1_sy_slot0_l0c = ascend_l0c.GetWithOffset<float>(8192, 0);
  auto c1_sy_slot1_l0c = ascend_l0c.GetWithOffset<float>(8192, 0);
  auto c1_sy_slot0_ki_l0b = ascend_l0b.GetWithOffset<bfloat16_t>(16384, 0);
  auto c1_sy_slot1_ki_l0b = ascend_l0b.GetWithOffset<bfloat16_t>(16384, 0);
  auto v1_p_max = ascend_ub.GetWithOffset<float>(128, 49152);
  auto v1_p_sum = ascend_ub.GetWithOffset<float>(128, 49664);
  auto v1_p_partial = ascend_ub.GetWithOffset<float>(128, 50176);
  auto v1_p_bmm1 = ascend_ub.GetWithOffset<float>(16384, 50688);
  auto v1_p_tmp = ascend_ub.GetWithOffset<float>(1024, 116224);
  auto v1_p_out = ascend_ub.GetWithOffset<float>(128, 120320);
  auto v1_sy_weight_half = ascend_ub.GetWithOffset<bfloat16_t>(64, 153600);
  auto v1_sy_weight = ascend_ub.GetWithOffset<float>(64, 153728);
  auto v1_sy_weight_2d = ascend_ub.GetWithOffset<float>(8192, 153984);
  auto v1_sy_max = ascend_ub.GetWithOffset<float>(1, 188800);
  auto v1_sy_storage = ascend_ub.GetWithOffset<float>(9216, 0);
  auto v1_sy_broadcast = ascend_ub.GetWithOffset<float>(128, 189344);
  auto v1_sy_floor = ascend_ub.GetWithOffset<float>(1, 189888);
  auto v1_sy_sum = ascend_ub.GetWithOffset<float>(1, 189920);
  auto v1_sy_eps = ascend_ub.GetWithOffset<float>(1, 189984);
  auto v1_sy_tmp = ascend_ub.GetWithOffset<float>(512, 186752);
  auto v1_sy_reduce = ascend_ub.GetWithOffset<float>(128, 188832);
  auto v1_sy_tmp_max = ascend_ub.GetWithOffset<float>(1, 189856);
  auto v1_sy_tmp_sum = ascend_ub.GetWithOffset<float>(1, 189952);
  auto v1_bw_dw_accum = ascend_ub.GetWithOffset<float>(32, 190272);
  auto v1_bw_loss_accum = ascend_ub.GetWithOffset<float>(1, 190400);
  auto v1_bw_weight_half = ascend_ub.GetWithOffset<bfloat16_t>(32, 190432);
  auto v1_bw_weight = ascend_ub.GetWithOffset<float>(32, 190496);
  auto v1_bw_weight_2d = ascend_ub.GetWithOffset<float>(2048, 0);
  auto v1_bw_row_tmp = ascend_ub.GetWithOffset<float>(256, 190624);
  auto v1_bw_p = ascend_ub.GetWithOffset<float>(64, 191648);
  auto v1_bw_sy = ascend_ub.GetWithOffset<float>(64, 191904);
  auto v1_bw_bmm2 = ascend_ub.GetWithOffset<float>(2048, 8192);
  auto v1_bw_diff = ascend_ub.GetWithOffset<float>(64, 192160);
  auto v1_bw_diff_2d = ascend_ub.GetWithOffset<float>(2048, 16384);
  auto v1_bw_rg = ascend_ub.GetWithOffset<float>(2048, 24576);
  auto v1_bw_dw = ascend_ub.GetWithOffset<float>(32, 192416);
  auto v1_bw_rg_half = ascend_ub.GetWithOffset<bfloat16_t>(2048, 16384);
  auto v1_bw_log_p = ascend_ub.GetWithOffset<float>(64, 192544);
  auto v1_bw_log_sy = ascend_ub.GetWithOffset<float>(64, 192800);
  auto v1_bw_loss_2d = ascend_ub.GetWithOffset<float>(64, 193056);
  auto v1_bw_loss_sum = ascend_ub.GetWithOffset<float>(1, 193312);
  auto v1_bw_dw_half = ascend_ub.GetWithOffset<bfloat16_t>(32, 193344);
  auto c2_dq_l0c = ascend_l0c.GetWithOffset<float>(8192, 0);
  auto c2_rg_l1 = ascend_l1.GetWithOffset<bfloat16_t>(8192, 262144);
  auto c2_gki_l1 = ascend_l1.GetWithOffset<bfloat16_t>(16384, 278528);
  auto c2_dk_rg_l0a = ascend_l0a.GetWithOffset<bfloat16_t>(8192, 32768);
  auto c2_dk_qi_l0b = ascend_l0b.GetWithOffset<bfloat16_t>(8192, 0);
  auto c2_dq_rg_l0a = ascend_l0a.GetWithOffset<bfloat16_t>(8192, 49152);
  auto c2_dq_gki_l0b = ascend_l0b.GetWithOffset<bfloat16_t>(16384, 16384);
  auto c2_dk_l0c = ascend_l0c.GetWithOffset<float>(16384, 0);
  auto v2_zero_batch = ascend_ub.GetWithOffset<float>(4096, 0);
  auto v2_dk_batch_ping = ascend_ub.GetWithOffset<float>(2048, 16384);
  auto v2_dk_batch_pong = ascend_ub.GetWithOffset<float>(2048, 24576);
  auto v2_finalize_dq_fp32 = ascend_ub.GetWithOffset<float>(128, 193408);
  auto v2_finalize_dq_half = ascend_ub.GetWithOffset<bfloat16_t>(128, 193920);
  auto v2_finalize_dk_fp32 = ascend_ub.GetWithOffset<float>(4096, 32768);
  auto v2_finalize_dk_half = ascend_ub.GetWithOffset<bfloat16_t>(4096, 49152);
  auto v2_loss_total = ascend_ub.GetWithOffset<float>(1, 194176);
  auto v2_loss_chunk = ascend_ub.GetWithOffset<float>(128, 194208);
  auto v2_loss_chunk_sum = ascend_ub.GetWithOffset<float>(1, 194720);
  auto vid = AscendC::GetSubBlockIdx();
  for (int32_t pipe_iter = 0; pipe_iter < 2; ++pipe_iter) {
    if (pipe_iter < 1) {
      if (((pipe_iter * 3) + cid) < 1) {
        if ASCEND_IS_AIV {
          AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
          for (int32_t lane = 0; lane < 16; ++lane) {
            int32_t sparse_idx = SparseIndices.GetValue(((((pipe_iter * 3072) + (cid * 1024)) + (vid * 512)) + lane));
            if (0 <= sparse_idx) {
              tl::ascend::copy_gm_to_ub<bfloat16_t, 512, 1>(v0_key_ub[(lane * 512)], Key[(((pipe_iter * 1967616) + (cid * 655872)) + (sparse_idx * 512))], 655872, 1, 512, bfloat16_t(0.000000e+00f));
              tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 1>(v0_ki_ub[(lane * 128)], KeyIndex[(((pipe_iter * 491904) + (cid * 163968)) + (sparse_idx * 128))], 163968, 1, 128, bfloat16_t(0.000000e+00f));
              tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 1>(v0_rope_ub[(lane * 128)], KeyRope[(((pipe_iter * 491904) + (cid * 163968)) + (sparse_idx * 128))], 163968, 1, 128, bfloat16_t(0.000000e+00f));
            }
            if (sparse_idx < 0) {
              tl::ascend::copy_gm_to_ub<bfloat16_t, 512, 1>(v0_key_ub[(lane * 512)], V0ZeroRow[0], 512, 1, 512, bfloat16_t(0.000000e+00f));
              tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 1>(v0_ki_ub[(lane * 128)], V0ZeroRow[0], 512, 1, 128, bfloat16_t(0.000000e+00f));
              tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 1>(v0_rope_ub[(lane * 128)], V0ZeroRow[0], 512, 1, 128, bfloat16_t(0.000000e+00f));
            }
          }
          AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
          for (int32_t chunk_id = 0; chunk_id < 32; ++chunk_id) {
            if (chunk_id < 31) {
              AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(((chunk_id + 1) % 2));
              for (int32_t lane_1 = 0; lane_1 < 16; ++lane_1) {
                int32_t sparse_idx_1 = SparseIndices.GetValue(((((((pipe_iter * 3072) + (cid * 1024)) + (vid * 512)) + (chunk_id * 16)) + lane_1) + 16));
                if (0 <= sparse_idx_1) {
                  tl::ascend::copy_gm_to_ub<bfloat16_t, 512, 1>(v0_key_ub[((((chunk_id + 1) % 2) * 8192) + (lane_1 * 512))], Key[(((pipe_iter * 1967616) + (cid * 655872)) + (sparse_idx_1 * 512))], 655872, 1, 512, bfloat16_t(0.000000e+00f));
                  tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 1>(v0_ki_ub[((((chunk_id + 1) % 2) * 2048) + (lane_1 * 128))], KeyIndex[(((pipe_iter * 491904) + (cid * 163968)) + (sparse_idx_1 * 128))], 163968, 1, 128, bfloat16_t(0.000000e+00f));
                  tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 1>(v0_rope_ub[((((chunk_id + 1) % 2) * 2048) + (lane_1 * 128))], KeyRope[(((pipe_iter * 491904) + (cid * 163968)) + (sparse_idx_1 * 128))], 163968, 1, 128, bfloat16_t(0.000000e+00f));
                }
                if (sparse_idx_1 < 0) {
                  tl::ascend::copy_gm_to_ub<bfloat16_t, 512, 1>(v0_key_ub[((((chunk_id + 1) % 2) * 8192) + (lane_1 * 512))], V0ZeroRow[0], 512, 1, 512, bfloat16_t(0.000000e+00f));
                  tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 1>(v0_ki_ub[((((chunk_id + 1) % 2) * 2048) + (lane_1 * 128))], V0ZeroRow[0], 512, 1, 128, bfloat16_t(0.000000e+00f));
                  tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 1>(v0_rope_ub[((((chunk_id + 1) % 2) * 2048) + (lane_1 * 128))], V0ZeroRow[0], 512, 1, 128, bfloat16_t(0.000000e+00f));
                }
              }
              AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(((chunk_id + 1) % 2));
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>((chunk_id % 2));
            tl::ascend::copy_ub_to_gm<bfloat16_t, 512, 16>(GatherKey_ws[((((pipe_iter * 1572864) + (cid * 524288)) + (vid * 262144)) + (chunk_id * 8192))], v0_key_ub[((chunk_id % 2) * 8192)], 512, 16, 512);
            tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 16>(GatherKeyIndex_ws[((((pipe_iter * 393216) + (cid * 131072)) + (vid * 65536)) + (chunk_id * 2048))], v0_ki_ub[((chunk_id % 2) * 2048)], 128, 16, 128);
            tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 16>(GatherKeyRope_ws[((((pipe_iter * 393216) + (cid * 131072)) + (vid * 65536)) + (chunk_id * 2048))], v0_rope_ub[((chunk_id % 2) * 2048)], 128, 16, 128);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((chunk_id % 2));
          }
          AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
          AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(2);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
        }
      }
    }
    if (pipe_iter == 1) {
      if (((pipe_iter * 3) + cid) < 4) {
        if ASCEND_IS_AIC {
          AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
          AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
          AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
          AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2);
          AscendC::SetFlag<AscendC::HardEvent::FIX_M>(2);
          AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_query_chunk_0_l1[0], Query[(((pipe_iter * 196608) + (cid * 65536)) - 196608)], 512, 128, 128, true);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_query_chunk_1_l1[0], Query[(((pipe_iter * 196608) + (cid * 65536)) - 196480)], 512, 128, 128, true);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_query_chunk_2_l1[0], Query[(((pipe_iter * 196608) + (cid * 65536)) - 196352)], 512, 128, 128, true);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_query_chunk_3_l1[0], Query[(((pipe_iter * 196608) + (cid * 65536)) - 196224)], 512, 128, 128, true);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_query_rope_cache_l1[0], QueryRope[(((pipe_iter * 49152) + (cid * 16384)) - 49152)], 128, 128, 128, true);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(4);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(4);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(c1_qi_l1[0], QueryIndex[(((pipe_iter * 24576) + (cid * 8192)) - 24576)], 128, 64, 128, true);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(5);
          AscendC::CrossCoreWaitFlag(0);
          for (int32_t k_inner = 0; k_inner < 8; ++k_inner) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
            tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_key_l1[0], GatherKey_ws[((((pipe_iter * 1572864) + (cid * 524288)) + (k_inner * 65536)) - 1572864)], 512, 128, 128, true);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
            if ((k_inner % 2) == 0) {
              for (int32_t d_idx = 0; d_idx < 4; ++d_idx) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
                if (d_idx == 0) {
                  tl::ascend::copy_l1_to_l0a<bfloat16_t, 128, 128, false>(c1_p_slot0_q_l0a[0], c1_query_chunk_0_l1[0], 128, 128);
                }
                if (d_idx == 1) {
                  tl::ascend::copy_l1_to_l0a<bfloat16_t, 128, 128, false>(c1_p_slot0_q_l0a[0], c1_query_chunk_1_l1[0], 128, 128);
                }
                if (d_idx == 2) {
                  tl::ascend::copy_l1_to_l0a<bfloat16_t, 128, 128, false>(c1_p_slot0_q_l0a[0], c1_query_chunk_2_l1[0], 128, 128);
                }
                if (d_idx == 3) {
                  tl::ascend::copy_l1_to_l0a<bfloat16_t, 128, 128, false>(c1_p_slot0_q_l0a[0], c1_query_chunk_3_l1[0], 128, 128);
                }
                tl::ascend::copy_l1_to_l0b<bfloat16_t, 128, 128, true>(c1_p_slot0_key_l0b[0], c1_key_l1[0], 128, 128);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
                if (d_idx < 3) {
                  AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
                  tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_key_l1[0], GatherKey_ws[(((((pipe_iter * 1572864) + (cid * 524288)) + (k_inner * 65536)) + (d_idx * 128)) - 1572736)], 512, 128, 128, true);
                  AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
                }
                if (d_idx == 3) {
                  AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
                  tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_key_l1[0], GatherKeyRope_ws[((((pipe_iter * 393216) + (cid * 131072)) + (k_inner * 16384)) - 393216)], 128, 128, 128, true);
                  AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
                tl::ascend::mma<bfloat16_t, float, 128, 128>(c1_p_slot0_q_l0a[0],c1_p_slot0_key_l0b[0],c1_p_slot0_l0c[0], (d_idx == 0), 128);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
              }
              AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
              tl::ascend::copy_l1_to_l0a<bfloat16_t, 128, 128, false>(c1_p_slot0_q_l0a[0], c1_query_rope_cache_l1[0], 128, 128);
              tl::ascend::copy_l1_to_l0b<bfloat16_t, 128, 128, true>(c1_p_slot0_key_l0b[0], c1_key_l1[0], 128, 128);
              AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
              AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
              AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
              tl::ascend::mma<bfloat16_t, float, 128, 128>(c1_p_slot0_q_l0a[0],c1_p_slot0_key_l0b[0],c1_p_slot0_l0c[0], (bool)0, 128);
              AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
              AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
              AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
              tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 128, 0>(bmm1Res[((((pipe_iter * 393216) + (cid * 131072)) + (k_inner * 128)) - 393216)], c1_p_slot0_l0c[0], 1024, 128, 128, 0);
              AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
            } else {
              for (int32_t d_idx_1 = 0; d_idx_1 < 4; ++d_idx_1) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
                if (d_idx_1 == 0) {
                  tl::ascend::copy_l1_to_l0a<bfloat16_t, 128, 128, false>(c1_p_slot1_q_l0a[0], c1_query_chunk_0_l1[0], 128, 128);
                }
                if (d_idx_1 == 1) {
                  tl::ascend::copy_l1_to_l0a<bfloat16_t, 128, 128, false>(c1_p_slot1_q_l0a[0], c1_query_chunk_1_l1[0], 128, 128);
                }
                if (d_idx_1 == 2) {
                  tl::ascend::copy_l1_to_l0a<bfloat16_t, 128, 128, false>(c1_p_slot1_q_l0a[0], c1_query_chunk_2_l1[0], 128, 128);
                }
                if (d_idx_1 == 3) {
                  tl::ascend::copy_l1_to_l0a<bfloat16_t, 128, 128, false>(c1_p_slot1_q_l0a[0], c1_query_chunk_3_l1[0], 128, 128);
                }
                tl::ascend::copy_l1_to_l0b<bfloat16_t, 128, 128, true>(c1_p_slot1_key_l0b[0], c1_key_l1[0], 128, 128);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(2);
                if (d_idx_1 < 3) {
                  AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
                  tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_key_l1[0], GatherKey_ws[(((((pipe_iter * 1572864) + (cid * 524288)) + (k_inner * 65536)) + (d_idx_1 * 128)) - 1572736)], 512, 128, 128, true);
                  AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
                }
                if (d_idx_1 == 3) {
                  AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
                  tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_key_l1[0], GatherKeyRope_ws[((((pipe_iter * 393216) + (cid * 131072)) + (k_inner * 16384)) - 393216)], 128, 128, 128, true);
                  AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(2);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(2);
                tl::ascend::mma<bfloat16_t, float, 128, 128>(c1_p_slot1_q_l0a[0],c1_p_slot1_key_l0b[0],c1_p_slot1_l0c[0], (d_idx_1 == 0), 128);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(2);
              }
              AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
              tl::ascend::copy_l1_to_l0a<bfloat16_t, 128, 128, false>(c1_p_slot1_q_l0a[0], c1_query_rope_cache_l1[0], 128, 128);
              tl::ascend::copy_l1_to_l0b<bfloat16_t, 128, 128, true>(c1_p_slot1_key_l0b[0], c1_key_l1[0], 128, 128);
              AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
              AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(2);
              AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(2);
              AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(2);
              tl::ascend::mma<bfloat16_t, float, 128, 128>(c1_p_slot1_q_l0a[0],c1_p_slot1_key_l0b[0],c1_p_slot1_l0c[0], (bool)0, 128);
              AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2);
              AscendC::SetFlag<AscendC::HardEvent::M_FIX>(2);
              AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(2);
              tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 128, 0>(bmm1Res[((((pipe_iter * 393216) + (cid * 131072)) + (k_inner * 128)) - 393216)], c1_p_slot1_l0c[0], 1024, 128, 128, 0);
              AscendC::SetFlag<AscendC::HardEvent::FIX_M>(2);
            }
          }
          AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
          AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
          AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
          AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2);
          AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(2);
          AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
          AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
          AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2);
          AscendC::SetFlag<AscendC::HardEvent::FIX_M>(2);
          AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(4);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(5);
          AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
          tl::ascend::copy_l1_to_l0a<bfloat16_t, 64, 128, false>(c1_sy_slot0_qi_l0a[0], c1_qi_l1[0], 64, 128);
          AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
          AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2);
          tl::ascend::copy_l1_to_l0a<bfloat16_t, 64, 128, false>(c1_sy_slot1_qi_l0a[0], c1_qi_l1[0], 64, 128);
          AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2);
          AscendC::CrossCoreWaitFlag(2);
          for (int32_t k_inner_1 = 0; k_inner_1 < 8; ++k_inner_1) {
            if ((k_inner_1 % 2) == 0) {
              AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
              tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_ki_l1[0], GatherKeyIndex_ws[((((pipe_iter * 393216) + (cid * 131072)) + (k_inner_1 * 16384)) - 393216)], 128, 128, 128, true);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(1);
              AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(1);
              tl::ascend::copy_l1_to_l0b<bfloat16_t, 128, 128, true>(c1_sy_slot0_ki_l0b[0], c1_ki_l1[0], 128, 128);
              AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
              AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
              AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
              tl::ascend::mma<bfloat16_t, float, 64, 128>(c1_sy_slot0_qi_l0a[0],c1_sy_slot0_ki_l0b[0],c1_sy_slot0_l0c[0], (bool)1, 128);
              AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
              AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
              AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
              tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 128, 1>(bmm2Res[((((pipe_iter * 196608) + (cid * 65536)) + (k_inner_1 * 128)) - 196608)], c1_sy_slot0_l0c[0], 1024, 64, 128, 0);
              AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
            } else {
              AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
              tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_ki_l1[0], GatherKeyIndex_ws[((((pipe_iter * 393216) + (cid * 131072)) + (k_inner_1 * 16384)) - 393216)], 128, 128, 128, true);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(1);
              AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(1);
              tl::ascend::copy_l1_to_l0b<bfloat16_t, 128, 128, true>(c1_sy_slot1_ki_l0b[0], c1_ki_l1[0], 128, 128);
              AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
              AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(2);
              AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(2);
              AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(2);
              tl::ascend::mma<bfloat16_t, float, 64, 128>(c1_sy_slot1_qi_l0a[0],c1_sy_slot1_ki_l0b[0],c1_sy_slot1_l0c[0], (bool)1, 128);
              AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2);
              AscendC::SetFlag<AscendC::HardEvent::M_FIX>(2);
              AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(2);
              tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 128, 1>(bmm2Res[((((pipe_iter * 196608) + (cid * 65536)) + (k_inner_1 * 128)) - 196608)], c1_sy_slot1_l0c[0], 1024, 64, 128, 0);
              AscendC::SetFlag<AscendC::HardEvent::FIX_M>(2);
            }
          }
          AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
          AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
          AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
          AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2);
          AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(2);
          AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(6);
        }
        if ASCEND_IS_AIV {
          if (vid == 0) {
            AscendC::CrossCoreWaitFlag(4);
          } else {
            AscendC::CrossCoreWaitFlag(6);
          }
          if (vid == 0) {
            tl::ascend::copy_gm_to_ub<float, 128>(v1_p_max[0], SoftmaxMax[(((pipe_iter * 384) + (cid * 128)) - 384)], 128, 1, 128, 0.000000e+00f);
            tl::ascend::copy_gm_to_ub<float, 128>(v1_p_sum[0], SoftmaxSum[(((pipe_iter * 384) + (cid * 128)) - 384)], 128, 1, 128, 0.000000e+00f);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
            AscendC::Abs(v1_p_sum[0], v1_p_sum[0], 128);
            tl::ascend::Fill<float>(v1_p_partial[0], 1.000000e-08f, 128);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Max(v1_p_sum[0], v1_p_sum[0], v1_p_partial[0], 128);
            for (int32_t n_idx = 0; n_idx < 8; ++n_idx) {
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
              AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
              tl::ascend::copy_gm_to_ub<float, 128, 128>(v1_p_bmm1[0], bmm1Res[((((pipe_iter * 393216) + (cid * 131072)) + (n_idx * 128)) - 393216)], 1024, 128, 128, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
              {
              AscendC::Muls(v1_p_bmm1[0], v1_p_bmm1[0], 4.419417e-02f, 16384);
              }
              AscendC::PipeBarrier<PIPE_V>();
              for (int32_t p_row_chunk = 0; p_row_chunk < 2; ++p_row_chunk) {
                for (int32_t p_col_chunk = 0; p_col_chunk < 2; ++p_col_chunk) {
                  tl::ascend::brcb<float>(v1_p_tmp[(p_row_chunk * 512)], v1_p_max[(p_row_chunk * 64)], 8, 1, 8);
                  AscendC::PipeBarrier<PIPE_V>();
                  tl::ascend::sub_mask<float>(v1_p_bmm1[((p_row_chunk * 8192) + (p_col_chunk * 64))], v1_p_bmm1[((p_row_chunk * 8192) + (p_col_chunk * 64))], v1_p_tmp[(p_row_chunk * 512)], 0xFFFFFFFFFFFFFFFF, 0, 64, 1, 1, 0, 16, 16, 1);
                }
              }
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Exp(v1_p_bmm1[0], v1_p_bmm1[0], 16384);
              AscendC::PipeBarrier<PIPE_V>();
              for (int32_t p_row_chunk_1 = 0; p_row_chunk_1 < 2; ++p_row_chunk_1) {
                for (int32_t p_col_chunk_1 = 0; p_col_chunk_1 < 2; ++p_col_chunk_1) {
                  tl::ascend::brcb<float>(v1_p_tmp[(p_row_chunk_1 * 512)], v1_p_sum[(p_row_chunk_1 * 64)], 8, 1, 8);
                  AscendC::PipeBarrier<PIPE_V>();
                  tl::ascend::div_mask<float>(v1_p_bmm1[((p_row_chunk_1 * 8192) + (p_col_chunk_1 * 64))], v1_p_bmm1[((p_row_chunk_1 * 8192) + (p_col_chunk_1 * 64))], v1_p_tmp[(p_row_chunk_1 * 512)], 0xFFFFFFFFFFFFFFFF, 0, 64, 1, 1, 0, 16, 16, 1);
                }
              }
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::Fill<float>(v1_p_out[0], 0.000000e+00f, 128);
              AscendC::PipeBarrier<PIPE_V>();
              for (int32_t p_reduce_chunk = 0; p_reduce_chunk < 2; ++p_reduce_chunk) {
                tl::ascend::reduce_sum<float,  64,  128,  0>(v1_p_partial[0], v1_p_bmm1[(p_reduce_chunk * 8192)], tmp_ub[0], true);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(v1_p_out[0], v1_p_out[0], v1_p_partial[0], 128);
                AscendC::PipeBarrier<PIPE_V>();
              }
              {
              AscendC::Muls(v1_p_out[0], v1_p_out[0], 7.812500e-03f, 128);
              }
              AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
              AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
              tl::ascend::copy_ub_to_gm<float, 128>(PReduced[((((pipe_iter * 3072) + (cid * 1024)) + (n_idx * 128)) - 3072)], v1_p_out[0], 1024, 1, 128);
              AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
          }
          if (vid == 1) {
            tl::ascend::copy_gm_to_ub<bfloat16_t, 64>(v1_sy_weight_half[0], Weights[(((pipe_iter * 192) + (cid * 64)) - 192)], 64, 1, 64, bfloat16_t(0.000000e+00f));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
            AscendC::Cast(v1_sy_weight[0], v1_sy_weight_half[0], AscendC::RoundMode::CAST_NONE, 64);
            tl::ascend::Fill<float>(v1_sy_weight_2d[0], 1.000000e+00f, 8192);
            AscendC::PipeBarrier<PIPE_V>();
            for (int32_t sy_col_chunk = 0; sy_col_chunk < 2; ++sy_col_chunk) {
              tl::ascend::brcb<float>(v1_sy_tmp[0], v1_sy_weight[0], 8, 1, 8);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::mul_mask<float>(v1_sy_weight_2d[(sy_col_chunk * 64)], v1_sy_weight_2d[(sy_col_chunk * 64)], v1_sy_tmp[0], 0xFFFFFFFFFFFFFFFF, 0, 64, 1, 1, 0, 16, 16, 1);
            }
            tl::ascend::Fill<float>(v1_sy_max[0], -CUDART_INF_F, 1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
            for (int32_t n_idx_1 = 0; n_idx_1 < 8; ++n_idx_1) {
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
              tl::ascend::copy_gm_to_ub<float, 128, 64>(v1_sy_storage[0], bmm2Res[((((pipe_iter * 196608) + (cid * 65536)) + (n_idx_1 * 128)) - 196608)], 1024, 64, 128, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);
              AscendC::Mul(v1_sy_storage[0], v1_sy_storage[0], v1_sy_weight_2d[0], 8192);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::reduce_sum<float,  64,  128,  0>(v1_sy_reduce[0], v1_sy_storage[0], tmp_ub[0], true);
              tl::ascend::copy_ub_to_ub<float, float, 128>(v1_sy_storage[((n_idx_1 * 128) + 8192)], v1_sy_reduce[0], 1, 128, 128, 1, 128, 128);
              AscendC::PipeBarrier<PIPE_V>();
              {
              AscendC::Muls(v1_sy_broadcast[0], v1_sy_reduce[0], 1.000000e+00f, 128);
              }
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::reduce_max<float, 1, 128, -1>(v1_sy_tmp_max[0], v1_sy_broadcast[0], tmp_ub[0], true);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Max(v1_sy_max[0], v1_sy_max[0], v1_sy_tmp_max[0], 1);
              AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
            }
            tl::ascend::Fill<float>(v1_sy_floor[0], -1.000000e+20f, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Max(v1_sy_max[0], v1_sy_max[0], v1_sy_floor[0], 1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
            tl::ascend::Fill<float>(v1_sy_sum[0], 0.000000e+00f, 1);
            for (int32_t n_idx_2 = 0; n_idx_2 < 8; ++n_idx_2) {
              tl::ascend::Broadcast<float, 2, 1, false>(v1_sy_broadcast[0],v1_sy_max[0],tmp_ub,(uint32_t[]){1, 128}, (uint32_t[]){1, 1});
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Sub(v1_sy_storage[((n_idx_2 * 128) + 8192)], v1_sy_storage[((n_idx_2 * 128) + 8192)], v1_sy_broadcast[0], 128);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Exp(v1_sy_storage[((n_idx_2 * 128) + 8192)], v1_sy_storage[((n_idx_2 * 128) + 8192)], 128);
              AscendC::PipeBarrier<PIPE_V>();
              {
              AscendC::Muls(v1_sy_broadcast[0], v1_sy_storage[((n_idx_2 * 128) + 8192)], 1.000000e+00f, 128);
              }
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::reduce_sum<float,  1,  128,  -1>(v1_sy_tmp_sum[0], v1_sy_broadcast[0], tmp_ub[0], true);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Add(v1_sy_sum[0], v1_sy_sum[0], v1_sy_tmp_sum[0], 1);
            }
            tl::ascend::Fill<float>(v1_sy_eps[0], 1.000000e-08f, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Max(v1_sy_sum[0], v1_sy_sum[0], v1_sy_eps[0], 1);
            for (int32_t n_idx_3 = 0; n_idx_3 < 8; ++n_idx_3) {
              AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
              tl::ascend::Broadcast<float, 2, 1, false>(v1_sy_broadcast[0],v1_sy_sum[0],tmp_ub,(uint32_t[]){1, 128}, (uint32_t[]){1, 1});
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Div(v1_sy_storage[((n_idx_3 * 128) + 8192)], v1_sy_storage[((n_idx_3 * 128) + 8192)], v1_sy_broadcast[0], 128);
              AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);
              tl::ascend::copy_ub_to_gm<float, 128, 1>(SoftmaxI[((((pipe_iter * 3072) + (cid * 1024)) + (n_idx_3 * 128)) - 3072)], v1_sy_storage[((n_idx_3 * 128) + 8192)], 1024, 1, 128);
              AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
          }
          AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(10);
          AscendC::CrossCoreWaitFlag(12);
        }
        if ASCEND_IS_AIC {
          AscendC::CrossCoreWaitFlag(10);
          AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(12);
        }
        if ASCEND_IS_AIV {
          tl::ascend::Fill<float>(v1_bw_eps[0], 1.000000e-08f, 64);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(2);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
          tl::ascend::Fill<float>(v1_bw_dw_accum[0], 0.000000e+00f, 32);
          tl::ascend::Fill<float>(v1_bw_loss_accum[0], 0.000000e+00f, 1);
          tl::ascend::copy_gm_to_ub<bfloat16_t, 32>(v1_bw_weight_half[0], Weights[((((pipe_iter * 192) + (cid * 64)) + (vid * 32)) - 192)], 64, 1, 32, bfloat16_t(0.000000e+00f));
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
          AscendC::Cast(v1_bw_weight[0], v1_bw_weight_half[0], AscendC::RoundMode::CAST_NONE, 32);
          tl::ascend::Fill<float>(v1_bw_weight_2d[0], 1.000000e+00f, 2048);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::brcb<float>(v1_bw_row_tmp[0], v1_bw_weight[0], 4, 1, 8);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::mul_mask<float>(v1_bw_weight_2d[0], v1_bw_weight_2d[0], v1_bw_row_tmp[0], 0xFFFFFFFFFFFFFFFF, 0, 32, 1, 1, 0, 8, 8, 1);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
          for (int32_t n_idx_4 = 0; n_idx_4 < 8; ++n_idx_4) {
            for (int32_t bw_col_id = 0; bw_col_id < 2; ++bw_col_id) {
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
              tl::ascend::copy_gm_to_ub<float, 64>(v1_bw_p[0], PReduced[(((((pipe_iter * 3072) + (cid * 1024)) + (n_idx_4 * 128)) + (bw_col_id * 64)) - 3072)], 1024, 1, 64, 0.000000e+00f);
              tl::ascend::copy_gm_to_ub<float, 64>(v1_bw_sy[0], SoftmaxI[(((((pipe_iter * 3072) + (cid * 1024)) + (n_idx_4 * 128)) + (bw_col_id * 64)) - 3072)], 1024, 1, 64, 0.000000e+00f);
              tl::ascend::copy_gm_to_ub<float, 64, 32>(v1_bw_bmm2[0], bmm2Res[((((((pipe_iter * 196608) + (cid * 65536)) + (vid * 32768)) + (n_idx_4 * 128)) + (bw_col_id * 64)) - 196608)], 1024, 32, 64, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);
              AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(2);
              AscendC::Sub(v1_bw_diff[0], v1_bw_sy[0], v1_bw_p[0], 64);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::Broadcast<float, 2, 0, false>(v1_bw_diff_2d[0],v1_bw_diff[0],tmp_ub,(uint32_t[]){32, 64}, (uint32_t[]){1, 64});
              {
              AscendC::Muls(v1_bw_rg[0], v1_bw_bmm2[0], -1.000000e+04f, 2048);
              }
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Exp(v1_bw_rg[0], v1_bw_rg[0], 2048);
              AscendC::PipeBarrier<PIPE_V>();
              {
              AscendC::Muls(v1_bw_rg[0], v1_bw_rg[0], -1.000000e+00f, 2048);
              }
              AscendC::PipeBarrier<PIPE_V>();
              {
              AscendC::Adds(v1_bw_rg[0], v1_bw_rg[0], 1.000000e+00f, 2048);
              }
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Mul(v1_bw_bmm2[0], v1_bw_bmm2[0], v1_bw_diff_2d[0], 2048);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::reduce_sum<float,  32,  64,  -1>(v1_bw_dw[0], v1_bw_bmm2[0], tmp_ub[0], true);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Add(v1_bw_dw_accum[0], v1_bw_dw_accum[0], v1_bw_dw[0], 32);
              AscendC::Mul(v1_bw_bmm2[0], v1_bw_weight_2d[0], v1_bw_diff_2d[0], 2048);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Mul(v1_bw_bmm2[0], v1_bw_bmm2[0], v1_bw_rg[0], 2048);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Cast(v1_bw_rg_half[0], v1_bw_bmm2[0], AscendC::RoundMode::CAST_RINT, 2048);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(2);
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(2);
              AscendC::PipeBarrier<PIPE_MTE3>();
              tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 32>(reluGradRes[((((((pipe_iter * 196608) + (cid * 65536)) + (vid * 32768)) + (n_idx_4 * 128)) + (bw_col_id * 64)) - 196608)], v1_bw_rg_half[0], 1024, 32, 64);
              AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(2);
              if ((n_idx_4 % 2) == vid) {
                AscendC::Max(v1_bw_p[0], v1_bw_p[0], v1_bw_eps[0], 64);
                AscendC::Max(v1_bw_sy[0], v1_bw_sy[0], v1_bw_eps[0], 64);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Ln(v1_bw_log_p[0], v1_bw_p[0], 64);
                AscendC::Ln(v1_bw_log_sy[0], v1_bw_sy[0], 64);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Sub(v1_bw_loss_2d[0], v1_bw_log_p[0], v1_bw_log_sy[0], 64);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(v1_bw_loss_2d[0], v1_bw_loss_2d[0], v1_bw_p[0], 64);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::reduce_sum<float,  1,  64,  -1>(v1_bw_loss_sum[0], v1_bw_loss_2d[0], tmp_ub[0], true);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(v1_bw_loss_accum[0], v1_bw_loss_accum[0], v1_bw_loss_sum[0], 1);
              }
              AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
            }
          }
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
          AscendC::Cast(v1_bw_dw_half[0], v1_bw_dw_accum[0], AscendC::RoundMode::CAST_RINT, 32);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
          tl::ascend::copy_ub_to_gm<bfloat16_t, 32>(dWeightOut[((((pipe_iter * 192) + (cid * 64)) + (vid * 32)) - 192)], v1_bw_dw_half[0], 64, 1, 32);
          tl::ascend::copy_ub_to_gm<float, 1>(LossPartial[((((pipe_iter * 48) + (cid * 16)) + (vid * 8)) - 48)], v1_bw_loss_accum[0], 128, 1, 1);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(2);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
          AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(8);
        }
        if ASCEND_IS_AIC {
          tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(c2_qi_l1[0], QueryIndex[(((pipe_iter * 24576) + (cid * 8192)) - 24576)], 128, 64, 128, true);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(6);
          AscendC::CrossCoreWaitFlag(8);
          AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(2);
          AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2);
          AscendC::SetFlag<AscendC::HardEvent::FIX_M>(2);
          AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3);
          AscendC::SetFlag<AscendC::HardEvent::FIX_M>(3);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(6);
          for (int32_t n_idx_5 = 0; n_idx_5 < 8; ++n_idx_5) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(2);
            tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(c2_rg_l1[0], reluGradRes[((((pipe_iter * 196608) + (cid * 65536)) + (n_idx_5 * 128)) - 196608)], 1024, 64, 128, true);
            tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c2_gki_l1[0], GatherKeyIndex_ws[((((pipe_iter * 393216) + (cid * 131072)) + (n_idx_5 * 16384)) - 393216)], 128, 128, 128, true);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(2);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(2);
            tl::ascend::copy_l1_to_l0a<bfloat16_t, 64, 128, true>(c2_dk_rg_l0a[0], c2_rg_l1[0], 128, 64);
            tl::ascend::copy_l1_to_l0b<bfloat16_t, 64, 128, false>(c2_dk_qi_l0b[0], c2_qi_l1[0], 64, 128);
            tl::ascend::copy_l1_to_l0a<bfloat16_t, 64, 128, false>(c2_dq_rg_l0a[0], c2_rg_l1[0], 64, 128);
            tl::ascend::copy_l1_to_l0b<bfloat16_t, 128, 128, false>(c2_dq_gki_l0b[0], c2_gki_l1[0], 128, 128);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(2);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(2);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(2);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(2);
            tl::ascend::mma<bfloat16_t, float, 128, 128>(c2_dk_rg_l0a[0],c2_dk_qi_l0b[0],c2_dk_l0c[0], (bool)1, 64);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(2);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(2);
            tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 128, 0>(dk_matmul_partial[((((pipe_iter * 393216) + (cid * 131072)) + (n_idx_5 * 16384)) - 393216)], c2_dk_l0c[0], 128, 128, 128, 0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(3);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(3);
            tl::ascend::mma<bfloat16_t, float, 64, 128>(c2_dq_rg_l0a[0],c2_dq_gki_l0b[0],c2_dq_l0c[0], (n_idx_5 == 0), 128);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3);
            if (n_idx_5 < 7) {
              AscendC::SetFlag<AscendC::HardEvent::FIX_M>(3);
            }
          }
          AscendC::SetFlag<AscendC::HardEvent::M_FIX>(3);
          AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(3);
          tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 128, 0>(dQueryIndexAccum[(((pipe_iter * 24576) + (cid * 8192)) - 24576)], c2_dq_l0c[0], 128, 64, 128, 0);
          AscendC::SetFlag<AscendC::HardEvent::FIX_M>(3);
          AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(2);
          AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2);
          AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(2);
          AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3);
          AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(3);
        }
      }
    }
  }
  AscendC::SyncAll<false>();
  if ASCEND_IS_AIV {
    tl::ascend::Fill<float>(v2_zero_batch[0], 0.000000e+00f, 4096);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(5);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(5);
    for (int32_t clear_iter = 0; clear_iter < 8; ++clear_iter) {
      tl::ascend::copy_ub_to_gm<float, 128, 32>(dKeyIndexAccum[(((cid * 65536) + (vid * 32768)) + (clear_iter * 4096))], v2_zero_batch[0], 128, 32, 128);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(5);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(5);
  }
  AscendC::SyncAll<false>();
  if ASCEND_IS_AIV {
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(6);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(7);
    int32_t condval;
    if ((cid < 1)) {
      condval = 1;
    } else {
      condval = 0;
    }
    int32_t condval_1;
    if ((cid < 1)) {
      condval_1 = 1;
    } else {
      condval_1 = 0;
    }
    if (-11 < ((condval / 2) - (vid * (1 - (condval_1 % 2))))) {
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(6);
      int32_t condval_2;
      if ((cid < 1)) {
        condval_2 = 1;
      } else {
        condval_2 = 0;
      }
      int32_t condval_3;
      if ((cid < 1)) {
        condval_3 = 1;
      } else {
        condval_3 = 0;
      }
      int32_t condval_4;
      if ((cid < 1)) {
        condval_4 = 1;
      } else {
        condval_4 = 0;
      }
      int32_t condval_5;
      if ((cid < 1)) {
        condval_5 = 1;
      } else {
        condval_5 = 0;
      }
      tl::ascend::copy_gm_to_ub<float, 128, 16>(v2_dk_batch_ping[0], dk_matmul_partial[(((cid * 43008) + ((cid < 1 ? cid : 1) * 2048)) + ((vid * ((condval_2 / 2) + 11)) * 2048))], 128, ((1 <= (((64 - (cid < 1 ? cid : 1)) - (vid * ((condval_3 / 2) + 11))) - (cid * 21))) ? 16 : (((((cid * 21) + (cid < 1 ? cid : 1)) + (vid * ((condval_4 / 2) + 11))) < 64) ? (((1024 - ((cid < 1 ? cid : 1) * 16)) - ((vid * ((condval_5 / 2) + 11)) * 16)) - (cid * 336)) : 0)), 128, 0.000000e+00f);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(6);
      int32_t condval_6;
      if ((cid < 1)) {
        condval_6 = 1;
      } else {
        condval_6 = 0;
      }
      int32_t condval_7;
      if ((cid < 1)) {
        condval_7 = 1;
      } else {
        condval_7 = 0;
      }
      for (int32_t batch_id = 0; batch_id < (((condval_6 / 2) + 11) - (vid * (1 - (condval_7 % 2)))); ++batch_id) {
        int32_t condval_8;
        if ((cid < 1)) {
          condval_8 = 1;
        } else {
          condval_8 = 0;
        }
        int32_t condval_9;
        if ((cid < 1)) {
          condval_9 = 1;
        } else {
          condval_9 = 0;
        }
        if (batch_id < (((condval_8 / 2) + 10) - (vid * (1 - (condval_9 % 2))))) {
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((((batch_id + 1) % 2) + 6));
          if (((batch_id + 1) % 2) == 0) {
            int32_t condval_10;
            if ((cid < 1)) {
              condval_10 = 1;
            } else {
              condval_10 = 0;
            }
            int32_t condval_11;
            if ((cid < 1)) {
              condval_11 = 1;
            } else {
              condval_11 = 0;
            }
            int32_t condval_12;
            if ((cid < 1)) {
              condval_12 = 1;
            } else {
              condval_12 = 0;
            }
            int32_t condval_13;
            if ((cid < 1)) {
              condval_13 = 1;
            } else {
              condval_13 = 0;
            }
            tl::ascend::copy_gm_to_ub<float, 128, 16>(v2_dk_batch_ping[0], dk_matmul_partial[(((((cid * 43008) + (batch_id * 2048)) + ((cid < 1 ? cid : 1) * 2048)) + ((vid * ((condval_10 / 2) + 11)) * 2048)) + 2048)], 128, ((1 <= ((((63 - batch_id) - (cid < 1 ? cid : 1)) - (vid * ((condval_11 / 2) + 11))) - (cid * 21))) ? 16 : ((((((cid * 21) + (cid < 1 ? cid : 1)) + batch_id) + (vid * ((condval_12 / 2) + 11))) < 63) ? ((((1008 - (batch_id * 16)) - ((cid < 1 ? cid : 1) * 16)) - ((vid * ((condval_13 / 2) + 11)) * 16)) - (cid * 336)) : 0)), 128, 0.000000e+00f);
          } else {
            int32_t condval_14;
            if ((cid < 1)) {
              condval_14 = 1;
            } else {
              condval_14 = 0;
            }
            int32_t condval_15;
            if ((cid < 1)) {
              condval_15 = 1;
            } else {
              condval_15 = 0;
            }
            int32_t condval_16;
            if ((cid < 1)) {
              condval_16 = 1;
            } else {
              condval_16 = 0;
            }
            int32_t condval_17;
            if ((cid < 1)) {
              condval_17 = 1;
            } else {
              condval_17 = 0;
            }
            tl::ascend::copy_gm_to_ub<float, 128, 16>(v2_dk_batch_pong[0], dk_matmul_partial[(((((cid * 43008) + (batch_id * 2048)) + ((cid < 1 ? cid : 1) * 2048)) + ((vid * ((condval_14 / 2) + 11)) * 2048)) + 2048)], 128, ((1 <= ((((63 - batch_id) - (cid < 1 ? cid : 1)) - (vid * ((condval_15 / 2) + 11))) - (cid * 21))) ? 16 : ((((((cid * 21) + (cid < 1 ? cid : 1)) + batch_id) + (vid * ((condval_16 / 2) + 11))) < 63) ? ((((1008 - (batch_id * 16)) - ((cid < 1 ? cid : 1) * 16)) - ((vid * ((condval_17 / 2) + 11)) * 16)) - (cid * 336)) : 0)), 128, 0.000000e+00f);
          }
          AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>((((batch_id + 1) % 2) + 6));
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(((batch_id % 2) + 6));
        for (int32_t pair_idx = 0; pair_idx < 8; ++pair_idx) {
          int64_t condval_18;
          if ((((int64_t)cid) < (int64_t)1)) {
            condval_18 = (int64_t)1;
          } else {
            condval_18 = (int64_t)0;
          }
          int32_t sparse_idx0 = SparseIndices.GetValue((((((((int64_t)cid) * (int64_t)336) + (((int64_t)batch_id) * (int64_t)16)) + ((((int64_t)cid) < (int64_t)1 ? ((int64_t)cid) : (int64_t)1) * (int64_t)16)) + ((((int64_t)vid) * ((condval_18 / (int64_t)2) + (int64_t)11)) * (int64_t)16)) + (((int64_t)pair_idx) * (int64_t)2)));
          int64_t condval_19;
          if ((((int64_t)cid) < (int64_t)1)) {
            condval_19 = (int64_t)1;
          } else {
            condval_19 = (int64_t)0;
          }
          int32_t sparse_idx1 = SparseIndices.GetValue(((((((((int64_t)cid) * (int64_t)336) + (((int64_t)batch_id) * (int64_t)16)) + ((((int64_t)cid) < (int64_t)1 ? ((int64_t)cid) : (int64_t)1) * (int64_t)16)) + ((((int64_t)vid) * ((condval_19 / (int64_t)2) + (int64_t)11)) * (int64_t)16)) + (((int64_t)pair_idx) * (int64_t)2)) + (int64_t)1));
          int32_t condval_20;
          if ((cid < 1)) {
            condval_20 = 1;
          } else {
            condval_20 = 0;
          }
          if (((((cid * 21) + (cid < 1 ? cid : 1)) + batch_id) + (vid * ((condval_20 / 2) + 11))) < 64) {
            if (0 <= sparse_idx0) {
              if ((batch_id % 2) == 0) {
                tl::ascend::atomic_add_ub_to_gm<float, 128, 1>(dKeyIndexAccum[(sparse_idx0 * 128)], v2_dk_batch_ping[(pair_idx * 256)], 196608, ((sparse_idx0 <= 1535) ? 1 : 0), 128);
              } else {
                tl::ascend::atomic_add_ub_to_gm<float, 128, 1>(dKeyIndexAccum[(sparse_idx0 * 128)], v2_dk_batch_pong[(pair_idx * 256)], 196608, ((sparse_idx0 <= 1535) ? 1 : 0), 128);
              }
            }
          }
          int32_t condval_21;
          if ((cid < 1)) {
            condval_21 = 1;
          } else {
            condval_21 = 0;
          }
          if ((((((cid * 336) + (batch_id * 16)) + ((cid < 1 ? cid : 1) * 16)) + ((vid * ((condval_21 / 2) + 11)) * 16)) + (pair_idx * 2)) < 1023) {
            if (0 <= sparse_idx1) {
              if ((batch_id % 2) == 0) {
                tl::ascend::atomic_add_ub_to_gm<float, 128, 1>(dKeyIndexAccum[(sparse_idx1 * 128)], v2_dk_batch_ping[((pair_idx * 256) + 128)], 196608, ((sparse_idx1 <= 1535) ? 1 : 0), 128);
              } else {
                tl::ascend::atomic_add_ub_to_gm<float, 128, 1>(dKeyIndexAccum[(sparse_idx1 * 128)], v2_dk_batch_pong[((pair_idx * 256) + 128)], 196608, ((sparse_idx1 <= 1535) ? 1 : 0), 128);
              }
            }
          }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(((batch_id % 2) + 6));
      }
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(6);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(7);
  }
  AscendC::SyncAll<false>();
  if ASCEND_IS_AIV {
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(4);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(7);
    if (cid < 1) {
      for (int32_t dq_group_local = 0; dq_group_local < 32; ++dq_group_local) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(4);
        tl::ascend::copy_gm_to_ub<float, 128, 1>(v2_finalize_dq_fp32[0], dQueryIndexAccum[(((cid * 8192) + (dq_group_local * 256)) + (vid * 128))], 8192, 1, 128, 0.000000e+00f);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(7);
        AscendC::Cast(v2_finalize_dq_half[0], v2_finalize_dq_fp32[0], AscendC::RoundMode::CAST_RINT, 128);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(4);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(5);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(5);
        tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 1>(dQueryIndex[(((cid * 8192) + (dq_group_local * 256)) + (vid * 128))], v2_finalize_dq_half[0], 8192, 1, 128);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(7);
      }
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(4);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(7);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(5);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(6);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(5);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(6);
    for (int32_t finalize_iter = 0; finalize_iter < 16; ++finalize_iter) {
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(((finalize_iter % 2) + 5));
      tl::ascend::copy_gm_to_ub<float, 128, 16>(v2_finalize_dk_fp32[((finalize_iter % 2) * 2048)], dKeyIndexAccum[(((cid * 65536) + (vid * 32768)) + (finalize_iter * 2048))], 128, 16, 128, 0.000000e+00f);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(((finalize_iter % 2) + 5));
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(((finalize_iter % 2) + 5));
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(((finalize_iter % 2) + 5));
      AscendC::Cast(v2_finalize_dk_half[((finalize_iter % 2) * 2048)], v2_finalize_dk_fp32[((finalize_iter % 2) * 2048)], AscendC::RoundMode::CAST_RINT, 2048);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(((finalize_iter % 2) + 5));
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(((finalize_iter % 2) + 5));
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(((finalize_iter % 2) + 5));
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 16>(dKeyIndexOut[(((cid * 65536) + (vid * 32768)) + (finalize_iter * 2048))], v2_finalize_dk_half[((finalize_iter % 2) * 2048)], 128, 16, 128);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(((finalize_iter % 2) + 5));
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(5);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(6);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(5);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(6);
    if (cid == 0) {
      if (vid == 0) {
        tl::ascend::Fill<float>(v2_loss_total[0], 0.000000e+00f, 1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(4);
        tl::ascend::copy_gm_to_ub<float, 128, 1>(v2_loss_chunk[0], LossPartial[0], 128, 1, 128, 0.000000e+00f);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(4);
        tl::ascend::reduce_sum<float,  1,  128,  -1>(v2_loss_chunk_sum[0], v2_loss_chunk[0], tmp_ub[0], true);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(v2_loss_total[0], v2_loss_total[0], v2_loss_chunk_sum[0], 1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(4);
        tl::ascend::copy_ub_to_gm<float, 1>(LossOut[0], v2_loss_total[0], 1, 1, 1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(4);
      }
    }
  }
}

void main_kernel_tiling() {
}

extern "C" void call(uint8_t* SparseIndices_handle, uint8_t* KRealSize_handle, uint8_t* ValidKMask_handle, uint8_t* V0ZeroRow_handle, uint8_t* Query_handle, uint8_t* QueryRope_handle, uint8_t* Key_handle, uint8_t* KeyRope_handle, uint8_t* QueryIndex_handle, uint8_t* KeyIndex_handle, uint8_t* Weights_handle, uint8_t* SoftmaxMax_handle, uint8_t* SoftmaxSum_handle, uint8_t* dQueryIndexAccum_handle, uint8_t* dKeyIndexAccum_handle, uint8_t* GatherKey_ws_handle, uint8_t* GatherKeyRope_ws_handle, uint8_t* GatherKeyIndex_ws_handle, uint8_t* bmm1Res_handle, uint8_t* bmm2Res_handle, uint8_t* PReduced_handle, uint8_t* SoftmaxI_handle, uint8_t* dk_matmul_partial_handle, uint8_t* reluGradRes_handle, uint8_t* LossPartial_handle, uint8_t* C2ZeroPad_handle, uint8_t* dQueryIndex_handle, uint8_t* dKeyIndexOut_handle, uint8_t* dWeightOut_handle, uint8_t* LossOut_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<3, nullptr, stream>>>(SparseIndices_handle, KRealSize_handle, ValidKMask_handle, V0ZeroRow_handle, Query_handle, QueryRope_handle, Key_handle, KeyRope_handle, QueryIndex_handle, KeyIndex_handle, Weights_handle, SoftmaxMax_handle, SoftmaxSum_handle, dQueryIndexAccum_handle, dKeyIndexAccum_handle, GatherKey_ws_handle, GatherKeyRope_ws_handle, GatherKeyIndex_ws_handle, bmm1Res_handle, bmm2Res_handle, PReduced_handle, SoftmaxI_handle, dk_matmul_partial_handle, reluGradRes_handle, LossPartial_handle, C2ZeroPad_handle, dQueryIndex_handle, dKeyIndexOut_handle, dWeightOut_handle, LossOut_handle, fftsAddr);
}
