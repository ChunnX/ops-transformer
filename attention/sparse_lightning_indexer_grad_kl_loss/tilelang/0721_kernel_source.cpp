#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR SparseIndices_handle,  GM_ADDR KRealSize_handle,  GM_ADDR V0ZeroRow_handle,  GM_ADDR Query_handle,  GM_ADDR QueryRope_handle,  GM_ADDR Key_handle,  GM_ADDR KeyRope_handle,  GM_ADDR QueryIndex_handle,  GM_ADDR KeyIndex_handle,  GM_ADDR Weights_handle,  GM_ADDR SoftmaxMax_handle,  GM_ADDR SoftmaxSum_handle,  GM_ADDR dQueryIndexAccum_handle,  GM_ADDR dKeyIndexAccum_handle,  GM_ADDR GatherKey_ws_handle,  GM_ADDR GatherKeyRope_ws_handle,  GM_ADDR GatherKeyIndex_ws_handle,  GM_ADDR bmm1Res_handle,  GM_ADDR bmm2Res_handle,  GM_ADDR PReduced_handle,  GM_ADDR V1Stats_handle,  GM_ADDR DWeightPartial_handle,  GM_ADDR dk_matmul_partial_handle,  GM_ADDR reluGradRes_handle,  GM_ADDR LossPartial_handle,  GM_ADDR C2ZeroPad_handle,  GM_ADDR dQueryIndex_handle,  GM_ADDR dKeyIndexOut_handle,  GM_ADDR dWeightOut_handle,  GM_ADDR LossOut_handle, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<int> SparseIndices;
  SparseIndices.SetGlobalBuffer((__gm__ int*)SparseIndices_handle);
  AscendC::GlobalTensor<int> KRealSize;
  KRealSize.SetGlobalBuffer((__gm__ int*)KRealSize_handle);
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
  AscendC::GlobalTensor<float> V1Stats;
  V1Stats.SetGlobalBuffer((__gm__ float*)V1Stats_handle);
  AscendC::GlobalTensor<float> DWeightPartial;
  DWeightPartial.SetGlobalBuffer((__gm__ float*)DWeightPartial_handle);
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
  auto tmp_ub = ascend_ub.GetWithOffset<uint8_t>(32768, 121344);
  auto v0_key_ub = ascend_ub.GetWithOffset<bfloat16_t>(16384, 0);
  auto v0_ki_ub = ascend_ub.GetWithOffset<bfloat16_t>(4096, 32768);
  auto v0_rope_ub = ascend_ub.GetWithOffset<bfloat16_t>(4096, 40960);
  auto v1_op_offsets = ascend_ub.GetWithOffset<float>(128, 49152);
  auto v1_om_offsets = ascend_ub.GetWithOffset<float>(128, 154112);
  auto v1_os_offsets = ascend_ub.GetWithOffset<float>(128, 190944);
  auto v1_ob_offsets = ascend_ub.GetWithOffset<float>(64, 195584);
  auto v1_ob_eps = ascend_ub.GetWithOffset<float>(64, 195840);
  auto c2_qi_l1 = ascend_l1.GetWithOffset<bfloat16_t>(8192, 245760);
  auto c1_query_chunk_0_l1 = ascend_l1.GetWithOffset<bfloat16_t>(16384, 0);
  auto c1_query_chunk_1_l1 = ascend_l1.GetWithOffset<bfloat16_t>(16384, 32768);
  auto c1_query_chunk_2_l1 = ascend_l1.GetWithOffset<bfloat16_t>(16384, 65536);
  auto c1_query_chunk_3_l1 = ascend_l1.GetWithOffset<bfloat16_t>(16384, 98304);
  auto c1_query_rope_cache_l1 = ascend_l1.GetWithOffset<bfloat16_t>(16384, 131072);
  auto c1_qi_l1 = ascend_l1.GetWithOffset<bfloat16_t>(8192, 163840);
  auto c1_key_l1 = ascend_l1.GetWithOffset<bfloat16_t>(16384, 180224);
  auto c1_q_l0a = ascend_l0a.GetWithOffset<bfloat16_t>(16384, 0);
  auto c1_key_l0b = ascend_l0b.GetWithOffset<bfloat16_t>(16384, 0);
  auto c1_p_l0c = ascend_l0c.GetWithOffset<float>(16384, 0);
  auto c1_qi_l0a = ascend_l0a.GetWithOffset<bfloat16_t>(8192, 32768);
  auto c1_ki_l1 = ascend_l1.GetWithOffset<bfloat16_t>(16384, 212992);
  auto c1_ki_l0b = ascend_l0b.GetWithOffset<bfloat16_t>(16384, 32768);
  auto c1_sy_l0c = ascend_l0c.GetWithOffset<float>(8192, 65536);
  auto v1_op_max = ascend_ub.GetWithOffset<float>(128, 49664);
  auto v1_op_sum = ascend_ub.GetWithOffset<float>(128, 50176);
  auto v1_op_partial = ascend_ub.GetWithOffset<float>(128, 50688);
  auto v1_op_bmm1 = ascend_ub.GetWithOffset<float>(16384, 51200);
  auto v1_op_tmp = ascend_ub.GetWithOffset<float>(1024, 116736);
  auto v1_op_out = ascend_ub.GetWithOffset<float>(128, 120832);
  auto v1_om_weight_half = ascend_ub.GetWithOffset<bfloat16_t>(64, 154624);
  auto v1_om_weight = ascend_ub.GetWithOffset<float>(64, 154752);
  auto v1_om_weight_2d = ascend_ub.GetWithOffset<float>(8192, 155008);
  auto v1_om_tmp = ascend_ub.GetWithOffset<float>(512, 187776);
  auto v1_om_local_max = ascend_ub.GetWithOffset<float>(1, 189824);
  auto v1_om_bmm2 = ascend_ub.GetWithOffset<float>(8192, 0);
  auto v1_om_score = ascend_ub.GetWithOffset<float>(128, 189856);
  auto v1_om_reduce_2d = ascend_ub.GetWithOffset<float>(128, 190368);
  auto v1_om_tile_max = ascend_ub.GetWithOffset<float>(1, 190880);
  auto v1_om_floor = ascend_ub.GetWithOffset<float>(1, 190912);
  auto v1_os_stats = ascend_ub.GetWithOffset<float>(16, 191456);
  auto v1_os_weight_half = ascend_ub.GetWithOffset<bfloat16_t>(64, 191520);
  auto v1_os_global_max = ascend_ub.GetWithOffset<float>(1, 191648);
  auto v1_os_weight = ascend_ub.GetWithOffset<float>(64, 191680);
  auto v1_os_weight_2d = ascend_ub.GetWithOffset<float>(8192, 0);
  auto v1_os_tmp = ascend_ub.GetWithOffset<float>(512, 191936);
  auto v1_os_global_max_2d = ascend_ub.GetWithOffset<float>(128, 193984);
  auto v1_os_local_sum = ascend_ub.GetWithOffset<float>(1, 194496);
  auto v1_os_bmm2 = ascend_ub.GetWithOffset<float>(8192, 51200);
  auto v1_os_score = ascend_ub.GetWithOffset<float>(128, 194528);
  auto v1_os_reduce_2d = ascend_ub.GetWithOffset<float>(128, 195040);
  auto v1_os_tile_sum = ascend_ub.GetWithOffset<float>(1, 195552);
  auto v1_ob_weight_half = ascend_ub.GetWithOffset<bfloat16_t>(64, 196096);
  auto v1_ob_stats = ascend_ub.GetWithOffset<float>(32, 196224);
  auto v1_ob_weight = ascend_ub.GetWithOffset<float>(64, 32768);
  auto v1_ob_weight_2d = ascend_ub.GetWithOffset<float>(4096, 33024);
  auto v1_ob_row_tmp = ascend_ub.GetWithOffset<float>(512, 51200);
  auto v1_ob_global_max = ascend_ub.GetWithOffset<float>(1, 32768);
  auto v1_ob_global_sum = ascend_ub.GetWithOffset<float>(1, 32800);
  auto v1_ob_global_max_2d = ascend_ub.GetWithOffset<float>(64, 49408);
  auto v1_ob_global_sum_2d = ascend_ub.GetWithOffset<float>(64, 51200);
  auto v1_ob_dw_accum = ascend_ub.GetWithOffset<float>(64, 32768);
  auto v1_ob_loss_accum = ascend_ub.GetWithOffset<float>(1, 51456);
  auto v1_ob_p = ascend_ub.GetWithOffset<float>(64, 51488);
  auto v1_ob_bmm2 = ascend_ub.GetWithOffset<float>(4096, 51744);
  auto v1_ob_work = ascend_ub.GetWithOffset<float>(4096, 68128);
  auto v1_ob_sy = ascend_ub.GetWithOffset<float>(64, 84512);
  auto v1_ob_diff = ascend_ub.GetWithOffset<float>(64, 84768);
  auto v1_ob_diff_2d = ascend_ub.GetWithOffset<float>(4096, 85024);
  auto v1_ob_gate = ascend_ub.GetWithOffset<float>(4096, 101408);
  auto v1_ob_dw = ascend_ub.GetWithOffset<float>(64, 84768);
  auto v1_ob_loss_p = ascend_ub.GetWithOffset<float>(64, 68128);
  auto v1_ob_log_p = ascend_ub.GetWithOffset<float>(64, 68384);
  auto v1_ob_log_sy = ascend_ub.GetWithOffset<float>(64, 68128);
  auto v1_ob_loss_2d = ascend_ub.GetWithOffset<float>(64, 68640);
  auto v1_ob_loss_sum = ascend_ub.GetWithOffset<float>(1, 51488);
  auto v1_ob_rg_half = ascend_ub.GetWithOffset<bfloat16_t>(4096, 68128);
  auto v1_fdw_left = ascend_ub.GetWithOffset<float>(32, 32768);
  auto v1_fdw_right = ascend_ub.GetWithOffset<float>(32, 32896);
  auto v1_fdw_half = ascend_ub.GetWithOffset<bfloat16_t>(32, 32896);
  auto c2_dq_l0c = ascend_l0c.GetWithOffset<float>(8192, 98304);
  auto c2_rg_l1 = ascend_l1.GetWithOffset<bfloat16_t>(8192, 262144);
  auto c2_gki_l1 = ascend_l1.GetWithOffset<bfloat16_t>(16384, 278528);
  auto c2_dk_rg_l0a = ascend_l0a.GetWithOffset<bfloat16_t>(8192, 49152);
  auto c2_dk_qi_l0b = ascend_l0b.GetWithOffset<bfloat16_t>(8192, 0);
  auto c2_dq_rg_l0a = ascend_l0a.GetWithOffset<bfloat16_t>(8192, 0);
  auto c2_dq_gki_l0b = ascend_l0b.GetWithOffset<bfloat16_t>(16384, 16384);
  auto c2_dk_l0c = ascend_l0c.GetWithOffset<float>(16384, 0);
  auto v2_zero_batch = ascend_ub.GetWithOffset<float>(4096, 0);
  auto v2_dk_batch_ping = ascend_ub.GetWithOffset<float>(2048, 16384);
  auto v2_dk_batch_pong = ascend_ub.GetWithOffset<float>(2048, 24576);
  auto v2_finalize_dq_fp32 = ascend_ub.GetWithOffset<float>(128, 32768);
  auto v2_finalize_dq_half = ascend_ub.GetWithOffset<bfloat16_t>(128, 33280);
  auto v2_finalize_dk_fp32 = ascend_ub.GetWithOffset<float>(4096, 24576);
  auto v2_finalize_dk_half = ascend_ub.GetWithOffset<bfloat16_t>(4096, 40960);
  auto v2_loss_total = ascend_ub.GetWithOffset<float>(1, 16384);
  auto v2_loss_chunk = ascend_ub.GetWithOffset<float>(128, 16416);
  auto v2_loss_chunk_sum = ascend_ub.GetWithOffset<float>(1, 16928);
  auto vid = AscendC::GetSubBlockIdx();
  for (int32_t pipe_iter = 0; pipe_iter < 87; ++pipe_iter) {
    if (pipe_iter < 86) {
      if (((pipe_iter * 3) + (cid / 8)) < 256) {
        if ASCEND_IS_AIV {
          AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
          for (int32_t lane = 0; lane < 16; ++lane) {
            int32_t sparse_idx = SparseIndices.GetValue(((((pipe_iter * 24576) + (cid * 1024)) + (vid * 512)) + lane));
            if (0 <= sparse_idx) {
              tl::ascend::copy_gm_to_ub<bfloat16_t, 512, 1>(v0_key_ub[(lane * 512)], Key[(((((pipe_iter * 3) + (cid / 8)) / 64) * 524288) + (sparse_idx * 512))], 2097152, 1, 512, bfloat16_t(0.000000e+00f));
              tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 1>(v0_ki_ub[(lane * 128)], KeyIndex[(((((pipe_iter * 3) + (cid / 8)) / 64) * 131072) + (sparse_idx * 128))], 524288, 1, 128, bfloat16_t(0.000000e+00f));
              tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 1>(v0_rope_ub[(lane * 128)], KeyRope[(((((pipe_iter * 3) + (cid / 8)) / 64) * 131072) + (sparse_idx * 128))], 524288, 1, 128, bfloat16_t(0.000000e+00f));
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
                int32_t sparse_idx_1 = SparseIndices.GetValue(((((((pipe_iter * 24576) + (cid * 1024)) + (vid * 512)) + (chunk_id * 16)) + lane_1) + 16));
                if (0 <= sparse_idx_1) {
                  tl::ascend::copy_gm_to_ub<bfloat16_t, 512, 1>(v0_key_ub[((((chunk_id + 1) % 2) * 8192) + (lane_1 * 512))], Key[(((((pipe_iter * 3) + (cid / 8)) / 64) * 524288) + (sparse_idx_1 * 512))], 2097152, 1, 512, bfloat16_t(0.000000e+00f));
                  tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 1>(v0_ki_ub[((((chunk_id + 1) % 2) * 2048) + (lane_1 * 128))], KeyIndex[(((((pipe_iter * 3) + (cid / 8)) / 64) * 131072) + (sparse_idx_1 * 128))], 524288, 1, 128, bfloat16_t(0.000000e+00f));
                  tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 1>(v0_rope_ub[((((chunk_id + 1) % 2) * 2048) + (lane_1 * 128))], KeyRope[(((((pipe_iter * 3) + (cid / 8)) / 64) * 131072) + (sparse_idx_1 * 128))], 524288, 1, 128, bfloat16_t(0.000000e+00f));
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
            tl::ascend::copy_ub_to_gm<bfloat16_t, 512, 16>(GatherKey_ws[((((pipe_iter * 12582912) + (cid * 524288)) + (vid * 262144)) + (chunk_id * 8192))], v0_key_ub[((chunk_id % 2) * 8192)], 512, 16, 512);
            tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 16>(GatherKeyIndex_ws[((((pipe_iter * 3145728) + (cid * 131072)) + (vid * 65536)) + (chunk_id * 2048))], v0_ki_ub[((chunk_id % 2) * 2048)], 128, 16, 128);
            tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 16>(GatherKeyRope_ws[((((pipe_iter * 3145728) + (cid * 131072)) + (vid * 65536)) + (chunk_id * 2048))], v0_rope_ub[((chunk_id % 2) * 2048)], 128, 16, 128);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((chunk_id % 2));
          }
          if ((pipe_iter % 2) == 0) {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(2);
          } else {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(1);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(3);
          }
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
        }
      }
    }
    if (1 <= pipe_iter) {
      if (((pipe_iter * 3) + (cid / 8)) < 259) {
        if ASCEND_IS_AIC {
          AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
          AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
          AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
          AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
          AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
          AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_query_chunk_0_l1[0], Query[(((pipe_iter * 1572864) + (cid * 65536)) - 1572864)], 512, 128, 128, true);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_query_chunk_1_l1[0], Query[(((pipe_iter * 1572864) + (cid * 65536)) - 1572736)], 512, 128, 128, true);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_query_chunk_2_l1[0], Query[(((pipe_iter * 1572864) + (cid * 65536)) - 1572608)], 512, 128, 128, true);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_query_chunk_3_l1[0], Query[(((pipe_iter * 1572864) + (cid * 65536)) - 1572480)], 512, 128, 128, true);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_query_rope_cache_l1[0], QueryRope[(((pipe_iter * 393216) + (cid * 16384)) - 393216)], 128, 128, 128, true);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(4);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(4);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(c1_qi_l1[0], QueryIndex[(((pipe_iter * 196608) + (cid * 8192)) - 196608)], 128, 64, 128, true);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(5);
          if (((pipe_iter + 1) % 2) == 0) {
            AscendC::CrossCoreWaitFlag(0);
          } else {
            AscendC::CrossCoreWaitFlag(1);
          }
          for (int32_t k_inner = 0; k_inner < 8; ++k_inner) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
            tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_key_l1[0], GatherKey_ws[((((pipe_iter * 12582912) + (cid * 524288)) + (k_inner * 65536)) - 12582912)], 512, 128, 128, true);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
            for (int32_t d_idx = 0; d_idx < 4; ++d_idx) {
              AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
              if (d_idx == 0) {
                tl::ascend::copy_l1_to_l0a<bfloat16_t, 128, 128, false>(c1_q_l0a[0], c1_query_chunk_0_l1[0], 128, 128);
              }
              if (d_idx == 1) {
                tl::ascend::copy_l1_to_l0a<bfloat16_t, 128, 128, false>(c1_q_l0a[0], c1_query_chunk_1_l1[0], 128, 128);
              }
              if (d_idx == 2) {
                tl::ascend::copy_l1_to_l0a<bfloat16_t, 128, 128, false>(c1_q_l0a[0], c1_query_chunk_2_l1[0], 128, 128);
              }
              if (d_idx == 3) {
                tl::ascend::copy_l1_to_l0a<bfloat16_t, 128, 128, false>(c1_q_l0a[0], c1_query_chunk_3_l1[0], 128, 128);
              }
              tl::ascend::copy_l1_to_l0b<bfloat16_t, 128, 128, true>(c1_key_l0b[0], c1_key_l1[0], 128, 128);
              AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
              AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
              if (d_idx < 3) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
                tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_key_l1[0], GatherKey_ws[(((((pipe_iter * 12582912) + (cid * 524288)) + (k_inner * 65536)) + (d_idx * 128)) - 12582784)], 512, 128, 128, true);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
              }
              if (d_idx == 3) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
                tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_key_l1[0], GatherKeyRope_ws[((((pipe_iter * 3145728) + (cid * 131072)) + (k_inner * 16384)) - 3145728)], 128, 128, 128, true);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
              }
              AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
              AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
              tl::ascend::mma<bfloat16_t, float, 128, 128>(c1_q_l0a[0],c1_key_l0b[0],c1_p_l0c[0], (d_idx == 0), 128);
              AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
              AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
            }
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
            tl::ascend::copy_l1_to_l0a<bfloat16_t, 128, 128, false>(c1_q_l0a[0], c1_query_rope_cache_l1[0], 128, 128);
            tl::ascend::copy_l1_to_l0b<bfloat16_t, 128, 128, true>(c1_key_l0b[0], c1_key_l1[0], 128, 128);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
            tl::ascend::mma<bfloat16_t, float, 128, 128>(c1_q_l0a[0],c1_key_l0b[0],c1_p_l0c[0], (bool)0, 128);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
            tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 128, 0>(bmm1Res[((((pipe_iter * 3145728) + (cid * 131072)) + (k_inner * 128)) - 3145728)], c1_p_l0c[0], 1024, 128, 128, 0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
          }
          AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
          AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
          AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
          if (((pipe_iter + 1) % 2) == 0) {
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(4);
          } else {
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(5);
          }
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(5);
          AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
          tl::ascend::copy_l1_to_l0a<bfloat16_t, 64, 128, false>(c1_qi_l0a[0], c1_qi_l1[0], 64, 128);
          AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
          if (((pipe_iter + 1) % 2) == 0) {
            AscendC::CrossCoreWaitFlag(2);
          } else {
            AscendC::CrossCoreWaitFlag(3);
          }
          for (int32_t k_inner_1 = 0; k_inner_1 < 8; ++k_inner_1) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
            tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_ki_l1[0], GatherKeyIndex_ws[((((pipe_iter * 3145728) + (cid * 131072)) + (k_inner_1 * 16384)) - 3145728)], 128, 128, 128, true);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(1);
            tl::ascend::copy_l1_to_l0b<bfloat16_t, 128, 128, true>(c1_ki_l0b[0], c1_ki_l1[0], 128, 128);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(1);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
            tl::ascend::mma<bfloat16_t, float, 64, 128>(c1_qi_l0a[0],c1_ki_l0b[0],c1_sy_l0c[0], (bool)1, 128);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(1);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(1);
            tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 128, 1>(bmm2Res[((((pipe_iter * 1572864) + (cid * 65536)) + (k_inner_1 * 128)) - 1572864)], c1_sy_l0c[0], 1024, 64, 128, 0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
          }
          AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
          AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
          AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
          if (((pipe_iter + 1) % 2) == 0) {
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(6);
          } else {
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(7);
          }
        }
        if ASCEND_IS_AIV {
          if (((pipe_iter + 1) % 2) == 0) {
            AscendC::CrossCoreWaitFlag(4);
          } else {
            AscendC::CrossCoreWaitFlag(5);
          }
          tl::ascend::ArithProgression<float>(v1_op_offsets[0], 0.000000e+00f, 1.000000e+00f, 128);
          tl::ascend::copy_gm_to_ub<float, 128>(v1_op_max[0], SoftmaxMax[(((pipe_iter * 3072) + (cid * 128)) - 3072)], 262144, 1, 128, 0.000000e+00f);
          tl::ascend::copy_gm_to_ub<float, 128>(v1_op_sum[0], SoftmaxSum[(((pipe_iter * 3072) + (cid * 128)) - 3072)], 262144, 1, 128, 0.000000e+00f);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
          AscendC::Abs(v1_op_sum[0], v1_op_sum[0], 128);
          tl::ascend::Fill<float>(v1_op_partial[0], 1.000000e-08f, 128);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Max(v1_op_sum[0], v1_op_sum[0], v1_op_partial[0], 128);
          for (int32_t local_n = 0; local_n < 4; ++local_n) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            tl::ascend::copy_gm_to_ub<float, 128, 128>(v1_op_bmm1[0], bmm1Res[(((((pipe_iter * 3145728) + (cid * 131072)) + (vid * 512)) + (local_n * 128)) - 3145728)], 1024, 128, 128, 0.000000e+00f);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
            {
            AscendC::Muls(v1_op_bmm1[0], v1_op_bmm1[0], 4.419417e-02f, 16384);
            }
            AscendC::PipeBarrier<PIPE_V>();
            for (int32_t p_row_chunk = 0; p_row_chunk < 2; ++p_row_chunk) {
              for (int32_t p_col_chunk = 0; p_col_chunk < 2; ++p_col_chunk) {
                tl::ascend::brcb<float>(v1_op_tmp[(p_row_chunk * 512)], v1_op_max[(p_row_chunk * 64)], 8, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::sub_mask<float>(v1_op_bmm1[((p_row_chunk * 8192) + (p_col_chunk * 64))], v1_op_bmm1[((p_row_chunk * 8192) + (p_col_chunk * 64))], v1_op_tmp[(p_row_chunk * 512)], 0xFFFFFFFFFFFFFFFF, 0, 64, 1, 1, 0, 16, 16, 1);
              }
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(v1_op_bmm1[0], v1_op_bmm1[0], 16384);
            AscendC::PipeBarrier<PIPE_V>();
            for (int32_t p_row_chunk_1 = 0; p_row_chunk_1 < 2; ++p_row_chunk_1) {
              for (int32_t p_col_chunk_1 = 0; p_col_chunk_1 < 2; ++p_col_chunk_1) {
                tl::ascend::brcb<float>(v1_op_tmp[(p_row_chunk_1 * 512)], v1_op_sum[(p_row_chunk_1 * 64)], 8, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
                tl::ascend::div_mask<float>(v1_op_bmm1[((p_row_chunk_1 * 8192) + (p_col_chunk_1 * 64))], v1_op_bmm1[((p_row_chunk_1 * 8192) + (p_col_chunk_1 * 64))], v1_op_tmp[(p_row_chunk_1 * 512)], 0xFFFFFFFFFFFFFFFF, 0, 64, 1, 1, 0, 16, 16, 1);
              }
            }
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::Fill<float>(v1_op_out[0], 0.000000e+00f, 128);
            AscendC::PipeBarrier<PIPE_V>();
            for (int32_t p_reduce_chunk = 0; p_reduce_chunk < 2; ++p_reduce_chunk) {
              tl::ascend::reduce_sum<float,  64,  128,  0>(v1_op_partial[0], v1_op_bmm1[(p_reduce_chunk * 8192)], tmp_ub[0], true);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Add(v1_op_out[0], v1_op_out[0], v1_op_partial[0], 128);
              AscendC::PipeBarrier<PIPE_V>();
            }
            {
            AscendC::Muls(v1_op_out[0], v1_op_out[0], 7.812500e-03f, 128);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            tl::ascend::copy_ub_to_gm<float, 128>(PReduced[(((((pipe_iter * 24576) + (cid * 1024)) + (vid * 512)) + (local_n * 128)) - 24576)], v1_op_out[0], 2097152, 1, 128);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
          }
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        }
        if ASCEND_IS_AIV {
          if (((pipe_iter + 1) % 2) == 0) {
            AscendC::CrossCoreWaitFlag(6);
          } else {
            AscendC::CrossCoreWaitFlag(7);
          }
          tl::ascend::ArithProgression<float>(v1_om_offsets[0], 0.000000e+00f, 1.000000e+00f, 128);
          tl::ascend::copy_gm_to_ub<bfloat16_t, 64>(v1_om_weight_half[0], Weights[(((pipe_iter * 1536) + (cid * 64)) - 1536)], 131072, 1, 64, bfloat16_t(0.000000e+00f));
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
          AscendC::Cast(v1_om_weight[0], v1_om_weight_half[0], AscendC::RoundMode::CAST_NONE, 64);
          tl::ascend::Fill<float>(v1_om_weight_2d[0], 1.000000e+00f, 8192);
          AscendC::PipeBarrier<PIPE_V>();
          for (int32_t sy_col_chunk = 0; sy_col_chunk < 2; ++sy_col_chunk) {
            tl::ascend::brcb<float>(v1_om_tmp[0], v1_om_weight[0], 8, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::mul_mask<float>(v1_om_weight_2d[(sy_col_chunk * 64)], v1_om_weight_2d[(sy_col_chunk * 64)], v1_om_tmp[0], 0xFFFFFFFFFFFFFFFF, 0, 64, 1, 1, 0, 16, 16, 1);
          }
          tl::ascend::Fill<float>(v1_om_local_max[0], -CUDART_INF_F, 1);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
          for (int32_t local_n_1 = 0; local_n_1 < 4; ++local_n_1) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
            tl::ascend::copy_gm_to_ub<float, 128, 64>(v1_om_bmm2[0], bmm2Res[(((((pipe_iter * 1572864) + (cid * 65536)) + (vid * 512)) + (local_n_1 * 128)) - 1572864)], 1024, 64, 128, 0.000000e+00f);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);
            AscendC::Mul(v1_om_bmm2[0], v1_om_bmm2[0], v1_om_weight_2d[0], 8192);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::reduce_sum<float,  64,  128,  0>(v1_om_score[0], v1_om_bmm2[0], tmp_ub[0], true);
            AscendC::PipeBarrier<PIPE_V>();
            {
            AscendC::Muls(v1_om_reduce_2d[0], v1_om_score[0], 1.000000e+00f, 128);
            }
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::reduce_max<float, 1, 128, -1>(v1_om_tile_max[0], v1_om_reduce_2d[0], tmp_ub[0], true);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Max(v1_om_local_max[0], v1_om_local_max[0], v1_om_tile_max[0], 1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
          }
          tl::ascend::Fill<float>(v1_om_floor[0], -1.000000e+20f, 1);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Max(v1_om_local_max[0], v1_om_local_max[0], v1_om_floor[0], 1);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);
          tl::ascend::copy_ub_to_gm<float, 1>(V1Stats[((((pipe_iter * 768) + (cid * 32)) + (vid * 8)) - 768)], v1_om_local_max[0], 65536, 1, 1);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
          if (((pipe_iter + 1) % 2) == 0) {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(10);
            AscendC::CrossCoreWaitFlag(12);
          } else {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(11);
            AscendC::CrossCoreWaitFlag(13);
          }
        }
        if ASCEND_IS_AIC {
          if (((pipe_iter + 1) % 2) == 0) {
            AscendC::CrossCoreWaitFlag(10);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(12);
          } else {
            AscendC::CrossCoreWaitFlag(11);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(13);
          }
        }
        if ASCEND_IS_AIV {
          tl::ascend::ArithProgression<float>(v1_os_offsets[0], 0.000000e+00f, 1.000000e+00f, 128);
          tl::ascend::copy_gm_to_ub<float, 8, 2>(v1_os_stats[0], V1Stats[(((pipe_iter * 768) + (cid * 32)) - 768)], 8, 2, 8, 0.000000e+00f);
          tl::ascend::copy_gm_to_ub<bfloat16_t, 64>(v1_os_weight_half[0], Weights[(((pipe_iter * 1536) + (cid * 64)) - 1536)], 131072, 1, 64, bfloat16_t(0.000000e+00f));
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
          AscendC::Max(v1_os_global_max[0], v1_os_stats[0], v1_os_stats[8], 1);
          AscendC::Cast(v1_os_weight[0], v1_os_weight_half[0], AscendC::RoundMode::CAST_NONE, 64);
          tl::ascend::Fill<float>(v1_os_weight_2d[0], 1.000000e+00f, 8192);
          AscendC::PipeBarrier<PIPE_V>();
          for (int32_t sy_col_chunk_1 = 0; sy_col_chunk_1 < 2; ++sy_col_chunk_1) {
            tl::ascend::brcb<float>(v1_os_tmp[0], v1_os_weight[0], 8, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::mul_mask<float>(v1_os_weight_2d[(sy_col_chunk_1 * 64)], v1_os_weight_2d[(sy_col_chunk_1 * 64)], v1_os_tmp[0], 0xFFFFFFFFFFFFFFFF, 0, 64, 1, 1, 0, 16, 16, 1);
          }
          tl::ascend::Broadcast<float, 2, 1, false>(v1_os_global_max_2d[0],v1_os_global_max[0],tmp_ub,(uint32_t[]){1, 128}, (uint32_t[]){1, 1});
          tl::ascend::Fill<float>(v1_os_local_sum[0], 0.000000e+00f, 1);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
          for (int32_t local_n_2 = 0; local_n_2 < 4; ++local_n_2) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
            tl::ascend::copy_gm_to_ub<float, 128, 64>(v1_os_bmm2[0], bmm2Res[(((((pipe_iter * 1572864) + (cid * 65536)) + (vid * 512)) + (local_n_2 * 128)) - 1572864)], 1024, 64, 128, 0.000000e+00f);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);
            AscendC::Mul(v1_os_bmm2[0], v1_os_bmm2[0], v1_os_weight_2d[0], 8192);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::reduce_sum<float,  64,  128,  0>(v1_os_score[0], v1_os_bmm2[0], tmp_ub[0], true);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(v1_os_score[0], v1_os_score[0], v1_os_global_max_2d[0], 128);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(v1_os_score[0], v1_os_score[0], 128);
            AscendC::PipeBarrier<PIPE_V>();
            {
            AscendC::Muls(v1_os_reduce_2d[0], v1_os_score[0], 1.000000e+00f, 128);
            }
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::reduce_sum<float,  1,  128,  -1>(v1_os_tile_sum[0], v1_os_reduce_2d[0], tmp_ub[0], true);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(v1_os_local_sum[0], v1_os_local_sum[0], v1_os_tile_sum[0], 1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
          }
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);
          tl::ascend::copy_ub_to_gm<float, 1>(V1Stats[((((pipe_iter * 768) + (cid * 32)) + (vid * 8)) - 752)], v1_os_local_sum[0], 65536, 1, 1);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
          if (((pipe_iter + 1) % 2) == 0) {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(10);
            AscendC::CrossCoreWaitFlag(12);
          } else {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(11);
            AscendC::CrossCoreWaitFlag(13);
          }
        }
        if ASCEND_IS_AIC {
          if (((pipe_iter + 1) % 2) == 0) {
            AscendC::CrossCoreWaitFlag(10);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(12);
          } else {
            AscendC::CrossCoreWaitFlag(11);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(13);
          }
        }
        if ASCEND_IS_AIV {
          tl::ascend::ArithProgression<float>(v1_ob_offsets[0], 0.000000e+00f, 1.000000e+00f, 64);
          tl::ascend::Fill<float>(v1_ob_eps[0], 1.000000e-08f, 64);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(2);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
          tl::ascend::copy_gm_to_ub<bfloat16_t, 64>(v1_ob_weight_half[0], Weights[(((pipe_iter * 1536) + (cid * 64)) - 1536)], 131072, 1, 64, bfloat16_t(0.000000e+00f));
          tl::ascend::copy_gm_to_ub<float, 8, 4>(v1_ob_stats[0], V1Stats[(((pipe_iter * 768) + (cid * 32)) - 768)], 8, 4, 8, 0.000000e+00f);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
          AscendC::Cast(v1_ob_weight[0], v1_ob_weight_half[0], AscendC::RoundMode::CAST_NONE, 64);
          tl::ascend::Fill<float>(v1_ob_weight_2d[0], 1.000000e+00f, 4096);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::brcb<float>(v1_ob_row_tmp[0], v1_ob_weight[0], 8, 1, 8);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::mul_mask<float>(v1_ob_weight_2d[0], v1_ob_weight_2d[0], v1_ob_row_tmp[0], 0xFFFFFFFFFFFFFFFF, 0, 64, 1, 1, 0, 8, 8, 1);
          AscendC::Max(v1_ob_global_max[0], v1_ob_stats[0], v1_ob_stats[8], 1);
          AscendC::Add(v1_ob_global_sum[0], v1_ob_stats[16], v1_ob_stats[24], 1);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Max(v1_ob_global_sum[0], v1_ob_global_sum[0], v1_ob_eps[0], 1);
          tl::ascend::Broadcast<float, 2, 1, false>(v1_ob_global_max_2d[0],v1_ob_global_max[0],tmp_ub,(uint32_t[]){1, 64}, (uint32_t[]){1, 1});
          tl::ascend::Broadcast<float, 2, 1, false>(v1_ob_global_sum_2d[0],v1_ob_global_sum[0],tmp_ub,(uint32_t[]){1, 64}, (uint32_t[]){1, 1});
          tl::ascend::Fill<float>(v1_ob_dw_accum[0], 0.000000e+00f, 64);
          tl::ascend::Fill<float>(v1_ob_loss_accum[0], 0.000000e+00f, 1);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
          for (int32_t local_n_3 = 0; local_n_3 < 4; ++local_n_3) {
            for (int32_t bw_col_id = 0; bw_col_id < 2; ++bw_col_id) {
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
              tl::ascend::copy_gm_to_ub<float, 64>(v1_ob_p[0], PReduced[((((((pipe_iter * 24576) + (cid * 1024)) + (vid * 512)) + (local_n_3 * 128)) + (bw_col_id * 64)) - 24576)], 2097152, 1, 64, 0.000000e+00f);
              tl::ascend::copy_gm_to_ub<float, 64, 64>(v1_ob_bmm2[0], bmm2Res[((((((pipe_iter * 1572864) + (cid * 65536)) + (vid * 512)) + (local_n_3 * 128)) + (bw_col_id * 64)) - 1572864)], 1024, 64, 64, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);
              AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(2);
              AscendC::Mul(v1_ob_work[0], v1_ob_bmm2[0], v1_ob_weight_2d[0], 4096);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::reduce_sum<float,  64,  64,  0>(v1_ob_sy[0], v1_ob_work[0], tmp_ub[0], true);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Sub(v1_ob_sy[0], v1_ob_sy[0], v1_ob_global_max_2d[0], 64);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Exp(v1_ob_sy[0], v1_ob_sy[0], 64);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Div(v1_ob_sy[0], v1_ob_sy[0], v1_ob_global_sum_2d[0], 64);
              AscendC::Sub(v1_ob_diff[0], v1_ob_sy[0], v1_ob_p[0], 64);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::Broadcast<float, 2, 0, false>(v1_ob_diff_2d[0],v1_ob_diff[0],tmp_ub,(uint32_t[]){64, 64}, (uint32_t[]){1, 64});
              {
              AscendC::Muls(v1_ob_gate[0], v1_ob_bmm2[0], -1.000000e+04f, 4096);
              }
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Exp(v1_ob_gate[0], v1_ob_gate[0], 4096);
              AscendC::PipeBarrier<PIPE_V>();
              {
              AscendC::Muls(v1_ob_gate[0], v1_ob_gate[0], -1.000000e+00f, 4096);
              }
              AscendC::PipeBarrier<PIPE_V>();
              {
              AscendC::Adds(v1_ob_gate[0], v1_ob_gate[0], 1.000000e+00f, 4096);
              }
              AscendC::Mul(v1_ob_work[0], v1_ob_bmm2[0], v1_ob_diff_2d[0], 4096);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::reduce_sum<float,  64,  64,  -1>(v1_ob_dw[0], v1_ob_work[0], tmp_ub[0], true);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Add(v1_ob_dw_accum[0], v1_ob_dw_accum[0], v1_ob_dw[0], 64);
              AscendC::Max(v1_ob_loss_p[0], v1_ob_p[0], v1_ob_eps[0], 64);
              AscendC::Max(v1_ob_sy[0], v1_ob_sy[0], v1_ob_eps[0], 64);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Ln(v1_ob_log_p[0], v1_ob_loss_p[0], 64);
              AscendC::Ln(v1_ob_log_sy[0], v1_ob_sy[0], 64);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Sub(v1_ob_loss_2d[0], v1_ob_log_p[0], v1_ob_log_sy[0], 64);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Mul(v1_ob_loss_2d[0], v1_ob_loss_2d[0], v1_ob_p[0], 64);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::reduce_sum<float,  1,  64,  -1>(v1_ob_loss_sum[0], v1_ob_loss_2d[0], tmp_ub[0], true);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Add(v1_ob_loss_accum[0], v1_ob_loss_accum[0], v1_ob_loss_sum[0], 1);
              AscendC::Mul(v1_ob_bmm2[0], v1_ob_weight_2d[0], v1_ob_diff_2d[0], 4096);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Mul(v1_ob_bmm2[0], v1_ob_bmm2[0], v1_ob_gate[0], 4096);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Cast(v1_ob_rg_half[0], v1_ob_bmm2[0], AscendC::RoundMode::CAST_RINT, 4096);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(2);
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(2);
              AscendC::PipeBarrier<PIPE_MTE3>();
              tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 64>(reluGradRes[((((((pipe_iter * 1572864) + (cid * 65536)) + (vid * 512)) + (local_n_3 * 128)) + (bw_col_id * 64)) - 1572864)], v1_ob_rg_half[0], 1024, 64, 64);
              AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(2);
              AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
            }
          }
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
          tl::ascend::copy_ub_to_gm<float, 64>(DWeightPartial[((((pipe_iter * 3072) + (cid * 128)) + (vid * 64)) - 3072)], v1_ob_dw_accum[0], 262144, 1, 64);
          tl::ascend::copy_ub_to_gm<float, 1>(LossPartial[(((((((pipe_iter * 24) + cid) - 24) / 512) * 8192) + (((((pipe_iter * 24) + cid) + 488) % 512) * 16)) + (vid * 8))], v1_ob_loss_accum[0], 32768, 1, 1);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(2);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
          if (((pipe_iter + 1) % 2) == 0) {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(10);
            AscendC::CrossCoreWaitFlag(12);
          } else {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(11);
            AscendC::CrossCoreWaitFlag(13);
          }
        }
        if ASCEND_IS_AIC {
          if (((pipe_iter + 1) % 2) == 0) {
            AscendC::CrossCoreWaitFlag(10);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(12);
          } else {
            AscendC::CrossCoreWaitFlag(11);
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(13);
          }
        }
        if ASCEND_IS_AIV {
          tl::ascend::copy_gm_to_ub<float, 32>(v1_fdw_left[0], DWeightPartial[((((pipe_iter * 3072) + (cid * 128)) + (vid * 32)) - 3072)], 262144, 1, 32, 0.000000e+00f);
          tl::ascend::copy_gm_to_ub<float, 32>(v1_fdw_right[0], DWeightPartial[((((pipe_iter * 3072) + (cid * 128)) + (vid * 32)) - 3008)], 262144, 1, 32, 0.000000e+00f);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
          AscendC::Add(v1_fdw_left[0], v1_fdw_left[0], v1_fdw_right[0], 32);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Cast(v1_fdw_half[0], v1_fdw_left[0], AscendC::RoundMode::CAST_RINT, 32);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
          tl::ascend::copy_ub_to_gm<bfloat16_t, 32>(dWeightOut[((((pipe_iter * 1536) + (cid * 64)) + (vid * 32)) - 1536)], v1_fdw_half[0], 131072, 1, 32);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
          if (((pipe_iter + 1) % 2) == 0) {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(8);
          } else {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(9);
          }
        }
        if ASCEND_IS_AIC {
          tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(c2_qi_l1[0], QueryIndex[(((pipe_iter * 196608) + (cid * 8192)) - 196608)], 128, 64, 128, true);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(6);
          if (((pipe_iter + 1) % 2) == 0) {
            AscendC::CrossCoreWaitFlag(8);
          } else {
            AscendC::CrossCoreWaitFlag(9);
          }
          AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(2);
          AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2);
          AscendC::SetFlag<AscendC::HardEvent::FIX_M>(2);
          AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3);
          AscendC::SetFlag<AscendC::HardEvent::FIX_M>(3);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(6);
          for (int32_t n_idx = 0; n_idx < 8; ++n_idx) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(2);
            tl::ascend::copy_gm_to_l1<bfloat16_t, 64, 128>(c2_rg_l1[0], reluGradRes[((((pipe_iter * 1572864) + (cid * 65536)) + (n_idx * 128)) - 1572864)], 1024, 64, 128, true);
            tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c2_gki_l1[0], GatherKeyIndex_ws[((((pipe_iter * 3145728) + (cid * 131072)) + (n_idx * 16384)) - 3145728)], 128, 128, 128, true);
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
            tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 128, 0>(dk_matmul_partial[((((pipe_iter * 3145728) + (cid * 131072)) + (n_idx * 16384)) - 3145728)], c2_dk_l0c[0], 128, 128, 128, 0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(3);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(3);
            tl::ascend::mma<bfloat16_t, float, 64, 128>(c2_dq_rg_l0a[0],c2_dq_gki_l0b[0],c2_dq_l0c[0], (n_idx == 0), 128);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3);
            if (n_idx < 7) {
              AscendC::SetFlag<AscendC::HardEvent::FIX_M>(3);
            }
          }
          AscendC::SetFlag<AscendC::HardEvent::M_FIX>(3);
          AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(3);
          tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 64, 128, 0>(dQueryIndexAccum[(((pipe_iter * 196608) + (cid * 8192)) - 196608)], c2_dq_l0c[0], 128, 64, 128, 0);
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
    int32_t condval;
    if ((cid < 16)) {
      condval = 1;
    } else {
      condval = 0;
    }
    for (int32_t clear_iter = 0; clear_iter < (condval + 2); ++clear_iter) {
      tl::ascend::copy_ub_to_gm<float, 128, 32>(dKeyIndexAccum[((((cid * 16384) + (vid * 8192)) + ((((cid * 2) + vid) < 32 ? ((cid * 2) + vid) : 32) * 4096)) + (clear_iter * 4096))], v2_zero_batch[0], 128, ((1 <= ((((128 - (((cid * 2) + vid) < 32 ? ((cid * 2) + vid) : 32)) - clear_iter) - (vid * 2)) - (cid * 4))) ? 32 : ((((((cid * 4) + (vid * 2)) + (((cid * 2) + vid) < 32 ? ((cid * 2) + vid) : 32)) + clear_iter) < 128) ? ((((4096 - ((((cid * 2) + vid) < 32 ? ((cid * 2) + vid) : 32) * 32)) - (clear_iter * 32)) - (vid * 64)) - (cid * 128)) : 0)), 128);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(5);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(5);
  }
  AscendC::SyncAll<false>();
  if ASCEND_IS_AIV {
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(6);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(7);
    for (int32_t task_id = 0; task_id < 2048; ++task_id) {
      int32_t valid_k_size = KRealSize.GetValue(task_id);
      int32_t condval_1;
      if ((cid < 16)) {
        condval_1 = 1;
      } else {
        condval_1 = 0;
      }
      if ((((cid * 32) + ((cid < 16 ? cid : 16) * 16)) + ((vid * ((condval_1 + 3) / 2)) * 16)) < valid_k_size) {
        int32_t condval_2;
        if ((cid < 16)) {
          condval_2 = 1;
        } else {
          condval_2 = 0;
        }
        int32_t condval_3;
        if ((cid < 16)) {
          condval_3 = 1;
        } else {
          condval_3 = 0;
        }
        int32_t condval_4;
        if ((cid < 16)) {
          condval_4 = 1;
        } else {
          condval_4 = 0;
        }
        if ((((((cid * 32) + (((condval_2 + 3) / 2) * 16)) + ((cid < 16 ? cid : 16) * 16)) + ((vid * ((condval_3 + 3) / 2)) * 16)) - ((vid * (1 - ((condval_4 + 1) % 2))) * 16)) <= valid_k_size) {
          if ASCEND_IS_AIV {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(6);
            int32_t condval_5;
            if ((cid < 16)) {
              condval_5 = 1;
            } else {
              condval_5 = 0;
            }
            int32_t condval_6;
            if ((cid < 16)) {
              condval_6 = 1;
            } else {
              condval_6 = 0;
            }
            int32_t condval_7;
            if ((cid < 16)) {
              condval_7 = 1;
            } else {
              condval_7 = 0;
            }
            int32_t condval_8;
            if ((cid < 16)) {
              condval_8 = 1;
            } else {
              condval_8 = 0;
            }
            tl::ascend::copy_gm_to_ub<float, 128, 16>(v2_dk_batch_ping[0], dk_matmul_partial[((((task_id * 131072) + (cid * 4096)) + ((cid < 16 ? cid : 16) * 2048)) + ((vid * ((condval_5 + 3) / 2)) * 2048))], 128, ((1 <= (((64 - (cid < 16 ? cid : 16)) - (vid * ((condval_6 + 3) / 2))) - (cid * 2))) ? 16 : ((((((cid < 16 ? cid : 16) + (vid * ((condval_7 + 3) / 2))) / 2) + cid) < 32) ? (((1024 - ((cid < 16 ? cid : 16) * 16)) - ((vid * ((condval_8 + 3) / 2)) * 16)) - (cid * 32)) : 0)), 128, 0.000000e+00f);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(6);
            int32_t condval_9;
            if ((cid < 16)) {
              condval_9 = 1;
            } else {
              condval_9 = 0;
            }
            int32_t condval_10;
            if ((cid < 16)) {
              condval_10 = 1;
            } else {
              condval_10 = 0;
            }
            for (int32_t batch_id = 0; batch_id < (((condval_9 + 3) / 2) - (vid * (1 - ((condval_10 + 1) % 2)))); ++batch_id) {
              int32_t condval_11;
              if ((cid < 16)) {
                condval_11 = 1;
              } else {
                condval_11 = 0;
              }
              int32_t condval_12;
              if ((cid < 16)) {
                condval_12 = 1;
              } else {
                condval_12 = 0;
              }
              if ((batch_id + 1) < (((condval_11 + 3) / 2) - (vid * (1 - ((condval_12 + 1) % 2))))) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((((batch_id + 1) % 2) + 6));
                if (((batch_id + 1) % 2) == 0) {
                  int32_t condval_13;
                  if ((cid < 16)) {
                    condval_13 = 1;
                  } else {
                    condval_13 = 0;
                  }
                  int32_t condval_14;
                  if ((cid < 16)) {
                    condval_14 = 1;
                  } else {
                    condval_14 = 0;
                  }
                  int32_t condval_15;
                  if ((cid < 16)) {
                    condval_15 = 1;
                  } else {
                    condval_15 = 0;
                  }
                  int32_t condval_16;
                  if ((cid < 16)) {
                    condval_16 = 1;
                  } else {
                    condval_16 = 0;
                  }
                  tl::ascend::copy_gm_to_ub<float, 128, 16>(v2_dk_batch_ping[0], dk_matmul_partial[((((((task_id * 131072) + (cid * 4096)) + (batch_id * 2048)) + ((cid < 16 ? cid : 16) * 2048)) + ((vid * ((condval_13 + 3) / 2)) * 2048)) + 2048)], 128, ((1 <= ((((63 - batch_id) - (cid < 16 ? cid : 16)) - (vid * ((condval_14 + 3) / 2))) - (cid * 2))) ? 16 : ((((((cid * 2) + (cid < 16 ? cid : 16)) + batch_id) + (vid * ((condval_15 + 3) / 2))) < 63) ? ((((1008 - (batch_id * 16)) - ((cid < 16 ? cid : 16) * 16)) - ((vid * ((condval_16 + 3) / 2)) * 16)) - (cid * 32)) : 0)), 128, 0.000000e+00f);
                } else {
                  int32_t condval_17;
                  if ((cid < 16)) {
                    condval_17 = 1;
                  } else {
                    condval_17 = 0;
                  }
                  int32_t condval_18;
                  if ((cid < 16)) {
                    condval_18 = 1;
                  } else {
                    condval_18 = 0;
                  }
                  int32_t condval_19;
                  if ((cid < 16)) {
                    condval_19 = 1;
                  } else {
                    condval_19 = 0;
                  }
                  int32_t condval_20;
                  if ((cid < 16)) {
                    condval_20 = 1;
                  } else {
                    condval_20 = 0;
                  }
                  tl::ascend::copy_gm_to_ub<float, 128, 16>(v2_dk_batch_pong[0], dk_matmul_partial[((((((task_id * 131072) + (cid * 4096)) + (batch_id * 2048)) + ((cid < 16 ? cid : 16) * 2048)) + ((vid * ((condval_17 + 3) / 2)) * 2048)) + 2048)], 128, ((1 <= ((((63 - batch_id) - (cid < 16 ? cid : 16)) - (vid * ((condval_18 + 3) / 2))) - (cid * 2))) ? 16 : ((((((cid * 2) + (cid < 16 ? cid : 16)) + batch_id) + (vid * ((condval_19 + 3) / 2))) < 63) ? ((((1008 - (batch_id * 16)) - ((cid < 16 ? cid : 16) * 16)) - ((vid * ((condval_20 + 3) / 2)) * 16)) - (cid * 32)) : 0)), 128, 0.000000e+00f);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>((((batch_id + 1) % 2) + 6));
              }
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(((batch_id % 2) + 6));
              for (int32_t pair_idx = 0; pair_idx < 8; ++pair_idx) {
                int64_t condval_21;
                if ((((int64_t)cid) < (int64_t)16)) {
                  condval_21 = (int64_t)1;
                } else {
                  condval_21 = (int64_t)0;
                }
                int32_t sparse_idx0 = SparseIndices.GetValue(((((((((int64_t)task_id) * (int64_t)1024) + (((int64_t)cid) * (int64_t)32)) + (((int64_t)batch_id) * (int64_t)16)) + ((((int64_t)cid) < (int64_t)16 ? ((int64_t)cid) : (int64_t)16) * (int64_t)16)) + ((((int64_t)vid) * ((condval_21 + (int64_t)3) / (int64_t)2)) * (int64_t)16)) + (((int64_t)pair_idx) * (int64_t)2)));
                int64_t condval_22;
                if ((((int64_t)cid) < (int64_t)16)) {
                  condval_22 = (int64_t)1;
                } else {
                  condval_22 = (int64_t)0;
                }
                int32_t sparse_idx1 = SparseIndices.GetValue((((((((((int64_t)task_id) * (int64_t)1024) + (((int64_t)cid) * (int64_t)32)) + (((int64_t)batch_id) * (int64_t)16)) + ((((int64_t)cid) < (int64_t)16 ? ((int64_t)cid) : (int64_t)16) * (int64_t)16)) + ((((int64_t)vid) * ((condval_22 + (int64_t)3) / (int64_t)2)) * (int64_t)16)) + (((int64_t)pair_idx) * (int64_t)2)) + (int64_t)1));
                if (0 <= sparse_idx0) {
                  if ((batch_id % 2) == 0) {
                    tl::ascend::atomic_add_ub_to_gm<float, 128, 1>(dKeyIndexAccum[(((task_id / 512) * 131072) + (sparse_idx0 * 128))], v2_dk_batch_ping[(pair_idx * 256)], 524288, ((-4095 <= ((0 - sparse_idx0) - ((task_id / 512) * 1024))) ? 1 : 0), 128);
                  } else {
                    tl::ascend::atomic_add_ub_to_gm<float, 128, 1>(dKeyIndexAccum[(((task_id / 512) * 131072) + (sparse_idx0 * 128))], v2_dk_batch_pong[(pair_idx * 256)], 524288, ((-4095 <= ((0 - sparse_idx0) - ((task_id / 512) * 1024))) ? 1 : 0), 128);
                  }
                }
                if (0 <= sparse_idx1) {
                  if ((batch_id % 2) == 0) {
                    tl::ascend::atomic_add_ub_to_gm<float, 128, 1>(dKeyIndexAccum[(((task_id / 512) * 131072) + (sparse_idx1 * 128))], v2_dk_batch_ping[((pair_idx * 256) + 128)], 524288, ((-4095 <= ((0 - sparse_idx1) - ((task_id / 512) * 1024))) ? 1 : 0), 128);
                  } else {
                    tl::ascend::atomic_add_ub_to_gm<float, 128, 1>(dKeyIndexAccum[(((task_id / 512) * 131072) + (sparse_idx1 * 128))], v2_dk_batch_pong[((pair_idx * 256) + 128)], 524288, ((-4095 <= ((0 - sparse_idx1) - ((task_id / 512) * 1024))) ? 1 : 0), 128);
                  }
                }
              }
              AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(((batch_id % 2) + 6));
            }
          }
        } else {
          if ASCEND_IS_AIV {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(6);
            int32_t condval_23;
            if ((cid < 16)) {
              condval_23 = 1;
            } else {
              condval_23 = 0;
            }
            int32_t condval_24;
            if ((cid < 16)) {
              condval_24 = 1;
            } else {
              condval_24 = 0;
            }
            int32_t condval_25;
            if ((cid < 16)) {
              condval_25 = 1;
            } else {
              condval_25 = 0;
            }
            int32_t condval_26;
            if ((cid < 16)) {
              condval_26 = 1;
            } else {
              condval_26 = 0;
            }
            tl::ascend::copy_gm_to_ub<float, 128, 16>(v2_dk_batch_ping[0], dk_matmul_partial[((((task_id * 131072) + (cid * 4096)) + ((cid < 16 ? cid : 16) * 2048)) + ((vid * ((condval_23 + 3) / 2)) * 2048))], 128, ((1 <= (((64 - (cid < 16 ? cid : 16)) - (vid * ((condval_24 + 3) / 2))) - (cid * 2))) ? 16 : ((((((cid < 16 ? cid : 16) + (vid * ((condval_25 + 3) / 2))) / 2) + cid) < 32) ? (((1024 - ((cid < 16 ? cid : 16) * 16)) - ((vid * ((condval_26 + 3) / 2)) * 16)) - (cid * 32)) : 0)), 128, 0.000000e+00f);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(6);
            int32_t condval_27;
            if ((cid < 16)) {
              condval_27 = 1;
            } else {
              condval_27 = 0;
            }
            int32_t condval_28;
            if ((cid < 16)) {
              condval_28 = 1;
            } else {
              condval_28 = 0;
            }
            for (int32_t batch_id_1 = 0; batch_id_1 < (((condval_27 + 3) / 2) - (vid * (1 - ((condval_28 + 1) % 2)))); ++batch_id_1) {
              int32_t condval_29;
              if ((cid < 16)) {
                condval_29 = 1;
              } else {
                condval_29 = 0;
              }
              int32_t condval_30;
              if ((cid < 16)) {
                condval_30 = 1;
              } else {
                condval_30 = 0;
              }
              if ((batch_id_1 + 1) < (((condval_29 + 3) / 2) - (vid * (1 - ((condval_30 + 1) % 2))))) {
                int32_t condval_32;
                if ((cid < 16)) {
                  condval_32 = 1;
                } else {
                  condval_32 = 0;
                }
                int32_t condval_31;
                if (((((((cid * 32) + (batch_id_1 * 16)) + ((cid < 16 ? cid : 16) * 16)) + ((vid * ((condval_32 + 3) / 2)) * 16)) + 16) < valid_k_size)) {
                  condval_31 = 1;
                } else {
                  condval_31 = 0;
                }
                if (0 < condval_31) {
                  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((((batch_id_1 + 1) % 2) + 6));
                  if (((batch_id_1 + 1) % 2) == 0) {
                    int32_t condval_33;
                    if ((cid < 16)) {
                      condval_33 = 1;
                    } else {
                      condval_33 = 0;
                    }
                    int32_t condval_34;
                    if ((cid < 16)) {
                      condval_34 = 1;
                    } else {
                      condval_34 = 0;
                    }
                    int32_t condval_35;
                    if ((cid < 16)) {
                      condval_35 = 1;
                    } else {
                      condval_35 = 0;
                    }
                    int32_t condval_36;
                    if ((cid < 16)) {
                      condval_36 = 1;
                    } else {
                      condval_36 = 0;
                    }
                    tl::ascend::copy_gm_to_ub<float, 128, 16>(v2_dk_batch_ping[0], dk_matmul_partial[((((((task_id * 131072) + (cid * 4096)) + (batch_id_1 * 2048)) + ((cid < 16 ? cid : 16) * 2048)) + ((vid * ((condval_33 + 3) / 2)) * 2048)) + 2048)], 128, ((1 <= ((((63 - batch_id_1) - (cid < 16 ? cid : 16)) - (vid * ((condval_34 + 3) / 2))) - (cid * 2))) ? 16 : ((((((cid * 2) + (cid < 16 ? cid : 16)) + batch_id_1) + (vid * ((condval_35 + 3) / 2))) < 63) ? ((((1008 - (batch_id_1 * 16)) - ((cid < 16 ? cid : 16) * 16)) - ((vid * ((condval_36 + 3) / 2)) * 16)) - (cid * 32)) : 0)), 128, 0.000000e+00f);
                  } else {
                    int32_t condval_37;
                    if ((cid < 16)) {
                      condval_37 = 1;
                    } else {
                      condval_37 = 0;
                    }
                    int32_t condval_38;
                    if ((cid < 16)) {
                      condval_38 = 1;
                    } else {
                      condval_38 = 0;
                    }
                    int32_t condval_39;
                    if ((cid < 16)) {
                      condval_39 = 1;
                    } else {
                      condval_39 = 0;
                    }
                    int32_t condval_40;
                    if ((cid < 16)) {
                      condval_40 = 1;
                    } else {
                      condval_40 = 0;
                    }
                    tl::ascend::copy_gm_to_ub<float, 128, 16>(v2_dk_batch_pong[0], dk_matmul_partial[((((((task_id * 131072) + (cid * 4096)) + (batch_id_1 * 2048)) + ((cid < 16 ? cid : 16) * 2048)) + ((vid * ((condval_37 + 3) / 2)) * 2048)) + 2048)], 128, ((1 <= ((((63 - batch_id_1) - (cid < 16 ? cid : 16)) - (vid * ((condval_38 + 3) / 2))) - (cid * 2))) ? 16 : ((((((cid * 2) + (cid < 16 ? cid : 16)) + batch_id_1) + (vid * ((condval_39 + 3) / 2))) < 63) ? ((((1008 - (batch_id_1 * 16)) - ((cid < 16 ? cid : 16) * 16)) - ((vid * ((condval_40 + 3) / 2)) * 16)) - (cid * 32)) : 0)), 128, 0.000000e+00f);
                  }
                  AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>((((batch_id_1 + 1) % 2) + 6));
                }
              }
              int32_t condval_42;
              if ((cid < 16)) {
                condval_42 = 1;
              } else {
                condval_42 = 0;
              }
              int32_t condval_41;
              if ((((((cid * 32) + (batch_id_1 * 16)) + ((cid < 16 ? cid : 16) * 16)) + ((vid * ((condval_42 + 3) / 2)) * 16)) < valid_k_size)) {
                condval_41 = 1;
              } else {
                condval_41 = 0;
              }
              if (0 < condval_41) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(((batch_id_1 % 2) + 6));
                for (int32_t pair_idx_1 = 0; pair_idx_1 < 8; ++pair_idx_1) {
                  int64_t condval_43;
                  if ((((int64_t)cid) < (int64_t)16)) {
                    condval_43 = (int64_t)1;
                  } else {
                    condval_43 = (int64_t)0;
                  }
                  int32_t sparse_idx0_1 = SparseIndices.GetValue(((((((((int64_t)task_id) * (int64_t)1024) + (((int64_t)cid) * (int64_t)32)) + (((int64_t)batch_id_1) * (int64_t)16)) + ((((int64_t)cid) < (int64_t)16 ? ((int64_t)cid) : (int64_t)16) * (int64_t)16)) + ((((int64_t)vid) * ((condval_43 + (int64_t)3) / (int64_t)2)) * (int64_t)16)) + (((int64_t)pair_idx_1) * (int64_t)2)));
                  int64_t condval_44;
                  if ((((int64_t)cid) < (int64_t)16)) {
                    condval_44 = (int64_t)1;
                  } else {
                    condval_44 = (int64_t)0;
                  }
                  int32_t sparse_idx1_1 = SparseIndices.GetValue((((((((((int64_t)task_id) * (int64_t)1024) + (((int64_t)cid) * (int64_t)32)) + (((int64_t)batch_id_1) * (int64_t)16)) + ((((int64_t)cid) < (int64_t)16 ? ((int64_t)cid) : (int64_t)16) * (int64_t)16)) + ((((int64_t)vid) * ((condval_44 + (int64_t)3) / (int64_t)2)) * (int64_t)16)) + (((int64_t)pair_idx_1) * (int64_t)2)) + (int64_t)1));
                  int32_t condval_46;
                  if ((cid < 16)) {
                    condval_46 = 1;
                  } else {
                    condval_46 = 0;
                  }
                  int32_t condval_45;
                  if (((((((cid * 32) + (batch_id_1 * 16)) + ((cid < 16 ? cid : 16) * 16)) + ((vid * ((condval_46 + 3) / 2)) * 16)) + (pair_idx_1 * 2)) < valid_k_size)) {
                    condval_45 = 1;
                  } else {
                    condval_45 = 0;
                  }
                  if (0 < condval_45) {
                    if (0 <= sparse_idx0_1) {
                      if ((batch_id_1 % 2) == 0) {
                        tl::ascend::atomic_add_ub_to_gm<float, 128, 1>(dKeyIndexAccum[(((task_id / 512) * 131072) + (sparse_idx0_1 * 128))], v2_dk_batch_ping[(pair_idx_1 * 256)], 524288, ((-4095 <= ((0 - sparse_idx0_1) - ((task_id / 512) * 1024))) ? 1 : 0), 128);
                      } else {
                        tl::ascend::atomic_add_ub_to_gm<float, 128, 1>(dKeyIndexAccum[(((task_id / 512) * 131072) + (sparse_idx0_1 * 128))], v2_dk_batch_pong[(pair_idx_1 * 256)], 524288, ((-4095 <= ((0 - sparse_idx0_1) - ((task_id / 512) * 1024))) ? 1 : 0), 128);
                      }
                    }
                  }
                  int32_t condval_48;
                  if ((cid < 16)) {
                    condval_48 = 1;
                  } else {
                    condval_48 = 0;
                  }
                  int32_t condval_47;
                  if ((((((((cid * 32) + (batch_id_1 * 16)) + ((cid < 16 ? cid : 16) * 16)) + ((vid * ((condval_48 + 3) / 2)) * 16)) + (pair_idx_1 * 2)) + 1) < valid_k_size)) {
                    condval_47 = 1;
                  } else {
                    condval_47 = 0;
                  }
                  if (0 < condval_47) {
                    if (0 <= sparse_idx1_1) {
                      if ((batch_id_1 % 2) == 0) {
                        tl::ascend::atomic_add_ub_to_gm<float, 128, 1>(dKeyIndexAccum[(((task_id / 512) * 131072) + (sparse_idx1_1 * 128))], v2_dk_batch_ping[((pair_idx_1 * 256) + 128)], 524288, ((-4095 <= ((0 - sparse_idx1_1) - ((task_id / 512) * 1024))) ? 1 : 0), 128);
                      } else {
                        tl::ascend::atomic_add_ub_to_gm<float, 128, 1>(dKeyIndexAccum[(((task_id / 512) * 131072) + (sparse_idx1_1 * 128))], v2_dk_batch_pong[((pair_idx_1 * 256) + 128)], 524288, ((-4095 <= ((0 - sparse_idx1_1) - ((task_id / 512) * 1024))) ? 1 : 0), 128);
                      }
                    }
                  }
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(((batch_id_1 % 2) + 6));
              }
            }
          }
        }
      }
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(6);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(7);
  }
  AscendC::SyncAll<false>();
  if ASCEND_IS_AIV {
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(4);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(7);
    for (int32_t dq_task_iter = 0; dq_task_iter < 86; ++dq_task_iter) {
      if (((dq_task_iter * 3) + (cid / 8)) < 256) {
        for (int32_t dq_group_local = 0; dq_group_local < 32; ++dq_group_local) {
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(4);
          tl::ascend::copy_gm_to_ub<float, 128, 1>(v2_finalize_dq_fp32[0], dQueryIndexAccum[((((dq_task_iter * 196608) + (cid * 8192)) + (dq_group_local * 256)) + (vid * 128))], 16777216, 1, 128, 0.000000e+00f);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(5);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(5);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(7);
          AscendC::Cast(v2_finalize_dq_half[0], v2_finalize_dq_fp32[0], AscendC::RoundMode::CAST_RINT, 128);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(4);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(5);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(5);
          tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 1>(dQueryIndex[((((dq_task_iter * 196608) + (cid * 8192)) + (dq_group_local * 256)) + (vid * 128))], v2_finalize_dq_half[0], 16777216, 1, 128);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(7);
        }
      }
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(4);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(7);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(5);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(6);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(5);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(6);
    int32_t condval_49;
    if ((cid < 8)) {
      condval_49 = 1;
    } else {
      condval_49 = 0;
    }
    for (int32_t finalize_iter = 0; finalize_iter < (condval_49 + 5); ++finalize_iter) {
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(((finalize_iter % 2) + 5));
      tl::ascend::copy_gm_to_ub<float, 128, 16>(v2_finalize_dk_fp32[((finalize_iter % 2) * 2048)], dKeyIndexAccum[((((cid * 20480) + (vid * 10240)) + ((((cid * 2) + vid) < 16 ? ((cid * 2) + vid) : 16) * 2048)) + (finalize_iter * 2048))], 128, ((1 <= ((((256 - (((cid * 2) + vid) < 16 ? ((cid * 2) + vid) : 16)) - finalize_iter) - (vid * 5)) - (cid * 10))) ? 16 : ((((((cid * 10) + (vid * 5)) + (((cid * 2) + vid) < 16 ? ((cid * 2) + vid) : 16)) + finalize_iter) < 256) ? ((((4096 - ((((cid * 2) + vid) < 16 ? ((cid * 2) + vid) : 16) * 16)) - (finalize_iter * 16)) - (vid * 80)) - (cid * 160)) : 0)), 128, 0.000000e+00f);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(((finalize_iter % 2) + 5));
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(((finalize_iter % 2) + 5));
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(((finalize_iter % 2) + 5));
      AscendC::Cast(v2_finalize_dk_half[((finalize_iter % 2) * 2048)], v2_finalize_dk_fp32[((finalize_iter % 2) * 2048)], AscendC::RoundMode::CAST_RINT, 2048);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(((finalize_iter % 2) + 5));
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(((finalize_iter % 2) + 5));
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(((finalize_iter % 2) + 5));
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 16>(dKeyIndexOut[((((cid * 20480) + (vid * 10240)) + ((((cid * 2) + vid) < 16 ? ((cid * 2) + vid) : 16) * 2048)) + (finalize_iter * 2048))], v2_finalize_dk_half[((finalize_iter % 2) * 2048)], 128, ((1 <= ((((256 - (((cid * 2) + vid) < 16 ? ((cid * 2) + vid) : 16)) - finalize_iter) - (vid * 5)) - (cid * 10))) ? 16 : ((((((cid * 10) + (vid * 5)) + (((cid * 2) + vid) < 16 ? ((cid * 2) + vid) : 16)) + finalize_iter) < 256) ? ((((4096 - ((((cid * 2) + vid) < 16 ? ((cid * 2) + vid) : 16) * 16)) - (finalize_iter * 16)) - (vid * 80)) - (cid * 160)) : 0)), 128);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(((finalize_iter % 2) + 5));
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(5);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(6);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(5);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(6);
    if (cid == 0) {
      if (vid == 0) {
        tl::ascend::Fill<float>(v2_loss_total[0], 0.000000e+00f, 1);
        for (int32_t loss_chunk_id = 0; loss_chunk_id < 256; ++loss_chunk_id) {
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(4);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(4);
          tl::ascend::copy_gm_to_ub<float, 128, 1>(v2_loss_chunk[0], LossPartial[(loss_chunk_id * 128)], 32768, 1, 128, 0.000000e+00f);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(4);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(4);
          tl::ascend::reduce_sum<float,  1,  128,  -1>(v2_loss_chunk_sum[0], v2_loss_chunk[0], tmp_ub[0], true);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Add(v2_loss_total[0], v2_loss_total[0], v2_loss_chunk_sum[0], 1);
        }
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

extern "C" void call(uint8_t* SparseIndices_handle, uint8_t* KRealSize_handle, uint8_t* V0ZeroRow_handle, uint8_t* Query_handle, uint8_t* QueryRope_handle, uint8_t* Key_handle, uint8_t* KeyRope_handle, uint8_t* QueryIndex_handle, uint8_t* KeyIndex_handle, uint8_t* Weights_handle, uint8_t* SoftmaxMax_handle, uint8_t* SoftmaxSum_handle, uint8_t* dQueryIndexAccum_handle, uint8_t* dKeyIndexAccum_handle, uint8_t* GatherKey_ws_handle, uint8_t* GatherKeyRope_ws_handle, uint8_t* GatherKeyIndex_ws_handle, uint8_t* bmm1Res_handle, uint8_t* bmm2Res_handle, uint8_t* PReduced_handle, uint8_t* V1Stats_handle, uint8_t* DWeightPartial_handle, uint8_t* dk_matmul_partial_handle, uint8_t* reluGradRes_handle, uint8_t* LossPartial_handle, uint8_t* C2ZeroPad_handle, uint8_t* dQueryIndex_handle, uint8_t* dKeyIndexOut_handle, uint8_t* dWeightOut_handle, uint8_t* LossOut_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<24, nullptr, stream>>>(SparseIndices_handle, KRealSize_handle, V0ZeroRow_handle, Query_handle, QueryRope_handle, Key_handle, KeyRope_handle, QueryIndex_handle, KeyIndex_handle, Weights_handle, SoftmaxMax_handle, SoftmaxSum_handle, dQueryIndexAccum_handle, dKeyIndexAccum_handle, GatherKey_ws_handle, GatherKeyRope_ws_handle, GatherKeyIndex_ws_handle, bmm1Res_handle, bmm2Res_handle, PReduced_handle, V1Stats_handle, DWeightPartial_handle, dk_matmul_partial_handle, reluGradRes_handle, LossPartial_handle, C2ZeroPad_handle, dQueryIndex_handle, dKeyIndexOut_handle, dWeightOut_handle, LossOut_handle, fftsAddr);
}
