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
  auto tmp_ub = ascend_ub.GetWithOffset<uint8_t>(16384, 68384);
  auto v0_key_ub = ascend_ub.GetWithOffset<bfloat16_t>(16384, 0);
  auto v0_ki_ub = ascend_ub.GetWithOffset<bfloat16_t>(4096, 32768);
  auto v0_rope_ub = ascend_ub.GetWithOffset<bfloat16_t>(4096, 40960);
  auto v1_op_offsets = ascend_ub.GetWithOffset<float>(128, 49152);
  auto v1_om_offsets = ascend_ub.GetWithOffset<float>(128, 84768);
  auto v1_os_offsets = ascend_ub.GetWithOffset<float>(128, 103424);
  auto v1_ob_offsets = ascend_ub.GetWithOffset<float>(64, 122656);
  auto v1_ob_eps = ascend_ub.GetWithOffset<float>(64, 122912);
  auto c2_qi_l1 = ascend_l1.GetWithOffset<bfloat16_t>(2048, 110592);
  auto c1_query_chunk_0_l1 = ascend_l1.GetWithOffset<bfloat16_t>(4096, 0);
  auto c1_query_chunk_1_l1 = ascend_l1.GetWithOffset<bfloat16_t>(4096, 8192);
  auto c1_query_chunk_2_l1 = ascend_l1.GetWithOffset<bfloat16_t>(4096, 16384);
  auto c1_query_chunk_3_l1 = ascend_l1.GetWithOffset<bfloat16_t>(4096, 24576);
  auto c1_query_rope_cache_l1 = ascend_l1.GetWithOffset<bfloat16_t>(4096, 32768);
  auto c1_qi_l1 = ascend_l1.GetWithOffset<bfloat16_t>(2048, 40960);
  auto c1_key_l1 = ascend_l1.GetWithOffset<bfloat16_t>(16384, 45056);
  auto c1_q_l0a = ascend_l0a.GetWithOffset<bfloat16_t>(4096, 0);
  auto c1_key_l0b = ascend_l0b.GetWithOffset<bfloat16_t>(16384, 0);
  auto c1_p_l0c = ascend_l0c.GetWithOffset<float>(4096, 0);
  auto c1_qi_l0a = ascend_l0a.GetWithOffset<bfloat16_t>(2048, 8192);
  auto c1_ki_l1 = ascend_l1.GetWithOffset<bfloat16_t>(16384, 77824);
  auto c1_ki_l0b = ascend_l0b.GetWithOffset<bfloat16_t>(16384, 32768);
  auto c1_sy_l0c = ascend_l0c.GetWithOffset<float>(2048, 16384);
  auto v1_op_max = ascend_ub.GetWithOffset<float>(32, 49664);
  auto v1_op_sum = ascend_ub.GetWithOffset<float>(32, 49792);
  auto v1_op_partial = ascend_ub.GetWithOffset<float>(128, 49920);
  auto v1_op_bmm1 = ascend_ub.GetWithOffset<float>(4096, 50432);
  auto v1_op_mask = ascend_ub.GetWithOffset<uint8_t>(16, 66816);
  auto v1_op_tmp = ascend_ub.GetWithOffset<float>(256, 66848);
  auto v1_op_out = ascend_ub.GetWithOffset<float>(128, 67872);
  auto v1_om_weight_half = ascend_ub.GetWithOffset<bfloat16_t>(16, 85280);
  auto v1_om_weight = ascend_ub.GetWithOffset<float>(16, 85312);
  auto v1_om_weight_2d = ascend_ub.GetWithOffset<float>(2048, 85376);
  auto v1_om_tmp = ascend_ub.GetWithOffset<float>(128, 93568);
  auto v1_om_local_max = ascend_ub.GetWithOffset<float>(1, 94080);
  auto v1_om_bmm2 = ascend_ub.GetWithOffset<float>(2048, 94112);
  auto v1_om_score = ascend_ub.GetWithOffset<float>(128, 102304);
  auto v1_om_mask = ascend_ub.GetWithOffset<uint8_t>(16, 102816);
  auto v1_om_reduce_2d = ascend_ub.GetWithOffset<float>(128, 102848);
  auto v1_om_tile_max = ascend_ub.GetWithOffset<float>(1, 103360);
  auto v1_om_floor = ascend_ub.GetWithOffset<float>(1, 103392);
  auto v1_os_stats = ascend_ub.GetWithOffset<float>(16, 103936);
  auto v1_os_weight_half = ascend_ub.GetWithOffset<bfloat16_t>(16, 104000);
  auto v1_os_global_max = ascend_ub.GetWithOffset<float>(1, 104032);
  auto v1_os_weight = ascend_ub.GetWithOffset<float>(16, 104064);
  auto v1_os_weight_2d = ascend_ub.GetWithOffset<float>(2048, 104128);
  auto v1_os_tmp = ascend_ub.GetWithOffset<float>(128, 112320);
  auto v1_os_global_max_2d = ascend_ub.GetWithOffset<float>(128, 112832);
  auto v1_os_local_sum = ascend_ub.GetWithOffset<float>(1, 113344);
  auto v1_os_bmm2 = ascend_ub.GetWithOffset<float>(2048, 113376);
  auto v1_os_score = ascend_ub.GetWithOffset<float>(128, 121568);
  auto v1_os_mask = ascend_ub.GetWithOffset<uint8_t>(16, 122080);
  auto v1_os_reduce_2d = ascend_ub.GetWithOffset<float>(128, 122112);
  auto v1_os_tile_sum = ascend_ub.GetWithOffset<float>(1, 122624);
  auto v1_ob_weight_half = ascend_ub.GetWithOffset<bfloat16_t>(16, 123168);
  auto v1_ob_stats = ascend_ub.GetWithOffset<float>(32, 123200);
  auto v1_ob_weight = ascend_ub.GetWithOffset<float>(16, 123328);
  auto v1_ob_weight_2d = ascend_ub.GetWithOffset<float>(1024, 123392);
  auto v1_ob_row_tmp = ascend_ub.GetWithOffset<float>(128, 127488);
  auto v1_ob_global_max = ascend_ub.GetWithOffset<float>(1, 128000);
  auto v1_ob_global_sum = ascend_ub.GetWithOffset<float>(1, 128032);
  auto v1_ob_global_max_2d = ascend_ub.GetWithOffset<float>(64, 128064);
  auto v1_ob_global_sum_2d = ascend_ub.GetWithOffset<float>(64, 128320);
  auto v1_ob_dw_accum = ascend_ub.GetWithOffset<float>(16, 128576);
  auto v1_ob_loss_accum = ascend_ub.GetWithOffset<float>(1, 128640);
  auto v1_ob_p = ascend_ub.GetWithOffset<float>(64, 128672);
  auto v1_ob_bmm2 = ascend_ub.GetWithOffset<float>(1024, 128928);
  auto v1_ob_work = ascend_ub.GetWithOffset<float>(1024, 133024);
  auto v1_ob_sy = ascend_ub.GetWithOffset<float>(64, 137120);
  auto v1_ob_mask = ascend_ub.GetWithOffset<uint8_t>(8, 137376);
  auto v1_ob_diff = ascend_ub.GetWithOffset<float>(64, 137408);
  auto v1_ob_diff_2d = ascend_ub.GetWithOffset<float>(1024, 137664);
  auto v1_ob_gate = ascend_ub.GetWithOffset<float>(1024, 141760);
  auto v1_ob_dw = ascend_ub.GetWithOffset<float>(16, 145856);
  auto v1_ob_loss_p = ascend_ub.GetWithOffset<float>(64, 145920);
  auto v1_ob_log_p = ascend_ub.GetWithOffset<float>(64, 146176);
  auto v1_ob_log_sy = ascend_ub.GetWithOffset<float>(64, 146432);
  auto v1_ob_loss_2d = ascend_ub.GetWithOffset<float>(64, 146688);
  auto v1_ob_loss_sum = ascend_ub.GetWithOffset<float>(1, 146944);
  auto v1_ob_rg_half = ascend_ub.GetWithOffset<bfloat16_t>(1024, 146976);
  auto v1_fdw_left = ascend_ub.GetWithOffset<float>(8, 149024);
  auto v1_fdw_right = ascend_ub.GetWithOffset<float>(8, 149056);
  auto v1_fdw_half = ascend_ub.GetWithOffset<bfloat16_t>(8, 149088);
  auto c2_dq_l0c = ascend_l0c.GetWithOffset<float>(2048, 90112);
  auto c2_rg_l1 = ascend_l1.GetWithOffset<bfloat16_t>(2048, 114688);
  auto c2_gki_l1 = ascend_l1.GetWithOffset<bfloat16_t>(16384, 118784);
  auto c2_dk_rg_l0a = ascend_l0a.GetWithOffset<bfloat16_t>(2048, 12288);
  auto c2_dk_qi_l0b = ascend_l0b.GetWithOffset<bfloat16_t>(2048, 0);
  auto c2_dq_rg_l0a = ascend_l0a.GetWithOffset<bfloat16_t>(2048, 16384);
  auto c2_dq_gki_l0b = ascend_l0b.GetWithOffset<bfloat16_t>(16384, 4096);
  auto c2_dk_l0c = ascend_l0c.GetWithOffset<float>(16384, 24576);
  auto v2_zero_batch = ascend_ub.GetWithOffset<float>(4096, 149120);
  auto v2_dk_batch_ping = ascend_ub.GetWithOffset<float>(2048, 165504);
  auto v2_dk_batch_pong = ascend_ub.GetWithOffset<float>(2048, 173696);
  auto v2_finalize_dq_fp32 = ascend_ub.GetWithOffset<float>(128, 181888);
  auto v2_finalize_dq_half = ascend_ub.GetWithOffset<bfloat16_t>(128, 182400);
  auto v2_finalize_dk_fp32 = ascend_ub.GetWithOffset<float>(4096, 0);
  auto v2_finalize_dk_half = ascend_ub.GetWithOffset<bfloat16_t>(4096, 182656);
  auto v2_loss_total = ascend_ub.GetWithOffset<float>(1, 190848);
  auto v2_loss_chunk = ascend_ub.GetWithOffset<float>(128, 190880);
  auto v2_loss_chunk_sum = ascend_ub.GetWithOffset<float>(1, 191392);
  auto vid = AscendC::GetSubBlockIdx();
  for (int32_t pipe_iter = 0; pipe_iter < 2; ++pipe_iter) {
    if (pipe_iter < 1) {
      if (((pipe_iter * 4) + cid) < 1) {
        if ASCEND_IS_AIV {
          AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
          for (int32_t k_outer = 0; k_outer < 2; ++k_outer) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            for (int32_t lane = 0; lane < 16; ++lane) {
              int32_t sparse_idx = SparseIndices.GetValue((((((pipe_iter * 16384) + (cid * 4096)) + (k_outer * 1024)) + (vid * 512)) + lane));
              if (0 <= sparse_idx) {
                tl::ascend::copy_gm_to_ub<bfloat16_t, 512, 1>(v0_key_ub[(lane * 512)], Key[(((pipe_iter * 2099200) + (cid * 524800)) + (sparse_idx * 512))], 524800, 1, 512, bfloat16_t(0.000000e+00f));
                tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 1>(v0_ki_ub[(lane * 128)], KeyIndex[(((pipe_iter * 524800) + (cid * 131200)) + (sparse_idx * 128))], 131200, 1, 128, bfloat16_t(0.000000e+00f));
                tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 1>(v0_rope_ub[(lane * 128)], KeyRope[(((pipe_iter * 524800) + (cid * 131200)) + (sparse_idx * 128))], 131200, 1, 128, bfloat16_t(0.000000e+00f));
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
                  int32_t sparse_idx_1 = SparseIndices.GetValue((((((((pipe_iter * 16384) + (cid * 4096)) + (k_outer * 1024)) + (vid * 512)) + (chunk_id * 16)) + lane_1) + 16));
                  if (0 <= sparse_idx_1) {
                    tl::ascend::copy_gm_to_ub<bfloat16_t, 512, 1>(v0_key_ub[((((chunk_id + 1) % 2) * 8192) + (lane_1 * 512))], Key[(((pipe_iter * 2099200) + (cid * 524800)) + (sparse_idx_1 * 512))], 524800, 1, 512, bfloat16_t(0.000000e+00f));
                    tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 1>(v0_ki_ub[((((chunk_id + 1) % 2) * 2048) + (lane_1 * 128))], KeyIndex[(((pipe_iter * 524800) + (cid * 131200)) + (sparse_idx_1 * 128))], 131200, 1, 128, bfloat16_t(0.000000e+00f));
                    tl::ascend::copy_gm_to_ub<bfloat16_t, 128, 1>(v0_rope_ub[((((chunk_id + 1) % 2) * 2048) + (lane_1 * 128))], KeyRope[(((pipe_iter * 524800) + (cid * 131200)) + (sparse_idx_1 * 128))], 131200, 1, 128, bfloat16_t(0.000000e+00f));
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
              tl::ascend::copy_ub_to_gm<bfloat16_t, 512, 16>(GatherKey_ws[(((((pipe_iter * 4194304) + (cid * 1048576)) + (k_outer * 524288)) + (vid * 262144)) + (chunk_id * 8192))], v0_key_ub[((chunk_id % 2) * 8192)], 512, 16, 512);
              tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 16>(GatherKeyIndex_ws[(((((pipe_iter * 1048576) + (cid * 262144)) + (k_outer * 131072)) + (vid * 65536)) + (chunk_id * 2048))], v0_ki_ub[((chunk_id % 2) * 2048)], 128, 16, 128);
              tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 16>(GatherKeyRope_ws[(((((pipe_iter * 1048576) + (cid * 262144)) + (k_outer * 131072)) + (vid * 65536)) + (chunk_id * 2048))], v0_rope_ub[((chunk_id % 2) * 2048)], 128, 16, 128);
              AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>((chunk_id % 2));
            }
            if ((pipe_iter + k_outer) == 0) {
              AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0);
              AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(2);
            } else {
              AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(1);
              AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(3);
            }
          }
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
        }
      }
    }
    if (pipe_iter == 1) {
      if (((pipe_iter * 4) + cid) < 5) {
        if ASCEND_IS_AIC {
          AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
          AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
          AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
          AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
          AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
          AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 32, 128>(c1_query_chunk_0_l1[0], Query[(((pipe_iter * 65536) + (cid * 16384)) - 65536)], 512, 32, 128, true);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 32, 128>(c1_query_chunk_1_l1[0], Query[(((pipe_iter * 65536) + (cid * 16384)) - 65408)], 512, 32, 128, true);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 32, 128>(c1_query_chunk_2_l1[0], Query[(((pipe_iter * 65536) + (cid * 16384)) - 65280)], 512, 32, 128, true);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 32, 128>(c1_query_chunk_3_l1[0], Query[(((pipe_iter * 65536) + (cid * 16384)) - 65152)], 512, 32, 128, true);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 32, 128>(c1_query_rope_cache_l1[0], QueryRope[(((pipe_iter * 16384) + (cid * 4096)) - 16384)], 128, 32, 128, true);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(4);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(4);
          tl::ascend::copy_gm_to_l1<bfloat16_t, 16, 128>(c1_qi_l1[0], QueryIndex[(((pipe_iter * 8192) + (cid * 2048)) - 8192)], 128, 16, 128, true);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(5);
          for (int32_t k_outer_1 = 0; k_outer_1 < 2; ++k_outer_1) {
            if ((pipe_iter + k_outer_1) == 1) {
              AscendC::CrossCoreWaitFlag(0);
            } else {
              AscendC::CrossCoreWaitFlag(1);
            }
            for (int32_t k_inner = 0; k_inner < 8; ++k_inner) {
              AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
              tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_key_l1[0], GatherKey_ws[(((((pipe_iter * 4194304) + (cid * 1048576)) + (k_outer_1 * 524288)) + (k_inner * 65536)) - 4194304)], 512, 128, 128, true);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
              for (int32_t d_idx = 0; d_idx < 4; ++d_idx) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
                if (d_idx == 0) {
                  tl::ascend::copy_l1_to_l0a<bfloat16_t, 32, 128, false>(c1_q_l0a[0], c1_query_chunk_0_l1[0], 32, 128);
                }
                if (d_idx == 1) {
                  tl::ascend::copy_l1_to_l0a<bfloat16_t, 32, 128, false>(c1_q_l0a[0], c1_query_chunk_1_l1[0], 32, 128);
                }
                if (d_idx == 2) {
                  tl::ascend::copy_l1_to_l0a<bfloat16_t, 32, 128, false>(c1_q_l0a[0], c1_query_chunk_2_l1[0], 32, 128);
                }
                if (d_idx == 3) {
                  tl::ascend::copy_l1_to_l0a<bfloat16_t, 32, 128, false>(c1_q_l0a[0], c1_query_chunk_3_l1[0], 32, 128);
                }
                tl::ascend::copy_l1_to_l0b<bfloat16_t, 128, 128, true>(c1_key_l0b[0], c1_key_l1[0], 128, 128);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
                if (d_idx < 3) {
                  AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
                  tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_key_l1[0], GatherKey_ws[((((((pipe_iter * 4194304) + (cid * 1048576)) + (k_outer_1 * 524288)) + (k_inner * 65536)) + (d_idx * 128)) - 4194176)], 512, 128, 128, true);
                  AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
                }
                if (d_idx == 3) {
                  AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
                  tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_key_l1[0], GatherKeyRope_ws[(((((pipe_iter * 1048576) + (cid * 262144)) + (k_outer_1 * 131072)) + (k_inner * 16384)) - 1048576)], 128, 128, 128, true);
                  AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
                tl::ascend::mma<bfloat16_t, float, 32, 128>(c1_q_l0a[0],c1_key_l0b[0],c1_p_l0c[0], (d_idx == 0), 128);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
              }
              AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
              tl::ascend::copy_l1_to_l0a<bfloat16_t, 32, 128, false>(c1_q_l0a[0], c1_query_rope_cache_l1[0], 32, 128);
              tl::ascend::copy_l1_to_l0b<bfloat16_t, 128, 128, true>(c1_key_l0b[0], c1_key_l1[0], 128, 128);
              AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
              AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
              AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
              AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
              tl::ascend::mma<bfloat16_t, float, 32, 128>(c1_q_l0a[0],c1_key_l0b[0],c1_p_l0c[0], (bool)0, 128);
              AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
              AscendC::SetFlag<AscendC::HardEvent::M_FIX>(0);
              AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(0);
              tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 32, 128, 0>(bmm1Res[(((((pipe_iter * 262144) + (cid * 65536)) + (k_outer_1 * 1024)) + (k_inner * 128)) - 262144)], c1_p_l0c[0], 2048, 32, 128, 0);
              AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
            }
          }
          AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
          AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
          AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
          AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(4);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(5);
          AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
          tl::ascend::copy_l1_to_l0a<bfloat16_t, 16, 128, false>(c1_qi_l0a[0], c1_qi_l1[0], 16, 128);
          AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
          for (int32_t k_outer_2 = 0; k_outer_2 < 2; ++k_outer_2) {
            if ((pipe_iter + k_outer_2) == 1) {
              AscendC::CrossCoreWaitFlag(2);
            } else {
              AscendC::CrossCoreWaitFlag(3);
            }
            for (int32_t k_inner_1 = 0; k_inner_1 < 8; ++k_inner_1) {
              AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
              tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c1_ki_l1[0], GatherKeyIndex_ws[(((((pipe_iter * 1048576) + (cid * 262144)) + (k_outer_2 * 131072)) + (k_inner_1 * 16384)) - 1048576)], 128, 128, 128, true);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(1);
              AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(1);
              tl::ascend::copy_l1_to_l0b<bfloat16_t, 128, 128, true>(c1_ki_l0b[0], c1_ki_l1[0], 128, 128);
              AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
              AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(1);
              AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(1);
              AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
              tl::ascend::mma<bfloat16_t, float, 16, 128>(c1_qi_l0a[0],c1_ki_l0b[0],c1_sy_l0c[0], (bool)1, 128);
              AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
              AscendC::SetFlag<AscendC::HardEvent::M_FIX>(1);
              AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(1);
              tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 16, 128, 1>(bmm2Res[(((((pipe_iter * 131072) + (cid * 32768)) + (k_outer_2 * 1024)) + (k_inner_1 * 128)) - 131072)], c1_sy_l0c[0], 2048, 16, 128, 0);
              AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
            }
          }
          AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
          AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
          AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
          AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(6);
        }
        if ASCEND_IS_AIV {
          AscendC::CrossCoreWaitFlag(4);
          tl::ascend::ArithProgression<float>(v1_op_offsets[0], 0.000000e+00f, 1.000000e+00f, 128);
          tl::ascend::copy_gm_to_ub<float, 32>(v1_op_max[0], SoftmaxMax[(((pipe_iter * 128) + (cid * 32)) - 128)], 32, 1, 32, 0.000000e+00f);
          tl::ascend::copy_gm_to_ub<float, 32>(v1_op_sum[0], SoftmaxSum[(((pipe_iter * 128) + (cid * 32)) - 128)], 32, 1, 32, 0.000000e+00f);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
          AscendC::Abs(v1_op_sum[0], v1_op_sum[0], 32);
          tl::ascend::Fill<float>(v1_op_partial[0], 1.000000e-08f, 32);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Max(v1_op_sum[0], v1_op_sum[0], v1_op_partial[0], 32);
          for (int32_t local_n = 0; local_n < 8; ++local_n) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            tl::ascend::copy_gm_to_ub<float, 128, 32>(v1_op_bmm1[0], bmm1Res[(((((pipe_iter * 262144) + (cid * 65536)) + (vid * 1024)) + (local_n * 128)) - 262144)], 2048, 32, 128, 0.000000e+00f);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
            {
            AscendC::Muls(v1_op_bmm1[0], v1_op_bmm1[0], 4.419417e-02f, 4096);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::CompareScalar(v1_op_mask[0], v1_op_offsets[0], float(((2048 - (local_n * 128)) - (vid * 1024))), AscendC::CMPMODE::LT, 128);
            AscendC::PipeBarrier<PIPE_V>();
            for (int32_t p_mask_row = 0; p_mask_row < 32; ++p_mask_row) {
AscendC::Select<float, uint8_t>(v1_op_bmm1[(p_mask_row * 128)], v1_op_mask[0], v1_op_bmm1[(p_mask_row * 128)], static_cast<float>(-CUDART_INF_F), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 128);
            }
            AscendC::PipeBarrier<PIPE_V>();
            for (int32_t p_col_chunk = 0; p_col_chunk < 2; ++p_col_chunk) {
              tl::ascend::brcb<float>(v1_op_tmp[0], v1_op_max[0], 4, 1, 8);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::sub_mask<float>(v1_op_bmm1[(p_col_chunk * 64)], v1_op_bmm1[(p_col_chunk * 64)], v1_op_tmp[0], 0xFFFFFFFFFFFFFFFF, 0, 32, 1, 1, 0, 16, 16, 1);
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(v1_op_bmm1[0], v1_op_bmm1[0], 4096);
            AscendC::PipeBarrier<PIPE_V>();
            for (int32_t p_col_chunk_1 = 0; p_col_chunk_1 < 2; ++p_col_chunk_1) {
              tl::ascend::brcb<float>(v1_op_tmp[0], v1_op_sum[0], 4, 1, 8);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::div_mask<float>(v1_op_bmm1[(p_col_chunk_1 * 64)], v1_op_bmm1[(p_col_chunk_1 * 64)], v1_op_tmp[0], 0xFFFFFFFFFFFFFFFF, 0, 32, 1, 1, 0, 16, 16, 1);
            }
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::Fill<float>(v1_op_out[0], 0.000000e+00f, 128);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::reduce_sum<float,  32,  128,  0>(v1_op_partial[0], v1_op_bmm1[0], tmp_ub[0], true);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(v1_op_out[0], v1_op_out[0], v1_op_partial[0], 128);
            AscendC::PipeBarrier<PIPE_V>();
            {
            AscendC::Muls(v1_op_out[0], v1_op_out[0], 3.125000e-02f, 128);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            tl::ascend::copy_ub_to_gm<float, 128>(PReduced[(((((pipe_iter * 8192) + (cid * 2048)) + (vid * 1024)) + (local_n * 128)) - 8192)], v1_op_out[0], 2048, 1, 128);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
          }
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        }
        if ASCEND_IS_AIV {
          AscendC::CrossCoreWaitFlag(6);
          tl::ascend::ArithProgression<float>(v1_om_offsets[0], 0.000000e+00f, 1.000000e+00f, 128);
          tl::ascend::copy_gm_to_ub<bfloat16_t, 16>(v1_om_weight_half[0], Weights[(((pipe_iter * 64) + (cid * 16)) - 64)], 16, 1, 16, bfloat16_t(0.000000e+00f));
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
          AscendC::Cast(v1_om_weight[0], v1_om_weight_half[0], AscendC::RoundMode::CAST_NONE, 16);
          tl::ascend::Fill<float>(v1_om_weight_2d[0], 1.000000e+00f, 2048);
          AscendC::PipeBarrier<PIPE_V>();
          for (int32_t sy_col_chunk = 0; sy_col_chunk < 2; ++sy_col_chunk) {
            tl::ascend::brcb<float>(v1_om_tmp[0], v1_om_weight[0], 2, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::mul_mask<float>(v1_om_weight_2d[(sy_col_chunk * 64)], v1_om_weight_2d[(sy_col_chunk * 64)], v1_om_tmp[0], 0xFFFFFFFFFFFFFFFF, 0, 16, 1, 1, 0, 16, 16, 1);
          }
          tl::ascend::Fill<float>(v1_om_local_max[0], -CUDART_INF_F, 1);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
          for (int32_t local_n_1 = 0; local_n_1 < 8; ++local_n_1) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
            tl::ascend::copy_gm_to_ub<float, 128, 16>(v1_om_bmm2[0], bmm2Res[(((((pipe_iter * 131072) + (cid * 32768)) + (vid * 1024)) + (local_n_1 * 128)) - 131072)], 2048, 16, 128, 0.000000e+00f);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);
            AscendC::Mul(v1_om_bmm2[0], v1_om_bmm2[0], v1_om_weight_2d[0], 2048);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::reduce_sum<float,  16,  128,  0>(v1_om_score[0], v1_om_bmm2[0], tmp_ub[0], true);
            AscendC::CompareScalar(v1_om_mask[0], v1_om_offsets[0], float(((2048 - (local_n_1 * 128)) - (vid * 1024))), AscendC::CMPMODE::LT, 128);
            AscendC::PipeBarrier<PIPE_V>();
AscendC::Select<float, uint8_t>(v1_om_score[0], v1_om_mask[0], v1_om_score[0], static_cast<float>(-CUDART_INF_F), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 128);
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
          tl::ascend::copy_ub_to_gm<float, 1>(V1Stats[((((pipe_iter * 128) + (cid * 32)) + (vid * 8)) - 128)], v1_om_local_max[0], 32, 1, 1);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
          AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(10);
          AscendC::CrossCoreWaitFlag(12);
        }
        if ASCEND_IS_AIC {
          AscendC::CrossCoreWaitFlag(10);
          AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(12);
        }
        if ASCEND_IS_AIV {
          tl::ascend::ArithProgression<float>(v1_os_offsets[0], 0.000000e+00f, 1.000000e+00f, 128);
          tl::ascend::copy_gm_to_ub<float, 8, 2>(v1_os_stats[0], V1Stats[(((pipe_iter * 128) + (cid * 32)) - 128)], 8, 2, 8, 0.000000e+00f);
          tl::ascend::copy_gm_to_ub<bfloat16_t, 16>(v1_os_weight_half[0], Weights[(((pipe_iter * 64) + (cid * 16)) - 64)], 16, 1, 16, bfloat16_t(0.000000e+00f));
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
          AscendC::Max(v1_os_global_max[0], v1_os_stats[0], v1_os_stats[8], 1);
          AscendC::Cast(v1_os_weight[0], v1_os_weight_half[0], AscendC::RoundMode::CAST_NONE, 16);
          tl::ascend::Fill<float>(v1_os_weight_2d[0], 1.000000e+00f, 2048);
          AscendC::PipeBarrier<PIPE_V>();
          for (int32_t sy_col_chunk_1 = 0; sy_col_chunk_1 < 2; ++sy_col_chunk_1) {
            tl::ascend::brcb<float>(v1_os_tmp[0], v1_os_weight[0], 2, 1, 8);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::mul_mask<float>(v1_os_weight_2d[(sy_col_chunk_1 * 64)], v1_os_weight_2d[(sy_col_chunk_1 * 64)], v1_os_tmp[0], 0xFFFFFFFFFFFFFFFF, 0, 16, 1, 1, 0, 16, 16, 1);
          }
          tl::ascend::Broadcast<float, 2, 1, false>(v1_os_global_max_2d[0],v1_os_global_max[0],tmp_ub,(uint32_t[]){1, 128}, (uint32_t[]){1, 1});
          tl::ascend::Fill<float>(v1_os_local_sum[0], 0.000000e+00f, 1);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
          for (int32_t local_n_2 = 0; local_n_2 < 8; ++local_n_2) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
            tl::ascend::copy_gm_to_ub<float, 128, 16>(v1_os_bmm2[0], bmm2Res[(((((pipe_iter * 131072) + (cid * 32768)) + (vid * 1024)) + (local_n_2 * 128)) - 131072)], 2048, 16, 128, 0.000000e+00f);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);
            AscendC::Mul(v1_os_bmm2[0], v1_os_bmm2[0], v1_os_weight_2d[0], 2048);
            AscendC::PipeBarrier<PIPE_V>();
            tl::ascend::reduce_sum<float,  16,  128,  0>(v1_os_score[0], v1_os_bmm2[0], tmp_ub[0], true);
            AscendC::CompareScalar(v1_os_mask[0], v1_os_offsets[0], float(((2048 - (local_n_2 * 128)) - (vid * 1024))), AscendC::CMPMODE::LT, 128);
            AscendC::PipeBarrier<PIPE_V>();
AscendC::Select<float, uint8_t>(v1_os_score[0], v1_os_mask[0], v1_os_score[0], static_cast<float>(-CUDART_INF_F), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 128);
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
          tl::ascend::copy_ub_to_gm<float, 1>(V1Stats[((((pipe_iter * 128) + (cid * 32)) + (vid * 8)) - 112)], v1_os_local_sum[0], 32, 1, 1);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
          AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(10);
          AscendC::CrossCoreWaitFlag(12);
        }
        if ASCEND_IS_AIC {
          AscendC::CrossCoreWaitFlag(10);
          AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(12);
        }
        if ASCEND_IS_AIV {
          tl::ascend::ArithProgression<float>(v1_ob_offsets[0], 0.000000e+00f, 1.000000e+00f, 64);
          tl::ascend::Fill<float>(v1_ob_eps[0], 1.000000e-08f, 64);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(2);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
          tl::ascend::copy_gm_to_ub<bfloat16_t, 16>(v1_ob_weight_half[0], Weights[(((pipe_iter * 64) + (cid * 16)) - 64)], 16, 1, 16, bfloat16_t(0.000000e+00f));
          tl::ascend::copy_gm_to_ub<float, 8, 4>(v1_ob_stats[0], V1Stats[(((pipe_iter * 128) + (cid * 32)) - 128)], 8, 4, 8, 0.000000e+00f);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
          AscendC::Cast(v1_ob_weight[0], v1_ob_weight_half[0], AscendC::RoundMode::CAST_NONE, 16);
          tl::ascend::Fill<float>(v1_ob_weight_2d[0], 1.000000e+00f, 1024);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::brcb<float>(v1_ob_row_tmp[0], v1_ob_weight[0], 2, 1, 8);
          AscendC::PipeBarrier<PIPE_V>();
          tl::ascend::mul_mask<float>(v1_ob_weight_2d[0], v1_ob_weight_2d[0], v1_ob_row_tmp[0], 0xFFFFFFFFFFFFFFFF, 0, 16, 1, 1, 0, 8, 8, 1);
          AscendC::Max(v1_ob_global_max[0], v1_ob_stats[0], v1_ob_stats[8], 1);
          AscendC::Add(v1_ob_global_sum[0], v1_ob_stats[16], v1_ob_stats[24], 1);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Max(v1_ob_global_sum[0], v1_ob_global_sum[0], v1_ob_eps[0], 1);
          tl::ascend::Broadcast<float, 2, 1, false>(v1_ob_global_max_2d[0],v1_ob_global_max[0],tmp_ub,(uint32_t[]){1, 64}, (uint32_t[]){1, 1});
          tl::ascend::Broadcast<float, 2, 1, false>(v1_ob_global_sum_2d[0],v1_ob_global_sum[0],tmp_ub,(uint32_t[]){1, 64}, (uint32_t[]){1, 1});
          tl::ascend::Fill<float>(v1_ob_dw_accum[0], 0.000000e+00f, 16);
          tl::ascend::Fill<float>(v1_ob_loss_accum[0], 0.000000e+00f, 1);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
          for (int32_t local_n_3 = 0; local_n_3 < 8; ++local_n_3) {
            for (int32_t bw_col_id = 0; bw_col_id < 2; ++bw_col_id) {
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
              tl::ascend::copy_gm_to_ub<float, 64>(v1_ob_p[0], PReduced[((((((pipe_iter * 8192) + (cid * 2048)) + (vid * 1024)) + (local_n_3 * 128)) + (bw_col_id * 64)) - 8192)], 2048, 1, 64, 0.000000e+00f);
              tl::ascend::copy_gm_to_ub<float, 64, 16>(v1_ob_bmm2[0], bmm2Res[((((((pipe_iter * 131072) + (cid * 32768)) + (vid * 1024)) + (local_n_3 * 128)) + (bw_col_id * 64)) - 131072)], 2048, 16, 64, 0.000000e+00f);
              AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(3);
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(3);
              AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(2);
              AscendC::Mul(v1_ob_work[0], v1_ob_bmm2[0], v1_ob_weight_2d[0], 1024);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::reduce_sum<float,  16,  64,  0>(v1_ob_sy[0], v1_ob_work[0], tmp_ub[0], true);
              AscendC::CompareScalar(v1_ob_mask[0], v1_ob_offsets[0], float((((2048 - (bw_col_id * 64)) - (local_n_3 * 128)) - (vid * 1024))), AscendC::CMPMODE::LT, 64);
              AscendC::PipeBarrier<PIPE_V>();
AscendC::Select<float, uint8_t>(v1_ob_sy[0], v1_ob_mask[0], v1_ob_sy[0], static_cast<float>(-CUDART_INF_F), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 64);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Sub(v1_ob_sy[0], v1_ob_sy[0], v1_ob_global_max_2d[0], 64);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Exp(v1_ob_sy[0], v1_ob_sy[0], 64);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Div(v1_ob_sy[0], v1_ob_sy[0], v1_ob_global_sum_2d[0], 64);
              AscendC::Sub(v1_ob_diff[0], v1_ob_sy[0], v1_ob_p[0], 64);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::Broadcast<float, 2, 0, false>(v1_ob_diff_2d[0],v1_ob_diff[0],tmp_ub,(uint32_t[]){16, 64}, (uint32_t[]){1, 64});
              {
              AscendC::Muls(v1_ob_gate[0], v1_ob_bmm2[0], -1.000000e+04f, 1024);
              }
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Exp(v1_ob_gate[0], v1_ob_gate[0], 1024);
              AscendC::PipeBarrier<PIPE_V>();
              {
              AscendC::Muls(v1_ob_gate[0], v1_ob_gate[0], -1.000000e+00f, 1024);
              }
              AscendC::PipeBarrier<PIPE_V>();
              {
              AscendC::Adds(v1_ob_gate[0], v1_ob_gate[0], 1.000000e+00f, 1024);
              }
              AscendC::Mul(v1_ob_work[0], v1_ob_bmm2[0], v1_ob_diff_2d[0], 1024);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::reduce_sum<float,  16,  64,  -1>(v1_ob_dw[0], v1_ob_work[0], tmp_ub[0], true);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Add(v1_ob_dw_accum[0], v1_ob_dw_accum[0], v1_ob_dw[0], 16);
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
              AscendC::CompareScalar(v1_ob_mask[0], v1_ob_offsets[0], float((((2048 - (bw_col_id * 64)) - (local_n_3 * 128)) - (vid * 1024))), AscendC::CMPMODE::LT, 64);
              AscendC::PipeBarrier<PIPE_V>();
AscendC::Select<float, uint8_t>(v1_ob_loss_2d[0], v1_ob_mask[0], v1_ob_loss_2d[0], static_cast<float>(0.000000e+00f), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 64);
              AscendC::PipeBarrier<PIPE_V>();
              tl::ascend::reduce_sum<float,  1,  64,  -1>(v1_ob_loss_sum[0], v1_ob_loss_2d[0], tmp_ub[0], true);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Add(v1_ob_loss_accum[0], v1_ob_loss_accum[0], v1_ob_loss_sum[0], 1);
              AscendC::Mul(v1_ob_bmm2[0], v1_ob_weight_2d[0], v1_ob_diff_2d[0], 1024);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Mul(v1_ob_bmm2[0], v1_ob_bmm2[0], v1_ob_gate[0], 1024);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::CompareScalar(v1_ob_mask[0], v1_ob_offsets[0], float((((2048 - (bw_col_id * 64)) - (local_n_3 * 128)) - (vid * 1024))), AscendC::CMPMODE::LT, 64);
              AscendC::PipeBarrier<PIPE_V>();
              for (int32_t rg_mask_row = 0; rg_mask_row < 16; ++rg_mask_row) {
AscendC::Select<float, uint8_t>(v1_ob_bmm2[(rg_mask_row * 64)], v1_ob_mask[0], v1_ob_bmm2[(rg_mask_row * 64)], static_cast<float>(0.000000e+00f), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 64);
              }
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::Cast(v1_ob_rg_half[0], v1_ob_bmm2[0], AscendC::RoundMode::CAST_RINT, 1024);
              AscendC::PipeBarrier<PIPE_V>();
              AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(2);
              AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(2);
              AscendC::PipeBarrier<PIPE_MTE3>();
              tl::ascend::copy_ub_to_gm<bfloat16_t, 64, 16>(reluGradRes[((((((pipe_iter * 131072) + (cid * 32768)) + (vid * 1024)) + (local_n_3 * 128)) + (bw_col_id * 64)) - 131072)], v1_ob_rg_half[0], 2048, 16, 64);
              AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(2);
              AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(3);
            }
          }
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(3);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
          tl::ascend::copy_ub_to_gm<float, 16>(DWeightPartial[((((pipe_iter * 128) + (cid * 32)) + (vid * 16)) - 128)], v1_ob_dw_accum[0], 32, 1, 16);
          tl::ascend::copy_ub_to_gm<float, 1>(LossPartial[((((pipe_iter * 64) + (cid * 16)) + (vid * 8)) - 64)], v1_ob_loss_accum[0], 128, 1, 1);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(2);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
          AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(10);
          AscendC::CrossCoreWaitFlag(12);
        }
        if ASCEND_IS_AIC {
          AscendC::CrossCoreWaitFlag(10);
          AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(12);
        }
        if ASCEND_IS_AIV {
          tl::ascend::copy_gm_to_ub<float, 8>(v1_fdw_left[0], DWeightPartial[((((pipe_iter * 128) + (cid * 32)) + (vid * 8)) - 128)], 32, 1, 8, 0.000000e+00f);
          tl::ascend::copy_gm_to_ub<float, 8>(v1_fdw_right[0], DWeightPartial[((((pipe_iter * 128) + (cid * 32)) + (vid * 8)) - 112)], 32, 1, 8, 0.000000e+00f);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2);
          AscendC::Add(v1_fdw_left[0], v1_fdw_left[0], v1_fdw_right[0], 8);
          AscendC::PipeBarrier<PIPE_V>();
          AscendC::Cast(v1_fdw_half[0], v1_fdw_left[0], AscendC::RoundMode::CAST_RINT, 8);
          AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(3);
          AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(3);
          tl::ascend::copy_ub_to_gm<bfloat16_t, 8>(dWeightOut[((((pipe_iter * 64) + (cid * 16)) + (vid * 8)) - 64)], v1_fdw_half[0], 16, 1, 8);
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(3);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(3);
          AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(8);
        }
        if ASCEND_IS_AIC {
          tl::ascend::copy_gm_to_l1<bfloat16_t, 16, 128>(c2_qi_l1[0], QueryIndex[(((pipe_iter * 8192) + (cid * 2048)) - 8192)], 128, 16, 128, true);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(6);
          AscendC::CrossCoreWaitFlag(8);
          AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(2);
          AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2);
          AscendC::SetFlag<AscendC::HardEvent::FIX_M>(2);
          AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3);
          AscendC::SetFlag<AscendC::HardEvent::FIX_M>(3);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(6);
          for (int32_t n_idx = 0; n_idx < 16; ++n_idx) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(2);
            tl::ascend::copy_gm_to_l1<bfloat16_t, 16, 128>(c2_rg_l1[0], reluGradRes[((((pipe_iter * 131072) + (cid * 32768)) + (n_idx * 128)) - 131072)], 2048, 16, 128, true);
            tl::ascend::copy_gm_to_l1<bfloat16_t, 128, 128>(c2_gki_l1[0], GatherKeyIndex_ws[((((pipe_iter * 1048576) + (cid * 262144)) + (n_idx * 16384)) - 1048576)], 128, 128, 128, true);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(2);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(2);
            tl::ascend::copy_l1_to_l0a<bfloat16_t, 16, 128, true>(c2_dk_rg_l0a[0], c2_rg_l1[0], 128, 16);
            tl::ascend::copy_l1_to_l0b<bfloat16_t, 16, 128, false>(c2_dk_qi_l0b[0], c2_qi_l1[0], 16, 128);
            tl::ascend::copy_l1_to_l0a<bfloat16_t, 16, 128, false>(c2_dq_rg_l0a[0], c2_rg_l1[0], 16, 128);
            tl::ascend::copy_l1_to_l0b<bfloat16_t, 128, 128, false>(c2_dq_gki_l0b[0], c2_gki_l1[0], 128, 128);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(2);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(2);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(2);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(2);
            tl::ascend::mma<bfloat16_t, float, 128, 128>(c2_dk_rg_l0a[0],c2_dk_qi_l0b[0],c2_dk_l0c[0], (bool)1, 16);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(2);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(2);
            tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 128, 128, 0>(dk_matmul_partial[((((pipe_iter * 1048576) + (cid * 262144)) + (n_idx * 16384)) - 1048576)], c2_dk_l0c[0], 128, 128, 128, 0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(3);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(3);
            tl::ascend::mma<bfloat16_t, float, 16, 128>(c2_dq_rg_l0a[0],c2_dq_gki_l0b[0],c2_dq_l0c[0], (n_idx == 0), 128);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3);
            if (n_idx < 15) {
              AscendC::SetFlag<AscendC::HardEvent::FIX_M>(3);
            }
          }
          AscendC::SetFlag<AscendC::HardEvent::M_FIX>(3);
          AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(3);
          tl::ascend::copy_l0c_to_gm<float, float, layout::RowMajor, 16, 128, 0>(dQueryIndexAccum[(((pipe_iter * 8192) + (cid * 2048)) - 8192)], c2_dq_l0c[0], 128, 16, 128, 0);
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
    for (int32_t clear_iter = 0; clear_iter < 5; ++clear_iter) {
      tl::ascend::copy_ub_to_gm<float, 128, 32>(dKeyIndexAccum[(((cid * 40960) + (vid * 20480)) + (clear_iter * 4096))], v2_zero_batch[0], 128, 32, 128);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(5);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(5);
  }
  AscendC::SyncAll<false>();
  if ASCEND_IS_AIV {
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(6);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(7);
    int32_t valid_k_size = KRealSize.GetValue(0);
    int32_t condval;
    if ((cid < 1)) {
      condval = 1;
    } else {
      condval = 0;
    }
    if ((((cid * 256) + ((cid < 1 ? cid : 1) * 16)) + ((vid * ((condval + 17) / 2)) * 16)) < valid_k_size) {
      int32_t condval_1;
      if ((cid < 1)) {
        condval_1 = 1;
      } else {
        condval_1 = 0;
      }
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
      if ((((((cid * 256) + (((condval_1 + 17) / 2) * 16)) + ((cid < 1 ? cid : 1) * 16)) + ((vid * ((condval_2 + 17) / 2)) * 16)) - ((vid * (1 - ((condval_3 + 1) % 2))) * 16)) <= valid_k_size) {
        if ASCEND_IS_AIV {
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(6);
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
          tl::ascend::copy_gm_to_ub<float, 128, 16>(v2_dk_batch_ping[0], dk_matmul_partial[(((cid * 32768) + ((cid < 1 ? cid : 1) * 2048)) + ((vid * ((condval_4 + 17) / 2)) * 2048))], 128, ((1 <= (((128 - (cid < 1 ? cid : 1)) - (vid * ((condval_5 + 17) / 2))) - (cid * 16))) ? 16 : ((((((cid < 1 ? cid : 1) + (vid * ((condval_6 + 17) / 2))) / 16) + cid) < 8) ? (((2048 - ((cid < 1 ? cid : 1) * 16)) - ((vid * ((condval_7 + 17) / 2)) * 16)) - (cid * 256)) : 0)), 128, 0.000000e+00f);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(6);
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
          for (int32_t batch_id = 0; batch_id < (((condval_8 + 17) / 2) - (vid * (1 - ((condval_9 + 1) % 2)))); ++batch_id) {
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
            if ((batch_id + 1) < (((condval_10 + 17) / 2) - (vid * (1 - ((condval_11 + 1) % 2))))) {
              AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((((batch_id + 1) % 2) + 6));
              if (((batch_id + 1) % 2) == 0) {
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
                tl::ascend::copy_gm_to_ub<float, 128, 16>(v2_dk_batch_ping[0], dk_matmul_partial[(((((cid * 32768) + (batch_id * 2048)) + ((cid < 1 ? cid : 1) * 2048)) + ((vid * ((condval_12 + 17) / 2)) * 2048)) + 2048)], 128, ((1 <= ((((127 - batch_id) - (cid < 1 ? cid : 1)) - (vid * ((condval_13 + 17) / 2))) - (cid * 16))) ? 16 : ((((((cid * 16) + (cid < 1 ? cid : 1)) + batch_id) + (vid * ((condval_14 + 17) / 2))) < 127) ? ((((2032 - (batch_id * 16)) - ((cid < 1 ? cid : 1) * 16)) - ((vid * ((condval_15 + 17) / 2)) * 16)) - (cid * 256)) : 0)), 128, 0.000000e+00f);
              } else {
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
                int32_t condval_18;
                if ((cid < 1)) {
                  condval_18 = 1;
                } else {
                  condval_18 = 0;
                }
                int32_t condval_19;
                if ((cid < 1)) {
                  condval_19 = 1;
                } else {
                  condval_19 = 0;
                }
                tl::ascend::copy_gm_to_ub<float, 128, 16>(v2_dk_batch_pong[0], dk_matmul_partial[(((((cid * 32768) + (batch_id * 2048)) + ((cid < 1 ? cid : 1) * 2048)) + ((vid * ((condval_16 + 17) / 2)) * 2048)) + 2048)], 128, ((1 <= ((((127 - batch_id) - (cid < 1 ? cid : 1)) - (vid * ((condval_17 + 17) / 2))) - (cid * 16))) ? 16 : ((((((cid * 16) + (cid < 1 ? cid : 1)) + batch_id) + (vid * ((condval_18 + 17) / 2))) < 127) ? ((((2032 - (batch_id * 16)) - ((cid < 1 ? cid : 1) * 16)) - ((vid * ((condval_19 + 17) / 2)) * 16)) - (cid * 256)) : 0)), 128, 0.000000e+00f);
              }
              AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>((((batch_id + 1) % 2) + 6));
            }
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(((batch_id % 2) + 6));
            for (int32_t pair_idx = 0; pair_idx < 8; ++pair_idx) {
              int64_t condval_20;
              if ((((int64_t)cid) < (int64_t)1)) {
                condval_20 = (int64_t)1;
              } else {
                condval_20 = (int64_t)0;
              }
              int32_t sparse_idx0 = SparseIndices.GetValue((((((((int64_t)cid) * (int64_t)256) + (((int64_t)batch_id) * (int64_t)16)) + ((((int64_t)cid) < (int64_t)1 ? ((int64_t)cid) : (int64_t)1) * (int64_t)16)) + ((((int64_t)vid) * ((condval_20 + (int64_t)17) / (int64_t)2)) * (int64_t)16)) + (((int64_t)pair_idx) * (int64_t)2)));
              int64_t condval_21;
              if ((((int64_t)cid) < (int64_t)1)) {
                condval_21 = (int64_t)1;
              } else {
                condval_21 = (int64_t)0;
              }
              int32_t sparse_idx1 = SparseIndices.GetValue(((((((((int64_t)cid) * (int64_t)256) + (((int64_t)batch_id) * (int64_t)16)) + ((((int64_t)cid) < (int64_t)1 ? ((int64_t)cid) : (int64_t)1) * (int64_t)16)) + ((((int64_t)vid) * ((condval_21 + (int64_t)17) / (int64_t)2)) * (int64_t)16)) + (((int64_t)pair_idx) * (int64_t)2)) + (int64_t)1));
              if (0 <= sparse_idx0) {
                if ((batch_id % 2) == 0) {
                  tl::ascend::atomic_add_ub_to_gm<float, 128, 1>(dKeyIndexAccum[(sparse_idx0 * 128)], v2_dk_batch_ping[(pair_idx * 256)], 163840, ((sparse_idx0 <= 1279) ? 1 : 0), 128);
                } else {
                  tl::ascend::atomic_add_ub_to_gm<float, 128, 1>(dKeyIndexAccum[(sparse_idx0 * 128)], v2_dk_batch_pong[(pair_idx * 256)], 163840, ((sparse_idx0 <= 1279) ? 1 : 0), 128);
                }
              }
              if (0 <= sparse_idx1) {
                if ((batch_id % 2) == 0) {
                  tl::ascend::atomic_add_ub_to_gm<float, 128, 1>(dKeyIndexAccum[(sparse_idx1 * 128)], v2_dk_batch_ping[((pair_idx * 256) + 128)], 163840, ((sparse_idx1 <= 1279) ? 1 : 0), 128);
                } else {
                  tl::ascend::atomic_add_ub_to_gm<float, 128, 1>(dKeyIndexAccum[(sparse_idx1 * 128)], v2_dk_batch_pong[((pair_idx * 256) + 128)], 163840, ((sparse_idx1 <= 1279) ? 1 : 0), 128);
                }
              }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(((batch_id % 2) + 6));
          }
        }
      } else {
        if ASCEND_IS_AIV {
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(6);
          int32_t condval_22;
          if ((cid < 1)) {
            condval_22 = 1;
          } else {
            condval_22 = 0;
          }
          int32_t condval_23;
          if ((cid < 1)) {
            condval_23 = 1;
          } else {
            condval_23 = 0;
          }
          int32_t condval_24;
          if ((cid < 1)) {
            condval_24 = 1;
          } else {
            condval_24 = 0;
          }
          int32_t condval_25;
          if ((cid < 1)) {
            condval_25 = 1;
          } else {
            condval_25 = 0;
          }
          tl::ascend::copy_gm_to_ub<float, 128, 16>(v2_dk_batch_ping[0], dk_matmul_partial[(((cid * 32768) + ((cid < 1 ? cid : 1) * 2048)) + ((vid * ((condval_22 + 17) / 2)) * 2048))], 128, ((1 <= (((128 - (cid < 1 ? cid : 1)) - (vid * ((condval_23 + 17) / 2))) - (cid * 16))) ? 16 : ((((((cid < 1 ? cid : 1) + (vid * ((condval_24 + 17) / 2))) / 16) + cid) < 8) ? (((2048 - ((cid < 1 ? cid : 1) * 16)) - ((vid * ((condval_25 + 17) / 2)) * 16)) - (cid * 256)) : 0)), 128, 0.000000e+00f);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(6);
          int32_t condval_26;
          if ((cid < 1)) {
            condval_26 = 1;
          } else {
            condval_26 = 0;
          }
          int32_t condval_27;
          if ((cid < 1)) {
            condval_27 = 1;
          } else {
            condval_27 = 0;
          }
          for (int32_t batch_id_1 = 0; batch_id_1 < (((condval_26 + 17) / 2) - (vid * (1 - ((condval_27 + 1) % 2)))); ++batch_id_1) {
            int32_t condval_28;
            if ((cid < 1)) {
              condval_28 = 1;
            } else {
              condval_28 = 0;
            }
            int32_t condval_29;
            if ((cid < 1)) {
              condval_29 = 1;
            } else {
              condval_29 = 0;
            }
            if ((batch_id_1 + 1) < (((condval_28 + 17) / 2) - (vid * (1 - ((condval_29 + 1) % 2))))) {
              int32_t condval_31;
              if ((cid < 1)) {
                condval_31 = 1;
              } else {
                condval_31 = 0;
              }
              int32_t condval_30;
              if (((((((cid * 256) + (batch_id_1 * 16)) + ((cid < 1 ? cid : 1) * 16)) + ((vid * ((condval_31 + 17) / 2)) * 16)) + 16) < valid_k_size)) {
                condval_30 = 1;
              } else {
                condval_30 = 0;
              }
              if (0 < condval_30) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>((((batch_id_1 + 1) % 2) + 6));
                if (((batch_id_1 + 1) % 2) == 0) {
                  int32_t condval_32;
                  if ((cid < 1)) {
                    condval_32 = 1;
                  } else {
                    condval_32 = 0;
                  }
                  int32_t condval_33;
                  if ((cid < 1)) {
                    condval_33 = 1;
                  } else {
                    condval_33 = 0;
                  }
                  int32_t condval_34;
                  if ((cid < 1)) {
                    condval_34 = 1;
                  } else {
                    condval_34 = 0;
                  }
                  int32_t condval_35;
                  if ((cid < 1)) {
                    condval_35 = 1;
                  } else {
                    condval_35 = 0;
                  }
                  tl::ascend::copy_gm_to_ub<float, 128, 16>(v2_dk_batch_ping[0], dk_matmul_partial[(((((cid * 32768) + (batch_id_1 * 2048)) + ((cid < 1 ? cid : 1) * 2048)) + ((vid * ((condval_32 + 17) / 2)) * 2048)) + 2048)], 128, ((1 <= ((((127 - batch_id_1) - (cid < 1 ? cid : 1)) - (vid * ((condval_33 + 17) / 2))) - (cid * 16))) ? 16 : ((((((cid * 16) + (cid < 1 ? cid : 1)) + batch_id_1) + (vid * ((condval_34 + 17) / 2))) < 127) ? ((((2032 - (batch_id_1 * 16)) - ((cid < 1 ? cid : 1) * 16)) - ((vid * ((condval_35 + 17) / 2)) * 16)) - (cid * 256)) : 0)), 128, 0.000000e+00f);
                } else {
                  int32_t condval_36;
                  if ((cid < 1)) {
                    condval_36 = 1;
                  } else {
                    condval_36 = 0;
                  }
                  int32_t condval_37;
                  if ((cid < 1)) {
                    condval_37 = 1;
                  } else {
                    condval_37 = 0;
                  }
                  int32_t condval_38;
                  if ((cid < 1)) {
                    condval_38 = 1;
                  } else {
                    condval_38 = 0;
                  }
                  int32_t condval_39;
                  if ((cid < 1)) {
                    condval_39 = 1;
                  } else {
                    condval_39 = 0;
                  }
                  tl::ascend::copy_gm_to_ub<float, 128, 16>(v2_dk_batch_pong[0], dk_matmul_partial[(((((cid * 32768) + (batch_id_1 * 2048)) + ((cid < 1 ? cid : 1) * 2048)) + ((vid * ((condval_36 + 17) / 2)) * 2048)) + 2048)], 128, ((1 <= ((((127 - batch_id_1) - (cid < 1 ? cid : 1)) - (vid * ((condval_37 + 17) / 2))) - (cid * 16))) ? 16 : ((((((cid * 16) + (cid < 1 ? cid : 1)) + batch_id_1) + (vid * ((condval_38 + 17) / 2))) < 127) ? ((((2032 - (batch_id_1 * 16)) - ((cid < 1 ? cid : 1) * 16)) - ((vid * ((condval_39 + 17) / 2)) * 16)) - (cid * 256)) : 0)), 128, 0.000000e+00f);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>((((batch_id_1 + 1) % 2) + 6));
              }
            }
            int32_t condval_41;
            if ((cid < 1)) {
              condval_41 = 1;
            } else {
              condval_41 = 0;
            }
            int32_t condval_40;
            if ((((((cid * 256) + (batch_id_1 * 16)) + ((cid < 1 ? cid : 1) * 16)) + ((vid * ((condval_41 + 17) / 2)) * 16)) < valid_k_size)) {
              condval_40 = 1;
            } else {
              condval_40 = 0;
            }
            if (0 < condval_40) {
              AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(((batch_id_1 % 2) + 6));
              for (int32_t pair_idx_1 = 0; pair_idx_1 < 8; ++pair_idx_1) {
                int64_t condval_42;
                if ((((int64_t)cid) < (int64_t)1)) {
                  condval_42 = (int64_t)1;
                } else {
                  condval_42 = (int64_t)0;
                }
                int32_t sparse_idx0_1 = SparseIndices.GetValue((((((((int64_t)cid) * (int64_t)256) + (((int64_t)batch_id_1) * (int64_t)16)) + ((((int64_t)cid) < (int64_t)1 ? ((int64_t)cid) : (int64_t)1) * (int64_t)16)) + ((((int64_t)vid) * ((condval_42 + (int64_t)17) / (int64_t)2)) * (int64_t)16)) + (((int64_t)pair_idx_1) * (int64_t)2)));
                int64_t condval_43;
                if ((((int64_t)cid) < (int64_t)1)) {
                  condval_43 = (int64_t)1;
                } else {
                  condval_43 = (int64_t)0;
                }
                int32_t sparse_idx1_1 = SparseIndices.GetValue(((((((((int64_t)cid) * (int64_t)256) + (((int64_t)batch_id_1) * (int64_t)16)) + ((((int64_t)cid) < (int64_t)1 ? ((int64_t)cid) : (int64_t)1) * (int64_t)16)) + ((((int64_t)vid) * ((condval_43 + (int64_t)17) / (int64_t)2)) * (int64_t)16)) + (((int64_t)pair_idx_1) * (int64_t)2)) + (int64_t)1));
                int32_t condval_45;
                if ((cid < 1)) {
                  condval_45 = 1;
                } else {
                  condval_45 = 0;
                }
                int32_t condval_44;
                if (((((((cid * 256) + (batch_id_1 * 16)) + ((cid < 1 ? cid : 1) * 16)) + ((vid * ((condval_45 + 17) / 2)) * 16)) + (pair_idx_1 * 2)) < valid_k_size)) {
                  condval_44 = 1;
                } else {
                  condval_44 = 0;
                }
                if (0 < condval_44) {
                  if (0 <= sparse_idx0_1) {
                    if ((batch_id_1 % 2) == 0) {
                      tl::ascend::atomic_add_ub_to_gm<float, 128, 1>(dKeyIndexAccum[(sparse_idx0_1 * 128)], v2_dk_batch_ping[(pair_idx_1 * 256)], 163840, ((sparse_idx0_1 <= 1279) ? 1 : 0), 128);
                    } else {
                      tl::ascend::atomic_add_ub_to_gm<float, 128, 1>(dKeyIndexAccum[(sparse_idx0_1 * 128)], v2_dk_batch_pong[(pair_idx_1 * 256)], 163840, ((sparse_idx0_1 <= 1279) ? 1 : 0), 128);
                    }
                  }
                }
                int32_t condval_47;
                if ((cid < 1)) {
                  condval_47 = 1;
                } else {
                  condval_47 = 0;
                }
                int32_t condval_46;
                if ((((((((cid * 256) + (batch_id_1 * 16)) + ((cid < 1 ? cid : 1) * 16)) + ((vid * ((condval_47 + 17) / 2)) * 16)) + (pair_idx_1 * 2)) + 1) < valid_k_size)) {
                  condval_46 = 1;
                } else {
                  condval_46 = 0;
                }
                if (0 < condval_46) {
                  if (0 <= sparse_idx1_1) {
                    if ((batch_id_1 % 2) == 0) {
                      tl::ascend::atomic_add_ub_to_gm<float, 128, 1>(dKeyIndexAccum[(sparse_idx1_1 * 128)], v2_dk_batch_ping[((pair_idx_1 * 256) + 128)], 163840, ((sparse_idx1_1 <= 1279) ? 1 : 0), 128);
                    } else {
                      tl::ascend::atomic_add_ub_to_gm<float, 128, 1>(dKeyIndexAccum[(sparse_idx1_1 * 128)], v2_dk_batch_pong[((pair_idx_1 * 256) + 128)], 163840, ((sparse_idx1_1 <= 1279) ? 1 : 0), 128);
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
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(6);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(7);
  }
  AscendC::SyncAll<false>();
  if ASCEND_IS_AIV {
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(4);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(7);
    if (cid < 1) {
      for (int32_t dq_group_local = 0; dq_group_local < 8; ++dq_group_local) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(4);
        tl::ascend::copy_gm_to_ub<float, 128, 1>(v2_finalize_dq_fp32[0], dQueryIndexAccum[(((cid * 2048) + (dq_group_local * 256)) + (vid * 128))], 2048, 1, 128, 0.000000e+00f);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(7);
        AscendC::Cast(v2_finalize_dq_half[0], v2_finalize_dq_fp32[0], AscendC::RoundMode::CAST_RINT, 128);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(4);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(5);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(5);
        tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 1>(dQueryIndex[(((cid * 2048) + (dq_group_local * 256)) + (vid * 128))], v2_finalize_dq_half[0], 2048, 1, 128);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(7);
      }
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(4);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(7);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(5);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(6);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(5);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(6);
    for (int32_t finalize_iter = 0; finalize_iter < 10; ++finalize_iter) {
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(((finalize_iter % 2) + 5));
      tl::ascend::copy_gm_to_ub<float, 128, 16>(v2_finalize_dk_fp32[((finalize_iter % 2) * 2048)], dKeyIndexAccum[(((cid * 40960) + (vid * 20480)) + (finalize_iter * 2048))], 128, 16, 128, 0.000000e+00f);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(((finalize_iter % 2) + 5));
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(((finalize_iter % 2) + 5));
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(((finalize_iter % 2) + 5));
      AscendC::Cast(v2_finalize_dk_half[((finalize_iter % 2) * 2048)], v2_finalize_dk_fp32[((finalize_iter % 2) * 2048)], AscendC::RoundMode::CAST_RINT, 2048);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(((finalize_iter % 2) + 5));
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(((finalize_iter % 2) + 5));
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(((finalize_iter % 2) + 5));
      tl::ascend::copy_ub_to_gm<bfloat16_t, 128, 16>(dKeyIndexOut[(((cid * 40960) + (vid * 20480)) + (finalize_iter * 2048))], v2_finalize_dk_half[((finalize_iter % 2) * 2048)], 128, 16, 128);
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

extern "C" void call(uint8_t* SparseIndices_handle, uint8_t* KRealSize_handle, uint8_t* V0ZeroRow_handle, uint8_t* Query_handle, uint8_t* QueryRope_handle, uint8_t* Key_handle, uint8_t* KeyRope_handle, uint8_t* QueryIndex_handle, uint8_t* KeyIndex_handle, uint8_t* Weights_handle, uint8_t* SoftmaxMax_handle, uint8_t* SoftmaxSum_handle, uint8_t* dQueryIndexAccum_handle, uint8_t* dKeyIndexAccum_handle, uint8_t* GatherKey_ws_handle, uint8_t* GatherKeyRope_ws_handle, uint8_t* GatherKeyIndex_ws_handle, uint8_t* bmm1Res_handle, uint8_t* bmm2Res_handle, uint8_t* PReduced_handle, uint8_t* V1Stats_handle, uint8_t* DWeightPartial_handle, uint8_t* dk_matmul_partial_handle, uint8_t* reluGradRes_handle, uint8_t* LossPartial_handle, uint8_t* C2ZeroPad_handle, uint8_t* dQueryIndex_handle, uint8_t* dKeyIndexOut_handle, uint8_t* dWeightOut_handle, uint8_t* LossOut_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<4, nullptr, stream>>>(SparseIndices_handle, KRealSize_handle, V0ZeroRow_handle, Query_handle, QueryRope_handle, Key_handle, KeyRope_handle, QueryIndex_handle, KeyIndex_handle, Weights_handle, SoftmaxMax_handle, SoftmaxSum_handle, dQueryIndexAccum_handle, dKeyIndexAccum_handle, GatherKey_ws_handle, GatherKeyRope_ws_handle, GatherKeyIndex_ws_handle, bmm1Res_handle, bmm2Res_handle, PReduced_handle, V1Stats_handle, DWeightPartial_handle, dk_matmul_partial_handle, reluGradRes_handle, LossPartial_handle, C2ZeroPad_handle, dQueryIndex_handle, dKeyIndexOut_handle, dWeightOut_handle, LossOut_handle, fftsAddr);
}
