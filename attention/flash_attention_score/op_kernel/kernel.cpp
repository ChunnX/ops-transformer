
#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;

class _TileGroupCursor {
public:
    int64_t cursor; 
};
class CubeCtx {
public:
    int64_t b_idx; int64_t n_idx; int64_t ki; int64_t task_id_mod2; int64_t task_id_mod3; 
};
class VecCtx {
public:
    int64_t b_idx; int64_t n_idx; int64_t qi; int64_t ki; int64_t skv_tiles; int64_t task_id_mod2; int64_t task_id_mod3; int64_t q_count_mod3; 
};
#include "OpTiling_tiling.h"
template <uint64_t NeedAttnMask>
__aicore__ inline void flash_attention_score_impl(__gm__ uint8_t* query_0, __gm__ uint8_t* key_0, __gm__ uint8_t* value_0, __gm__ uint8_t* real_shift_0, __gm__ uint8_t* drop_mask_0, __gm__ uint8_t* padding_mask_0, __gm__ uint8_t* atten_mask_0, __gm__ uint8_t* prefix_0, __gm__ uint8_t* actual_seq_qlen_0, __gm__ uint8_t* actual_seq_kvlen_0, __gm__ uint8_t* q_start_idx_0, __gm__ uint8_t* kv_start_idx_0, __gm__ uint8_t* d_scale_q_0, __gm__ uint8_t* d_scale_k_0, __gm__ uint8_t* d_scale_v_0, __gm__ uint8_t* query_rope_0, __gm__ uint8_t* key_rope_0, __gm__ uint8_t* sink_0, __gm__ uint8_t* p_scale_0, __gm__ uint8_t* softmax_max_0, __gm__ uint8_t* softmax_sum_0, __gm__ uint8_t* softmax_out_0, __gm__ uint8_t* attention_out_0, __gm__ uint8_t* workspace_0, __gm__ uint8_t* tiling_0_ptr)
{

    constexpr uint32_t tiling_0_all_bytes = sizeof(OpTiling);
    OpTiling tiling_0;
    #if defined(__DAV_CUBE__)
    copy_data_align64((uint8_t*)&tiling_0, (__gm__ uint8_t *)tiling_0_ptr, tiling_0_all_bytes);
    #endif
    #if defined(__DAV_VEC__)
    __ubuf__ uint8_t *tiling_0_in_ub = (__ubuf__ uint8_t *)get_imm(0);
    constexpr uint32_t tiling_0_len_burst = (tiling_0_all_bytes + 31) / 32;
    copy_gm_to_ubuf_align_v2((__ubuf__ uint8_t *)tiling_0_in_ub, (__gm__ uint8_t *)tiling_0_ptr, 0, 1, tiling_0_len_burst * 32, 0, 0, false, 0, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    copy_data_align64((uint8_t*)&tiling_0, (__ubuf__ uint8_t *)tiling_0_in_ub, tiling_0_all_bytes);
    #endif

    using _tg_qk_vec_db_tiles_0_Type = Tile<TileType::Vec, float, 128, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null>;
    _tg_qk_vec_db_tiles_0_Type _tg_qk_vec_db_tiles_0_0(128, 64); TASSIGN(_tg_qk_vec_db_tiles_0_0, 0x0);
    _tg_qk_vec_db_tiles_0_Type _tg_qk_vec_db_tiles_1_0(128, 64); TASSIGN(_tg_qk_vec_db_tiles_1_0, 0x8000);
    using _tg_pv_vec_db_tiles_0_Type = Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null>;
    _tg_pv_vec_db_tiles_0_Type _tg_pv_vec_db_tiles_0_0(64, 128); TASSIGN(_tg_pv_vec_db_tiles_0_0, 0x18200);
    _tg_pv_vec_db_tiles_0_Type _tg_pv_vec_db_tiles_1_0(64, 128); TASSIGN(_tg_pv_vec_db_tiles_1_0, 0x20200);
    using _tg_p_mat_db_tiles_0_Type = Tile<TileType::Mat, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null>;
    _tg_p_mat_db_tiles_0_Type _tg_p_mat_db_tiles_0_0(128, 128); TASSIGN(_tg_p_mat_db_tiles_0_0, 0x20000);
    _tg_p_mat_db_tiles_0_Type _tg_p_mat_db_tiles_1_0(128, 128); TASSIGN(_tg_p_mat_db_tiles_1_0, 0x28000);
    _tg_p_mat_db_tiles_0_Type _tg_p_mat_db_tiles_2_0(128, 128); TASSIGN(_tg_p_mat_db_tiles_2_0, 0x30000);

    #if defined(__DAV_CUBE__)
    _tg_p_mat_db_tiles_0_Type _tg_q_l1_db_tiles_0_0(128, 128); TASSIGN(_tg_q_l1_db_tiles_0_0, 0x0);
    _tg_p_mat_db_tiles_0_Type _tg_q_l1_db_tiles_1_0(128, 128); TASSIGN(_tg_q_l1_db_tiles_1_0, 0x8000);
    using _tg_k_l1_db_tiles_0_Type = Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null>;
    _tg_k_l1_db_tiles_0_Type _tg_k_l1_db_tiles_0_0(128, 128); TASSIGN(_tg_k_l1_db_tiles_0_0, 0x10000);
    _tg_k_l1_db_tiles_0_Type _tg_k_l1_db_tiles_1_0(128, 128); TASSIGN(_tg_k_l1_db_tiles_1_0, 0x18000);
    _tg_k_l1_db_tiles_0_Type _tg_v_l1_db_tiles_0_0(128, 128); TASSIGN(_tg_v_l1_db_tiles_0_0, 0x38000);
    _tg_k_l1_db_tiles_0_Type _tg_v_l1_db_tiles_1_0(128, 128); TASSIGN(_tg_v_l1_db_tiles_1_0, 0x40000);
    using _tg_left_db_tiles_0_Type = Tile<TileType::Left, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null>;
    _tg_left_db_tiles_0_Type _tg_left_db_tiles_0_0(128, 128); TASSIGN(_tg_left_db_tiles_0_0, 0x0);
    _tg_left_db_tiles_0_Type _tg_left_db_tiles_1_0(128, 128); TASSIGN(_tg_left_db_tiles_1_0, 0x8000);
    using _tg_right_db_tiles_0_Type = Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null>;
    _tg_right_db_tiles_0_Type _tg_right_db_tiles_0_0(128, 128); TASSIGN(_tg_right_db_tiles_0_0, 0x0);
    _tg_right_db_tiles_0_Type _tg_right_db_tiles_1_0(128, 128); TASSIGN(_tg_right_db_tiles_1_0, 0x8000);
    using _tg_acc_db_tiles_0_Type = Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null>;
    _tg_acc_db_tiles_0_Type _tg_acc_db_tiles_0_0(128, 128); TASSIGN(_tg_acc_db_tiles_0_0, 0x0);
    _tg_acc_db_tiles_0_Type _tg_acc_db_tiles_1_0(128, 128); TASSIGN(_tg_acc_db_tiles_1_0, 0x10000);
    _tg_acc_db_tiles_0_Type _tg_acc_db_tiles_2_0(128, 128); TASSIGN(_tg_acc_db_tiles_2_0, 0x20000);
    _tg_acc_db_tiles_0_Type _tg_acc_db_tiles_3_0(128, 128); TASSIGN(_tg_acc_db_tiles_3_0, 0x30000);
    #endif

    #if defined(__DAV_VEC__)
    using _tg_p_f16_db_tiles_0_Type = Tile<TileType::Vec, half, 65, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null>;
    _tg_p_f16_db_tiles_0_Type _tg_p_f16_db_tiles_0_0(65, 128); TASSIGN(_tg_p_f16_db_tiles_0_0, 0x10000);
    _tg_p_f16_db_tiles_0_Type _tg_p_f16_db_tiles_1_0(65, 128); TASSIGN(_tg_p_f16_db_tiles_1_0, 0x14100);
    using _tg_p_f16_main_db_tiles_0_Type = Tile<TileType::Vec, half, 65, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null>;
    _tg_p_f16_main_db_tiles_0_Type _tg_p_f16_main_db_tiles_0_0; TASSIGN(_tg_p_f16_main_db_tiles_0_0, 0x10000);
    _tg_p_f16_main_db_tiles_0_Type _tg_p_f16_main_db_tiles_1_0; TASSIGN(_tg_p_f16_main_db_tiles_1_0, 0x14100);
    _tg_p_f16_main_db_tiles_0_Type _tg_p_f16_back_db_tiles_0_0; TASSIGN(_tg_p_f16_back_db_tiles_0_0, 0x12080);
    _tg_p_f16_main_db_tiles_0_Type _tg_p_f16_back_db_tiles_1_0; TASSIGN(_tg_p_f16_back_db_tiles_1_0, 0x16180);
    using gmax_rm_0_Type = Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null>;
    gmax_rm_0_Type gmax_rm_0_0(1, 64); TASSIGN(gmax_rm_0_0, 0x34200);
    gmax_rm_0_Type gmax_rm_1_0(1, 64); TASSIGN(gmax_rm_1_0, 0x34300);
    gmax_rm_0_Type gmax_rm_2_0(1, 64); TASSIGN(gmax_rm_2_0, 0x34400);
    gmax_rm_0_Type gsum_rm_0_0(1, 64); TASSIGN(gsum_rm_0_0, 0x34500);
    gmax_rm_0_Type gsum_rm_1_0(1, 64); TASSIGN(gsum_rm_1_0, 0x34600);
    gmax_rm_0_Type gsum_rm_2_0(1, 64); TASSIGN(gsum_rm_2_0, 0x34700);
    gmax_rm_0_Type ec_rm_0_0(1, 64); TASSIGN(ec_rm_0_0, 0x34800);
    gmax_rm_0_Type ec_rm_1_0(1, 64); TASSIGN(ec_rm_1_0, 0x34900);
    gmax_rm_0_Type ec_rm_2_0(1, 64); TASSIGN(ec_rm_2_0, 0x34a00);
    using _tg_o_f16_g_tiles_0_Type = Tile<TileType::Vec, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null>;
    _tg_o_f16_g_tiles_0_Type _tg_o_f16_g_tiles_0_0(64, 128); TASSIGN(_tg_o_f16_g_tiles_0_0, 0x30200);
    _tg_pv_vec_db_tiles_0_Type _tg_running_o_g_tiles_0_0(64, 128); TASSIGN(_tg_running_o_g_tiles_0_0, 0x28200);
    using _tg_mask_db_tiles_0_Type = Tile<TileType::Vec, uint8_t, 128, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null>;
    _tg_mask_db_tiles_0_Type _tg_mask_db_tiles_0_0(128, 64); TASSIGN(_tg_mask_db_tiles_0_0, 0x34b00);
    _tg_mask_db_tiles_0_Type _tg_mask_db_tiles_1_0(128, 64); TASSIGN(_tg_mask_db_tiles_1_0, 0x36b00);
    #endif

    const int64_t QK_READY_BARKWARD_IDS_0[] = {2, 3};
    const int64_t PV_READY_BARKWARD_IDS_0[] = {9, 10};
    using tensor_q_0ShapeDim5 = pto::Shape<1, 1, 1, -1, -1>;
    using tensor_q_0StrideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
    using tensor_q_0Type = GlobalTensor<half, tensor_q_0ShapeDim5, tensor_q_0StrideDim5, Layout::DN>;
    tensor_q_0Type tensor_q_0((__gm__ half*)(query_0), tensor_q_0ShapeDim5(), tensor_q_0StrideDim5(1, 1, 1, 1, tiling_0.n*tiling_0.d));
    using tensor_k_0ShapeDim5 = pto::Shape<1, 1, 1, -1, -1>;
    using tensor_k_0StrideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
    using tensor_k_0Type = GlobalTensor<half, tensor_k_0ShapeDim5, tensor_k_0StrideDim5>;
    tensor_k_0Type tensor_k_0((__gm__ half*)(key_0), tensor_k_0ShapeDim5(), tensor_k_0StrideDim5(1, 1, 1, tiling_0.n*tiling_0.d, 1));
    using tensor_v_0ShapeDim5 = pto::Shape<1, 1, 1, -1, -1>;
    using tensor_v_0StrideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
    using tensor_v_0Type = GlobalTensor<half, tensor_v_0ShapeDim5, tensor_v_0StrideDim5>;
    tensor_v_0Type tensor_v_0((__gm__ half*)(value_0), tensor_v_0ShapeDim5(), tensor_v_0StrideDim5(1, 1, 1, tiling_0.n*tiling_0.d, 1));
    using tensor_o_0ShapeDim5 = pto::Shape<1, 1, 1, -1, -1>;
    using tensor_o_0StrideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
    using tensor_o_0Type = GlobalTensor<half, tensor_o_0ShapeDim5, tensor_o_0StrideDim5>;
    tensor_o_0Type tensor_o_0((__gm__ half*)(attention_out_0), tensor_o_0ShapeDim5(), tensor_o_0StrideDim5(1, 1, 1, tiling_0.n*tiling_0.d, 1));
    using tensor_attn_mask_0ShapeDim5 = pto::Shape<1, 1, 1, -1, -1>;
    using tensor_attn_mask_0StrideDim5 = pto::Stride<-1, -1, -1, -1, -1>;
    using tensor_attn_mask_0Type = GlobalTensor<uint8_t, tensor_attn_mask_0ShapeDim5, tensor_attn_mask_0StrideDim5>;
    tensor_attn_mask_0Type tensor_attn_mask_0(atten_mask_0, tensor_attn_mask_0ShapeDim5(), tensor_attn_mask_0StrideDim5(1, 1, 1, 2048, 1));
    auto n_dim_0 = tiling_0.n;
    auto sq_dim_0 = tiling_0.sq;
    auto skv_dim_0 = tiling_0.skv;
    auto sq_tiles_0 = ((sq_dim_0 + 127) / 128);
    auto skv_tiles_0 = ((skv_dim_0 + 127) / 128);
    auto core_id_0 = get_block_idx();
    auto num_cores_0 = get_block_num();
    auto total_work_0 = (tiling_0.b * tiling_0.n);
    auto work_per_core_0 = (((total_work_0 + num_cores_0) - 1) / num_cores_0);
    auto work_start_0 = (core_id_0 * work_per_core_0);
    auto work_end_0 = (work_start_0 + work_per_core_0);
    int64_t work_end_2;
    if ((work_end_0 > total_work_0)) {
        work_end_2 = total_work_0;
    } else {
        work_end_2 = work_end_0;
    }
    Tile<TileType::Vec, float, 128, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null> _tg_qk_vec_db_tiles_0[] = {_tg_qk_vec_db_tiles_0_0, _tg_qk_vec_db_tiles_1_0};
    const int64_t _tg_qk_vec_db_mutex_ids_0[] = {14, 15};
    _TileGroupCursor _tg_qk_vec_db_cursor_0 = {.cursor=1};
    Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null> _tg_pv_vec_db_tiles_0[] = {_tg_pv_vec_db_tiles_0_0, _tg_pv_vec_db_tiles_1_0};
    const int64_t _tg_pv_vec_db_mutex_ids_0[] = {16, 17};
    _TileGroupCursor _tg_pv_vec_db_cursor_0 = {.cursor=1};
    Tile<TileType::Mat, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null> _tg_p_mat_db_tiles_0[] = {_tg_p_mat_db_tiles_0_0, _tg_p_mat_db_tiles_1_0, _tg_p_mat_db_tiles_2_0};
    const int64_t _tg_p_mat_db_mutex_ids_0[] = {18, 19, 20};
    _TileGroupCursor _tg_p_mat_db_cursor_0 = {.cursor=2};
    #if defined(__DAV_CUBE__)
    Tile<TileType::Mat, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null> _tg_q_l1_db_tiles_0[] = {_tg_q_l1_db_tiles_0_0, _tg_q_l1_db_tiles_1_0};
    const int64_t _tg_q_l1_db_mutex_ids_0[] = {0, 1};
    _TileGroupCursor _tg_q_l1_db_cursor_0 = {.cursor=1};
    Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null> _tg_k_l1_db_tiles_0[] = {_tg_k_l1_db_tiles_0_0, _tg_k_l1_db_tiles_1_0};
    const int64_t _tg_k_l1_db_mutex_ids_0[] = {2, 3};
    _TileGroupCursor _tg_k_l1_db_cursor_0 = {.cursor=1};
    Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null> _tg_v_l1_db_tiles_0[] = {_tg_v_l1_db_tiles_0_0, _tg_v_l1_db_tiles_1_0};
    const int64_t _tg_v_l1_db_mutex_ids_0[] = {4, 5};
    _TileGroupCursor _tg_v_l1_db_cursor_0 = {.cursor=1};
    Tile<TileType::Left, half, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 512, PadValue::Null> _tg_left_db_tiles_0[] = {_tg_left_db_tiles_0_0, _tg_left_db_tiles_1_0};
    const int64_t _tg_left_db_mutex_ids_0[] = {6, 7};
    _TileGroupCursor _tg_left_db_cursor_0 = {.cursor=1};
    Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, -1, -1, SLayout::ColMajor, 512, PadValue::Null> _tg_right_db_tiles_0[] = {_tg_right_db_tiles_0_0, _tg_right_db_tiles_1_0};
    const int64_t _tg_right_db_mutex_ids_0[] = {8, 9};
    _TileGroupCursor _tg_right_db_cursor_0 = {.cursor=1};
    Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024, PadValue::Null> _tg_acc_db_tiles_0[] = {_tg_acc_db_tiles_0_0, _tg_acc_db_tiles_1_0, _tg_acc_db_tiles_2_0, _tg_acc_db_tiles_3_0};
    const int64_t _tg_acc_db_mutex_ids_0[] = {10, 11, 12, 13};
    _TileGroupCursor _tg_acc_db_cursor_0 = {.cursor=3};
    auto task_id_0 = 0;
    CubeCtx ctx_arr_0_0 = {.b_idx=0, .n_idx=0, .ki=0, .task_id_mod2=0, .task_id_mod3=0};
    CubeCtx ctx_arr_1_0 = {.b_idx=0, .n_idx=0, .ki=0, .task_id_mod2=0, .task_id_mod3=0};
    CubeCtx ctx_arr_2_0 = {.b_idx=0, .n_idx=0, .ki=0, .task_id_mod2=0, .task_id_mod3=0};
    CubeCtx ctx_arr_3_0 = {.b_idx=0, .n_idx=0, .ki=0, .task_id_mod2=0, .task_id_mod3=0};
    CubeCtx ctx_arr_0[] = {ctx_arr_0_0, ctx_arr_1_0, ctx_arr_2_0, ctx_arr_3_0};
    for (uint64_t work_id_0 = work_start_0; work_id_0 < work_end_2; work_id_0 += 1) {
        auto b_idx_0 = (work_id_0 / n_dim_0);
        auto n_idx_0 = (work_id_0 % n_dim_0);
        for (uint64_t qi_0 = 0; qi_0 < sq_tiles_0; qi_0 += 1) {
            _tg_q_l1_db_cursor_0.cursor = (_tg_q_l1_db_cursor_0.cursor + 1);
            auto _bufidx_0_0 = ((_tg_q_l1_db_cursor_0.cursor + 1) % 2);
            auto causal_skv_0 = ((NeedAttnMask * ((uint64_t)(qi_0 + 1))) + ((uint64_t)((1 - ((int64_t)NeedAttnMask)) * skv_tiles_0)));
            uint64_t ki_end_3;
            if (((work_id_0 + 1) >= work_end_2)) {
                uint64_t ki_end_2;
                if (((qi_0 + 1) >= sq_tiles_0)) {
                    auto ki_end_1 = (causal_skv_0 + ((uint64_t)2));
                    ki_end_2 = ki_end_1;
                } else {
                    ki_end_2 = causal_skv_0;
                }
                ki_end_3 = ki_end_2;
            } else {
                ki_end_3 = causal_skv_0;
            }
            for (uint64_t ki_0 = 0; ki_0 < ki_end_3; ki_0 += 1) {
                if ((ki_0 < ((int64_t)causal_skv_0))) {
                    auto& ctx_curr_0 = ctx_arr_0[(task_id_0 % 4)];
                    ctx_curr_0.b_idx = b_idx_0;
                    ctx_curr_0.n_idx = n_idx_0;
                    ctx_curr_0.ki = ki_0;
                    ctx_curr_0.task_id_mod2 = (task_id_0 % 2);
                    ctx_curr_0.task_id_mod3 = (task_id_0 % 3);
                    const int64_t _inl0_QK_READY_BARKWARD_IDS_0[] = {2, 3};
                    const int64_t _inl0_QK_READY_FORWARD_IDS_0[] = {0, 1};
                    if ((ki_0 == 0)) {
                        get_buf(PIPE_MTE2, _tg_q_l1_db_mutex_ids_0[_bufidx_0_0], 0);
                        tensor_q_0.SetShape<pto::GlobalTensorDim::DIM_3, pto::GlobalTensorDim::DIM_4>(static_cast<int64_t>(128), static_cast<int64_t>(128));
                        TASSIGN(tensor_q_0, (__gm__ half*)(query_0) + (b_idx_0 * tiling_0.sq * tiling_0.n * tiling_0.d + (qi_0 * 128) * tiling_0.n * tiling_0.d + n_idx_0 * tiling_0.d + 0));
                        TLOAD(_tg_q_l1_db_tiles_0[_bufidx_0_0], tensor_q_0);
                        rls_buf(PIPE_MTE2, _tg_q_l1_db_mutex_ids_0[_bufidx_0_0], 0);
                    }
                    _tg_k_l1_db_cursor_0.cursor = (_tg_k_l1_db_cursor_0.cursor + 1);
                    auto _inl0__bufidx_0_0 = ((_tg_k_l1_db_cursor_0.cursor + 1) % 2);
                    get_buf(PIPE_MTE2, _tg_k_l1_db_mutex_ids_0[_inl0__bufidx_0_0], 0);
                    tensor_k_0.SetShape<pto::GlobalTensorDim::DIM_3, pto::GlobalTensorDim::DIM_4>(static_cast<int64_t>(128), static_cast<int64_t>(128));
                    TASSIGN(tensor_k_0, (__gm__ half*)(key_0) + (b_idx_0 * tiling_0.skv * tiling_0.n * tiling_0.d + (ki_0 * 128) * tiling_0.n * tiling_0.d + n_idx_0 * tiling_0.d + 0));
                    TLOAD(_tg_k_l1_db_tiles_0[_inl0__bufidx_0_0], tensor_k_0);
                    rls_buf(PIPE_MTE2, _tg_k_l1_db_mutex_ids_0[_inl0__bufidx_0_0], 0);
                    _tg_left_db_cursor_0.cursor = (_tg_left_db_cursor_0.cursor + 1);
                    auto _inl0__bufidx_1_0 = ((_tg_left_db_cursor_0.cursor + 1) % 2);
                    _tg_right_db_cursor_0.cursor = (_tg_right_db_cursor_0.cursor + 1);
                    auto _inl0__bufidx_2_0 = ((_tg_right_db_cursor_0.cursor + 1) % 2);
                    _tg_acc_db_cursor_0.cursor = (_tg_acc_db_cursor_0.cursor + 1);
                    auto _inl0__bufidx_3_0 = ((_tg_acc_db_cursor_0.cursor + 1) % 4);
                    get_buf(PIPE_MTE1, _tg_left_db_mutex_ids_0[_inl0__bufidx_1_0], 0);
                    get_buf(PIPE_MTE1, _tg_k_l1_db_mutex_ids_0[_inl0__bufidx_0_0], 0);
                    TMOV<std::remove_reference_t<decltype(_tg_left_db_tiles_0[_inl0__bufidx_1_0])>, std::remove_reference_t<decltype(_tg_k_l1_db_tiles_0[_inl0__bufidx_0_0])>>(_tg_left_db_tiles_0[_inl0__bufidx_1_0], _tg_k_l1_db_tiles_0[_inl0__bufidx_0_0]);
                    rls_buf(PIPE_MTE1, _tg_left_db_mutex_ids_0[_inl0__bufidx_1_0], 0);
                    rls_buf(PIPE_MTE1, _tg_k_l1_db_mutex_ids_0[_inl0__bufidx_0_0], 0);
                    get_buf(PIPE_MTE1, _tg_right_db_mutex_ids_0[_inl0__bufidx_2_0], 0);
                    get_buf(PIPE_MTE1, _tg_q_l1_db_mutex_ids_0[_inl0__bufidx_0_0], 0);
                    TMOV<std::remove_reference_t<decltype(_tg_right_db_tiles_0[_inl0__bufidx_2_0])>, std::remove_reference_t<decltype(_tg_q_l1_db_tiles_0[_bufidx_0_0])>>(_tg_right_db_tiles_0[_inl0__bufidx_2_0], _tg_q_l1_db_tiles_0[_bufidx_0_0]);
                    rls_buf(PIPE_MTE1, _tg_right_db_mutex_ids_0[_inl0__bufidx_2_0], 0);
                    rls_buf(PIPE_MTE1, _tg_q_l1_db_mutex_ids_0[_inl0__bufidx_0_0], 0);
                    get_buf(PIPE_M, _tg_acc_db_mutex_ids_0[_inl0__bufidx_3_0], 0);
                    get_buf(PIPE_M, _tg_left_db_mutex_ids_0[_inl0__bufidx_1_0], 0);
                    get_buf(PIPE_M, _tg_right_db_mutex_ids_0[_inl0__bufidx_2_0], 0);
                    TMATMUL(_tg_acc_db_tiles_0[_inl0__bufidx_3_0], _tg_left_db_tiles_0[_inl0__bufidx_1_0], _tg_right_db_tiles_0[_inl0__bufidx_2_0]);
                    rls_buf(PIPE_M, _tg_acc_db_mutex_ids_0[_inl0__bufidx_3_0], 0);
                    rls_buf(PIPE_M, _tg_left_db_mutex_ids_0[_inl0__bufidx_1_0], 0);
                    rls_buf(PIPE_M, _tg_right_db_mutex_ids_0[_inl0__bufidx_2_0], 0);
                    _tg_qk_vec_db_cursor_0.cursor = (_tg_qk_vec_db_cursor_0.cursor + 1);
                    auto _inl0__bufidx_4_0 = ((_tg_qk_vec_db_cursor_0.cursor + 1) % 2);
                    const int64_t _dyn_arr_0[] = {2, 3};
                    wait_intra_block(PIPE_FIX, _dyn_arr_0[(task_id_0 % 2)]);
                    const int64_t _dyn_arr_1[] = {2, 3};
                    wait_intra_block(PIPE_FIX, _dyn_arr_1[(task_id_0 % 2)] + 16);
                    get_buf(PIPE_FIX, _tg_qk_vec_db_mutex_ids_0[_inl0__bufidx_4_0], 0);
                    get_buf(PIPE_FIX, _tg_acc_db_mutex_ids_0[_inl0__bufidx_3_0], 0);
                    TMOV<std::remove_reference_t<decltype(_tg_qk_vec_db_tiles_0[_inl0__bufidx_4_0])>, std::remove_reference_t<decltype(_tg_acc_db_tiles_0[_inl0__bufidx_3_0])>, AccToVecMode::DualModeSplitN>(_tg_qk_vec_db_tiles_0[_inl0__bufidx_4_0], _tg_acc_db_tiles_0[_inl0__bufidx_3_0]);
                    rls_buf(PIPE_FIX, _tg_qk_vec_db_mutex_ids_0[_inl0__bufidx_4_0], 0);
                    rls_buf(PIPE_FIX, _tg_acc_db_mutex_ids_0[_inl0__bufidx_3_0], 0);
                    const int64_t _dyn_arr_2[] = {0, 1};
                    set_intra_block(PIPE_FIX, _dyn_arr_2[(task_id_0 % 2)]);
                    set_intra_block(PIPE_FIX, _dyn_arr_2[(task_id_0 % 2)] + 16);
                }
                if ((task_id_0 > 1)) {
                    auto& prev_0 = ctx_arr_0[((task_id_0 + 2) % 4)];
                    const int64_t _inl1_P_READY_FORWARD_IDS_0[] = {4, 5, 6};
                    const int64_t _inl1_PV_READY_BARKWARD_IDS_0[] = {9, 10};
                    const int64_t _inl1_PV_READY_FORWARD_IDS_0[] = {7, 8};
                    const int64_t _dyn_arr_3[] = {4, 5, 6};
                    wait_intra_block(PIPE_MTE1, _dyn_arr_3[prev_0.task_id_mod3]);
                    const int64_t _dyn_arr_4[] = {4, 5, 6};
                    wait_intra_block(PIPE_MTE1, _dyn_arr_4[prev_0.task_id_mod3] + 16);
                    _tg_v_l1_db_cursor_0.cursor = (_tg_v_l1_db_cursor_0.cursor + 1);
                    auto _inl1__bufidx_0_0 = ((_tg_v_l1_db_cursor_0.cursor + 1) % 2);
                    _tg_p_mat_db_cursor_0.cursor = (_tg_p_mat_db_cursor_0.cursor + 1);
                    auto _inl1__bufidx_1_0 = ((_tg_p_mat_db_cursor_0.cursor + 1) % 3);
                    _tg_left_db_cursor_0.cursor = (_tg_left_db_cursor_0.cursor + 1);
                    auto _inl1__bufidx_2_0 = ((_tg_left_db_cursor_0.cursor + 1) % 2);
                    _tg_right_db_cursor_0.cursor = (_tg_right_db_cursor_0.cursor + 1);
                    auto _inl1__bufidx_3_0 = ((_tg_right_db_cursor_0.cursor + 1) % 2);
                    _tg_acc_db_cursor_0.cursor = (_tg_acc_db_cursor_0.cursor + 1);
                    auto _inl1__bufidx_4_0 = ((_tg_acc_db_cursor_0.cursor + 1) % 4);
                    get_buf(PIPE_MTE2, _tg_v_l1_db_mutex_ids_0[_inl1__bufidx_0_0], 0);
                    tensor_v_0.SetShape<pto::GlobalTensorDim::DIM_3, pto::GlobalTensorDim::DIM_4>(static_cast<int64_t>(128), static_cast<int64_t>(128));
                    TASSIGN(tensor_v_0, (__gm__ half*)(value_0) + (prev_0.b_idx * tiling_0.skv * tiling_0.n * tiling_0.d + (prev_0.ki * 128) * tiling_0.n * tiling_0.d + prev_0.n_idx * tiling_0.d + 0));
                    TLOAD(_tg_v_l1_db_tiles_0[_inl1__bufidx_0_0], tensor_v_0);
                    rls_buf(PIPE_MTE2, _tg_v_l1_db_mutex_ids_0[_inl1__bufidx_0_0], 0);
                    get_buf(PIPE_MTE1, _tg_left_db_mutex_ids_0[_inl1__bufidx_2_0], 0);
                    get_buf(PIPE_MTE1, _tg_p_mat_db_mutex_ids_0[_inl1__bufidx_1_0], 0);
                    TMOV<std::remove_reference_t<decltype(_tg_left_db_tiles_0[_inl1__bufidx_2_0])>, std::remove_reference_t<decltype(_tg_p_mat_db_tiles_0[_inl1__bufidx_1_0])>>(_tg_left_db_tiles_0[_inl1__bufidx_2_0], _tg_p_mat_db_tiles_0[_inl1__bufidx_1_0]);
                    rls_buf(PIPE_MTE1, _tg_left_db_mutex_ids_0[_inl1__bufidx_2_0], 0);
                    rls_buf(PIPE_MTE1, _tg_p_mat_db_mutex_ids_0[_inl1__bufidx_1_0], 0);
                    get_buf(PIPE_MTE1, _tg_right_db_mutex_ids_0[_inl1__bufidx_3_0], 0);
                    get_buf(PIPE_MTE1, _tg_v_l1_db_mutex_ids_0[_inl1__bufidx_0_0], 0);
                    TMOV<std::remove_reference_t<decltype(_tg_right_db_tiles_0[_inl1__bufidx_3_0])>, std::remove_reference_t<decltype(_tg_v_l1_db_tiles_0[_inl1__bufidx_0_0])>>(_tg_right_db_tiles_0[_inl1__bufidx_3_0], _tg_v_l1_db_tiles_0[_inl1__bufidx_0_0]);
                    rls_buf(PIPE_MTE1, _tg_right_db_mutex_ids_0[_inl1__bufidx_3_0], 0);
                    rls_buf(PIPE_MTE1, _tg_v_l1_db_mutex_ids_0[_inl1__bufidx_0_0], 0);
                    get_buf(PIPE_M, _tg_acc_db_mutex_ids_0[_inl1__bufidx_4_0], 0);
                    get_buf(PIPE_M, _tg_left_db_mutex_ids_0[_inl1__bufidx_2_0], 0);
                    get_buf(PIPE_M, _tg_right_db_mutex_ids_0[_inl1__bufidx_3_0], 0);
                    TMATMUL(_tg_acc_db_tiles_0[_inl1__bufidx_4_0], _tg_left_db_tiles_0[_inl1__bufidx_2_0], _tg_right_db_tiles_0[_inl1__bufidx_3_0]);
                    rls_buf(PIPE_M, _tg_acc_db_mutex_ids_0[_inl1__bufidx_4_0], 0);
                    rls_buf(PIPE_M, _tg_left_db_mutex_ids_0[_inl1__bufidx_2_0], 0);
                    rls_buf(PIPE_M, _tg_right_db_mutex_ids_0[_inl1__bufidx_3_0], 0);
                    _tg_pv_vec_db_cursor_0.cursor = (_tg_pv_vec_db_cursor_0.cursor + 1);
                    auto _inl1__bufidx_5_0 = ((_tg_pv_vec_db_cursor_0.cursor + 1) % 2);
                    const int64_t _dyn_arr_5[] = {9, 10};
                    wait_intra_block(PIPE_FIX, _dyn_arr_5[prev_0.task_id_mod2]);
                    const int64_t _dyn_arr_6[] = {9, 10};
                    wait_intra_block(PIPE_FIX, _dyn_arr_6[prev_0.task_id_mod2] + 16);
                    get_buf(PIPE_FIX, _tg_pv_vec_db_mutex_ids_0[_inl1__bufidx_5_0], 0);
                    get_buf(PIPE_FIX, _tg_acc_db_mutex_ids_0[_inl1__bufidx_4_0], 0);
                    TMOV<std::remove_reference_t<decltype(_tg_pv_vec_db_tiles_0[_inl1__bufidx_5_0])>, std::remove_reference_t<decltype(_tg_acc_db_tiles_0[_inl1__bufidx_4_0])>, AccToVecMode::DualModeSplitM>(_tg_pv_vec_db_tiles_0[_inl1__bufidx_5_0], _tg_acc_db_tiles_0[_inl1__bufidx_4_0]);
                    rls_buf(PIPE_FIX, _tg_pv_vec_db_mutex_ids_0[_inl1__bufidx_5_0], 0);
                    rls_buf(PIPE_FIX, _tg_acc_db_mutex_ids_0[_inl1__bufidx_4_0], 0);
                    const int64_t _dyn_arr_7[] = {7, 8};
                    set_intra_block(PIPE_FIX, _dyn_arr_7[prev_0.task_id_mod2]);
                    set_intra_block(PIPE_FIX, _dyn_arr_7[prev_0.task_id_mod2] + 16);
                }
                auto task_id_7 = (task_id_0 + 1);
                task_id_0 = task_id_7;
            }
        }
    }
    #endif
    #if defined(__DAV_VEC__)
    Tile<TileType::Vec, half, 65, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null> _tg_p_f16_db_tiles_0[] = {_tg_p_f16_db_tiles_0_0, _tg_p_f16_db_tiles_1_0};
    const int64_t _tg_p_f16_db_mutex_ids_0[] = {0, 1};
    _TileGroupCursor _tg_p_f16_db_cursor_0 = {.cursor=1};
    Tile<TileType::Vec, half, 65, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> _tg_p_f16_main_db_tiles_0[] = {_tg_p_f16_main_db_tiles_0_0, _tg_p_f16_main_db_tiles_1_0};
    const int64_t _tg_p_f16_main_db_mutex_ids_0[] = {0, 1};
    _TileGroupCursor _tg_p_f16_main_db_cursor_0 = {.cursor=1};
    Tile<TileType::Vec, half, 65, 64, BLayout::ColMajor, 64, 64, SLayout::RowMajor, 512, PadValue::Null> _tg_p_f16_back_db_tiles_0[] = {_tg_p_f16_back_db_tiles_0_0, _tg_p_f16_back_db_tiles_1_0};
    const int64_t _tg_p_f16_back_db_mutex_ids_0[] = {0, 1};
    _TileGroupCursor _tg_p_f16_back_db_cursor_0 = {.cursor=1};
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null> global_max_rm_buf_0[] = {gmax_rm_0_0, gmax_rm_1_0, gmax_rm_2_0};
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null> global_sum_rm_buf_0[] = {gsum_rm_0_0, gsum_rm_1_0, gsum_rm_2_0};
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null> exp_corr_rm_fifo_0[] = {ec_rm_0_0, ec_rm_1_0, ec_rm_2_0};
    Tile<TileType::Vec, half, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null> _tg_o_f16_g_tiles_0[] = {_tg_o_f16_g_tiles_0_0};
    const int64_t _tg_o_f16_g_mutex_ids_0[] = {11};
    Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null> _tg_running_o_g_tiles_0[] = {_tg_running_o_g_tiles_0_0};
    const int64_t _tg_running_o_g_mutex_ids_0[] = {12};
    Tile<TileType::Vec, uint8_t, 128, 64, BLayout::RowMajor, -1, -1, SLayout::NoneBox, 512, PadValue::Null> _tg_mask_db_tiles_0[] = {_tg_mask_db_tiles_0_0, _tg_mask_db_tiles_1_0};
    const int64_t _tg_mask_db_mutex_ids_0[] = {8, 9};
    _TileGroupCursor _tg_mask_db_cursor_0 = {.cursor=1};
    set_intra_block(PIPE_V, 2);
    set_intra_block(PIPE_V, 3);
    set_intra_block(PIPE_V, 9);
    set_intra_block(PIPE_V, 10);
    auto _expr_tmp_0_0 = get_subblockid();
    auto sub_id_0 = (int32_t)(_expr_tmp_0_0);
    auto task_id_8 = 0;
    auto q_count_0 = 0;
    VecCtx ctx_arr_0_1 = {.b_idx=0, .n_idx=0, .qi=0, .ki=0, .skv_tiles=0, .task_id_mod2=0, .task_id_mod3=0, .q_count_mod3=0};
    VecCtx ctx_arr_1_1 = {.b_idx=0, .n_idx=0, .qi=0, .ki=0, .skv_tiles=0, .task_id_mod2=0, .task_id_mod3=0, .q_count_mod3=0};
    VecCtx ctx_arr_2_1 = {.b_idx=0, .n_idx=0, .qi=0, .ki=0, .skv_tiles=0, .task_id_mod2=0, .task_id_mod3=0, .q_count_mod3=0};
    VecCtx ctx_arr_3_1 = {.b_idx=0, .n_idx=0, .qi=0, .ki=0, .skv_tiles=0, .task_id_mod2=0, .task_id_mod3=0, .q_count_mod3=0};
    VecCtx ctx_arr_1[] = {ctx_arr_0_1, ctx_arr_1_1, ctx_arr_2_1, ctx_arr_3_1};
    for (uint64_t work_id_1 = work_start_0; work_id_1 < work_end_2; work_id_1 += 1) {
        auto b_idx_1 = (work_id_1 / n_dim_0);
        auto n_idx_1 = (work_id_1 % n_dim_0);
        for (uint64_t qi_1 = 0; qi_1 < sq_tiles_0; qi_1 += 1) {
            auto causal_skv_1 = ((NeedAttnMask * ((uint64_t)(qi_1 + 1))) + ((uint64_t)((1 - ((int64_t)NeedAttnMask)) * skv_tiles_0)));
            uint64_t ki_end_7;
            if (((work_id_1 + 1) >= work_end_2)) {
                uint64_t ki_end_6;
                if (((qi_1 + 1) >= sq_tiles_0)) {
                    auto ki_end_5 = (causal_skv_1 + ((uint64_t)3));
                    ki_end_6 = ki_end_5;
                } else {
                    ki_end_6 = causal_skv_1;
                }
                ki_end_7 = ki_end_6;
            } else {
                ki_end_7 = causal_skv_1;
            }
            for (uint64_t ki_1 = 0; ki_1 < ki_end_7; ki_1 += 1) {
                if ((ki_1 < ((int64_t)causal_skv_1))) {
                    auto& ctx_curr_1 = ctx_arr_1[(task_id_8 % 4)];
                    ctx_curr_1.b_idx = b_idx_1;
                    ctx_curr_1.n_idx = n_idx_1;
                    ctx_curr_1.qi = qi_1;
                    ctx_curr_1.ki = ki_1;
                    ctx_curr_1.skv_tiles = causal_skv_1;
                    ctx_curr_1.task_id_mod2 = (task_id_8 % 2);
                    ctx_curr_1.task_id_mod3 = (task_id_8 % 3);
                    ctx_curr_1.q_count_mod3 = (q_count_0 % 3);
                }
                if ((task_id_8 > 0)) {
                    if ((ki_1 < ((int64_t)(causal_skv_1 + ((uint64_t)1))))) {
                        auto& p_ctx_0 = ctx_arr_1[((task_id_8 + 3) % 4)];
                        const int64_t _inl2_QK_READY_FORWARD_IDS_0[] = {0, 1};
                        const int64_t _inl2_QK_READY_BARKWARD_IDS_0[] = {2, 3};
                        const int64_t _inl2_P_READY_FORWARD_IDS_0[] = {4, 5, 6};
                        const int64_t _dyn_arr_8[] = {0, 1};
                        wait_intra_block(PIPE_V, _dyn_arr_8[p_ctx_0.task_id_mod2]);
                        _tg_qk_vec_db_cursor_0.cursor = (_tg_qk_vec_db_cursor_0.cursor + 1);
                        auto _inl2__bufidx_0_0 = ((_tg_qk_vec_db_cursor_0.cursor + 1) % 2);
                        _tg_p_f16_db_cursor_0.cursor = (_tg_p_f16_db_cursor_0.cursor + 1);
                        auto _inl2__bufidx_1_0 = ((_tg_p_f16_db_cursor_0.cursor + 1) % 2);
                        _tg_p_f16_main_db_cursor_0.cursor = (_tg_p_f16_main_db_cursor_0.cursor + 1);
                        auto _inl2__bufidx_2_0 = ((_tg_p_f16_main_db_cursor_0.cursor + 1) % 2);
                        _tg_p_f16_back_db_cursor_0.cursor = (_tg_p_f16_back_db_cursor_0.cursor + 1);
                        auto _inl2__bufidx_3_0 = ((_tg_p_f16_back_db_cursor_0.cursor + 1) % 2);
                        _tg_p_mat_db_cursor_0.cursor = (_tg_p_mat_db_cursor_0.cursor + 1);
                        auto _inl2__bufidx_4_0 = ((_tg_p_mat_db_cursor_0.cursor + 1) % 3);
                        if (NeedAttnMask) {
                            if ((p_ctx_0.ki == (p_ctx_0.skv_tiles - 1))) {
                                _tg_mask_db_cursor_0.cursor = (_tg_mask_db_cursor_0.cursor + 1);
                                auto _inl2__bufidx_5_0 = ((_tg_mask_db_cursor_0.cursor + 1) % 2);
                                get_buf(PIPE_MTE2, _tg_mask_db_mutex_ids_0[_inl2__bufidx_5_0], 0);
                                tensor_attn_mask_0.SetShape<pto::GlobalTensorDim::DIM_3, pto::GlobalTensorDim::DIM_4>(static_cast<int64_t>(128), static_cast<int64_t>(64));
                                TASSIGN(tensor_attn_mask_0, atten_mask_0 + (0 * 2048 + (sub_id_0 * 64)));
                                TLOAD(_tg_mask_db_tiles_0[_inl2__bufidx_5_0], tensor_attn_mask_0);
                                rls_buf(PIPE_MTE2, _tg_mask_db_mutex_ids_0[_inl2__bufidx_5_0], 0);
                                if ((p_ctx_0.ki == 0)) {
                                    get_buf(PIPE_V, _tg_p_f16_db_mutex_ids_0[_inl2__bufidx_1_0], 0);
                                    get_buf(PIPE_V, _tg_mask_db_mutex_ids_0[_inl2__bufidx_5_0], 0);
                                    __ubuf__ float *vf_tile_ptr_0 = (__ubuf__ float *)_tg_qk_vec_db_tiles_0[_inl2__bufidx_0_0].data();
                                    __ubuf__ half *vf_tile_ptr_1 = (__ubuf__ half *)_tg_p_f16_db_tiles_0[_inl2__bufidx_1_0].data();
                                    __ubuf__ uint8_t *vf_tile_ptr_2 = (__ubuf__ uint8_t *)_tg_mask_db_tiles_0[_inl2__bufidx_5_0].data();
                                    __ubuf__ float *vf_tile_ptr_3 = (__ubuf__ float *)global_max_rm_buf_0[p_ctx_0.q_count_mod3].data();
                                    __ubuf__ half *vf_tile_ptr_4 = (__ubuf__ half *)_tg_p_f16_db_tiles_0[_inl2__bufidx_1_0].data();
                                    __ubuf__ half *vf_tile_ptr_5 = (vf_tile_ptr_1 + (512));
                                    __ubuf__ float *vf_tile_ptr_6 = (__ubuf__ float *)global_sum_rm_buf_0[p_ctx_0.q_count_mod3].data();
                                    __VEC_SCOPE__ {
                                        RegTensor<float> _inl3_src0_0;
                                        RegTensor<float> _inl3_src1_0;
                                        RegTensor<float> _inl3_src2_0;
                                        RegTensor<float> _inl3_src3_0;
                                        RegTensor<float> _inl3_max0_0;
                                        RegTensor<float> _inl3_max1_0;
                                        RegTensor<float> _inl3_max2_0;
                                        RegTensor<float> _inl3_max3_0;
                                        RegTensor<float> _inl3_vreg_x_f32_0_0;
                                        RegTensor<float> _inl3_vreg_x_f32_1_0;
                                        RegTensor<float> _inl3_vreg_x_f32_2_0;
                                        RegTensor<float> _inl3_vreg_x_f32_3_0;
                                        RegTensor<float> _inl3_vreg_x_exp_0_0;
                                        RegTensor<float> _inl3_vreg_x_exp_1_0;
                                        RegTensor<float> _inl3_vreg_x_exp_2_0;
                                        RegTensor<float> _inl3_vreg_x_exp_3_0;
                                        RegTensor<float> _inl3_vreg_x_sum_0_0;
                                        RegTensor<float> _inl3_vreg_x_sum_1_0;
                                        RegTensor<float> _inl3_vreg_x_sum_2_0;
                                        RegTensor<float> _inl3_vreg_x_sum_3_0;
                                        RegTensor<float> _inl3_vreg_x_sum0_0;
                                        RegTensor<float> _inl3_vreg_x_sum1_0;
                                        RegTensor<float> _inl3_vreg_x_max_f32_b_0;
                                        RegTensor<float> _inl3_vreg_l0_0;
                                        RegTensor<half> _inl3_vreg_x_exp_even_f16_0;
                                        RegTensor<half> _inl3_vreg_x_exp_odd_f16_0;
                                        RegTensor<half> _inl3_vreg_x_exp_even_f16_1_0;
                                        RegTensor<half> _inl3_vreg_x_exp_odd_f16_1_0;
                                        RegTensor<half> _inl3_vreg_x_exp_f16_pack_0;
                                        RegTensor<half> _inl3_vreg_x_exp_f16_packa_0;
                                        RegTensor<half> _inl3_vreg_x_exp_f16_1_pack_0;
                                        RegTensor<half> _inl3_vreg_x_exp_f16_1_packa_0;
                                        MaskReg _inl3_preg_108_0;
                                        _inl3_preg_108_0 = pset_b32(PAT_ALL);
                                        uint32_t _vf_mask_scalar_0 = (uint32_t)128;
                                        MaskReg _inl3_preg_136_0;
                                        _inl3_preg_136_0 = plt_b16(_vf_mask_scalar_0, POST_UPDATE);
                                        MaskReg _inl3_preg_compare0_0;
                                        _inl3_preg_compare0_0 = pset_b32(PAT_ALL);
                                        MaskReg _inl3_preg_compare1_0;
                                        _inl3_preg_compare1_0 = pset_b32(PAT_ALL);
                                        MaskReg _inl3_preg_compare2_0;
                                        _inl3_preg_compare2_0 = pset_b32(PAT_ALL);
                                        MaskReg _inl3_preg_compare3_0;
                                        _inl3_preg_compare3_0 = pset_b32(PAT_ALL);
                                        RegTensor<float> _inl3_vreg_min_0;
                                        vbr(_inl3_max0_0, -1000000000.000000);
                                        vbr(_inl3_max1_0, -1000000000.000000);
                                        vbr(_inl3_max2_0, -1000000000.000000);
                                        vbr(_inl3_max3_0, -1000000000.000000);
                                        vbr(_inl3_vreg_min_0, -1000000000.000000);
                                        for (uint16_t _inl3_iter_m_0 = 0; _inl3_iter_m_0 < (uint16_t)(32); _inl3_iter_m_0 += 1) {
                                            vlds(_inl3_src0_0, vf_tile_ptr_0, ((_inl3_iter_m_0 * 64) * 4), NORM);
                                            vlds(_inl3_src1_0, (vf_tile_ptr_0 + (64)), ((_inl3_iter_m_0 * 64) * 4), NORM);
                                            vlds(_inl3_src2_0, (vf_tile_ptr_0 + (128)), ((_inl3_iter_m_0 * 64) * 4), NORM);
                                            vlds(_inl3_src3_0, (vf_tile_ptr_0 + (192)), ((_inl3_iter_m_0 * 64) * 4), NORM);
                                            plds(_inl3_preg_compare0_0, (__ubuf__ uint32_t *)vf_tile_ptr_2, ((_inl3_iter_m_0 * 64) * 4), DS);
                                            plds(_inl3_preg_compare1_0, (__ubuf__ uint32_t *)vf_tile_ptr_2, (64 + ((_inl3_iter_m_0 * 64) * 4)), DS);
                                            plds(_inl3_preg_compare2_0, (__ubuf__ uint32_t *)vf_tile_ptr_2, (128 + ((_inl3_iter_m_0 * 64) * 4)), DS);
                                            plds(_inl3_preg_compare3_0, (__ubuf__ uint32_t *)vf_tile_ptr_2, (192 + ((_inl3_iter_m_0 * 64) * 4)), DS);
                                            vsel(_inl3_src0_0, _inl3_vreg_min_0, _inl3_src0_0, _inl3_preg_compare0_0);
                                            vsel(_inl3_src1_0, _inl3_vreg_min_0, _inl3_src1_0, _inl3_preg_compare1_0);
                                            vsel(_inl3_src2_0, _inl3_vreg_min_0, _inl3_src2_0, _inl3_preg_compare2_0);
                                            vsel(_inl3_src3_0, _inl3_vreg_min_0, _inl3_src3_0, _inl3_preg_compare3_0);
                                            vsts(_inl3_src0_0, (vf_tile_ptr_0 + (((_inl3_iter_m_0 * 64) * 4))), 0, NORM_B32, _inl3_preg_108_0);
                                            vsts(_inl3_src1_0, (vf_tile_ptr_0 + ((64 + ((_inl3_iter_m_0 * 64) * 4)))), 0, NORM_B32, _inl3_preg_108_0);
                                            vsts(_inl3_src2_0, (vf_tile_ptr_0 + ((128 + ((_inl3_iter_m_0 * 64) * 4)))), 0, NORM_B32, _inl3_preg_108_0);
                                            vsts(_inl3_src3_0, (vf_tile_ptr_0 + ((192 + ((_inl3_iter_m_0 * 64) * 4)))), 0, NORM_B32, _inl3_preg_108_0);
                                            vmax(_inl3_max0_0, _inl3_max0_0, _inl3_src0_0, _inl3_preg_108_0, MODE_ZEROING);
                                            vmax(_inl3_max1_0, _inl3_max1_0, _inl3_src1_0, _inl3_preg_108_0, MODE_ZEROING);
                                            vmax(_inl3_max2_0, _inl3_max2_0, _inl3_src2_0, _inl3_preg_108_0, MODE_ZEROING);
                                            vmax(_inl3_max3_0, _inl3_max3_0, _inl3_src3_0, _inl3_preg_108_0, MODE_ZEROING);
                                        }
                                        vmax(_inl3_max0_0, _inl3_max0_0, _inl3_max2_0, _inl3_preg_108_0, MODE_ZEROING);
                                        vmax(_inl3_max1_0, _inl3_max1_0, _inl3_max3_0, _inl3_preg_108_0, MODE_ZEROING);
                                        vmax(_inl3_max0_0, _inl3_max0_0, _inl3_max1_0, _inl3_preg_108_0, MODE_ZEROING);
                                        vmuls(_inl3_max0_0, _inl3_max0_0, 0.088388, _inl3_preg_108_0);
                                        vsts(_inl3_max0_0, vf_tile_ptr_3, 0, NORM_B32, _inl3_preg_108_0);
                                        vdup(_inl3_vreg_x_sum_0_0, 0.000000, _inl3_preg_108_0, MODE_ZEROING);
                                        vdup(_inl3_vreg_x_sum_1_0, 0.000000, _inl3_preg_108_0, MODE_ZEROING);
                                        vdup(_inl3_vreg_x_sum_2_0, 0.000000, _inl3_preg_108_0, MODE_ZEROING);
                                        vdup(_inl3_vreg_x_sum_3_0, 0.000000, _inl3_preg_108_0, MODE_ZEROING);
                                        mem_bar(VST_VLD);
                                        for (uint16_t _inl3_i0_0 = 0; _inl3_i0_0 < (uint16_t)(32); _inl3_i0_0 += 1) {
                                            vlds(_inl3_vreg_x_f32_0_0, vf_tile_ptr_0, (_inl3_i0_0 * 64), NORM);
                                            vlds(_inl3_vreg_x_f32_1_0, vf_tile_ptr_0, (2048 + (_inl3_i0_0 * 64)), NORM);
                                            vlds(_inl3_vreg_x_f32_2_0, vf_tile_ptr_0, (4096 + (_inl3_i0_0 * 64)), NORM);
                                            vlds(_inl3_vreg_x_f32_3_0, vf_tile_ptr_0, (6144 + (_inl3_i0_0 * 64)), NORM);
                                            vmuls(_inl3_vreg_x_f32_0_0, _inl3_vreg_x_f32_0_0, 0.088388, _inl3_preg_108_0);
                                            vmuls(_inl3_vreg_x_f32_1_0, _inl3_vreg_x_f32_1_0, 0.088388, _inl3_preg_108_0);
                                            vmuls(_inl3_vreg_x_f32_2_0, _inl3_vreg_x_f32_2_0, 0.088388, _inl3_preg_108_0);
                                            vmuls(_inl3_vreg_x_f32_3_0, _inl3_vreg_x_f32_3_0, 0.088388, _inl3_preg_108_0);
                                            vexpdif(_inl3_vreg_x_exp_0_0, _inl3_vreg_x_f32_0_0, _inl3_max0_0, _inl3_preg_108_0, PART_EVEN);
                                            vexpdif(_inl3_vreg_x_exp_1_0, _inl3_vreg_x_f32_1_0, _inl3_max0_0, _inl3_preg_108_0, PART_EVEN);
                                            vexpdif(_inl3_vreg_x_exp_2_0, _inl3_vreg_x_f32_2_0, _inl3_max0_0, _inl3_preg_108_0, PART_EVEN);
                                            vexpdif(_inl3_vreg_x_exp_3_0, _inl3_vreg_x_f32_3_0, _inl3_max0_0, _inl3_preg_108_0, PART_EVEN);
                                            vcvt(_inl3_vreg_x_exp_even_f16_0, _inl3_vreg_x_exp_0_0, _inl3_preg_108_0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                                            vcvt(_inl3_vreg_x_exp_odd_f16_0, _inl3_vreg_x_exp_2_0, _inl3_preg_108_0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                                            vdintlv(_inl3_vreg_x_exp_f16_pack_0, _inl3_vreg_x_exp_f16_packa_0, _inl3_vreg_x_exp_even_f16_0, _inl3_vreg_x_exp_odd_f16_0);
                                            vcvt(_inl3_vreg_x_exp_even_f16_1_0, _inl3_vreg_x_exp_1_0, _inl3_preg_108_0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                                            vcvt(_inl3_vreg_x_exp_odd_f16_1_0, _inl3_vreg_x_exp_3_0, _inl3_preg_108_0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                                            vdintlv(_inl3_vreg_x_exp_f16_1_pack_0, _inl3_vreg_x_exp_f16_1_packa_0, _inl3_vreg_x_exp_even_f16_1_0, _inl3_vreg_x_exp_odd_f16_1_0);
                                            vsstb(_inl3_vreg_x_exp_f16_pack_0, vf_tile_ptr_4, (65 << 16u) | (1 & 0xFFFFU), _inl3_preg_136_0, POST_UPDATE);
                                            vadd(_inl3_vreg_x_sum_0_0, _inl3_vreg_x_exp_0_0, _inl3_vreg_x_sum_0_0, _inl3_preg_108_0, MODE_ZEROING);
                                            vadd(_inl3_vreg_x_sum_2_0, _inl3_vreg_x_exp_2_0, _inl3_vreg_x_sum_2_0, _inl3_preg_108_0, MODE_ZEROING);
                                            vsstb(_inl3_vreg_x_exp_f16_1_pack_0, vf_tile_ptr_5, (65 << 16u) | (1 & 0xFFFFU), _inl3_preg_136_0, POST_UPDATE);
                                            vadd(_inl3_vreg_x_sum_1_0, _inl3_vreg_x_exp_1_0, _inl3_vreg_x_sum_1_0, _inl3_preg_108_0, MODE_ZEROING);
                                            vadd(_inl3_vreg_x_sum_3_0, _inl3_vreg_x_exp_3_0, _inl3_vreg_x_sum_3_0, _inl3_preg_108_0, MODE_ZEROING);
                                        }
                                        vadd(_inl3_vreg_x_sum0_0, _inl3_vreg_x_sum_2_0, _inl3_vreg_x_sum_0_0, _inl3_preg_108_0, MODE_ZEROING);
                                        vadd(_inl3_vreg_x_sum1_0, _inl3_vreg_x_sum_3_0, _inl3_vreg_x_sum_1_0, _inl3_preg_108_0, MODE_ZEROING);
                                        vadd(_inl3_vreg_x_sum0_0, _inl3_vreg_x_sum0_0, _inl3_vreg_x_sum1_0, _inl3_preg_108_0, MODE_ZEROING);
                                        vsts(_inl3_vreg_x_sum0_0, vf_tile_ptr_6, 0, NORM_B32, _inl3_preg_108_0);
                                    }
                                    rls_buf(PIPE_V, _tg_p_f16_db_mutex_ids_0[_inl2__bufidx_1_0], 0);
                                    rls_buf(PIPE_V, _tg_mask_db_mutex_ids_0[_inl2__bufidx_5_0], 0);
                                } else {
                                    get_buf(PIPE_V, _tg_p_f16_db_mutex_ids_0[_inl2__bufidx_1_0], 0);
                                    get_buf(PIPE_V, _tg_mask_db_mutex_ids_0[_inl2__bufidx_5_0], 0);
                                    __ubuf__ float *vf_tile_ptr_7 = (__ubuf__ float *)_tg_qk_vec_db_tiles_0[_inl2__bufidx_0_0].data();
                                    __ubuf__ half *vf_tile_ptr_8 = (__ubuf__ half *)_tg_p_f16_db_tiles_0[_inl2__bufidx_1_0].data();
                                    __ubuf__ uint8_t *vf_tile_ptr_9 = (__ubuf__ uint8_t *)_tg_mask_db_tiles_0[_inl2__bufidx_5_0].data();
                                    __ubuf__ float *vf_tile_ptr_10 = (__ubuf__ float *)global_max_rm_buf_0[p_ctx_0.q_count_mod3].data();
                                    __ubuf__ float *vf_tile_ptr_11 = (__ubuf__ float *)exp_corr_rm_fifo_0[p_ctx_0.task_id_mod3].data();
                                    __ubuf__ half *vf_tile_ptr_12 = (__ubuf__ half *)_tg_p_f16_db_tiles_0[_inl2__bufidx_1_0].data();
                                    __ubuf__ half *vf_tile_ptr_13 = (vf_tile_ptr_8 + (512));
                                    __ubuf__ float *vf_tile_ptr_14 = (__ubuf__ float *)global_sum_rm_buf_0[p_ctx_0.q_count_mod3].data();
                                    __VEC_SCOPE__ {
                                        RegTensor<float> _inl4_src0_0;
                                        RegTensor<float> _inl4_src1_0;
                                        RegTensor<float> _inl4_src2_0;
                                        RegTensor<float> _inl4_src3_0;
                                        RegTensor<float> _inl4_max0_0;
                                        RegTensor<float> _inl4_max1_0;
                                        RegTensor<float> _inl4_max2_0;
                                        RegTensor<float> _inl4_max3_0;
                                        RegTensor<float> _inl4_vreg_x_f32_0_0;
                                        RegTensor<float> _inl4_vreg_x_f32_1_0;
                                        RegTensor<float> _inl4_vreg_x_f32_2_0;
                                        RegTensor<float> _inl4_vreg_x_f32_3_0;
                                        RegTensor<float> _inl4_vreg_x_exp_0_0;
                                        RegTensor<float> _inl4_vreg_x_exp_1_0;
                                        RegTensor<float> _inl4_vreg_x_exp_2_0;
                                        RegTensor<float> _inl4_vreg_x_exp_3_0;
                                        RegTensor<float> _inl4_vreg_x_sum_0_0;
                                        RegTensor<float> _inl4_vreg_x_sum_1_0;
                                        RegTensor<float> _inl4_vreg_x_sum_2_0;
                                        RegTensor<float> _inl4_vreg_x_sum_3_0;
                                        RegTensor<float> _inl4_vreg_x_sum0_0;
                                        RegTensor<float> _inl4_vreg_x_sum1_0;
                                        RegTensor<float> _inl4_vreg_x_max_f32_b_0;
                                        RegTensor<float> _inl4_vreg_l0_0;
                                        RegTensor<half> _inl4_vreg_x_exp_even_f16_0;
                                        RegTensor<half> _inl4_vreg_x_exp_odd_f16_0;
                                        RegTensor<half> _inl4_vreg_x_exp_even_f16_1_0;
                                        RegTensor<half> _inl4_vreg_x_exp_odd_f16_1_0;
                                        RegTensor<half> _inl4_vreg_x_exp_f16_pack_0;
                                        RegTensor<half> _inl4_vreg_x_exp_f16_packa_0;
                                        RegTensor<half> _inl4_vreg_x_exp_f16_1_pack_0;
                                        RegTensor<half> _inl4_vreg_x_exp_f16_1_packa_0;
                                        MaskReg _inl4_preg_108_0;
                                        _inl4_preg_108_0 = pset_b32(PAT_ALL);
                                        uint32_t _vf_mask_scalar_1 = (uint32_t)128;
                                        MaskReg _inl4_preg_136_0;
                                        _inl4_preg_136_0 = plt_b16(_vf_mask_scalar_1, POST_UPDATE);
                                        MaskReg _inl4_preg_compare0_0;
                                        _inl4_preg_compare0_0 = pset_b32(PAT_ALL);
                                        MaskReg _inl4_preg_compare1_0;
                                        _inl4_preg_compare1_0 = pset_b32(PAT_ALL);
                                        MaskReg _inl4_preg_compare2_0;
                                        _inl4_preg_compare2_0 = pset_b32(PAT_ALL);
                                        MaskReg _inl4_preg_compare3_0;
                                        _inl4_preg_compare3_0 = pset_b32(PAT_ALL);
                                        RegTensor<float> _inl4_vreg_min_0;
                                        vbr(_inl4_max0_0, -1000000000.000000);
                                        vbr(_inl4_max1_0, -1000000000.000000);
                                        vbr(_inl4_max2_0, -1000000000.000000);
                                        vbr(_inl4_max3_0, -1000000000.000000);
                                        vbr(_inl4_vreg_min_0, -1000000000.000000);
                                        for (uint16_t _inl4_iter_m_0 = 0; _inl4_iter_m_0 < (uint16_t)(32); _inl4_iter_m_0 += 1) {
                                            vlds(_inl4_src0_0, vf_tile_ptr_7, ((_inl4_iter_m_0 * 64) * 4), NORM);
                                            vlds(_inl4_src1_0, (vf_tile_ptr_7 + (64)), ((_inl4_iter_m_0 * 64) * 4), NORM);
                                            vlds(_inl4_src2_0, (vf_tile_ptr_7 + (128)), ((_inl4_iter_m_0 * 64) * 4), NORM);
                                            vlds(_inl4_src3_0, (vf_tile_ptr_7 + (192)), ((_inl4_iter_m_0 * 64) * 4), NORM);
                                            plds(_inl4_preg_compare0_0, (__ubuf__ uint32_t *)vf_tile_ptr_9, ((_inl4_iter_m_0 * 64) * 4), DS);
                                            plds(_inl4_preg_compare1_0, (__ubuf__ uint32_t *)vf_tile_ptr_9, (64 + ((_inl4_iter_m_0 * 64) * 4)), DS);
                                            plds(_inl4_preg_compare2_0, (__ubuf__ uint32_t *)vf_tile_ptr_9, (128 + ((_inl4_iter_m_0 * 64) * 4)), DS);
                                            plds(_inl4_preg_compare3_0, (__ubuf__ uint32_t *)vf_tile_ptr_9, (192 + ((_inl4_iter_m_0 * 64) * 4)), DS);
                                            vsel(_inl4_src0_0, _inl4_vreg_min_0, _inl4_src0_0, _inl4_preg_compare0_0);
                                            vsel(_inl4_src1_0, _inl4_vreg_min_0, _inl4_src1_0, _inl4_preg_compare1_0);
                                            vsel(_inl4_src2_0, _inl4_vreg_min_0, _inl4_src2_0, _inl4_preg_compare2_0);
                                            vsel(_inl4_src3_0, _inl4_vreg_min_0, _inl4_src3_0, _inl4_preg_compare3_0);
                                            vsts(_inl4_src0_0, (vf_tile_ptr_7 + (((_inl4_iter_m_0 * 64) * 4))), 0, NORM_B32, _inl4_preg_108_0);
                                            vsts(_inl4_src1_0, (vf_tile_ptr_7 + ((64 + ((_inl4_iter_m_0 * 64) * 4)))), 0, NORM_B32, _inl4_preg_108_0);
                                            vsts(_inl4_src2_0, (vf_tile_ptr_7 + ((128 + ((_inl4_iter_m_0 * 64) * 4)))), 0, NORM_B32, _inl4_preg_108_0);
                                            vsts(_inl4_src3_0, (vf_tile_ptr_7 + ((192 + ((_inl4_iter_m_0 * 64) * 4)))), 0, NORM_B32, _inl4_preg_108_0);
                                            vmax(_inl4_max0_0, _inl4_max0_0, _inl4_src0_0, _inl4_preg_108_0, MODE_ZEROING);
                                            vmax(_inl4_max1_0, _inl4_max1_0, _inl4_src1_0, _inl4_preg_108_0, MODE_ZEROING);
                                            vmax(_inl4_max2_0, _inl4_max2_0, _inl4_src2_0, _inl4_preg_108_0, MODE_ZEROING);
                                            vmax(_inl4_max3_0, _inl4_max3_0, _inl4_src3_0, _inl4_preg_108_0, MODE_ZEROING);
                                        }
                                        vlds(_inl4_vreg_x_max_f32_b_0, vf_tile_ptr_10, 0, NORM);
                                        vmax(_inl4_max0_0, _inl4_max0_0, _inl4_max2_0, _inl4_preg_108_0, MODE_ZEROING);
                                        vmax(_inl4_max1_0, _inl4_max1_0, _inl4_max3_0, _inl4_preg_108_0, MODE_ZEROING);
                                        vmax(_inl4_max0_0, _inl4_max0_0, _inl4_max1_0, _inl4_preg_108_0, MODE_ZEROING);
                                        vmuls(_inl4_max0_0, _inl4_max0_0, 0.088388, _inl4_preg_108_0);
                                        vmax(_inl4_max0_0, _inl4_max0_0, _inl4_vreg_x_max_f32_b_0, _inl4_preg_108_0, MODE_ZEROING);
                                        vexpdif(_inl4_vreg_x_max_f32_b_0, _inl4_vreg_x_max_f32_b_0, _inl4_max0_0, _inl4_preg_108_0, PART_EVEN);
                                        vsts(_inl4_max0_0, vf_tile_ptr_10, 0, NORM_B32, _inl4_preg_108_0);
                                        vsts(_inl4_vreg_x_max_f32_b_0, vf_tile_ptr_11, 0, NORM_B32, _inl4_preg_108_0);
                                        vdup(_inl4_vreg_x_sum_0_0, 0.000000, _inl4_preg_108_0, MODE_ZEROING);
                                        vdup(_inl4_vreg_x_sum_1_0, 0.000000, _inl4_preg_108_0, MODE_ZEROING);
                                        vdup(_inl4_vreg_x_sum_2_0, 0.000000, _inl4_preg_108_0, MODE_ZEROING);
                                        vdup(_inl4_vreg_x_sum_3_0, 0.000000, _inl4_preg_108_0, MODE_ZEROING);
                                        mem_bar(VST_VLD);
                                        for (uint16_t _inl4_i0_0 = 0; _inl4_i0_0 < (uint16_t)(32); _inl4_i0_0 += 1) {
                                            vlds(_inl4_vreg_x_f32_0_0, vf_tile_ptr_7, (_inl4_i0_0 * 64), NORM);
                                            vlds(_inl4_vreg_x_f32_1_0, vf_tile_ptr_7, (2048 + (_inl4_i0_0 * 64)), NORM);
                                            vlds(_inl4_vreg_x_f32_2_0, vf_tile_ptr_7, (4096 + (_inl4_i0_0 * 64)), NORM);
                                            vlds(_inl4_vreg_x_f32_3_0, vf_tile_ptr_7, (6144 + (_inl4_i0_0 * 64)), NORM);
                                            vmuls(_inl4_vreg_x_f32_0_0, _inl4_vreg_x_f32_0_0, 0.088388, _inl4_preg_108_0);
                                            vmuls(_inl4_vreg_x_f32_1_0, _inl4_vreg_x_f32_1_0, 0.088388, _inl4_preg_108_0);
                                            vmuls(_inl4_vreg_x_f32_2_0, _inl4_vreg_x_f32_2_0, 0.088388, _inl4_preg_108_0);
                                            vmuls(_inl4_vreg_x_f32_3_0, _inl4_vreg_x_f32_3_0, 0.088388, _inl4_preg_108_0);
                                            vexpdif(_inl4_vreg_x_exp_0_0, _inl4_vreg_x_f32_0_0, _inl4_max0_0, _inl4_preg_108_0, PART_EVEN);
                                            vexpdif(_inl4_vreg_x_exp_1_0, _inl4_vreg_x_f32_1_0, _inl4_max0_0, _inl4_preg_108_0, PART_EVEN);
                                            vexpdif(_inl4_vreg_x_exp_2_0, _inl4_vreg_x_f32_2_0, _inl4_max0_0, _inl4_preg_108_0, PART_EVEN);
                                            vexpdif(_inl4_vreg_x_exp_3_0, _inl4_vreg_x_f32_3_0, _inl4_max0_0, _inl4_preg_108_0, PART_EVEN);
                                            vcvt(_inl4_vreg_x_exp_even_f16_0, _inl4_vreg_x_exp_0_0, _inl4_preg_108_0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                                            vcvt(_inl4_vreg_x_exp_odd_f16_0, _inl4_vreg_x_exp_2_0, _inl4_preg_108_0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                                            vdintlv(_inl4_vreg_x_exp_f16_pack_0, _inl4_vreg_x_exp_f16_packa_0, _inl4_vreg_x_exp_even_f16_0, _inl4_vreg_x_exp_odd_f16_0);
                                            vcvt(_inl4_vreg_x_exp_even_f16_1_0, _inl4_vreg_x_exp_1_0, _inl4_preg_108_0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                                            vcvt(_inl4_vreg_x_exp_odd_f16_1_0, _inl4_vreg_x_exp_3_0, _inl4_preg_108_0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                                            vdintlv(_inl4_vreg_x_exp_f16_1_pack_0, _inl4_vreg_x_exp_f16_1_packa_0, _inl4_vreg_x_exp_even_f16_1_0, _inl4_vreg_x_exp_odd_f16_1_0);
                                            vsstb(_inl4_vreg_x_exp_f16_pack_0, vf_tile_ptr_12, (65 << 16u) | (1 & 0xFFFFU), _inl4_preg_136_0, POST_UPDATE);
                                            vadd(_inl4_vreg_x_sum_0_0, _inl4_vreg_x_exp_0_0, _inl4_vreg_x_sum_0_0, _inl4_preg_108_0, MODE_ZEROING);
                                            vadd(_inl4_vreg_x_sum_2_0, _inl4_vreg_x_exp_2_0, _inl4_vreg_x_sum_2_0, _inl4_preg_108_0, MODE_ZEROING);
                                            vsstb(_inl4_vreg_x_exp_f16_1_pack_0, vf_tile_ptr_13, (65 << 16u) | (1 & 0xFFFFU), _inl4_preg_136_0, POST_UPDATE);
                                            vadd(_inl4_vreg_x_sum_1_0, _inl4_vreg_x_exp_1_0, _inl4_vreg_x_sum_1_0, _inl4_preg_108_0, MODE_ZEROING);
                                            vadd(_inl4_vreg_x_sum_3_0, _inl4_vreg_x_exp_3_0, _inl4_vreg_x_sum_3_0, _inl4_preg_108_0, MODE_ZEROING);
                                        }
                                        vadd(_inl4_vreg_x_sum0_0, _inl4_vreg_x_sum_2_0, _inl4_vreg_x_sum_0_0, _inl4_preg_108_0, MODE_ZEROING);
                                        vadd(_inl4_vreg_x_sum1_0, _inl4_vreg_x_sum_3_0, _inl4_vreg_x_sum_1_0, _inl4_preg_108_0, MODE_ZEROING);
                                        vadd(_inl4_vreg_x_sum0_0, _inl4_vreg_x_sum0_0, _inl4_vreg_x_sum1_0, _inl4_preg_108_0, MODE_ZEROING);
                                        vlds(_inl4_vreg_l0_0, vf_tile_ptr_14, 0, NORM);
                                        vmul(_inl4_vreg_l0_0, _inl4_vreg_x_max_f32_b_0, _inl4_vreg_l0_0, _inl4_preg_108_0, MODE_ZEROING);
                                        vadd(_inl4_vreg_l0_0, _inl4_vreg_l0_0, _inl4_vreg_x_sum0_0, _inl4_preg_108_0, MODE_ZEROING);
                                        vsts(_inl4_vreg_l0_0, vf_tile_ptr_14, 0, NORM_B32, _inl4_preg_108_0);
                                    }
                                    rls_buf(PIPE_V, _tg_p_f16_db_mutex_ids_0[_inl2__bufidx_1_0], 0);
                                    rls_buf(PIPE_V, _tg_mask_db_mutex_ids_0[_inl2__bufidx_5_0], 0);
                                }
                            } else {
                                if ((p_ctx_0.ki == 0)) {
                                    get_buf(PIPE_V, _tg_p_f16_db_mutex_ids_0[_inl2__bufidx_1_0], 0);
                                    __ubuf__ float *vf_tile_ptr_15 = (__ubuf__ float *)_tg_qk_vec_db_tiles_0[_inl2__bufidx_0_0].data();
                                    __ubuf__ half *vf_tile_ptr_16 = (__ubuf__ half *)_tg_p_f16_db_tiles_0[_inl2__bufidx_1_0].data();
                                    __ubuf__ float *vf_tile_ptr_17 = (__ubuf__ float *)global_max_rm_buf_0[p_ctx_0.q_count_mod3].data();
                                    __ubuf__ half *vf_tile_ptr_18 = (__ubuf__ half *)_tg_p_f16_db_tiles_0[_inl2__bufidx_1_0].data();
                                    __ubuf__ half *vf_tile_ptr_19 = (vf_tile_ptr_16 + (512));
                                    __ubuf__ float *vf_tile_ptr_20 = (__ubuf__ float *)global_sum_rm_buf_0[p_ctx_0.q_count_mod3].data();
                                    __VEC_SCOPE__ {
                                        RegTensor<float> _inl5_src0_0;
                                        RegTensor<float> _inl5_src1_0;
                                        RegTensor<float> _inl5_src2_0;
                                        RegTensor<float> _inl5_src3_0;
                                        RegTensor<float> _inl5_max0_0;
                                        RegTensor<float> _inl5_max1_0;
                                        RegTensor<float> _inl5_max2_0;
                                        RegTensor<float> _inl5_max3_0;
                                        RegTensor<float> _inl5_vreg_x_f32_0_0;
                                        RegTensor<float> _inl5_vreg_x_f32_1_0;
                                        RegTensor<float> _inl5_vreg_x_f32_2_0;
                                        RegTensor<float> _inl5_vreg_x_f32_3_0;
                                        RegTensor<float> _inl5_vreg_x_exp_0_0;
                                        RegTensor<float> _inl5_vreg_x_exp_1_0;
                                        RegTensor<float> _inl5_vreg_x_exp_2_0;
                                        RegTensor<float> _inl5_vreg_x_exp_3_0;
                                        RegTensor<float> _inl5_vreg_x_sum_0_0;
                                        RegTensor<float> _inl5_vreg_x_sum_1_0;
                                        RegTensor<float> _inl5_vreg_x_sum_2_0;
                                        RegTensor<float> _inl5_vreg_x_sum_3_0;
                                        RegTensor<float> _inl5_vreg_x_sum0_0;
                                        RegTensor<float> _inl5_vreg_x_sum1_0;
                                        RegTensor<float> _inl5_vreg_x_max_f32_b_0;
                                        RegTensor<float> _inl5_vreg_l0_0;
                                        RegTensor<half> _inl5_vreg_x_exp_even_f16_0;
                                        RegTensor<half> _inl5_vreg_x_exp_odd_f16_0;
                                        RegTensor<half> _inl5_vreg_x_exp_even_f16_1_0;
                                        RegTensor<half> _inl5_vreg_x_exp_odd_f16_1_0;
                                        RegTensor<half> _inl5_vreg_x_exp_f16_pack_0;
                                        RegTensor<half> _inl5_vreg_x_exp_f16_packa_0;
                                        RegTensor<half> _inl5_vreg_x_exp_f16_1_pack_0;
                                        RegTensor<half> _inl5_vreg_x_exp_f16_1_packa_0;
                                        MaskReg _inl5_preg_108_0;
                                        _inl5_preg_108_0 = pset_b32(PAT_ALL);
                                        uint32_t _vf_mask_scalar_2 = (uint32_t)128;
                                        MaskReg _inl5_preg_136_0;
                                        _inl5_preg_136_0 = plt_b16(_vf_mask_scalar_2, POST_UPDATE);
                                        MaskReg _inl5_preg_compare0_0;
                                        _inl5_preg_compare0_0 = pset_b32(PAT_ALL);
                                        MaskReg _inl5_preg_compare1_0;
                                        _inl5_preg_compare1_0 = pset_b32(PAT_ALL);
                                        MaskReg _inl5_preg_compare2_0;
                                        _inl5_preg_compare2_0 = pset_b32(PAT_ALL);
                                        MaskReg _inl5_preg_compare3_0;
                                        _inl5_preg_compare3_0 = pset_b32(PAT_ALL);
                                        RegTensor<float> _inl5_vreg_min_0;
                                        vbr(_inl5_max0_0, -1000000000.000000);
                                        vbr(_inl5_max1_0, -1000000000.000000);
                                        vbr(_inl5_max2_0, -1000000000.000000);
                                        vbr(_inl5_max3_0, -1000000000.000000);
                                        vbr(_inl5_vreg_min_0, -1000000000.000000);
                                        for (uint16_t _inl5_iter_m_0 = 0; _inl5_iter_m_0 < (uint16_t)(32); _inl5_iter_m_0 += 1) {
                                            vlds(_inl5_src0_0, vf_tile_ptr_15, ((_inl5_iter_m_0 * 64) * 4), NORM);
                                            vlds(_inl5_src1_0, (vf_tile_ptr_15 + (64)), ((_inl5_iter_m_0 * 64) * 4), NORM);
                                            vlds(_inl5_src2_0, (vf_tile_ptr_15 + (128)), ((_inl5_iter_m_0 * 64) * 4), NORM);
                                            vlds(_inl5_src3_0, (vf_tile_ptr_15 + (192)), ((_inl5_iter_m_0 * 64) * 4), NORM);
                                            vmax(_inl5_max0_0, _inl5_max0_0, _inl5_src0_0, _inl5_preg_108_0, MODE_ZEROING);
                                            vmax(_inl5_max1_0, _inl5_max1_0, _inl5_src1_0, _inl5_preg_108_0, MODE_ZEROING);
                                            vmax(_inl5_max2_0, _inl5_max2_0, _inl5_src2_0, _inl5_preg_108_0, MODE_ZEROING);
                                            vmax(_inl5_max3_0, _inl5_max3_0, _inl5_src3_0, _inl5_preg_108_0, MODE_ZEROING);
                                        }
                                        vmax(_inl5_max0_0, _inl5_max0_0, _inl5_max2_0, _inl5_preg_108_0, MODE_ZEROING);
                                        vmax(_inl5_max1_0, _inl5_max1_0, _inl5_max3_0, _inl5_preg_108_0, MODE_ZEROING);
                                        vmax(_inl5_max0_0, _inl5_max0_0, _inl5_max1_0, _inl5_preg_108_0, MODE_ZEROING);
                                        vmuls(_inl5_max0_0, _inl5_max0_0, 0.088388, _inl5_preg_108_0);
                                        vsts(_inl5_max0_0, vf_tile_ptr_17, 0, NORM_B32, _inl5_preg_108_0);
                                        vdup(_inl5_vreg_x_sum_0_0, 0.000000, _inl5_preg_108_0, MODE_ZEROING);
                                        vdup(_inl5_vreg_x_sum_1_0, 0.000000, _inl5_preg_108_0, MODE_ZEROING);
                                        vdup(_inl5_vreg_x_sum_2_0, 0.000000, _inl5_preg_108_0, MODE_ZEROING);
                                        vdup(_inl5_vreg_x_sum_3_0, 0.000000, _inl5_preg_108_0, MODE_ZEROING);
                                        for (uint16_t _inl5_i0_0 = 0; _inl5_i0_0 < (uint16_t)(32); _inl5_i0_0 += 1) {
                                            vlds(_inl5_vreg_x_f32_0_0, vf_tile_ptr_15, (_inl5_i0_0 * 64), NORM);
                                            vlds(_inl5_vreg_x_f32_1_0, vf_tile_ptr_15, (2048 + (_inl5_i0_0 * 64)), NORM);
                                            vlds(_inl5_vreg_x_f32_2_0, vf_tile_ptr_15, (4096 + (_inl5_i0_0 * 64)), NORM);
                                            vlds(_inl5_vreg_x_f32_3_0, vf_tile_ptr_15, (6144 + (_inl5_i0_0 * 64)), NORM);
                                            vmuls(_inl5_vreg_x_f32_0_0, _inl5_vreg_x_f32_0_0, 0.088388, _inl5_preg_108_0);
                                            vmuls(_inl5_vreg_x_f32_1_0, _inl5_vreg_x_f32_1_0, 0.088388, _inl5_preg_108_0);
                                            vmuls(_inl5_vreg_x_f32_2_0, _inl5_vreg_x_f32_2_0, 0.088388, _inl5_preg_108_0);
                                            vmuls(_inl5_vreg_x_f32_3_0, _inl5_vreg_x_f32_3_0, 0.088388, _inl5_preg_108_0);
                                            vexpdif(_inl5_vreg_x_exp_0_0, _inl5_vreg_x_f32_0_0, _inl5_max0_0, _inl5_preg_108_0, PART_EVEN);
                                            vexpdif(_inl5_vreg_x_exp_1_0, _inl5_vreg_x_f32_1_0, _inl5_max0_0, _inl5_preg_108_0, PART_EVEN);
                                            vexpdif(_inl5_vreg_x_exp_2_0, _inl5_vreg_x_f32_2_0, _inl5_max0_0, _inl5_preg_108_0, PART_EVEN);
                                            vexpdif(_inl5_vreg_x_exp_3_0, _inl5_vreg_x_f32_3_0, _inl5_max0_0, _inl5_preg_108_0, PART_EVEN);
                                            vcvt(_inl5_vreg_x_exp_even_f16_0, _inl5_vreg_x_exp_0_0, _inl5_preg_108_0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                                            vcvt(_inl5_vreg_x_exp_odd_f16_0, _inl5_vreg_x_exp_2_0, _inl5_preg_108_0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                                            vdintlv(_inl5_vreg_x_exp_f16_pack_0, _inl5_vreg_x_exp_f16_packa_0, _inl5_vreg_x_exp_even_f16_0, _inl5_vreg_x_exp_odd_f16_0);
                                            vcvt(_inl5_vreg_x_exp_even_f16_1_0, _inl5_vreg_x_exp_1_0, _inl5_preg_108_0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                                            vcvt(_inl5_vreg_x_exp_odd_f16_1_0, _inl5_vreg_x_exp_3_0, _inl5_preg_108_0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                                            vdintlv(_inl5_vreg_x_exp_f16_1_pack_0, _inl5_vreg_x_exp_f16_1_packa_0, _inl5_vreg_x_exp_even_f16_1_0, _inl5_vreg_x_exp_odd_f16_1_0);
                                            vsstb(_inl5_vreg_x_exp_f16_pack_0, vf_tile_ptr_18, (65 << 16u) | (1 & 0xFFFFU), _inl5_preg_136_0, POST_UPDATE);
                                            vadd(_inl5_vreg_x_sum_0_0, _inl5_vreg_x_exp_0_0, _inl5_vreg_x_sum_0_0, _inl5_preg_108_0, MODE_ZEROING);
                                            vadd(_inl5_vreg_x_sum_2_0, _inl5_vreg_x_exp_2_0, _inl5_vreg_x_sum_2_0, _inl5_preg_108_0, MODE_ZEROING);
                                            vsstb(_inl5_vreg_x_exp_f16_1_pack_0, vf_tile_ptr_19, (65 << 16u) | (1 & 0xFFFFU), _inl5_preg_136_0, POST_UPDATE);
                                            vadd(_inl5_vreg_x_sum_1_0, _inl5_vreg_x_exp_1_0, _inl5_vreg_x_sum_1_0, _inl5_preg_108_0, MODE_ZEROING);
                                            vadd(_inl5_vreg_x_sum_3_0, _inl5_vreg_x_exp_3_0, _inl5_vreg_x_sum_3_0, _inl5_preg_108_0, MODE_ZEROING);
                                        }
                                        vadd(_inl5_vreg_x_sum0_0, _inl5_vreg_x_sum_2_0, _inl5_vreg_x_sum_0_0, _inl5_preg_108_0, MODE_ZEROING);
                                        vadd(_inl5_vreg_x_sum1_0, _inl5_vreg_x_sum_3_0, _inl5_vreg_x_sum_1_0, _inl5_preg_108_0, MODE_ZEROING);
                                        vadd(_inl5_vreg_x_sum0_0, _inl5_vreg_x_sum0_0, _inl5_vreg_x_sum1_0, _inl5_preg_108_0, MODE_ZEROING);
                                        vsts(_inl5_vreg_x_sum0_0, vf_tile_ptr_20, 0, NORM_B32, _inl5_preg_108_0);
                                    }
                                    rls_buf(PIPE_V, _tg_p_f16_db_mutex_ids_0[_inl2__bufidx_1_0], 0);
                                } else {
                                    get_buf(PIPE_V, _tg_p_f16_db_mutex_ids_0[_inl2__bufidx_1_0], 0);
                                    __ubuf__ float *vf_tile_ptr_21 = (__ubuf__ float *)_tg_qk_vec_db_tiles_0[_inl2__bufidx_0_0].data();
                                    __ubuf__ half *vf_tile_ptr_22 = (__ubuf__ half *)_tg_p_f16_db_tiles_0[_inl2__bufidx_1_0].data();
                                    __ubuf__ float *vf_tile_ptr_23 = (__ubuf__ float *)global_max_rm_buf_0[p_ctx_0.q_count_mod3].data();
                                    __ubuf__ float *vf_tile_ptr_24 = (__ubuf__ float *)exp_corr_rm_fifo_0[p_ctx_0.task_id_mod3].data();
                                    __ubuf__ half *vf_tile_ptr_25 = (__ubuf__ half *)_tg_p_f16_db_tiles_0[_inl2__bufidx_1_0].data();
                                    __ubuf__ half *vf_tile_ptr_26 = (vf_tile_ptr_22 + (512));
                                    __ubuf__ float *vf_tile_ptr_27 = (__ubuf__ float *)global_sum_rm_buf_0[p_ctx_0.q_count_mod3].data();
                                    __VEC_SCOPE__ {
                                        RegTensor<float> _inl6_src0_0;
                                        RegTensor<float> _inl6_src1_0;
                                        RegTensor<float> _inl6_src2_0;
                                        RegTensor<float> _inl6_src3_0;
                                        RegTensor<float> _inl6_max0_0;
                                        RegTensor<float> _inl6_max1_0;
                                        RegTensor<float> _inl6_max2_0;
                                        RegTensor<float> _inl6_max3_0;
                                        RegTensor<float> _inl6_vreg_x_f32_0_0;
                                        RegTensor<float> _inl6_vreg_x_f32_1_0;
                                        RegTensor<float> _inl6_vreg_x_f32_2_0;
                                        RegTensor<float> _inl6_vreg_x_f32_3_0;
                                        RegTensor<float> _inl6_vreg_x_exp_0_0;
                                        RegTensor<float> _inl6_vreg_x_exp_1_0;
                                        RegTensor<float> _inl6_vreg_x_exp_2_0;
                                        RegTensor<float> _inl6_vreg_x_exp_3_0;
                                        RegTensor<float> _inl6_vreg_x_sum_0_0;
                                        RegTensor<float> _inl6_vreg_x_sum_1_0;
                                        RegTensor<float> _inl6_vreg_x_sum_2_0;
                                        RegTensor<float> _inl6_vreg_x_sum_3_0;
                                        RegTensor<float> _inl6_vreg_x_sum0_0;
                                        RegTensor<float> _inl6_vreg_x_sum1_0;
                                        RegTensor<float> _inl6_vreg_x_max_f32_b_0;
                                        RegTensor<float> _inl6_vreg_l0_0;
                                        RegTensor<half> _inl6_vreg_x_exp_even_f16_0;
                                        RegTensor<half> _inl6_vreg_x_exp_odd_f16_0;
                                        RegTensor<half> _inl6_vreg_x_exp_even_f16_1_0;
                                        RegTensor<half> _inl6_vreg_x_exp_odd_f16_1_0;
                                        RegTensor<half> _inl6_vreg_x_exp_f16_pack_0;
                                        RegTensor<half> _inl6_vreg_x_exp_f16_packa_0;
                                        RegTensor<half> _inl6_vreg_x_exp_f16_1_pack_0;
                                        RegTensor<half> _inl6_vreg_x_exp_f16_1_packa_0;
                                        MaskReg _inl6_preg_108_0;
                                        _inl6_preg_108_0 = pset_b32(PAT_ALL);
                                        uint32_t _vf_mask_scalar_3 = (uint32_t)128;
                                        MaskReg _inl6_preg_136_0;
                                        _inl6_preg_136_0 = plt_b16(_vf_mask_scalar_3, POST_UPDATE);
                                        MaskReg _inl6_preg_compare0_0;
                                        _inl6_preg_compare0_0 = pset_b32(PAT_ALL);
                                        MaskReg _inl6_preg_compare1_0;
                                        _inl6_preg_compare1_0 = pset_b32(PAT_ALL);
                                        MaskReg _inl6_preg_compare2_0;
                                        _inl6_preg_compare2_0 = pset_b32(PAT_ALL);
                                        MaskReg _inl6_preg_compare3_0;
                                        _inl6_preg_compare3_0 = pset_b32(PAT_ALL);
                                        RegTensor<float> _inl6_vreg_min_0;
                                        vbr(_inl6_max0_0, -1000000000.000000);
                                        vbr(_inl6_max1_0, -1000000000.000000);
                                        vbr(_inl6_max2_0, -1000000000.000000);
                                        vbr(_inl6_max3_0, -1000000000.000000);
                                        vbr(_inl6_vreg_min_0, -1000000000.000000);
                                        for (uint16_t _inl6_iter_m_0 = 0; _inl6_iter_m_0 < (uint16_t)(32); _inl6_iter_m_0 += 1) {
                                            vlds(_inl6_src0_0, vf_tile_ptr_21, ((_inl6_iter_m_0 * 64) * 4), NORM);
                                            vlds(_inl6_src1_0, (vf_tile_ptr_21 + (64)), ((_inl6_iter_m_0 * 64) * 4), NORM);
                                            vlds(_inl6_src2_0, (vf_tile_ptr_21 + (128)), ((_inl6_iter_m_0 * 64) * 4), NORM);
                                            vlds(_inl6_src3_0, (vf_tile_ptr_21 + (192)), ((_inl6_iter_m_0 * 64) * 4), NORM);
                                            vmax(_inl6_max0_0, _inl6_max0_0, _inl6_src0_0, _inl6_preg_108_0, MODE_ZEROING);
                                            vmax(_inl6_max1_0, _inl6_max1_0, _inl6_src1_0, _inl6_preg_108_0, MODE_ZEROING);
                                            vmax(_inl6_max2_0, _inl6_max2_0, _inl6_src2_0, _inl6_preg_108_0, MODE_ZEROING);
                                            vmax(_inl6_max3_0, _inl6_max3_0, _inl6_src3_0, _inl6_preg_108_0, MODE_ZEROING);
                                        }
                                        vlds(_inl6_vreg_x_max_f32_b_0, vf_tile_ptr_23, 0, NORM);
                                        vmax(_inl6_max0_0, _inl6_max0_0, _inl6_max2_0, _inl6_preg_108_0, MODE_ZEROING);
                                        vmax(_inl6_max1_0, _inl6_max1_0, _inl6_max3_0, _inl6_preg_108_0, MODE_ZEROING);
                                        vmax(_inl6_max0_0, _inl6_max0_0, _inl6_max1_0, _inl6_preg_108_0, MODE_ZEROING);
                                        vmuls(_inl6_max0_0, _inl6_max0_0, 0.088388, _inl6_preg_108_0);
                                        vmax(_inl6_max0_0, _inl6_max0_0, _inl6_vreg_x_max_f32_b_0, _inl6_preg_108_0, MODE_ZEROING);
                                        vexpdif(_inl6_vreg_x_max_f32_b_0, _inl6_vreg_x_max_f32_b_0, _inl6_max0_0, _inl6_preg_108_0, PART_EVEN);
                                        vsts(_inl6_max0_0, vf_tile_ptr_23, 0, NORM_B32, _inl6_preg_108_0);
                                        vsts(_inl6_vreg_x_max_f32_b_0, vf_tile_ptr_24, 0, NORM_B32, _inl6_preg_108_0);
                                        vdup(_inl6_vreg_x_sum_0_0, 0.000000, _inl6_preg_108_0, MODE_ZEROING);
                                        vdup(_inl6_vreg_x_sum_1_0, 0.000000, _inl6_preg_108_0, MODE_ZEROING);
                                        vdup(_inl6_vreg_x_sum_2_0, 0.000000, _inl6_preg_108_0, MODE_ZEROING);
                                        vdup(_inl6_vreg_x_sum_3_0, 0.000000, _inl6_preg_108_0, MODE_ZEROING);
                                        for (uint16_t _inl6_i0_0 = 0; _inl6_i0_0 < (uint16_t)(32); _inl6_i0_0 += 1) {
                                            vlds(_inl6_vreg_x_f32_0_0, vf_tile_ptr_21, (_inl6_i0_0 * 64), NORM);
                                            vlds(_inl6_vreg_x_f32_1_0, vf_tile_ptr_21, (2048 + (_inl6_i0_0 * 64)), NORM);
                                            vlds(_inl6_vreg_x_f32_2_0, vf_tile_ptr_21, (4096 + (_inl6_i0_0 * 64)), NORM);
                                            vlds(_inl6_vreg_x_f32_3_0, vf_tile_ptr_21, (6144 + (_inl6_i0_0 * 64)), NORM);
                                            vmuls(_inl6_vreg_x_f32_0_0, _inl6_vreg_x_f32_0_0, 0.088388, _inl6_preg_108_0);
                                            vmuls(_inl6_vreg_x_f32_1_0, _inl6_vreg_x_f32_1_0, 0.088388, _inl6_preg_108_0);
                                            vmuls(_inl6_vreg_x_f32_2_0, _inl6_vreg_x_f32_2_0, 0.088388, _inl6_preg_108_0);
                                            vmuls(_inl6_vreg_x_f32_3_0, _inl6_vreg_x_f32_3_0, 0.088388, _inl6_preg_108_0);
                                            vexpdif(_inl6_vreg_x_exp_0_0, _inl6_vreg_x_f32_0_0, _inl6_max0_0, _inl6_preg_108_0, PART_EVEN);
                                            vexpdif(_inl6_vreg_x_exp_1_0, _inl6_vreg_x_f32_1_0, _inl6_max0_0, _inl6_preg_108_0, PART_EVEN);
                                            vexpdif(_inl6_vreg_x_exp_2_0, _inl6_vreg_x_f32_2_0, _inl6_max0_0, _inl6_preg_108_0, PART_EVEN);
                                            vexpdif(_inl6_vreg_x_exp_3_0, _inl6_vreg_x_f32_3_0, _inl6_max0_0, _inl6_preg_108_0, PART_EVEN);
                                            vcvt(_inl6_vreg_x_exp_even_f16_0, _inl6_vreg_x_exp_0_0, _inl6_preg_108_0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                                            vcvt(_inl6_vreg_x_exp_odd_f16_0, _inl6_vreg_x_exp_2_0, _inl6_preg_108_0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                                            vdintlv(_inl6_vreg_x_exp_f16_pack_0, _inl6_vreg_x_exp_f16_packa_0, _inl6_vreg_x_exp_even_f16_0, _inl6_vreg_x_exp_odd_f16_0);
                                            vcvt(_inl6_vreg_x_exp_even_f16_1_0, _inl6_vreg_x_exp_1_0, _inl6_preg_108_0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                                            vcvt(_inl6_vreg_x_exp_odd_f16_1_0, _inl6_vreg_x_exp_3_0, _inl6_preg_108_0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                                            vdintlv(_inl6_vreg_x_exp_f16_1_pack_0, _inl6_vreg_x_exp_f16_1_packa_0, _inl6_vreg_x_exp_even_f16_1_0, _inl6_vreg_x_exp_odd_f16_1_0);
                                            vsstb(_inl6_vreg_x_exp_f16_pack_0, vf_tile_ptr_25, (65 << 16u) | (1 & 0xFFFFU), _inl6_preg_136_0, POST_UPDATE);
                                            vadd(_inl6_vreg_x_sum_0_0, _inl6_vreg_x_exp_0_0, _inl6_vreg_x_sum_0_0, _inl6_preg_108_0, MODE_ZEROING);
                                            vadd(_inl6_vreg_x_sum_2_0, _inl6_vreg_x_exp_2_0, _inl6_vreg_x_sum_2_0, _inl6_preg_108_0, MODE_ZEROING);
                                            vsstb(_inl6_vreg_x_exp_f16_1_pack_0, vf_tile_ptr_26, (65 << 16u) | (1 & 0xFFFFU), _inl6_preg_136_0, POST_UPDATE);
                                            vadd(_inl6_vreg_x_sum_1_0, _inl6_vreg_x_exp_1_0, _inl6_vreg_x_sum_1_0, _inl6_preg_108_0, MODE_ZEROING);
                                            vadd(_inl6_vreg_x_sum_3_0, _inl6_vreg_x_exp_3_0, _inl6_vreg_x_sum_3_0, _inl6_preg_108_0, MODE_ZEROING);
                                        }
                                        vadd(_inl6_vreg_x_sum0_0, _inl6_vreg_x_sum_2_0, _inl6_vreg_x_sum_0_0, _inl6_preg_108_0, MODE_ZEROING);
                                        vadd(_inl6_vreg_x_sum1_0, _inl6_vreg_x_sum_3_0, _inl6_vreg_x_sum_1_0, _inl6_preg_108_0, MODE_ZEROING);
                                        vadd(_inl6_vreg_x_sum0_0, _inl6_vreg_x_sum0_0, _inl6_vreg_x_sum1_0, _inl6_preg_108_0, MODE_ZEROING);
                                        vlds(_inl6_vreg_l0_0, vf_tile_ptr_27, 0, NORM);
                                        vmul(_inl6_vreg_l0_0, _inl6_vreg_x_max_f32_b_0, _inl6_vreg_l0_0, _inl6_preg_108_0, MODE_ZEROING);
                                        vadd(_inl6_vreg_l0_0, _inl6_vreg_l0_0, _inl6_vreg_x_sum0_0, _inl6_preg_108_0, MODE_ZEROING);
                                        vsts(_inl6_vreg_l0_0, vf_tile_ptr_27, 0, NORM_B32, _inl6_preg_108_0);
                                    }
                                    rls_buf(PIPE_V, _tg_p_f16_db_mutex_ids_0[_inl2__bufidx_1_0], 0);
                                }
                            }
                        } else {
                            if ((p_ctx_0.ki == 0)) {
                                get_buf(PIPE_V, _tg_p_f16_db_mutex_ids_0[_inl2__bufidx_1_0], 0);
                                __ubuf__ float *vf_tile_ptr_28 = (__ubuf__ float *)_tg_qk_vec_db_tiles_0[_inl2__bufidx_0_0].data();
                                __ubuf__ half *vf_tile_ptr_29 = (__ubuf__ half *)_tg_p_f16_db_tiles_0[_inl2__bufidx_1_0].data();
                                __ubuf__ float *vf_tile_ptr_30 = (__ubuf__ float *)global_max_rm_buf_0[p_ctx_0.q_count_mod3].data();
                                __ubuf__ half *vf_tile_ptr_31 = (__ubuf__ half *)_tg_p_f16_db_tiles_0[_inl2__bufidx_1_0].data();
                                __ubuf__ half *vf_tile_ptr_32 = (vf_tile_ptr_29 + (512));
                                __ubuf__ float *vf_tile_ptr_33 = (__ubuf__ float *)global_sum_rm_buf_0[p_ctx_0.q_count_mod3].data();
                                __VEC_SCOPE__ {
                                    RegTensor<float> _inl7_src0_0;
                                    RegTensor<float> _inl7_src1_0;
                                    RegTensor<float> _inl7_src2_0;
                                    RegTensor<float> _inl7_src3_0;
                                    RegTensor<float> _inl7_max0_0;
                                    RegTensor<float> _inl7_max1_0;
                                    RegTensor<float> _inl7_max2_0;
                                    RegTensor<float> _inl7_max3_0;
                                    RegTensor<float> _inl7_vreg_x_f32_0_0;
                                    RegTensor<float> _inl7_vreg_x_f32_1_0;
                                    RegTensor<float> _inl7_vreg_x_f32_2_0;
                                    RegTensor<float> _inl7_vreg_x_f32_3_0;
                                    RegTensor<float> _inl7_vreg_x_exp_0_0;
                                    RegTensor<float> _inl7_vreg_x_exp_1_0;
                                    RegTensor<float> _inl7_vreg_x_exp_2_0;
                                    RegTensor<float> _inl7_vreg_x_exp_3_0;
                                    RegTensor<float> _inl7_vreg_x_sum_0_0;
                                    RegTensor<float> _inl7_vreg_x_sum_1_0;
                                    RegTensor<float> _inl7_vreg_x_sum_2_0;
                                    RegTensor<float> _inl7_vreg_x_sum_3_0;
                                    RegTensor<float> _inl7_vreg_x_sum0_0;
                                    RegTensor<float> _inl7_vreg_x_sum1_0;
                                    RegTensor<float> _inl7_vreg_x_max_f32_b_0;
                                    RegTensor<float> _inl7_vreg_l0_0;
                                    RegTensor<half> _inl7_vreg_x_exp_even_f16_0;
                                    RegTensor<half> _inl7_vreg_x_exp_odd_f16_0;
                                    RegTensor<half> _inl7_vreg_x_exp_even_f16_1_0;
                                    RegTensor<half> _inl7_vreg_x_exp_odd_f16_1_0;
                                    RegTensor<half> _inl7_vreg_x_exp_f16_pack_0;
                                    RegTensor<half> _inl7_vreg_x_exp_f16_packa_0;
                                    RegTensor<half> _inl7_vreg_x_exp_f16_1_pack_0;
                                    RegTensor<half> _inl7_vreg_x_exp_f16_1_packa_0;
                                    MaskReg _inl7_preg_108_0;
                                    _inl7_preg_108_0 = pset_b32(PAT_ALL);
                                    uint32_t _vf_mask_scalar_4 = (uint32_t)128;
                                    MaskReg _inl7_preg_136_0;
                                    _inl7_preg_136_0 = plt_b16(_vf_mask_scalar_4, POST_UPDATE);
                                    MaskReg _inl7_preg_compare0_0;
                                    _inl7_preg_compare0_0 = pset_b32(PAT_ALL);
                                    MaskReg _inl7_preg_compare1_0;
                                    _inl7_preg_compare1_0 = pset_b32(PAT_ALL);
                                    MaskReg _inl7_preg_compare2_0;
                                    _inl7_preg_compare2_0 = pset_b32(PAT_ALL);
                                    MaskReg _inl7_preg_compare3_0;
                                    _inl7_preg_compare3_0 = pset_b32(PAT_ALL);
                                    RegTensor<float> _inl7_vreg_min_0;
                                    vbr(_inl7_max0_0, -1000000000.000000);
                                    vbr(_inl7_max1_0, -1000000000.000000);
                                    vbr(_inl7_max2_0, -1000000000.000000);
                                    vbr(_inl7_max3_0, -1000000000.000000);
                                    vbr(_inl7_vreg_min_0, -1000000000.000000);
                                    for (uint16_t _inl7_iter_m_0 = 0; _inl7_iter_m_0 < (uint16_t)(32); _inl7_iter_m_0 += 1) {
                                        vlds(_inl7_src0_0, vf_tile_ptr_28, ((_inl7_iter_m_0 * 64) * 4), NORM);
                                        vlds(_inl7_src1_0, (vf_tile_ptr_28 + (64)), ((_inl7_iter_m_0 * 64) * 4), NORM);
                                        vlds(_inl7_src2_0, (vf_tile_ptr_28 + (128)), ((_inl7_iter_m_0 * 64) * 4), NORM);
                                        vlds(_inl7_src3_0, (vf_tile_ptr_28 + (192)), ((_inl7_iter_m_0 * 64) * 4), NORM);
                                        vmax(_inl7_max0_0, _inl7_max0_0, _inl7_src0_0, _inl7_preg_108_0, MODE_ZEROING);
                                        vmax(_inl7_max1_0, _inl7_max1_0, _inl7_src1_0, _inl7_preg_108_0, MODE_ZEROING);
                                        vmax(_inl7_max2_0, _inl7_max2_0, _inl7_src2_0, _inl7_preg_108_0, MODE_ZEROING);
                                        vmax(_inl7_max3_0, _inl7_max3_0, _inl7_src3_0, _inl7_preg_108_0, MODE_ZEROING);
                                    }
                                    vmax(_inl7_max0_0, _inl7_max0_0, _inl7_max2_0, _inl7_preg_108_0, MODE_ZEROING);
                                    vmax(_inl7_max1_0, _inl7_max1_0, _inl7_max3_0, _inl7_preg_108_0, MODE_ZEROING);
                                    vmax(_inl7_max0_0, _inl7_max0_0, _inl7_max1_0, _inl7_preg_108_0, MODE_ZEROING);
                                    vmuls(_inl7_max0_0, _inl7_max0_0, 0.088388, _inl7_preg_108_0);
                                    vsts(_inl7_max0_0, vf_tile_ptr_30, 0, NORM_B32, _inl7_preg_108_0);
                                    vdup(_inl7_vreg_x_sum_0_0, 0.000000, _inl7_preg_108_0, MODE_ZEROING);
                                    vdup(_inl7_vreg_x_sum_1_0, 0.000000, _inl7_preg_108_0, MODE_ZEROING);
                                    vdup(_inl7_vreg_x_sum_2_0, 0.000000, _inl7_preg_108_0, MODE_ZEROING);
                                    vdup(_inl7_vreg_x_sum_3_0, 0.000000, _inl7_preg_108_0, MODE_ZEROING);
                                    for (uint16_t _inl7_i0_0 = 0; _inl7_i0_0 < (uint16_t)(32); _inl7_i0_0 += 1) {
                                        vlds(_inl7_vreg_x_f32_0_0, vf_tile_ptr_28, (_inl7_i0_0 * 64), NORM);
                                        vlds(_inl7_vreg_x_f32_1_0, vf_tile_ptr_28, (2048 + (_inl7_i0_0 * 64)), NORM);
                                        vlds(_inl7_vreg_x_f32_2_0, vf_tile_ptr_28, (4096 + (_inl7_i0_0 * 64)), NORM);
                                        vlds(_inl7_vreg_x_f32_3_0, vf_tile_ptr_28, (6144 + (_inl7_i0_0 * 64)), NORM);
                                        vmuls(_inl7_vreg_x_f32_0_0, _inl7_vreg_x_f32_0_0, 0.088388, _inl7_preg_108_0);
                                        vmuls(_inl7_vreg_x_f32_1_0, _inl7_vreg_x_f32_1_0, 0.088388, _inl7_preg_108_0);
                                        vmuls(_inl7_vreg_x_f32_2_0, _inl7_vreg_x_f32_2_0, 0.088388, _inl7_preg_108_0);
                                        vmuls(_inl7_vreg_x_f32_3_0, _inl7_vreg_x_f32_3_0, 0.088388, _inl7_preg_108_0);
                                        vexpdif(_inl7_vreg_x_exp_0_0, _inl7_vreg_x_f32_0_0, _inl7_max0_0, _inl7_preg_108_0, PART_EVEN);
                                        vexpdif(_inl7_vreg_x_exp_1_0, _inl7_vreg_x_f32_1_0, _inl7_max0_0, _inl7_preg_108_0, PART_EVEN);
                                        vexpdif(_inl7_vreg_x_exp_2_0, _inl7_vreg_x_f32_2_0, _inl7_max0_0, _inl7_preg_108_0, PART_EVEN);
                                        vexpdif(_inl7_vreg_x_exp_3_0, _inl7_vreg_x_f32_3_0, _inl7_max0_0, _inl7_preg_108_0, PART_EVEN);
                                        vcvt(_inl7_vreg_x_exp_even_f16_0, _inl7_vreg_x_exp_0_0, _inl7_preg_108_0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                                        vcvt(_inl7_vreg_x_exp_odd_f16_0, _inl7_vreg_x_exp_2_0, _inl7_preg_108_0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                                        vdintlv(_inl7_vreg_x_exp_f16_pack_0, _inl7_vreg_x_exp_f16_packa_0, _inl7_vreg_x_exp_even_f16_0, _inl7_vreg_x_exp_odd_f16_0);
                                        vcvt(_inl7_vreg_x_exp_even_f16_1_0, _inl7_vreg_x_exp_1_0, _inl7_preg_108_0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                                        vcvt(_inl7_vreg_x_exp_odd_f16_1_0, _inl7_vreg_x_exp_3_0, _inl7_preg_108_0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                                        vdintlv(_inl7_vreg_x_exp_f16_1_pack_0, _inl7_vreg_x_exp_f16_1_packa_0, _inl7_vreg_x_exp_even_f16_1_0, _inl7_vreg_x_exp_odd_f16_1_0);
                                        vsstb(_inl7_vreg_x_exp_f16_pack_0, vf_tile_ptr_31, (65 << 16u) | (1 & 0xFFFFU), _inl7_preg_136_0, POST_UPDATE);
                                        vadd(_inl7_vreg_x_sum_0_0, _inl7_vreg_x_exp_0_0, _inl7_vreg_x_sum_0_0, _inl7_preg_108_0, MODE_ZEROING);
                                        vadd(_inl7_vreg_x_sum_2_0, _inl7_vreg_x_exp_2_0, _inl7_vreg_x_sum_2_0, _inl7_preg_108_0, MODE_ZEROING);
                                        vsstb(_inl7_vreg_x_exp_f16_1_pack_0, vf_tile_ptr_32, (65 << 16u) | (1 & 0xFFFFU), _inl7_preg_136_0, POST_UPDATE);
                                        vadd(_inl7_vreg_x_sum_1_0, _inl7_vreg_x_exp_1_0, _inl7_vreg_x_sum_1_0, _inl7_preg_108_0, MODE_ZEROING);
                                        vadd(_inl7_vreg_x_sum_3_0, _inl7_vreg_x_exp_3_0, _inl7_vreg_x_sum_3_0, _inl7_preg_108_0, MODE_ZEROING);
                                    }
                                    vadd(_inl7_vreg_x_sum0_0, _inl7_vreg_x_sum_2_0, _inl7_vreg_x_sum_0_0, _inl7_preg_108_0, MODE_ZEROING);
                                    vadd(_inl7_vreg_x_sum1_0, _inl7_vreg_x_sum_3_0, _inl7_vreg_x_sum_1_0, _inl7_preg_108_0, MODE_ZEROING);
                                    vadd(_inl7_vreg_x_sum0_0, _inl7_vreg_x_sum0_0, _inl7_vreg_x_sum1_0, _inl7_preg_108_0, MODE_ZEROING);
                                    vsts(_inl7_vreg_x_sum0_0, vf_tile_ptr_33, 0, NORM_B32, _inl7_preg_108_0);
                                }
                                rls_buf(PIPE_V, _tg_p_f16_db_mutex_ids_0[_inl2__bufidx_1_0], 0);
                            } else {
                                get_buf(PIPE_V, _tg_p_f16_db_mutex_ids_0[_inl2__bufidx_1_0], 0);
                                __ubuf__ float *vf_tile_ptr_34 = (__ubuf__ float *)_tg_qk_vec_db_tiles_0[_inl2__bufidx_0_0].data();
                                __ubuf__ half *vf_tile_ptr_35 = (__ubuf__ half *)_tg_p_f16_db_tiles_0[_inl2__bufidx_1_0].data();
                                __ubuf__ float *vf_tile_ptr_36 = (__ubuf__ float *)global_max_rm_buf_0[p_ctx_0.q_count_mod3].data();
                                __ubuf__ float *vf_tile_ptr_37 = (__ubuf__ float *)exp_corr_rm_fifo_0[p_ctx_0.task_id_mod3].data();
                                __ubuf__ half *vf_tile_ptr_38 = (__ubuf__ half *)_tg_p_f16_db_tiles_0[_inl2__bufidx_1_0].data();
                                __ubuf__ half *vf_tile_ptr_39 = (vf_tile_ptr_35 + (512));
                                __ubuf__ float *vf_tile_ptr_40 = (__ubuf__ float *)global_sum_rm_buf_0[p_ctx_0.q_count_mod3].data();
                                __VEC_SCOPE__ {
                                    RegTensor<float> _inl8_src0_0;
                                    RegTensor<float> _inl8_src1_0;
                                    RegTensor<float> _inl8_src2_0;
                                    RegTensor<float> _inl8_src3_0;
                                    RegTensor<float> _inl8_max0_0;
                                    RegTensor<float> _inl8_max1_0;
                                    RegTensor<float> _inl8_max2_0;
                                    RegTensor<float> _inl8_max3_0;
                                    RegTensor<float> _inl8_vreg_x_f32_0_0;
                                    RegTensor<float> _inl8_vreg_x_f32_1_0;
                                    RegTensor<float> _inl8_vreg_x_f32_2_0;
                                    RegTensor<float> _inl8_vreg_x_f32_3_0;
                                    RegTensor<float> _inl8_vreg_x_exp_0_0;
                                    RegTensor<float> _inl8_vreg_x_exp_1_0;
                                    RegTensor<float> _inl8_vreg_x_exp_2_0;
                                    RegTensor<float> _inl8_vreg_x_exp_3_0;
                                    RegTensor<float> _inl8_vreg_x_sum_0_0;
                                    RegTensor<float> _inl8_vreg_x_sum_1_0;
                                    RegTensor<float> _inl8_vreg_x_sum_2_0;
                                    RegTensor<float> _inl8_vreg_x_sum_3_0;
                                    RegTensor<float> _inl8_vreg_x_sum0_0;
                                    RegTensor<float> _inl8_vreg_x_sum1_0;
                                    RegTensor<float> _inl8_vreg_x_max_f32_b_0;
                                    RegTensor<float> _inl8_vreg_l0_0;
                                    RegTensor<half> _inl8_vreg_x_exp_even_f16_0;
                                    RegTensor<half> _inl8_vreg_x_exp_odd_f16_0;
                                    RegTensor<half> _inl8_vreg_x_exp_even_f16_1_0;
                                    RegTensor<half> _inl8_vreg_x_exp_odd_f16_1_0;
                                    RegTensor<half> _inl8_vreg_x_exp_f16_pack_0;
                                    RegTensor<half> _inl8_vreg_x_exp_f16_packa_0;
                                    RegTensor<half> _inl8_vreg_x_exp_f16_1_pack_0;
                                    RegTensor<half> _inl8_vreg_x_exp_f16_1_packa_0;
                                    MaskReg _inl8_preg_108_0;
                                    _inl8_preg_108_0 = pset_b32(PAT_ALL);
                                    uint32_t _vf_mask_scalar_5 = (uint32_t)128;
                                    MaskReg _inl8_preg_136_0;
                                    _inl8_preg_136_0 = plt_b16(_vf_mask_scalar_5, POST_UPDATE);
                                    MaskReg _inl8_preg_compare0_0;
                                    _inl8_preg_compare0_0 = pset_b32(PAT_ALL);
                                    MaskReg _inl8_preg_compare1_0;
                                    _inl8_preg_compare1_0 = pset_b32(PAT_ALL);
                                    MaskReg _inl8_preg_compare2_0;
                                    _inl8_preg_compare2_0 = pset_b32(PAT_ALL);
                                    MaskReg _inl8_preg_compare3_0;
                                    _inl8_preg_compare3_0 = pset_b32(PAT_ALL);
                                    RegTensor<float> _inl8_vreg_min_0;
                                    vbr(_inl8_max0_0, -1000000000.000000);
                                    vbr(_inl8_max1_0, -1000000000.000000);
                                    vbr(_inl8_max2_0, -1000000000.000000);
                                    vbr(_inl8_max3_0, -1000000000.000000);
                                    vbr(_inl8_vreg_min_0, -1000000000.000000);
                                    for (uint16_t _inl8_iter_m_0 = 0; _inl8_iter_m_0 < (uint16_t)(32); _inl8_iter_m_0 += 1) {
                                        vlds(_inl8_src0_0, vf_tile_ptr_34, ((_inl8_iter_m_0 * 64) * 4), NORM);
                                        vlds(_inl8_src1_0, (vf_tile_ptr_34 + (64)), ((_inl8_iter_m_0 * 64) * 4), NORM);
                                        vlds(_inl8_src2_0, (vf_tile_ptr_34 + (128)), ((_inl8_iter_m_0 * 64) * 4), NORM);
                                        vlds(_inl8_src3_0, (vf_tile_ptr_34 + (192)), ((_inl8_iter_m_0 * 64) * 4), NORM);
                                        vmax(_inl8_max0_0, _inl8_max0_0, _inl8_src0_0, _inl8_preg_108_0, MODE_ZEROING);
                                        vmax(_inl8_max1_0, _inl8_max1_0, _inl8_src1_0, _inl8_preg_108_0, MODE_ZEROING);
                                        vmax(_inl8_max2_0, _inl8_max2_0, _inl8_src2_0, _inl8_preg_108_0, MODE_ZEROING);
                                        vmax(_inl8_max3_0, _inl8_max3_0, _inl8_src3_0, _inl8_preg_108_0, MODE_ZEROING);
                                    }
                                    vlds(_inl8_vreg_x_max_f32_b_0, vf_tile_ptr_36, 0, NORM);
                                    vmax(_inl8_max0_0, _inl8_max0_0, _inl8_max2_0, _inl8_preg_108_0, MODE_ZEROING);
                                    vmax(_inl8_max1_0, _inl8_max1_0, _inl8_max3_0, _inl8_preg_108_0, MODE_ZEROING);
                                    vmax(_inl8_max0_0, _inl8_max0_0, _inl8_max1_0, _inl8_preg_108_0, MODE_ZEROING);
                                    vmuls(_inl8_max0_0, _inl8_max0_0, 0.088388, _inl8_preg_108_0);
                                    vmax(_inl8_max0_0, _inl8_max0_0, _inl8_vreg_x_max_f32_b_0, _inl8_preg_108_0, MODE_ZEROING);
                                    vexpdif(_inl8_vreg_x_max_f32_b_0, _inl8_vreg_x_max_f32_b_0, _inl8_max0_0, _inl8_preg_108_0, PART_EVEN);
                                    vsts(_inl8_max0_0, vf_tile_ptr_36, 0, NORM_B32, _inl8_preg_108_0);
                                    vsts(_inl8_vreg_x_max_f32_b_0, vf_tile_ptr_37, 0, NORM_B32, _inl8_preg_108_0);
                                    vdup(_inl8_vreg_x_sum_0_0, 0.000000, _inl8_preg_108_0, MODE_ZEROING);
                                    vdup(_inl8_vreg_x_sum_1_0, 0.000000, _inl8_preg_108_0, MODE_ZEROING);
                                    vdup(_inl8_vreg_x_sum_2_0, 0.000000, _inl8_preg_108_0, MODE_ZEROING);
                                    vdup(_inl8_vreg_x_sum_3_0, 0.000000, _inl8_preg_108_0, MODE_ZEROING);
                                    for (uint16_t _inl8_i0_0 = 0; _inl8_i0_0 < (uint16_t)(32); _inl8_i0_0 += 1) {
                                        vlds(_inl8_vreg_x_f32_0_0, vf_tile_ptr_34, (_inl8_i0_0 * 64), NORM);
                                        vlds(_inl8_vreg_x_f32_1_0, vf_tile_ptr_34, (2048 + (_inl8_i0_0 * 64)), NORM);
                                        vlds(_inl8_vreg_x_f32_2_0, vf_tile_ptr_34, (4096 + (_inl8_i0_0 * 64)), NORM);
                                        vlds(_inl8_vreg_x_f32_3_0, vf_tile_ptr_34, (6144 + (_inl8_i0_0 * 64)), NORM);
                                        vmuls(_inl8_vreg_x_f32_0_0, _inl8_vreg_x_f32_0_0, 0.088388, _inl8_preg_108_0);
                                        vmuls(_inl8_vreg_x_f32_1_0, _inl8_vreg_x_f32_1_0, 0.088388, _inl8_preg_108_0);
                                        vmuls(_inl8_vreg_x_f32_2_0, _inl8_vreg_x_f32_2_0, 0.088388, _inl8_preg_108_0);
                                        vmuls(_inl8_vreg_x_f32_3_0, _inl8_vreg_x_f32_3_0, 0.088388, _inl8_preg_108_0);
                                        vexpdif(_inl8_vreg_x_exp_0_0, _inl8_vreg_x_f32_0_0, _inl8_max0_0, _inl8_preg_108_0, PART_EVEN);
                                        vexpdif(_inl8_vreg_x_exp_1_0, _inl8_vreg_x_f32_1_0, _inl8_max0_0, _inl8_preg_108_0, PART_EVEN);
                                        vexpdif(_inl8_vreg_x_exp_2_0, _inl8_vreg_x_f32_2_0, _inl8_max0_0, _inl8_preg_108_0, PART_EVEN);
                                        vexpdif(_inl8_vreg_x_exp_3_0, _inl8_vreg_x_f32_3_0, _inl8_max0_0, _inl8_preg_108_0, PART_EVEN);
                                        vcvt(_inl8_vreg_x_exp_even_f16_0, _inl8_vreg_x_exp_0_0, _inl8_preg_108_0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                                        vcvt(_inl8_vreg_x_exp_odd_f16_0, _inl8_vreg_x_exp_2_0, _inl8_preg_108_0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                                        vdintlv(_inl8_vreg_x_exp_f16_pack_0, _inl8_vreg_x_exp_f16_packa_0, _inl8_vreg_x_exp_even_f16_0, _inl8_vreg_x_exp_odd_f16_0);
                                        vcvt(_inl8_vreg_x_exp_even_f16_1_0, _inl8_vreg_x_exp_1_0, _inl8_preg_108_0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                                        vcvt(_inl8_vreg_x_exp_odd_f16_1_0, _inl8_vreg_x_exp_3_0, _inl8_preg_108_0, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
                                        vdintlv(_inl8_vreg_x_exp_f16_1_pack_0, _inl8_vreg_x_exp_f16_1_packa_0, _inl8_vreg_x_exp_even_f16_1_0, _inl8_vreg_x_exp_odd_f16_1_0);
                                        vsstb(_inl8_vreg_x_exp_f16_pack_0, vf_tile_ptr_38, (65 << 16u) | (1 & 0xFFFFU), _inl8_preg_136_0, POST_UPDATE);
                                        vadd(_inl8_vreg_x_sum_0_0, _inl8_vreg_x_exp_0_0, _inl8_vreg_x_sum_0_0, _inl8_preg_108_0, MODE_ZEROING);
                                        vadd(_inl8_vreg_x_sum_2_0, _inl8_vreg_x_exp_2_0, _inl8_vreg_x_sum_2_0, _inl8_preg_108_0, MODE_ZEROING);
                                        vsstb(_inl8_vreg_x_exp_f16_1_pack_0, vf_tile_ptr_39, (65 << 16u) | (1 & 0xFFFFU), _inl8_preg_136_0, POST_UPDATE);
                                        vadd(_inl8_vreg_x_sum_1_0, _inl8_vreg_x_exp_1_0, _inl8_vreg_x_sum_1_0, _inl8_preg_108_0, MODE_ZEROING);
                                        vadd(_inl8_vreg_x_sum_3_0, _inl8_vreg_x_exp_3_0, _inl8_vreg_x_sum_3_0, _inl8_preg_108_0, MODE_ZEROING);
                                    }
                                    vadd(_inl8_vreg_x_sum0_0, _inl8_vreg_x_sum_2_0, _inl8_vreg_x_sum_0_0, _inl8_preg_108_0, MODE_ZEROING);
                                    vadd(_inl8_vreg_x_sum1_0, _inl8_vreg_x_sum_3_0, _inl8_vreg_x_sum_1_0, _inl8_preg_108_0, MODE_ZEROING);
                                    vadd(_inl8_vreg_x_sum0_0, _inl8_vreg_x_sum0_0, _inl8_vreg_x_sum1_0, _inl8_preg_108_0, MODE_ZEROING);
                                    vlds(_inl8_vreg_l0_0, vf_tile_ptr_40, 0, NORM);
                                    vmul(_inl8_vreg_l0_0, _inl8_vreg_x_max_f32_b_0, _inl8_vreg_l0_0, _inl8_preg_108_0, MODE_ZEROING);
                                    vadd(_inl8_vreg_l0_0, _inl8_vreg_l0_0, _inl8_vreg_x_sum0_0, _inl8_preg_108_0, MODE_ZEROING);
                                    vsts(_inl8_vreg_l0_0, vf_tile_ptr_40, 0, NORM_B32, _inl8_preg_108_0);
                                }
                                rls_buf(PIPE_V, _tg_p_f16_db_mutex_ids_0[_inl2__bufidx_1_0], 0);
                            }
                        }
                        const int64_t _dyn_arr_9[] = {2, 3};
                        set_intra_block(PIPE_V, _dyn_arr_9[p_ctx_0.task_id_mod2]);
                        get_buf(PIPE_MTE3, _tg_p_mat_db_mutex_ids_0[_inl2__bufidx_4_0], 0);
                        get_buf(PIPE_MTE3, _tg_p_f16_main_db_mutex_ids_0[_inl2__bufidx_2_0], 0);
                        TINSERT(_tg_p_mat_db_tiles_0[_inl2__bufidx_4_0], _tg_p_f16_main_db_tiles_0[_inl2__bufidx_2_0], 0, (64 * sub_id_0));
                        rls_buf(PIPE_MTE3, _tg_p_mat_db_mutex_ids_0[_inl2__bufidx_4_0], 0);
                        rls_buf(PIPE_MTE3, _tg_p_f16_main_db_mutex_ids_0[_inl2__bufidx_2_0], 0);
                        get_buf(PIPE_MTE3, _tg_p_mat_db_mutex_ids_0[_inl2__bufidx_4_0], 0);
                        get_buf(PIPE_MTE3, _tg_p_f16_back_db_mutex_ids_0[_inl2__bufidx_3_0], 0);
                        TINSERT(_tg_p_mat_db_tiles_0[_inl2__bufidx_4_0], _tg_p_f16_back_db_tiles_0[_inl2__bufidx_3_0], 64, (64 * sub_id_0));
                        rls_buf(PIPE_MTE3, _tg_p_mat_db_mutex_ids_0[_inl2__bufidx_4_0], 0);
                        rls_buf(PIPE_MTE3, _tg_p_f16_back_db_mutex_ids_0[_inl2__bufidx_3_0], 0);
                        const int64_t _dyn_arr_10[] = {4, 5, 6};
                        set_intra_block(PIPE_MTE3, _dyn_arr_10[p_ctx_0.task_id_mod3]);
                    }
                }
                if ((task_id_8 > 2)) {
                    auto& g_ctx_0 = ctx_arr_1[((task_id_8 + 1) % 4)];
                    const int64_t _inl9_PV_READY_FORWARD_IDS_0[] = {7, 8};
                    const int64_t _inl9_PV_READY_BARKWARD_IDS_0[] = {9, 10};
                    _tg_pv_vec_db_cursor_0.cursor = (_tg_pv_vec_db_cursor_0.cursor + 1);
                    auto _inl9__bufidx_0_0 = ((_tg_pv_vec_db_cursor_0.cursor + 1) % 2);
                    const int64_t _dyn_arr_11[] = {7, 8};
                    wait_intra_block(PIPE_V, _dyn_arr_11[g_ctx_0.task_id_mod2]);
                    if ((g_ctx_0.ki == 0)) {
                        TMOV<std::remove_reference_t<decltype(_tg_running_o_g_tiles_0_0)>, std::remove_reference_t<decltype(_tg_pv_vec_db_tiles_0[_inl9__bufidx_0_0])>>(_tg_running_o_g_tiles_0_0, _tg_pv_vec_db_tiles_0[_inl9__bufidx_0_0]);
                    } else {
                        if ((g_ctx_0.ki < (g_ctx_0.skv_tiles - 1))) {
                            __ubuf__ float *vf_tile_ptr_41 = (__ubuf__ float *)exp_corr_rm_fifo_0[g_ctx_0.task_id_mod3].data();
                            __ubuf__ float *vf_tile_ptr_42 = (__ubuf__ float *)_tg_running_o_g_tiles_0_0.data();
                            __ubuf__ float *vf_tile_ptr_43 = (__ubuf__ float *)_tg_pv_vec_db_tiles_0[_inl9__bufidx_0_0].data();
                            __VEC_SCOPE__ {
                                RegTensor<float> _inl10_vreg_exp_max_0;
                                RegTensor<float> _inl10_vreg_input_pre_0;
                                RegTensor<float> _inl10_vreg_input_cur_0;
                                RegTensor<float> _inl10_vreg_mul_0;
                                RegTensor<float> _inl10_vreg_add_0;
                                MaskReg _inl10_preg_all_0;
                                _inl10_preg_all_0 = pset_b32(PAT_ALL);
                                for (uint16_t _inl10_i_0 = 0; _inl10_i_0 < (uint16_t)(64); _inl10_i_0 += 1) {
                                    vlds(_inl10_vreg_exp_max_0, vf_tile_ptr_41, _inl10_i_0, BRC_B32);
                                    for (uint16_t _inl10_j_0 = 0; _inl10_j_0 < (uint16_t)(2); _inl10_j_0 += 1) {
                                        vlds(_inl10_vreg_input_pre_0, vf_tile_ptr_42, ((_inl10_i_0 * 128) + (_inl10_j_0 * 64)), NORM);
                                        vlds(_inl10_vreg_input_cur_0, vf_tile_ptr_43, ((_inl10_i_0 * 128) + (_inl10_j_0 * 64)), NORM);
                                        vmul(_inl10_vreg_mul_0, _inl10_vreg_exp_max_0, _inl10_vreg_input_pre_0, _inl10_preg_all_0, MODE_ZEROING);
                                        vadd(_inl10_vreg_add_0, _inl10_vreg_mul_0, _inl10_vreg_input_cur_0, _inl10_preg_all_0, MODE_ZEROING);
                                        vsts(_inl10_vreg_add_0, (vf_tile_ptr_42 + (((_inl10_i_0 * 128) + (_inl10_j_0 * 64)))), 0, NORM_B32, _inl10_preg_all_0);
                                    }
                                }
                            }
                        } else {
                            __ubuf__ float *vf_tile_ptr_44 = (__ubuf__ float *)exp_corr_rm_fifo_0[g_ctx_0.task_id_mod3].data();
                            __ubuf__ float *vf_tile_ptr_45 = (__ubuf__ float *)global_sum_rm_buf_0[g_ctx_0.q_count_mod3].data();
                            __ubuf__ float *vf_tile_ptr_46 = (__ubuf__ float *)_tg_running_o_g_tiles_0_0.data();
                            __ubuf__ float *vf_tile_ptr_47 = (__ubuf__ float *)_tg_pv_vec_db_tiles_0[_inl9__bufidx_0_0].data();
                            __VEC_SCOPE__ {
                                RegTensor<float> _inl11_vreg_exp_max_0;
                                RegTensor<float> _inl11_vreg_exp_sum_0;
                                RegTensor<float> _inl11_vreg_input_pre_0;
                                RegTensor<float> _inl11_vreg_input_cur_0;
                                RegTensor<float> _inl11_vreg_mul_0;
                                RegTensor<float> _inl11_vreg_add_0;
                                RegTensor<float> _inl11_vreg_div_0;
                                MaskReg _inl11_preg_all_0;
                                _inl11_preg_all_0 = pset_b32(PAT_ALL);
                                for (uint16_t _inl11_i_0 = 0; _inl11_i_0 < (uint16_t)(64); _inl11_i_0 += 1) {
                                    vlds(_inl11_vreg_exp_max_0, vf_tile_ptr_44, _inl11_i_0, BRC_B32);
                                    vlds(_inl11_vreg_exp_sum_0, vf_tile_ptr_45, _inl11_i_0, BRC_B32);
                                    for (uint16_t _inl11_j_0 = 0; _inl11_j_0 < (uint16_t)(2); _inl11_j_0 += 1) {
                                        vlds(_inl11_vreg_input_pre_0, vf_tile_ptr_46, ((_inl11_i_0 * 128) + (_inl11_j_0 * 64)), NORM);
                                        vlds(_inl11_vreg_input_cur_0, vf_tile_ptr_47, ((_inl11_i_0 * 128) + (_inl11_j_0 * 64)), NORM);
                                        vmul(_inl11_vreg_mul_0, _inl11_vreg_exp_max_0, _inl11_vreg_input_pre_0, _inl11_preg_all_0, MODE_ZEROING);
                                        vadd(_inl11_vreg_add_0, _inl11_vreg_mul_0, _inl11_vreg_input_cur_0, _inl11_preg_all_0, MODE_ZEROING);
                                        vdiv(_inl11_vreg_div_0, _inl11_vreg_add_0, _inl11_vreg_exp_sum_0, _inl11_preg_all_0, MODE_ZEROING);
                                        vsts(_inl11_vreg_div_0, (vf_tile_ptr_46 + (((_inl11_i_0 * 128) + (_inl11_j_0 * 64)))), 0, NORM_B32, _inl11_preg_all_0);
                                    }
                                }
                            }
                            get_buf(PIPE_V, 11, 0);
                            TCVT(_tg_o_f16_g_tiles_0_0, _tg_running_o_g_tiles_0_0, RoundMode::CAST_ROUND);
                            rls_buf(PIPE_V, 11, 0);
                            get_buf(PIPE_MTE3, 11, 0);
                            tensor_o_0.SetShape<pto::GlobalTensorDim::DIM_3, pto::GlobalTensorDim::DIM_4>(static_cast<int64_t>(64), static_cast<int64_t>(128));
                            TASSIGN(tensor_o_0, (__gm__ half*)(attention_out_0) + (g_ctx_0.b_idx * tiling_0.sq * tiling_0.n * tiling_0.d + (((g_ctx_0.qi * 2) + sub_id_0) * 64) * tiling_0.n * tiling_0.d + g_ctx_0.n_idx * tiling_0.d + 0));
                            TSTORE(tensor_o_0, _tg_o_f16_g_tiles_0_0);
                            rls_buf(PIPE_MTE3, 11, 0);
                        }
                    }
                    const int64_t _dyn_arr_12[] = {9, 10};
                    set_intra_block(PIPE_V, _dyn_arr_12[g_ctx_0.task_id_mod2]);
                    if ((g_ctx_0.ki == (g_ctx_0.skv_tiles - 1))) {
                        if ((g_ctx_0.ki == 0)) {
                            __ubuf__ float *vf_tile_ptr_48 = (__ubuf__ float *)global_sum_rm_buf_0[g_ctx_0.q_count_mod3].data();
                            __ubuf__ float *vf_tile_ptr_49 = (__ubuf__ float *)_tg_running_o_g_tiles_0_0.data();
                            __VEC_SCOPE__ {
                                RegTensor<float> _inl12_vreg_input_cur_0;
                                RegTensor<float> _inl12_vreg_div_0;
                                RegTensor<float> _inl12_vreg_exp_sum_0;
                                MaskReg _inl12_preg_all_0;
                                _inl12_preg_all_0 = pset_b32(PAT_ALL);
                                for (uint16_t _inl12_i_0 = 0; _inl12_i_0 < (uint16_t)(64); _inl12_i_0 += 1) {
                                    vlds(_inl12_vreg_exp_sum_0, vf_tile_ptr_48, _inl12_i_0, BRC_B32);
                                    for (uint16_t _inl12_j_0 = 0; _inl12_j_0 < (uint16_t)(2); _inl12_j_0 += 1) {
                                        vlds(_inl12_vreg_input_cur_0, vf_tile_ptr_49, ((_inl12_i_0 * 128) + (_inl12_j_0 * 64)), NORM);
                                        vdiv(_inl12_vreg_div_0, _inl12_vreg_input_cur_0, _inl12_vreg_exp_sum_0, _inl12_preg_all_0, MODE_ZEROING);
                                        vsts(_inl12_vreg_div_0, (vf_tile_ptr_49 + (((_inl12_i_0 * 128) + (_inl12_j_0 * 64)))), 0, NORM_B32, _inl12_preg_all_0);
                                    }
                                }
                            }
                            get_buf(PIPE_V, 11, 0);
                            TCVT(_tg_o_f16_g_tiles_0_0, _tg_running_o_g_tiles_0_0, RoundMode::CAST_ROUND);
                            rls_buf(PIPE_V, 11, 0);
                            get_buf(PIPE_MTE3, 11, 0);
                            tensor_o_0.SetShape<pto::GlobalTensorDim::DIM_3, pto::GlobalTensorDim::DIM_4>(static_cast<int64_t>(64), static_cast<int64_t>(128));
                            TASSIGN(tensor_o_0, (__gm__ half*)(attention_out_0) + (g_ctx_0.b_idx * tiling_0.sq * tiling_0.n * tiling_0.d + (((g_ctx_0.qi * 2) + sub_id_0) * 64) * tiling_0.n * tiling_0.d + g_ctx_0.n_idx * tiling_0.d + 0));
                            TSTORE(tensor_o_0, _tg_o_f16_g_tiles_0_0);
                            rls_buf(PIPE_MTE3, 11, 0);
                        }
                    }
                }
                auto task_id_15 = (task_id_8 + 1);
                task_id_8 = task_id_15;
            }
            auto q_count_5 = (q_count_0 + 1);
            q_count_0 = q_count_5;
        }
    }
    #endif
}
