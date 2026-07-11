/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "eye_decoupled_attention_tiling_data.h"
#include "kernel_operator.h"
#include "tiling_utils.h"
#include "fp16_utils.h"
#include "matmul_utils.h"

using namespace AscendC;

class EyeDecoupledAttentionFwd {
public:
    __aicore__ inline EyeDecoupledAttentionFwd() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR pos, GM_ADDR w_short_qkv, GM_ADDR w_short_o, GM_ADDR w_long_qkv, GM_ADDR w_long_o,
                                GM_ADDR w_ffn1, GM_ADDR w_ffn2, GM_ADDR w_ffn3, GM_ADDR w_ffn4, GM_ADDR gamma1, GM_ADDR beta1, GM_ADDR gamma2, GM_ADDR beta2,
                                GM_ADDR gamma3, GM_ADDR beta3, GM_ADDR gamma4, GM_ADDR beta4, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {

        REGISTER_TILING_DEFAULT(EyeDecoupledAttentionTilingData);
    GET_TILING_DATA_WITH_STRUCT(EyeDecoupledAttentionTilingData, tiling_data, tiling);
        B_ = tiling_data.B;
        N_ = tiling_data.N;
        C_ = tiling_data.C;
        hb_ = tiling_data.hb;
        wb_ = tiling_data.wb;
        bn_h_ = tiling_data.bn_h;
        bn_w_ = tiling_data.bn_w;

        x_gm_.SetGlobalBuffer((__gm__ half*)x, B_ * N_ * C_);
        pos_gm_.SetGlobalBuffer((__gm__ half*)pos, B_ * N_ * C_);
        y_gm_.SetGlobalBuffer((__gm__ half*)y, B_ * N_ * C_);

        w_short_qkv_gm_.SetGlobalBuffer((__gm__ half*)w_short_qkv, C_ * C_ * 3);
        w_short_o_gm_.SetGlobalBuffer((__gm__ half*)w_short_o, C_ * C_);
        w_long_qkv_gm_.SetGlobalBuffer((__gm__ half*)w_long_qkv, C_ * C_ * 3);
        w_long_o_gm_.SetGlobalBuffer((__gm__ half*)w_long_o, C_ * C_);

        w_ffn1_gm_.SetGlobalBuffer((__gm__ half*)w_ffn1, C_ * C_);
        w_ffn2_gm_.SetGlobalBuffer((__gm__ half*)w_ffn2, C_ * C_);
        w_ffn3_gm_.SetGlobalBuffer((__gm__ half*)w_ffn3, C_ * C_);
        w_ffn4_gm_.SetGlobalBuffer((__gm__ half*)w_ffn4, C_ * C_);

        gamma1_gm_.SetGlobalBuffer((__gm__ half*)gamma1, C_);
        beta1_gm_.SetGlobalBuffer((__gm__ half*)beta1, C_);
        gamma2_gm_.SetGlobalBuffer((__gm__ half*)gamma2, C_);
        beta2_gm_.SetGlobalBuffer((__gm__ half*)beta2, C_);
        gamma3_gm_.SetGlobalBuffer((__gm__ half*)gamma3, C_);
        beta3_gm_.SetGlobalBuffer((__gm__ half*)beta3, C_);
        gamma4_gm_.SetGlobalBuffer((__gm__ half*)gamma4, C_);
        beta4_gm_.SetGlobalBuffer((__gm__ half*)beta4, C_);

        qkv_workspace_ = (__gm__ half*)workspace;

        uint32_t buf_size = C_;
        if (N_ > buf_size) buf_size = N_;

        pipe_.InitBuffer(xrow_buf_, buf_size * sizeof(half));
        pipe_.InitBuffer(posrow_buf_, buf_size * sizeof(half));
        pipe_.InitBuffer(qrow_buf_, buf_size * sizeof(half));
        pipe_.InitBuffer(krow_buf_, buf_size * sizeof(half));
        pipe_.InitBuffer(vrow_buf_, buf_size * sizeof(half));
        pipe_.InitBuffer(outrow_buf_, buf_size * sizeof(half));
        pipe_.InitBuffer(hidden_buf_, buf_size * sizeof(half));
        pipe_.InitBuffer(score_buf_, buf_size * sizeof(half));
        pipe_.InitBuffer(gamma_buf_, buf_size * sizeof(half));
        pipe_.InitBuffer(beta_buf_, buf_size * sizeof(half));
        pipe_.InitBuffer(wcol_buf_, buf_size * sizeof(half));
        pipe_.InitBuffer(tmp_buf_, buf_size * sizeof(half));
    }

    __aicore__ inline void Process() {
        int32_t total_tokens = B_ * N_;
        int32_t C = (int32_t)C_;

        // Phase 1: precompute Q/K/V into workspace for short attention
        for (int32_t t = 0; t < total_tokens; ++t) {
            ComputeAndStoreQKV(t, w_short_qkv_gm_, 0, C);
        }

        // Phase 2: per-token short attention + FFN
        for (int32_t t = 0; t < total_tokens; ++t) {
            LocalTensor<half> xrow = xrow_buf_.Get<half>();
            LocalTensor<half> out = outrow_buf_.Get<half>();
            ReadRow(x_gm_, xrow, t, C);
            DataCopy(out, xrow, C);
            PipeBarrier<PIPE_ALL>();

            AttentionFromWorkspace(t, out, xrow, 0, w_short_o_gm_, gamma1_gm_, beta1_gm_, C);
            FFN(out, xrow, w_ffn1_gm_, w_ffn2_gm_, gamma2_gm_, beta2_gm_, C);
            DataCopy(y_gm_[t * C], out, C);
            PipeBarrier<PIPE_ALL>();
        }

        // Phase 3: precompute Q/K/V for long attention using y as input
        for (int32_t t = 0; t < total_tokens; ++t) {
            ComputeAndStoreQKVFromY(t, w_long_qkv_gm_, 0, C);
        }

        // Phase 4: per-token long attention + FFN
        for (int32_t t = 0; t < total_tokens; ++t) {
            LocalTensor<half> yrow = xrow_buf_.Get<half>();
            LocalTensor<half> out = outrow_buf_.Get<half>();
            ReadRow(y_gm_, yrow, t, C);
            DataCopy(out, yrow, C);
            PipeBarrier<PIPE_ALL>();

            AttentionFromWorkspace(t, out, yrow, 0, w_long_o_gm_, gamma3_gm_, beta3_gm_, C);
            FFN(out, yrow, w_ffn3_gm_, w_ffn4_gm_, gamma4_gm_, beta4_gm_, C);
            DataCopy(y_gm_[t * C], out, C);
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    __aicore__ inline __gm__ half* WorkspaceQ(int32_t token, int32_t C) {
        return qkv_workspace_ + token * C;
    }
    __aicore__ inline __gm__ half* WorkspaceK(int32_t token, int32_t C) {
        return qkv_workspace_ + B_ * N_ * C + token * C;
    }
    __aicore__ inline __gm__ half* WorkspaceV(int32_t token, int32_t C) {
        return qkv_workspace_ + 2 * B_ * N_ * C + token * C;
    }

    __aicore__ inline void ReadRow(GlobalTensor<half>& gm, LocalTensor<half>& dst,
                                   int32_t token, int32_t C) {
        DataCopy(dst, gm[token * C], C);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void WriteRow(__gm__ half* gm, LocalTensor<half>& src,
                                    int32_t token, int32_t C) {
        GlobalTensor<half> gt;
        gt.SetGlobalBuffer(gm + token * C, C);
        DataCopy(gt, src, C);
        PipeBarrier<PIPE_ALL>();
    }

    // Compute q/k/v from (x + pos) using W_qkv and store to workspace
    __aicore__ inline void ComputeAndStoreQKV(int32_t token,
                                              GlobalTensor<half>& W_qkv,
                                              int32_t w_offset,
                                              int32_t C) {
        LocalTensor<half> xrow = xrow_buf_.Get<half>();
        LocalTensor<half> posrow = posrow_buf_.Get<half>();
        LocalTensor<half> q = qrow_buf_.Get<half>();
        LocalTensor<half> k = krow_buf_.Get<half>();
        LocalTensor<half> v = vrow_buf_.Get<half>();

        ReadRow(x_gm_, xrow, token, C);
        ReadRow(pos_gm_, posrow, token, C);
        ComputeQKV(q, k, v, xrow, posrow, W_qkv, C);

        WriteRow(WorkspaceQ(token, C), q, 0, C);
        WriteRow(WorkspaceK(token, C), k, 0, C);
        WriteRow(WorkspaceV(token, C), v, 0, C);
    }

    __aicore__ inline void ComputeAndStoreQKVFromY(int32_t token,
                                                   GlobalTensor<half>& W_qkv,
                                                   int32_t w_offset,
                                                   int32_t C) {
        LocalTensor<half> yrow = xrow_buf_.Get<half>();
        LocalTensor<half> posrow = posrow_buf_.Get<half>();
        LocalTensor<half> q = qrow_buf_.Get<half>();
        LocalTensor<half> k = krow_buf_.Get<half>();
        LocalTensor<half> v = vrow_buf_.Get<half>();

        ReadRow(y_gm_, yrow, token, C);
        ReadRow(pos_gm_, posrow, token, C);
        ComputeQKV(q, k, v, yrow, posrow, W_qkv, C);

        WriteRow(WorkspaceQ(token, C), q, 0, C);
        WriteRow(WorkspaceK(token, C), k, 0, C);
        WriteRow(WorkspaceV(token, C), v, 0, C);
    }

    // Compute q/k/v from (src + pos) using W_qkv [C, 3C]
    __aicore__ inline void ComputeQKV(LocalTensor<half>& q,
                                      LocalTensor<half>& k,
                                      LocalTensor<half>& v,
                                      LocalTensor<half>& src,
                                      LocalTensor<half>& pos,
                                      GlobalTensor<half>& W_qkv,
                                      int32_t C) {
        for (int32_t c = 0; c < C; ++c) {
            q[c] = (half)0.0f;
            k[c] = (half)0.0f;
            v[c] = (half)0.0f;
        }
        LocalTensor<half> wcol = wcol_buf_.Get<half>();
        for (int32_t d = 0; d < C; ++d) {
            half xp = src[d] + pos[d];
            DataCopy(wcol, W_qkv[d * (3 * C)], C);
            PipeBarrier<PIPE_ALL>();
            for (int32_t c = 0; c < C; ++c) q[c] += xp * wcol[c];

            DataCopy(wcol, W_qkv[d * (3 * C) + C], C);
            PipeBarrier<PIPE_ALL>();
            for (int32_t c = 0; c < C; ++c) k[c] += xp * wcol[c];

            DataCopy(wcol, W_qkv[d * (3 * C) + 2 * C], C);
            PipeBarrier<PIPE_ALL>();
            for (int32_t c = 0; c < C; ++c) v[c] += xp * wcol[c];
        }
    }

    // Attention using precomputed Q/K/V in workspace, then output projection and residual+LN
    __aicore__ inline void AttentionFromWorkspace(int32_t token,
                                                  LocalTensor<half>& out,
                                                  LocalTensor<half>& residual,
                                                  int32_t workspace_offset,
                                                  GlobalTensor<half>& W_o,
                                                  GlobalTensor<half>& gamma,
                                                  GlobalTensor<half>& beta,
                                                  int32_t C) {
        LocalTensor<half> q = qrow_buf_.Get<half>();
        LocalTensor<half> k = krow_buf_.Get<half>();
        LocalTensor<half> v = vrow_buf_.Get<half>();
        LocalTensor<half> scores = score_buf_.Get<half>();
        LocalTensor<half> scratch = tmp_buf_.Get<half>();

        // Load q for current token
        GlobalTensor<half> q_gm;
        q_gm.SetGlobalBuffer(WorkspaceQ(token, C), C);
        DataCopy(q, q_gm, C);
        PipeBarrier<PIPE_ALL>();

        // Compute attention scores
        for (int32_t j = 0; j < (int32_t)N_; ++j) {
            GlobalTensor<half> k_gm;
            k_gm.SetGlobalBuffer(WorkspaceK(j, C), C);
            DataCopy(k, k_gm, C);
            PipeBarrier<PIPE_ALL>();
            half s = (half)0.0f;
            for (int32_t c = 0; c < C; ++c) s += q[c] * k[c];
            scores[j] = s;
        }

        // Softmax
        SoftmaxUb(scores, scores, scratch, N_);

        // Weighted sum of values
        for (int32_t c = 0; c < C; ++c) out[c] = (half)0.0f;
        for (int32_t j = 0; j < (int32_t)N_; ++j) {
            GlobalTensor<half> v_gm;
            v_gm.SetGlobalBuffer(WorkspaceV(j, C), C);
            DataCopy(v, v_gm, C);
            PipeBarrier<PIPE_ALL>();
            for (int32_t c = 0; c < C; ++c) out[c] += scores[j] * v[c];
        }

        // Output projection out = out @ W_o
        LocalTensor<half> proj = qrow_buf_.Get<half>();
        for (int32_t c = 0; c < C; ++c) proj[c] = out[c];
        for (int32_t c = 0; c < C; ++c) out[c] = (half)0.0f;
        LocalTensor<half> wcol = wcol_buf_.Get<half>();
        for (int32_t d = 0; d < C; ++d) {
            DataCopy(wcol, W_o[d * C], C);
            PipeBarrier<PIPE_ALL>();
            for (int32_t c = 0; c < C; ++c) out[c] += proj[d] * wcol[c];
        }

        // Residual + LayerNorm
        LocalTensor<half> g = gamma_buf_.Get<half>();
        LocalTensor<half> b = beta_buf_.Get<half>();
        DataCopy(g, gamma, C);
        DataCopy(b, beta, C);
        PipeBarrier<PIPE_ALL>();
        for (int32_t c = 0; c < C; ++c) out[c] += residual[c];
        LayerNormUb(out, out, scratch, g, b, C);
    }

    __aicore__ inline void FFN(LocalTensor<half>& x,
                               LocalTensor<half>& residual,
                               GlobalTensor<half>& W1,
                               GlobalTensor<half>& W2,
                               GlobalTensor<half>& gamma,
                               GlobalTensor<half>& beta,
                               int32_t C) {
        LocalTensor<half> hidden = hidden_buf_.Get<half>();
        LocalTensor<half> out = x;
        LocalTensor<half> wcol = wcol_buf_.Get<half>();

        // hidden = relu(x @ W1)
        for (int32_t c = 0; c < C; ++c) hidden[c] = (half)0.0f;
        for (int32_t d = 0; d < C; ++d) {
            DataCopy(wcol, W1[d * C], C);
            PipeBarrier<PIPE_ALL>();
            for (int32_t c = 0; c < C; ++c) hidden[c] += x[d] * wcol[c];
        }
        for (int32_t c = 0; c < C; ++c) {
            hidden[c] = (hidden[c] > (half)0.0f) ? hidden[c] : (half)0.0f;
        }

        // out = hidden @ W2
        for (int32_t c = 0; c < C; ++c) out[c] = (half)0.0f;
        for (int32_t d = 0; d < C; ++d) {
            DataCopy(wcol, W2[d * C], C);
            PipeBarrier<PIPE_ALL>();
            for (int32_t c = 0; c < C; ++c) out[c] += hidden[d] * wcol[c];
        }

        // residual + LN
        LocalTensor<half> g = gamma_buf_.Get<half>();
        LocalTensor<half> b = beta_buf_.Get<half>();
        LocalTensor<half> tmp = tmp_buf_.Get<half>();
        DataCopy(g, gamma, C);
        DataCopy(b, beta, C);
        PipeBarrier<PIPE_ALL>();
        for (int32_t c = 0; c < C; ++c) out[c] += residual[c];
        LayerNormUb(out, out, tmp, g, b, C);
    }

    uint32_t B_, N_, C_, hb_, wb_, bn_h_, bn_w_;
    __gm__ half* qkv_workspace_;

    GlobalTensor<half> x_gm_, pos_gm_, y_gm_;
    GlobalTensor<half> w_short_qkv_gm_, w_short_o_gm_;
    GlobalTensor<half> w_long_qkv_gm_, w_long_o_gm_;
    GlobalTensor<half> w_ffn1_gm_, w_ffn2_gm_, w_ffn3_gm_, w_ffn4_gm_;
    GlobalTensor<half> gamma1_gm_, beta1_gm_;
    GlobalTensor<half> gamma2_gm_, beta2_gm_;
    GlobalTensor<half> gamma3_gm_, beta3_gm_;
    GlobalTensor<half> gamma4_gm_, beta4_gm_;

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> xrow_buf_, posrow_buf_, qrow_buf_,
        krow_buf_, vrow_buf_, outrow_buf_, hidden_buf_, score_buf_,
        gamma_buf_, beta_buf_, wcol_buf_, tmp_buf_;
};

extern "C" __global__ __aicore__ void eye_decoupled_attention_fwd(
    GM_ADDR x, GM_ADDR pos, GM_ADDR w_short_qkv, GM_ADDR w_short_o, GM_ADDR w_long_qkv, GM_ADDR w_long_o,
    GM_ADDR w_ffn1, GM_ADDR w_ffn2, GM_ADDR w_ffn3, GM_ADDR w_ffn4, GM_ADDR gamma1, GM_ADDR beta1, GM_ADDR gamma2, GM_ADDR beta2,
    GM_ADDR gamma3, GM_ADDR beta3, GM_ADDR gamma4, GM_ADDR beta4, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {

    EyeDecoupledAttentionFwd op;
    op.Init(x, pos,w_short_qkv, w_short_o,w_long_qkv, w_long_o,w_ffn1, w_ffn2, w_ffn3, w_ffn4,gamma1, beta1,gamma2, beta2,gamma3, beta3,gamma4, beta4,y, workspace, tiling);
    op.Process();
}
