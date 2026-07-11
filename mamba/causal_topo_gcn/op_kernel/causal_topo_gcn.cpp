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
#include "causal_topo_gcn_tiling_data.h"
#include "kernel_operator.h"
#include "tiling_utils.h"
#include "fp16_utils.h"
#include "matmul_utils.h"

using namespace AscendC;

class CausalTopoGCNFwd {
public:
    __aicore__ inline CausalTopoGCNFwd() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w_gcn1, GM_ADDR b_gcn1, GM_ADDR w_gcn2, GM_ADDR b_gcn2, GM_ADDR w_gcn3, GM_ADDR b_gcn3, GM_ADDR w_cls, 
                                GM_ADDR bn_scale, GM_ADDR bn_bias, GM_ADDR bn_mean, GM_ADDR bn_var, GM_ADDR embed, GM_ADDR logits, GM_ADDR workspace, GM_ADDR tiling) {
        REGISTER_TILING_DEFAULT(CausalTopoGcnTilingData);
        GET_TILING_DATA_WITH_STRUCT(CausalTopoGcnTilingData, tiling_data, tiling);
        B_           = tiling_data.B;
        K_           = tiling_data.K;
        F_           = tiling_data.F;
        H_           = tiling_data.H;
        num_classes_ = tiling_data.num_classes;
        use_effect_  = tiling_data.use_effect;
        alpha_       = (half)tiling_data.alpha;
        scale_       = (half)tiling_data.scale;
        gamma_       = (half)tiling_data.gamma;
        num_head_    = tiling_data.num_head;

        x_gm_.SetGlobalBuffer((__gm__ half*)x, B_ * K_ * F_);
        w_gcn1_gm_.SetGlobalBuffer((__gm__ half*)w_gcn1, F_ * H_);
        b_gcn1_gm_.SetGlobalBuffer((__gm__ half*)b_gcn1, H_);
        w_gcn2_gm_.SetGlobalBuffer((__gm__ half*)w_gcn2, H_ * H_);
        b_gcn2_gm_.SetGlobalBuffer((__gm__ half*)b_gcn2, H_);
        w_gcn3_gm_.SetGlobalBuffer((__gm__ half*)w_gcn3, H_ * H_);
        b_gcn3_gm_.SetGlobalBuffer((__gm__ half*)b_gcn3, H_);
        w_cls_gm_.SetGlobalBuffer((__gm__ half*)w_cls, num_classes_ * H_);
        bn_scale_gm_.SetGlobalBuffer((__gm__ half*)bn_scale, (F_ > H_ ? F_ : H_));
        bn_bias_gm_.SetGlobalBuffer((__gm__ half*)bn_bias, (F_ > H_ ? F_ : H_));
        bn_mean_gm_.SetGlobalBuffer((__gm__ half*)bn_mean, (F_ > H_ ? F_ : H_));
        bn_var_gm_.SetGlobalBuffer((__gm__ half*)bn_var, (F_ > H_ ? F_ : H_));
        logits_gm_.SetGlobalBuffer((__gm__ half*)logits, B_ * num_classes_);

        workspace_gm_ = (__gm__ half*)workspace;

        adj_offset_     = 0;
        gcn_out_offset_ = B_ * K_ * K_;
        gcn_tmp_offset_ = gcn_out_offset_ + B_ * K_ * H_;

        uint32_t max_dim = F_;
        if (K_ > max_dim) max_dim = K_;
        if (H_ > max_dim) max_dim = H_;
        if (num_classes_ > max_dim) max_dim = num_classes_;
        uint32_t buf_elems = K_ * max_dim;
        if (H_ * num_classes_ > buf_elems) buf_elems = H_ * num_classes_;
        if (K_ * K_ > buf_elems) buf_elems = K_ * K_;

        pipe_.InitBuffer(x_buf_, buf_elems * sizeof(half));
        pipe_.InitBuffer(y_buf_, buf_elems * sizeof(half));
        pipe_.InitBuffer(z_buf_, buf_elems * sizeof(half));
        pipe_.InitBuffer(tmp_buf_, buf_elems * sizeof(half));
        pipe_.InitBuffer(bn_scale_buf_, max_dim * sizeof(half));
        pipe_.InitBuffer(bn_bias_buf_, max_dim * sizeof(half));
        pipe_.InitBuffer(bn_mean_buf_, max_dim * sizeof(half));
        pipe_.InitBuffer(bn_var_buf_, max_dim * sizeof(half));
    }

    __aicore__ inline void Process() {
        int32_t b = GetBlockIdx();
        if (b >= (int32_t)B_) return;
        ComputeInputAndInitialAdj(b);
        ComputeGcnLayer1(b);
        ComputeGcnLayer2And3(b);
        ComputePoolAndLogits(b);
    }

private:
    __aicore__ inline void ComputeInputAndInitialAdj(int32_t b) {
        LocalTensor<half> x = x_buf_.Get<half>();
        LocalTensor<half> y = y_buf_.Get<half>();
        LocalTensor<half> z = z_buf_.Get<half>();
        ReadX(b, x);
        LoadBNStats(F_);
        BatchNorm1d(y, x, F_);
        PipeBarrier<PIPE_ALL>();
        WriteGcnOut(b, y);
        PipeBarrier<PIPE_ALL>();

        MatmulTransposeScore(z, y, K_, F_);
        PipeBarrier<PIPE_ALL>();
        SoftmaxRow(z, z, x, K_);
        PipeBarrier<PIPE_ALL>();
        WriteAdj(b, z);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ComputeGcnLayer1(int32_t b) {
        LocalTensor<half> x = x_buf_.Get<half>();
        LocalTensor<half> y = y_buf_.Get<half>();
        LocalTensor<half> z = z_buf_.Get<half>();
        ReadGcnOut(b, x);
        PipeBarrier<PIPE_ALL>();
        LoadBNStats(H_);
        GcnLayer(y, x, z, w_gcn1_gm_, b_gcn1_gm_, K_, F_, H_);
        PipeBarrier<PIPE_ALL>();
        WriteGcnTmp(b, y);
        PipeBarrier<PIPE_ALL>();

        ReadGcnTmp(b, x);
        PipeBarrier<PIPE_ALL>();
        MatmulTransposeScore(z, x, K_, H_);
        PipeBarrier<PIPE_ALL>();
        SoftmaxRow(z, z, x, K_);
        PipeBarrier<PIPE_ALL>();
        WriteAdj(b, z);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ComputeGcnLayer2And3(int32_t b) {
        LocalTensor<half> x = x_buf_.Get<half>();
        LocalTensor<half> y = y_buf_.Get<half>();
        LocalTensor<half> z = z_buf_.Get<half>();
        ReadGcnTmp(b, x);
        PipeBarrier<PIPE_ALL>();
        LoadBNStats(H_);
        GcnLayer(y, x, z, w_gcn2_gm_, b_gcn2_gm_, K_, H_, H_);
        PipeBarrier<PIPE_ALL>();
        WriteGcnOut(b, y);
        PipeBarrier<PIPE_ALL>();

        ReadGcnOut(b, x);
        PipeBarrier<PIPE_ALL>();
        MatmulTransposeScore(z, x, K_, H_);
        PipeBarrier<PIPE_ALL>();
        SoftmaxRow(z, z, x, K_);
        PipeBarrier<PIPE_ALL>();
        WriteAdj(b, z);
        PipeBarrier<PIPE_ALL>();

        ReadGcnOut(b, x);
        PipeBarrier<PIPE_ALL>();
        LoadBNStats(H_);
        GcnLayer(y, x, z, w_gcn3_gm_, b_gcn3_gm_, K_, H_, H_);
        PipeBarrier<PIPE_ALL>();
        WriteGcnTmp(b, y);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ComputePoolAndLogits(int32_t b) {
        LocalTensor<half> x = x_buf_.Get<half>();
        LocalTensor<half> y = y_buf_.Get<half>();
        LocalTensor<half> z = z_buf_.Get<half>();
        ReadGcnTmp(b, x);
        PipeBarrier<PIPE_ALL>();
        MeanPool(y, x, K_, H_);
        PipeBarrier<PIPE_ALL>();

        Classifier(z, y, w_cls_gm_, H_, num_classes_, scale_);
        PipeBarrier<PIPE_ALL>();
        WriteLogits(b, z);
        PipeBarrier<PIPE_ALL>();
    }

    // GM helpers
    __aicore__ inline void ReadX(int32_t b, LocalTensor<half>& dst) {
        DataCopy(dst, x_gm_[b * K_ * F_], K_ * F_);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ReadAdj(int32_t b, LocalTensor<half>& dst) {
        GlobalTensor<half> gt;
        gt.SetGlobalBuffer(workspace_gm_ + adj_offset_ + b * K_ * K_, K_ * K_);
        DataCopy(dst, gt, K_ * K_);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void WriteAdj(int32_t b, LocalTensor<half>& src) {
        GlobalTensor<half> gt;
        gt.SetGlobalBuffer(workspace_gm_ + adj_offset_ + b * K_ * K_, K_ * K_);
        DataCopy(gt, src, K_ * K_);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ReadGcnOut(int32_t b, LocalTensor<half>& dst) {
        GlobalTensor<half> gt;
        gt.SetGlobalBuffer(workspace_gm_ + gcn_out_offset_ + b * K_ * H_, K_ * H_);
        DataCopy(dst, gt, K_ * H_);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void WriteGcnOut(int32_t b, LocalTensor<half>& src) {
        GlobalTensor<half> gt;
        gt.SetGlobalBuffer(workspace_gm_ + gcn_out_offset_ + b * K_ * H_, K_ * H_);
        DataCopy(gt, src, K_ * H_);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ReadGcnTmp(int32_t b, LocalTensor<half>& dst) {
        GlobalTensor<half> gt;
        gt.SetGlobalBuffer(workspace_gm_ + gcn_tmp_offset_ + b * K_ * H_, K_ * H_);
        DataCopy(dst, gt, K_ * H_);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void WriteGcnTmp(int32_t b, LocalTensor<half>& src) {
        GlobalTensor<half> gt;
        gt.SetGlobalBuffer(workspace_gm_ + gcn_tmp_offset_ + b * K_ * H_, K_ * H_);
        DataCopy(gt, src, K_ * H_);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void WriteLogits(int32_t b, LocalTensor<half>& src) {
        DataCopy(logits_gm_[b * num_classes_], src, num_classes_);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadBNStats(int32_t len) {
        LocalTensor<half> s = bn_scale_buf_.Get<half>();
        LocalTensor<half> b = bn_bias_buf_.Get<half>();
        LocalTensor<half> m = bn_mean_buf_.Get<half>();
        LocalTensor<half> v = bn_var_buf_.Get<half>();
        DataCopy(s, bn_scale_gm_, len);
        DataCopy(b, bn_bias_gm_, len);
        DataCopy(m, bn_mean_gm_, len);
        DataCopy(v, bn_var_gm_, len);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void BatchNorm1d(LocalTensor<half>& out, LocalTensor<half>& in, int32_t len) {
        LocalTensor<half> s = bn_scale_buf_.Get<half>();
        LocalTensor<half> b = bn_bias_buf_.Get<half>();
        LocalTensor<half> m = bn_mean_buf_.Get<half>();
        LocalTensor<half> v = bn_var_buf_.Get<half>();
        half eps = (half)1e-5f;
        for (int32_t k = 0; k < (int32_t)K_; ++k) {
            for (int32_t i = 0; i < len; ++i) {
                half val = in(k * len + i);
                half mean = m(i);
                half var  = v(i);
                half std_inv = (half)(1.0f / __builtin_sqrtf((float)var + (float)eps));
                out(k * len + i) = (val - mean) * std_inv * s(i) + b(i);
            }
        }
    }

    __aicore__ inline void MatmulTransposeScore(LocalTensor<half>& adj, LocalTensor<half>& x, int32_t K, int32_t F) {
        for (int32_t i = 0; i < K; ++i) {
            for (int32_t j = 0; j < K; ++j) {
                half acc = (half)0.0f;
                for (int32_t f = 0; f < F; ++f) {
                    acc += x(i * F + f) * x(j * F + f);
                }
                adj(i * K + j) = acc;
            }
        }
    }

    __aicore__ inline void SoftmaxRow(LocalTensor<half>& out, LocalTensor<half>& in, LocalTensor<half>& tmp, int32_t K) {
        for (int32_t i = 0; i < K; ++i) {
            half max_val = in(i * K);
            for (int32_t j = 1; j < K; ++j) {
                half v = in(i * K + j);
                max_val = (v > max_val) ? v : max_val;
            }
            for (int32_t j = 0; j < K; ++j) {
                tmp(i * K + j) = in(i * K + j) - max_val;
            }
            for (int32_t j = 0; j < K; ++j) {
                float f = (float)tmp(i * K + j);
                tmp(i * K + j) = (half)__builtin_expf(f);
            }
            half sum = (half)0.0f;
            for (int32_t j = 0; j < K; ++j) {
                sum += tmp(i * K + j);
            }
            half inv_sum = (half)1.0f / sum;
            for (int32_t j = 0; j < K; ++j) {
                out(i * K + j) = tmp(i * K + j) * inv_sum;
            }
        }
    }

    __aicore__ inline void GcnLayer(LocalTensor<half>& out, LocalTensor<half>& x, LocalTensor<half>& adj_scratch,
                                    GlobalTensor<half>& W, GlobalTensor<half>& bias, int32_t K, int32_t Fin, int32_t Fout) {
        LocalTensor<half> support = tmp_buf_.Get<half>();
        for (int32_t i = 0; i < K; ++i) {
            for (int32_t j = 0; j < Fout; ++j) {
                half acc = (half)0.0f;
                for (int32_t k = 0; k < Fin; ++k) {
                    acc += x(i * Fin + k) * W(k * Fout + j);
                }
                support(i * Fout + j) = acc;
            }
        }
        int32_t b = GetBlockIdx();
        ReadAdj(b, adj_scratch);
        for (int32_t i = 0; i < K; ++i) {
            for (int32_t j = 0; j < Fout; ++j) {
                half acc = bias(j);
                for (int32_t k = 0; k < K; ++k) {
                    acc += adj_scratch(i * K + k) * support(k * Fout + j);
                }
                out(i * Fout + j) = EluHalf(acc);
            }
        }
        BatchNorm1d(out, out, Fout);
        for (int32_t i = 0; i < K * Fout; ++i) {
            out(i) = EluHalf(out(i));
        }
    }

    __aicore__ inline void MeanPool(LocalTensor<half>& out, LocalTensor<half>& x, int32_t K, int32_t H) {
        for (int32_t h = 0; h < H; ++h) {
            half acc = (half)0.0f;
            for (int32_t k = 0; k < K; ++k) {
                acc += x(k * H + h);
            }
            out(h) = acc / (half)(float)K;
        }
    }

    __aicore__ inline void L2NormRows(LocalTensor<half>& out, LocalTensor<half>& in, LocalTensor<half>& tmp, int32_t rows, int32_t cols) {
        for (int32_t i = 0; i < rows; ++i) {
            half sum_sq = (half)0.0f;
            for (int32_t j = 0; j < cols; ++j) {
                half v = in(i * cols + j);
                sum_sq += v * v;
            }
            half norm = (half)__builtin_sqrtf((float)sum_sq);
            half inv_norm = (norm > (half)0.0f) ? ((half)1.0f / norm) : (half)0.0f;
            for (int32_t j = 0; j < cols; ++j) {
                out(i * cols + j) = in(i * cols + j) * inv_norm;
            }
        }
    }

    __aicore__ inline void CausalNormRows(LocalTensor<half>& out, LocalTensor<half>& in, LocalTensor<half>& tmp, half gamma, int32_t rows, int32_t cols) {
        for (int32_t i = 0; i < rows; ++i) {
            half sum_sq = (half)0.0f;
            for (int32_t j = 0; j < cols; ++j) {
                half v = in(i * cols + j);
                sum_sq += v * v;
            }
            half norm = (half)__builtin_sqrtf((float)sum_sq);
            half inv = (half)1.0f / (norm + gamma);
            for (int32_t j = 0; j < cols; ++j) {
                out(i * cols + j) = in(i * cols + j) * inv;
            }
        }
    }

    __aicore__ inline void Classifier(LocalTensor<half>& logits, LocalTensor<half>& x_mean, GlobalTensor<half>& w_cls, int32_t H, int32_t num_classes, half scale) {
        LocalTensor<half> normed_x = y_buf_.Get<half>();
        LocalTensor<half> scratch = tmp_buf_.Get<half>();
        for (int32_t h = 0; h < H; ++h) {
            normed_x(h) = x_mean(h);
        }
        L2NormRows(normed_x, normed_x, scratch, 1, H);
        PipeBarrier<PIPE_ALL>();

        LocalTensor<half> normed_w = x_buf_.Get<half>();
        DataCopy(normed_w, w_cls, num_classes * H);
        PipeBarrier<PIPE_ALL>();
        CausalNormRows(normed_w, normed_w, scratch, gamma_, num_classes, H);
        PipeBarrier<PIPE_ALL>();

        for (int32_t c = 0; c < num_classes; ++c) {
            half acc = (half)0.0f;
            for (int32_t h = 0; h < H; ++h) {
                acc += normed_x(h) * normed_w(c * H + h);
            }
            logits(c) = acc * scale;
        }
    }

    uint32_t B_, K_, F_, H_, num_classes_, use_effect_, num_head_;
    half alpha_, scale_, gamma_;
    uint32_t adj_offset_, gcn_out_offset_, gcn_tmp_offset_;
    __gm__ half* workspace_gm_;

    GlobalTensor<half> x_gm_;
    GlobalTensor<half> w_gcn1_gm_, b_gcn1_gm_, w_gcn2_gm_, b_gcn2_gm_, w_gcn3_gm_, b_gcn3_gm_, w_cls_gm_;
    GlobalTensor<half> bn_scale_gm_, bn_bias_gm_, bn_mean_gm_, bn_var_gm_, logits_gm_;

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> x_buf_, y_buf_, z_buf_, tmp_buf_;
    TBuf<QuePosition::VECCALC> bn_scale_buf_, bn_bias_buf_, bn_mean_buf_, bn_var_buf_;
};

extern "C" __global__ __aicore__ void causal_topo_gcn_fwd(GM_ADDR x, GM_ADDR w_gcn1, GM_ADDR b_gcn1, GM_ADDR w_gcn2, GM_ADDR b_gcn2, GM_ADDR w_gcn3, GM_ADDR b_gcn3, GM_ADDR w_cls, 
                                                          GM_ADDR bn_scale, GM_ADDR bn_bias, GM_ADDR bn_mean, GM_ADDR bn_var, GM_ADDR embed, GM_ADDR logits, GM_ADDR workspace, GM_ADDR tiling) {
    CausalTopoGCNFwd op;
    op.Init(x, w_gcn1, b_gcn1, w_gcn2, b_gcn2, w_gcn3, b_gcn3, w_cls, bn_scale, bn_bias, bn_mean, bn_var, embed, logits, workspace, tiling);
    op.Process();
}