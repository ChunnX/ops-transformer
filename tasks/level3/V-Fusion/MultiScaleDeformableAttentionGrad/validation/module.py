import math
import torch
import torch.nn as nn
import kernel_gen_ops


def ms_deform_attn_grad_ref(value, spatial_shapes, level_start_index,
                            sampling_locations, attention_weights, grad_output):
    """
    Reference implementation matching the kernel's data layout.
    - value: [B, N, nH, D]
    - spatial_shapes: [L, 2]
    - level_start_index: [L]
    - sampling_locations: [B, nH, L, nP, 2, nQ]
    - attention_weights: [B, nH, L, nP, nQ]
    - grad_output: [B, nQ, nH, D]
    """
    B, N, nH, D = value.shape
    nL = spatial_shapes.shape[0]
    nP = sampling_locations.shape[3]
    nQ = sampling_locations.shape[5]

    value_f = value.float()
    grad_output_f = grad_output.float()
    sampling_loc_f = sampling_locations.float()
    attn_w_f = attention_weights.float()

    grad_value = torch.zeros_like(value_f)
    grad_sampling_loc = torch.zeros_like(sampling_loc_f)
    grad_attn_weight = torch.zeros_like(attn_w_f)

    for b in range(B):
        for h in range(nH):
            base_ptr = h * D
            for l in range(nL):
                H_l = spatial_shapes[l, 0].item()
                W_l = spatial_shapes[l, 1].item()
                start = level_start_index[l].item()
                wStride = nH * D
                hStride = W_l * wStride
                val_offset = b * N * nH * D + start * wStride

                for p in range(nP):
                    for q in range(nQ):
                        loc_w = sampling_loc_f[b, h, l, p, 0, q].item()
                        loc_h = sampling_loc_f[b, h, l, p, 1, q].item()

                        w_im = loc_w * W_l - 0.5
                        h_im = loc_h * H_l - 0.5

                        if h_im <= -1 or w_im <= -1 or h_im >= H_l or w_im >= W_l:
                            continue

                        h_low = int(math.floor(h_im))
                        w_low = int(math.floor(w_im))
                        h_high = h_low + 1
                        w_high = w_low + 1

                        dist_h = h_im - h_low
                        dist_w = w_im - w_low
                        hh = 1.0 - dist_h
                        hw = 1.0 - dist_w

                        aw = attn_w_f[b, h, l, p, q].item()
                        go = grad_output_f[b, q, h]

                        top_grad_value = aw * go

                        w1 = hh * hw
                        w2 = hh * dist_w
                        w3 = dist_h * hw
                        w4 = dist_h * dist_w

                        v = [torch.zeros(D), torch.zeros(D), torch.zeros(D), torch.zeros(D)]

                        if h_low >= 0 and w_low >= 0:
                            idx = val_offset + h_low * hStride + w_low * wStride + base_ptr
                            v[0] = value_f.flatten()[idx:idx + D].clone()
                            grad_value.flatten()[idx:idx + D] += top_grad_value * w1

                        if h_low >= 0 and w_high < W_l:
                            idx = val_offset + h_low * hStride + w_high * wStride + base_ptr
                            v[1] = value_f.flatten()[idx:idx + D].clone()
                            grad_value.flatten()[idx:idx + D] += top_grad_value * w2

                        if h_high < H_l and w_low >= 0:
                            idx = val_offset + h_high * hStride + w_low * wStride + base_ptr
                            v[2] = value_f.flatten()[idx:idx + D].clone()
                            grad_value.flatten()[idx:idx + D] += top_grad_value * w3

                        if h_high < H_l and w_high < W_l:
                            idx = val_offset + h_high * hStride + w_high * wStride + base_ptr
                            v[3] = value_f.flatten()[idx:idx + D].clone()
                            grad_value.flatten()[idx:idx + D] += top_grad_value * w4

                        bilinear_val = w1 * v[0] + w2 * v[1] + w3 * v[2] + w4 * v[3]
                        grad_attn_weight[b, h, l, p, q] = torch.dot(go, bilinear_val)

                        grad_w = torch.zeros(D)
                        grad_h = torch.zeros(D)

                        if h_low >= 0 and w_low >= 0:
                            grad_h -= hw * v[0]
                            grad_w -= hh * v[0]
                        if h_low >= 0 and w_high < W_l:
                            grad_h -= dist_w * v[1]
                            grad_w += hh * v[1]
                        if h_high < H_l and w_low >= 0:
                            grad_h += hw * v[2]
                            grad_w -= dist_h * v[2]
                        if h_high < H_l and w_high < W_l:
                            grad_h += dist_w * v[3]
                            grad_w += dist_h * v[3]

                        grad_sampling_loc[b, h, l, p, 0, q] = torch.dot(top_grad_value, W_l * grad_w)
                        grad_sampling_loc[b, h, l, p, 1, q] = torch.dot(top_grad_value, H_l * grad_h)

    return grad_value, grad_sampling_loc, grad_attn_weight


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, value, spatial_shapes, level_start_index,
                sampling_locations, attention_weights, grad_output):
        return ms_deform_attn_grad_ref(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, grad_output)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, value, spatial_shapes, level_start_index,
                sampling_locations, attention_weights, grad_output):
        results = kernel_gen_ops.multi_scale_deformable_attention_grad(
            value, spatial_shapes, level_start_index,
            sampling_locations, attention_weights, grad_output)
        return results
