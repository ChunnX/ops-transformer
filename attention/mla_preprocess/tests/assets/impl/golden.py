"""MlaPreprocess CPU Golden 实现 — 从 ATK executor.py 移植

包含两套 CPU 计算路径:
  1. 高精度路径 (cpu_mla_preprocess) — 使用 fp32 中间计算, 作为三方对比的 golden
  2. 低精度路径 (cpu_mla_preprocess_lowprec) — 匹配 NPU kernel 中间精度截断, 作为三方对比的 benchmark

C API 参数顺序 (aclnnMlaPreprocessGetWorkspaceSize):
  inputs: input, gamma0, beta0, quantScale0, quantOffset0, wdqkv, deScale0, bias0,
          gamma1, beta1, quantScale1, quantOffset1, wuq, deScale1, bias1, gamma2,
          cos, sin, wuk, kvCache, kvCacheRope, slotmapping, ctkvScale, qNopeScale
  attr:   wdqDim, qRopeDim, kRopeDim, epsilon, qRotaryCoeff, kRotaryCoeff,
          transposeWdq, transposeWuq, transposeWuk, cacheMode, quantMode, doRmsNorm, wdkvSplitCount
  outputs: qOutOut, kvCacheOutOut, qRopeOutOut, krCacheOutOut
"""

import torch
import torch.nn.functional as F
import numpy as np

QUANTMAX = 127
QUANTMIN = -128
BLOCK_NUM = 192
BLOCK_SIZE = 128


def round_up(val, align):
    if align == 0:
        return 0
    return -(val // -align) * align


def transdata_nd2nz(x):
    align = 32 if x.dtype == torch.int8 else 16
    x = x if x.dim() == 3 else x.unsqueeze(0)
    batch, row, col = x.shape[0], round_up(x.shape[1], 16), round_up(x.shape[2], align)
    pad_row = row - x.shape[1]
    pad_col = col - x.shape[2]
    ret = F.pad(x, (0, pad_col, 0, pad_row))
    ret = torch.reshape(ret, (batch, row // 16, 16, col // align, align))
    ret = torch.permute(ret, [0, 3, 1, 2, 4])
    ret = torch.reshape(ret, (ret.shape[0], ret.shape[1], ret.shape[2] * ret.shape[3], ret.shape[4]))
    return ret


def transdata_nz2nd(nz_tensor, orig_shape, dtype):
    align = 32 if dtype == torch.int8 else 16
    block = 16
    if nz_tensor.dim() == 4:
        batch, col_blocks, row_padded, _ = nz_tensor.shape
    else:
        batch = 1
        col_blocks, row_padded, _ = nz_tensor.shape
    row_blocks = row_padded // block
    padded_row = row_blocks * block
    padded_col = col_blocks * align
    nz = nz_tensor.reshape(batch, col_blocks, row_blocks, block, align)
    nd = nz.permute(0, 2, 1, 3, 4).reshape(batch, padded_row, padded_col)
    if batch == 1:
        nd = nd.squeeze(0)
    return nd[:orig_shape[0], :orig_shape[1]].contiguous()


def reshape_and_cache_cpu(key, key_rope, key_cache_in, key_cache_rope_in, slot_mapping, cache_mode):
    key_cache_in, key_cache_rope_in = key_cache_in.clone(), key_cache_rope_in.clone()
    valid_indicies = torch.where(slot_mapping >= 0)[0]
    valid_slots = slot_mapping[valid_indicies]
    key_nope_valid = key[valid_indicies]
    key_rope_valid = key_rope[valid_indicies]
    block_idx = valid_slots // BLOCK_SIZE
    block_offset = valid_slots % BLOCK_SIZE
    cache_dtype = key_cache_in.dtype
    rope_cache_dtype = key_cache_rope_in.dtype
    if cache_mode == 0:
        key_cache_in[block_idx, block_offset] = torch.unsqueeze(
            torch.cat((key_nope_valid, key_rope_valid), dim=-1), dim=1
        ).to(cache_dtype)
        return key_cache_in, torch.zeros_like(key_cache_in)
    key_cache_in[block_idx, block_offset] = torch.unsqueeze(key_nope_valid, dim=1).to(cache_dtype)
    key_cache_rope_in[block_idx, block_offset] = torch.unsqueeze(key_rope_valid, dim=1).to(rope_cache_dtype)
    return key_cache_in, key_cache_rope_in


def rms_norm_cpu(x, gamma, beta=None, epsilon=1e-6):
    square_sum = torch.sum(torch.square(x.float()), axis=-1, keepdims=True)
    rms_factor = 1.0 / torch.sqrt(square_sum / x.shape[-1] + epsilon)
    ret = x * rms_factor * gamma.float()
    if beta is not None:
        ret += beta.float()
    return ret


def rms_norm_quant_cpu(x, gamma, beta, quant_scale=None, quant_offset=None):
    x_norm = rms_norm_cpu(x, gamma, beta)
    dynamic_quant = quant_scale is None and quant_offset is None
    if dynamic_quant:
        max_abs_val, _ = torch.max(torch.abs(x_norm), dim=-1, keepdim=True)
        scale = 127.0 / max_abs_val
        ret = torch.clamp(torch.round(x_norm * scale).half(), min=-128, max=127)
        return ret, 1 / scale
    else:
        quant_scale = 1.0 / quant_scale.float().item()
        quant_offset = quant_offset.float().item()
        x_quant = x_norm * quant_scale + quant_offset
        x_quant_fp16 = x_quant.to(torch.float16)
        ret = torch.clamp(torch.round(x_quant_fp16.float()).half(), min=-128, max=127).to(torch.int8)
    return ret


def mm_w8a8_cpu(x, weight, deq_scale, per_token_scale=None, deq_offset=None):
    ret = torch.matmul(x.to(torch.int32), weight.transpose(0, 1).to(torch.int32))
    deq_scale = deq_scale.reshape((1, -1))
    deq_offset = deq_offset.reshape((1, -1))
    if per_token_scale is not None:
        ret = ret.to(torch.float32) * deq_scale * per_token_scale
    else:
        ret += deq_offset
        ret = ret.to(torch.float32) * deq_scale
    return ret


def quant_cpu(x, scale, offset=None):
    x = x.to(torch.float32)
    scale = 1.0 / scale.to(torch.float32)
    ret = torch.round(x * scale)
    if offset is not None:
        ret += offset
    ret = torch.clamp(torch.round(ret).to(torch.int32), min=-128, max=127)
    return ret.to(torch.int8)


def mm_cpu(x, weight):
    x = x.to(torch.float32)
    weight = weight.transpose(1, 0).to(torch.float32)
    return torch.matmul(x, weight)


def mm_einsum_cpu(x, weight):
    x = torch.permute(x, (1, 0, 2)).to(torch.float32)
    weight = weight.to(torch.float32)
    return torch.permute(torch.matmul(x, weight), (1, 0, 2))


def rope_cpu(x, cos, sin):
    def rotate_half(x):
        x0, x1 = x.chunk(2, -1)
        return torch.cat((-x1, x0), dim=x0.ndim - 1)
    x = x.to(torch.float32)
    sin = sin.to(torch.float32)
    cos = cos.to(torch.float32)
    ntoken, head_num, head_size = x.shape
    batch = ntoken
    x = x.view(batch, 1, head_num, head_size)
    cos = cos.view(batch, 1, head_size).unsqueeze(2)
    sin = sin.view(batch, 1, head_size).unsqueeze(2)
    return ((x * cos) + (rotate_half(x) * sin)).view(ntoken, head_num, head_size)


def ein_sum_out_quant_golden(input, scale):
    scale = scale[None, :, None]
    if scale.dtype == torch.bfloat16:
        input = input.float()
        scale = scale.float()
    quant = input * scale
    output = torch.round(quant.float()).half()
    output = torch.min(output, torch.tensor(QUANTMAX, dtype=torch.half))
    output = torch.max(output, torch.tensor(QUANTMIN, dtype=torch.half))
    return output.to(torch.int8)


# ==================== 低精度 CPU 路径 ====================

def rms_norm_quant_cpu_lowprec(x, gamma, beta, quant_scale=None, quant_offset=None, data_type=torch.float16):
    x_norm = rms_norm_cpu(x, gamma, beta)
    dynamic_quant = quant_scale is None and quant_offset is None
    if dynamic_quant:
        max_abs_val, _ = torch.max(torch.abs(x_norm), dim=-1, keepdim=True)
        scale = 127.0 / max_abs_val
        ret = torch.clamp(torch.round(x_norm * scale).half(), min=-128, max=127)
        return ret, 1 / scale
    else:
        quant_scale_dt = quant_scale.to(data_type)
        input_scale = (torch.tensor(1.0, dtype=torch.float32) / quant_scale_dt.float()).item()
        input_offset = quant_offset.float().item()
        x_quant = x_norm * input_scale + input_offset
        x_quant_fp16 = x_quant.to(torch.float16)
        ret = torch.clamp(torch.round(x_quant_fp16.float()).half(), min=-128, max=127).to(torch.int8)
        return ret


def mm_w8a8_cpu_lowprec(x, weight, deq_scale, per_token_scale=None, deq_offset=None, data_type=torch.float16):
    ret = torch.matmul(x.to(torch.int32), weight.transpose(0, 1).to(torch.int32))
    deq_scale = deq_scale.reshape((1, -1))
    deq_offset = deq_offset.reshape((1, -1))
    if per_token_scale is not None:
        ret = ret.to(torch.float32) * deq_scale * per_token_scale
    else:
        ret += deq_offset
        ret = ret.to(torch.float32) * deq_scale
    ret = ret.to(data_type).float()
    return ret


def mm_cpu_lowprec(x, weight, data_type=torch.float16):
    x_dt = x.to(data_type)
    weight_dt = weight.transpose(1, 0).to(data_type)
    ret = torch.matmul(x_dt, weight_dt).float()
    ret = ret.to(data_type).float()
    return ret


def mm_einsum_cpu_lowprec(x, weight, data_type=torch.float16):
    x_dt = torch.permute(x, (1, 0, 2)).to(data_type)
    weight_dt = weight.to(data_type)
    ret = torch.permute(torch.matmul(x_dt, weight_dt), (1, 0, 2)).float()
    ret = ret.to(data_type).float()
    return ret


def rope_cpu_lowprec(x, cos, sin, data_type=torch.float16):
    def rotate_half(x):
        x0, x1 = x.chunk(2, -1)
        return torch.cat((-x1, x0), dim=x0.ndim - 1)
    x = x.to(torch.float32)
    sin = sin.to(torch.float32)
    cos = cos.to(torch.float32)
    ntoken, head_num, head_size = x.shape
    batch = ntoken
    x = x.view(batch, 1, head_num, head_size)
    cos = cos.view(batch, 1, head_size).unsqueeze(2)
    sin = sin.view(batch, 1, head_size).unsqueeze(2)
    result = ((x * cos) + (rotate_half(x) * sin)).view(ntoken, head_num, head_size)
    result = result.to(data_type).float()
    return result


def quant_cpu_lowprec(x, scale, offset=None, data_type=torch.float16):
    x = x.to(torch.float32)
    scale_dt = scale.to(data_type)
    inv_scale = 1.0 / scale_dt.float()
    ret = torch.round(x * inv_scale)
    if offset is not None:
        ret += offset.float()
    ret = torch.clamp(torch.round(ret).to(torch.int32), min=-128, max=127)
    return ret.to(torch.int8)


# ==================== 公共辅助 ====================

def _restore_descale(tensor):
    """如果 deScale 被 customize_inputs 处理为 int64, 还原为 fp32"""
    if tensor is None:
        return tensor
    if isinstance(tensor, torch.Tensor):
        if tensor.dtype == torch.int64:
            return tensor.to(torch.int32).view(torch.float32).clone()
        return tensor.float()
    return tensor


def _restore_nz(tensor, orig_shape):
    """如果 tensor 被 customize_inputs 转为 NZ 格式, 还原为 ND"""
    if tensor is None:
        return tensor
    if isinstance(tensor, torch.Tensor):
        t = tensor.detach().cpu()
        if t.dim() == 3 and orig_shape is not None and len(orig_shape) == 2:
            if t.shape[0] != orig_shape[0]:
                return transdata_nz2nd(t, orig_shape, t.dtype)
        return t
    return tensor


def _to_int(val, default=0):
    """Safely convert val to int (may be tensor, list, or scalar)"""
    if isinstance(val, torch.Tensor):
        return int(val.item()) if val.numel() == 1 else int(val[0].item())
    if isinstance(val, (list, tuple)):
        return int(val[0]) if len(val) > 0 else default
    return int(val)

def _to_bool(val, default=True):
    """Safely convert val to bool"""
    if isinstance(val, torch.Tensor):
        return bool(val.item()) if val.numel() == 1 else bool(val[0].item())
    if isinstance(val, (list, tuple)):
        return bool(val[0]) if len(val) > 0 else default
    return bool(val)

def _to_float(val, default=1e-6):
    """Safely convert val to float"""
    if isinstance(val, torch.Tensor):
        return float(val.item()) if val.numel() == 1 else float(val[0].item())
    if isinstance(val, (list, tuple)):
        return float(val[0]) if len(val) > 0 else default
    return float(val)


def _compute_outputs(input1, gamma0, beta0, quantScale0, quantOffset0,
                     wdqkv, deScale0, bias0, gamma1, beta1,
                     quantScale1, quantOffset1, wuq, deScale1, bias1,
                     gamma2, cos, sin, wuk, kvCache, kvCacheRope,
                     slotmapping, ctkvScale, qNopeScale,
                     cacheMode, doRmsNorm, data_type, use_lowprec=False):
    """公共计算逻辑 — 高精度/低精度共用, 通过 use_lowprec 切换"""

    en_w8a8 = wdqkv.dtype == torch.int8

    if use_lowprec:
        _lp = data_type
        if en_w8a8:
            if doRmsNorm:
                mm1_in = rms_norm_quant_cpu_lowprec(input1, gamma0, beta0, quantScale0, quantOffset0, data_type=_lp)
            else:
                mm1_in = quant_cpu_lowprec(input1, quantScale0, quantOffset0, data_type=_lp)
            mm1_out = mm_w8a8_cpu_lowprec(mm1_in, wdqkv, deScale0, None, bias0, data_type=_lp).to(data_type)
        else:
            mm1_out = mm_cpu_lowprec(input1, wdqkv, data_type=_lp).to(data_type)
    else:
        if en_w8a8:
            if doRmsNorm:
                mm1_in = rms_norm_quant_cpu(input1, gamma0, beta0, quantScale0, quantOffset0)
            else:
                mm1_in = quant_cpu(input1, quantScale0, quantOffset0)
            mm1_out = mm_w8a8_cpu(mm1_in, wdqkv, deScale0, None, bias0).to(data_type)
        else:
            mm1_out = mm_cpu(input1, wdqkv).to(data_type)

    split1_out = [mm1_out[..., :512], mm1_out[..., 512:576], mm1_out[..., 576:]]

    if use_lowprec:
        _lp = data_type
        if en_w8a8:
            mm2_in = rms_norm_quant_cpu_lowprec(split1_out[2], gamma1, beta1, quantScale1, quantOffset1, data_type=_lp)
            mm2_out = mm_w8a8_cpu_lowprec(mm2_in, wuq, deScale1, None, bias1, data_type=_lp).to(data_type)
        else:
            mm2_in = rms_norm_cpu(split1_out[2], gamma1)
            mm2_out = mm_cpu_lowprec(mm2_in, wuq, data_type=_lp).to(data_type)
    else:
        if en_w8a8:
            mm2_in = rms_norm_quant_cpu(split1_out[2], gamma1, beta1, quantScale1, quantOffset1)
            mm2_out = mm_w8a8_cpu(mm2_in, wuq, deScale1, None, bias1).to(data_type)
        else:
            mm2_in = rms_norm_cpu(split1_out[2], gamma1)
            mm2_out = mm_cpu(mm2_in, wuq).to(data_type)

    input_token_num = input1.shape[0]
    head_num = wuk.shape[0]
    nope_dim = wuk.shape[1]
    rope_dim = 64
    mm2_out = mm2_out.reshape(input_token_num, head_num, nope_dim + rope_dim)
    split2_out = [mm2_out[..., :nope_dim], mm2_out[..., nope_dim:]]

    if use_lowprec:
        einsum_out = mm_einsum_cpu_lowprec(split2_out[0], wuk, data_type=data_type).to(data_type)
        q_rope_out = rope_cpu_lowprec(split2_out[1], cos, sin, data_type=data_type).to(data_type)
    else:
        einsum_out = mm_einsum_cpu(split2_out[0], wuk).to(data_type)
        q_rope_out = rope_cpu(split2_out[1], cos, sin).to(data_type)

    q_out = quant_cpu(einsum_out, qNopeScale[None, :, None].cpu()) if cacheMode == 2 else einsum_out

    rms3_out = rms_norm_cpu(split1_out[0], gamma2).to(data_type)
    if cacheMode == 2:
        key_out = quant_cpu(rms3_out, ctkvScale)
    else:
        key_out = rms3_out

    if use_lowprec:
        key_rope_out = rope_cpu_lowprec(
            split1_out[1].unsqueeze(dim=1), cos, sin, data_type=data_type
        ).to(data_type).squeeze(dim=1)
    else:
        key_rope_out = rope_cpu(
            split1_out[1].unsqueeze(dim=1), cos, sin
        ).to(data_type).squeeze(dim=1)

    kv_cache_out, kv_cache_rope_out = reshape_and_cache_cpu(
        key_out, key_rope_out, kvCache, kvCacheRope, slotmapping, cacheMode
    )

    if cacheMode > 1:
        # NPU 输出 NZ 格式: [BLOCK_NUM, last_dim//16, BLOCK_SIZE, 16]
        kv_cache_out = transdata_nd2nz(kv_cache_out.reshape((BLOCK_NUM, BLOCK_SIZE, -1)))
        kv_cache_rope_out = transdata_nd2nz(kv_cache_rope_out.reshape((BLOCK_NUM, BLOCK_SIZE, -1)))

    if cacheMode == 0:
        q_out_final = torch.cat((q_out, q_rope_out), dim=-1)
        q_rope_out_final = torch.zeros_like(q_out_final)
    elif cacheMode == 2:
        q_out_final = ein_sum_out_quant_golden(q_out, qNopeScale)
        q_rope_out_final = q_rope_out
    else:
        q_out_final = q_out
        q_rope_out_final = q_rope_out

    return [q_out_final, kv_cache_out, q_rope_out_final, kv_cache_rope_out]


# ==================== 公开接口 ====================

def cpu_mla_preprocess(input, gamma0, beta0, quantScale0, quantOffset0,
                       wdqkv, descale0, bias0, gamma1, beta1,
                       quantScale1, quantOffset1, wuq, descale1, bias1,
                       gamma2, cos, sin, wuk, kvCache, kvCacheRope,
                       slotMapping, ctkvScale, qNopeScale,
                       wdqDim, qRopeDim, kRopeDim, epsilon,
                       qRotaryCoeff, kRotaryCoeff, transeposeWdq, transeposeWuq,
                       transeposeWuk, cacheMode, quantMode, doRmsNorm, wdkvSplitCount,
                       qOutOut, kvCacheOutOut, qRopeOutOut, krCacheOutOut, **kwargs):
    """高精度 CPU golden — 使用 fp32 中间计算

    参数签名与 aclnnMlaPreprocessGetWorkspaceSize 一致 (含输出 tensor 占位)。
    返回 [qOut, kvCacheOut, qRopeOut, krCacheOut]。
    ATK 高精度路径中所有浮点输入实际是 fp32 (ATK 数据生成时把 fp16 转成了 fp32)。
    """
    _to_cpu = lambda t: t.detach().cpu() if isinstance(t, torch.Tensor) else t
    input1 = _to_cpu(input)
    gamma0 = _to_cpu(gamma0)
    beta0 = _to_cpu(beta0)
    quantScale0 = _to_cpu(quantScale0)
    quantOffset0 = _to_cpu(quantOffset0)
    wdqkv = _to_cpu(wdqkv)
    deScale0 = _restore_descale(_to_cpu(descale0))
    bias0 = _to_cpu(bias0)
    gamma1 = _to_cpu(gamma1)
    beta1 = _to_cpu(beta1)
    quantScale1 = _to_cpu(quantScale1)
    quantOffset1 = _to_cpu(quantOffset1)
    wuq = _to_cpu(wuq)
    deScale1 = _restore_descale(_to_cpu(descale1))
    bias1 = _to_cpu(bias1)
    gamma2 = _to_cpu(gamma2)
    cos = _to_cpu(cos)
    sin = _to_cpu(sin)
    wuk = _to_cpu(wuk)
    kvCache = _to_cpu(kvCache)
    kvCacheRope = _to_cpu(kvCacheRope)
    slotmapping = _to_cpu(slotMapping)
    ctkvScale = _to_cpu(ctkvScale)
    qNopeScale = _to_cpu(qNopeScale)

    # ATK 代码: self.data_type = self.input1.dtype (行 689)
    # 但 ATK .so 内部把 fp16 input 转成了 fp32 (运行时观察 golden 输出 fp32)
    # ATK .so 不可见, 无法从代码推断, 但运行时 golden=fp32, lowprec 有 fp16 和 fp32 两份
    # TTK 高精度 golden 需输出 fp32 (与 ATK result_cpu.bin 一致)
    # TTK 低精度 benchmark 保持 input1.dtype (与 ATK result_cpu_lowprec_float16.bin 一致)
    data_type = input1.dtype
    if data_type == torch.float16:
        _hp = torch.float32
        input1 = input1.to(_hp)
        gamma0 = gamma0.to(_hp)
        beta0 = beta0.to(_hp)
        quantScale0 = quantScale0.to(_hp)
        gamma1 = gamma1.to(_hp)
        beta1 = beta1.to(_hp)
        quantScale1 = quantScale1.to(_hp)
        gamma2 = gamma2.to(_hp)
        cos = cos.to(_hp)
        sin = sin.to(_hp)
        wuk = wuk.to(_hp)
        kvCache = kvCache.to(_hp)
        kvCacheRope = kvCacheRope.to(_hp)
        ctkvScale = ctkvScale.to(_hp)
        qNopeScale = qNopeScale.to(_hp)
        data_type = _hp
    # bf16: 保持原 dtype, 高精度路径中 .to(data_type) 截断为 bf16 (与 ATK 一致)
    # fp16: data_type=fp32, .to(fp32) 不截断 (与 ATK 一致)
    cacheMode = _to_int(cacheMode, 3)
    doRmsNorm = _to_bool(doRmsNorm, True)

    result = _compute_outputs(
        input1, gamma0, beta0, quantScale0, quantOffset0,
        wdqkv, deScale0, bias0, gamma1, beta1,
        quantScale1, quantOffset1, wuq, deScale1, bias1,
        gamma2, cos, sin, wuk, kvCache, kvCacheRope,
        slotmapping, ctkvScale, qNopeScale,
        cacheMode, doRmsNorm, data_type, use_lowprec=False
    )
    return result


def cpu_mla_preprocess_lowprec(input, gamma0, beta0, quantScale0, quantOffset0,
                               wdqkv, descale0, bias0, gamma1, beta1,
                               quantScale1, quantOffset1, wuq, descale1, bias1,
                               gamma2, cos, sin, wuk, kvCache, kvCacheRope,
                               slotMapping, ctkvScale, qNopeScale,
                               wdqDim, qRopeDim, kRopeDim, epsilon,
                               qRotaryCoeff, kRotaryCoeff, transeposeWdq, transeposeWuq,
                               transeposeWuk, cacheMode, quantMode, doRmsNorm, wdkvSplitCount,
                               qOutOut, kvCacheOutOut, qRopeOutOut, krCacheOutOut, **kwargs):
    """低精度 CPU golden — 匹配 NPU kernel 中间精度截断

    与 cpu_mla_preprocess 相同的参数签名和计算流程,
    但中间计算使用 data_type 截断 (匹配 NPU kernel 的 MMAD/Fixpipe 行为)。
    """
    _to_cpu = lambda t: t.detach().cpu() if isinstance(t, torch.Tensor) else t
    input1 = _to_cpu(input)
    gamma0 = _to_cpu(gamma0)
    beta0 = _to_cpu(beta0)
    quantScale0 = _to_cpu(quantScale0)
    quantOffset0 = _to_cpu(quantOffset0)
    wdqkv = _to_cpu(wdqkv)
    deScale0 = _restore_descale(_to_cpu(descale0))
    bias0 = _to_cpu(bias0)
    gamma1 = _to_cpu(gamma1)
    beta1 = _to_cpu(beta1)
    quantScale1 = _to_cpu(quantScale1)
    quantOffset1 = _to_cpu(quantOffset1)
    wuq = _to_cpu(wuq)
    deScale1 = _restore_descale(_to_cpu(descale1))
    bias1 = _to_cpu(bias1)
    gamma2 = _to_cpu(gamma2)
    cos = _to_cpu(cos)
    sin = _to_cpu(sin)
    wuk = _to_cpu(wuk)
    kvCache = _to_cpu(kvCache)
    kvCacheRope = _to_cpu(kvCacheRope)
    slotmapping = _to_cpu(slotMapping)
    ctkvScale = _to_cpu(ctkvScale)
    qNopeScale = _to_cpu(qNopeScale)

    # ATK 低精度路径: _lp_dtype = self.data_type
    # 高精度和低精度用同一个 data_type, 区别在于低精度函数内部有 .to(data_type).float() 截断
    # TTK 低精度路径: data_type = input1.dtype (与高精度路径一致)
    data_type = input1.dtype
    cacheMode = _to_int(cacheMode, 3)
    doRmsNorm = _to_bool(doRmsNorm, True)

    result = _compute_outputs(
        input1, gamma0, beta0, quantScale0, quantOffset0,
        wdqkv, deScale0, bias0, gamma1, beta1,
        quantScale1, quantOffset1, wuq, deScale1, bias1,
        gamma2, cos, sin, wuk, kvCache, kvCacheRope,
        slotmapping, ctkvScale, qNopeScale,
        cacheMode, doRmsNorm, data_type, use_lowprec=True
    )
    return result
