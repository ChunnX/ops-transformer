"""Independent ATK accuracy plugin for BSA V3 HIF8 full quantization."""

import ctypes
import math
import os
import struct
import sys
import zlib
from typing import List, Sequence, Tuple

import torch

from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import (
    AclDataType, AclFormat, AclIntArray, AclTensorStruct, Int64, TensorPtr,
    TORCH_TO_ACLTYPE, nnopbase, ascendcl,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import hif8_encode_decode

logging = Logger().get_logger()
HIF8_MAX = 32768.0
Q_BASE_TILE = 128
KV_BASE_TILE = 256
MIN_FLOAT = -3.402823466e38
QUANT_MODE_HIF8 = 5
NOT_LIMITED = 2147483647
SPARSITY_RATIO_CHOICES = (1.0, 1.0, 0.9, 0.8, 0.75, 0.5, 0.4, 0.25, 0.1)
DATA_RANGE_CHOICES = (0.77, 0.91, 1.0, 2.0, 5.0)
P_SCALE_CHOICES = (1.0, 2.0, 7.0, 16.0, 32.0, 64.0)


def _f32(value) -> float:
    if isinstance(value, torch.Tensor):
        value = value.item()
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _f32_mul(lhs, rhs) -> float:
    return _f32(_f32(lhs) * _f32(rhs))


def mm1_dequant_scale(q_scale, k_scale, attention_scale) -> float:
    """Match ``float q * float k * float attentionScale``.

    The MM1 FixPipe runs with NO_QUANT (hardware dual-destination mode) and the
    online-softmax epilogue applies this scale instead, so the 19-bit scalar
    truncation no longer exists.
    """
    return _f32_mul(_f32_mul(q_scale, k_scale), attention_scale)


def quantize_hif8(src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    src = src.to(torch.float32).contiguous()
    max_abs = float(src.abs().max()) if src.numel() else 0.0
    scale = max_abs / HIF8_MAX if max_abs else 1.0
    encoded = hif8_encode_decode.trans_float_tensor_to_hifuint8(
        src / scale, round_mode="round", over_mode=True
    )
    return encoded.contiguous(), torch.tensor([scale], dtype=torch.float32)


def decode_hif8(src: torch.Tensor) -> torch.Tensor:
    return hif8_encode_decode.trans_hifuint8_tensor_to_float(
        src.to(torch.uint8).contiguous(), over_mode=True
    ).to(torch.float32)


def cast_hif8(src: torch.Tensor) -> torch.Tensor:
    bits = hif8_encode_decode.trans_float_tensor_to_hifuint8(
        src.to(torch.float32).contiguous(), round_mode="round", over_mode=True
    )
    return decode_hif8(bits)


def _contiguous_strides(shape: Sequence[int]) -> List[int]:
    strides = [1] * len(shape)
    for idx in range(len(shape) - 2, -1, -1):
        strides[idx] = strides[idx + 1] * int(shape[idx + 1])
    return strides


def _create_acl_tensor(npu_tensor: torch.Tensor, acl_dtype, logical_shape) -> AclTensorStruct:
    npu_tensor = npu_tensor.contiguous()
    if npu_tensor.device.type != "npu":
        npu_tensor = npu_tensor.npu()
    logical_shape = tuple(int(x) for x in logical_shape)
    ndim = len(logical_shape)
    dims = (Int64 * ndim)(*logical_shape)
    strides = (Int64 * ndim)(*_contiguous_strides(logical_shape))
    addr = ctypes.c_void_p(npu_tensor.data_ptr())
    acl_tensor = nnopbase.aclCreateTensor(
        dims, ndim, acl_dtype, strides, 0, AclFormat.ACL_FORMAT_ND,
        dims, ndim, addr,
    )
    if not acl_tensor:
        raise RuntimeError(f"aclCreateTensor failed: dtype={acl_dtype}, shape={logical_shape}")
    return AclTensorStruct(
        acl_tensor, addr.value, npu_tensor, ascendcl.aclDataTypeSize(acl_dtype)
    )


def create_hif8_acl_tensor(storage: torch.Tensor) -> AclTensorStruct:
    if not hasattr(AclDataType, "ACL_HIFLOAT8"):
        raise RuntimeError("this ATK/CANN acl wrapper does not expose ACL_HIFLOAT8")
    storage = storage.to(torch.uint8).contiguous().npu()
    return _create_acl_tensor(storage, AclDataType.ACL_HIFLOAT8, storage.shape)


def create_acl_tensor(tensor: torch.Tensor) -> AclTensorStruct:
    tensor = tensor.contiguous().npu()
    return _create_acl_tensor(tensor, TORCH_TO_ACLTYPE[str(tensor.dtype)], tensor.shape)


def create_int64_array(values):
    if values is None:
        return ctypes.POINTER(AclIntArray)()
    return nnopbase.create_x_list([Int64(int(v)) for v in values])


def _case_seed(*values) -> int:
    return zlib.crc32(repr(values).encode("utf-8")) & 0x7FFFFFFF


def _as_lengths(raw, default: int, batch: int) -> Tuple[int, ...]:
    if raw is None or (isinstance(raw, str) and raw == "None"):
        return tuple([int(default)] * batch)
    values = tuple(int(x) for x in raw)
    if len(values) != batch:
        raise ValueError(f"actual sequence length count {len(values)} != batch {batch}")
    return values


def _bnsd_to_layout(tensor: torch.Tensor, layout: str, lengths) -> torch.Tensor:
    if layout == "BNSD":
        return tensor.contiguous()
    bsnd = tensor.permute(0, 2, 1, 3).contiguous()
    if layout == "BSND":
        return bsnd
    return torch.cat(
        [bsnd[b, :int(lengths[b])] for b in range(tensor.shape[0])], dim=0
    ).contiguous()


def _lse_to_layout(tensor: torch.Tensor, layout: str, lengths) -> torch.Tensor:
    if layout == "BNSD":
        return tensor.unsqueeze(-1).contiguous()
    if layout == "BSND":
        return tensor.permute(0, 2, 1).unsqueeze(-1).contiguous()
    return torch.cat(
        [tensor[b, :, :int(lengths[b])].transpose(0, 1) for b in range(tensor.shape[0])],
        dim=0,
    ).unsqueeze(-1).contiguous()


def build_sparse_mask(batch, heads, q_storage, kv_storage, q_lengths, kv_lengths,
                      block_shape, ratio, seed, layout):
    bx, by = (int(x) for x in block_shape)
    mask_q = max(q_lengths) if layout == "TND" else q_storage
    mask_kv = max(kv_lengths) if layout == "TND" else kv_storage
    cq = (mask_q + bx - 1) // bx
    ck = (mask_kv + by - 1) // by
    mask = torch.zeros((batch, heads, cq, ck), dtype=torch.int8)
    gen = torch.Generator().manual_seed(seed)
    for b in range(batch):
        valid_cq = (q_lengths[b] + bx - 1) // bx
        valid_ck = (kv_lengths[b] + by - 1) // by
        keep = max(1, math.ceil(valid_ck * ratio))
        for h in range(heads):
            for qb in range(valid_cq):
                chosen = torch.randperm(valid_ck, generator=gen)[:keep]
                mask[b, h, qb, chosen] = 1
    return mask


def _gather_positions(mask_row, kv_len, block_y):
    result = []
    for block_id, enabled in enumerate(mask_row.tolist()):
        if enabled:
            begin = block_id * block_y
            result.extend(range(begin, min(begin + block_y, kv_len)))
    return result


def _mm2_tiled(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out = torch.zeros((p.shape[0], v.shape[1]), dtype=torch.float32)
    for begin in range(0, p.shape[1], 128):
        end = min(begin + 128, p.shape[1])
        out = (out + torch.matmul(p[:, begin:end], v[begin:end])).to(torch.float32)
    return out


def hif8_attention_block(q_raw, k_raw, v_raw, q_scale, k_scale, v_scale,
                         p_scale, attention_scale):
    mm1_scale = mm1_dequant_scale(q_scale, k_scale, attention_scale)
    mm2_scale = _f32(v_scale)
    ln_p_scale = torch.log(torch.tensor(_f32(p_scale), dtype=torch.float32))
    out = torch.zeros((q_raw.shape[0], v_raw.shape[1]), dtype=torch.float32)
    lse = torch.full((q_raw.shape[0],), float("inf"), dtype=torch.float32)
    for q0 in range(0, q_raw.shape[0], Q_BASE_TILE):
        q1 = min(q0 + Q_BASE_TILE, q_raw.shape[0])
        q = q_raw[q0:q1]
        running_max = torch.full((q.shape[0],), MIN_FLOAT, dtype=torch.float32)
        running_sum = torch.zeros((q.shape[0],), dtype=torch.float32)
        running_out = torch.zeros((q.shape[0], v_raw.shape[1]), dtype=torch.float32)
        for k0 in range(0, k_raw.shape[0], KV_BASE_TILE):
            k1 = min(k0 + KV_BASE_TILE, k_raw.shape[0])
            raw_s = torch.matmul(k_raw[k0:k1], q.transpose(0, 1)).transpose(0, 1)
            scores = (raw_s * mm1_scale).to(torch.float32)
            tile_max = scores.max(dim=-1).values.to(torch.float32)
            new_max = torch.maximum(running_max, tile_max).to(torch.float32)
            exp_max = torch.exp(running_max - new_max).to(torch.float32)
            p_fp32 = torch.exp(scores - (new_max - ln_p_scale)[:, None]).to(torch.float32)
            new_sum = (exp_max * running_sum + p_fp32.sum(dim=-1)).to(torch.float32)
            p_hif8 = cast_hif8(p_fp32)
            # MM2 FixPipe runs with NO_QUANT: UB holds the raw P @ V, and the V dequant
            # scale is applied by the rescale-O epilogue on the normalized output.
            tile_out = _mm2_tiled(p_hif8, v_raw[k0:k1]).to(torch.float32)
            running_out = (exp_max[:, None] * running_out + tile_out).to(torch.float32)
            running_max, running_sum = new_max, new_sum
        out[q0:q1] = (running_out / running_sum[:, None] * mm2_scale).to(torch.float32)
        lse[q0:q1] = running_max + torch.log(running_sum) - ln_p_scale
    return out, lse


def bsa_hif8_reference(q_bits, k_bits, v_bits, mask, block_shape, q_lengths,
                       kv_lengths, scales, attention_scale, out_dtype, layout):
    """Reference branch matching ``fp8_test_bsa.py::ref_attention``.

    The ATK CPU node is called twice: the ordinary task is the quantized
    benchmark path above, while the benchmark task must use the anti-quantized
    Q/K/V values and a plain block-sparse attention calculation.  HIF8 storage
    is decoded first and the corresponding dequant scale is applied, exactly
    like ``q_blk_antiquant = q_blk.astype(float32) * q_deq_scale`` in the
    reference ATK implementation.
    """
    q_raw = decode_hif8(q_bits).to(torch.float32) * _f32(scales[0])
    k_raw = decode_hif8(k_bits).to(torch.float32) * _f32(scales[1])
    v_raw = decode_hif8(v_bits).to(torch.float32) * _f32(scales[2])
    batch, q_heads, q_storage, dim = q_raw.shape
    kv_heads = k_raw.shape[1]
    group = q_heads // kv_heads
    bx, by = (int(x) for x in block_shape)
    out = torch.zeros((batch, q_heads, q_storage, dim), dtype=torch.float32)
    lse = torch.zeros((batch, q_heads, q_storage), dtype=torch.float32)
    for b in range(batch):
        for qh in range(q_heads):
            kvh = qh // group
            for qb in range((q_lengths[b] + bx - 1) // bx):
                positions = _gather_positions(mask[b, qh, qb], kv_lengths[b], by)
                index = torch.tensor(positions, dtype=torch.long)
                q0, q1 = qb * bx, min((qb + 1) * bx, q_lengths[b])
                q = q_raw[b, qh, q0:q1]
                k = k_raw[b, kvh].index_select(0, index)
                v = v_raw[b, kvh].index_select(0, index)
                scores = (torch.matmul(q, k.transpose(0, 1)) *
                          _f32(attention_scale)).to(torch.float32)
                row_lse = torch.logsumexp(scores, dim=-1)
                p = torch.softmax(scores, dim=-1).to(torch.float32)
                out[b, qh, q0:q1] = torch.matmul(p, v)
                lse[b, qh, q0:q1] = row_lse
    return (
        _bnsd_to_layout(out.to(out_dtype), layout, q_lengths).to(torch.float32),
        _lse_to_layout(lse, layout, q_lengths).to(torch.float32),
    )


def bsa_hif8_golden(q_bits, k_bits, v_bits, mask, block_shape, q_lengths,
                    kv_lengths, scales, p_scale, attention_scale, out_dtype, layout):
    q_raw, k_raw, v_raw = decode_hif8(q_bits), decode_hif8(k_bits), decode_hif8(v_bits)
    batch, q_heads, q_storage, dim = q_raw.shape
    kv_heads = k_raw.shape[1]
    group = q_heads // kv_heads
    bx, by = (int(x) for x in block_shape)
    out = torch.zeros((batch, q_heads, q_storage, dim), dtype=torch.float32)
    lse = torch.full((batch, q_heads, q_storage), float("inf"), dtype=torch.float32)
    for b in range(batch):
        for qh in range(q_heads):
            kvh = qh // group
            for qb in range((q_lengths[b] + bx - 1) // bx):
                positions = _gather_positions(mask[b, qh, qb], kv_lengths[b], by)
                index = torch.tensor(positions, dtype=torch.long)
                q0, q1 = qb * bx, min((qb + 1) * bx, q_lengths[b])
                block_out, block_lse = hif8_attention_block(
                    q_raw[b, qh, q0:q1], k_raw[b, kvh].index_select(0, index),
                    v_raw[b, kvh].index_select(0, index), scales[0], scales[1],
                    scales[2], p_scale, attention_scale,
                )
                out[b, qh, q0:q1] = block_out
                lse[b, qh, q0:q1] = block_lse
        # The kernel does not define padded rows.  Normalize both references
        # to zero so ATK compares only the semantically valid prefix.
        lse[b, :, q_lengths[b]:] = 0
    return (
        _bnsd_to_layout(out.to(out_dtype), layout, q_lengths).to(torch.float32),
        _lse_to_layout(lse, layout, q_lengths).to(torch.float32),
    )


class Hif8Prepared:
    def prepare(self, input_data: InputDataset):
        kw = input_data.kwargs
        self.layout, self.kv_layout = str(kw["qInputLayout"]), str(kw["kvInputLayout"])
        if self.layout != self.kv_layout or self.layout not in ("BNSD", "BSND", "TND"):
            raise ValueError("HIF8 ATK requires matching BNSD/BSND/TND Q/KV layouts")
        q_desc, k_desc = kw["query"], kw["key"]
        self.kv_heads = int(kw["numKeyValueHeads"])
        self.block_shape = tuple(int(x) for x in kw["blockShape"])
        self.scale_value = float(kw["scaleValue"])
        self.inner_precise = int(kw["innerPrecise"])
        self.softmax_lse_flag = int(kw["softmaxLseFlag"])
        self.quant_mode = int(kw["quantMode"])
        self.dst_type_max = float(kw["dstTypeMax"])
        if self.quant_mode != QUANT_MODE_HIF8:
            raise ValueError(f"HIF8 ATK requires quantMode=5, got {self.quant_mode}")
        if self.layout == "TND":
            self.batch = len(kw["actualSeqLengths"])
            self.q_heads, self.dim = int(q_desc.shape[1]), int(q_desc.shape[2])
            q_storage = max(int(x) for x in kw["actualSeqLengths"])
            kv_storage = max(int(x) for x in kw["actualSeqLengthsKv"])
        elif self.layout == "BNSD":
            self.batch, self.q_heads = int(q_desc.shape[0]), int(q_desc.shape[1])
            q_storage, self.dim = int(q_desc.shape[2]), int(q_desc.shape[3])
            kv_storage = int(k_desc.shape[2])
        else:
            self.batch, q_storage = int(q_desc.shape[0]), int(q_desc.shape[1])
            self.q_heads, self.dim = int(q_desc.shape[2]), int(q_desc.shape[3])
            kv_storage = int(k_desc.shape[1])
        if self.dim not in (64, 128) or self.q_heads % self.kv_heads:
            raise ValueError("HIF8 quantMode=5 requires D in {64, 128} and Nq divisible by Nkv")
        self.q_storage, self.kv_storage = q_storage, kv_storage
        self.q_lengths = _as_lengths(kw.get("actualSeqLengths"), q_storage, self.batch)
        self.kv_lengths = _as_lengths(kw.get("actualSeqLengthsKv"), kv_storage, self.batch)
        if (any(x <= 0 or x > q_storage for x in self.q_lengths) or
                any(x <= 0 or x > kv_storage for x in self.kv_lengths)):
            raise ValueError("actual sequence lengths must be positive and within storage shapes")
        if (self.block_shape[0] <= 0 or self.block_shape[1] <= 0 or
                self.block_shape[0] % 16 or self.block_shape[1] % 16):
            raise ValueError("HIF8 blockShape X/Y must be positive multiples of 16")
        self.out_dtype = torch.bfloat16 if kw["attentionOut"].dtype == torch.bfloat16 else torch.float16
        seed = _case_seed(self.layout, self.batch, self.q_heads, self.kv_heads,
                          self.q_lengths, self.kv_lengths, self.block_shape,
                          self.out_dtype, self.scale_value)
        amp = DATA_RANGE_CHOICES[(seed // 7) % len(DATA_RANGE_CHOICES)]
        p_value = P_SCALE_CHOICES[(seed // 17) % len(P_SCALE_CHOICES)]
        ratio = SPARSITY_RATIO_CHOICES[seed % len(SPARSITY_RATIO_CHOICES)]
        gen = torch.Generator().manual_seed(seed)
        q_src = (torch.rand((self.batch, self.q_heads, q_storage, self.dim), generator=gen) * 2 - 1) * amp
        k_src = (torch.rand((self.batch, self.kv_heads, kv_storage, self.dim), generator=gen) * 2 - 1) * amp
        v_src = (torch.rand((self.batch, self.kv_heads, kv_storage, self.dim), generator=gen) * 2 - 1) * amp
        for b in range(self.batch):
            q_src[b, :, self.q_lengths[b]:] = 0
            k_src[b, :, self.kv_lengths[b]:] = 0
            v_src[b, :, self.kv_lengths[b]:] = 0
        self.q_bits_bnsd, self.q_scale = quantize_hif8(q_src)
        self.k_bits_bnsd, self.k_scale = quantize_hif8(k_src)
        self.v_bits_bnsd, self.v_scale = quantize_hif8(v_src)
        self.q_bits = _bnsd_to_layout(self.q_bits_bnsd, self.layout, self.q_lengths)
        self.k_bits = _bnsd_to_layout(self.k_bits_bnsd, self.layout, self.kv_lengths)
        self.v_bits = _bnsd_to_layout(self.v_bits_bnsd, self.layout, self.kv_lengths)
        self.p_scale = torch.tensor([p_value], dtype=torch.float32)
        self.mask = build_sparse_mask(self.batch, self.q_heads, q_storage, kv_storage,
                                      self.q_lengths, self.kv_lengths, self.block_shape,
                                      ratio, seed, self.layout)
        logging.info(f"BSA HIF8 seed={seed} layout={self.layout} B={self.batch} "
                     f"Nq={self.q_heads} Nkv={self.kv_heads} q={self.q_lengths} "
                     f"kv={self.kv_lengths} block={self.block_shape} ratio={ratio} p={p_value}")

    def out_shape(self):
        if self.layout == "TND":
            return (sum(self.q_lengths), self.q_heads, self.dim)
        if self.layout == "BNSD":
            return (self.batch, self.q_heads, self.q_storage, self.dim)
        return (self.batch, self.q_storage, self.q_heads, self.dim)

    def lse_shape(self):
        if self.layout == "TND":
            return (sum(self.q_lengths), self.q_heads, 1)
        if self.layout == "BNSD":
            return (self.batch, self.q_heads, self.q_storage, 1)
        return (self.batch, self.q_storage, self.q_heads, 1)

    def golden(self, is_benchmark=False):
        scales = (self.q_scale.item(), self.k_scale.item(), self.v_scale.item())
        if is_benchmark:
            return bsa_hif8_reference(
                self.q_bits_bnsd, self.k_bits_bnsd, self.v_bits_bnsd, self.mask,
                self.block_shape, self.q_lengths, self.kv_lengths,
                scales, self.scale_value, self.out_dtype, self.layout,
            )
        return bsa_hif8_golden(
            self.q_bits_bnsd, self.k_bits_bnsd, self.v_bits_bnsd, self.mask,
            self.block_shape, self.q_lengths, self.kv_lengths, scales,
            self.p_scale.item(), self.scale_value, self.out_dtype, self.layout,
        )


@register("aclnn_blocksparseattention_v3_hif8")
class BlockSparseAttentionV3Hif8Api(BaseApi, Hif8Prepared):
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)

    def init_by_input_data(self, input_data: InputDataset):
        self.prepare(input_data)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        out, lse = self.golden(self.task_result.is_benchmark_task)
        return (out, lse) if self.softmax_lse_flag else out


@register("pyaclnn_aclnn_blocksparseattention_v3_hif8")
class PyAclnnBlockSparseAttentionV3Hif8Api(AclnnBaseApi):
    def _get_prepared(self, input_data):
        cached = getattr(self.backend, "api_executor", None)
        if isinstance(cached, BlockSparseAttentionV3Hif8Api) and hasattr(cached, "q_bits"):
            return cached
        prep = BlockSparseAttentionV3Hif8Api(self.task_result)
        prep.init_by_input_data(input_data)
        return prep

    def init_by_input_data(self, input_data: InputDataset):
        p = self._get_prepared(input_data)
        self.prep = p
        # These tensors are output buffers and are fully written by aclnn BSA.
        # ``torch.zeros(..., device="npu")`` launches the unrelated ZerosLike
        # operator, which is unavailable with the user-readable OPP overlay on
        # this host.  Empty allocation needs no auxiliary kernel; padded tails
        # are normalized on CPU in ``after_call`` below.
        self.out_npu = torch.empty(p.out_shape(), dtype=p.out_dtype, device="npu")
        out_pack = create_acl_tensor(self.out_npu)
        if p.softmax_lse_flag:
            self.lse_npu = torch.empty(p.lse_shape(), dtype=torch.float32, device="npu")
            lse_pack = create_acl_tensor(self.lse_npu)
        else:
            self.lse_npu, lse_pack = None, TensorPtr()
        args = [
            create_hif8_acl_tensor(p.q_bits), create_hif8_acl_tensor(p.k_bits),
            create_hif8_acl_tensor(p.v_bits), create_acl_tensor(p.mask), TensorPtr(),
            create_int64_array(p.block_shape), create_int64_array(p.q_lengths),
            create_int64_array(p.kv_lengths), TensorPtr(), create_acl_tensor(p.q_scale),
            create_acl_tensor(p.k_scale), create_acl_tensor(p.v_scale),
            create_acl_tensor(p.p_scale), ctypes.c_char_p(p.layout.encode()),
            ctypes.c_char_p(p.kv_layout.encode()), Int64(p.kv_heads), Int64(0),
            ctypes.c_double(p.scale_value), Int64(p.inner_precise), Int64(0),
            Int64(NOT_LIMITED), Int64(NOT_LIMITED), Int64(p.softmax_lse_flag),
            Int64(QUANT_MODE_HIF8), ctypes.c_double(p.dst_type_max), out_pack, lse_pack,
        ]
        torch.npu.synchronize()
        return args, [out_pack, lse_pack] if p.softmax_lse_flag else [out_pack]

    def after_call(self, output_packages):
        p = self.prep
        result = [self.out_npu.cpu().to(torch.float32)]
        if p.softmax_lse_flag:
            result.append(self.lse_npu.cpu().to(torch.float32))
        if p.layout != "TND":
            for b, valid_q in enumerate(p.q_lengths):
                if p.layout == "BNSD":
                    result[0][b, :, valid_q:] = 0
                    if p.softmax_lse_flag:
                        result[1][b, :, valid_q:] = 0
                else:
                    result[0][b, valid_q:] = 0
                    if p.softmax_lse_flag:
                        result[1][b, valid_q:] = 0
        return result

    def get_cpp_func_signature_type(self):
        return ("aclnnStatus aclnnBlockSparseAttentionV3GetWorkspaceSize("
                "const aclTensor *query, const aclTensor *key, const aclTensor *value, "
                "const aclTensor *blockSparseMaskOptional, const aclTensor *attenMaskOptional, "
                "const aclIntArray *blockShapeOptional, const aclIntArray *actualSeqLengthsOptional, "
                "const aclIntArray *actualSeqLengthsKvOptional, const aclTensor *blockTableOptional, "
                "const aclTensor *qDequantScaleOptional, const aclTensor *kDequantScaleOptional, "
                "const aclTensor *vDequantScaleOptional, const aclTensor *pQuantScaleOptional, "
                "char *qInputLayout, char *kvInputLayout, int64_t numKeyValueHeads, int64_t maskType, "
                "double scaleValue, int64_t innerPrecise, int64_t blockSize, int64_t preTokens, "
                "int64_t nextTokens, int64_t softmaxLseFlag, int64_t quantMode, double dstTypeMax, "
                "aclTensor *attentionOut, aclTensor *softmaxLseOptional, uint64_t *workspaceSize, "
                "aclOpExecutor **executor)")
