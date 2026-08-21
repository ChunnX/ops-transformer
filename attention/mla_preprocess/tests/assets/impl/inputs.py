"""MlaPreprocess customize_inputs — 从 ATK executor.py init_by_input_data 移植

参数签名与 aclnnMlaPreprocessGetWorkspaceSize 一致 (含输出 tensor 占位)。
in-place 修改 tensor，不返回值。

处理项:
  1. slotmapping 随机生成 (np.random.RandomState(2025))
  2. deScale0/deScale1 process_deq_scale (fp32→int32→int64, 仅 fp16 输入)
  3. wdqkv/wuq NZ 格式转换
  4. kvCache/kvCacheRope NZ 格式转换 (cacheMode > 1)
"""

import importlib.util
import os

import numpy as np
import torch
import torch_npu

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_golden():
    path = os.path.join(_THIS_DIR, "golden.py")
    spec = importlib.util.spec_from_file_location("_mla_golden_for_inputs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_golden_mod = _load_golden()
transdata_nd2nz = _golden_mod.transdata_nd2nz


def _process_deq_scale(deq_scale):
    """fp32 → int32 reinterpret → int64"""
    ret = torch.frombuffer(deq_scale.cpu().numpy().tobytes(), dtype=torch.int32).to(torch.int64)
    return ret


def _to_nz_inplace(tensor):
    """将 tensor 转为 NZ 格式 (in-place) — 仅在 NPU 设备上有效

    TTK ACLNN 模式下 customize_inputs 接收 CPU tensor,
    NZ 格式转换通过 CSV tensor_formats 字段由框架处理。
    此函数仅在 tensor 已在 NPU 上时执行转换。
    """
    if tensor is None:
        return
    if tensor.device.type != 'npu':
        # CPU tensor: skip NZ conversion, framework handles via tensor_formats
        return
    cpu_data = tensor.detach().cpu()
    nz_data = transdata_nd2nz(cpu_data).contiguous()
    nz_view = torch_npu.npu_format_cast(tensor, 29)
    nz_view.copy_(nz_data.to(tensor.device))


def customize_inputs(input, gamma0, beta0, quantScale0, quantOffset0,
                     wdqkv, descale0, bias0, gamma1, beta1,
                     quantScale1, quantOffset1, wuq, descale1, bias1,
                     gamma2, cos, sin, wuk, kvCache, kvCacheRope,
                     slotMapping, ctkvScale, qNopeScale,
                     wdqDim, qRopeDim, kRopeDim, epsilon,
                     qRotaryCoeff, kRotaryCoeff, transeposeWdq, transeposeWuq,
                     transeposeWuk, cacheMode, quantMode, doRmsNorm, wdkvSplitCount,
                     qOutOut, kvCacheOutOut, qRopeOutOut, krCacheOutOut, **kwargs):
    """定制输入修正 — 参数签名与 aclnnMlaPreprocessGetWorkspaceSize 一致

    必须 in-place 修改 (用 copy_()), 不返回值。
    """

    # Safely extract cacheMode value (may be tensor, list, or scalar)
    cm = cacheMode
    if isinstance(cm, torch.Tensor):
        cm = cm.item() if cm.numel() == 1 else int(cm[0].item())
    elif isinstance(cm, (list, tuple)):
        cm = cm[0] if len(cm) > 0 else 3
    cacheMode_val = int(cm)
    testcase_name = kwargs.get('testcase_name', '')
    print(f"[CUSTOMIZE-MLA] testcase_name={testcase_name}, cacheMode={cacheMode_val}", flush=True)

    # 1. slotmapping 随机生成 (与 ATK set_seeds(2025) + np.random.choice 一致)
    input_token_num = input.shape[0]
    _slot_rng = np.random.RandomState(2025)
    new_slotmapping = torch.from_numpy(
        _slot_rng.choice(192 * 128, input_token_num, replace=False).astype(np.int32)
    ).to(torch.int32)
    slotMapping.copy_(new_slotmapping.to(slotMapping.device))
    print(f"[SLOT-GEN-MLA] token_num={input_token_num}", flush=True)

    # 2. deScale0/deScale1 process_deq_scale (仅 fp16 输入)
    # 需要将 fp32 tensor reinterpret 为 int64 (不改数据, 只改 dtype view)
    # copy_() 不能改 dtype, 必须用 view + storage 替换
    if input.dtype == torch.float16:
        descale0_cpu = descale0.detach().cpu()
        deScale0_int64 = torch.frombuffer(descale0_cpu.numpy().tobytes(), dtype=torch.int32).to(torch.int64)
        descale0.data = deScale0_int64.to(descale0.device).contiguous()
        descale1_cpu = descale1.detach().cpu()
        deScale1_int64 = torch.frombuffer(descale1_cpu.numpy().tobytes(), dtype=torch.int32).to(torch.int64)
        descale1.data = deScale1_int64.to(descale1.device).contiguous()
        print(f"[DESCALE-MLA] deScale0 dtype after data=: {descale0.dtype}", flush=True)
        print(f"[DESCALE-MLA] deScale1 dtype after data=: {descale1.dtype}", flush=True)

    # 3. wdqkv/wuq NZ 格式转换
    _to_nz_inplace(wdqkv)
    _to_nz_inplace(wuq)

    # 4. kvCache/kvCacheRope NZ 格式转换 (cacheMode > 1)
    if cacheMode_val > 1:
        _to_nz_inplace(kvCache)
        _to_nz_inplace(kvCacheRope)
        print(f"[NZ-MLA] kvCache/kvCacheRope NZ conversion requested (cacheMode={cacheMode_val})", flush=True)
