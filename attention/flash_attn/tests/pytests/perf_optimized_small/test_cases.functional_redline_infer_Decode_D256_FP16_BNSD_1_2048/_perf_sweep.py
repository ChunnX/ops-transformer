import sys, os
_pytests_dir = '/workspace/ops-transformer-d256-opt/attention/flash_attn/tests/pytests'
sys.path.insert(0, _pytests_dir)
import torch, torch_npu
torch.npu.set_device(0)
from core.data import flash_attn_inputs
from cann_ops_transformer.ops import flash_attn, flash_attn_metadata
device = torch.device('npu:0')
c = {'B': 1, 'N1': 5, 'N2': 1, 'S1': 1, 'S2': 2048, 'D': 256, 'layout_q': 'BNSD', 'layout_kv': 'BNSD', 'layout_out': 'BNSD', 'Dtype': 'fp16', 'q_range': [-10, 10], 'k_range': [-10, 10], 'v_range': [-10, 10], 'scale': 0.0625, 'mask_mode': 0, 'return_softmax_lse': 0, 'input_layout': 'BNSD', 'DV': 256, 'DRope': 0, 'win_left': 2147483647, 'win_right': 2147483647}
print('CASE test_cases.functional_redline_infer/Decode_D256_FP16_BNSD_1_2048  cold  runs=10')

def _to_int32_tensor(v):
    if v is None: return None
    return torch.tensor(v, dtype=torch.int32, device=device)

layout_q = c.get('layout_q', c.get('input_layout', 'BNSD'))
layout_kv = c.get('layout_kv', layout_q)
cu_q = c.get('cu_seqlens_q')
cu_kv = c.get('cu_seqlens_kv')
sq = c.get('seqused_q') or c.get('actual_seq_qlen')
skv = c.get('seqused_kv') or c.get('actual_seq_kvlen')
cu_q_t = _to_int32_tensor(cu_q)
cu_kv_t = _to_int32_tensor(cu_kv)
sq_t = _to_int32_tensor(sq)
skv_t = _to_int32_tensor(skv)
if layout_q == 'TND' and cu_q_t is not None:
    _bs = cu_q_t.shape[0] - 1
else:
    _bs = c.get('B', 1)
_msq = c.get('max_seqlen_q')
if _msq is None:
    _msq = max(sq) if sq else (int((cu_q_t[1:]-cu_q_t[:-1]).max()) if cu_q_t is not None else c.get('S1', 1))
else:
    _msq = int(_msq)
_msk = c.get('max_seqlen_kv')
if _msk is None:
    _msk = max(skv) if skv else (int((cu_kv_t[1:]-cu_kv_t[:-1]).max()) if cu_kv_t is not None else c.get('S2', c.get('S1', 1)))
else:
    _msk = int(_msk)

meta = flash_attn_metadata(
    cu_seqlens_q=cu_q_t, cu_seqlens_kv=cu_kv_t,
    seqused_q=sq_t, seqused_kv=skv_t,
    num_heads_q=c['N1'], num_heads_kv=c.get('N2',c['N1']), head_dim=c['D'],
    batch_size=_bs, max_seqlen_q=_msq, max_seqlen_kv=_msk,
    mask_mode=c.get('mask_mode',0), win_left=-1, win_right=-1,
    layout_q=layout_q, layout_kv=layout_kv, layout_out=c.get('layout_out',layout_q),
)
if c.get('mask_mode', 0) == 3:
    _m = torch.triu(torch.ones(2048, 2048), diagonal=1).to(dtype=torch.int8, device=device)
else:
    _m = None
for _ in range(10):
    inputs = flash_attn_inputs.generate(c, device=device, layout=layout_q, layout_kv=layout_kv)
    bt = c.get('block_table')
    bt_t = None
    if bt is not None and isinstance(bt, list) and str(layout_kv).startswith('PA_'):
        bt_t = torch.tensor(bt, dtype=torch.int32, device=device)
        if bt_t.dim() == 1:
            bt_t = bt_t.unsqueeze(0)
    elif str(layout_kv).startswith('PA_'):
        print(f'WARN: block_table missing for PA case, bt={type(bt)}')
    inputs['block_table'] = bt_t
    q, k, v = inputs['q'], inputs['k'], inputs['v']
    bt_dev = inputs.get('block_table')
    _f1 = torch.randn(32, 32768, 1, 128, dtype=q.dtype, device=device)
    _f2 = torch.randn(32, 32768, 1, 128, dtype=q.dtype, device=device)
    torch.npu.synchronize()
    flash_attn(q, k, v, block_table=bt_dev, sinks=None, attn_mask=_m,
        metadata=meta, cu_seqlens_q=cu_q_t, cu_seqlens_kv=cu_kv_t,
        seqused_q=sq_t, seqused_kv=skv_t,
        softmax_scale=1/(c['D']**0.5), mask_mode=c.get('mask_mode',0),
        win_left=-1, win_right=-1, max_seqlen_q=_msq, max_seqlen_kv=_msk,
        layout_q=layout_q, layout_kv=layout_kv, layout_out=c.get('layout_out',layout_q),
        return_softmax_lse=0)
torch.npu.synchronize()
torch.npu.empty_cache()
print('CASE_DONE')