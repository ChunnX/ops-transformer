import os
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '0'
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
import torch, torch_npu, numpy as np, io, contextlib, struct, math
torch_npu.npu.set_device(0)
torch.manual_seed(42); np.random.seed(42)

batch_size = 1; q_seq = 8192; k_seq = 8192
q_head_num = 64; k_head_num = 1; head_dim = 128
block_size = 1024; block_num = 10591
qk_dtype = torch.bfloat16; weight_dtype = torch.bfloat16
query = torch.randn(q_seq, q_head_num, head_dim, dtype=qk_dtype).npu()
weights = torch.randn(q_seq, q_head_num, dtype=weight_dtype).npu()
key = torch.randn(block_num, block_size, k_head_num, head_dim, dtype=qk_dtype).npu()
block_table = torch.tensor([[i for i in range(math.ceil(k_seq/block_size))]], dtype=torch.int32).npu()
actual_seq_q = torch.tensor([float(q_seq)], dtype=torch.int32).npu()
actual_seq_k = torch.tensor([float(k_seq)], dtype=torch.int32).npu()
r, v = torch_npu.npu_lightning_indexer(
    query, key, weights,
    actual_seq_lengths_query=actual_seq_q,
    actual_seq_lengths_key=actual_seq_k,
    block_table=block_table,
    layout_query="TND",
    layout_key="PA_BSND",
    sparse_count=512,
    sparse_mode=3)
torch.npu.synchronize()

# batch 512 debug data
vals = r[512, 0].cpu().tolist()
idx = 0
def rd(label):
    global idx
    if idx >= len(vals):
        print(f"  {label}: OUT OF DATA (idx={idx})")
        return -999999
    v = vals[idx]; idx += 1
    return v

print("=== Round 7 tile 0 intermediate values ===")
bf16_vals = [rd(f"bf16[{i}]") for i in range(4)]
print(f"  After Cast(float->bf16): {[hex(v & 0xFFFF) for v in bf16_vals]}")
mask_vals = [rd(f"mask[{i}]") for i in range(4)]
accumMask = rd("accumMask")
print(f"  After DoAndMask: {[hex(v & 0xFFFF) for v in mask_vals]}, accumMask={hex(accumMask & 0xFFFF)}")
float_vals = [rd(f"float[{i}]") for i in range(4)]
print(f"  After Cast(int16->float): {[struct.unpack('f', struct.pack('i', v))[0] for v in float_vals]}")
print(f"  Compares:")
for bm in [3, 2, 1]:
    im = rd(f"im{bm}"); rc = rd(f"rc{bm}"); cs = rd(f"cs{bm}")
    print(f"    bin{bm}: involvedMask=0x{im&0xFFFFFFFF:08x}, rsvdCnt={rc}, cumSum={cs}")

print("\n=== Per-round state ===")
for rnd in range(7, -1, -1):
    rd_id = rd("roundId")
    if rd_id == -999999: break
    gh = [rd(f"gh[{i}]") for i in range(4)]
    im = rd("involvedMask"); am = rd("andMask"); rk = rd("remainK"); ghbn = rd("ghbn")
    th_t0 = [rd(f"th_t0[{i}]") for i in range(4)]
    th_t1 = [rd(f"th_t1[{i}]") for i in range(4)]
    bins = [gh[3], gh[2]-gh[3], gh[1]-gh[2], gh[0]-gh[1]]
    print(f"  R{rd_id}: gh={gh}, bins={bins}, inv=0x{im&0xFFFF:04x}, mask=0x{am&0xFFFF:04x}, rk={rk}, ghbn={ghbn}")
    print(f"        th_t0={th_t0}, th_t1={th_t1}")
    bb = rd("bb"); bbp = rd("bbp"); im2 = rd("inv2"); tdtk = rd("tdtk"); rk2 = rd("rk2")
    tpk = [rd(f"tpk[{i}]") for i in range(2)]
    print(f"        bb={bb}, bbp={bbp}, inv2=0x{im2&0xFFFF:04x}, tdtk={tdtk}, rk={rk2}, tpk={tpk}")

print("\n=== Extraction ===")
inv_final = rd("inv_final"); bf_bits = rd("boundaryFloat")
bf = struct.unpack('f', struct.pack('i', bf_bits))[0]
tpk_final = [rd(f"tpk_final[{i}]") for i in range(2)]
print(f"  involvedMask=0x{inv_final&0xFFFF:04x}, boundaryFloat={bf}, tileTopK={tpk_final}")
print(f"  Total debug values: {idx}")
