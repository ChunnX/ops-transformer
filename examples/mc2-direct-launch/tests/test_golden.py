#!/usr/bin/env python3
"""
MC2 Direct Launch Test — 用 pybind 分支跑通版本的同款测试框架
"""
import os, sys, time, ctypes, argparse
import numpy as np, torch, torch.distributed as dist, torch_npu
from torch.multiprocessing import Process, Manager
BASEDIR = os.path.dirname(os.path.abspath(__file__))
ROOTDIR = os.path.normpath(os.path.join(BASEDIR, '..'))
sys.path.insert(0, os.path.join(ROOTDIR, 'pta-moe-test'))
from toolss import data_compare_np
from moe_operator_test_v3v4 import compare_output, ComparedOutput

# ==== 常量 ====
BS, H, K = 55, 1038, 1
EP, MOE, LOCAL_MOE = 2, 2, 1
GBS, A = BS * EP, BS * EP * LOCAL_MOE
BD = 32
WS_SIZE = 16 * 1024 * 1024
MASTER_PORT = "29700"

def cwsz():
    tc = ((H * 2 + 511) // 512) * 512
    td = ((H * 2 + 31) // 32 * 32 + 44 + 511) // 512 * 512
    return (A * K * tc + A * K * td) * 2

# ==== tiling structs（匹配 tiling_host.h） ====
class DT(ctypes.Structure):
    _fields_ = [
        ('epWorldSize',ctypes.c_uint32),('tpWorldSize',ctypes.c_uint32),('epRankId',ctypes.c_uint32),('tpRankId',ctypes.c_uint32),
        ('expertShardType',ctypes.c_uint32),('sharedExpertNum',ctypes.c_uint32),('sharedExpertRankNum',ctypes.c_uint32),
        ('moeExpertNum',ctypes.c_uint32),('quantMode',ctypes.c_uint32),('globalBs',ctypes.c_uint32),
        ('bs',ctypes.c_uint32),('k',ctypes.c_uint32),('h',ctypes.c_uint32),('a',ctypes.c_uint32),('aivNum',ctypes.c_uint32),
        ('isTokenMask',ctypes.c_bool),('isExpertMask',ctypes.c_bool),('hasElasticInfo',ctypes.c_bool),('isPerformance',ctypes.c_bool),
        ('isQuant',ctypes.c_bool),('isMc2Context',ctypes.c_bool),('reserved1',ctypes.c_bool),('reserved2',ctypes.c_bool),
        ('totalUbSize',ctypes.c_uint64),('totalWinSizeEp',ctypes.c_uint64),
        ('expertTokenNumsType',ctypes.c_uint32),('zeroComputeExpertNum',ctypes.c_int32),('maxSizeForUbBuffer',ctypes.c_uint32),
        ('scalesRow',ctypes.c_uint64),('scalesCol',ctypes.c_uint64),('scalesTypeSize',ctypes.c_uint32),('scalesCount',ctypes.c_uint64),
    ]

class CT(ctypes.Structure):
    _fields_ = [
        ('epWorldSize',ctypes.c_uint32),('tpWorldSize',ctypes.c_uint32),('epRankId',ctypes.c_uint32),('tpRankId',ctypes.c_uint32),
        ('expertShardType',ctypes.c_uint32),('sharedExpertNum',ctypes.c_uint32),('sharedExpertRankNum',ctypes.c_uint32),
        ('moeExpertNum',ctypes.c_uint32),('moeExpertPerRankNum',ctypes.c_uint32),('zeroExpertNum',ctypes.c_uint32),
        ('copyExpertNum',ctypes.c_uint32),('constExpertNum',ctypes.c_uint32),('globalBs',ctypes.c_uint32),
        ('bs',ctypes.c_uint32),('k',ctypes.c_uint32),('h',ctypes.c_uint32),('a',ctypes.c_uint32),('aivNum',ctypes.c_uint32),
        ('isTokenMask',ctypes.c_bool),('isExpertMask',ctypes.c_bool),('hasSharedExpertX',ctypes.c_bool),('hasElasticInfo',ctypes.c_bool),
        ('isPerformance',ctypes.c_bool),('isMc2Context',ctypes.c_bool),('reserved1',ctypes.c_bool),('reserved2',ctypes.c_bool),
        ('totalUbSize',ctypes.c_uint64),('totalWinSizeEp',ctypes.c_uint64),
        ('armAvgFactor',ctypes.c_float),('epsilon',ctypes.c_float),('bufferNum',ctypes.c_uint32),
    ]

def mdt(rank):
    t = DT(); t.epWorldSize=EP;t.tpWorldSize=1;t.epRankId=rank;t.moeExpertNum=MOE;t.globalBs=GBS;t.bs=BS;t.k=K;t.h=H;t.a=A
    t.aivNum=BD;t.isMc2Context=True;t.totalUbSize=192*1024;t.totalWinSizeEp=cwsz();t.expertTokenNumsType=1
    return torch.frombuffer(bytearray(bytes(t)), dtype=torch.uint8)

def mct(rank):
    t = CT(); t.epWorldSize=EP;t.tpWorldSize=1;t.epRankId=rank;t.moeExpertNum=MOE;t.moeExpertPerRankNum=LOCAL_MOE;t.globalBs=GBS
    t.bs=BS;t.k=K;t.h=H;t.a=A;t.aivNum=BD;t.isMc2Context=True;t.totalUbSize=192*1024;t.totalWinSizeEp=cwsz();t.bufferNum=2
    return torch.frombuffer(bytearray(bytes(t)), dtype=torch.uint8)

def fb(rank, ws, tag):
    open(f'/tmp/mc2_{tag}_{rank}', 'w').close()
    while not all(os.path.exists(f'/tmp/mc2_{tag}_{r}') for r in range(ws)): time.sleep(0.01)
    time.sleep(0.1)

# ==== comm_context 编译 ====
def build_comm_context():
    tnp = os.path.dirname(os.path.abspath(torch_npu.__file__))
    cann = os.environ['ASCEND_HOME_PATH']
    ext = os.path.normpath(os.path.join(BASEDIR, '../../../torch_extension/npu_ops_transformer'))
    from torch.utils.cpp_extension import load as jit_load
    return jit_load(name='mc2_ctx_g', sources=[f'{ext}/ops/csrc/comm_context.cpp'],
        extra_include_paths=[f'{tnp}/include', f'{tnp}/include/third_party/hccl/inc',
                             f'{tnp}/include/third_party/acl/inc', f'{cann}/include', f'{ext}/common/inc'],
        extra_ldflags=[f'-L{tnp}/lib', '-ltorch_npu'], extra_cflags=['-D_GLIBCXX_USE_CXX11_ABI=1'], verbose=False)

# ==== worker ====
def run(rank, ws, mode, rq, ctx_m):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = MASTER_PORT
    torch_npu.npu.set_device(rank % 2)
    dist.init_process_group(backend="hccl", rank=rank, world_size=ws, init_method=f"tcp://127.0.0.1:{MASTER_PORT}")
    eg = dist.new_group(backend="hccl", ranks=list(range(ws)))
    en = eg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    mgr = ctx_m.CommContextManager(en, EP, "channel"); ctx = mgr.create_context()
    sys.path.insert(0, os.path.join(ROOTDIR, 'build', 'csrc'))
    import mc2_direct_launch as mc2

    torch.manual_seed(42 + rank)
    x = torch.randn(BS, H, dtype=torch.bfloat16).npu()
    eid = torch.randint(0, MOE, (BS, K), dtype=torch.int32).npu()
    scales = (torch.rand(BS, K, dtype=torch.float32) + 0.5).npu()
    res = {'rank': rank, 'x': x.cpu(), 'eid': eid.cpu(), 'scales': scales.cpu()}

    if mode in ('dispatch', 'cascade'):
        ex = torch.zeros(A, H, dtype=torch.bfloat16).npu()
        ds = torch.zeros(A, dtype=torch.float32).npu()
        ai = torch.zeros(A * 3, dtype=torch.int32).npu()
        et = torch.zeros(MOE, dtype=torch.int64).npu()
        er = torch.zeros(EP * MOE, dtype=torch.int32).npu()
        tr = torch.zeros(1, dtype=torch.int32).npu()
        es = torch.zeros(1, dtype=torch.uint8).npu()
        w = torch.zeros(WS_SIZE // 4, dtype=torch.int32).npu()
        mc2.mc2_dispatch(ctx, x, eid, 1, BD, None, None, None, None, None,
                         ex, ds, ai, et, er, tr, es, w, mdt(rank))
        torch.npu.synchronize()
        n = int(er.sum().item())
        res['expand_x'] = ex[:max(n,1)].cpu()
        res['ep_recv'] = er.cpu()
        res['expert_token_nums'] = et.cpu()
        res['assist_info'] = ai.cpu()
        print(f"[R{rank}] dispatch done", flush=True)

        if mode == 'cascade':
            fb(rank, ws, 'gc')
            xo = torch.zeros(BS, H, dtype=torch.bfloat16).npu()
            w2 = torch.zeros(WS_SIZE // 4, dtype=torch.int32).npu()
            mc2.mc2_combine(ctx, ex, eid, ai, er, scales, 1, BD,
                            None, None, None, None, None, None, xo, w2, mct(rank))
            torch.npu.synchronize()
            res['x_out'] = xo.cpu()

    elif mode == 'combine':
        expand_x = torch.randn(A, H, dtype=torch.bfloat16).npu()
        assist_data = torch.zeros(A * 3, dtype=torch.int32)
        for i in range(BS):
            assist_data[i*3] = rank; assist_data[i*3+1] = i; assist_data[i*3+2] = 0
        ai = assist_data.npu()
        ep_send = torch.zeros(EP * MOE, dtype=torch.int32); ep_send[-1] = BS; ep_send = ep_send.npu()
        xo = torch.zeros(BS, H, dtype=torch.bfloat16).npu()
        w = torch.zeros(WS_SIZE // 4, dtype=torch.int32).npu()
        mc2.mc2_combine(ctx, expand_x, eid, ai, ep_send, scales, 1, BD,
                        None, None, None, None, None, None, xo, w, mct(rank))
        torch.npu.synchronize()
        res['expand_x_input'] = expand_x[:BS].cpu()
        res['x_out'] = xo.cpu()

    rq.put(res)
    dist.destroy_process_group()

# ==== compare ====
def compare_dispatch(results):
    print("\n################## dispatch result ##################")
    ok = True
    for rank in range(EP):
        r = results[rank]
        ex_ok = r['expand_x'].numel() > 0 and r['expand_x'].float().abs().sum() > 0
        recv_ok = r['ep_recv'].sum().item() > 0
        rank_ok = ex_ok and recv_ok
        print(f"  R{rank}: expand_x_nonzero={ex_ok} recv_valid={recv_ok} -> {'PASS' if rank_ok else 'FAIL'}")
        if not rank_ok: ok = False
    return ok

def compare_combine(results, mode):
    print("\n################## combine compare ##################")
    golden_x, npu_x = [], []
    for rank in range(EP):
        r = results[rank]
        if mode == 'combine':
            golden = (r['expand_x_input'].float() * r['scales']).to(torch.bfloat16)
        else:
            golden = (r['x'].float() * r['scales']).to(torch.bfloat16)
        golden_x.append(golden)
        npu_x.append(r['x_out'])
    os.makedirs('csv_result', exist_ok=True)
    res = compare_output(ComparedOutput.X, golden_x, npu_x, "float32", True, list(range(EP)))
    print(f"result: {res}")
    return all(res.values())

# ==== main ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dispatch', action='store_true')
    parser.add_argument('--combine', action='store_true')
    parser.add_argument('--cascade', action='store_true')
    args = parser.parse_args()
    if args.dispatch: mode = 'dispatch'
    elif args.combine: mode = 'combine'
    else: mode = 'cascade'

    ctx_m = build_comm_context()
    print(f"\n{'='*60}\n  <<<>>> {mode} test (bs={BS} h={H} k={K} ep={EP} bf16)\n{'='*60}")
    os.system("fuser -k /dev/npu* 2>/dev/null; fuser -k /dev/davinci* 2>/dev/null")
    time.sleep(5)
    os.system("rm -f /tmp/mc2_gc_* /tmp/mc2_* 2>/dev/null")
    manager = Manager()
    rq = manager.Queue()
    ps = [Process(target=run, args=(r, EP, mode, rq, ctx_m)) for r in range(EP)]
    for p in ps: p.start()
    for p in ps: p.join(timeout=120)

    results = {}
    while not rq.empty():
        r = rq.get(); results[r['rank']] = r

    if mode == 'dispatch':
        ok = compare_dispatch(results)
    elif mode == 'combine':
        ok = compare_combine(results, 'combine')
    else:
        ok = compare_dispatch(results) and compare_combine(results, 'cascade')

    print(f"\n  {mode} test {'PASS' if ok else 'FAILED'}")
    os.system("rm -f /tmp/mc2_gc_*")
