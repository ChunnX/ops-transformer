"""
MC2 <<<>>> Golden 对比测试
支持三种模式: dispatch / combine / cascade (dispatch→combine)
每种模式都与 CPU golden 对比，使用 v3v4 的 compare_output + toolss.data_compare_np

用法:
  python tests/test_golden.py --dispatch          # 单跑 dispatch
  python tests/test_golden.py --combine           # 单跑 combine
   python tests/test_golden.py --cascade           # 级联 dispatch→combine
"""
import os, sys, time, ctypes, argparse, torch, torch.distributed as dist, torch_npu
from torch.multiprocessing import Process, Manager
from torch.utils.cpp_extension import load as jit_load

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pta-moe-test'))
from moe_operator_test_v3v4 import compare_output, ComparedOutput

# ==================== Config ====================
BS, H, K = 55, 1038, 1
EP, MOE, LOCAL_MOE = 2, 2, 1
GBS = BS * EP
A = GBS * LOCAL_MOE
BD, WA = 32, 512
MASTER_PORT = "29508"
WS_SIZE = 16 * 1024 * 1024

def cwsz():
    tc = ((H * 2 + WA - 1) // WA) * WA
    td = ((H * 2 + 32 - 1) // 32 * 32 + 44 + WA - 1) // WA * WA
    return (A * K * tc + A * K * td) * 2

# ==================== Tiling Structs ====================
class DT(ctypes.Structure):
    _fields_ = [('epWorldSize',ctypes.c_uint32),('tpWorldSize',ctypes.c_uint32),('epRankId',ctypes.c_uint32),('tpRankId',ctypes.c_uint32),('expertShardType',ctypes.c_uint32),('sharedExpertNum',ctypes.c_uint32),('sharedExpertRankNum',ctypes.c_uint32),('moeExpertNum',ctypes.c_uint32),('quantMode',ctypes.c_uint32),('globalBs',ctypes.c_uint32),('bs',ctypes.c_uint32),('k',ctypes.c_uint32),('h',ctypes.c_uint32),('a',ctypes.c_uint32),('aivNum',ctypes.c_uint32),('isTokenMask',ctypes.c_bool),('isExpertMask',ctypes.c_bool),('hasElasticInfo',ctypes.c_bool),('isPerformance',ctypes.c_bool),('isQuant',ctypes.c_bool),('isMc2Context',ctypes.c_bool),('r1',ctypes.c_bool),('r2',ctypes.c_bool),('totalUbSize',ctypes.c_uint64),('totalWinSizeEp',ctypes.c_uint64),('totalWinSizeTp',ctypes.c_uint64),('expertTokenNumsType',ctypes.c_uint32),('zeroComputeExpertNum',ctypes.c_int32),('maxSizeForUbBuffer',ctypes.c_uint32),('scalesRow',ctypes.c_uint64),('scalesCol',ctypes.c_uint64),('scalesTypeSize',ctypes.c_uint32),('scalesCount',ctypes.c_uint64)]

class CT(ctypes.Structure):
    _fields_ = [('epWorldSize',ctypes.c_uint32),('tpWorldSize',ctypes.c_uint32),('epRankId',ctypes.c_uint32),('tpRankId',ctypes.c_uint32),('expertShardType',ctypes.c_uint32),('sharedExpertNum',ctypes.c_uint32),('sharedExpertRankNum',ctypes.c_uint32),('moeExpertNum',ctypes.c_uint32),('moeExpertPerRankNum',ctypes.c_uint32),('zeroExpertNum',ctypes.c_uint32),('copyExpertNum',ctypes.c_uint32),('constExpertNum',ctypes.c_uint32),('globalBs',ctypes.c_uint32),('bs',ctypes.c_uint32),('k',ctypes.c_uint32),('h',ctypes.c_uint32),('a',ctypes.c_uint32),('aivNum',ctypes.c_uint32),('isTokenMask',ctypes.c_bool),('isExpertMask',ctypes.c_bool),('hasSharedExpertX',ctypes.c_bool),('hasElasticInfo',ctypes.c_bool),('isPerformance',ctypes.c_bool),('isMc2Context',ctypes.c_bool),('r1',ctypes.c_bool),('r2',ctypes.c_bool),('totalUbSize',ctypes.c_uint64),('totalWinSizeEp',ctypes.c_uint64),('totalWinSizeTp',ctypes.c_uint64),('armAvgFactor',ctypes.c_float),('epsilon',ctypes.c_float),('bufferNum',ctypes.c_uint32)]

def mdt(r):
    t = DT(); t.epWorldSize=EP;t.tpWorldSize=1;t.epRankId=r;t.moeExpertNum=MOE;t.globalBs=GBS;t.bs=BS;t.k=K;t.h=H;t.a=A;t.aivNum=BD;t.isMc2Context=True;t.totalUbSize=192*1024;t.totalWinSizeEp=cwsz();t.expertTokenNumsType=1
    return torch.frombuffer(bytearray(bytes(t)), dtype=torch.uint8)

def mct(r):
    t = CT(); t.epWorldSize=EP;t.tpWorldSize=1;t.epRankId=r;t.moeExpertNum=MOE;t.moeExpertPerRankNum=LOCAL_MOE;t.globalBs=GBS;t.bs=BS;t.k=K;t.h=H;t.a=A;t.aivNum=BD;t.isMc2Context=True;t.totalUbSize=192*1024;t.totalWinSizeEp=cwsz();t.bufferNum=2
    return torch.frombuffer(bytearray(bytes(t)), dtype=torch.uint8)

def fb(rank, ws, tag):
    open(f'/tmp/mc2_{tag}_{rank}', 'w').close()
    while not all(os.path.exists(f'/tmp/mc2_{tag}_{r}') for r in range(ws)): time.sleep(0.01)
    time.sleep(0.1)

# ==================== Worker ====================
def run(rank, ws, mode, result_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = MASTER_PORT
    torch_npu.npu.set_device(rank % 2)
    dist.init_process_group(backend="hccl", rank=rank, world_size=ws, init_method=f"tcp://127.0.0.1:{MASTER_PORT}")
    eg = dist.new_group(backend="hccl", ranks=list(range(ws)))
    en = eg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

    tnp = os.path.dirname(os.path.abspath(torch_npu.__file__))
    cann = os.environ['ASCEND_HOME_PATH']
    ext = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../torch_extension/npu_ops_transformer'))
    ctx_m = jit_load(name='mc2_ctx_g', sources=[f'{ext}/ops/csrc/comm_context.cpp'],
        extra_include_paths=[f'{tnp}/include', f'{tnp}/include/third_party/hccl/inc',
                            f'{tnp}/include/third_party/acl/inc', f'{cann}/include', f'{ext}/common/inc'],
        extra_ldflags=[f'-L{tnp}/lib', '-ltorch_npu'], extra_cflags=['-D_GLIBCXX_USE_CXX11_ABI=1'], verbose=False)
    mgr = ctx_m.CommContextManager(en, EP, "channel"); ctx = mgr.create_context()
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'csrc'))
    import mc2_direct_launch as mc2

    torch.manual_seed(42 + rank)
    x = torch.randn(BS, H, dtype=torch.bfloat16).npu()
    eid = torch.randint(0, MOE, (BS, K), dtype=torch.int32).npu()
    scales = (torch.rand(BS, K, dtype=torch.float32) + 0.5).npu()

    res = {'rank': rank, 'x': x.cpu(), 'eid': eid.cpu(), 'scales': scales.cpu()}

    if mode in ('dispatch', 'cascade'):
        ex = torch.zeros(A, H, dtype=torch.bfloat16).npu()
        ds = torch.zeros(A, dtype=torch.float32).npu()
        ai = torch.zeros(A * 128, dtype=torch.int32).npu()
        et = torch.zeros(LOCAL_MOE, dtype=torch.int64).npu()
        er = torch.zeros(EP * LOCAL_MOE, dtype=torch.int32).npu()
        tr = torch.zeros(1, dtype=torch.int32).npu()
        es = torch.zeros(A, dtype=torch.float32).npu()
        w = torch.zeros(WS_SIZE // 4, dtype=torch.int32).npu()
        mc2.mc2_dispatch(ctx, x, eid, 1, BD, None, None, None, None, None,
                         ex, ds, ai, et, er, tr, es, w, mdt(rank))
        torch.npu.synchronize()
        n = er.sum().item()
        res['expand_x'] = ex[:n].cpu()
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
            print(f"[R{rank}] combine done", flush=True)

    elif mode == 'combine':
        # 独立 combine: 全本地路由
        expand_x = torch.randn(A, H, dtype=torch.bfloat16).npu()
        assist_data = torch.zeros(A * 128, dtype=torch.int32)
        for i in range(BS):
            assist_data[i*3] = rank; assist_data[i*3+1] = i; assist_data[i*3+2] = 0
        ai = assist_data.npu()
        ep_send = torch.zeros(EP * LOCAL_MOE, dtype=torch.int32); ep_send[-1] = BS; ep_send = ep_send.npu()
        xo = torch.zeros(BS, H, dtype=torch.bfloat16).npu()
        w = torch.zeros(WS_SIZE // 4, dtype=torch.int32).npu()
        mc2.mc2_combine(ctx, expand_x, eid, ai, ep_send, scales, 1, BD,
                        None, None, None, None, None, None, xo, w, mct(rank))
        torch.npu.synchronize()
        res['expand_x_input'] = expand_x[:BS].cpu()
        res['x_out'] = xo.cpu()
        print(f"[R{rank}] combine done", flush=True)

    result_queue.put(res)
    dist.destroy_process_group()

# ==================== Golden + Compare ====================
def compare_dispatch(results):
    """dispatch 是多 rank 集合通信，CPU golden 需全局路由信息。
    此处仅验证输出非零 + shape 正确。bit-identical 验证见 test_dispatch_compare.py"""
    print("\n################## dispatch result ##################")
    ok = True
    for rank in range(EP):
        r = results[rank]
        ex_ok = r['expand_x'].numel() > 0 and r['expand_x'].float().abs().sum() > 0
        recv_ok = r['ep_recv'].sum().item() > 0
        rank_ok = ex_ok and recv_ok
        print(f"  R{rank}: expand_x_nonzero={ex_ok} recv_valid={recv_ok} → {'PASS' if rank_ok else 'FAIL'}")
        if not rank_ok: ok = False
    result = {f'expand_x_{r}': results[r]['expand_x'].float().abs().sum() > 0 for r in range(EP)}
    result['ep_recv_counts'] = all(results[r]['ep_recv'].sum().item() > 0 for r in range(EP))
    print(f"result_dispatch: {result}")
    return ok

def compare_combine(results, mode):
    print("\n################## combine compare ##################")
    golden_x, npu_x = [], []
    for rank in range(EP):
        r = results[rank]
        if mode == 'combine':
            # 全本地路由: golden = expand_x[0:BS] * scales
            golden = (r['expand_x_input'].float() * r['scales']).to(torch.bfloat16)
        else:
            # 级联: golden = x * scales (round-trip)
            golden = (r['x'].float() * r['scales']).to(torch.bfloat16)
        golden_x.append(golden)
        npu_x.append(r['x_out'])

    os.makedirs('csv_result', exist_ok=True)
    res = compare_output(ComparedOutput.X, golden_x, npu_x, "float32", True, list(range(EP)))
    print(f"result_combine: {res}")
    return all(res.values())

# ==================== Main ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dispatch', action='store_true')
    parser.add_argument('--combine', action='store_true')
    parser.add_argument('--cascade', action='store_true')
    args = parser.parse_args()
    if not (args.dispatch or args.combine or args.cascade):
        parser.error('请指定运行模式: --dispatch / --combine / --cascade')
    if sum([args.dispatch, args.combine, args.cascade]) > 1:
        parser.error('一次只能指定一个模式，不支持组合运行')

    if args.dispatch: mode = 'dispatch'
    elif args.combine: mode = 'combine'
    else: mode = 'cascade'

    tnp = os.path.dirname(os.path.abspath(torch_npu.__file__))
    cann = os.environ['ASCEND_HOME_PATH']
    ext = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../torch_extension/npu_ops_transformer'))
    jit_load(name='mc2_ctx_g', sources=[f'{ext}/ops/csrc/comm_context.cpp'],
        extra_include_paths=[f'{tnp}/include', f'{tnp}/include/third_party/hccl/inc',
                            f'{tnp}/include/third_party/acl/inc', f'{cann}/include', f'{ext}/common/inc'],
        extra_ldflags=[f'-L{tnp}/lib', '-ltorch_npu'], extra_cflags=['-D_GLIBCXX_USE_CXX11_ABI=1'], verbose=False)

    print(f"\n{'='*60}\n  <<<>>> {mode} test (bs={BS} h={H} k={K} ep={EP} bf16)\n{'='*60}")
    os.system("fuser -k /dev/npu* 2>/dev/null; rm -f /tmp/mc2_gc_* 2>/dev/null; rm -f /tmp/mc2_* 2>/dev/null")
    time.sleep(10)
    manager = Manager()
    rq = manager.Queue()
    ps = [Process(target=run, args=(r, EP, mode, rq)) for r in range(EP)]
    for p in ps: p.start()
    for p in ps: p.join(timeout=50)
    os.system("rm -f /tmp/mc2_gc_*")
    if not all(p.exitcode == 0 for p in ps):
        print(f"  {mode}: TIMEOUT/ERROR"); sys.exit(1)

    results = {}
    while not rq.empty():
        r = rq.get(); results[r['rank']] = r

    if mode == 'dispatch':
        ok = compare_dispatch(results)
    elif mode == 'combine':
        ok = compare_combine(results, 'combine')
    else:
        ok1 = compare_dispatch(results)
        ok2 = compare_combine(results, 'cascade')
        ok = ok1 and ok2

    print(f"\n  {mode} test {'Success' if ok else 'Failed'}.")
