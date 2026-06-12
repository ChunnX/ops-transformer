"""
脚本2: 运行 <<<>>> dispatch+combine 并保存结果为 .pt 文件
用法: python tests/run_kernel_launch.py
输入: tests/golden.pt (读取 inputs)
输出: tests/npu_output.pt
"""
import os, sys, time, ctypes, torch, torch.distributed as dist, torch_npu
from torch.multiprocessing import Process, Manager
from torch.utils.cpp_extension import load as jit_load

BS, H, K, EP, MOE, GBS, A = 55, 1038, 1, 2, 2, 110, 110
BD, WA = 32, 512
MASTER_PORT = "29507"

def cwsz():
    tc = ((H * 2 + WA - 1) // WA) * WA
    td = ((H * 2 + 32 - 1) // 32 * 32 + 44 + WA - 1) // WA * WA
    return (A * K * tc + A * K * td) * 2

class DT(ctypes.Structure):
    _fields_ = [('epWorldSize',ctypes.c_uint32),('tpWorldSize',ctypes.c_uint32),('epRankId',ctypes.c_uint32),('tpRankId',ctypes.c_uint32),('expertShardType',ctypes.c_uint32),('sharedExpertNum',ctypes.c_uint32),('sharedExpertRankNum',ctypes.c_uint32),('moeExpertNum',ctypes.c_uint32),('quantMode',ctypes.c_uint32),('globalBs',ctypes.c_uint32),('bs',ctypes.c_uint32),('k',ctypes.c_uint32),('h',ctypes.c_uint32),('a',ctypes.c_uint32),('aivNum',ctypes.c_uint32),('isTokenMask',ctypes.c_bool),('isExpertMask',ctypes.c_bool),('hasElasticInfo',ctypes.c_bool),('isPerformance',ctypes.c_bool),('isQuant',ctypes.c_bool),('isMc2Context',ctypes.c_bool),('r1',ctypes.c_bool),('r2',ctypes.c_bool),('totalUbSize',ctypes.c_uint64),('totalWinSizeEp',ctypes.c_uint64),('totalWinSizeTp',ctypes.c_uint64),('expertTokenNumsType',ctypes.c_uint32),('zeroComputeExpertNum',ctypes.c_int32),('maxSizeForUbBuffer',ctypes.c_uint32),('scalesRow',ctypes.c_uint64),('scalesCol',ctypes.c_uint64),('scalesTypeSize',ctypes.c_uint32),('scalesCount',ctypes.c_uint64)]

class CT(ctypes.Structure):
    _fields_ = [('epWorldSize',ctypes.c_uint32),('tpWorldSize',ctypes.c_uint32),('epRankId',ctypes.c_uint32),('tpRankId',ctypes.c_uint32),('expertShardType',ctypes.c_uint32),('sharedExpertNum',ctypes.c_uint32),('sharedExpertRankNum',ctypes.c_uint32),('moeExpertNum',ctypes.c_uint32),('moeExpertPerRankNum',ctypes.c_uint32),('zeroExpertNum',ctypes.c_uint32),('copyExpertNum',ctypes.c_uint32),('constExpertNum',ctypes.c_uint32),('globalBs',ctypes.c_uint32),('bs',ctypes.c_uint32),('k',ctypes.c_uint32),('h',ctypes.c_uint32),('a',ctypes.c_uint32),('aivNum',ctypes.c_uint32),('isTokenMask',ctypes.c_bool),('isExpertMask',ctypes.c_bool),('hasSharedExpertX',ctypes.c_bool),('hasElasticInfo',ctypes.c_bool),('isPerformance',ctypes.c_bool),('isMc2Context',ctypes.c_bool),('r1',ctypes.c_bool),('r2',ctypes.c_bool),('totalUbSize',ctypes.c_uint64),('totalWinSizeEp',ctypes.c_uint64),('totalWinSizeTp',ctypes.c_uint64),('armAvgFactor',ctypes.c_float),('epsilon',ctypes.c_float),('bufferNum',ctypes.c_uint32)]

def mdt(r):
    t = DT(); t.epWorldSize=EP;t.tpWorldSize=1;t.epRankId=r;t.moeExpertNum=MOE;t.globalBs=GBS;t.bs=BS;t.k=K;t.h=H;t.a=A;t.aivNum=BD;t.isMc2Context=True;t.totalUbSize=192*1024;t.totalWinSizeEp=cwsz();t.expertTokenNumsType=1
    return torch.frombuffer(bytearray(bytes(t)), dtype=torch.uint8)

def mct(r):
    t = CT(); t.epWorldSize=EP;t.tpWorldSize=1;t.epRankId=r;t.moeExpertNum=MOE;t.moeExpertPerRankNum=1;t.globalBs=GBS;t.bs=BS;t.k=K;t.h=H;t.a=A;t.aivNum=BD;t.isMc2Context=True;t.totalUbSize=192*1024;t.totalWinSizeEp=cwsz();t.bufferNum=2
    return torch.frombuffer(bytearray(bytes(t)), dtype=torch.uint8)

def file_barrier(rank, ws):
    open(f'/tmp/mc2_kl_{rank}', 'w').close()
    while not all(os.path.exists(f'/tmp/mc2_kl_{r}') for r in range(ws)):
        time.sleep(0.01)
    time.sleep(0.1)

def run(rank, ws, inputs, result_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = MASTER_PORT
    torch_npu.npu.set_device(rank % 2)
    dist.init_process_group(backend="hccl", rank=rank, world_size=ws,
                            init_method=f"tcp://127.0.0.1:{MASTER_PORT}")
    eg = dist.new_group(backend="hccl", ranks=list(range(ws)))
    en = eg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

    tnp = os.path.dirname(os.path.abspath(torch_npu.__file__))
    cann = os.environ['ASCEND_HOME_PATH']
    ext = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../torch_extension/npu_ops_transformer'))
    ctx_m = jit_load(name='mc2_ctx_kl', sources=[f'{ext}/ops/csrc/comm_context.cpp'],
        extra_include_paths=[f'{tnp}/include', f'{tnp}/include/third_party/hccl/inc',
                            f'{tnp}/include/third_party/acl/inc', f'{cann}/include', f'{ext}/common/inc'],
        extra_ldflags=[f'-L{tnp}/lib', '-ltorch_npu'],
        extra_cflags=['-D_GLIBCXX_USE_CXX11_ABI=1'], verbose=False)
    mgr = ctx_m.CommContextManager(en, EP, "channel")
    ctx = mgr.create_context()

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'build', 'csrc'))
    import mc2_direct_launch as mc2

    # 从 inputs 读取（与 gen_cpu_golden.py 相同的数据）
    x = inputs[rank]['x'].npu()
    eid = inputs[rank]['expert_ids'].npu()
    scales = inputs[rank]['expert_scales'].npu()

    # <<<>>> dispatch
    ex = torch.zeros(A, H, dtype=torch.bfloat16).npu()
    ds = torch.zeros(A, dtype=torch.float32).npu()
    ai = torch.zeros(A * 128, dtype=torch.int32).npu()
    et = torch.zeros(1, dtype=torch.int64).npu()
    er = torch.zeros(EP, dtype=torch.int32).npu()
    tr = torch.zeros(1, dtype=torch.int32).npu()
    es = torch.zeros(A, dtype=torch.float32).npu()
    w = torch.zeros(16 * 1024 * 1024 // 4, dtype=torch.int32).npu()
    mc2.mc2_dispatch(ctx, x, eid, 1, BD, None, None, None, None, None,
                     ex, ds, ai, et, er, tr, es, w, mdt(rank))
    torch.npu.synchronize()
    file_barrier(rank, ws)

    # <<<>>> combine
    xo = torch.zeros(BS, H, dtype=torch.bfloat16).npu()
    w2 = torch.zeros(16 * 1024 * 1024 // 4, dtype=torch.int32).npu()
    mc2.mc2_combine(ctx, ex, eid, ai, er, scales, 1, BD,
                    None, None, None, None, None, None, xo, w2, mct(rank))
    torch.npu.synchronize()

    result_queue.put((rank, xo.cpu()))
    print(f"[R{rank}] done, abs_sum={xo.float().abs().sum().item():.2f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    print("=" * 50)
    print("Run <<<>>> dispatch+combine")
    print("=" * 50)

    # 读取 inputs
    data_path = os.path.join(os.path.dirname(__file__), 'golden.pt')
    assert os.path.exists(data_path), f"Run gen_cpu_golden.py first! {data_path} not found"
    data = torch.load(data_path)
    inputs = data['inputs']

    # 预编译
    tnp = os.path.dirname(os.path.abspath(torch_npu.__file__))
    cann = os.environ['ASCEND_HOME_PATH']
    ext = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../torch_extension/npu_ops_transformer'))
    jit_load(name='mc2_ctx_kl', sources=[f'{ext}/ops/csrc/comm_context.cpp'],
        extra_include_paths=[f'{tnp}/include', f'{tnp}/include/third_party/hccl/inc',
                            f'{tnp}/include/third_party/acl/inc', f'{cann}/include', f'{ext}/common/inc'],
        extra_ldflags=[f'-L{tnp}/lib', '-ltorch_npu'],
        extra_cflags=['-D_GLIBCXX_USE_CXX11_ABI=1'], verbose=False)
    print("Pre-compile done.")

    os.system("fuser -k /dev/npu* 2>/dev/null; rm -f /tmp/mc2_kl_*")
    time.sleep(3)

    manager = Manager()
    result_queue = manager.Queue()
    procs = [Process(target=run, args=(r, EP, inputs, result_queue)) for r in range(EP)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=50)
    os.system("rm -f /tmp/mc2_kl_*")

    if not all(p.exitcode == 0 for p in procs):
        print("TIMEOUT/ERROR")
        sys.exit(1)

    # 收集结果并保存
    npu_output = {}
    while not result_queue.empty():
        rank, xo = result_queue.get()
        npu_output[rank] = xo

    out_path = os.path.join(os.path.dirname(__file__), 'npu_output.pt')
    torch.save(npu_output, out_path)
    print(f"\nSaved <<<>>> output to {out_path}")
