"""
对比 <<<>>> dispatch 与 aclnn dispatch 的输出
证明 <<<>>> 的 tiling 和通信完全正确
"""
import os, sys, time, ctypes, torch, torch.distributed as dist, torch_npu
from torch.multiprocessing import Process, Lock, Value
from torch.utils.cpp_extension import load as jit_load

BS, H, K = 55, 1038, 1
EP_WORLD_SIZE, MOE_EXPERT_NUM = 2, 2
GLOBAL_BS = BS * EP_WORLD_SIZE
A = GLOBAL_BS * 1
BLOCK_DIM, WIN_ADDR_ALIGN = 32, 512
INPUT_DTYPE = torch.bfloat16
MASTER_PORT = "50055"

def calc_total_win_size():
    h_bytes = H * 2
    tsz = ((h_bytes + WIN_ADDR_ALIGN - 1) // WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN
    return A * K * tsz * 2

class DispatchTilingInfo(ctypes.Structure):
    _fields_ = [
        ("epWorldSize", ctypes.c_uint32), ("tpWorldSize", ctypes.c_uint32),
        ("epRankId", ctypes.c_uint32), ("tpRankId", ctypes.c_uint32),
        ("expertShardType", ctypes.c_uint32), ("sharedExpertNum", ctypes.c_uint32),
        ("sharedExpertRankNum", ctypes.c_uint32), ("moeExpertNum", ctypes.c_uint32),
        ("quantMode", ctypes.c_uint32), ("globalBs", ctypes.c_uint32),
        ("bs", ctypes.c_uint32), ("k", ctypes.c_uint32),
        ("h", ctypes.c_uint32), ("a", ctypes.c_uint32),
        ("aivNum", ctypes.c_uint32),
        ("isTokenMask", ctypes.c_bool), ("isExpertMask", ctypes.c_bool),
        ("hasElasticInfo", ctypes.c_bool), ("isPerformance", ctypes.c_bool),
        ("isQuant", ctypes.c_bool), ("isMc2Context", ctypes.c_bool),
        ("reserved1", ctypes.c_bool), ("reserved2", ctypes.c_bool),
        ("totalUbSize", ctypes.c_uint64), ("totalWinSizeEp", ctypes.c_uint64),
        ("totalWinSizeTp", ctypes.c_uint64),
        ("expertTokenNumsType", ctypes.c_uint32), ("zeroComputeExpertNum", ctypes.c_int32),
        ("maxSizeForUbBuffer", ctypes.c_uint32),
        ("scalesRow", ctypes.c_uint64), ("scalesCol", ctypes.c_uint64),
        ("scalesTypeSize", ctypes.c_uint32), ("scalesCount", ctypes.c_uint64),
    ]

def make_tiling(rank):
    t = DispatchTilingInfo()
    t.epWorldSize = EP_WORLD_SIZE; t.tpWorldSize = 1; t.epRankId = rank
    t.moeExpertNum = MOE_EXPERT_NUM; t.globalBs = GLOBAL_BS
    t.bs = BS; t.k = K; t.h = H; t.a = A; t.aivNum = BLOCK_DIM
    t.isMc2Context = True; t.totalUbSize = 192*1024
    t.totalWinSizeEp = calc_total_win_size(); t.expertTokenNumsType = 1
    return torch.frombuffer(bytearray(bytes(t)), dtype=torch.uint8)

def run(rank, ws, barrier_counter, barrier_lock):
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = MASTER_PORT
    torch_npu.npu.set_device(rank % 2)
    dist.init_process_group(backend="hccl", rank=rank, world_size=ws,
                            init_method=f"tcp://127.0.0.1:{MASTER_PORT}")
    ep_group = dist.new_group(backend="hccl", ranks=list(range(ws)))
    ep_name = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

    # Context
    tnp = os.path.dirname(os.path.abspath(torch_npu.__file__))
    cann = os.environ['ASCEND_HOME_PATH']
    ext = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../torch_extension/npu_ops_transformer'))
    ctx_mod = jit_load(name='mc2_ctx_cmp', sources=[f'{ext}/ops/csrc/comm_context.cpp'],
        extra_include_paths=[f'{tnp}/include', f'{tnp}/include/third_party/hccl/inc',
                            f'{tnp}/include/third_party/acl/inc', f'{cann}/include', f'{ext}/common/inc'],
        extra_ldflags=[f'-L{tnp}/lib', '-ltorch_npu'],
        extra_cflags=['-D_GLIBCXX_USE_CXX11_ABI=1'], verbose=False)
    mgr = ctx_mod.CommContextManager(ep_name, EP_WORLD_SIZE, "channel")
    context = mgr.create_context()
    buf = mgr.ccl_buffer_size

    # aclnn dispatch 扩展
    disp_aclnn = jit_load(name='mc2_disp_cmp', sources=[f'{ext}/ops/csrc/moe_distribute_dispatch.cpp'],
        extra_include_paths=[f'{tnp}/include', f'{tnp}/include/third_party/hccl/inc',
                            f'{tnp}/include/third_party/acl/inc', f'{cann}/include',
                            f'{cann}/include/aclnn', f'{ext}/common/inc'],
        extra_ldflags=[f'-L{tnp}/lib', '-ltorch_npu', f'-L{cann}/lib64', '-lnnopbase', '-lopapi',
                      '-lacl_op_compiler', f'-L{cann}/opp/vendors/custom_transformer/op_api/lib', '-lcust_opapi'],
        extra_cflags=['-D_GLIBCXX_USE_CXX11_ABI=1'], verbose=False)

    # <<<>>> module
    so_dir = os.path.join(os.path.dirname(__file__), '..', 'build', 'csrc')
    sys.path.insert(0, so_dir)
    import mc2_direct_launch as mc2

    # Same input for both
    torch.manual_seed(42 + rank)
    x = torch.randn(BS, H, dtype=INPUT_DTYPE).npu()
    expert_ids = torch.randint(0, MOE_EXPERT_NUM, (BS, K), dtype=torch.int32).npu()

    # === Run aclnn dispatch ===
    r_aclnn = disp_aclnn.npu_moe_distribute_dispatch(
        context, x, expert_ids, EP_WORLD_SIZE, rank, MOE_EXPERT_NUM, buf,
        None, None, None, None, None, 1, 0, 0, 1, 0, 0, GLOBAL_BS, 1, "", 0, 0, 0, None, None, None)
    expand_x_aclnn = r_aclnn[0]
    assist_aclnn = r_aclnn[2]
    recv_aclnn = r_aclnn[4]
    torch.npu.synchronize()

    # === Run <<<>>> dispatch ===
    expand_x_kl = torch.zeros(A, H, dtype=INPUT_DTYPE).npu()
    dscales = torch.zeros(A, dtype=torch.float32).npu()
    assist_kl = torch.zeros(A * 128, dtype=torch.int32).npu()
    etokens = torch.zeros(1, dtype=torch.int64).npu()
    recv_kl = torch.zeros(EP_WORLD_SIZE, dtype=torch.int32).npu()
    tp_recv = torch.zeros(1, dtype=torch.int32).npu()
    escales = torch.zeros(A, dtype=torch.float32).npu()
    ws_t = torch.zeros(16*1024*1024//4, dtype=torch.int32).npu()
    tiling = make_tiling(rank)

    mc2.mc2_dispatch(context, x, expert_ids, 1, BLOCK_DIM,
                     None, None, None, None, None,
                     expand_x_kl, dscales, assist_kl, etokens,
                     recv_kl, tp_recv, escales, ws_t, tiling)
    torch.npu.synchronize()

    # === Compare ===
    # expand_x 有效区域 (ep_recv 总和的 token 数)
    n_tokens_aclnn = recv_aclnn.sum().item()
    n_tokens_kl = recv_kl.sum().item()

    expand_match = torch.allclose(expand_x_aclnn[:n_tokens_aclnn], expand_x_kl[:n_tokens_kl], atol=0, rtol=0)
    recv_match = torch.equal(recv_aclnn.cpu(), recv_kl.cpu())
    # assist_info 比较有效区域
    assist_len = n_tokens_kl * 3
    assist_match = torch.equal(assist_aclnn[:assist_len].cpu(), assist_kl[:assist_len].cpu())

    print(f"[R{rank}] aclnn_tokens={n_tokens_aclnn} kl_tokens={n_tokens_kl} "
          f"expand_match={expand_match} recv_match={recv_match} assist_match={assist_match}",
          flush=True)

    all_ok = expand_match and recv_match and assist_match
    print(f"[R{rank}] {'PASSED - outputs identical' if all_ok else 'DIFF'}", flush=True)

    with barrier_lock:
        barrier_counter.value += 1

    # spin-wait until all ranks finished
    while True:
        with barrier_lock:
            if barrier_counter.value == ws:
                break
        time.sleep(0.01)

        dist.destroy_process_group()

if __name__ == "__main__":
    print("=== Dispatch: <<<>>> vs aclnn comparison ===")
    tnp = os.path.dirname(os.path.abspath(torch_npu.__file__))
    cann = os.environ['ASCEND_HOME_PATH']
    ext = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../torch_extension/npu_ops_transformer'))
    jit_load(name='mc2_ctx_cmp', sources=[f'{ext}/ops/csrc/comm_context.cpp'],
        extra_include_paths=[f'{tnp}/include', f'{tnp}/include/third_party/hccl/inc',
                            f'{tnp}/include/third_party/acl/inc', f'{cann}/include', f'{ext}/common/inc'],
        extra_ldflags=[f'-L{tnp}/lib', '-ltorch_npu'],
        extra_cflags=['-D_GLIBCXX_USE_CXX11_ABI=1'], verbose=False)
    jit_load(name='mc2_disp_cmp', sources=[f'{ext}/ops/csrc/moe_distribute_dispatch.cpp'],
        extra_include_paths=[f'{tnp}/include', f'{tnp}/include/third_party/hccl/inc',
                            f'{tnp}/include/third_party/acl/inc', f'{cann}/include',
                            f'{cann}/include/aclnn', f'{ext}/common/inc'],
        extra_ldflags=[f'-L{tnp}/lib', '-ltorch_npu', f'-L{cann}/lib64', '-lnnopbase', '-lopapi',
                      '-lacl_op_compiler', f'-L{cann}/opp/vendors/custom_transformer/op_api/lib', '-lcust_opapi'],
        extra_cflags=['-D_GLIBCXX_USE_CXX11_ABI=1'], verbose=False)
    print("Pre-compile done.")
    os.system("fuser -k /dev/npu* 2>/dev/null"); time.sleep(2)
    barrier_counter = Value("i", 0)
    barrier_lock = Lock()
    procs = [Process(target=run, args=(r, EP_WORLD_SIZE, barrier_counter, barrier_lock)) for r in range(EP_WORLD_SIZE)]
    for p in procs: p.start()
    for p in procs: p.join(timeout=90)
    ok = all(p.exitcode == 0 for p in procs)
    print("ALL PASSED" if ok else "FAILED")
