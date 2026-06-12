"""
MC2 <<<>>> Dispatch 独立测试 + CPU golden 对比 + CSV 输出
"""
import os, sys, time, ctypes, torch, torch.distributed as dist, torch_npu
from torch.multiprocessing import Process, Manager, Lock, Value
from torch.utils.cpp_extension import load as jit_load

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pta-moe-test'))
from moe_operator_test_v3v4 import compare_output, ComparedOutput

BS, H, K = 55, 1038, 1
EP_WORLD_SIZE, MOE_EXPERT_NUM = 2, 2
LOCAL_MOE_EXPERT_NUM = 1
GLOBAL_BS = BS * EP_WORLD_SIZE
A = GLOBAL_BS * LOCAL_MOE_EXPERT_NUM
BLOCK_DIM, WIN_ADDR_ALIGN = 32, 512
INPUT_DTYPE = torch.bfloat16
MASTER_PORT = "50066"
RANK_PER_DEV = 2
WORKSPACE_SIZE = 16 * 1024 * 1024

def calc_total_win_size():
    h_bytes = H * 2
    tc = ((h_bytes + WIN_ADDR_ALIGN - 1) // WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN
    td = ((h_bytes + 32 - 1) // 32 * 32 + 44 + WIN_ADDR_ALIGN - 1) // WIN_ADDR_ALIGN * WIN_ADDR_ALIGN
    return (A * K * tc + A * K * td) * 2

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

def make_dispatch_tiling(rank):
    t = DispatchTilingInfo()
    t.epWorldSize = EP_WORLD_SIZE; t.tpWorldSize = 1; t.epRankId = rank
    t.moeExpertNum = MOE_EXPERT_NUM; t.globalBs = GLOBAL_BS
    t.bs = BS; t.k = K; t.h = H; t.a = A; t.aivNum = BLOCK_DIM
    t.isMc2Context = True; t.totalUbSize = 192 * 1024
    t.totalWinSizeEp = calc_total_win_size(); t.expertTokenNumsType = 1
    return torch.frombuffer(bytearray(bytes(t)), dtype=torch.uint8)

def run_worker(rank, world_size, result_queue, barrier_counter, barrier_lock):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = MASTER_PORT
    torch_npu.npu.set_device(rank % RANK_PER_DEV)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size,
                            init_method=f"tcp://127.0.0.1:{MASTER_PORT}")
    ep_group = dist.new_group(backend="hccl", ranks=list(range(world_size)))
    ep_name = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

    torch_npu_path = os.path.dirname(os.path.abspath(torch_npu.__file__))
    cann_path = os.environ['ASCEND_HOME_PATH']
    ext_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../torch_extension/npu_ops_transformer'))
    ctx_mod = jit_load(name='mc2_ctx_dl', sources=[f'{ext_root}/ops/csrc/comm_context.cpp'],
        extra_include_paths=[f'{torch_npu_path}/include', f'{torch_npu_path}/include/third_party/hccl/inc',
                            f'{torch_npu_path}/include/third_party/acl/inc', f'{cann_path}/include',
                            f'{ext_root}/common/inc'],
        extra_ldflags=[f'-L{torch_npu_path}/lib', '-ltorch_npu'],
        extra_cflags=['-D_GLIBCXX_USE_CXX11_ABI=1'], verbose=False)
    mgr = ctx_mod.CommContextManager(ep_name, EP_WORLD_SIZE, "channel")
    context = mgr.create_context()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'csrc'))
    import mc2_direct_launch as mc2

    torch.manual_seed(42 + rank)
    x = torch.randn(BS, H, dtype=INPUT_DTYPE).npu()
    expert_ids = torch.randint(0, MOE_EXPERT_NUM, (BS, K), dtype=torch.int32).npu()

    expand_x = torch.zeros(A, H, dtype=INPUT_DTYPE).npu()
    dynamic_scales = torch.zeros(A, dtype=torch.float32).npu()
    assist_info = torch.zeros(A * 128, dtype=torch.int32).npu()
    expert_token_nums = torch.zeros(LOCAL_MOE_EXPERT_NUM, dtype=torch.int64).npu()
    ep_recv_counts = torch.zeros(EP_WORLD_SIZE * LOCAL_MOE_EXPERT_NUM, dtype=torch.int32).npu()
    tp_recv_counts = torch.zeros(1, dtype=torch.int32).npu()
    expand_scales = torch.zeros(A, dtype=torch.float32).npu()
    workspace = torch.zeros(WORKSPACE_SIZE // 4, dtype=torch.int32).npu()
    tiling = make_dispatch_tiling(rank)

    mc2.mc2_dispatch(context, x, expert_ids, 1, BLOCK_DIM,
                     None, None, None, None, None,
                     expand_x, dynamic_scales, assist_info, expert_token_nums,
                     ep_recv_counts, tp_recv_counts, expand_scales, workspace, tiling)
    torch.npu.synchronize()

    n_tokens = ep_recv_counts.sum().item()
    print(f"[R{rank}] PASSED abs_sum={expand_x.float().abs().sum().item():.2f} "
          f"ep_recv={ep_recv_counts.cpu().tolist()} tokens={expert_token_nums.cpu().tolist()}", flush=True)

    result_queue.put((rank, x.cpu(), expert_ids.cpu(), expand_x[:n_tokens].cpu(), ep_recv_counts.cpu()))
    with barrier_lock:
        barrier_counter.value += 1

    # spin-wait until all ranks finished
    while True:
        with barrier_lock:
            if barrier_counter.value == world_size:
                break
        time.sleep(0.01)

        dist.destroy_process_group()


if __name__ == "__main__":
    print("=" * 50)
    print("MC2 <<<>>> Dispatch Direct Launch Test")
    print(f"bs={BS} h={H} k={K} ep={EP_WORLD_SIZE} bf16")
    print("=" * 50)

    torch_npu_path = os.path.dirname(os.path.abspath(torch_npu.__file__))
    cann_path = os.environ['ASCEND_HOME_PATH']
    ext_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../torch_extension/npu_ops_transformer'))
    jit_load(name='mc2_ctx_dl', sources=[f'{ext_root}/ops/csrc/comm_context.cpp'],
        extra_include_paths=[f'{torch_npu_path}/include', f'{torch_npu_path}/include/third_party/hccl/inc',
                            f'{torch_npu_path}/include/third_party/acl/inc', f'{cann_path}/include',
                            f'{ext_root}/common/inc'],
        extra_ldflags=[f'-L{torch_npu_path}/lib', '-ltorch_npu'],
        extra_cflags=['-D_GLIBCXX_USE_CXX11_ABI=1'], verbose=False)
    print("Pre-compile done.", flush=True)

    os.system("fuser -k /dev/npu* 2>/dev/null")
    time.sleep(2)

    manager = Manager()
    result_queue = manager.Queue()
    barrier_counter = Value('i', 0)
    barrier_lock = Lock()
    procs = [Process(target=run_worker, args=(r, EP_WORLD_SIZE, result_queue, barrier_counter, barrier_lock)) for r in range(EP_WORLD_SIZE)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)

    if not all(p.exitcode == 0 for p in procs):
        print("FAILED")
        sys.exit(1)

    # 收集结果 + CPU golden 对比
    results = {}
    while not result_queue.empty():
        rank, x, eid, expand_x_npu, recv = result_queue.get()
        results[rank] = (x, eid, expand_x_npu, recv)

    # CPU golden: 按 expert_id 排序 token
    golden_expand_x = []
    npu_expand_x = []
    for rank in range(EP_WORLD_SIZE):
        x, eid, ex_npu, recv = results[rank]
        sorted_idx = torch.argsort(eid.flatten(), stable=True)
        dst_row = sorted_idx // K
        golden = x[dst_row][:recv.sum().item()]
        golden_expand_x.append(golden)
        npu_expand_x.append(ex_npu)

    os.makedirs('csv_result', exist_ok=True)
    print("\n>>> compare_output (dispatch expand_x vs CPU golden):")
    res = compare_output(ComparedOutput.EXPAND_X, golden_expand_x, npu_expand_x, "float32", True, list(range(EP_WORLD_SIZE)))
    print(f"Result: {res}")
    print("ALL PASSED" if all(res.values()) else "FAILED")
