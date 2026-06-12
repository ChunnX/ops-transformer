"""
MC2 <<<>>> Combine 单独测试
独立构造 combine 输入数据，不依赖 dispatch 输出
"""
import os, sys, time, ctypes, torch, torch.distributed as dist, torch_npu
from torch.multiprocessing import Process, Lock, Value
from torch.utils.cpp_extension import load as jit_load

BS, H, K = 55, 1038, 1
EP_WORLD_SIZE, MOE_EXPERT_NUM, LOCAL_MOE = 2, 2, 1
GLOBAL_BS = BS * EP_WORLD_SIZE
A = GLOBAL_BS * LOCAL_MOE  # 110
BLOCK_DIM, WIN_ADDR_ALIGN = 32, 512
INPUT_DTYPE = torch.bfloat16
MASTER_PORT = "50077"
WORKSPACE_SIZE = 16 * 1024 * 1024
TILING_KEY_BF16 = 1


def calc_total_win_size():
    h_bytes = H * 2
    tc = ((h_bytes + WIN_ADDR_ALIGN - 1) // WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN
    td = ((h_bytes + 32 - 1) // 32 * 32 + 44 + WIN_ADDR_ALIGN - 1) // WIN_ADDR_ALIGN * WIN_ADDR_ALIGN
    return (A * K * tc + A * K * td) * 2


class CombineTilingInfo(ctypes.Structure):
    _fields_ = [
        ("epWorldSize", ctypes.c_uint32), ("tpWorldSize", ctypes.c_uint32),
        ("epRankId", ctypes.c_uint32), ("tpRankId", ctypes.c_uint32),
        ("expertShardType", ctypes.c_uint32), ("sharedExpertNum", ctypes.c_uint32),
        ("sharedExpertRankNum", ctypes.c_uint32), ("moeExpertNum", ctypes.c_uint32),
        ("moeExpertPerRankNum", ctypes.c_uint32),
        ("zeroExpertNum", ctypes.c_uint32), ("copyExpertNum", ctypes.c_uint32),
        ("constExpertNum", ctypes.c_uint32),
        ("globalBs", ctypes.c_uint32), ("bs", ctypes.c_uint32),
        ("k", ctypes.c_uint32), ("h", ctypes.c_uint32),
        ("a", ctypes.c_uint32), ("aivNum", ctypes.c_uint32),
        ("isTokenMask", ctypes.c_bool), ("isExpertMask", ctypes.c_bool),
        ("hasSharedExpertX", ctypes.c_bool), ("hasElasticInfo", ctypes.c_bool),
        ("isPerformance", ctypes.c_bool), ("isMc2Context", ctypes.c_bool),
        ("reserved1", ctypes.c_bool), ("reserved2", ctypes.c_bool),
        ("totalUbSize", ctypes.c_uint64), ("totalWinSizeEp", ctypes.c_uint64),
        ("totalWinSizeTp", ctypes.c_uint64),
        ("armAvgFactor", ctypes.c_float), ("epsilon", ctypes.c_float),
        ("bufferNum", ctypes.c_uint32),
    ]


def make_combine_tiling(rank):
    t = CombineTilingInfo()
    t.epWorldSize = EP_WORLD_SIZE
    t.tpWorldSize = 1
    t.epRankId = rank
    t.moeExpertNum = MOE_EXPERT_NUM
    t.moeExpertPerRankNum = LOCAL_MOE
    t.sharedExpertNum = 0
    t.sharedExpertRankNum = 0
    t.globalBs = GLOBAL_BS
    t.bs = BS
    t.k = K
    t.h = H
    t.a = A
    t.aivNum = BLOCK_DIM
    t.isMc2Context = True
    t.totalUbSize = 192 * 1024
    t.totalWinSizeEp = calc_total_win_size()
    t.bufferNum = 2
    return torch.frombuffer(bytearray(bytes(t)), dtype=torch.uint8)


def make_combine_inputs(rank):
    """构造 combine 的独立输入（不依赖 dispatch）
    关键：routing 必须覆盖全部 BS 个目标位置，否则 WaitDispatch 会死等
    """
    torch.manual_seed(100 + rank)

    expand_x = torch.randn(A, H, dtype=INPUT_DTYPE).npu()
    expert_ids = torch.zeros(BS, K, dtype=torch.int32).npu()

    # assist_info: 所有 token 路由到本 rank（全本地），覆盖位置 0 ~ BS-1
    self_send_cnt = BS
    assist_data = torch.zeros(A * 128, dtype=torch.int32)
    for i in range(self_send_cnt):
        assist_data[i * 3 + 0] = rank  # 路由到自己
        assist_data[i * 3 + 1] = i     # 目标位置 = i (覆盖 0~54)
        assist_data[i * 3 + 2] = 0     # topk_id
    assist_info = assist_data.npu()

    ep_send_counts = torch.zeros(EP_WORLD_SIZE * LOCAL_MOE, dtype=torch.int32)
    ep_send_counts[-1] = self_send_cnt
    ep_send_counts = ep_send_counts.npu()

    expert_scales = torch.ones(BS, K, dtype=torch.float32).npu()

    return expand_x, expert_ids, assist_info, ep_send_counts, expert_scales, self_send_cnt


def run(rank, ws, barrier_counter, barrier_lock):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = MASTER_PORT
    torch_npu.npu.set_device(rank % 2)
    dist.init_process_group(backend="hccl", rank=rank, world_size=ws,
                            init_method=f"tcp://127.0.0.1:{MASTER_PORT}")
    eg = dist.new_group(backend="hccl", ranks=list(range(ws)))
    en = eg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    print(f"[R{rank}] group={en}", flush=True)

    # Context
    tnp = os.path.dirname(os.path.abspath(torch_npu.__file__))
    cann = os.environ['ASCEND_HOME_PATH']
    ext = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../torch_extension/npu_ops_transformer'))
    ctx_m = jit_load(name='mc2_ctx_cb', sources=[f'{ext}/ops/csrc/comm_context.cpp'],
        extra_include_paths=[f'{tnp}/include', f'{tnp}/include/third_party/hccl/inc',
                            f'{tnp}/include/third_party/acl/inc', f'{cann}/include', f'{ext}/common/inc'],
        extra_ldflags=[f'-L{tnp}/lib', '-ltorch_npu'],
        extra_cflags=['-D_GLIBCXX_USE_CXX11_ABI=1'], verbose=False)
    mgr = ctx_m.CommContextManager(en, EP_WORLD_SIZE, "channel")
    context = mgr.create_context()
    print(f"[R{rank}] ctx OK", flush=True)

    # Load <<<>>>
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'csrc'))
    import mc2_direct_launch as mc2

    # 构造 combine 独立输入
    expand_x, expert_ids, assist_info, ep_send_counts, expert_scales, send_cnt = make_combine_inputs(rank)
    x_out = torch.zeros(BS, H, dtype=INPUT_DTYPE).npu()
    workspace = torch.zeros(WORKSPACE_SIZE // 4, dtype=torch.int32).npu()
    tiling = make_combine_tiling(rank)

    print(f"[R{rank}] combine <<<>>> (selfSendCnt={send_cnt})...", flush=True)
    mc2.mc2_combine(context, expand_x, expert_ids, assist_info,
                    ep_send_counts, expert_scales, TILING_KEY_BF16, BLOCK_DIM,
                    None, None, None, None, None, None,
                    x_out, workspace, tiling)
    torch.npu.synchronize()

    abs_sum = x_out.float().abs().sum().item()
    ok = (x_out.shape == (BS, H)) and (abs_sum > 0)
    print(f"[R{rank}] {'PASSED' if ok else 'FAILED'} abs_sum={abs_sum:.2f}", flush=True)

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
    print("=" * 50)
    print("MC2 <<<>>> Combine Standalone Test")
    print(f"bs={BS} h={H} k={K} ep={EP_WORLD_SIZE} bf16")
    print("=" * 50)

    tnp = os.path.dirname(os.path.abspath(torch_npu.__file__))
    cann = os.environ['ASCEND_HOME_PATH']
    ext = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../torch_extension/npu_ops_transformer'))
    print("Pre-compiling...", flush=True)
    jit_load(name='mc2_ctx_cb', sources=[f'{ext}/ops/csrc/comm_context.cpp'],
        extra_include_paths=[f'{tnp}/include', f'{tnp}/include/third_party/hccl/inc',
                            f'{tnp}/include/third_party/acl/inc', f'{cann}/include', f'{ext}/common/inc'],
        extra_ldflags=[f'-L{tnp}/lib', '-ltorch_npu'],
        extra_cflags=['-D_GLIBCXX_USE_CXX11_ABI=1'], verbose=False)
    print("Done.", flush=True)

    os.system("fuser -k /dev/npu* 2>/dev/null")
    time.sleep(2)
    barrier_counter = Value('i', 0)
    barrier_lock = Lock()
    ps = [Process(target=run, args=(r, EP_WORLD_SIZE, barrier_counter, barrier_lock)) for r in range(EP_WORLD_SIZE)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(timeout=45)
    ok = all(p.exitcode == 0 for p in ps)
    print("\n" + ("ALL PASSED" if ok else "TIMEOUT/FAILED"))
