"""
MC2 <<<>>> Direct Launch Test
dispatch_v2 单独测试：bisheng -xasc 编译 + <<<>>> kernel 直调 + context 通信
"""
import os
import sys
import time
import ctypes
import torch
import torch.distributed as dist
import torch_npu
from torch.multiprocessing import Process

# ==================== Config ====================
BS = 55
H = 1038
K = 1
EP_WORLD_SIZE = 2
MOE_EXPERT_NUM = 2
LOCAL_MOE_EXPERT_NUM = 1
GLOBAL_BS = BS * EP_WORLD_SIZE
A = GLOBAL_BS * LOCAL_MOE_EXPERT_NUM
BLOCK_DIM = 32
TILING_KEY_BF16 = 1
INPUT_DTYPE = torch.bfloat16
MASTER_PORT = "50066"
RANK_PER_DEV = 2
WORKSPACE_SIZE = 16 * 1024 * 1024
WIN_ADDR_ALIGN = 512


def calc_total_win_size():
    """window = [combine区 + dispatch区] × 2(双缓冲)"""
    h_bytes = H * 2
    token_combine = ((h_bytes + WIN_ADDR_ALIGN - 1) // WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN
    token_dispatch = ((h_bytes + 32 - 1) // 32 * 32 + 44 + WIN_ADDR_ALIGN - 1) // WIN_ADDR_ALIGN * WIN_ADDR_ALIGN
    combine_area = A * (K + 0) * token_combine
    dispatch_area = A * K * token_dispatch
    return (combine_area + dispatch_area) * 2


class DispatchTilingInfo(ctypes.Structure):
    _fields_ = [
        ("epWorldSize", ctypes.c_uint32),
        ("tpWorldSize", ctypes.c_uint32),
        ("epRankId", ctypes.c_uint32),
        ("tpRankId", ctypes.c_uint32),
        ("expertShardType", ctypes.c_uint32),
        ("sharedExpertNum", ctypes.c_uint32),
        ("sharedExpertRankNum", ctypes.c_uint32),
        ("moeExpertNum", ctypes.c_uint32),
        ("quantMode", ctypes.c_uint32),
        ("globalBs", ctypes.c_uint32),
        ("bs", ctypes.c_uint32),
        ("k", ctypes.c_uint32),
        ("h", ctypes.c_uint32),
        ("a", ctypes.c_uint32),
        ("aivNum", ctypes.c_uint32),
        ("isTokenMask", ctypes.c_bool),
        ("isExpertMask", ctypes.c_bool),
        ("hasElasticInfo", ctypes.c_bool),
        ("isPerformance", ctypes.c_bool),
        ("isQuant", ctypes.c_bool),
        ("isMc2Context", ctypes.c_bool),
        ("reserved1", ctypes.c_bool),
        ("reserved2", ctypes.c_bool),
        ("totalUbSize", ctypes.c_uint64),
        ("totalWinSizeEp", ctypes.c_uint64),
        ("totalWinSizeTp", ctypes.c_uint64),
        ("expertTokenNumsType", ctypes.c_uint32),
        ("zeroComputeExpertNum", ctypes.c_int32),
        ("maxSizeForUbBuffer", ctypes.c_uint32),
        ("scalesRow", ctypes.c_uint64),
        ("scalesCol", ctypes.c_uint64),
        ("scalesTypeSize", ctypes.c_uint32),
        ("scalesCount", ctypes.c_uint64),
    ]


def make_dispatch_tiling(rank):
    t = DispatchTilingInfo()
    t.epWorldSize = EP_WORLD_SIZE
    t.tpWorldSize = 1
    t.epRankId = rank
    t.moeExpertNum = MOE_EXPERT_NUM
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
    t.expertTokenNumsType = 1
    return torch.frombuffer(bytearray(bytes(t)), dtype=torch.uint8)


def run_worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = MASTER_PORT
    torch_npu.npu.set_device(rank % RANK_PER_DEV)

    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size,
                            init_method=f"tcp://127.0.0.1:{MASTER_PORT}")
    ep_group = dist.new_group(backend="hccl", ranks=list(range(world_size)))
    ep_name = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    print(f"[R{rank}] group={ep_name}", flush=True)

    # Context
    from torch.utils.cpp_extension import load as jit_load
    torch_npu_path = os.path.dirname(os.path.abspath(torch_npu.__file__))
    cann_path = os.environ.get('ASCEND_HOME_PATH', '/home/l00946875/Ascend/cann-9.1.0')
    ext_root = '/home/l00946875/ops-transformer/torch_extension/npu_ops_transformer'
    ctx_mod = jit_load(
        name='mc2_ctx_dl', sources=[f'{ext_root}/ops/csrc/comm_context.cpp'],
        extra_include_paths=[f'{torch_npu_path}/include', f'{torch_npu_path}/include/third_party/hccl/inc',
                            f'{torch_npu_path}/include/third_party/acl/inc', f'{cann_path}/include',
                            f'{ext_root}/common/inc'],
        extra_ldflags=[f'-L{torch_npu_path}/lib', '-ltorch_npu'],
        extra_cflags=['-D_GLIBCXX_USE_CXX11_ABI=1'], verbose=False)
    mgr = ctx_mod.CommContextManager(ep_name, EP_WORLD_SIZE, "channel")
    context = mgr.create_context()
    print(f"[R{rank}] ctx OK buf={mgr.ccl_buffer_size}", flush=True)

    # Load <<<>>> module
    so_dir = os.path.join(os.path.dirname(__file__), '..', 'build', 'csrc')
    sys.path.insert(0, so_dir)
    import mc2_direct_launch as mc2
    print(f"[R{rank}] <<<>>> module loaded", flush=True)

    # Input
    torch.manual_seed(42 + rank)
    x = torch.randn(BS, H, dtype=INPUT_DTYPE).npu()
    expert_ids = torch.randint(0, MOE_EXPERT_NUM, (BS, K), dtype=torch.int32).npu()

    # Output tensors
    expand_x = torch.zeros(A, H, dtype=INPUT_DTYPE).npu()
    dynamic_scales = torch.zeros(A, dtype=torch.float32).npu()
    assist_info = torch.zeros(A * 128, dtype=torch.int32).npu()
    expert_token_nums = torch.zeros(LOCAL_MOE_EXPERT_NUM, dtype=torch.int64).npu()
    ep_recv_counts = torch.zeros(EP_WORLD_SIZE * LOCAL_MOE_EXPERT_NUM, dtype=torch.int32).npu()
    tp_recv_counts = torch.zeros(1, dtype=torch.int32).npu()
    expand_scales = torch.zeros(A, dtype=torch.float32).npu()
    workspace = torch.zeros(WORKSPACE_SIZE // 4, dtype=torch.int32).npu()
    tiling = make_dispatch_tiling(rank)

    # <<<>>> Dispatch
    print(f"[R{rank}] dispatch <<<>>>...", flush=True)
    mc2.mc2_dispatch(context, x, expert_ids, TILING_KEY_BF16, BLOCK_DIM,
                     None, None, None, None, None,
                     expand_x, dynamic_scales, assist_info, expert_token_nums,
                     ep_recv_counts, tp_recv_counts, expand_scales, workspace, tiling)
    torch.npu.synchronize()

    # 验证 dispatch
    abs_sum = expand_x.float().abs().sum().item()
    recv = ep_recv_counts.cpu().tolist()
    tokens = expert_token_nums.cpu().tolist()
    ok = (expand_x.shape == (A, H)) and (abs_sum > 0) and (sum(recv) > 0)
    print(f"[R{rank}] {'PASSED' if ok else 'FAILED'} "
          f"abs_sum={abs_sum:.2f} ep_recv={recv} tokens={tokens}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    print("=" * 50)
    print("MC2 <<<>>> Dispatch Direct Launch Test")
    print(f"bs={BS} h={H} k={K} ep={EP_WORLD_SIZE} bf16")
    print("=" * 50)

    # 预编译 context 扩展
    print("Pre-compiling ctx extension...", flush=True)
    torch_npu_path = os.path.dirname(os.path.abspath(torch_npu.__file__))
    cann_path = os.environ.get('ASCEND_HOME_PATH', '/home/l00946875/Ascend/cann-9.1.0')
    ext_root = '/home/l00946875/ops-transformer/torch_extension/npu_ops_transformer'
    from torch.utils.cpp_extension import load as jit_load
    jit_load(name='mc2_ctx_dl', sources=[f'{ext_root}/ops/csrc/comm_context.cpp'],
        extra_include_paths=[f'{torch_npu_path}/include', f'{torch_npu_path}/include/third_party/hccl/inc',
                            f'{torch_npu_path}/include/third_party/acl/inc', f'{cann_path}/include',
                            f'{ext_root}/common/inc'],
        extra_ldflags=[f'-L{torch_npu_path}/lib', '-ltorch_npu'],
        extra_cflags=['-D_GLIBCXX_USE_CXX11_ABI=1'], verbose=False)
    print("Done.", flush=True)

    os.system("fuser -k /dev/npu* 2>/dev/null")
    time.sleep(2)
    procs = [Process(target=run_worker, args=(r, EP_WORLD_SIZE)) for r in range(EP_WORLD_SIZE)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
    ok = all(p.exitcode == 0 for p in procs)
    print("\n" + ("ALL PASSED" if ok else "FAILED"))
