# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.hiascend.com/license/
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------

"""engram_fetch 端到端验证用例（PTA ElasticBuffer 全链路：HCCL 组 → 通信上下文构建 → aclnn → kernel）。

通过 torchrun 以多进程方式启动，每个进程绑定一张 NPU，组成一个 HCCL 通信组，
验证 engram_fetch 的两种取数模式端到端功能是否正确：

  - fetch_mode=0（sliced，切片跨 NPU 通信，默认，推理/训练均支持）:
      全局表按 rank 均匀切片分存，各 rank 经 engram_write 写入本 rank 分片，
      fetch 任意全局 index 时由持有该 entry 的远端 rank 通过跨 NPU 通信取回。
  - fetch_mode=1（local，全量本 NPU，仅推理）:
      全局全量表冗余到每个 NPU，各 rank 经 engram_write 写入同一份全量表，
      fetch 任意全局 index 均在本卡本地完成，无任何跨 NPU 通信。

验证项（与 CPU golden 比对）：
  1. 推理:   engram_fetch 结果与 CPU 全表 embedding gather 逐行完全一致（两种模式）；
  2. 训练:   仅 sliced 模式。前向结果一致 + engram_fetch_grad 输出的
             (grad_unique, unique_local_entry) 与 CPU 去重 scatter-add 聚合一致
             （各 rank 得到其所持分片 entry 的跨 rank 全局聚合梯度）；
  3. 零拷贝: with_grad 模式下原地更新 storage 后, 立即 fetch 可读到更新后的数据。

运行方式（详见 README.md，或直接使用 run_example.sh）:
  torchrun --nproc_per_node=2 test_engram_fetch_e2e.py --fetch-mode local
  torchrun --nproc_per_node=2 test_engram_fetch_e2e.py --fetch-mode sliced --phase train
"""

import argparse
import os
import sys
import traceback

import torch
import torch_npu
import torch.distributed as dist

from cann_ops_transformer import ElasticBuffer

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser(description="engram_fetch end-to-end verification")
    parser.add_argument(
        "--fetch-mode",
        choices=["sliced", "local", "0", "1"],
        default="sliced",
        help="engram fetch mode: sliced(0)=全局表按rank切片+跨NPU通信(默认), local(1)=全量表冗余到每个NPU纯本地取数",
    )
    parser.add_argument(
        "--phase",
        choices=["infer", "train", "both"],
        default="both",
        help="验证阶段: infer=仅推理, train=仅训练(含grad与零拷贝), both=全部(默认)",
    )
    parser.add_argument("--dtype", choices=list(_DTYPE_MAP.keys()), default="bf16", help="表数据类型, 默认 bf16")
    parser.add_argument("--num-entries", type=int, default=4096, help="全局表 entry 总行数(需能被rank数整除), 默认 4096")
    parser.add_argument("--hidden", type=int, default=256, help="隐藏维大小, 默认 256")
    parser.add_argument("--num-tokens", type=int, default=512, help="每 rank 参与取数的 token 数, 默认 512")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子(全量表各rank生成一致, indices/grad按rank区分), 默认 2026")
    args = parser.parse_args()
    args.fetch_mode = "sliced" if args.fetch_mode in ("sliced", "0") else "local"
    return args


def fetch_mode_value(mode_str):
    if mode_str in ("sliced", "0"):
        return ElasticBuffer.FETCH_MODE_SLICED
    return ElasticBuffer.FETCH_MODE_LOCAL


def gen_table(seed, num_rows, hidden, dtype):
    """按固定种子生成数据表, 保证各 rank 的全量表完全一致。"""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(num_rows, hidden, generator=g, dtype=torch.float32).to(dtype)


def gen_indices(seed, num_entries, num_tokens, pool_size=64):
    """生成 int32 索引: 一半来自小池(制造重复 entry, 触发去重聚合), 一半全局均匀。"""
    pool = min(pool_size, num_entries)
    g = torch.Generator().manual_seed(seed)
    n_pool = max(num_tokens // 2, 1) if num_tokens >= 2 else num_tokens
    n_uni = num_tokens - n_pool
    dup = torch.randint(0, pool, (n_pool,), generator=g, dtype=torch.int64)
    uni = torch.randint(0, num_entries, (n_uni,), generator=g, dtype=torch.int64)
    idx = torch.cat([dup, uni])
    return idx[torch.randperm(idx.numel(), generator=g)].to(torch.int32)


class Checker:
    def __init__(self, rank):
        self.rank = rank
        self.failures = []

    def check(self, name, ok, detail=""):
        tag = "PASS" if ok else "FAIL"
        print(f"[rank {self.rank}] [{tag}] {name}" + (f" ({detail})" if detail else ""), flush=True)
        if not ok:
            self.failures.append(name)

    def check_equal(self, name, actual, expected):
        self.check(name, torch.equal(actual, expected),
                   f"max_abs_diff={(actual.float() - expected.float()).abs().max().item() if actual.numel() else 0}")

    def check_close(self, name, actual, expected, rtol, atol):
        ok = torch.allclose(actual.float(), expected.float(), rtol=rtol, atol=atol)
        diff = (actual.float() - expected.float()).abs().max().item() if actual.numel() else 0.0
        self.check(name, ok, f"max_abs_diff={diff}, rtol={rtol}, atol={atol}")

    def ok(self):
        return not self.failures


def rank_storage(full_table, rank, world_size, fetch_mode):
    """按模式返回本 rank 应写入的表: sliced=本rank分片(连续), local=全量表。"""
    if fetch_mode == ElasticBuffer.FETCH_MODE_LOCAL:
        return full_table
    rows_per_rank = full_table.size(0) // world_size
    return full_table[rank * rows_per_rank:(rank + 1) * rows_per_rank]


def run_infer_phase(args, group, rank, world_size, device, checker, full_table):
    fetch_mode = fetch_mode_value(args.fetch_mode)
    storage = rank_storage(full_table, rank, world_size, fetch_mode).contiguous()
    num_cpu_bytes = ElasticBuffer.get_engram_storage_size_hint(storage.size(0), args.hidden, full_table.dtype)

    buf = ElasticBuffer(group, num_cpu_bytes=num_cpu_bytes, explicitly_destroy=True, fetch_mode=fetch_mode)
    buf.engram_write(storage)

    indices = gen_indices(args.seed + 10 + rank, full_table.size(0), args.num_tokens)
    fetched = buf.engram_fetch(indices.to(device))()

    torch.npu.synchronize()
    golden = full_table[indices.long()]
    checker.check_equal(f"[infer/{args.fetch_mode}] fetch 结果与全表 gather golden 一致", fetched.cpu(), golden)

    buf.destroy()


def golden_grad_sliced(rank, world_size, rows_per_rank, idx_cpu_list, grad_cpu_list, hidden, dtype):
    """mode=0 golden: 本 rank 分片 range 内, 聚合所有 rank token 的梯度(按 local entry 去重升序)。"""
    begin = rank * rows_per_rank
    acc = {}
    for src in range(world_size):
        idx_src = idx_cpu_list[src]
        for t in range(idx_src.numel()):
            gid = int(idx_src[t])
            if begin <= gid < begin + rows_per_rank:
                acc.setdefault(gid - begin, []).append(grad_cpu_list[src][t])
    ids = sorted(acc.keys())
    if not ids:
        return torch.empty(0, dtype=torch.int32), torch.empty(0, hidden, dtype=dtype)
    vals = torch.stack([torch.stack(acc[i]).sum(0) for i in ids]).to(dtype)
    return torch.tensor(ids, dtype=torch.int32), vals


def run_train_phase(args, group, rank, world_size, device, checker, full_table):
    # 训练场景仅支持 sliced 模式(fetch_mode=0); local 模式(全量本NPU)仅支持推理
    fetch_mode = ElasticBuffer.FETCH_MODE_SLICED
    mode_name = "sliced"
    rows_per_rank = full_table.size(0) // world_size

    shard = rank_storage(full_table, rank, world_size, fetch_mode)
    storage = torch.empty_like(shard, pin_memory=True)
    storage.copy_(shard)

    num_cpu_bytes = ElasticBuffer.get_engram_storage_size_hint(storage.size(0), args.hidden, full_table.dtype)
    buf = ElasticBuffer(group, num_cpu_bytes=num_cpu_bytes, with_grad=True,
                        num_max_tokens_per_rank=args.num_tokens,
                        explicitly_destroy=True, fetch_mode=fetch_mode)
    buf.engram_write(storage)  # with_grad: 零拷贝映射, storage 地址需保持不变

    indices = gen_indices(args.seed + 100 + rank, full_table.size(0), args.num_tokens)
    fetched, ctx = buf.engram_fetch(indices.to(device))()

    torch.npu.synchronize()
    checker.check_equal(f"[train/{mode_name}] 前向 fetched 与全表 gather golden 一致",
                        fetched.cpu(), full_table[indices.long()])

    g = torch.Generator().manual_seed(args.seed + 200 + rank)
    grad_fetched = torch.randn(args.num_tokens, args.hidden, generator=g, dtype=torch.float32)
    grad_fetched_npu = grad_fetched.to(full_table.dtype).to(device)

    grad_unique, unique_local_entry = buf.engram_fetch_grad(grad_fetched_npu, ctx)
    torch.npu.synchronize()

    idx_gather = [torch.zeros(args.num_tokens, dtype=torch.int64, device=device) for _ in range(world_size)]
    grad_gather = [torch.zeros(args.num_tokens, args.hidden, dtype=full_table.dtype, device=device)
                   for _ in range(world_size)]
    dist.all_gather(idx_gather, indices.long().to(device))
    dist.all_gather(grad_gather, grad_fetched_npu)
    idx_cpu_list = [t.cpu() for t in idx_gather]
    grad_cpu_list = [t.cpu().float() for t in grad_gather]

    golden_ids, golden_vals = golden_grad_sliced(rank, world_size, rows_per_rank,
                                                 idx_cpu_list, grad_cpu_list,
                                                 args.hidden, full_table.dtype)
    checker.check_equal(f"[train/{mode_name}] unique_local_entry 与 golden 去重索引一致",
                        unique_local_entry.cpu(), golden_ids)
    if golden_ids.numel() > 0:
        rtol = atol = 1e-2 if full_table.dtype != torch.float32 else 1e-4
        checker.check_close(f"[train/{mode_name}] grad_unique 与 golden 梯度聚合一致",
                            grad_unique.cpu(), golden_vals, rtol, atol)
    else:
        checker.check(f"[train/{mode_name}] grad_unique 为空表(无命中)", grad_unique.cpu().numel() == 0)

    updated_full = gen_table(args.seed + 300, full_table.size(0), args.hidden, full_table.dtype)
    storage.copy_(rank_storage(updated_full, rank, world_size, fetch_mode))  # 原地更新, 地址不变
    buf.barrier(with_cpu_sync=True)

    indices_upd = gen_indices(args.seed + 400 + rank, full_table.size(0), args.num_tokens)
    fetched_upd = buf.engram_fetch(indices_upd.to(device))()
    torch.npu.synchronize()
    checker.check_equal(f"[train/{mode_name}] 零拷贝: storage 原地更新后 fetch 可见新数据",
                        fetched_upd.cpu(), updated_full[indices_upd.long()])

    buf.destroy()


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if world_size < 2:
        print("[ERROR] engram_fetch 验证需要至少 2 个 rank 组成 HCCL 通信组", flush=True)
        sys.exit(1)
    if args.num_entries % world_size != 0:
        print(f"[ERROR] --num-entries({args.num_entries}) 需能被 rank 数({world_size})整除", flush=True)
        sys.exit(1)
    if args.fetch_mode == "local" and args.phase == "train":
        print("[ERROR] local 模式(fetch_mode=1)仅支持推理场景, 不支持训练(--phase train)", flush=True)
        sys.exit(1)
    if args.fetch_mode == "local" and args.phase == "both":
        print("[INFO] local 模式(fetch_mode=1)仅支持推理场景, 自动跳过训练阶段验证", flush=True)
        args.phase = "infer"

    torch_npu.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group(backend="hccl")
    group = dist.new_group(backend="hccl", ranks=list(range(world_size)))

    checker = Checker(rank)
    full_table = gen_table(args.seed, args.num_entries, args.hidden, _DTYPE_MAP[args.dtype])

    try:
        if args.phase in ("infer", "both"):
            run_infer_phase(args, group, rank, world_size, device, checker, full_table)
        if args.phase in ("train", "both"):
            run_train_phase(args, group, rank, world_size, device, checker, full_table)
    except Exception:
        print(f"[rank {rank}] [FAIL] 执行异常:\n{traceback.format_exc()}", flush=True)
        dist.destroy_process_group()
        sys.exit(1)

    dist.barrier()
    if checker.ok():
        print(f"[rank {rank}] engram_fetch({args.fetch_mode}, {args.dtype}) 端到端验证全部通过", flush=True)
    else:
        print(f"[rank {rank}] [FAIL] 失败项: {checker.failures}", flush=True)
    dist.destroy_process_group()
    sys.exit(0 if checker.ok() else 1)


if __name__ == "__main__":
    main()
