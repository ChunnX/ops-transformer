# CatCCOS UDMA operators for PyTorch

This standalone project delivers the Ascend950 BF16
`alltoallvgroupedmatmul` UDMA kernel as an installable Wheel. The Wheel contains
the C++/AscendC extension (`_C.abi3.so`) and the SHMEM shared libraries required
at runtime. The actual device launch remains a Fast Kernel Launch expression:
`KernelAdapter<Kernel><<<blockDim, nullptr, stream>>>(...)`.

The CatCCOS dependency comes from the official
`https://gitcode.com/cann/catccos.git` repository and is pinned to
`878c2e0a504be8c49c03b5e1157a7c714a628e29`; its own CATLASS and SHMEM gitlinks
are used, so this project does not change the repository-wide CATLASS version.
The operator-local `csrc/catccos_overlay` directory contains only the
UDMA all-to-all-v headers that have not yet landed in that official revision.

## Build and install

On an Ascend950 build host, source the CANN and torch_npu environments first.
The build requires CMake 3.18.4 or newer from `PATH`; a system or Conda
installation is supported and the PyPI `cmake` package is not required.

```bash
cd experimental/mc2/catccos_udma_ops
bash scripts/prepare_dependencies.sh
export CATCCOS_SOURCE_PATH="$PWD/../../../3rdparty/catccos"
export SHMEM_HOME_PATH="$CATCCOS_SOURCE_PATH/3rdparty/shmem/install"
bash scripts/build_wheel.sh
python3 -m pip install --force-reinstall --no-deps dist/*.whl
```

`prepare_dependencies.sh` refuses an existing CatCCOS checkout at another
revision. SHMEM is built with `-soc_type Ascend950` and the same C++ ABI setting
as the active PyTorch installation.

## PyTorch API

Run one process per NPU. `UDMA_MASTER_PORT` must be reachable by every rank and
must differ from torchrun's rendezvous port; if omitted, the package uses
`MASTER_PORT + 1`.

```python
import os
import torch
import torch_npu
import ops_transformer_catccos_udma as udma

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch_npu.npu.set_device(local_rank)
udma.init_udma_group(rank=rank, world_size=world_size, local_rank=local_rank)

m, n, k, expert_num = 128, 256, 256, 8
device = f"npu:{local_rank}"
x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
weight = torch.randn(expert_num // world_size, k, n,
                     dtype=torch.bfloat16, device=device)
local_tokens, global_tokens = udma.build_rank_route_tensors(
    m, rank, world_size, expert_num, device
)

output = udma.all_to_allv_grouped_mat_mul(
    x, weight, local_tokens, global_tokens,
    ep_size=world_size, expert_num=expert_num,
)
torch_npu.npu.synchronize()
udma.destroy_udma_group()
```

`x` must be packed by global expert in the order described by
`local_tokens_per_expert`. `weight` is `[local_expert_num, k, n]`. The returned
capacity is `[world_size * m, n]`; valid rows are packed first and their count is
`global_tokens_per_local_expert.sum()`.

For a performance-critical repeated launch, remove the otherwise necessary
NPU-to-SHMEM staging copy and output allocation from the timed region:

```python
symmetric_x = udma.to_symmetric(x)  # once, or write the producer result here
out = torch.zeros(world_size * m, n, dtype=torch.bfloat16, device=device)
output = udma.all_to_allv_grouped_mat_mul(
    symmetric_x, weight, local_tokens, global_tokens,
    ep_size=world_size, expert_num=expert_num, out=out,
)
# Synchronize, delete all symmetric tensors, then finalize the group.
```

Symmetric allocation is a collective SHMEM operation. Every rank must call
`empty_symmetric`/`to_symmetric` in the same order with the same allocation
size. Views with a nonzero storage offset fall back to the safe staging path.

## Validation

The test reproduces the CatCCOS route algorithm, constructs an HCCL/PyTorch
golden result, validates both normal and zero-staging paths, then measures the
same launch-only scope as the original example:

```bash
NPROC_PER_NODE=2 bash scripts/run_validation.sh \
  --m 128 --n 256 --k 256 --expert-num 8
```

To make no-regression gating explicit, pass the original example's latency and
an allowed ratio (zero by default):

```bash
NPROC_PER_NODE=2 bash scripts/run_validation.sh \
  --reference-us 123.4 --max-regression 0.0
```

Current kernel constraints are BF16, TP=1 (`ep_size == world_size`), at most 8
ranks, at most 64 local experts, contiguous row-major tensors, and the original
CatCCOS peer-memory/tiling limit.
