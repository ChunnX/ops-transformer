# Ascend950 MTE versions

Current local checks and outstanding target-host verification are recorded in
[MTE_VALIDATION.md](MTE_VALIDATION.md).

The MTE implementation starts from operator commit
`17adcfad8949b1aa9f29ced0e685f3dc25831f28`. It uses the same public mathematical
APIs, BF16 input/output conventions, routing order and fixed GMM profiles.
Communication is a separate MTE implementation. The original UDMA package and
its dependency revision remain available through the default build profile.

## Independent build

On the Ascend950 Linux host, source the CANN and torch_npu environments. From
`experimental/mc2/catccos_udma_ops`, run:

```bash
export CATCCOS_SOURCE_PATH="$PWD/../../../3rdparty/catccos"
export SHMEM_SOURCE_PATH="$PWD/../../../3rdparty/shmem-mte"
export SHMEM_HOME_PATH="$SHMEM_SOURCE_PATH/install"
bash scripts/prepare_dependencies.sh --backend mte
bash scripts/build_wheel.sh --backend mte
python3 -m pip install --force-reinstall --no-deps dist/mte/*.whl
```

MTE pins CatCCOS to `878c2e0a504be8c49c03b5e1157a7c714a628e29` and SHMEM to
`aee374a396b389626f68fab74ec8c555e18978a7`. A separate SHMEM checkout prevents
preparing MTE from changing an existing UDMA dependency. The script rejects an
existing MTE checkout at another revision. Both headers and installed libraries
must match the pin and build provenance marker; local SHMEM modifications are
rejected. SHMEM uses the active PyTorch C++ ABI.
MTE configuration uses `llvm-nm` or `nm` to verify that `libshmem.so` exports the
pinned bootstrap symbol `g_boot_handle`; a missing symbol or inspection tool
stops the build before compiling the extension.

`--backend mte` creates only `catccos_mte_extension`, using an explicit list of
MTE sources. It neither builds nor imports the UDMA operator extension. The
official SHMEM library may contain other backend implementations, but the MTE
runtime enables only `ACLSHMEM_DATA_OP_MTE` and does not select a fallback.

Build, binary staging and Wheel output are isolated by backend:

| Backend | Build | Staged package | Wheel output |
|---|---|---|---|
| MTE | `build/mte/cmake` | `build/mte/cmake/stage/ops_transformer_catccos_mte` | `dist/mte` |
| UDMA | `build/udma/cmake` | `build/udma/cmake/stage/ops_transformer_catccos_udma` | `dist` |

The original command `bash scripts/build_wheel.sh` still selects UDMA and
preserves its `dist/*.whl` installation path. Cleaning either profile leaves the
other profile's products intact. UDMA retains its existing top-level
`validation` package and CLI entry points; MTE installs validation only inside
`ops_transformer_catccos_mte`, so the Wheels do not overwrite each other's files.

For the private communication-tuning bridge, build with:

```bash
bash scripts/build_wheel.sh --backend mte --with-tuning
```

Production builds reject tuning overrides. Tuning does not add parameters to
either public mathematical API. See the tuning package documentation for the
search workflow; each candidate uses the same mathematical entry points.

## Runtime and API

Run one process per NPU. Load one communication backend per process: the MTE
and UDMA extensions cannot safely share SHMEM globals, default device context
or differently pinned `libshmem.so` implementations. Use separate `torchrun`
jobs for comparisons. Loading a second backend gives an explicit error before
its extension is imported. Importing the MTE package or its CPU planning code
does not import torch_npu or load `_C`.
Host calls serialize stream selection and enqueue through one runtime launch
mutex. Switching streams waits for the previous stream, and group destruction
uses the same mutex before synchronizing and releasing shared workspace.

```python
import os
import torch
import torch_npu
import ops_transformer_catccos_mte as mte

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch_npu.npu.set_device(local_rank)
mte.init_mte_group(rank=rank, world_size=world_size, local_rank=local_rank)

# x, weight and both route tensors follow the existing UDMA mathematical API.
y, dispatched_x = mte.all_to_allv_grouped_mat_mul(
    x, weight, local_tokens, global_tokens,
    ep_size=world_size, expert_num=expert_num,
    received_rows=received_rows, output_all_to_allv=True,
)
returned = mte.grouped_mat_mul_all_to_allv(
    expert_x, return_weight, local_tokens, global_tokens,
    ep_size=world_size, expert_num=expert_num, out=reusable_output,
)
torch_npu.npu.synchronize()
mte.destroy_mte_group()
```

The mathematical functions have exactly the same signatures as the UDMA
facade, including `out`, optional AllToAllV export and host/device route inputs.
Inputs and outputs are ordinary contiguous BF16 NPU tensors. Only the remotely
addressable communication workspace uses symmetric SHMEM memory. Reuse output
and device route tensors to keep allocations and route transfers out of timing.

Bootstrap uses `MTE_MASTER_ADDR` and `MTE_MASTER_PORT`; defaults follow
`MASTER_ADDR` and `MASTER_PORT + 1`. The port must be distinct from torchrun's
rendezvous port. Hostnames are resolved to a numeric IPv4 address.

Initialization enables only MTE, reserves 400 MiB of data workspace plus two
64 KiB control areas, and clears that workspace. The default heap is 1 GiB.
Each operator keeps its barrier generation in the last 64 bytes of its control
area. That state is initialized once; per-call resets cover only transient
synchronization state. A private MTE barrier joins AIVs through hardware flag 8
and does not depend on SHMEM's fixed-size software AIV pool.
Before the first operator it verifies all peer MTE capabilities and performs
actual transfers on the mapped peer memory; a non-null pointer alone does not
establish reachability. Unsupported topology or failed transfers cause an error.
After an operation whose device completion cannot be established, restart all
workers; potentially in-flight memory is retained until process exit.

MTE signal scratch is explicitly configured at UB offset 0 with size 2048 and
event ID 7. Payload buffers begin after that area and use separate event IDs.
The communication budget also reserves 8192 bytes for system use.
Initialization checks the reported UB against the pinned 248 KiB kernel budget,
and each selected kernel checks its available communication AIVs. The physical
256 KiB UB capacity is not a claim that all bytes are usable for payload.

The low-level `empty_symmetric` helper remains available for interoperability.
Release all such tensors before destroying the group. Public mathematical
operators do not require this allocator.

## Correctness and timing

The MTE package contains the baseline validation cases and CPU FP32 Golden,
with backend-specific imports and profiler names:

```bash
ops-transformer-mte-accuracy
ops-transformer-mte-performance
ops-transformer-mte-accuracy \
  --cases "$PWD/ops_transformer_catccos_mte/validation/cases/grouped_mat_mul_all_to_allv.yaml"
```

Validation uses CPU Gloo for coordination. Profiled operator rows exclude
allocation and route preprocessing; each iteration is measured by the slowest
rank, followed by the median across iterations. BF16 correctness and rank-skew
gates follow the existing framework. MTE fingerprints record the backend and
dependency pins. Compare equivalent workloads and capture layouts in separate
processes; an imported baseline with incompatible fingerprints is rejected.

Source-level and CPU tests can run without a CANN installation. Compiling the
device extension, verifying generated UB allocation, checking all peer paths,
and establishing performance require the target Ascend950 host. No performance
gain is implied by passing CPU tests.
