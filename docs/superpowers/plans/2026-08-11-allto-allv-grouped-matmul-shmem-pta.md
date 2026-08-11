# AllToAllV GroupedMatMul SHMEM PTA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independently loadable torch bridge and a deterministic multi-process PTA driver that exercise the Ascend950 SHMEM `aclnnAlltoAllvGroupedMatMulV2` path end to end.

**Architecture:** Python owns process orchestration, ordinary NPU tensors, route generation, and the CPU FP32 golden. A C++ `TORCH_LIBRARY` bridge owns the per-process SHMEM runtime, copies ordinary NPU input into symmetric memory, allocates the symmetric aclnn workspace, calls V2 on the current torch_npu stream, synchronizes, barriers, and returns an ordinary NPU output.

**Tech Stack:** C++17, LibTorch dispatcher, torch_npu C++ runtime, AscendCL/aclnn, CANN SHMEM/UDMA, Python 3, `torch.multiprocessing`, CMake, Bash.

## Global Constraints

- Work only in `mc2/allto_allv_grouped_mat_mul/examples/arch35/pta` plus the existing SHMEM static checker and documentation.
- Preserve the approved V2 ABI: `gmmX`, `gmmWeight`, `shmemContext`, `epWorldSize`, `sendCounts`, `recvCounts`, `gmmY`.
- Support only BF16, ND, contiguous tensors, local expert count `1..64`, and SHMEM world size `1..8`.
- Do not initialize HCCL or call op-plugin's HCCL V1 wrapper.
- Allocate temporary `gmmX` with `aclshmem_malloc` and execution workspace with `aclshmem_calloc` from the initialized symmetric heap.
- Use the current torch_npu stream for D2D copy and aclnn launch; synchronize it before SHMEM barrier/free.
- Every rank must allocate and free symmetric objects in identical order; test cases keep local input byte size equal across ranks.
- Output order is local-expert-major then source-rank-minor.
- Precision follows the source verifier: BF16 `rtol=2^-7` for `K<2048`, otherwise `2^-6`, with normalized mean bias at most `2^-7`.
- The local environment may only claim syntax/static checks. Only an Ascend950 target environment may claim successful linking or multi-card precision.

---

## File Map

- Create `mc2/allto_allv_grouped_mat_mul/examples/arch35/pta/allto_allv_grouped_mat_mul_shmem_torch.cpp`: bridge state, validation, SHMEM lifecycle, ACL descriptor ownership, V2 call, torch schemas.
- Create `mc2/allto_allv_grouped_mat_mul/examples/arch35/pta/CMakeLists.txt`: standalone shared-library build and both SHMEM SDK layouts.
- Create `mc2/allto_allv_grouped_mat_mul/examples/arch35/pta/build_pta.sh`: environment validation and CMake invocation.
- Create `mc2/allto_allv_grouped_mat_mul/examples/arch35/pta/test_allto_allv_grouped_mat_mul_shmem.py`: pure route helpers, worker lifecycle, CPU golden, precision metrics, parent-process aggregation.
- Create `mc2/allto_allv_grouped_mat_mul/examples/arch35/pta/run_pta_test.sh`: timeout wrapper and CLI forwarding.
- Create `mc2/allto_allv_grouped_mat_mul/examples/arch35/pta/README.md`: target build/run contract and troubleshooting.
- Modify `mc2/allto_allv_grouped_mat_mul/tests/check_allto_allv_grouped_mat_mul_shmem_static.py`: offline contract checks for every new artifact.

---

### Task 1: SHMEM Torch Bridge

**Files:**
- Create: `mc2/allto_allv_grouped_mat_mul/examples/arch35/pta/allto_allv_grouped_mat_mul_shmem_torch.cpp`
- Modify: `mc2/allto_allv_grouped_mat_mul/tests/check_allto_allv_grouped_mat_mul_shmem_static.py`

**Interfaces:**
- Consumes: installed V2 aclnn header/library, SHMEM host API, current torch_npu device and stream.
- Produces: torch namespace `allto_allv_grouped_mat_mul_shmem_pta` with `init`, `run`, `barrier`, and `finalize`.

- [ ] **Step 1: Add the failing bridge contract check**

Add `check_pta_bridge()` to the static checker and append it to `main()`'s checks list. The check must read the new C++ path and require these exact behaviors:

```python
def check_pta_bridge() -> None:
    bridge = read("examples/arch35/pta/allto_allv_grouped_mat_mul_shmem_torch.cpp")
    for token in (
        "TORCH_LIBRARY(allto_allv_grouped_mat_mul_shmem_pta, m)",
        'm.def("init(int rank, int world_size, int heap_bytes, str ip_port) -> int")',
        'm.def("run(Tensor x, Tensor weight, int[] send_counts, int[] recv_counts) -> Tensor")',
        'm.def("barrier() -> int")',
        'm.def("finalize() -> int")',
        "aclshmemx_init_attr",
        "shmemx_get_ffts_config",
        "aclshmem_malloc",
        "aclshmem_calloc",
        "aclrtMemcpyAsync",
        "aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize",
        "aclnnAlltoAllvGroupedMatMulV2(",
        "aclrtSynchronizeStream",
        "aclshmem_barrier_all",
        "shmem_free",
    ):
        require(token in bridge, f"PTA bridge missing contract token: {token}")
    require("init_process_group" not in bridge and "Hccl" not in bridge and "HCCL" not in bridge,
            "PTA bridge must not initialize or call HCCL")
```

- [ ] **Step 2: Run the checker and observe the intended failure**

Run:

```bash
python3 mc2/allto_allv_grouped_mat_mul/tests/check_allto_allv_grouped_mat_mul_shmem_static.py
```

Expected: FAIL with `FileNotFoundError` for `examples/arch35/pta/allto_allv_grouped_mat_mul_shmem_torch.cpp`.

- [ ] **Step 3: Implement bridge state and lifecycle**

Implement a process-local state with exact-match idempotent initialization:

```cpp
struct BridgeState {
    bool initialized = false;
    int64_t rank = -1;
    int64_t worldSize = 0;
    int64_t heapBytes = 0;
    std::string ipPort;
    void *context = nullptr;
};

int64_t Init(int64_t rank, int64_t worldSize, int64_t heapBytes, const std::string &ipPort);
int64_t Barrier();
int64_t Finalize();
```

`Init` must validate rank/world/heap/IP, call `aclshmemx_set_conf_store_tls(false, nullptr, 0)`, fill `aclshmemx_init_attr_t` with `ACLSHMEM_DATA_OP_UDMA`, call `aclshmemx_init_attr`, validate `shmem_my_pe()`/`shmem_n_pes()`, and save the nonzero `shmemx_get_ffts_config()` pointer. `Barrier` calls `aclshmem_barrier_all()`. `Finalize` synchronizes the current stream when present, calls `shmem_finalize()`, clears all state, and is idempotent.

- [ ] **Step 4: Implement descriptors, validation, and `Run`**

Use non-owning ACL descriptors with owning descriptor guards:

```cpp
using TensorGuard = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;
using IntArrayGuard = std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)>;

at::Tensor Run(const at::Tensor &x, const at::Tensor &weight,
               at::IntArrayRef sendCounts, at::IntArrayRef recvCounts);
```

`Run` must:

1. reject uninitialized state, non-NPU/device mismatch, non-BF16, non-ND, non-contiguous, wrong ranks/shapes, invalid expert/world/count values, and `sum(sendCounts) != M` before symmetric allocation;
2. create `y = at::empty({sum(recvCounts), N}, x.options())`;
3. allocate symmetric input, copy `x` with `aclrtMemcpyAsync(..., ACL_MEMCPY_DEVICE_TO_DEVICE, currentStream)`;
4. create BF16 ND `aclTensor` descriptors and `aclIntArray` objects;
5. call V2 workspace query with `state.context` and `state.worldSize`;
6. allocate workspace through `aclshmem_calloc`, call V2 stage two, and rely on the non-repeatable executor's documented automatic release by stage two;
7. synchronize the current stream, barrier all ranks, then free workspace followed by symmetric input;
8. return ordinary NPU `y`.

Every ACL/SHMEM/aclnn status check must include API name, rank, and return code in `TORCH_CHECK`. A cleanup guard must ensure descriptors and any successfully allocated symmetric pointers are released on exceptions; collective failures remain protected by the outer process timeout.

- [ ] **Step 5: Register exact torch schemas**

```cpp
TORCH_LIBRARY(allto_allv_grouped_mat_mul_shmem_pta, m) {
    m.def("init(int rank, int world_size, int heap_bytes, str ip_port) -> int");
    m.def("run(Tensor x, Tensor weight, int[] send_counts, int[] recv_counts) -> Tensor");
    m.def("barrier() -> int");
    m.def("finalize() -> int");
    m.impl("init", &Init);
    m.impl("run", torch::kPrivateUse1, &Run);
    m.impl("barrier", &Barrier);
    m.impl("finalize", &Finalize);
}
```

- [ ] **Step 6: Run static checks**

Run the existing static checker with a real Python interpreter. Expected: all previous groups and `check_pta_bridge` pass.

- [ ] **Step 7: Commit**

```bash
git add mc2/allto_allv_grouped_mat_mul/examples/arch35/pta/allto_allv_grouped_mat_mul_shmem_torch.cpp \
        mc2/allto_allv_grouped_mat_mul/tests/check_allto_allv_grouped_mat_mul_shmem_static.py
git commit -m "test: add SHMEM PTA torch bridge"
```

---

### Task 2: Standalone Build

**Files:**
- Create: `mc2/allto_allv_grouped_mat_mul/examples/arch35/pta/CMakeLists.txt`
- Create: `mc2/allto_allv_grouped_mat_mul/examples/arch35/pta/build_pta.sh`
- Modify: `mc2/allto_allv_grouped_mat_mul/tests/check_allto_allv_grouped_mat_mul_shmem_static.py`

**Interfaces:**
- Consumes: `ASCEND_HOME_PATH`, one of `SHMEM_HOME_PATH`/`SHMEM_HOME`, `CUST_OPAPI_LIBRARY`, optional `CUST_OPAPI_INCLUDE_DIR`, optional `PYTHON_BIN`.
- Produces: `build/pta/liballto_allv_grouped_mat_mul_shmem_pta.so`.

- [ ] **Step 1: Add the failing build contract check**

Require `find_package(Torch REQUIRED)`, C++17, both SHMEM layouts, `torch_npu`, `ascendcl`, `libshmem`, `libshmem_utils`, and the explicit custom op-api library. Require the build script to validate variables with `${name:?message}` and query `torch.utils.cmake_prefix_path` from `PYTHON_BIN`.

- [ ] **Step 2: Run the checker and observe missing CMake/build files**

Expected: FAIL at the first missing build artifact.

- [ ] **Step 3: Implement `CMakeLists.txt`**

The CMake target must have this shape:

```cmake
cmake_minimum_required(VERSION 3.16)
project(allto_allv_grouped_mat_mul_shmem_pta LANGUAGES CXX)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
add_library(allto_allv_grouped_mat_mul_shmem_pta SHARED
    allto_allv_grouped_mat_mul_shmem_torch.cpp)
target_compile_features(allto_allv_grouped_mat_mul_shmem_pta PRIVATE cxx_std_17)
set_target_properties(allto_allv_grouped_mat_mul_shmem_pta PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
```

Resolve source SDK includes/libs under `${SHMEM_HOME_PATH}/shmem/{include,src,lib}` and integrated SDK includes/libs under `${SHMEM_HOME}/{include,src/device,build/lib}`. Resolve the V2 header from `CUST_OPAPI_INCLUDE_DIR`, `${ASCEND_HOME_PATH}/include`, or a sibling `include` directory derived from `CUST_OPAPI_LIBRARY`. Fail with the complete attempted path list if any dependency is absent.

- [ ] **Step 4: Implement `build_pta.sh`**

The script must use `set -euo pipefail`, resolve its own directory, validate required environment variables, query torch and torch_npu paths using `PYTHON_BIN`, and run:

```bash
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
  -DPython3_EXECUTABLE="$PYTHON_BIN" \
  -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PREFIX" \
  -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
  -DSHMEM_HOME_PATH="${SHMEM_HOME_PATH:-}" \
  -DSHMEM_HOME="${SHMEM_HOME:-}" \
  -DCUST_OPAPI_LIBRARY="$CUST_OPAPI_LIBRARY" \
  -DCUST_OPAPI_INCLUDE_DIR="${CUST_OPAPI_INCLUDE_DIR:-}"
cmake --build "$BUILD_DIR" --parallel "${BUILD_JOBS:-8}"
```

Print the absolute `.so` path after a successful build.

- [ ] **Step 5: Verify shell/static behavior**

Run `bash -n build_pta.sh`, the static checker, and a dependency-negative CMake invocation. Expected: shell syntax passes; static check passes; CMake fails early with a named missing dependency in the local non-CANN environment.

- [ ] **Step 6: Commit**

```bash
git add mc2/allto_allv_grouped_mat_mul/examples/arch35/pta/CMakeLists.txt \
        mc2/allto_allv_grouped_mat_mul/examples/arch35/pta/build_pta.sh \
        mc2/allto_allv_grouped_mat_mul/tests/check_allto_allv_grouped_mat_mul_shmem_static.py
git commit -m "build: add standalone SHMEM PTA bridge build"
```

---

### Task 3: Deterministic PTA Driver

**Files:**
- Create: `mc2/allto_allv_grouped_mat_mul/examples/arch35/pta/test_allto_allv_grouped_mat_mul_shmem.py`
- Modify: `mc2/allto_allv_grouped_mat_mul/tests/check_allto_allv_grouped_mat_mul_shmem_static.py`

**Interfaces:**
- Consumes: bridge `.so`, world/device/IP arguments, deterministic `balanced` or `uneven` route case.
- Produces: per-rank metrics and process exit code 0 only when every rank passes.

- [ ] **Step 1: Add a failing driver contract check**

Parse the Python file with `ast.parse`, reject `torch.distributed` and `init_process_group`, and require functions named:

```python
build_global_counts
rank_send_counts
rank_recv_counts
build_rank_input
build_rank_weight
build_golden
precision_metrics
worker
main
```

Require `mp.get_context("spawn")`, `torch.ops.load_library`, bridge `init/run/barrier/finalize`, structured result queue handling, and default world size 2.

- [ ] **Step 2: Run the checker and observe the missing driver failure**

Expected: FAIL with `FileNotFoundError` for the PTA Python file.

- [ ] **Step 3: Implement pure route helpers first**

Use standard-library-only route generation so `--self-check` can run without torch:

```python
def build_global_counts(case: str, world_size: int, experts: int,
                        tokens_per_route: int, seed: int) -> list[list[list[int]]]:
    counts = [[[tokens_per_route for _ in range(experts)]
               for _ in range(world_size)] for _ in range(world_size)]
    if case == "uneven":
        for src in range(world_size):
            rng = random.Random(seed + src)
            donor = rng.randrange(world_size * experts)
            receiver = rng.randrange(world_size * experts - 1)
            if receiver >= donor:
                receiver += 1
            flat = [value for peer in counts[src] for value in peer]
            flat[receiver] += flat[donor]
            flat[donor] = 0
            counts[src] = [flat[i:i + experts]
                           for i in range(0, len(flat), experts)]
    return counts
```

`rank_send_counts` returns `C[rank]` flattened; `rank_recv_counts` returns `C[src][rank][expert]` in source-major order. `--self-check` must verify equal per-source totals, transpose consistency, one zero route in every uneven source row, and expected output row count for all ranks.

- [ ] **Step 4: Implement deterministic tensors and CPU golden**

Import torch/torch_npu only in runtime functions after argument parsing. Build every source input on CPU from `case_seed + source * 1009`, build each destination weight from `case_seed + 100000 + destination * 1013`, quantize both to BF16, and move only the local rank's input/weight to its NPU.

`build_golden` must slice each source input by destination/expert prefix sums and concatenate:

```python
chunks = []
for expert in range(local_experts):
    for source in range(world_size):
        rows = route_slice(source_inputs[source], counts[source], rank, expert)
        chunks.append(rows.float().matmul(local_weight[expert].float()))
golden = torch.cat(chunks, dim=0) if chunks else torch.empty((0, n), dtype=torch.float32)
```

- [ ] **Step 5: Implement source-compatible precision metrics**

Reject shape mismatch and non-finite values. Compute:

```python
scale = torch.maximum(torch.ones_like(golden), golden.abs())
diff = actual.float() - golden.float()
rtol = 2 ** (-7) if k < 2048 else 2 ** (-6)
within = diff.abs() <= rtol * scale
mean_bias = (diff / scale).mean().abs().item() if diff.numel() else 0.0
passed = bool(within.all().item()) and mean_bias <= 2 ** (-7)
```

Return `passed`, max absolute error, max relative error, mean relative error, and normalized mean bias.

- [ ] **Step 6: Implement worker and parent orchestration**

Each worker sets its assigned device, loads the bridge, calls `init`, executes selected cases, calls `barrier` after every verification before raising a precision failure, writes a structured dict to the result queue, and calls `finalize` in `finally`. The parent starts one process per device, collects at most one result per rank, joins all workers, prints a rank/case summary, and returns nonzero for missing results, nonzero child exit code, bridge exception, or precision failure.

- [ ] **Step 7: Run offline driver checks**

Run:

```bash
python3 -m py_compile test_allto_allv_grouped_mat_mul_shmem.py
python3 test_allto_allv_grouped_mat_mul_shmem.py --self-check
python3 mc2/allto_allv_grouped_mat_mul/tests/check_allto_allv_grouped_mat_mul_shmem_static.py
```

Expected without torch: py_compile passes, self-check prints both case invariants as PASS, and all static groups pass.

- [ ] **Step 8: Commit**

```bash
git add mc2/allto_allv_grouped_mat_mul/examples/arch35/pta/test_allto_allv_grouped_mat_mul_shmem.py \
        mc2/allto_allv_grouped_mat_mul/tests/check_allto_allv_grouped_mat_mul_shmem_static.py
git commit -m "test: add deterministic SHMEM PTA driver"
```

---

### Task 4: Run Wrapper and Operator-Facing Documentation

**Files:**
- Create: `mc2/allto_allv_grouped_mat_mul/examples/arch35/pta/run_pta_test.sh`
- Create: `mc2/allto_allv_grouped_mat_mul/examples/arch35/pta/README.md`
- Modify: `mc2/allto_allv_grouped_mat_mul/tests/check_allto_allv_grouped_mat_mul_shmem_static.py`

**Interfaces:**
- Consumes: built `.so`, Python executable, driver arguments, `PTA_TIMEOUT_SECONDS`.
- Produces: bounded target-machine test command and complete handoff documentation.

- [ ] **Step 1: Add failing run/documentation checks**

Require `set -euo pipefail`, system `timeout`, a default 600-second bound, passthrough of all driver arguments, `.so` existence validation, and no non-loopback fixed IP. Require README sections for prerequisites, build, 2-card run, 1/4/8-card variants, output interpretation, timeout, and known offline limits.

- [ ] **Step 2: Run checker and observe missing file failure**

Expected: FAIL at `run_pta_test.sh` or `README.md`.

- [ ] **Step 3: Implement `run_pta_test.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
PTA_LIBRARY=${PTA_LIBRARY:-$SCRIPT_DIR/build/pta/liballto_allv_grouped_mat_mul_shmem_pta.so}
PTA_TIMEOUT_SECONDS=${PTA_TIMEOUT_SECONDS:-600}
test -f "$PTA_LIBRARY" || { echo "missing PTA library: $PTA_LIBRARY" >&2; exit 1; }
timeout --signal=TERM --kill-after=30 "${PTA_TIMEOUT_SECONDS}s" \
  "$PYTHON_BIN" "$SCRIPT_DIR/test_allto_allv_grouped_mat_mul_shmem.py" \
  --library "$PTA_LIBRARY" "$@"
```

- [ ] **Step 4: Write README with exact target commands**

Document environment setup, `bash build_pta.sh`, `bash run_pta_test.sh --world-size 2 --device-ids 0,1 --case all`, custom `--ip-port`, 1/4/8-card variants, expected PASS summary fields, nonzero conditions, and the fact that local static checks are not a build or hardware-pass claim.

- [ ] **Step 5: Verify scripts and full static suite**

Run `bash -n` on both shell scripts, Python py_compile/self-check, static checker, `git diff --check`, and scan new artifacts for user-specific absolute paths or non-loopback IP literals.

- [ ] **Step 6: Commit**

```bash
git add mc2/allto_allv_grouped_mat_mul/examples/arch35/pta/run_pta_test.sh \
        mc2/allto_allv_grouped_mat_mul/examples/arch35/pta/README.md \
        mc2/allto_allv_grouped_mat_mul/tests/check_allto_allv_grouped_mat_mul_shmem_static.py
git commit -m "docs: add SHMEM PTA run guide"
```

---

### Task 5: Final Offline Verification and Target Handoff

**Files:**
- Verify only: every file listed in the File Map.

**Interfaces:**
- Consumes: all previous task outputs.
- Produces: evidence-backed local verification summary plus explicit target-machine commands.

- [ ] **Step 1: Run all locally available checks fresh**

Run the bundled/available Python interpreter for static check, py_compile, and `--self-check`; run `bash -n` for both scripts; run `git diff --check`; inspect `git status --short --branch` and the commit list.

- [ ] **Step 2: Audit design-to-implementation coverage**

Confirm all four schemas, SHMEM allocation order, same-stream D2D/aclnn launch, stream sync before barrier/free, deterministic global count transpose, expert/source output ordering, source BF16 tolerance, timeout, and documentation are each present in code or tests.

- [ ] **Step 3: Record non-runnable checks honestly**

State that C++ compilation, `.so` linkage/load, and Ascend950 2-card precision were not executed locally. Provide these exact target commands:

```bash
cd mc2/allto_allv_grouped_mat_mul/examples/arch35/pta
bash build_pta.sh
bash run_pta_test.sh --world-size 2 --device-ids 0,1 --case all
```

- [ ] **Step 4: Present branch and commits**

Report the `a2ava5` worktree path, final commit hashes, modified files, fresh check outputs, and target-machine next action. Do not push or merge unless explicitly requested.
