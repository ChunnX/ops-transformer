# Ascend950 AllToAllV GroupedMatMul SHMEM Migration Design

## Background

Migrate the BF16 Ascend950 AllToAllV-GroupedMatMul UDMA implementation from
`D:/1Code/catccos` branch `mingzhea2a_udma` into
`mc2/allto_allv_grouped_mat_mul` in ops-transformer. The source reference is
commit `1e04165` and the target baseline is commit `95d31c5a4`.

The source implementation uses CANN SHMEM with UDMA for remote transfers and
MTE for the self transfer. The target implementation currently uses HCCL. The
new implementation must preserve the source algorithm and communication
schedule while integrating with the ops-transformer aclnn, tiling, build, and
test frameworks.

## Goals

- Replace the existing Ascend950 implementation with a self-contained SHMEM
  implementation derived from the source algorithm.
- Change the public V2 aclnn API to accept an initialized SHMEM context pointer
  instead of HCCL `group` and `commMode` arguments.
- Support the source feature set: BF16, AllToAllV followed by GroupedMatMul,
  TP=1, EP equal to the SHMEM rank count, and EP in the inclusive range 1-8.
- Preserve the existing A3/HCCL implementation and V1 behavior on A3.
- Avoid a build-time or runtime dependency on the sibling catccos checkout.
- Provide host, tiling, kernel, multi-rank precision, and profiler validation.

## Non-goals

- Preserving V2 ABI compatibility.
- Supporting V2 on A3.
- Supporting V1 on Ascend950 after the A5 implementation is replaced.
- Supporting FP16, shared-expert MatMul, transposed weights, or `permuteOut` in
  the new V2 API.
- Importing CATCCOS or CATLASS as a third-party dependency.
- Internally allocating or owning the caller's SHMEM context, input, or
  workspace.

## Public V2 API

The V2 API is intentionally ABI-breaking and is reduced to the source feature
set:

```cpp
ACLNN_API aclnnStatus aclnnAlltoAllvGroupedMatMulV2GetWorkspaceSize(
    const aclTensor *gmmX,
    const aclTensor *gmmWeight,
    void *shmemContext,
    int64_t epWorldSize,
    const aclIntArray *sendCounts,
    const aclIntArray *recvCounts,
    aclTensor *gmmY,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnAlltoAllvGroupedMatMulV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

### Tensor and route-table contract

- `gmmX` has shape `[M, K]`, dtype BF16, ND layout, and storage allocated from
  the initialized SHMEM heap.
- `gmmWeight` has shape `[E, K, N]`, dtype BF16, and ND layout. `E` is the
  number of local experts.
- `sendCounts` has length `EP * E`. Its flattened order is destination rank,
  then destination-local expert. It maps to the source implementation's
  `localTokensPerExpert` table. Every value is nonnegative and the sum is `M`.
- `recvCounts` has length `EP * E`. Its flattened order is source rank, then
  local expert. It maps to the source implementation's
  `globalTokensPerLocalExpert` table. Every value is nonnegative.
- `gmmY` has shape `[sum(recvCounts), N]`, dtype BF16, and ND layout.
- `shmemContext` is the initialized device context/FFTS configuration address
  returned by the caller's SHMEM initialization flow. The API borrows this
  pointer and never frees it.

The op_api layer wraps `shmemContext` in a non-owning internal `aclTensor`,
following the existing MC2 context-tensor pattern. Public V2 dispatches to a
dedicated internal `AlltoAllvGroupedMatMulShmem` op whose required
`shmem_context` input replaces the HCCL group metadata. The existing internal
`AlltoAllvGroupedMatMul` op remains isolated for V1/A3 so its
`MC2().HcclGroup({"group"})` registration cannot accidentally initialize HCCL
for a SHMEM executor. The A5 kernel uses the received address to set the SHMEM
synchronization base before any SHMEM operation.

## Memory Contract

The V2 execution workspace must be allocated by the caller with
`aclshmem_calloc` from the same initialized SHMEM heap as `gmmX`. The workspace
size returned by `GetWorkspaceSize` is the source implementation's fixed
registered window:

```text
200 * 1024 * 1024 BF16 elements * 2 bytes + 4096 bytes = 400 MiB + 4096 bytes
```

The 400 MiB region holds the two-stage distributed input window. The final
page stores synchronization words and alignment padding. The combined SHMEM
allocation for `gmmX` and workspace must not exceed the source limit of
1004 MiB.

No public API available to op_api reliably proves that an arbitrary pointer
belongs to the SHMEM heap. Pointer provenance is therefore a documented caller
contract. Host code validates nullness, shapes, sizes, and the declared
workspace size, but does not claim to validate SHMEM ownership.

## Architecture

### Integration boundary

The implementation stays inside `mc2/allto_allv_grouped_mat_mul` and uses the
existing ops-transformer registration, tiling-data serialization, binary
generation, and aclnn executor flow. A focused Ascend950 kernel subtree will
contain native AscendC implementations of the source communication scheduler,
UDMA copy helper, route-table mapping, and AIC/AIV pipeline. It will reuse
appropriate target-repository MatMul primitives where their behavior matches
the source, but it will not depend on CATCCOS or CATLASS headers.

The existing A3 HCCL kernel and tiling implementation remain available under
their current op type and tiling keys. Ascend950 V2 selects only the dedicated
SHMEM op and tiling key; the SHMEM op is registered only for Ascend950.

### A5 tiling

The A5 tiling path validates the public contract and emits all scalar values and
route tables required by the kernel. It selects between the two source tile
families:

- `M0=128, N0=256, K0=256`
- `M0=256, N0=128, K0=256`

Communication constants match the source:

- configured communication tile M: 64 (the two AIV subcores process 32 rows each)
- communication block M: 64
- communication interval: 10
- workspace stages: 2
- UB stages: 2
- maximum peer blocks: 2

The tiler enforces:

- `epWorldSize` is in the inclusive range 1-8;
- SHMEM rank size equals `epWorldSize` at kernel execution;
- `E > 0`, `E <= 64`, and global expert count is `EP * E`;
- `sendCounts` and `recvCounts` both contain exactly `EP * E` nonnegative
  values;
- `sum(sendCounts) == M`;
- BF16 input, weight, and output with compatible K and N dimensions;
- fixed workspace and per-peer UDMA window bounds from the source checks.

The serialized tiling data owns copies of both counts arrays. Kernel code uses
their addresses directly as the source route tables; no extra device-side
counts tensor or allocation is introduced.

The SHMEM tiling payload is a fixed-width structure. Counts are validated as
nonnegative `int64_t` host values and narrowed to `uint32_t` only after an
overflow check. With at most `8 * 64 = 512` route entries, the two embedded
tables occupy 4096 bytes. Scalar dimensions, stage offsets, communication
constants, and the target-repository BF16 GroupedMatMul tiling data follow the
tables. A compile-time size assertion must keep the complete payload within the
ops-transformer raw tiling buffer.

### On-chip resource plan

Remote UDMA uses `PIPE_S` to publish byte-copy WQEs and passes a null UB scratch
pointer, matching the source. It therefore consumes no operator-owned UB and
does not share the kernel's MTE event IDs. The self-copy path retains the
source bounded MTE tile of `32 x 256` BF16 elements (16 KiB) with two UB stages,
for a maximum communication UB allocation of 32 KiB. Both addresses and sizes
are 32-byte aligned.

| Resource | Per-stage size | Stages | Maximum | Purpose |
|---|---:|---:|---:|---|
| AIV local-copy UB | `32 * 256 * 2 = 16384` bytes | 2 | 32768 bytes | Self-rank MTE copy |
| AIV remote-copy UB | 0 | 0 | 0 | UDMA WQE is submitted through `PIPE_S` |
| AIC L1/L0 | owned by selected MatMul primitive | 2-way pipeline | platform tiling result | BF16 input/weight and FP32 accumulation |
| Cross-core flags | hardware flag IDs 0 and 1 per stage | 2 | 4 logical flags | AIV-complete and AIC-store-complete |

For this mixed Cube/Vector operator a single elementwise `bufferCoefficient`
does not describe the AIC path. The AIV coefficient is instead fixed by the
two 16 KiB communication tiles above; the AIC resource sizes are validated by
the target MatMul tiler for the chosen `M0/N0/K0` family. BF16 multiplication
accumulates in FP32 inside the Cube primitive and writes BF16, so no separate
vector BF16-to-FP32 temporary is allocated.

### Workspace layout

```text
workspace + 0
  stage 0 distributed BF16 rows
  stage 1 distributed BF16 rows
workspace + 400 MiB
  syncMmadFinish (4 * int64)
  syncCommFinish (4 * int64)
  reserved alignment/padding to 4096 bytes
```

The data-stage stride is derived from `EP * commInterval * M0 * K` and checked
against the fixed 400 MiB window. The signal page is never used as MatMul or
communication payload.

### AIV communication path

For each communication loop and workspace stage:

1. The root AIV initializes local compute-finished and communication-finished
   signals, then all participating AIVs synchronize.
2. Each destination peer has exactly one owning AIV subcore. The local peer is
   handled by subcore 0 with MTE; remote peers are handled by subcore 1 with
   UDMA PUT. This preserves the source rule that a UDMA queue and sync ID are
   not shared by multiple subcores.
3. The owner waits for the destination peer's prior compute-complete signal
   only when the current route has payload.
4. Local data is copied in bounded communication blocks. A remote contiguous
   peer slice is submitted as one UDMA transfer when layouts permit; a
   row-wise fallback is used for non-contiguous layouts.
5. The owner fences, publishes the number of completed blocks, and the root
   waits until all data expected by the local rank has arrived.
6. A cross-core flag releases AIC computation for the completed stage.

### AIC GroupedMatMul path

For each communication loop and workspace stage:

1. AIC waits for the AIV communication-complete flag.
2. For every local expert and source rank, route-table prefix sums remap the
   received BF16 rows to that expert's weight matrix.
3. The selected `M0/N0/K0` configuration executes GroupedMatMul and writes the
   valid rows contiguously into `gmmY`.
4. When the stage will be reused, AIC publishes its compute-complete flag so
   the corresponding AIV owner may overwrite the symmetric window.

This retains the source two-stage communication/computation overlap and peer
serialization order.

## Error Handling and Ownership

- Null required tensors, arrays, context, workspace metadata, or executor
  pointers return the corresponding aclnn null-parameter status.
- Unsupported platform, dtype, rank, shape, EP size, counts length, negative
  count, route sum, or workspace size returns `ACLNN_ERR_PARAM_INVALID`.
- A V2 call on a non-Ascend950 platform fails before executor construction.
- The context, `gmmX`, workspace, weights, output, SHMEM runtime, and stream are
  owned by the caller. The operator creates only a borrowed tensor descriptor
  for the context and releases descriptor metadata through the executor's
  normal lifetime.
- SHMEM runtime failures occurring after launch are surfaced through the
  standard stream synchronization and recent-error mechanisms.

## Build and Documentation Changes

- Update the V2 public header and implementation to the new ABI.
- Add a dedicated internal SHMEM op definition and propagate its context tensor
  through inner aclnn generation, kernel signature, and A5 tiling.
- Route public V2 to the SHMEM op and keep public V1 on the existing HCCL op;
  remove Ascend950 registration from the existing HCCL op while retaining the
  A3 files and tiling keys.
- Add SHMEM include/link requirements only for Ascend950 builds. Accept both
  the source SDK `SHMEM_HOME_PATH/shmem/...` layout and the HCCL-integrated
  `SHMEM_HOME/{include,src/device,build/lib}` layout.
- Replace the A5 example with a caller-managed SHMEM initialization and
  allocation flow.
- Document the optional HCCL transport environment (`HCCL_SOCKET_IFNAME`,
  connection/execution timeouts, and rendezvous configuration) without
  hard-coding machine-specific IP addresses or interface names.
- Rewrite the V2 interface document to state the reduced capability and memory
  contract. Keep V1/A3 documentation intact and state that V1 is unsupported on
  Ascend950.

## Test Strategy

Development follows test-first red/green cycles for host behavior.

### op_api tests

- New V2 signature constructs an executor on Ascend950 with valid BF16 inputs.
- Null `shmemContext` is rejected.
- A3 invocation is rejected by V2 while V1/A3 remains unchanged.
- FP16, invalid dimensions, mismatched K/N, unsupported EP, malformed counts,
  negative counts, and incorrect send total are rejected.
- Workspace size equals 400 MiB + 4096 bytes and undersized execution workspace
  is rejected.

### tiling tests

- Source reference case: `M=4084, K=2048, N=4096, EP=2, E=4`.
- Both M0 tile families.
- EP 1 through 8, with multi-rank execution focused on EP 2, 4, and 8.
- Uneven routes, zero-token experts, residue M/K/N tiles, maximum local experts,
  and registered-window boundaries.
- A3 regression cases continue selecting the original HCCL tiling path.

### kernel and build tests

- Ascend950 kernel compiles with SHMEM headers and the dedicated tiling key.
- Kernel argument order includes `shmem_context` without shifting the A3
  semantic inputs incorrectly.
- Existing A3 kernel UT remains green.

### multi-rank precision and performance

An Ascend950 C++ aclnn example initializes SHMEM with the UDMA engine, allocates
`gmmX` and workspace from the SHMEM heap, and invokes V2. At least 30 generated
precision cases cover every EP from 1 through 8, both tile families, uneven
routes, residue shapes, zero-token experts, and the source reference shape. Results are compared
with an FP32 CPU route-plus-GroupedMatMul golden and converted to BF16 for the
same tolerance policy used by the source validation.

Representative cases run with profiler collection after warmup. The report
compares the migrated aclnn implementation with the source implementation and
separates communication, GMM, and total kernel duration where trace data makes
that distinction available.

If the available hardware has fewer than eight devices, all available ranks are
executed and higher-rank topology is covered by host/tiling tests. The final
report explicitly lists every unexecuted hardware case; it must not present an
unexecuted case as passing.

## Acceptance Criteria

- The new branch builds the operator for Ascend950 and preserves A3 build/UT.
- V2 exposes only the approved SHMEM interface and rejects unsupported source
  features.
- The source reference shape passes multi-rank precision validation.
- All executed precision cases pass the documented tolerance.
- Workspace, context, and SHMEM allocation requirements are documented and
  demonstrated by the example.
- Profiler results and any hardware-coverage limitations are recorded without
  unsupported performance claims.
