# Host execution of the real Sparse MLA metadata AICPU kernel

`attention/sparse_flash_mla_metadata` is an AICPU-only operator:

* `op_kernel_aicpu/sparse_flash_mla_metadata_aicpu.json` sets
  `"engine": "DNN_VM_AICPU"`, `"opKernelLib": "CUSTAICPUKernel"`;
* there is no `sparse_flash_mla_metadata_def.cpp`, so TTK's TBE op-info registry
  (`ttk/core_modules/operator/op_info_keeper.py` reading
  `aic-<soc>-ops-info.json`) has no entry and CAMODEL cannot select it.

So the MLA case cannot execute its real metadata pre-op. The kernel body,
however, is plain CPU C++ (`Prepare` / `BalanceSchedule` / `GenMetadata`) over
the public AICPU `CpuKernelContext` API. This directory runs that exact code on
the host.

## What this is and is not

* It compiles the **unmodified** kernel source
  `attention/sparse_flash_mla_metadata/op_kernel_aicpu/sparse_flash_mla_metadata_aicpu.cpp`
  and calls the real `aicpu::SparseFlashMlaMetadataCpuKernel::Compute`.
* It builds the NodeDef through the toolkit's own `aicpu::NodeDefBuilder`
  (`pkg_inc/aicpu_common/context/common/node_def_builder.h`) — the same API the
  repo's official AICPU UT factory uses (`examples/add_example/tests/ut/op_kernel_aicpu/test_add_example.cpp`,
  macro `CREATE_NODEDEF` / `RUN_KERNEL`).
* It does **not** re-implement, stub, or shortcut the scheduling algorithm, and
  it does not fabricate metadata.
* It is a *host* execution of a real AICPU implementation. It is **not** the
  board AICPU runtime (`libruntime` on the SoC), and it does not prove board
  AICPU behaviour, on-device ordering, or the real `RunCpuKernel` dispatch path.

## Why the host context libraries are linkable

The toolkit ships x86-64 host builds of the AICPU context layer (verified with
`nm`/`file` on `$CANN_ROOT/x86_64-linux/lib64`):

| library | arch | ABI | provides |
| --- | --- | --- | --- |
| `libaicpu_context_host.a` | x86-64 | `_GLIBCXX_USE_CXX11_ABI=0` | `CpuKernelContext`, `NodeDef`/`Tensor`/`AttrValue` impls, `CpuKernelUtils::CreateNodeDef/CreateTensor/CreateAttrValue`, `RegistCpuKernel` |
| `libaicpu_nodedef_host.a` | x86-64 | ABI0 | `aicpu::NodeDefBuilder::{Input,Output,Attr,...}` |
| `libhost_ascend_protobuf.a` | x86-64 | ABI0 | `ascend_private::protobuf::*` and `absl::lts_ascend_private::*` used by the context layer |

(The non-`_host` `libaicpu_context.a` / `libaicpu_nodedef.a` are aarch64 and
belong to the device link, so they are not used here. `libbase_ascend_protobuf.a`
is the aarch64 twin of `libhost_ascend_protobuf.a`.)

The device link is the model for this one: `cmake/symbol.cmake` links the
shipped `libtransformer_aicpu_kernels.so` from `libaicpu_context.a` +
`libbase_ascend_protobuf.a` with `--whole-archive`.

## Files

| file | purpose |
| --- | --- |
| `metadata_host_main.cpp` | host driver: reads a work dir, builds the NodeDef, runs `Compute`, writes the metadata tensor, prints a summary |
| `build_metadata_host.sh` | host `g++` build of the driver + the real kernel source, linking the three host libs |
| `run_metadata_host.py` | per-case wrapper: derives the metadata attrs from each built MLA case, runs the driver, validates the tensor, writes `metadata_delay<N>.npy` |
| `gen_smla_metadata.sh` | end-to-end: build driver → build cases → real metadata per delay → rebuild cases → CAMODEL run |

## I/O contract of `metadata_host`

`metadata_host <workdir>` reads three text files:

```
attrs.txt   "int <name> <value>" | "str <name> <value>"
inputs.txt  "<name> <dtype> <d0,d1,...> <raw-file>"   (op input-index order)
output.txt  "<dtype> <d0,d1,...> <raw-file>"
```

`run_metadata_host.py` writes them from the MLA case `.npy` files. All tensors
are raw little-endian int32. The driver writes the output tensor and prints
`METADATA_OK fa_enabled_cores=… fd_enabled_cores=… fa0=[enable,bn2s,ms,s2s,bn2e,me,s2e,fdws,s2max]`.

## Parameters that must match the MLA case

The metadata op shares the MLA op's names for `cmp_ratio`, `ori_mask_mode`,
`cmp_mask_mode`, `ori_win_left`, `ori_win_right`, `layout_q`, `layout_kv`; these
are copied verbatim from each case's `case.csv`. Two groups are **not** in the
MLA case and are explicit inputs to `run_metadata_host.py` (recorded per case in
`metadata_params_delay<N>.json`):

* `num_heads_q` (default 1), `num_heads_kv` (default 1), `head_dim` (default
  512) — the tiny fixture's single-head, `head_dim=512` MLA shape;
* `ori_topk` / `cmp_topk` (default 0). With the fixture's
  `ori_mask_mode=BAND` and `cmp_mask_mode=RIGHT_DOWN_CAUSAL`, the metadata
  generator uses the mask-window path and does not read top-k, so 0 keeps it on
  the same semantics the case declares. Set them if you want the sparse path.

`GetAttrValueOpt` leaves the class defaults for the bool attrs
(`has_ori_kv=true`, `has_cmp_kv=true`, `is_batch_consistency=false`) and for
`aic_core_num`/`aiv_core_num` (36/72), so the driver does not send them.

## Run

```bash
# inside the shared local-resource-runner lock
bash build_metadata_host.sh                                   # host g++ only
python3 ../build_smla_linked_cases.py --out /tmp/pro0921_latent_mla
python3 run_metadata_host.py --mla-root /tmp/pro0921_latent_mla \
    --out /tmp/pro0921_latent_mla/meta
python3 ../build_smla_linked_cases.py --out /tmp/pro0921_latent_mla \
    --metadata-dir /tmp/pro0921_latent_mla/meta
bash ../run_tiny_sim.sh /tmp/pro0921_latent_mla sparse_mla

# or the one-shot orchestrator
bash gen_smla_metadata.sh /tmp/pro0921_latent_mla
```

Each delay case gets its own `metadata_delay<N>.npy`; `build_smla_linked_cases.py`
copies the resolved tensor into that case dir and records the source in
`expected.json`, so cross-delay metadata reuse is visible rather than silent.

## Residual limits

* Host context libraries are only present when the CANN install ships the
  `*_host.a` archives; `build_metadata_host.sh` fails with `MISSING:` lines when
  they are absent (no silent fallback, no stubbing).
* Compiling the driver is a host C++ build; if the toolkit's host archives were
  built with a different ABI the link fails loudly (the script pins ABI0 on
  purpose).
* The produced metadata reflects the metadata op's own cost model. Where the MLA
  case declares per-row `ori_topk_length` but the metadata op takes a single
  `ori_topk`, the scheduling hint is the metadata op's uniform model; this is
  real implementation output, not a claim that it equals a board run of the same
  graph.
